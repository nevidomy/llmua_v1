"""Profile a training step and verify SDPA is taking a fast backend.

Checks several things:

  1. Do q, k, v reach scaled_dot_product_attention with the SAME dtype?
     A dtype mismatch (e.g. fp32 q/k from an fp32 RoPE buffer against bf16 v)
     silently disqualifies the flash and mem-efficient kernels. No error is
     raised -- SDPA falls back to the math backend, which materialises the full
     B x H x T x T score matrix.

  2. Does PyTorch consider flash attention usable for these tensors?
     can_use_flash_attention(debug=True) prints the specific reason if not.

  3. Does a forced flash-only context actually run?
     The definitive test: if it raises, the real run is not using flash.

  4. Where does the time go?
     Kernel-level breakdown plus tokens/sec and peak memory.

Usage:
    python -m perf.profile_step --shape P1 --T 2048 --batch 8 --amp bf16
    python -m perf.profile_step --shape P4 --T 2048 --batch 4 --amp bf16 --trace
"""

import argparse
import contextlib
import time

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from model.transformer import (model_P1, model_P2, model_P3, model_P4,
                               model_target)

FACTORIES = {"P1": model_P1, "P2": model_P2, "P3": model_P3,
             "P4": model_P4, "target": model_target}

# ---------------------------------------------------------------- dtype probe

_seen = []
_real_sdpa = F.scaled_dot_product_attention


def _probe_sdpa(q, k, v, *a, **kw):
    _seen.append((tuple(q.shape), q.dtype, k.dtype, v.dtype))
    return _real_sdpa(q, k, v, *a, **kw)


@contextlib.contextmanager
def probe_sdpa():
    _seen.clear()
    F.scaled_dot_product_attention = _probe_sdpa
    import model.transformer as mt
    mt.F.scaled_dot_product_attention = _probe_sdpa
    try:
        yield _seen
    finally:
        F.scaled_dot_product_attention = _real_sdpa
        mt.F.scaled_dot_product_attention = _real_sdpa


# ------------------------------------------------------------------- backends

def backend_report(shape, dtype, device):
    """Ask PyTorch directly whether each fast backend is usable."""
    if device != "cuda":
        print("  (backend checks require CUDA -- skipped)")
        return
    B, H, T, W = shape
    q = torch.randn(B, H, T, W, dtype=dtype, device="cuda")
    params = torch.backends.cuda.SDPAParams(q, q, q, None, 0.0, True, False)
    print(f"  can_use_flash_attention     : "
          f"{torch.backends.cuda.can_use_flash_attention(params, debug=True)}")
    print(f"  can_use_efficient_attention : "
          f"{torch.backends.cuda.can_use_efficient_attention(params, debug=True)}")


def forced_backend_test(model, x, y, amp_ctx, device):
    """The definitive check: run under a flash-only context and see if it works."""
    if device != "cuda":
        print("  (forced-backend test requires CUDA -- skipped)")
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel
    for name, backend in (("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
                          ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION)):
        try:
            with sdpa_kernel([backend]), amp_ctx(), torch.no_grad():
                model(x, y)
            print(f"  {name:<20} USABLE")
        except Exception as e:
            print(f"  {name:<20} NOT USABLE -> {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:110]}")


# --------------------------------------------------------------------- timing

def timed_steps(model, x, y, opt, amp_ctx, steps, device):
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    for _ in range(5):                      # warmup
        with amp_ctx():
            loss = model(x, y)[1]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    sync()

    t0 = time.perf_counter()
    for _ in range(steps):
        with amp_ctx():
            loss = model(x, y)[1]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    sync()
    return (time.perf_counter() - t0) / steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="P1", choices=list(FACTORIES))
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--vocab", type=int, default=32768)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--trace", action="store_true",
                   help="write a chrome trace to trace.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compile", action="store_true",
                   help="compile the model")
    p.add_argument("--optimizer", default="AdamW", choices=["AdamW", "CustomAdamW", "CustomAdamWPlusMuon"])
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")

    dev = args.device
    model = FACTORIES[args.shape](
        context_width=args.T, vocab_size=args.vocab,
        enable_init=True, enable_swiglu=True,
        enable_rms_norm=True, enable_qk_norm=True).to(dev)
    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)
        print("Model compiled.")

    # Choose optimizer
    opt = None
    if args.optimizer == "AdamW":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=(dev == "cuda"))
    else:
        from model.adamw_muon import muon_param_groups, MuonAdamW
        groups = muon_param_groups(
            model, lr=1e-4, muon_lr_mult=30.0,
            weight_decay=0.1, use_muon=(args.optimizer == "CustomAdamWPlusMuon"))
        opt = MuonAdamW(groups, betas=(0.9, 0.95), ns_steps=1)

    if args.amp == "fp32":
        amp_ctx = contextlib.nullcontext
        adtype = torch.float32
    else:
        adtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        amp_ctx = lambda: torch.autocast(dev, dtype=adtype)

    x = torch.randint(0, args.vocab, (args.batch, args.T), device=dev)
    y = torch.randint(0, args.vocab, (args.batch, args.T), device=dev)

    n_head = model.blocks[0].mha.headcount
    head_dim = model.blocks[0].mha.headsize
    print(f"shape={args.shape}  d={model.embed.weight.shape[1]}  "
          f"L={len(model.blocks)}  heads={n_head}x{head_dim}  "
          f"h={model.ff_width}")
    print(f"N={model.num_params():,}  N_noemb={model.num_params_no_embedings():,}")
    print(f"T={args.T}  batch={args.batch}  amp={args.amp}  device={dev}")
    print()

    # 1 -----------------------------------------------------------------------
    print("[1] dtypes reaching scaled_dot_product_attention")
    with probe_sdpa() as seen, amp_ctx(), torch.no_grad():
        model(x, y)
    dtypes = {(a, b, c) for _, a, b, c in seen}
    for shp, dq, dk, dv in seen[:1]:
        print(f"  q{tuple(shp)} {dq}   k {dk}   v {dv}")
    if len(dtypes) == 1 and len(next(iter(dtypes))) == len(set(next(iter(dtypes)))) == 1:
        pass
    ok = all(dq == dk == dv for _, dq, dk, dv in seen)
    print(f"  all {len(seen)} calls have matching q/k/v dtype: {ok}"
          f"{'' if ok else '   <-- FLASH DISQUALIFIED'}")
    print()

    # 2 -----------------------------------------------------------------------
    print("[2] backend availability")
    backend_report((args.batch, n_head, args.T, head_dim), adtype, dev)
    print()

    # 3 -----------------------------------------------------------------------
    print("[3] forced-backend test")
    forced_backend_test(model, x, y, amp_ctx, dev)
    print()

    # 4 -----------------------------------------------------------------------
    print("[4] throughput")
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sec = timed_steps(model, x, y, opt, amp_ctx, args.steps, dev)
    tok = args.batch * args.T
    print(f"  {sec*1e3:.2f} ms/step   {tok/sec:,.0f} tok/s")
    if dev == "cuda":
        print(f"  peak memory {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print()

    print("[5] kernel breakdown")
    acts = [ProfilerActivity.CPU]
    if dev == "cuda":
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts, record_shapes=False) as prof:
        for _ in range(3):
            with amp_ctx():
                loss = model(x, y)[1]
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        if dev == "cuda":
            torch.cuda.synchronize()
    key = "cuda_time_total" if dev == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=key, row_limit=18))
    if args.trace:
        prof.export_chrome_trace("trace.json")
        print("wrote trace.json  (open at chrome://tracing or ui.perfetto.dev)")


if __name__ == "__main__":
    main()