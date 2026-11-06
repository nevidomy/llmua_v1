"""
with mps device support

    python -m perf.profile_step2 --shape P1 --T 2048 --batch 8 --amp bf16
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
        print(f"  (backend introspection is CUDA-only; on {device} SDPA uses "
              f"the platform kernel, NOT FlashAttention)")
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
        elif device == "mps":
            torch.mps.synchronize()

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
    p.add_argument("--tf32", default="high",
                   choices=["highest", "high", "medium"],
                   help="float32 matmul precision; only affects fp32 matmuls, "
                        "so it is a no-op under --amp bf16")
    p.add_argument("--compile", action="store_true",
                   help="wrap the model in torch.compile before profiling")
    p.add_argument("--trace", action="store_true",
                   help="write a chrome trace to trace.json")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu"))
    args = p.parse_args()

    dev = args.device
    torch.set_float32_matmul_precision(args.tf32)
    model = FACTORIES[args.shape](
        context_width=args.T, vocab_size=args.vocab,
        enable_init=True, enable_swiglu=True,
        enable_rms_norm=True, enable_qk_norm=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=(dev == "cuda"))
    if args.compile:
        model = torch.compile(model)

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
    print(f"T={args.T}  batch={args.batch}  amp={args.amp}  device={dev}"
          f"  compile={args.compile}")
    print(f"fp32_matmul_precision={torch.get_float32_matmul_precision()}  "
          f"cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}  "
          f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}")
    print("  ^ these must match your trainer, or throughput is not comparable")
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
    elif dev == "mps":
        print(f"  allocated {torch.mps.current_allocated_memory()/2**30:.2f} GiB"
              f"  driver {torch.mps.driver_allocated_memory()/2**30:.2f} GiB")
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
    n_triton = sum(1 for e in prof.key_averages() if e.key.startswith("triton_"))
    print(f"\ntriton kernels in profile: {n_triton}"
          f"{'   <-- compiled code is running' if n_triton else '   <-- EAGER, no compiled kernels'}")
    if args.trace:
        prof.export_chrome_trace("trace.json")
        print("wrote trace.json  (open at chrome://tracing or ui.perfetto.dev)")


if __name__ == "__main__":
    main()