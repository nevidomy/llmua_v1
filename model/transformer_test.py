"""Model invariants. Rerun after any architecture change.

The 32-combination loop is the point: flags interact, and a change that looks
local (LayerNorm bias, SwiGLU width, QK-norm) can break one combination while
the one you tested by hand still works.
"""

import itertools
import math

import torch
import torch.nn.functional as F

from model.ff_width import (effective_expansion, ff_width_for,
                            gelu_width_matching, mlp_params)
from model.transformer import Transformer

V, T, d, L, hs = 512, 64, 128, 4, 32
FLAGS = ("no_biases", "custom_init", "use_swiglu", "rms_norm", "qk_norm")


def build(**kw):
    kw.setdefault("headsize", hs)
    return Transformer("t", V, T, d, L, **kw)


def main():
    print("=== 1. all 32 flag combinations: build / forward / backward ===")
    fails = []
    for combo in itertools.product([False, True], repeat=len(FLAGS)):
        kw = dict(zip(FLAGS, combo))
        try:
            torch.manual_seed(0)
            m = build(**kw)
            x = torch.randint(0, V, (2, T))
            y = torch.randint(0, V, (2, T))
            logits, loss = m(x, y)
            assert logits.shape == (2, T, V), logits.shape
            assert torch.isfinite(loss), "non-finite loss"
            loss.backward()
            missing = [n for n, p in m.named_parameters() if p.grad is None]
            assert not missing, f"no grad: {missing[:3]}"
        except Exception as e:
            fails.append((kw, f"{type(e).__name__}: {e}"))
    print(f"  {2**len(FLAGS) - len(fails)}/{2**len(FLAGS)} ok")
    for kw, err in fails:
        print("  FAIL", {k: v for k, v in kw.items() if v}, err)
    assert not fails

    print("\n=== 2. ff_width is computed; GELU is matched to rounded SwiGLU ===")
    print(f"  {'d':>5}{'swiglu':>8}{'ff_width':>10}{'eff exp':>9}"
          f"{'per-layer d^2':>15}")
    for dd in (128, 192, 384, 768):
        h_sw = ff_width_for(dd, True)
        widths = {}
        for sw in (False, True):
            m = Transformer("t", V, T, dd, 2, 64, use_swiglu=sw)
            want = h_sw if sw else gelu_width_matching(h_sw)
            eff = effective_expansion(dd, m.ff_width, sw)
            pl = (4 * dd * dd + (3 if sw else 2) * dd * m.ff_width) / dd ** 2
            print(f"  {dd:>5}{str(sw):>8}{m.ff_width:>10}{eff:>9.2f}{pl:>15.2f}")
            assert m.ff_width == want, (dd, sw, m.ff_width, want)
            # same relative band as validate_ff_width (rounding at small d)
            assert abs(eff - 4.0) / 4.0 <= 0.15, f"effective expansion {eff}"
            widths[sw] = m.ff_width
        assert mlp_params(dd, widths[True], True) == \
               mlp_params(dd, widths[False], False)

    print("\n=== 3. explicit ff_width is validated, not trusted ===")
    ok = Transformer("t", V, T, 128, 2, 32, ff_width=512, use_swiglu=False)
    print(f"   512 + GELU        -> accepted (ff_width={ok.ff_width})")
    for bad, sw, why in ((512, True, "4d with SwiGLU"),
                         (341, True, "not a multiple of 8"),
                         (64, False, "narrower than d")):
        try:
            Transformer("t", V, T, 128, 2, 32, ff_width=bad, use_swiglu=sw)
            print(f"  {bad} + swiglu={sw} -> NOT CAUGHT")
            raise SystemExit(1)
        except ValueError:
            print(f"  {bad:>4} + swiglu={str(sw):<6} -> raised ({why})")

    print("\n=== 4. no_biases removes every bias, under both norms ===")
    for rn in (False, True):
        m = build(no_biases=True, rms_norm=rn)
        biases = [n for n, _ in m.named_parameters() if n.endswith(".bias")]
        print(f"  rms_norm={str(rn):<6} bias tensors {len(biases)}")
        assert not biases, biases

    print("\n=== 5. custom_init holds logit_std ~0.55 across widths ===")
    for dd in (128, 384, 768):
        torch.manual_seed(0)
        m = Transformer("t", V, T, dd, 6, 64, custom_init=True, no_biases=True,
                        use_swiglu=True, rms_norm=True, qk_norm=True)
        with torch.no_grad():
            lg = m(torch.randint(0, V, (2, T)))
            ls = F.cross_entropy(lg.reshape(-1, V),
                                 torch.randint(0, V, (2 * T,))).item()
        print(f"  d={dd:<5} logit_std {lg.std():.3f}   init loss {ls:.4f}"
              f"   ln(V) {math.log(V):.4f}")
        assert 0.45 < lg.std() < 0.70, f"logit_std {lg.std()} drifted"

    print("\n=== 6. tied weight is one tensor, counted once ===")
    m = build()
    assert m.embed.weight.data_ptr() == m.lmh.weight.data_ptr(), "not tied"
    uniq = len({id(p) for p in m.parameters()})
    assert len(list(m.parameters())) == uniq, "duplicate in parameters()"
    assert any(n.startswith("blocks.0.") for n, _ in m.named_parameters()), (
        "blocks not registered -- use nn.ModuleList, not a Python list")
    print(f"  tied, {uniq} unique tensors, num_params {m.num_params():,}")

    print("\n=== 7. sweep shapes ===")
    print(f"  {'d':>5}{'L':>4}{'ff':>7}{'non-emb':>12}{'total@32k':>12}")
    for dd, LL in ((128, 6), (192, 8), (256, 10), (384, 12), (768, 12)):
        m = Transformer("t", 32768, 256, dd, LL, 64, no_biases=True,
                        use_swiglu=True, rms_norm=True, qk_norm=True,
                        custom_init=True)
        ne = sum(p.numel() for n, p in m.named_parameters()
                 if p.ndim >= 2 and not n.startswith(("embed", "lmh")))
        print(f"  {dd:>5}{LL:>4}{m.ff_width:>7}{ne:>12,}{m.num_params():>12,}")

    print("\n=== 8. generate, and T>context_width is caught ===")
    m = build()
    out = m.generate(12, prompt=[1, 2, 3])
    assert len(out) == 15 and max(out) < V
    print(f"  generate: 3 -> {len(out)} ids, max {max(out)} < {V}")
    try:
        m(torch.randint(0, V, (1, T + 1)))
        print("  over-long input NOT caught")
        raise SystemExit(1)
    except AssertionError:
        print(f"  T={T+1} > context_width={T} -> assert fired")

    print("\n=== 9. attention_mask=None matches a causal allow-mask ===")
    m = build().eval()
    x = torch.randint(0, V, (2, T))
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    allow = causal.expand(2, 1, T, T)
    with torch.no_grad():
        a = m(x)
        b = m(x, attention_mask=allow)
    max_diff = (a - b).abs().max().item()
    print(f"  max |logits_A - logits_B| = {max_diff:.4e}")
    assert max_diff < 1e-4, max_diff

    print("\n=== 10. doc_masking/doc_rope default off; arch round-trips ===")
    m = build().eval()
    assert m.doc_masking is False and m.doc_rope is False
    x = torch.randint(1, V, (2, T))          # no BOS -> doc_rope == vanilla RoPE
    y = torch.randint(0, V, (2, T))
    with torch.no_grad():
        a = m(x)
        logits, loss = m(x, y)
    assert logits.shape == (2, T, V) and torch.isfinite(loss)
    clone = Transformer(**m.arch).eval()
    clone.load_state_dict(m.state_dict())
    with torch.no_grad():
        b = clone(x)
    assert (a - b).abs().max().item() == 0.0
    # old checkpoints omit the new keys
    old = {k: v for k, v in m.arch.items()
           if k not in ("doc_masking", "doc_rope", "bos_id")}
    legacy = Transformer(**old)
    assert legacy.doc_masking is False and legacy.doc_rope is False
    m_rope = build(doc_rope=True).eval()
    m_rope.load_state_dict(m.state_dict())
    with torch.no_grad():
        c = m_rope(x)
    rope_diff = (a - c).abs().max().item()
    print(f"  no-BOS doc_rope vs vanilla max |d| = {rope_diff:.4e}")
    assert rope_diff < 1e-5, rope_diff
    x_bos = x.clone(); x_bos[:, T // 3] = 0
    with torch.no_grad():
        d_van = m(x_bos)
        d_rope = m_rope(x_bos)
    bos_diff = (d_van - d_rope).abs().max().item()
    print(f"  with-BOS doc_rope vs vanilla max |d| = {bos_diff:.4e}")
    assert bos_diff > 1e-3, bos_diff
    with torch.no_grad():
        d_off = m_rope(x_bos, doc_rope=False)
    assert (d_van - d_off).abs().max().item() < 1e-5
    print(f"  arch flags doc_masking={m.arch['doc_masking']} "
          f"doc_rope={m.arch['doc_rope']}")

    print("\n=== 11. doc_masking without a mask raises ===")
    try:
        md = build(doc_masking=True).eval()
    except RuntimeError as e:
        if "flex_attention" in str(e):
            print("  skip (no flex_attention)")
        else:
            raise
    else:
        x = torch.randint(0, V, (2, T))
        try:
            md(x)
            raise SystemExit("missing block_mask was silently allowed")
        except RuntimeError as e:
            assert "no block_mask" in str(e), e
        print("  missing block_mask raises")
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        with torch.no_grad():
            md(x, attention_mask=causal.expand(2, 1, T, T))
        print("  attention_mask still accepted")

    print("\nall checks passed")


if __name__ == "__main__":
    main()