"""
Measure the QK-norm attention-logit table at arbitrary head_dim.

Reproduces the appendix table:

    | placement  | logits at init | after 4x weight growth | ratio |

The measured quantity is the pre-softmax attention logit

    logit[b,h,i,j] = (q[b,h,i,:] . k[b,h,j,:]) / sqrt(head_dim)

over causal positions only (j <= i), which is what a
"max attention logit" training monitor sees.

Two configurations:
  no QK-norm : q,k = split(x @ W_qkv)
  QK-norm    : q,k = rmsnorm_over_head_dim(split(x @ W_qkv))

"Weight growth" simulates training-time growth of the QKV weights by
multiplying W_qkv by `growth`. Because q and k each scale linearly,
the un-normalised logit scales quadratically -> growth^2.

RoPE is omitted deliberately: it is a rotation, so it preserves norms and
cannot change any statistic measured here. (Placement of QK-norm relative
to RoPE matters for *positional* correctness, which is a separate test.)

Sample-count control: the number of logits is B*H*T*(T+1)/2, and H = d/W
depends on head_dim. Batch size is chosen so that B*H is constant across
head_dim settings, otherwise the *max* statistic would be compared across
different sample sizes and W=32 would look larger for free.
"""

import argparse
import math

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last dimension."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(dtype)


@torch.no_grad()
def measure(
    d: int,
    head_dim: int,
    qk_norm: bool,
    growth: float,
    seq_len: int,
    bh_target: int,
    init_std: float,
    seed: int,
    device: str,
) -> dict:
    """Return statistics of the causal pre-softmax attention logits."""
    assert d % head_dim == 0, "d must be divisible by head_dim"
    n_head = d // head_dim
    assert bh_target % n_head == 0, "bh_target must be divisible by n_head"
    batch = bh_target // n_head

    g = torch.Generator(device="cpu").manual_seed(seed)

    # Residual stream entering the block. Pre-norm means the attention input
    # is RMSNorm'd, so every component has unit RMS. Simulate exactly that.
    x = torch.randn(batch, seq_len, d, generator=g)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    # QKV projection, no bias. init_std is the model's weight init.
    w_qkv = torch.randn(3 * d, d, generator=g) * init_std
    w_qkv = w_qkv * growth  # simulated training-time weight growth

    x = x.to(device)
    w_qkv = w_qkv.to(device)

    qkv = x @ w_qkv.T                                  # (B, T, 3d)
    qkv = qkv.view(batch, seq_len, 3, n_head, head_dim)
    q, k = qkv[:, :, 0], qkv[:, :, 1]                  # (B, T, H, W)

    if qk_norm:
        gain = torch.ones(head_dim, device=device)     # gain at init
        q = rms_norm(q, gain)
        k = rms_norm(k, gain)

    q = q.transpose(1, 2)                              # (B, H, T, W)
    k = k.transpose(1, 2)

    logits = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)

    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    vals = logits[:, :, causal].flatten().float()

    return {
        "n_head": n_head,
        "batch": batch,
        "count": vals.numel(),
        "std": vals.std().item(),
        "abs_mean": vals.abs().mean().item(),
        "p99.9": vals.abs().quantile(0.999).item(),
        "max": vals.abs().max().item(),
        "bound": math.sqrt(head_dim),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=768, help="model width")
    p.add_argument("--head-dims", type=int, nargs="+", default=[32, 64])
    p.add_argument("--growth", type=float, default=4.0, help="QKV weight multiplier")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--bh-target", type=int, default=96, help="constant B*H across head_dims")
    p.add_argument("--init-std", type=float, default=None, help="default 0.55/sqrt(d)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    init_std = args.init_std if args.init_std is not None else 0.55 / math.sqrt(args.d)

    print(f"d={args.d}  init_std={init_std:.5f}  seq_len={args.seq_len}  "
          f"growth={args.growth}x  device={args.device}  seeds={args.seeds}")
    print()

    for W in args.head_dims:
        rows = []
        for qk in (False, True):
            stats = {}
            for scale, gr in (("init", 1.0), ("grown", args.growth)):
                runs = [
                    measure(args.d, W, qk, gr, args.seq_len, args.bh_target,
                            init_std, s, args.device)
                    for s in args.seeds
                ]
                stats[scale] = {
                    key: sum(r[key] for r in runs) / len(runs)
                    for key in ("std", "abs_mean", "p99.9", "max")
                }
                stats["meta"] = runs[0]
            rows.append((qk, stats))

        meta = rows[0][1]["meta"]
        print(f"head_dim = {W}   n_head = {meta['n_head']}   batch = {meta['batch']}   "
              f"logits/run = {meta['count']:,}   bound = sqrt(W) = {meta['bound']:.2f}")
        print(f"  {'placement':<12} {'max@init':>10} {'max@grown':>11} {'ratio':>8} "
              f"{'std@init':>10} {'std@grown':>10} {'p99.9@grown':>12}")
        for qk, s in rows:
            name = "QK-norm" if qk else "no QK-norm"
            ratio = s["grown"]["max"] / s["init"]["max"]
            print(f"  {name:<12} {s['init']['max']:>10.2f} {s['grown']['max']:>11.2f} "
                  f"{ratio:>7.2f}x {s['init']['std']:>10.3f} {s['grown']['std']:>10.3f} "
                  f"{s['grown']['p99.9']:>12.2f}")
        print()


if __name__ == "__main__":
    main()