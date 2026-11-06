"""Causal vs document-masked val loss ("junk" leftover from the previous doc).

Packed windows of val.bin, same tiling as position_bucketing.py. Two
forwards per batch under no_grad:

  A  model(x, y)                         is_causal=True
  B  model(x, y, attention_mask=mask)    same-doc & causal

Delta = B − A. Bucketed by junk = t − own, where own is the offset from
the last BOS (0 at a document start) so junk is the window-relative
position of that BOS: how many previous-doc tokens sit in the window
before this token's document. Cross-doc attention can only exist for
junk > 0.

    python -m analysis.junk_test /path/to/run_dir 47
    python -m analysis.junk_test llmua_P2/P2_lr_sweep_..._seed_47 47
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from tqdm import tqdm

from data.index import ROOT
from sample_generator import load_model


BOS = 0


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="run directory (or path to a .pt)")
    p.add_argument("seed", type=int, help="recorded / torch seed")
    p.add_argument("--ckpt", default="ckpt_final.pt")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--split", default="val")
    p.add_argument("--bins_dir", default=None)
    p.add_argument("--bos", type=int, default=BOS)
    p.add_argument("--out", default="junk_test.png")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model", default=None)
    p.add_argument("--enable_init", action="store_true")
    p.add_argument("--enable_swiglu", action="store_true")
    p.add_argument("--enable_rms_norm", action="store_true")
    p.add_argument("--enable_qk_norm", action="store_true")
    return p.parse_args()


def resolve_ckpt(run_dir, ckpt):
    if os.path.isfile(run_dir):
        return run_dir
    for cand in (run_dir, os.path.join(ROOT, "runs", run_dir)):
        path = os.path.join(cand, ckpt)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"no {ckpt} under {run_dir!r} or {ROOT}/runs/{run_dir}")


def junk_edges(T):
    """[0], [1], [2-3], [4-7], ... covering junk in 0..T-1.
    junk < 0 (last BOS before the window) is reported separately."""
    edges = [(0, 0)]
    lo = 1
    while lo < T:
        hi = min(2 * lo - 1, T - 1)
        edges.append((lo, hi))
        lo = hi + 1
    return edges


def last_bos_index(data, bos):
    idx = np.arange(len(data), dtype=np.int64)
    return np.maximum.accumulate(np.where(data == bos, idx, np.int64(-1)))


def plot_junk(edges, nats, se, counts, pre, out_path, title):
    import matplotlib.pyplot as plt

    labels = ["[<0]"] + [
        f"[{lo}]" if lo == hi else f"[{lo}-{hi}]" for lo, hi in edges]
    xs = np.arange(len(labels))
    ys = np.array([pre["nats"], *nats])
    yerr = np.array([pre["se"], *se])
    cnt = np.array([pre["n"], *counts])

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.axhline(0.0, color="0.65", linewidth=0.8)
    ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color="C0", capsize=3,
                zorder=3, label="Δ nats (doc-mask − causal)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("junk = t − own  (previous-doc tokens packed in the window)")
    ax.set_ylabel("Δ nats / token")
    ax.set_title(title)

    ax2 = ax.twinx()
    ax2.bar(xs, cnt, color="0.75", alpha=0.35, zorder=1, label="tokens")
    ax2.set_ylabel("tokens")
    ax2.set_yscale("log")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ckpt_path = resolve_ckpt(args.run_dir, args.ckpt)
    m, ck = load_model(ckpt_path, args.device, model=args.model,
                       enable_init=args.enable_init,
                       enable_swiglu=args.enable_swiglu,
                       enable_rms_norm=args.enable_rms_norm,
                       enable_qk_norm=args.enable_qk_norm)
    T = m.context_width
    device = args.device

    bins_dir = args.bins_dir or f"{ROOT}/bins/{m.vocab_size}"
    meta = json.load(open(os.path.join(bins_dir, "meta.json")))
    assert meta["vocab_size"] == m.vocab_size, (
        f"bins are vocab {meta['vocab_size']}, model is "
        f"{m.vocab_size} -- WRONG BINS FOR THIS MODEL")
    cpt = meta[f"chars_per_token_{args.split}"]
    to_bpc = 1.0 / (cpt * math.log(2))

    data = np.memmap(os.path.join(bins_dir, f"{args.split}.bin"),
                     dtype=np.uint16, mode="r")
    last_bos = last_bos_index(data, args.bos)
    n = (len(data) - 1) // T
    edges = junk_edges(T)
    n_edges = len(edges)

    ctx = (torch.autocast(device_type=device.split(":")[0],
                          dtype=torch.bfloat16)
           if device != "cpu" else torch.autocast("cpu", enabled=False))

    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    # running sums for overall mean and per-bucket mean ± se
    tot_a = tot_b = 0.0
    n_tok = 0
    b_sum = np.zeros(n_edges, dtype=np.float64)
    b_sumsq = np.zeros(n_edges, dtype=np.float64)
    b_n = np.zeros(n_edges, dtype=np.int64)
    pre_sum = pre_sumsq = 0.0
    pre_n = 0

    total_batches = (n + args.batch - 1) // args.batch
    with torch.no_grad():
        for start in tqdm(range(0, n, args.batch), total=total_batches,
                           desc="junk eval"):
            b = min(args.batch, n - start)
            ix = (start + np.arange(b)) * T
            win_idx = ix[:, None] + np.arange(T)
            x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
            y = np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)

            seg = torch.from_numpy(last_bos[win_idx]).to(device)
            same_doc = seg[:, :, None] == seg[:, None, :]
            mask = (same_doc & causal).unsqueeze(1)       # [b, 1, T, T]

            with ctx:
                _, loss_a = m(x, y, reduce=False)
                _, loss_b = m(x, y, reduce=False, attention_mask=mask)
            la = loss_a.float().cpu().numpy().astype(np.float64)
            lb = loss_b.float().cpu().numpy().astype(np.float64)
            d = lb - la

            tot_a += la.sum()
            tot_b += lb.sum()
            n_tok += la.size

            own = win_idx - last_bos[win_idx]
            junk = np.arange(T)[None, :] - own            # t − own
            pre_m = junk < 0
            if pre_m.any():
                v = d[pre_m]
                pre_sum += v.sum()
                pre_sumsq += (v * v).sum()
                pre_n += v.size
            for i, (lo, hi) in enumerate(edges):
                msk = (junk >= lo) & (junk <= hi)
                if not msk.any():
                    continue
                v = d[msk]
                b_sum[i] += v.sum()
                b_sumsq[i] += (v * v).sum()
                b_n[i] += v.size

    def stats(s, s2, k):
        if k == 0:
            return float("nan"), float("nan")
        mean = s / k
        var = max(s2 / k - mean * mean, 0.0)
        se = math.sqrt(var / k)
        return mean, se

    mean_a, mean_b = tot_a / n_tok, tot_b / n_tok
    delta = mean_b - mean_a
    pre_m, pre_se = stats(pre_sum, pre_sumsq, pre_n)
    means, ses = [], []
    for i in range(n_edges):
        mu, se = stats(b_sum[i], b_sumsq[i], int(b_n[i]))
        means.append(mu)
        ses.append(se)

    a = m.arch
    print(f"\nmodel  {a['name']} -- {a.get('description') or ''}")
    print(f"ckpt   {ckpt_path}: step {ck.get('step')}, seed {args.seed}")
    print(f"data   {args.split}.bin: {n:,} windows of T={T} "
          f"({n_tok / 1e6:.2f}M tokens), batch {args.batch}")
    print(f"\ncausal      {mean_a:.6f} nats/tok  = {mean_a * to_bpc:.6f} bpc")
    print(f"doc-masked  {mean_b:.6f} nats/tok  = {mean_b * to_bpc:.6f} bpc")
    print(f"delta       {delta:+.6f} nats/tok  = {delta * to_bpc:+.6f} bpc"
          f"   (doc-mask − causal; >0 means the model used junk context)\n")

    print(f"{'junk':>12} {'n_tok':>10} {'d_nats':>10} {'SE':>8} {'d_bpc':>10}")
    print(f"{'[<0]':>12} {pre_n:10d} {pre_m:10.6f} {pre_se:8.6f} "
          f"{pre_m * to_bpc:10.6f}")
    for (lo, hi), mu, se, k in zip(edges, means, ses, b_n):
        lab = f"[{lo}]" if lo == hi else f"[{lo}-{hi}]"
        print(f"{lab:>12} {int(k):10d} {mu:10.6f} {se:8.6f} "
              f"{mu * to_bpc:10.6f}")

    plot_junk(edges, means, ses, b_n,
              {"nats": pre_m, "se": pre_se, "n": pre_n},
              args.out,
              title=f"{a['name']}  seed {args.seed}  "
                    f"Δ={delta:+.4f} nats")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
