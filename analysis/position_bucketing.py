"""Per-position loss analysis ("position bucketing") for a trained run.

Answers: is the model still using context at the end of its window?
At window position t the model has t tokens of context, so L(t) falling
means context is still being used. If it is still falling at t = T-1,
a longer window has something to buy; if it flatlines by t ~ 512, it
doesn't.

    python -m analysis.position_bucketing /path/to/ckpt_final.pt
    python -m analysis.position_bucketing ckpt.pt --windows 1000 --batch 32

Per-token losses come from the model's reduce=False forward over
non-overlapping windows of the val split (N ~ 6,098 for the 32k bins).
Positions are bucketed logarithmically -- [1], [2-3], [4-7], ... -- and
the per-WINDOW bucket mean is the statistical unit: tokens within a
window are correlated, so raw token counts would overstate precision.
SEs are across windows (N = windows). bpc = nats / chars_per_token / ln2,
same conversion as train.log.

Architecture is restored from the checkpoint's model_arch field; for
legacy checkpoints re-pass the train-time flags (--model P1 --enable_*).
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from tqdm import tqdm

from data.index import ROOT
from model.transformer import (Transformer, model_P1, model_P2, model_P3,
                               model_P4, model_target, model_test,
                               doc_block_mask)

FACTORIES = {"test": model_test, "P1": model_P1, "P2": model_P2,
             "P3": model_P3, "P4": model_P4, "target": model_target}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt", help="checkpoint file to analyze")
    p.add_argument("--windows", type=int, default=0,
                   help="val windows to eval; 0 = all that fit the split "
                        "(~6,098 for the 32k bins)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--split", default="val")
    p.add_argument("--bins_dir", default=None,
                   help="default derived from model vocab: "
                        "{ROOT}/bins/{vocab_size}")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    # legacy checkpoints only (no model_arch stored)
    p.add_argument("--model", choices=list(FACTORIES), default=None)
    p.add_argument("--enable_init", action="store_true")
    p.add_argument("--enable_swiglu", action="store_true")
    p.add_argument("--enable_rms_norm", action="store_true")
    p.add_argument("--enable_qk_norm", action="store_true")
    return p.parse_args()


from sample_generator import load_model  # shared: model_arch or legacy flags


def bucket_edges(T):
    """Logarithmic position buckets: [1], [2-3], [4-7], ... Position 0
    (zero context) is excluded from bucketing and reported separately."""
    edges, lo = [], 1
    while lo < T:
        hi = min(2 * lo - 1, T - 1)
        edges.append((lo, hi))
        lo = hi + 1
    return edges


def analyze(ckpt_path, windows=0, batch=16, split="val", bins_dir=None,
            device=None, model=None, enable_init=False, enable_swiglu=False,
            enable_rms_norm=False, enable_qk_norm=False, progress_cb=None):
    """Full position-bucketing analysis; returns a plain-JSON-able dict.
    progress_cb(done_batches, total_batches) is called after each batch."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m, ck = load_model(ckpt_path, device, model=model,
                       enable_init=enable_init, enable_swiglu=enable_swiglu,
                       enable_rms_norm=enable_rms_norm,
                       enable_qk_norm=enable_qk_norm)
    T = m.context_width

    bins_dir = bins_dir or f"{ROOT}/bins/{m.vocab_size}"
    meta = json.load(open(os.path.join(bins_dir, "meta.json")))
    assert meta["vocab_size"] == m.vocab_size, (
        f"bins are vocab {meta['vocab_size']}, model is "
        f"{m.vocab_size} -- WRONG BINS FOR THIS MODEL")
    cpt = meta[f"chars_per_token_{split}"]
    to_bpc = 1.0 / (cpt * math.log(2))

    data = np.memmap(os.path.join(bins_dir, f"{split}.bin"),
                     dtype=np.uint16, mode="r")
    n_all = (len(data) - 1) // T
    n = min(windows or n_all, n_all)

    edges = bucket_edges(T)
    ctx = (torch.autocast(device_type=device.split(":")[0],
                          dtype=torch.bfloat16)
           if device != "cpu" else torch.autocast("cpu", enabled=False))

    W, pos0 = [], []          # per-window bucket means; per-window pos-0 loss
    total_batches = (n + batch - 1) // batch
    with torch.no_grad():
        for bi, start in enumerate(range(0, n, batch)):
            b = min(batch, n - start)
            ix = (start + np.arange(b)) * T
            x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
            y = np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)
            # create_block_mask stays outside torch.compile(model), same as
            # the trainer; doc_rope positions are derived inside forward.
            bm = (doc_block_mask(x, m.bos_id)
                  if getattr(m, "doc_masking", False) else None)
            with ctx:
                _, losses = m(x, y, reduce=False, block_mask=bm)  # [B, T] nats
            losses = losses.float().cpu().numpy().astype(np.float64)
            W.append(np.stack([losses[:, lo:hi + 1].mean(axis=1)
                               for lo, hi in edges], axis=1))
            pos0.append(losses[:, 0])
            if progress_cb:
                progress_cb(bi + 1, total_batches)
    W = np.concatenate(W)                                  # [N, n_buckets]
    pos0 = np.concatenate(pos0)
    N = W.shape[0]

    buckets, prev = [], None
    for b, (lo, hi) in enumerate(edges):
        w = W[:, b]
        mean, se = w.mean(), w.std(ddof=1) / math.sqrt(N)
        ent = {"label": f"[{lo}]" if lo == hi else f"[{lo}-{hi}]",
               "lo": lo, "hi": hi, "pos": hi - lo + 1,
               "nats": float(mean), "bpc": float(mean * to_bpc),
               "se_nats": float(se), "se_bpc": float(se * to_bpc),
               "delta_bpc": None, "delta_se_bpc": None,
               "delta_nats": None, "delta_se_nats": None}
        if prev is not None:
            d = w - prev                        # paired: same windows
            dm, dse = d.mean(), d.std(ddof=1) / math.sqrt(N)
            ent["delta_bpc"] = float(dm * to_bpc)
            ent["delta_se_bpc"] = float(dse * to_bpc)
            ent["delta_nats"] = float(dm)
            ent["delta_se_nats"] = float(dse)
        buckets.append(ent)
        prev = w

    # the decision lives in the tail: is the last bucket still falling?
    d = W[:, -1] - W[:, -2]
    dm, dse = d.mean() * to_bpc, d.std(ddof=1) / math.sqrt(N) * to_bpc
    z = dm / dse if dse > 0 else 0.0
    verdict = ("STILL FALLING -- longer window has something to buy"
               if dm < 0 and abs(z) > 2 else
               "flat -- longer window buys nothing")

    val = ck.get("final") or (ck.get("history") or [None])[-1] or {}
    a = m.arch
    return {
        "model": a["name"], "desc": a.get("description") or "",
        "ckpt": ckpt_path, "step": ck.get("step"),
        "tokens": ck.get("tokens", 0),
        "val_nats": val.get("val_nats"), "val_bpc": val.get("val_bpc"),
        "split": split, "n_windows": int(N), "T": T,
        "pos0_nats": float(pos0.mean()), "pos0_bpc": float(pos0.mean() * to_bpc),
        "buckets": buckets,
        "tail": {"from": buckets[-2]["label"], "to": buckets[-1]["label"],
                 "dm_bpc": float(dm), "dse_bpc": float(dse),
                 "dm_nats": float(dm / to_bpc), "dse_nats": float(dse / to_bpc),
                 "z": float(z), "verdict": verdict},
    }


def main():
    args = parse_args()
    # tqdm shim over the analyze() progress callback
    state = {}
    def cb(done, total):
        if "bar" not in state:
            state["bar"] = tqdm(total=total, desc="eval")
        state["bar"].update(done - state["bar"].n)
    r = analyze(args.ckpt, windows=args.windows, batch=args.batch,
                split=args.split, bins_dir=args.bins_dir, device=args.device,
                model=args.model, enable_init=args.enable_init,
                enable_swiglu=args.enable_swiglu,
                enable_rms_norm=args.enable_rms_norm,
                enable_qk_norm=args.enable_qk_norm, progress_cb=cb)
    if "bar" in state:
        state["bar"].close()

    print(f"\nmodel  {r['model']} -- {r['desc']}")
    print(f"ckpt   {r['ckpt']}: step {r['step']}, "
          f"{r['tokens']/1e6:.1f}M tokens trained")
    if r["val_nats"] is not None:
        print(f"val    {r['val_nats']:.4f} nats/tok = "
              f"{r['val_bpc']:.4f} bpc (from checkpoint)")
    print(f"data   {r['split']}.bin: {r['n_windows']:,} non-overlapping "
          f"windows of T={r['T']} "
          f"({r['n_windows'] * r['T'] / 1e6:.1f}M tokens); "
          f"SEs use N={r['n_windows']:,} windows")
    print(f"pos 0  (no context): {r['pos0_nats']:.4f} nats = "
          f"{r['pos0_bpc']:.4f} bpc -- excluded from buckets\n")

    print(f"{'bucket':>12} {'pos':>5} {'nats':>8} {'bpc':>8}  "
          f"{'SE bpc':>7}  {'delta bpc vs prev':>18}")
    for b in r["buckets"]:
        delta = (f"{b['delta_bpc']:+.4f} +/- {b['delta_se_bpc']:.4f}"
                 if b["delta_bpc"] is not None else "--")
        print(f"{b['label']:>12} {b['pos']:>5} {b['nats']:>8.4f} "
              f"{b['bpc']:>8.4f}  {b['se_bpc']:>7.4f}  {delta:>18}")

    t = r["tail"]
    print(f"\ntail   {t['from']} -> {t['to']}: {t['dm_bpc']:+.4f} +/- "
          f"{t['dse_bpc']:.4f} bpc (z={t['z']:+.1f}) -- {t['verdict']}")


if __name__ == "__main__":
    main()
