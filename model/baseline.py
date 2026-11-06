"""Entropy floors: what a model that learned NOTHING but token statistics gets.

Four references, all computed exactly from the bins, all fit on train and
evaluated on val:

    uniform   ln(V)          -- the loss at initialisation
    unigram   token frequency only
    bigram    previous token only, add-1 smoothed
    bigram*   previous token only, smoothing tuned on val

The bigram is the one that matters. It has V^2 free parameters -- 1.07 BILLION
at V=32768, ten times a 110M transformer -- and its only handicap is one token
of context. A transformer that has not beaten it has not learned anything a
lookup table could not.

TWO SMOOTHING METHODOLOGIES, and the choice is not cosmetic. Add-1 (Laplace)
adds V pseudo-counts to every row. At V=8192 with 45.7M training tokens the
average row holds ~5.6k real observations, so 59% of each row's probability
mass is smoothing rather than data -- and the floor comes out inflated,
flattering whatever you compare against it. Measured at V=8192: add-1 gave
5.4939 nats/token, tuned add-0.03 gave 5.0761. The 0.42-nat difference was
entirely an artefact of the estimator.

  add-alpha       p = (c + a) / (r + a*V)
  interpolated    p = L * c/r + (1-L) * p_unigram      (Jelinek-Mercer)

Interpolation backs off to the unigram distribution rather than toward
uniform, so it wins when many val contexts are unseen. At V=8192 with 96.3%
of val bigrams seen in training, add-alpha won -- interpolation solved a
problem that did not exist.

TWO COUNTING PATHS. A dense V x V table is fine at V=8192 (537MB) and
impossible at V=32768 (8.6GB as int64). The sparse path counts only the pairs
that actually occur in val, which is at most len(val) distinct keys, by
streaming train through a sorted lookup. Both paths are checked to agree.
"""

import json
import math
import os
import sys

import numpy as np

DENSE_CELL_LIMIT = 200_000_000       # V*V above this -> sparse path


# ------------------------------------------------------------------ loading
def _load(bins_dir, split, limit=None):
    a = np.memmap(os.path.join(bins_dir, f"{split}.bin"), dtype=np.uint16,
                  mode="r")
    if limit and len(a) > limit:
        a = a[:limit]
    return np.asarray(a, dtype=np.int64)


def _meta(bins_dir):
    return json.load(open(os.path.join(bins_dir, "meta.json")))


# ------------------------------------------------------------------ unigram
def _unigram(train, val, V):
    c = np.bincount(train, minlength=V).astype(np.float64)
    p = (c + 1.0) / (c.sum() + V)
    return float(-np.log(p)[val].mean())


# ------------------------------------------------------------------- bigram
def _bigram_dense(train, val, V):
    """Returns (val_pair_counts, val_context_counts) via a full V x V table."""
    joint = np.bincount(train[:-1] * V + train[1:], minlength=V * V)
    row = np.bincount(train[:-1], minlength=V)
    vp, vn = val[:-1], val[1:]
    c = joint[vp * V + vn].astype(np.float64)
    del joint
    return c, row[vp].astype(np.float64), row


def _bigram_sparse(train, val, V, chunk=20_000_000, verbose=True):
    """Same outputs without materialising V x V.

    Only the pairs that occur in val are ever needed, so build a sorted array
    of those keys and stream train through np.searchsorted. Memory is
    O(unique val pairs), not O(V^2).
    """
    vp, vn = val[:-1], val[1:]
    vkeys = vp * V + vn
    uniq = np.unique(vkeys)                       # sorted
    counts = np.zeros(len(uniq), dtype=np.int64)
    # train[:-1]: the final token has no successor, so it is not a context
    row = np.bincount(train[:-1], minlength=V)

    n = len(train) - 1
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        seg = train[i:j + 1]                      # +1 spans the boundary pair
        keys = seg[:-1] * V + seg[1:]
        idx = np.searchsorted(uniq, keys)
        np.clip(idx, 0, len(uniq) - 1, out=idx)
        hit = uniq[idx] == keys
        if hit.any():
            counts += np.bincount(idx[hit], minlength=len(uniq))
        if verbose:
            print(f"\r    counting bigrams {j/n*100:5.1f}%", end="", flush=True)
    if verbose:
        print("\r" + " " * 40 + "\r", end="")

    back = np.searchsorted(uniq, vkeys)
    return (counts[back].astype(np.float64),
            row[vp].astype(np.float64), row)


def _bigram_scores(c, r, p_uni, vn, V, alphas, lambdas):
    """c: val pair counts, r: val context counts. Returns {name: nats}."""
    out = {}
    for a in alphas:
        out[f"add-{a:g}"] = float(-np.log((c + a) / (r + a * V)).mean())
    with np.errstate(divide="ignore", invalid="ignore"):
        mle = np.where(r > 0, c / np.maximum(r, 1), 0.0)
    for lam in lambdas:
        p = lam * mle + (1 - lam) * p_uni[vn]
        out[f"interp-{lam:g}"] = float(-np.log(p).mean())
    return out


# --------------------------------------------------------------------- api
def baselines(bins_dir, train_limit=None, val_limit=None,
              alphas=(1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001),
              lambdas=(0.5, 0.7, 0.9, 0.95, 0.99), verbose=True):
    meta = _meta(bins_dir)
    V = meta["vocab_size"]
    cpt = meta.get("chars_per_token_val")
    train = _load(bins_dir, "train", train_limit)
    val = _load(bins_dir, "val", val_limit)

    dense = V * V <= DENSE_CELL_LIMIT
    if verbose:
        print(f"{bins_dir}")
        print(f"  V {V:,}   train {len(train):,} tok   val {len(val):,} tok"
              f"   cpt(val) {cpt:.4f}")
        print(f"  bigram path: {'dense' if dense else 'sparse'} "
              f"(V^2 = {V*V:,} cells)")

    uni_c = np.bincount(train, minlength=V).astype(np.float64)
    p_uni = (uni_c + 1.0) / (uni_c.sum() + V)

    if dense:
        c, r, _ = _bigram_dense(train, val, V)
    else:
        c, r, _ = _bigram_sparse(train, val, V, verbose=verbose)

    seen = float((c > 0).mean())
    unseen_ctx = float((r == 0).mean())
    smooth_share = V / (len(train) / V + V)

    scores = _bigram_scores(c, r, p_uni, val[1:], V, alphas, lambdas)
    best = min(scores, key=scores.get)

    floors = {
        "uniform": float(math.log(V)),
        "unigram": _unigram(train, val, V),
        "bigram_add1": scores["add-1"],
        "bigram_best": scores[best],
    }

    res = {"vocab_size": V, "chars_per_token": cpt, "floors": floors,
           "bigram_variants": scores, "bigram_best_estimator": best,
           "val_bigrams_seen": seen, "val_contexts_unseen": unseen_ctx,
           "add1_smoothing_share": smooth_share,
           "n_train": len(train), "n_val": len(val)}
    if verbose:
        report(res)
    return res


def report(res, model_loss=None):
    cpt = res["chars_per_token"]
    L2 = math.log(2)

    def row(name, v):
        line = f"  {name:<28}{v:>10.4f}"
        if cpt:
            line += f"{v/cpt:>11.4f}{v/cpt/L2:>11.4f}"
        return line

    print(f"\n  add-1 puts {res['add1_smoothing_share']*100:.0f}% of each row's"
          f" mass in smoothing")
    print(f"  {res['val_bigrams_seen']*100:.1f}% of val bigrams seen in train;"
          f" {res['val_contexts_unseen']*100:.2f}% of contexts unseen")
    hdr = f"  {'floor':<28}{'nats/tok':>10}"
    if cpt:
        hdr += f"{'nats/char':>11}{'bits/char':>11}"
    print("\n" + hdr)
    for k in ("uniform", "unigram", "bigram_add1", "bigram_best"):
        label = k
        if k == "bigram_best":
            label = f"bigram_best ({res['bigram_best_estimator']})"
        print(row(label, res["floors"][k]))
    if model_loss is not None:
        print(row("YOUR MODEL", model_loss))
        b = res["floors"]["bigram_best"]
        gap = b - model_loss
        print(f"\n  {'BEATS' if gap > 0 else 'WORSE THAN'} the tuned bigram by "
              f"{abs(gap):.4f} nats/token ({abs(gap)/b*100:.1f}%)")
        u = res["floors"]["uniform"]
        print(f"  progress from uniform: model {u-model_loss:.4f} nats, "
              f"bigram {u-b:.4f} nats "
              f"({(u-model_loss)/(u-b):.2f}x as far)")


def variants_table(res):
    print(f"\n  {'estimator':<18}{'nats/tok':>10}")
    for k, v in sorted(res["bigram_variants"].items(), key=lambda kv: kv[1]):
        mark = "  <- best" if k == res["bigram_best_estimator"] else ""
        print(f"  {k:<18}{v:>10.4f}{mark}")


# -------------------------------------------------------------------- test
if __name__ == "__main__":
    print("=== sparse path must equal dense path ===")
    rng = np.random.default_rng(0)
    V = 256
    trans = rng.dirichlet(np.full(V, 0.05), size=V)
    cum = trans.cumsum(axis=1)
    n = 300_000
    seq = np.zeros(n, dtype=np.int64)
    r = rng.random(n)
    for i in range(1, n):
        seq[i] = np.searchsorted(cum[seq[i - 1]], r[i])
    tr, va = seq[:260_000], seq[260_000:]

    cd, rd, _ = _bigram_dense(tr, va, V)
    cs, rs, _ = _bigram_sparse(tr, va, V, chunk=7_000, verbose=False)
    assert np.array_equal(cd, cs), "pair counts differ"
    assert np.array_equal(rd, rs), "context counts differ"
    print(f"  {len(cd):,} val pairs, counts identical (dense == sparse)")

    print("\n  chunk size must not matter (boundary pairs):")
    for ch in (1_000, 7_000, 100_000, 10_000_000):
        c2, _, _ = _bigram_sparse(tr, va, V, chunk=ch, verbose=False)
        assert np.array_equal(cd, c2), f"chunk={ch} differs"
        print(f"    chunk {ch:>10,}: identical")

    print("\n=== end to end on synthetic bins ===")
    os.makedirs("/tmp/bl", exist_ok=True)
    tr.astype(np.uint16).tofile("/tmp/bl/train.bin")
    va.astype(np.uint16).tofile("/tmp/bl/val.bin")
    json.dump({"vocab_size": V, "chars_per_token_val": 3.5},
              open("/tmp/bl/meta.json", "w"))
    res = baselines("/tmp/bl", verbose=True)
    variants_table(res)
    f = res["floors"]
    assert f["uniform"] > f["unigram"] > f["bigram_add1"] >= f["bigram_best"]
    print("\n  floors correctly ordered, tuned <= add-1")
    report(res, model_loss=f["bigram_best"] * 0.8)