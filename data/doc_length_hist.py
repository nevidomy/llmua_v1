"""
Document-length distribution over a tokenized .bin (uint16 token ids).

Documents are delimited by BOS. Length is measured BOS-to-BOS, i.e. it
includes the document's own BOS and EOS, which is what the tokenizer's
reported chars/token denominator uses.

Two numbers matter for choosing sequence length T, and they are different:

  - % of DOCUMENTS that fit in T        -- doc-weighted, dominated by short docs
  - % of TOKENS in documents that fit   -- token-weighted, this is the one that
                                           says how much of your training signal
                                           comes from documents that get split

A corpus can have 80% of documents fitting in T while only 40% of tokens do,
because the long documents carry disproportionate mass.

Also reported: under plain packing with no document masking, the fraction of
positions whose full within-document history is available -- an upper bound on
how much context the model can actually use.

Usage:
    python doc_length_hist.py /path/to/train_32768.bin --bos 0
    python doc_length_hist.py ... --bos 0 --candidates 512 1024 2048
"""

import argparse

import numpy as np


def find_bos_positions(path: str, bos_id: int, chunk: int = 1 << 27) -> tuple:
    """Stream the bin, collecting positions of BOS. Returns (positions, total)."""
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    total = arr.shape[0]
    out = []
    for start in range(0, total, chunk):
        block = np.asarray(arr[start : start + chunk])
        hits = np.flatnonzero(block == bos_id)
        if hits.size:
            out.append(hits.astype(np.int64) + start)
        del block
    positions = np.concatenate(out) if out else np.zeros(0, dtype=np.int64)
    return positions, total


def doc_lengths(positions: np.ndarray, total: int) -> np.ndarray:
    """BOS-to-BOS distances, with the final document running to end of file."""
    if positions.size == 0:
        return np.zeros(0, dtype=np.int64)
    lengths = np.diff(positions)
    tail = total - positions[-1]
    return np.append(lengths, tail)


def ascii_hist(lengths: np.ndarray, bins: int = 20, width: int = 50) -> None:
    """Log-spaced histogram, since document lengths are heavy-tailed."""
    lo = max(1, int(lengths.min()))
    hi = int(lengths.max())
    edges = np.unique(np.geomspace(lo, hi, bins + 1).astype(np.int64))
    counts, edges = np.histogram(lengths, bins=edges)
    peak = counts.max() if counts.size else 1
    print(f"  {'range (tokens)':>22}  {'docs':>10}  {'%':>6}")
    for i, c in enumerate(counts):
        bar = "#" * int(width * c / peak)
        rng = f"{edges[i]:,}-{edges[i+1]:,}"
        pct = 100.0 * c / lengths.size
        print(f"  {rng:>22}  {c:>10,}  {pct:>5.1f}%  {bar}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", help="path to the uint16 .bin")
    p.add_argument("--bos", type=int, required=True, help="BOS token id")
    p.add_argument("--expect-docs", type=int, default=None,
                   help="assert this many documents were found")
    p.add_argument("--candidates", type=int, nargs="+",
                   default=[256, 512, 1024, 2048, 4096])
    args = p.parse_args()

    positions, total = find_bos_positions(args.path, args.bos)
    lengths = doc_lengths(positions, total)

    if lengths.size == 0:
        raise SystemExit(f"no BOS (id={args.bos}) found -- wrong id or wrong dtype?")
    if args.expect_docs is not None and lengths.size != args.expect_docs:
        raise SystemExit(
            f"found {lengths.size:,} docs, expected {args.expect_docs:,}. "
            "BOS id may also occur as an ordinary token."
        )

    tok_total = lengths.sum()
    print(f"file        {args.path}")
    print(f"tokens      {total:,}")
    print(f"documents   {lengths.size:,}")
    print(f"mean len    {lengths.mean():,.1f}")
    print()

    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    vals = np.percentile(lengths, qs)
    print("  quantile   tokens")
    for q, v in zip(qs, vals):
        label = "median" if q == 50 else f"p{q:g}"
        print(f"  {label:>8}   {v:>10,.0f}")
    print(f"  {'max':>8}   {lengths.max():>10,}")
    print()

    print("document length distribution (log-spaced)")
    ascii_hist(lengths)
    print()

    print("sequence-length candidates")
    print(f"  {'T':>6}  {'% docs fit':>11}  {'% tokens in fitting docs':>25}  "
          f"{'% positions w/ full ctx':>24}")
    for T in args.candidates:
        fits = lengths <= T
        pct_docs = 100.0 * fits.sum() / lengths.size
        pct_tok = 100.0 * lengths[fits].sum() / tok_total
        # Under plain packing, a position has its full within-doc history only
        # if its offset in the document is < T. For a doc of length L that is
        # min(L, T) positions.
        full_ctx = np.minimum(lengths, T).sum()
        pct_ctx = 100.0 * full_ctx / tok_total
        print(f"  {T:>6}  {pct_docs:>10.1f}%  {pct_tok:>24.1f}%  {pct_ctx:>23.1f}%")


if __name__ == "__main__":
    main()