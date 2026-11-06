"""Sizing sources, taking one in full, and subsampling for tokenizer training.

Two things this solves:

1. "Wikipedia in full, FineWeb fills the rest to 8B." Wiki's size is not known
   until consumed, so measure it from Parquet footers first (fast, no
   decompression), then set FineWeb's budget to the remainder.

2. "Does the BPE train on the whole corpus?" No, and it should not -- merge
   statistics converge on 1-3 GB. But the tokenizer sample MUST have the same
   source mix as the training corpus, and the corpus is written source by
   source, so taking the first N lines gives pure FineWeb. Sample every Nth.
"""

import glob
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def estimate_source_chars(pattern, column="text", probe_row_groups=3):
    """Character count for a Parquet source.

    Footers give UNCOMPRESSED BYTES for the column without decompressing
    anything. One decompressed row group calibrates bytes-per-character, which
    for Ukrainian runs ~1.8 (Cyrillic is 2 bytes in UTF-8, ASCII is 1).
    """
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise FileNotFoundError(pattern)

    nbytes = nrows = 0
    for p in shards:
        md = pq.ParquetFile(p).metadata
        nrows += md.num_rows
        for rg in range(md.num_row_groups):
            g = md.row_group(rg)
            for c in range(md.num_columns):
                cc = g.column(c)
                if cc.path_in_schema == column:
                    nbytes += cc.total_uncompressed_size

    # calibrate bytes/char on a few real row groups
    pf = pq.ParquetFile(shards[0])
    ch = by = 0
    for rg in range(min(probe_row_groups, pf.metadata.num_row_groups)):
        for b in pf.iter_batches(batch_size=2000, columns=[column],
                                 row_groups=[rg]):
            for d in b.column(column).to_pylist():
                if d:
                    ch += len(d)
                    by += len(d.encode())
    bpc = by / ch if ch else 1.8

    chars = nbytes / bpc
    print(f"{pattern}")
    print(f"  {len(shards)} shards, {nrows:,} rows")
    print(f"  {nbytes/1e9:.2f} GB uncompressed  @ {bpc:.3f} bytes/char")
    print(f"  ~= {chars/1e9:.2f}B chars  (footer estimate, pre-cleaning)")
    return chars


def _shard_stats(path, column, batch_size):
    """Exact nonempty-doc / character counts for one Parquet file.

    Arrow utf8_length on the column — no Python strings. Empty and null rows
    are skipped, matching sample_sources.
    """
    ndocs = nchars = 0
    for b in pq.ParquetFile(path).iter_batches(batch_size=batch_size,
                                               columns=[column]):
        col = b.column(column)
        lengths = pc.utf8_length(col)
        nonempty = pc.and_(pc.is_valid(col), pc.greater(lengths, 0))
        ndocs += int(pc.sum(pc.cast(nonempty, pa.int64())).as_py() or 0)
        nchars += int(pc.sum(lengths).as_py() or 0)
    return ndocs, nchars


def source_stats(sources, column="text", batch_size=1 << 16, workers=None,
                 verbose=True):
    """Exact document and character counts per source.

    Same plan as sample_sources: [(glob_pattern, target_chars), ...].
    Inventories the WHOLE source — the budget is only printed for context.
    For a footer estimate that does not decompress, see estimate_source_chars.

    Faster than summing len(d) over the generator: batch utf8_length, shards
    in parallel (decode releases the GIL).
    """
    rows = []
    for pattern, target in sources:
        shards = sorted(glob.glob(pattern))
        if not shards:
            raise FileNotFoundError(f"no files match {pattern!r}")
        n = workers if workers is not None else min(16, len(shards),
                                                    os.cpu_count() or 4)
        if n <= 1 or len(shards) == 1:
            parts = [_shard_stats(p, column, batch_size) for p in shards]
        else:
            with ThreadPoolExecutor(max_workers=n) as pool:
                parts = list(pool.map(
                    lambda p: _shard_stats(p, column, batch_size), shards))
        ndocs = sum(d for d, _ in parts)
        nchars = sum(c for _, c in parts)
        row = {"pattern": pattern, "shards": len(shards),
               "ndocs": ndocs, "nchars": nchars, "target": target}
        rows.append(row)
        if verbose:
            scale, unit = (1e9, "B") if nchars >= 1e9 else (1e6, "M")
            msg = (f"{pattern}: {ndocs:,} docs, {nchars/scale:.2f}{unit} chars "
                   f"({len(shards)} shards)")
            if target < 1e18:
                msg += f"  ({nchars / target * 100:.0f}% of budget)"
                if nchars < target:
                    msg += "  <-- EXHAUSTED, under budget"
            print(msg)
    return rows


def plan_mix(sources_full, total_chars, verbose=True):
    """sources_full: [(pattern, 'all' | fraction | absolute_chars), ...]

    'all' sources are measured and taken whole; the remainder is split across
    the rest. Returns the (pattern, budget) list sample_sources expects.
    """
    measured = {}
    used = 0
    for pat, spec in sources_full:
        if spec == "all":
            measured[pat] = estimate_source_chars(pat)
            used += measured[pat]

    remainder = max(0.0, total_chars - used)
    if used > total_chars:
        print(f"\nWARNING: 'all' sources supply {used/1e9:.2f}B, above the "
              f"{total_chars/1e9:.2f}B target -- nothing left for the rest")
    out = []
    for pat, spec in sources_full:
        if spec == "all":
            # a budget above the source's size makes sample_sources exhaust it
            out.append((pat, int(measured[pat] * 1.5)))
        elif isinstance(spec, float) and spec <= 1.0:
            out.append((pat, int(remainder * spec)))
        else:
            out.append((pat, int(spec)))
    if verbose:
        print(f"\nplan for {total_chars/1e9:.2f}B chars total:")
        for pat, budget in out:
            tag = " (full source)" if any(p == pat and s == "all"
                                          for p, s in sources_full) else ""
            print(f"  {budget/1e9:>6.2f}B  {pat.split('/')[-1]:<28}{tag}")
        print(f"  wiki share of total: {used/total_chars*100:.1f}%")
    return out


def subsample_jsonl(src, dst, target_chars, seed=0, verbose=True):
    """Mix-preserving subsample for tokenizer training.

    The corpus is written source by source, so the first N lines are one
    source only. Systematic every-Nth sampling holds the mix; a random
    Bernoulli keep does too and needs no length pass.
    """
    rng = random.Random(seed)
    # pass 1: total size, to pick the keep probability
    total = n = 0
    with open(src, encoding="utf-8") as f:
        for line in f:
            total += len(line) - 2          # minus quotes; close enough
            n += 1
    p = min(1.0, target_chars / max(total, 1))

    kept = kept_chars = 0
    with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
        for line in fi:
            if rng.random() < p:
                fo.write(line)
                kept += 1
                kept_chars += len(line) - 2
    if verbose:
        print(f"{n:,} docs / {total/1e9:.2f}B chars -> "
              f"{kept:,} docs / {kept_chars/1e9:.2f}B chars "
              f"(p={p:.4f}) -> {dst}")
    return kept
