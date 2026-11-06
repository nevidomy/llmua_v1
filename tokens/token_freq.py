"""Token-id frequencies in a bins directory (train.bin / val.bin / shards).

Counts every id in [0, vocab_size). Prints the most- and least-used pieces
(decoded via the tokenizer named in meta.json) so unused vocab slots are
easy to spot -- e.g. candidates to steal for SFT specials.

    python -m tokens.token_freq $LLMUA_ROOT/bins/32768
    python -m tokens.token_freq $LLMUA_ROOT/bins/32768 --show 40
"""
import argparse
import json
import os

import numpy as np

from data.index import ROOT
from tokens.sharded_bins import ShardedBins


def count_split(bins, vocab_size, chunk=1 << 22):
    counts = np.zeros(vocab_size, dtype=np.int64)
    n_oob = 0
    for shard in bins.shards:
        off, n = 0, len(shard)
        while off < n:
            block = np.asarray(shard[off:off + chunk], dtype=np.int64)
            oob = block >= vocab_size
            n_oob += int(oob.sum())
            if oob.any():
                block = block[~oob]
            counts += np.bincount(block, minlength=vocab_size)
            off += chunk
    return counts, n_oob


def _fmt_tok(tok, i):
    raw = tok.id_to_token(i)
    if raw is None:
        return "?", "?"
    try:
        decoded = tok.decode([i], skip_special_tokens=False)
    except Exception:
        decoded = raw
    return raw, decoded


def print_table(ids, counts, total, tok, title):
    print(f"\n{title}")
    print(f"  {'id':>7}  {'count':>14}  {'%':>8}  {'raw':<24}  decoded")
    for i in ids:
        c = int(counts[i])
        raw, decoded = _fmt_tok(tok, int(i))
        pct = 100.0 * c / total if total else 0.0
        print(f"  {int(i):>7}  {c:>14,}  {pct:>7.4f}%  {raw!s:<24}  {decoded!r}")


def run(bins_dir, show=20, splits=("train", "val")):
    bins_dir = os.path.abspath(os.path.expanduser(bins_dir))
    meta_path = os.path.join(bins_dir, "meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    V = int(meta.get("vocab_size") or 0)
    tok_name = meta.get("tokenizer")
    tok_path = None
    if tok_name:
        cand = tok_name if os.path.isabs(tok_name) else os.path.join(ROOT, "tokenizers", tok_name)
        if os.path.exists(cand):
            tok_path = cand
    if not tok_path:
        raise SystemExit(f"no tokenizer (meta.tokenizer={tok_name!r}) under {ROOT}/tokenizers")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    if not V:
        V = tok.get_vocab_size()
    if tok.get_vocab_size() != V:
        print(f"WARNING: meta vocab {V} vs tokenizer {tok.get_vocab_size()}")

    print(f"bins       {bins_dir}")
    print(f"tokenizer  {tok_path}")
    print(f"vocab      {V:,}  bos {meta.get('bos_id')}  eos {meta.get('eos_id')}")

    total_counts = np.zeros(V, dtype=np.int64)
    grand = 0
    for split in splits:
        try:
            bins = ShardedBins.open(bins_dir, split)
        except FileNotFoundError:
            print(f"  {split}: missing, skip")
            continue
        counts, n_oob = count_split(bins, V)
        n = int(counts.sum())
        grand += n
        total_counts += counts
        n_unused = int((counts == 0).sum())
        print(f"  {split:<6}  {n:,} tokens  {n_unused:,} unused ids"
              f"{f'  {n_oob:,} id>=vocab' if n_oob else ''}")

    if grand == 0:
        raise SystemExit("no tokens counted")

    n_unused = int((total_counts == 0).sum())
    n_once = int((total_counts == 1).sum())
    mx = int(total_counts.max())
    print(f"  ALL     {grand:,} tokens  {n_unused:,} unused  "
          f"{n_once:,} used once  max {mx:,}")

    if show > 0:
        order = np.argsort(total_counts, kind="stable")
        least = order[:show]
        most = order[-show:][::-1]
        print_table(most, total_counts, grand, tok, f"most used (top {show})")
        print_table(least, total_counts, grand, tok, f"least used (bottom {show})")
    if n_unused:
        unused = np.flatnonzero(total_counts == 0)
        print(f"\n{n_unused:,} unused ids "
              f"(first {min(show, n_unused)}): {unused[:show].tolist()}")
    return total_counts


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bins_dir",
                   help="bins directory (meta.json + train.bin/val.bin or shards)")
    p.add_argument("--show", type=int, default=20,
                   help="how many most-used and least-used ids to print")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.bins_dir, show=args.show, splits=tuple(args.splits))
