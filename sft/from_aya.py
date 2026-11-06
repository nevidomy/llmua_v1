"""Aya dataset parquet -> {question, answer} JSON for SFT.

Keeps rows with language_code=ukr. Q is `inputs`, A is `targets`.

    python prepare_sft.py aya \\
      --dir ~/data/CohereLabs/aya_dataset/data \\
      --out-dir ~/data/llmua/sft_data/text --yes
"""
import argparse
import glob
import os

import pyarrow.parquet as pq

from data.index import ROOT
from data.shard_util import confirm, utcnow, write_json
from sft.pairs import clean_part, drop_old_splits, write_pairs

LANG = "ukr"


def convert_row(question, answer):
    q = clean_part(question)
    a = clean_part(answer)
    if not q or not a:
        return None
    return {"question": q, "answer": a}


def _pick_col(names, *cands):
    lower = {n.lower(): n for n in names}
    for c in cands:
        if c in names:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    raise SystemExit(f"parquet missing {cands}; have {list(names)}")


def from_parquet(path, lang=LANG):
    """Return (kept, n_rows, n_other, n_empty)."""
    path = os.path.expanduser(path)
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    q_col = _pick_col(names, "inputs", "input")
    a_col = _pick_col(names, "targets", "target")
    lang_col = _pick_col(names, "language_code", "language_column", "language")
    kept, n_rows, n_other, n_empty = [], 0, 0, 0
    want = lang.strip().lower()
    for batch in pf.iter_batches(batch_size=4096,
                                 columns=[q_col, a_col, lang_col]):
        cols = batch.to_pydict()
        for q, a, code in zip(cols[q_col], cols[a_col], cols[lang_col]):
            n_rows += 1
            if (code or "").strip().lower() != want:
                n_other += 1
                continue
            pair = convert_row(q, a)
            if pair is None:
                n_empty += 1
            else:
                kept.append(pair)
    return kept, n_rows, n_other, n_empty


def _split_files(data_dir, split):
    """train-*.parquet / test-*.parquet under data_dir, sorted."""
    pat = os.path.join(os.path.expanduser(data_dir), f"{split}-*.parquet")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"no files match {pat!r}")
    return files


def run(args):
    data_dir = os.path.abspath(os.path.expanduser(args.dir))
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    dest = os.path.join(out_dir, "aya.json")
    jobs = []
    if args.train:
        jobs.append(("train", _split_files(data_dir, "train")))
    if args.test:
        jobs.append(("test", _split_files(data_dir, "test")))
    if not jobs:
        raise SystemExit("give --train and/or --test (default: both)")

    print(f"aya -> {{question, answer}}  lang={args.lang}")
    print(f"  dir   {data_dir}")
    print(f"         -> {dest}")
    for split, srcs in jobs:
        for src in srcs:
            sz = os.path.getsize(src) if os.path.exists(src) else 0
            print(f"  {split:<5}  {src}  ({sz/1e6:.1f} MB)")

    if not args.dry_run:
        confirm(args.yes)
        os.makedirs(out_dir, exist_ok=True)
    man = {"step": "sft_aya", "created": utcnow(), "dir": data_dir,
           "lang": args.lang, "out": out_dir,
           "json": os.path.basename(dest), "files": []}
    all_kept = []
    for split, srcs in jobs:
        kept, n_rows, n_other, n_empty = [], 0, 0, 0
        for src in srcs:
            k, nr, no, ne = from_parquet(src, lang=args.lang)
            kept.extend(k)
            n_rows += nr
            n_other += no
            n_empty += ne
        all_kept.extend(kept)
        rec = {"split": split, "src": srcs, "json": os.path.basename(dest),
               "n_in": n_rows, "n_other_lang": n_other, "n_empty": n_empty,
               "n_out": len(kept)}
        man["files"].append(rec)
        print(f"  {split}: {n_rows:,} rows  {n_other:,} other-lang  "
              f"{n_empty:,} empty  -> {len(kept):,} {args.lang} pairs")
    print(f"  total {len(all_kept):,} {args.lang} pairs")
    if args.dry_run:
        print("\ndry-run: not writing")
        return man
    if not all_kept:
        print(f"         skip write (no {args.lang} rows)")
        man["json"] = None
    else:
        write_pairs(dest, all_kept)
        drop_old_splits(out_dir, "aya_train.json", "aya_test.json")
    write_json(os.path.join(out_dir, "manifest_aya.json"), man)
    print(f"manifest -> {out_dir}/manifest_aya.json")
    return man


def parse_args(argv=None, prog=None):
    p = argparse.ArgumentParser(
        prog=prog, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir",
                   default=os.path.expanduser(
                       "~/data/CohereLabs/aya_dataset/data"))
    p.add_argument("--lang", default=LANG,
                   help="language_code to keep (default: ukr)")
    p.add_argument("--out-dir",
                   default=os.path.join(ROOT, "sft_data", "text"))
    p.add_argument("--yes", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--no-test", action="store_true")
    args = p.parse_args(argv)
    args.train = not args.no_train
    args.test = not args.no_test
    return args


def _self_test():
    import tempfile
    import pyarrow as pa

    with tempfile.TemporaryDirectory() as td:
        data = os.path.join(td, "data")
        os.makedirs(data)
        table = pa.table({
            "inputs": ["київ?", "hello", ""],
            "targets": ["столиця", "hi", "x"],
            "language_code": ["ukr", "eng", "ukr"],
        })
        pq.write_table(table, os.path.join(data, "train-00000-of-00001.parquet"))
        pq.write_table(table.slice(1, 1),
                       os.path.join(data, "test-00000-of-00001.parquet"))
        kept, n_rows, n_other, n_empty = from_parquet(
            os.path.join(data, "train-00000-of-00001.parquet"))
        assert n_rows == 3 and n_other == 1 and n_empty == 1 and len(kept) == 1
        assert kept[0]["question"] == "Київ?"
        assert kept[0]["answer"] == "Столиця"
        ns = argparse.Namespace(dir=data, lang="ukr", out_dir=td,
                                yes=True, dry_run=False,
                                train=True, test=True)
        man = run(ns)
        assert man["files"][0]["n_out"] == 1
        assert os.path.exists(os.path.join(td, "aya.json"))
        assert man["files"][1]["n_out"] == 0
        assert not os.path.exists(os.path.join(td, "aya_test.json"))
    print("sft.from_aya self-test ok")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        run(parse_args())
