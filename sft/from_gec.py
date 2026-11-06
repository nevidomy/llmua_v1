"""UA-GEC fluency sentence files -> {question, answer} JSON for SFT.

Each `{id}.src.txt` in source-sentences is line-aligned with every
`{id}.a*.txt` in target-sentences (a1/a2 are separate examples).

    Q = 'Виправ помилки: "{source}"'
    A = '{target}'

    python prepare_sft.py gec \\
      --root ~/data/ua-gec/data/gec-fluency \\
      --out-dir ~/data/llmua/sft_data/text --yes
"""
import argparse
import os

from data.index import ROOT
from data.shard_util import confirm, utcnow, write_json
from sft.pairs import clean_part, drop_old_splits, write_pairs

PROMPT = "Виправ помилки:"


def convert_row(source, target):
    source = clean_part(source)
    target = clean_part(target)
    if not source or not target:
        return None
    pair = {"question": f'{PROMPT} "{source}"', "answer": target}
    return pair, source == target


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def _src_files(src_dir):
    return sorted(
        os.path.join(src_dir, n) for n in os.listdir(src_dir)
        if n.endswith(".src.txt"))


def _tgt_files(tgt_dir, stem):
    prefix = stem + "."
    return sorted(
        os.path.join(tgt_dir, n) for n in os.listdir(tgt_dir)
        if n.startswith(prefix) and n.endswith(".txt"))


def from_split(src_dir, tgt_dir):
    """Return (kept, skipped, n_in, n_identical, n_tgt_files)."""
    kept, skipped, n_in, n_ident, n_tgt = [], 0, 0, 0, 0
    src_dir = os.path.expanduser(src_dir)
    tgt_dir = os.path.expanduser(tgt_dir)
    sources = _src_files(src_dir)
    if not sources:
        raise SystemExit(f"no *.src.txt in {src_dir}")
    for src_path in sources:
        stem = os.path.basename(src_path).split(".")[0]
        src_lines = _read_lines(src_path)
        tgts = _tgt_files(tgt_dir, stem)
        if not tgts:
            raise SystemExit(f"no target files for {stem} in {tgt_dir}")
        for tgt_path in tgts:
            n_tgt += 1
            tgt_lines = _read_lines(tgt_path)
            if len(src_lines) != len(tgt_lines):
                raise SystemExit(
                    f"line count mismatch {src_path} ({len(src_lines)}) vs "
                    f"{tgt_path} ({len(tgt_lines)})")
            for src, tgt in zip(src_lines, tgt_lines):
                n_in += 1
                conv = convert_row(src, tgt)
                if conv is None:
                    skipped += 1
                    continue
                pair, same = conv
                if same:
                    n_ident += 1
                kept.append(pair)
    return kept, skipped, n_in, n_ident, n_tgt


def run(args):
    root = os.path.abspath(os.path.expanduser(args.root))
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    dest = os.path.join(out_dir, "uagec.json")
    jobs = []
    if args.train:
        jobs.append(("train",
                     os.path.join(root, "train", "source-sentences"),
                     os.path.join(root, "train", "target-sentences")))
    if args.test:
        jobs.append(("test",
                     os.path.join(root, "test", "source-sentences"),
                     os.path.join(root, "test", "target-sentences")))
    if not jobs:
        raise SystemExit("give --train and/or --test (default: both)")

    print("ua-gec fluency -> {question, answer}")
    print(f"  root  {root}")
    print(f"         -> {dest}")
    for split, src_dir, tgt_dir in jobs:
        n_src = (len(_src_files(src_dir)) if os.path.isdir(src_dir) else 0)
        n_tgt = (len(os.listdir(tgt_dir)) if os.path.isdir(tgt_dir) else 0)
        print(f"  {split:<5}  {src_dir}  ({n_src} src files)")
        print(f"         {tgt_dir}  ({n_tgt} tgt files)")

    if not args.dry_run:
        confirm(args.yes)
        os.makedirs(out_dir, exist_ok=True)
    man = {"step": "sft_gec", "created": utcnow(), "root": root,
           "out": out_dir, "json": os.path.basename(dest), "files": []}
    all_kept = []
    for split, src_dir, tgt_dir in jobs:
        kept, skipped, n_in, n_ident, n_tgt = from_split(src_dir, tgt_dir)
        all_kept.extend(kept)
        rec = {"split": split, "src_dir": src_dir, "tgt_dir": tgt_dir,
               "json": os.path.basename(dest), "n_in": n_in,
               "n_out": len(kept), "n_skipped": skipped,
               "n_identical": n_ident, "n_tgt_files": n_tgt}
        man["files"].append(rec)
        print(f"  {split}: {n_in:,} line-pairs -> {len(kept):,} "
              f"(skipped {skipped:,} empty; {n_ident:,} already correct)")
    print(f"  total {len(all_kept):,} pairs")
    if args.dry_run:
        print("\ndry-run: not writing")
        return man
    write_pairs(dest, all_kept)
    drop_old_splits(out_dir, "uagec_train.json", "uagec_test.json")
    write_json(os.path.join(out_dir, "manifest_gec.json"), man)
    print(f"manifest -> {out_dir}/manifest_gec.json")
    return man


def parse_args(argv=None, prog=None):
    p = argparse.ArgumentParser(
        prog=prog, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root",
                   default=os.path.expanduser(
                       "~/data/ua-gec/data/gec-fluency"))
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

    def write(path, text):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "gec-fluency")
        src = os.path.join(root, "train", "source-sentences")
        tgt = os.path.join(root, "train", "target-sentences")
        write(os.path.join(src, "0001.src.txt"),
              "я йду до школи\n\nкиїв це місто\n")
        write(os.path.join(tgt, "0001.a1.txt"),
              "я йду до школи\n\nкиїв — це місто\n")
        write(os.path.join(tgt, "0001.a2.txt"),
              "я йду у школу\n\nкиїв — це місто\n")
        kept, skipped, n_in, n_ident, n_tgt = from_split(src, tgt)
        assert n_tgt == 2 and n_in == 6 and skipped == 2 and len(kept) == 4
        assert n_ident == 1  # "Я йду до школи" == a1
        assert kept[0]["question"] == f'{PROMPT} "Я йду до школи"'
        assert kept[0]["answer"] == "Я йду до школи"
        assert kept[1]["answer"] == "Київ — це місто"
        assert kept[2]["answer"] == "Я йду у школу"
        ns = argparse.Namespace(root=root, out_dir=td, yes=True,
                                dry_run=False, train=True, test=False)
        man = run(ns)
        assert man["files"][0]["n_out"] == 4
        assert os.path.exists(os.path.join(td, "uagec.json"))
    print("sft.from_gec self-test ok")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        run(parse_args())
