"""fido-ua-news CSV -> {question, answer} JSON for SFT.

Columns: title, text, tags, target. Prompt is classification:

    Q = "До якої категорії належить стаття: {text}"
    A = "{target}"

    python prepare_sft.py news \\
      --train ~/data/fido-ua-news/train.csv \\
      --test ~/data/fido-ua-news/test.csv \\
      --out-dir ~/data/llmua/sft_data/text --yes
"""
import argparse
import csv
import os
from collections import Counter

from data.index import ROOT
from data.shard_util import confirm, utcnow, write_json
from sft.pairs import clean_part, drop_old_splits, write_pairs

PROMPT = "До якої категорії належить стаття:"


def convert_row(text, target):
    text = clean_part(text)
    target = clean_part(target)
    if not text or not target:
        return None
    return {"question": f"{PROMPT} {text}", "answer": target}


def from_csv(path):
    kept, skipped, n_in = [], 0, 0
    with open(os.path.expanduser(path), encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return kept, skipped, n_in
        fields = {n.strip().lower(): n for n in reader.fieldnames}
        text_key = fields.get("text") or fields.get("test")
        tgt_key = fields.get("target")
        if not text_key or not tgt_key:
            raise SystemExit(
                f"{path}: need text/target columns, have {reader.fieldnames}")
        for row in reader:
            n_in += 1
            pair = convert_row(row.get(text_key), row.get(tgt_key))
            if pair is None:
                skipped += 1
            else:
                kept.append(pair)
    return kept, skipped, n_in


def run(args):
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    dest = os.path.join(out_dir, "fidonews.json")
    jobs = []
    if args.train:
        jobs.append(("train", os.path.expanduser(args.train)))
    if args.test:
        jobs.append(("test", os.path.expanduser(args.test)))
    if not jobs:
        raise SystemExit("give --train and/or --test")

    print("fido-ua-news -> {question, answer}")
    print(f"         -> {dest}")
    for split, src in jobs:
        sz = os.path.getsize(src) if os.path.exists(src) else 0
        print(f"  {split:<5}  {src}  ({sz/1e6:.1f} MB)")

    if not args.dry_run:
        confirm(args.yes)
        os.makedirs(out_dir, exist_ok=True)
    man = {"step": "sft_news", "created": utcnow(), "out": out_dir,
           "json": os.path.basename(dest), "files": []}
    all_kept = []
    for split, src in jobs:
        kept, skipped, n_in = from_csv(src)
        all_kept.extend(kept)
        rec = {"split": split, "src": src, "json": os.path.basename(dest),
               "n_in": n_in, "n_out": len(kept), "n_skipped": skipped,
               "targets": dict(Counter(p["answer"] for p in kept))}
        man["files"].append(rec)
        print(f"  {split}: {n_in:,} in -> {len(kept):,} pairs "
              f"(skipped {skipped:,} empty text/target)")
        if rec["targets"]:
            print("         " + ", ".join(
                f"{k} {v:,}" for k, v in
                sorted(rec["targets"].items(), key=lambda kv: -kv[1])))
    print(f"  total {len(all_kept):,} pairs")
    if args.dry_run:
        print("\ndry-run: not writing")
        return man
    write_pairs(dest, all_kept)
    drop_old_splits(out_dir, "fidonews_train.json", "fidonews_test.json")
    write_json(os.path.join(out_dir, "manifest_news.json"), man)
    print(f"manifest -> {out_dir}/manifest_news.json")
    return man


def parse_args(argv=None, prog=None):
    p = argparse.ArgumentParser(
        prog=prog, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train",
                   default=os.path.expanduser("~/data/fido-ua-news/train.csv"))
    p.add_argument("--test",
                   default=os.path.expanduser("~/data/fido-ua-news/test.csv"))
    p.add_argument("--out-dir",
                   default=os.path.join(ROOT, "sft_data", "text"))
    p.add_argument("--yes", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--no-test", action="store_true")
    args = p.parse_args(argv)
    if args.no_train:
        args.train = None
    if args.no_test:
        args.test = None
    return args


def _self_test():
    import tempfile
    csv_body = (
        "title,text,tags,target\n"
        '"київський заголовок","перший абзац новини, з комою.",x,новини\n'
        "ok title,,x,новини\n"
        "title only,some text,x,\n"
        '"quoted, title","body\nwith newline",x,спорт\n'
    )
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "train.csv")
        with open(src, "w", encoding="utf-8", newline="") as f:
            f.write(csv_body)
        kept, skipped, n_in = from_csv(src)
        assert n_in == 4 and skipped == 2 and len(kept) == 2
        assert kept[0]["answer"] == "Новини"
        assert kept[0]["question"].startswith(PROMPT + " ")
        assert kept[0]["question"].endswith("Перший абзац новини, з комою.")
        assert kept[1]["answer"] == "Спорт"
        assert "with newline" in kept[1]["question"]
        ns = argparse.Namespace(train=src, test=None, out_dir=td,
                                yes=True, dry_run=False)
        man = run(ns)
        assert man["files"][0]["n_out"] == 2
        assert os.path.exists(os.path.join(td, "fidonews.json"))
    print("sft.from_news self-test ok")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        run(parse_args())
