"""UA-SQuAD (fido-ai) -> {question, answer} JSON for SFT.

train.json is a flat list under `data` of {context, question, answer}.
val.json is SQuAD 2.0: data -> articles -> paragraphs -> {context, qas}.

    python prepare_sft.py fido \\
      --train ~/data/fido-ai/train.json \\
      --val ~/data/fido-ai/val.json \\
      --out-dir ~/data/llmua/sft_data/text --yes
"""
import argparse
import json
import os

from data.index import ROOT
from data.shard_util import confirm, utcnow, write_json
from sft.pairs import clean_part, first_span, join_question, write_pairs, drop_old_splits


def _load_json(path):
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        return json.load(f)


def convert_train_row(row):
    q = join_question(clean_part(row.get("context")),
                      clean_part(row.get("question")))
    a = clean_part(row.get("answer"))
    if not q or not a:
        return None
    return {"question": q, "answer": a}


def convert_val_qa(context, qa):
    raw = first_span(qa.get("answers")) or first_span(qa.get("plausible_answers"))
    q = join_question(clean_part(context), clean_part(qa.get("question")))
    a = clean_part(raw)
    if not q or not a:
        return None
    return {"question": q, "answer": a}


def from_train(obj):
    rows = obj["data"] if isinstance(obj, dict) else obj
    kept, skipped = [], 0
    for row in rows:
        pair = convert_train_row(row)
        if pair is None:
            skipped += 1
        else:
            kept.append(pair)
    return kept, skipped, len(rows)


def from_val(obj):
    articles = obj["data"] if isinstance(obj, dict) else obj
    kept, skipped, n_in = [], 0, 0
    for art in articles:
        for para in art.get("paragraphs") or []:
            ctx = para.get("context")
            for qa in para.get("qas") or []:
                n_in += 1
                pair = convert_val_qa(ctx, qa)
                if pair is None:
                    skipped += 1
                else:
                    kept.append(pair)
    return kept, skipped, n_in


def run(args):
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    dest = os.path.join(out_dir, "fidoai.json")
    jobs = []
    if args.train:
        jobs.append(("train", os.path.expanduser(args.train), from_train))
    if args.val:
        jobs.append(("val", os.path.expanduser(args.val), from_val))
    if not jobs:
        raise SystemExit("give --train and/or --val")

    print("fido-ai -> {question, answer}")
    print(f"         -> {dest}")
    for split, src, _ in jobs:
        sz = os.path.getsize(src) if os.path.exists(src) else 0
        print(f"  {split:<5}  {src}  ({sz/1e6:.1f} MB)")

    if not args.dry_run:
        confirm(args.yes)
        os.makedirs(out_dir, exist_ok=True)
    man = {"step": "sft_fido", "created": utcnow(), "out": out_dir,
           "json": os.path.basename(dest), "files": []}
    all_kept = []
    for split, src, conv in jobs:
        kept, skipped, n_in = conv(_load_json(src))
        all_kept.extend(kept)
        rec = {"split": split, "src": src, "json": os.path.basename(dest),
               "n_in": n_in, "n_out": len(kept), "n_skipped": skipped}
        man["files"].append(rec)
        print(f"  {split}: {n_in:,} in -> {len(kept):,} pairs "
              f"(skipped {skipped:,} empty q/a)")
    print(f"  total {len(all_kept):,} pairs")
    if args.dry_run:
        print("\ndry-run: not writing")
        return man
    write_pairs(dest, all_kept)
    drop_old_splits(out_dir, "fidoai_train.json", "fidoai_val.json")
    write_json(os.path.join(out_dir, "manifest.json"), man)
    print(f"manifest -> {out_dir}/manifest.json")
    return man


def parse_args(argv=None, prog=None):
    p = argparse.ArgumentParser(
        prog=prog, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default=os.path.expanduser("~/data/fido-ai/train.json"))
    p.add_argument("--val", default=os.path.expanduser("~/data/fido-ai/val.json"))
    p.add_argument("--out-dir",
                   default=os.path.join(ROOT, "sft_data", "text"))
    p.add_argument("--yes", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-train", action="store_true",
                   help="skip the train split (default still converts both)")
    p.add_argument("--no-val", action="store_true")
    args = p.parse_args(argv)
    if args.no_train:
        args.train = None
    if args.no_val:
        args.val = None
    return args


def _self_test():
    import tempfile
    train = {"data": [
        {"context": "київ є столицею.", "question": "яке місто?",
         "answer": "київ"},
        {"context": "x", "question": "y", "answer": ""},
    ]}
    val = {"data": [{"title": "t", "paragraphs": [{
        "context": "київ є столицею України.",
        "qas": [
            {"question": "що є столицею?",
             "answers": [{"text": "київ", "answer_start": 0}]},
            {"question": "хто?", "answers": [],
             "plausible_answers": [{"text": "українці", "answer_start": 1}]},
            {"question": "нічого?", "answers": [], "plausible_answers": []},
        ]}]}]}
    kept_t, skip_t, n_t = from_train(train)
    assert n_t == 2 and skip_t == 1 and len(kept_t) == 1
    assert kept_t[0]["question"].startswith("Київ")
    assert "Яке місто?" in kept_t[0]["question"]
    assert kept_t[0]["answer"].startswith("Київ")
    kept_v, skip_v, n_v = from_val(val)
    assert n_v == 3 and skip_v == 1 and len(kept_v) == 2
    assert kept_v[0]["answer"].startswith("Київ")
    assert kept_v[1]["answer"].startswith("Українці")
    with tempfile.TemporaryDirectory() as td:
        src_t = os.path.join(td, "train.json")
        src_v = os.path.join(td, "val.json")
        write_json(src_t, train)
        write_json(src_v, val)
        ns = argparse.Namespace(train=src_t, val=src_v, out_dir=td,
                                yes=True, dry_run=False, no_train=False,
                                no_val=False)
        man = run(ns)
        assert man["files"][0]["n_out"] == 1
        assert man["files"][1]["n_out"] == 2
        out = os.path.join(td, "fidoai.json")
        assert os.path.exists(out)
        assert len(json.load(open(out, encoding="utf-8"))) == 3
    print("sft.from_fido self-test ok")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        run(parse_args())
