"""Build the pretraining corpus, tokenizers and token bins (Paper 1).

Reproduces tokenization.ipynb as one script. Raw datasets are read from
$LLMUA_SOURCES (default ~/data); everything is written under $LLMUA_ROOT
(default ~/data/llmua).

Run the stages in THIS order:

    python prepare_corpus.py sample --stage 1   # 8e9 chars: wiki 20%, fineweb 80%
    python prepare_corpus.py tokenizers         # on the 8e9 corpus, BEFORE stage 2
    python prepare_corpus.py sample --stage 2   # extend to 9e9, appended
    python prepare_corpus.py sample --stage 3   # extend to 10.2e9, appended
    python prepare_corpus.py bins [--vocab 32768]
    python prepare_corpus.py report

Why the order matters: the corpus was grown in three appended stages, not drawn
once at 10.2e9. tk_convert draws one RNG value per document in file order and
the val split is decided by that draw, so the val set depends on the exact line
order of clean/corpus.jsonl. A one-shot 10.2e9 draw (same seed) yields a
different document order and therefore a different val set. The tokenizers were
likewise trained on the stage-1 corpus only.

The paper's corpus.jsonl was built with unordered workers (imap_unordered), so
its line order depended on worker timing. write_sample now uses imap, so a
rebuild is deterministic but has the same documents in a different line order,
and a plain `bins` run draws a different val set. To get the paper's val split
by content, pass `bins --val-hashes results/val_hashes.json` (add --val-only
to write only val.bin; --corpus and --out redirect input and output).

sample_sources is prefix-stable under a seed: a larger budget yields the
smaller draw as an exact prefix. Stages 2 and 3 use
data.pipeline_ua_resume.sample_sources, which skips what the previous stage
already emitted. --scale F multiplies every budget (debug only; a scaled
stage 1 is a byte prefix of the full one). --only wiki|fineweb restricts the
sources.
"""
import argparse
import os
import sys

from data.index import ROOT, fineweb_pattern, wiki_pattern

VOCABS = (8192, 16384, 32768, 49152)
STAGE_TOTALS = {1: 8e9, 2: 9e9, 3: 10.2e9}      # chars; wiki 20%, fineweb 80%
WIKI_FRAC, FW_FRAC = 0.20, 0.80


def paths():
    clean = f"{ROOT}/clean"
    return {
        "clean": clean,
        "corpus": f"{clean}/corpus.jsonl",
        "tok_train": f"{clean}/tok_train.jsonl",
        "held": f"{clean}/held.jsonl",
        "tokenizers": f"{ROOT}/tokenizers",
    }


def budgets(total, scale=1.0, only=None):
    """[(pattern, chars)] with wiki FIRST (source order fixes corpus order)."""
    wiki, fw = wiki_pattern(), fineweb_pattern()
    plan = [(wiki, int(WIKI_FRAC * total * scale)),
            (fw, int(FW_FRAC * total * scale))]
    if only == "wiki":
        plan = plan[:1]
    elif only == "fineweb":
        plan = plan[1:]
    return plan


def stage_done(p):
    """Last finished stage, kept in clean/corpus.stage (0 = none)."""
    try:
        with open(f"{p['clean']}/corpus.stage") as f:
            return int(f.read())
    except FileNotFoundError:
        return 0


def mark_stage(p, stage):
    with open(f"{p['clean']}/corpus.stage", "w") as f:
        f.write(str(stage))


def cmd_sample(a):
    p = paths()
    os.makedirs(p["clean"], exist_ok=True)
    done = stage_done(p)
    if done != a.stage - 1:
        raise SystemExit(f"stage {a.stage} must follow stage {a.stage - 1}; "
                         f"clean/corpus.stage says {done} is the last one done")
    plan = budgets(STAGE_TOTALS[a.stage], a.scale, a.only)
    if a.stage == 1:
        from data.pipeline_ua import sample_sources, write_sample
        write_sample(sample_sources(plan), p["corpus"], workers=a.workers)
        mark_stage(p, 1)
        return
    from data.pipeline_ua import write_sample
    from data.pipeline_ua_resume import sample_sources
    if not os.path.exists(p["corpus"]):
        raise SystemExit(f"{p['corpus']} missing; run stage {a.stage - 1} first")
    old = dict(budgets(STAGE_TOTALS[a.stage - 1], a.scale, a.only))
    ext = f"{p['clean']}/corpus_stage{a.stage}.jsonl"
    if os.path.exists(ext):
        raise SystemExit(f"{ext} exists: a previous stage {a.stage} was "
                         "interrupted; check corpus.jsonl before retrying")
    write_sample(sample_sources(plan, resume=old), ext, workers=a.workers)
    with open(p["corpus"], "ab") as dst, open(ext, "rb") as src:
        while True:
            chunk = src.read(1 << 24)
            if not chunk:
                break
            dst.write(chunk)
    os.remove(ext)
    mark_stage(p, a.stage)
    print(f"appended stage {a.stage} -> {p['corpus']}")


def cmd_tokenizers(a):
    from tokens.tokenizer_definition import tk_train
    from tokens.training import split_jsonl, train_all, check, compare
    p = paths()
    split_jsonl(p["corpus"], p["tok_train"], p["held"],
                held_every=200, train_target_chars=2_000_000_000)
    out = train_all(p["tok_train"], p["tokenizers"], vocabs=VOCABS,
                    tk_train=tk_train)
    for v, path in sorted(out.items()):
        check(path, p["held"])
    compare(out, p["held"])


def cmd_bins(a):
    import json
    from tokenizers import Tokenizer
    from tokens.tokenizer_utils import tk_convert, tk_verify
    from tokens.training import read_jsonl
    p = paths()
    tok_path = f"{p['tokenizers']}/ua_bpe_{a.vocab}.json"
    tok = Tokenizer.from_file(tok_path)
    out = a.out or f"{ROOT}/bins/{a.vocab}"
    corpus = a.corpus or p["corpus"]
    if a.val_only and not (a.val_hashes and a.out):
        raise SystemExit("--val-only needs --val-hashes and an explicit --out, "
                         "so it cannot replace a real bins folder")
    if a.val_hashes:
        with open(a.val_hashes, encoding="utf-8") as f:
            hashes = json.load(f)["val_hashes"]
        tk_convert(read_jsonl(corpus), out, tok, tok_path=tok_path,
                   val_hashes=hashes, val_only=a.val_only)
        tk_verify(out, tok, split="val" if a.val_only else "train")
        return
    tk_convert(read_jsonl(corpus), out, tok, tok_path=tok_path)
    tk_verify(out, tok)


def cmd_report(a):
    from data.cleanup_ua_fast import dup_report
    from data.length_report import length_report, sample_short
    corpus = paths()["corpus"]
    dup_report(corpus)
    length_report(corpus)
    sample_short(corpus, 0, 300, n=10)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="write or extend clean/corpus.jsonl")
    s.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    s.add_argument("--scale", type=float, default=1.0,
                   help="debug: multiply every budget")
    s.add_argument("--only", choices=("wiki", "fineweb"), default=None,
                   help="debug: restrict to one source")
    s.add_argument("--workers", type=int, default=None,
                   help="cleaning processes (default: all cores)")
    s.set_defaults(fn=cmd_sample)

    s = sub.add_parser("tokenizers",
                       help="split, train the four vocabularies, check, compare")
    s.set_defaults(fn=cmd_tokenizers)

    s = sub.add_parser("bins", help="tokenize the corpus into bins/{vocab}")
    s.add_argument("--vocab", type=int, default=32768, choices=VOCABS)
    s.add_argument("--corpus", default=None,
                   help="input jsonl (default: $LLMUA_ROOT/clean/corpus.jsonl)")
    s.add_argument("--out", default=None,
                   help="output dir (default: $LLMUA_ROOT/bins/VOCAB)")
    s.add_argument("--val-hashes", default=None,
                   help="results/val_hashes.json: split val by content")
    s.add_argument("--val-only", action="store_true",
                   help="with --val-hashes: write val.bin only")
    s.set_defaults(fn=cmd_bins)

    s = sub.add_parser("report", help="dup, length and short-doc reports")
    s.set_defaults(fn=cmd_report)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
