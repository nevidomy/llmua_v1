"""BRUK good/ text -> val-only uint16 bins (BOS/EOS via tokenizer template).

    # one val doc per non-empty line
    python -m data.from_bruk --mode lines \\
      --src $LLMUA_SOURCES/bruk/data/good \\
      --out $LLMUA_ROOT/bins/bruk_uk4b --yes

    # one val doc per file
    python -m data.from_bruk --mode files \\
      --src $LLMUA_SOURCES/bruk/data/good \\
      --out $LLMUA_ROOT/bins/bruk --yes

    # rebuild val from strings.jsonl (kept sentences only)
    python -m data.from_bruk --mode sentences \\
      --strings $LLMUA_ROOT/bins/bruk_uk4b_sentenses/strings.jsonl \\
      --out $LLMUA_ROOT/bins/bruk_uk4b_sentenses --yes
"""
import argparse
import glob
import hashlib
import json
import os
import shutil

import numpy as np

from data.cleanup_ua_fast import clean_ua_str
from data.index import ROOT, bruk_good_dir
from data.shard_util import confirm, utcnow, write_json

BRUK_SRC = bruk_good_dir()


def list_src(src):
    src = os.path.abspath(os.path.expanduser(src))
    files = sorted(p for p in glob.glob(os.path.join(src, "*"))
                   if os.path.isfile(p))
    if not files:
        raise FileNotFoundError(f"no files in {src}")
    return files


def docs_lines(paths):
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                out = clean_ua_str(s)
                if out:
                    yield out


def docs_files(paths):
    for path in paths:
        raw = open(path, encoding="utf-8", errors="replace").read()
        out = clean_ua_str(raw)
        if out:
            yield out


def docs_sentences(strings_path, keep="not_exact"):
    """Yield cleaned sentence texts from strings.jsonl.

    keep:
      all          every unique sentence
      not_exact    drop exact-containment hits (uk4b rule)
      not_exact_mh drop exact hits and MinHash-only near-copies
    """
    with open(strings_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if isinstance(obj, str):
                if obj:
                    yield obj
                continue
            if keep == "not_exact" and obj.get("exact_hit"):
                continue
            if keep == "not_exact_mh" and (
                    obj.get("exact_hit") or obj.get("minhash_hit")):
                continue
            text = obj.get("text")
            if text:
                yield text


def write_val(docs, out_dir, tok, tok_path, src, mode, n_files, batch=512):
    os.makedirs(out_dir, exist_ok=True)
    V = tok.get_vocab_size()
    assert V < 2 ** 16, f"vocab {V} does not fit uint16"
    bin_path = os.path.join(out_dir, "val.bin")
    ntok = nchar = ndoc = 0
    tmp = bin_path + ".tmp"
    buf = []
    with open(tmp, "wb") as f:
        def flush():
            nonlocal ntok, nchar, ndoc
            if not buf:
                return
            for enc, d in zip(tok.encode_batch(buf), buf):
                ids = enc.ids
                np.asarray(ids, dtype=np.uint16).tofile(f)
                ntok += len(ids)
                nchar += len(d)
                ndoc += 1
            buf.clear()

        for d in docs:
            buf.append(d)
            if len(buf) >= batch:
                flush()
        flush()
    os.replace(tmp, bin_path)
    train_path = os.path.join(out_dir, "train.bin")
    shutil.copyfile(bin_path, train_path)
    meta = {
        "step": "bruk_val",
        "created": utcnow(),
        "mode": mode,
        "src": src,
        "n_src_files": n_files,
        "vocab_size": V,
        "dtype": "uint16",
        "bos_id": tok.token_to_id("<|bos|>"),
        "eos_id": tok.token_to_id("<|eos|>"),
        "tokenizer": os.path.basename(tok_path),
        "tokenizer_sha": hashlib.blake2b(
            open(tok_path, "rb").read(), digest_size=8).hexdigest(),
        "n_docs_val": ndoc,
        "n_docs_train": ndoc,
        "n_tokens_val": ntok,
        "n_tokens_train": ntok,
        "n_chars_val": nchar,
        "n_chars_train": nchar,
        "chars_per_token_val": (nchar / ntok) if ntok else None,
        "file_val": "val.bin",
        "file_train": "train.bin",
        "template": "<|bos|> $A <|eos|>",
    }
    write_json(os.path.join(out_dir, "meta.json"), meta)
    print(f"\n{ndoc:,} docs  {ntok:,} tokens  {nchar:,} chars  "
          f"cpt {meta['chars_per_token_val']:.4f}")
    print(f"  val.bin -> {bin_path}")
    print(f"  meta    -> {out_dir}/meta.json")
    return meta


def run(args):
    out_dir = os.path.abspath(os.path.expanduser(args.out))
    tok_path = os.path.abspath(os.path.expanduser(args.tokenizer))
    print(f"bruk -> val.bin  mode={args.mode}")
    print(f"  out   {out_dir}")
    print(f"  tok   {tok_path}")
    if args.mode == "sentences":
        strings = os.path.abspath(os.path.expanduser(
            args.strings or os.path.join(out_dir, "strings.jsonl")))
        src = strings
        n_files = 0
        print(f"  strings {strings}  keep={args.keep}")
        docs = docs_sentences(strings, keep=args.keep)
    else:
        src = os.path.abspath(os.path.expanduser(args.src))
        files = list_src(src)
        n_files = len(files)
        print(f"  src   {src}  ({n_files} files)")
        docs = docs_lines(files) if args.mode == "lines" else docs_files(files)
    if not args.dry_run:
        confirm(args.yes)
    else:
        print("\ndry-run: not writing")
        return {"n_src_files": n_files, "dry_run": True}

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    meta = write_val(docs, out_dir, tok, tok_path, src, args.mode, n_files)
    from tokens.tokenizer_utils import tk_verify
    tk_verify(out_dir, tok, split="val")
    return meta


def parse_args(argv=None, prog=None):
    p = argparse.ArgumentParser(
        prog=prog, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("lines", "files", "sentences"),
                   required=True)
    p.add_argument("--src", default=BRUK_SRC)
    p.add_argument("--strings", default=None,
                   help="strings.jsonl for --mode sentences")
    p.add_argument("--keep", default="not_exact",
                   choices=("all", "not_exact", "not_exact_mh"))
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer",
                   default=os.path.join(str(ROOT), "tokenizers",
                                        "ua_bpe_32768.json"))
    p.add_argument("--yes", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
