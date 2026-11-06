"""Replay the original val split without touching train.bin / val.bin.

Streams the same jsonl tk_convert saw (clean/corpus.jsonl, seed=0,
val_frac=0.005, empty docs skipped), records blake2 hashes of every val
document, tokenizes them in that order to val_recreated.bin, then
compares the new file to the original val.bin.

Does not import or call tk_convert -- the pick() loop is copied so the
existing convert path stays untouched.
"""
import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np
from tokenizers import Tokenizer
from tqdm.auto import tqdm

from data.index import ROOT
from tokens.training import read_jsonl

V = 32768
SEED = 0
VAL_FRAC = 0.005
BATCH = 2000
DIGEST_SIZE = 8

CORPUS = f"{ROOT}/clean/corpus.jsonl"
TOK_PATH = f"{ROOT}/tokenizers/ua_bpe_{V}.json"
BIN_DIR = f"{ROOT}/bins/{V}"
VAL_BIN = os.path.join(BIN_DIR, "val.bin")
OUT_BIN = os.path.join(BIN_DIR, "val_recreated.bin")
OUT_HASHES = os.path.join(BIN_DIR, "val_hashes.json")


def doc_hash(s):
    return hashlib.blake2b(s.encode("utf-8"), digest_size=DIGEST_SIZE).hexdigest()


def file_digest(path):
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick(rng):
    # same as tk_convert with test_frac=0: fracs = {"val": val_frac}
    r, acc = rng.random(), 0.0
    acc += VAL_FRAC
    if r < acc:
        return "val"
    return "train"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    meta = json.load(open(os.path.join(BIN_DIR, "meta.json")))
    tok = Tokenizer.from_file(TOK_PATH)
    rng = random.Random(SEED)

    hashes, lines, docs = [], [], []
    n_kept = 0
    nchar = 0
    print(f"scanning {CORPUS}")
    for line_i, doc in enumerate(tqdm(read_jsonl(CORPUS), unit="doc",
                                      file=sys.stdout)):
        if not doc:
            continue
        n_kept += 1
        if pick(rng) != "val":
            continue
        hashes.append(doc_hash(doc))
        lines.append(line_i)
        docs.append(doc)
        nchar += len(doc)

    print(f"kept {n_kept:,} docs, val {len(docs):,} "
          f"(meta n_docs_val {meta['n_docs_val']:,})")
    if len(docs) != meta["n_docs_val"]:
        print("WARNING: val doc count does not match meta -- wrong jsonl "
              "or seed/frac")

    rec = {
        "source": CORPUS,
        "tokenizer": os.path.basename(TOK_PATH),
        "seed": SEED,
        "val_frac": VAL_FRAC,
        "digest": f"blake2b-{DIGEST_SIZE}",
        "n_docs_kept": n_kept,
        "n_docs_val": len(docs),
        "n_chars_val": nchar,
        "val_lines": lines,
        "val_hashes": hashes,
    }
    with open(OUT_HASHES, "w") as f:
        json.dump(rec, f)
    print(f"hashes -> {OUT_HASHES}")

    print(f"encoding {len(docs):,} val docs -> {OUT_BIN}")
    ntok = 0
    with open(OUT_BIN, "wb") as f:
        for i in tqdm(range(0, len(docs), BATCH), unit="batch", file=sys.stdout):
            chunk = docs[i:i + BATCH]
            parts = [enc.ids for enc in tok.encode_batch(chunk)]
            arr = np.fromiter((t for c in parts for t in c),
                              dtype=np.uint16,
                              count=sum(len(c) for c in parts))
            arr.tofile(f)
            ntok += arr.size

    print(f"recreated {ntok:,} tokens "
          f"(meta n_tokens_val {meta['n_tokens_val']:,})")

    a, b = file_digest(VAL_BIN), file_digest(OUT_BIN)
    sa, sb = os.path.getsize(VAL_BIN), os.path.getsize(OUT_BIN)
    print(f"val.bin           {sa:>12,} B  blake2b-16 {a}")
    print(f"val_recreated.bin {sb:>12,} B  blake2b-16 {b}")
    if a == b and sa == sb:
        print("MATCH")
    else:
        print("MISMATCH")
        sys.exit(1)


if __name__ == "__main__":
    main()
