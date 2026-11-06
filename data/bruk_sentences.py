"""BrUK sentence units: tokenize_uk split, uk4b exact-containment drop, bins.

Units are nlp_uk / tokenize_uk sentences from each native line of good/,
then strip + clean_ua_str + whitespace collapse. Internal SHA-1 dedup.
Exact substring hits against corpus.jsonl are dropped from val.bin
(the uk4b rule). MinHash-only near-copies are flagged, not dropped.

    python -m data.bruk_sentences --yes
"""
import argparse
import glob
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
from tokenize_uk import tokenize_sents

from data.bruk_contamination import (
    BANDS, CORPUS, JACCARD_THRESH, MINHASH_MIN_WORDS, NGRAM, NPERM,
    PREFILTER_NGRAM, SHORT_WORDS, band_keys, build_ac, collapse,
    jaccard_sigs, loose_key, minhash_words, sha1, split_windows,
)
from data.cleanup_ua_fast import clean_ua_str
from data.from_bruk import BRUK_SRC, write_val
from data.index import ROOT
from data.shard_util import confirm, write_json

OUT_DEFAULT = f"{ROOT}/bins/bruk_uk4b_sentenses"


def load_bruk_sentences(src):
    rows = []
    n_blank = 0
    for path in sorted(glob.glob(os.path.join(src, "*"))):
        if not os.path.isfile(path):
            continue
        name = os.path.basename(path)
        for line_no, line in enumerate(
                open(path, encoding="utf-8", errors="replace"), 1):
            raw = line.strip()
            if not raw:
                n_blank += 1
                continue
            for sent_i, sent in enumerate(tokenize_sents(raw)):
                text = collapse(clean_ua_str(sent.strip()))
                if not text:
                    n_blank += 1
                    continue
                rows.append({
                    "file": name, "line_no": line_no, "sent_i": sent_i,
                    "text": text, "n_words": len(text.split()),
                    "n_chars": len(text), "sha1": sha1(text),
                    "loose": loose_key(text),
                })
    return rows, n_blank


def dedup(rows):
    by_sha, unique, n_dup = {}, [], 0
    for r in rows:
        if r["sha1"] in by_sha:
            n_dup += 1
            continue
        r["idx"] = len(unique)
        by_sha[r["sha1"]] = r["idx"]
        unique.append(r)
    return unique, n_dup


def word_hist(unique):
    words = np.array([r["n_words"] for r in unique])
    return {
        "1-5": int(((words >= 1) & (words <= 5)).sum()),
        "6-8": int(((words >= 6) & (words <= 8)).sum()),
        "9-20": int(((words >= 9) & (words <= 20)).sum()),
        "21-50": int(((words >= 21) & (words <= 50)).sum()),
        "51+": int((words >= 51).sum()),
    }


def contaminate(unique, corpus=CORPUS):
    texts = [r["text"] for r in unique]
    A = build_ac(texts)
    loose_map = defaultdict(list)
    for i, r in enumerate(unique):
        if len(r["loose"]) >= 8:
            loose_map[r["loose"]].append(i)
    loose_keys = list(loose_map.keys())
    A_lo = build_ac(loose_keys)

    mh_idx, bands, grams = [], defaultdict(list), {}
    for r in unique:
        if r["n_words"] < MINHASH_MIN_WORDS:
            continue
        w = r["text"].split()
        sig = minhash_words(w)
        if sig is None:
            continue
        li = len(mh_idx)
        mh_idx.append((r["idx"], sig))
        for bk in band_keys(sig):
            bands[bk].append(li)
        if len(w) >= PREFILTER_NGRAM:
            for i in range(len(w) - PREFILTER_NGRAM + 1):
                grams.setdefault(" ".join(w[i:i + PREFILTER_NGRAM]), set()).add(li)

    hit_exact, hit_loose, hit_mh = Counter(), Counter(), {}
    n_docs = n_chars = 0
    t1 = time.time()
    with open(corpus, encoding="utf-8") as f:
        for line in f:
            n_docs += 1
            doc = json.loads(line)
            n_chars += len(doc)
            collapsed = collapse(doc)
            if not collapsed:
                continue
            seen_e, seen_l = set(), set()
            for _, i in A.iter(collapsed):
                if i not in seen_e:
                    seen_e.add(i)
                    hit_exact[i] += 1
            loose_doc = loose_key(collapsed)
            for _, ki in A_lo.iter(loose_doc):
                for i in loose_map[loose_keys[ki]]:
                    if i not in seen_l:
                        seen_l.add(i)
                        hit_loose[i] += 1
            words_d = collapsed.split()
            cands = set()
            if len(words_d) >= PREFILTER_NGRAM:
                for i in range(len(words_d) - PREFILTER_NGRAM + 1):
                    ids = grams.get(" ".join(words_d[i:i + PREFILTER_NGRAM]))
                    if ids:
                        cands.update(ids)
            if cands:
                for win in split_windows(collapsed):
                    sig = minhash_words(win)
                    if sig is None:
                        continue
                    near = set()
                    for bk in band_keys(sig):
                        near.update(bands.get(bk, ()))
                    near &= cands
                    for li in near:
                        uid, osig = mh_idx[li]
                        jac = jaccard_sigs(sig, osig)
                        if jac >= JACCARD_THRESH:
                            prev = hit_mh.get(uid, 0.0)
                            if jac > prev:
                                hit_mh[uid] = jac
            if n_docs % 200_000 == 0:
                dt = time.time() - t1
                print(f"  {n_docs:,} docs  {n_chars/1e9:.2f}G  "
                      f"{n_docs/max(dt, 1e-6):,.0f}/s  exact {len(hit_exact)}  "
                      f"loose {len(hit_loose)}  minhash {len(hit_mh)}",
                      flush=True)
    return {
        "hit_exact": hit_exact, "hit_loose": hit_loose, "hit_mh": hit_mh,
        "n_docs": n_docs, "n_chars": n_chars, "n_indexed_mh": len(mh_idx),
    }


def bucket_hits(unique, idxs):
    b = Counter()
    for i in idxs:
        w = unique[i]["n_words"]
        if w <= 5:
            b["1-5"] += 1
        elif w <= 8:
            b["6-8"] += 1
        else:
            b["9+"] += 1
    return dict(b)


def examples(unique, hit_exact, hit_loose, hit_mh, ids, n=8, short=None):
    out = []
    for i in ids:
        r = unique[i]
        if short is True and r["n_words"] > SHORT_WORDS:
            continue
        if short is False and r["n_words"] <= SHORT_WORDS:
            continue
        out.append({
            "file": r["file"], "n_words": r["n_words"],
            "n_chars": r["n_chars"], "text": r["text"][:220],
            "corpus_hits": hit_exact.get(i, 0),
            "loose_hits": hit_loose.get(i, 0),
            "minhash_jaccard": hit_mh.get(i),
        })
        if len(out) >= n:
            break
    return out


def subset_sums(rows, pred):
    chars = bits = tokens = n = 0
    bits_ok = True
    for r in rows:
        if not pred(r):
            continue
        n += 1
        chars += r["n_chars"]
        tokens += int(r.get("n_tokens") or 0)
        if r.get("bits") is None:
            bits_ok = False
        else:
            bits += float(r["bits"])
    bpc = (bits / chars) if (bits_ok and chars) else None
    return {
        "n": n, "n_chars": chars, "n_tokens": tokens,
        "bits": bits if bits_ok else None, "bpc": bpc,
    }


def write_strings(path, unique):
    with open(path, "w", encoding="utf-8") as f:
        for r in unique:
            rec = {
                "text": r["text"],
                "n_chars": r["n_chars"],
                "n_words": r["n_words"],
                "n_tokens": r.get("n_tokens"),
                "bits": r.get("bits"),
                "exact_hit": bool(r.get("exact_hit")),
                "loose_hit": bool(r.get("loose_hit")),
                "minhash_hit": bool(r.get("minhash_hit")),
                "minhash_jaccard": r.get("minhash_jaccard"),
                "file": r["file"],
                "line_no": r["line_no"],
                "sent_i": r["sent_i"],
                "sha1": r["sha1"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def attach_hits(unique, hits):
    he, hl, hm = hits["hit_exact"], hits["hit_loose"], hits["hit_mh"]
    for r in unique:
        i = r["idx"]
        r["exact_hit"] = i in he
        r["loose_hit"] = i in hl
        r["minhash_hit"] = i in hm
        r["minhash_jaccard"] = hm.get(i)
        r["corpus_hits"] = int(he.get(i, 0))
        r["loose_hits"] = int(hl.get(i, 0))


def encode_token_counts(unique, tok):
    texts = [r["text"] for r in unique]
    for i in range(0, len(texts), 512):
        chunk = texts[i:i + 512]
        for r, enc in zip(unique[i:i + 512], tok.encode_batch(chunk)):
            r["n_tokens"] = len(enc.ids)


def build_report(unique, n_raw, n_dup, n_blank, n_files, hits, elapsed,
                 tok_path, out_dir):
    he, hl, hm = hits["hit_exact"], hits["hit_loose"], hits["hit_mh"]
    exact_ids, loose_ids, mh_ids = set(he), set(hl), set(hm)
    hist = word_hist(unique)
    sums_all = subset_sums(unique, lambda r: True)
    sums_ex = subset_sums(unique, lambda r: not r["exact_hit"])
    sums_mh = subset_sums(
        unique, lambda r: not r["exact_hit"] and not r["minhash_hit"])
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "units": "tokenize_uk / nlp_uk sentences from each BrUK native line: "
                 "strip, clean_ua_str, collapse whitespace",
        "splitter": "tokenize_uk.tokenize_sents",
        "bruk_src": BRUK_SRC,
        "corpus": CORPUS,
        "out": out_dir,
        "n_bruk_files": n_files,
        "n_blank_or_empty": n_blank,
        "n_sentences_raw": n_raw,
        "n_internal_exact_dups": n_dup,
        "n_unique_sentences": len(unique),
        "word_hist_unique": hist,
        "n_unique_short_le5": hist["1-5"],
        "corpus_docs": hits["n_docs"],
        "corpus_chars": hits["n_chars"],
        "elapsed_s": round(elapsed, 1),
        "tokenizer": os.path.basename(tok_path),
        "exact": {
            "n_unique_hit": len(exact_ids),
            "n_corpus_doc_hits": int(sum(he.values())),
            "frac_unique": len(exact_ids) / len(unique),
            "by_length": bucket_hits(unique, exact_ids),
            "n_short_le5": sum(1 for i in exact_ids
                               if unique[i]["n_words"] <= 5),
            "n_long_gt5": sum(1 for i in exact_ids
                              if unique[i]["n_words"] > 5),
            "examples_short": examples(
                unique, he, hl, hm, exact_ids, short=True),
            "examples_long": examples(
                unique, he, hl, hm, exact_ids, short=False),
        },
        "loose": {
            "n_unique_hit": len(loose_ids),
            "n_loose_only": len(loose_ids - exact_ids),
            "frac_unique": len(loose_ids) / len(unique),
            "by_length": bucket_hits(unique, loose_ids),
            "examples_loose_only": examples(
                unique, he, hl, hm, sorted(loose_ids - exact_ids)),
        },
        "minhash": {
            "min_words": MINHASH_MIN_WORDS,
            "ngram": NGRAM,
            "nperm": NPERM,
            "threshold": JACCARD_THRESH,
            "n_indexed": hits["n_indexed_mh"],
            "n_unique_hit": len(mh_ids),
            "frac_of_indexed": (
                len(mh_ids) / hits["n_indexed_mh"]
                if hits["n_indexed_mh"] else 0),
            "n_also_exact": len(mh_ids & exact_ids),
            "n_new": len(mh_ids - exact_ids),
            "examples": examples(
                unique, he, hl, hm,
                sorted(mh_ids, key=lambda i: -hm[i])),
        },
        "subsets": {
            "all_unique": sums_all,
            "minus_exact": sums_ex,
            "minus_exact_and_minhash": sums_mh,
        },
        "filter": "val.bin drops exact_hit only (uk4b rule). "
                  "MinHash-only hits are flagged, not dropped.",
        "bits": "null until `python -m data.bruk_sentence_bits --ckpt CKPT`. "
                "bpc(subset) = sum(bits) / sum(n_chars).",
    }


def run(args):
    src = os.path.abspath(os.path.expanduser(args.src))
    out_dir = os.path.abspath(os.path.expanduser(args.out))
    tok_path = os.path.abspath(os.path.expanduser(args.tokenizer))
    print(f"bruk sentences -> {out_dir}")
    print(f"  src  {src}")
    print(f"  tok  {tok_path}")
    if args.dry_run:
        print("\ndry-run: not writing")
        return {"dry_run": True}
    confirm(args.yes)
    t0 = time.time()
    rows, n_blank = load_bruk_sentences(src)
    unique, n_dup = dedup(rows)
    hist = word_hist(unique)
    n_files = sum(1 for p in glob.glob(os.path.join(src, "*"))
                  if os.path.isfile(p))
    print(f"sentences={len(rows)} unique={len(unique)} dups={n_dup} "
          f"short<=5={hist['1-5']}", flush=True)
    hits = contaminate(unique, args.corpus)
    attach_hits(unique, hits)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    encode_token_counts(unique, tok)

    os.makedirs(out_dir, exist_ok=True)
    strings_path = os.path.join(out_dir, "strings.jsonl")
    write_strings(strings_path, unique)
    chars = np.array([r["n_chars"] for r in unique], dtype=np.int32)
    tokens = np.array([r["n_tokens"] for r in unique], dtype=np.int32)
    bits = np.full(len(unique), np.nan, dtype=np.float64)
    exact = np.array([r["exact_hit"] for r in unique], dtype=np.bool_)
    mh = np.array([r["minhash_hit"] for r in unique], dtype=np.bool_)
    np.save(os.path.join(out_dir, "n_chars.npy"), chars)
    np.save(os.path.join(out_dir, "n_tokens.npy"), tokens)
    np.save(os.path.join(out_dir, "bits.npy"), bits)
    np.save(os.path.join(out_dir, "exact_hit.npy"), exact)
    np.save(os.path.join(out_dir, "minhash_hit.npy"), mh)

    kept = [r["text"] for r in unique if not r["exact_hit"]]
    meta = write_val(kept, out_dir, tok, tok_path, src, "sentences", n_files)
    meta["strings"] = "strings.jsonl"
    meta["splitter"] = "tokenize_uk.tokenize_sents"
    meta["filter"] = "exact_containment"
    meta["n_unique_sentences"] = len(unique)
    meta["n_internal_exact_dups"] = n_dup
    meta["n_exact_hits_dropped"] = int(exact.sum())
    meta["n_minhash_only"] = int((mh & ~exact).sum())
    write_json(os.path.join(out_dir, "meta.json"), meta)

    report = build_report(
        unique, len(rows), n_dup, n_blank, n_files, hits,
        time.time() - t0, tok_path, out_dir)
    report_path = os.path.join(out_dir, "contamination_report.json")
    write_json(report_path, report)

    from tokens.tokenizer_utils import tk_verify
    tk_verify(out_dir, tok, split="val")

    print(json.dumps({
        "n_sentences_raw": len(rows),
        "n_internal_exact_dups": n_dup,
        "n_unique": len(unique),
        "exact_hits": report["exact"]["n_unique_hit"],
        "exact_short_le5": report["exact"]["n_short_le5"],
        "exact_long_gt5": report["exact"]["n_long_gt5"],
        "minhash_new": report["minhash"]["n_new"],
        "val_docs": meta["n_docs_val"],
        "elapsed_s": report["elapsed_s"],
    }, indent=2))
    print("wrote", strings_path)
    print("wrote", report_path)
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=BRUK_SRC)
    p.add_argument("--out", default=OUT_DEFAULT)
    p.add_argument("--corpus", default=CORPUS)
    p.add_argument("--tokenizer",
                   default=os.path.join(str(ROOT), "tokenizers",
                                        "ua_bpe_32768.json"))
    p.add_argument("--yes", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
