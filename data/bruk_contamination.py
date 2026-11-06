"""BrUK line-level internal dedup + training-corpus contamination.

Units are BrUK native lines (strip + drop empty + clean_ua_str + whitespace
collapse), not nlp_uk sentence splits.

    python -m data.bruk_contamination
"""
import glob
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict

import ahocorasick
import numpy as np

from data.cleanup_ua_fast import clean_ua_str, _LOOSE
from data.index import ROOT, bruk_good_dir

BRUK = bruk_good_dir()
CORPUS = f"{ROOT}/clean/corpus.jsonl"
OUT = f"{ROOT}/bins/bruk_uk4b/contamination_report.json"

SHORT_WORDS = 5
MINHASH_MIN_WORDS = 8
NGRAM = 3
PREFILTER_NGRAM = 5
NPERM = 32
BANDS, ROWS = 8, 4
JACCARD_THRESH = 0.80


def collapse(s):
    return " ".join(s.split())


def loose_key(s):
    return collapse(_LOOSE.sub("", s.casefold()))


def sha1(s):
    return hashlib.sha1(s.encode()).hexdigest()


def load_bruk_lines():
    rows = []
    n_blank = 0
    for path in sorted(glob.glob(os.path.join(BRUK, "*"))):
        name = os.path.basename(path)
        for i, line in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
            raw = line.strip()
            if not raw:
                n_blank += 1
                continue
            text = collapse(clean_ua_str(raw))
            if not text:
                n_blank += 1
                continue
            rows.append({
                "file": name, "line_no": i, "text": text,
                "n_words": len(text.split()), "n_chars": len(text),
                "sha1": sha1(text), "loose": loose_key(text),
            })
    return rows, n_blank


def build_ac(texts):
    A = ahocorasick.Automaton()
    for i, s in enumerate(texts):
        if s:
            A.add_word(s, i)
    A.make_automaton()
    return A


def minhash_words(words, ngram=NGRAM, nperm=NPERM):
    if len(words) < ngram:
        return None
    mins = np.full(nperm, np.iinfo(np.uint64).max, dtype=np.uint64)
    gold = 0x9E3779B97F4A7C15
    for i in range(len(words) - ngram + 1):
        g = " ".join(words[i:i + ngram]).encode()
        h0 = int.from_bytes(hashlib.blake2b(g, digest_size=8).digest(), "little")
        x = h0
        for p in range(nperm):
            if x < mins[p]:
                mins[p] = x
            x = (x + gold + h0) & 0xFFFFFFFFFFFFFFFF
    return mins


def band_keys(sig):
    keys = []
    for b in range(BANDS):
        sl = sig[b * ROWS:(b + 1) * ROWS].tobytes()
        keys.append(hashlib.blake2b(sl, digest_size=8).digest())
    return keys


def jaccard_sigs(a, b):
    return float((a == b).mean())


def split_windows(doc, width=34, step=17):
    words = doc.split()
    if len(words) < MINHASH_MIN_WORDS:
        return
    if len(words) <= width + 8:
        yield words
        return
    for i in range(0, max(1, len(words) - width + 1), step):
        yield words[i:i + width]


def run():
    t0 = time.time()
    rows, n_blank = load_bruk_lines()
    n_raw = len(rows)
    by_sha = {}
    internal_dup = 0
    unique = []
    for r in rows:
        if r["sha1"] in by_sha:
            internal_dup += 1
            continue
        by_sha[r["sha1"]] = len(unique)
        r["idx"] = len(unique)
        unique.append(r)

    words = np.array([r["n_words"] for r in unique])
    word_hist = {
        "1-5": int(((words >= 1) & (words <= 5)).sum()),
        "6-8": int(((words >= 6) & (words <= 8)).sum()),
        "9-20": int(((words >= 9) & (words <= 20)).sum()),
        "21-50": int(((words >= 21) & (words <= 50)).sum()),
        "51+": int((words >= 51).sum()),
    }

    print(f"BrUK lines={n_raw} unique={len(unique)} internal_dups={internal_dup} "
          f"short<=5={word_hist['1-5']}", flush=True)
    texts = [r["text"] for r in unique]
    loose_of = [r["loose"] for r in unique]
    A = build_ac(texts)
    # unique loose keys -> list of unique-line idxs (skip tiny after loose)
    loose_map = defaultdict(list)
    for i, k in enumerate(loose_of):
        if len(k) >= 8:
            loose_map[k].append(i)
    loose_keys = list(loose_map.keys())
    A_lo = build_ac(loose_keys)

    # MinHash index of longer unique lines
    mh_idx = []
    bands = defaultdict(list)
    trigrams = {}
    for r in unique:
        if r["n_words"] < MINHASH_MIN_WORDS:
            continue
        w = r["text"].split()
        sig = minhash_words(w)
        if sig is None:
            continue
        li = len(mh_idx)
        mh_idx.append((r["idx"], sig, w))
        for bk in band_keys(sig):
            bands[bk].append(li)
        for i in range(len(w) - PREFILTER_NGRAM + 1):
            trigrams.setdefault(" ".join(w[i:i + PREFILTER_NGRAM]), set()).add(li)

    hit_exact = Counter()   # unique idx -> corpus docs
    hit_loose = Counter()
    hit_mh = {}             # unique idx -> max jaccard
    n_docs = n_chars = 0
    t1 = time.time()

    with open(CORPUS, encoding="utf-8") as f:
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
            # MinHash: only windows that share a 5-gram with a BrUK line
            words_d = collapsed.split()
            cands = set()
            if len(words_d) >= PREFILTER_NGRAM:
                for i in range(len(words_d) - PREFILTER_NGRAM + 1):
                    g = " ".join(words_d[i:i + PREFILTER_NGRAM])
                    ids = trigrams.get(g)
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
                        uid, osig, _ = mh_idx[li]
                        jac = jaccard_sigs(sig, osig)
                        if jac >= JACCARD_THRESH:
                            prev = hit_mh.get(uid, 0.0)
                            if jac > prev:
                                hit_mh[uid] = jac
            if n_docs % 200_000 == 0:
                dt = time.time() - t1
                print(f"  {n_docs:,} docs  {n_chars/1e9:.2f}G  "
                      f"{n_docs/dt:,.0f}/s  exact {len(hit_exact)}  "
                      f"loose {len(hit_loose)}  minhash {len(hit_mh)}",
                      flush=True)

    def bucket_hits(idxs):
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

    exact_ids = set(hit_exact)
    loose_ids = set(hit_loose)
    mh_ids = set(hit_mh)
    loose_only = loose_ids - exact_ids
    remaining = [r for r in unique if r["idx"] not in exact_ids]
    remaining_no_mh = [r for r in remaining if r["idx"] not in mh_ids]

    def examples(ids, n=8, short=None):
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

    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "units": "BrUK native lines (not nlp_uk sentences): "
                 "strip, drop empty, clean_ua_str, collapse whitespace",
        "bruk_src": BRUK,
        "corpus": CORPUS,
        "n_bruk_files": 805,
        "n_blank_or_empty": n_blank,
        "n_lines_raw": n_raw,
        "n_internal_exact_dups": internal_dup,
        "n_unique_lines": len(unique),
        "word_hist_unique": word_hist,
        "n_unique_short_le5": word_hist["1-5"],
        "corpus_docs": n_docs,
        "corpus_chars": n_chars,
        "elapsed_s": round(time.time() - t0, 1),
        "exact": {
            "n_unique_hit": len(exact_ids),
            "n_corpus_doc_hits": int(sum(hit_exact.values())),
            "frac_unique": len(exact_ids) / len(unique),
            "by_length": bucket_hits(exact_ids),
            "n_short_le5": sum(1 for i in exact_ids if unique[i]["n_words"] <= 5),
            "n_long_gt5": sum(1 for i in exact_ids if unique[i]["n_words"] > 5),
            "examples_short": examples(exact_ids, short=True),
            "examples_long": examples(exact_ids, short=False),
        },
        "loose": {
            "n_unique_hit": len(loose_ids),
            "n_loose_only": len(loose_only),
            "frac_unique": len(loose_ids) / len(unique),
            "by_length": bucket_hits(loose_ids),
            "examples_loose_only": examples(sorted(loose_only), n=8),
        },
        "minhash": {
            "min_words": MINHASH_MIN_WORDS,
            "ngram": NGRAM,
            "nperm": NPERM,
            "threshold": JACCARD_THRESH,
            "n_indexed_lines": len(mh_idx),
            "n_unique_hit": len(mh_ids),
            "frac_of_indexed": (len(mh_ids) / len(mh_idx)) if mh_idx else 0,
            "frac_of_unique": len(mh_ids) / len(unique),
            "n_also_exact": len(mh_ids & exact_ids),
            "n_new": len(mh_ids - exact_ids),
            "examples": examples(sorted(mh_ids, key=lambda i: -hit_mh[i]), n=8),
        },
        "after_drop_exact": {
            "n_remain": len(remaining),
            "n_dropped": len(exact_ids),
        },
        "after_drop_exact_and_minhash": {
            "n_remain": len(remaining_no_mh),
            "n_dropped": len(unique) - len(remaining_no_mh),
            "delta_vs_exact_only": len(remaining) - len(remaining_no_mh),
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({k: report[k] for k in (
        "n_lines_raw", "n_internal_exact_dups", "n_unique_lines",
        "n_unique_short_le5", "corpus_docs", "elapsed_s")}, indent=2))
    print("exact", report["exact"]["n_unique_hit"],
          "short", report["exact"]["n_short_le5"],
          "long", report["exact"]["n_long_gt5"])
    print("loose", report["loose"]["n_unique_hit"],
          "loose_only", report["loose"]["n_loose_only"])
    print("minhash", report["minhash"]["n_unique_hit"],
          "new", report["minhash"]["n_new"])
    print("remain after exact", report["after_drop_exact"]["n_remain"])
    print("remain after exact+mh", report["after_drop_exact_and_minhash"]["n_remain"])
    print("wrote", OUT)
    return report


if __name__ == "__main__":
    run()
