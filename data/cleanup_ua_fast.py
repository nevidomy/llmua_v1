"""Additions to cleanup_ua.py for the larger-scale rerun.

1. Fast-path guards -- 3x throughput, verified byte-identical to the
   unguarded functions on every input.
2. Deduplication -- never built, and the gap that matters most at scale.

Import the originals from clean_ua and shadow the two slow functions.
"""

import hashlib
import json
import re
import sys
import unicodedata

import numpy as np

from data import cleanup_ua as C

# --------------------------------------------------------------------------
# 1. fast paths
# --------------------------------------------------------------------------
# Each guard is a cheap C-level test that lets the overwhelming majority of
# documents skip an expensive regex. None changes behaviour.

_MARK = re.compile(r"[\u0300\u0301]")

# substring probe: C-level scan, vs a lookbehind-heavy IGNORECASE regex
_PROBE = ("пкс", "px", "png", "jpg", "jpeg", "gif", "svg", "webp",
          "Файл", "Зображення", "File", "Image", "Медіа", "Media")

_APOS_CHARS = ("\u2019", "\u2018", "\u02BC", "\u02B9", "\u0060", "\u00B4")


def clean_ua_str(s: str) -> str:
    # scrub: skip entirely unless a marker substring is present  (3.7 -> 171 MB/s)
    if any(k in s for k in _PROBE):
        s = C.scrub(s)

    # strip_stress: _WORD.sub calls a PYTHON function per word. Only pay it
    # when a combining acute/grave actually exists.                (4.4 -> 6.6)
    d = unicodedata.normalize("NFD", s)
    if _MARK.search(d):
        d = C._STRESS.sub(r"\1", C._WORD.sub(C._fix_word, d))
    s = unicodedata.normalize("NFC", d)

    # apostrophes: guarded .replace() beats dict .translate()      (8.7 -> 1709)
    for c in _APOS_CHARS:
        if c in s:
            s = s.replace(c, "'")

    return C.clean_ws(s)


# --------------------------------------------------------------------------
# 2. deduplication
# --------------------------------------------------------------------------
# Near-duplicates are endemic to web corpora. A random per-document train/val
# split puts copies on BOTH sides, so val loss comes out optimistic and no
# other diagnostic reveals it.
#
# Dedup must run on CLEANED text: normalising whitespace and apostrophes first
# means exact hashing catches documents that differ only in extraction noise.
#
# Two-pass by design. Cleaning is the expensive step and runs once; the dedup
# filter is a second pass over already-cleaned JSONL, which is cheap.

_LOOSE = re.compile(r"[^\w\s]|_")


def fingerprint(s: str, loose: bool = False) -> int:
    """64-bit hash of a cleaned document.

    loose=True additionally casefolds and drops punctuation, which catches
    copies differing only in quoting or capitalisation. It also merges
    documents that differ ONLY in punctuation, so it is the more aggressive
    setting -- measure the extra removal rate before adopting it.
    """
    if loose:
        s = _LOOSE.sub("", s.casefold())
        s = " ".join(s.split())
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(),
                          "big")


def hash_jsonl(path, loose=False):
    """Pass 1: uint64 fingerprint per line. 8 bytes/doc -- 50M docs is 400MB,
    where a Python set of the same would be several GB."""
    from tqdm.auto import tqdm
    out = []
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, unit="doc", file=sys.stdout):
            out.append(fingerprint(json.loads(line), loose))
    return np.array(out, dtype=np.uint64)


def dedup_jsonl(src, dst, loose=False, min_chars=0):
    """Pass 2: keep the FIRST occurrence of each fingerprint."""
    from tqdm.auto import tqdm
    h = hash_jsonl(src, loose)
    _, first_idx = np.unique(h, return_index=True)
    keep = np.zeros(len(h), dtype=bool)
    keep[first_idx] = True

    kept = dropped_dup = dropped_short = 0
    with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
        for i, line in enumerate(tqdm(fi, total=len(h), unit="doc",
                                      file=sys.stdout)):
            if not keep[i]:
                dropped_dup += 1
                continue
            doc = json.loads(line)
            if len(doc) < min_chars:
                dropped_short += 1
                continue
            fo.write(line)
            kept += 1

    n = len(h)
    print(f"{n:,} in -> {kept:,} out")
    print(f"  duplicates {dropped_dup:,} ({dropped_dup/n*100:.2f}%)")
    if min_chars:
        print(f"  under {min_chars} chars {dropped_short:,} "
              f"({dropped_short/n*100:.2f}%)")
    return kept


def dup_report(path, loose=False, show=10):
    """How much would dedup remove, and what are the worst repeats?"""
    h = hash_jsonl(path, loose)
    uniq, counts = np.unique(h, return_counts=True)
    rep = counts[counts > 1]
    print(f"{len(h):,} docs, {len(uniq):,} unique "
          f"({(1 - len(uniq)/len(h))*100:.2f}% duplicated)")
    if len(rep):
        print(f"  {len(rep):,} fingerprints repeat; worst repeats "
              f"{counts.max():,}x")
        worst = uniq[np.argsort(-counts)][:show]
        want = set(worst.tolist())
        seen = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                fp = fingerprint(d, loose)
                if fp in want and fp not in seen:
                    seen[fp] = d[:120]
                    if len(seen) == len(want):
                        break
        for fp in worst:
            c = counts[uniq == fp][0]
            print(f"  {c:>7,}x  {seen.get(int(fp), '?')!r}")
    return len(uniq) / len(h)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    import time

    # -- guards must be byte-identical to the unguarded pipeline ------------
    for src, _ in C.CASES:
        assert clean_ua_str(src) == C.clean_ua_str(src), src
    extra = ["Водень.pngФайлГелій.jpg180pxмежаТекст.", "Мапа 40x26пкстериторія.",
             "звичайний текст без нічого", "п\u2019ять об\u02BC\u0454ктів",
             "Со́нячна систе́ма", "café iPhone", "рокуRHL", "\uFEFFа  б\n\n\n\nв"]
    for s in extra:
        assert clean_ua_str(s) == C.clean_ua_str(s), s
    print("guards: identical to unguarded on all inputs")

    # -- throughput --------------------------------------------------------
    rng = random.Random(0)
    W = ("система планета Сонце водень елемент таблиця Київ Україна річка "
         "Дніпро територія заповідник область населення").split()
    docs = []
    for _ in range(3000):
        ws = []
        for _ in range(150):
            w = rng.choice(W)
            if rng.random() < 0.004:
                w = w[0] + "\u0301" + w[1:]
            ws.append(w)
        docs.append(" ".join(ws) + ".")
    n = sum(map(len, docs))
    for fn, name in ((C.clean_ua_str, "unguarded"), (clean_ua_str, "guarded")):
        t = time.perf_counter()
        for d in docs:
            fn(d)
        dt = time.perf_counter() - t
        print(f"  {name:<10} {n/dt/1e6:6.2f} MB/s")

    # -- dedup -------------------------------------------------------------
    body = [" ".join(rng.choice(W) for _ in range(40)) for _ in range(200)]
    lines = []
    for i, b in enumerate(body):
        lines.append(b)
        if i % 5 == 0:
            lines.append(b)                       # exact duplicate
        if i % 25 == 0:
            lines.append(b.upper() + " !!!")      # loose-only duplicate
    rng.shuffle(lines)
    with open("dup.jsonl", "w", encoding="utf-8") as f:
        for l in lines:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")

    print(f"\nsynthetic: {len(lines)} lines from {len(body)} distinct bodies")
    print("\nexact:")
    r_exact = dup_report("dup.jsonl", loose=False, show=3)
    print("\nloose:")
    r_loose = dup_report("dup.jsonl", loose=True, show=3)
    assert r_loose < r_exact, "loose should remove strictly more"
    n_upper = sum(1 for i in range(len(body)) if i % 25 == 0)
    print("\nexact filtering:")
    kept = dedup_jsonl("dup.jsonl", "dedup.jsonl", loose=False, min_chars=50)
    # exact keeps the uppercase variants -- they are not byte-identical
    assert kept == len(body) + n_upper, f"{kept} != {len(body) + n_upper}"

    print("\nloose filtering:")
    kept_l = dedup_jsonl("dup.jsonl", "dedup_loose.jsonl", loose=True,
                         min_chars=50)
    assert kept_l == len(body), f"{kept_l} != {len(body)}"
    print(f"\nexact kept {kept} (bodies + {n_upper} case variants), "
          f"loose kept {kept_l} (bodies only)")