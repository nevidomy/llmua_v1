"""Train and compare tokenizers at 8k / 16k / 32k / 48k.

The held-out set is carved out by LINE INDEX before any sampling, so it is
disjoint from the training sample by construction -- not by hoping two random
draws miss each other. Earlier fertility numbers (2.82, 3.56) came from a
held set of about two documents that shared a seed with the training sample;
both were unreliable.

Systematic (every Nth) rather than random for the held-out split: the corpus
is written source by source, so a prefix would be one source only, but every
Nth preserves the mix in both halves.
"""

import json
import math
import os
import random
import sys

from tokenizers import Tokenizer

from data.index import ROOT

VOCABS = (8192, 16384, 32768, 49152)


def split_jsonl(src, train_dst, held_dst, held_every=200,
                train_target_chars=None, seed=0):
    """Every held_every-th line -> held-out. The rest are eligible for the
    training sample, thinned by Bernoulli to hit train_target_chars.

    Disjoint by construction: a line is either an index multiple or it is not.
    """
    from tqdm.auto import tqdm
    rng = random.Random(seed)

    total = n = 0
    with open(src, encoding="utf-8") as f:
        for line in f:
            total += len(line) - 2
            n += 1
    eligible = total * (1 - 1 / held_every)
    p = 1.0 if train_target_chars is None else min(
        1.0, train_target_chars / max(eligible, 1))

    tn = hn = tc = hc = 0
    with open(src, encoding="utf-8") as fi, \
         open(train_dst, "w", encoding="utf-8") as ft, \
         open(held_dst, "w", encoding="utf-8") as fh:
        for i, line in enumerate(tqdm(fi, total=n, unit="doc", file=sys.stdout)):
            if i % held_every == 0:
                fh.write(line)
                hn += 1
                hc += len(line) - 2
            elif rng.random() < p:
                ft.write(line)
                tn += 1
                tc += len(line) - 2

    print(f"train {tn:,} docs / {tc/1e9:.2f}B chars -> {train_dst}")
    print(f"held  {hn:,} docs / {hc/1e6:.0f}M chars -> {held_dst}")
    return tn, hn


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def count_lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def train_all(train_path, out_dir, vocabs=VOCABS, tk_train=None):
    """tk_train: your tk_train from the tokenizer cell."""
    n = count_lines(train_path)
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for v in vocabs:
        print(f"\n=== vocab {v:,} ===")
        tok = tk_train(read_jsonl(train_path), vocab_size=v, length=n)
        p = os.path.join(out_dir, f"ua_bpe_{v}.json")
        tok.save(p)
        paths[v] = p
        print(f"saved {p}")
    return paths


def compare(paths, held_path, n_docs=20_000, bytes_per_char=1.803):
    """Fertility on held-out text, plus what each vocab costs in parameters."""
    held = []
    for i, d in enumerate(read_jsonl(held_path)):
        if i >= n_docs:
            break
        held.append(d)
    nch = sum(len(d) for d in held)
    print(f"\nheld-out: {len(held):,} docs, {nch/1e6:.1f}M chars\n")

    hdr = (f"{'vocab':>7}{'chars/tok':>11}{'tok/doc':>9}{'emb@768':>10}"
           f"{'D=20N':>9}{'chars':>8}{'vs 8k':>8}")
    print(hdr)
    base = None
    rows = {}
    for v in sorted(paths):
        tok = Tokenizer.from_file(paths[v])
        encs = tok.encode_batch(held, add_special_tokens=False)
        ntk = sum(len(e.ids) for e in encs)
        cpt = nch / ntk
        base = base or cpt
        emb = v * 768
        tot = 84_934_656 + emb            # L=12 d=768 SwiGLU non-emb
        D = 20 * tot
        print(f"{v:>7,}{cpt:>11.3f}{ntk/len(held):>9.0f}{emb/1e6:>9.1f}M"
              f"{D/1e9:>8.2f}B{D*cpt/1e9:>7.1f}B{(cpt/base-1)*100:>+7.1f}%")
        rows[v] = cpt
    return rows


def check(path, held_path, n_docs=2000):
    """Invariants that must hold before spending GPU time."""
    tok = Tokenizer.from_file(path)
    V = tok.get_vocab_size()
    assert V < 2 ** 16, "uint16 memmap would wrap"

    ids = {}
    for ctx in ["{d}", " {d}", "-{d}", ".{d}", "рік {d} рік", "({d})"]:
        for d in "0123456789":
            e = tok.encode(ctx.format(d=d), add_special_tokens=False)
            pos = [i for i, t in enumerate(e.tokens) if t == d]
            assert len(pos) == 1, f"{ctx.format(d=d)!r} -> {e.tokens}"
            ids.setdefault(d, e.ids[pos[0]])
            assert ids[d] == e.ids[pos[0]], f"digit {d} unstable in {ctx!r}"

    bad = 0
    for i, doc in enumerate(read_jsonl(held_path)):
        if i >= n_docs:
            break
        if tok.decode(tok.encode(doc).ids, skip_special_tokens=True) != doc:
            bad += 1
    assert bad == 0, f"{bad} roundtrip failures"

    nl = sum(1 for t in tok.get_vocab() if "Ċ" in t)
    assert nl > 0, "no newline-bearing token -- paragraph breaks were flattened"
    e = tok.encode("Київ")
    assert e.tokens[0] == "<|bos|>" and e.tokens[-1] == "<|eos|>"
    print(f"  {V:,}: digits stable, {n_docs:,} roundtrips exact, "
          f"{nl} newline tokens, template ok")
