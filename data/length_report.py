"""Document-length distribution, for choosing min_chars.

Exact dedup found almost nothing (FineWeb-2 dedups upstream), but the repeats
that did show up were bot-generated Wikipedia stubs -- same template, different
place name. Those are near-duplicates, invisible to exact hashing, and short.
A length floor removes most of them without any similarity machinery.

The decision is a trade: docs are cheap to drop, characters are not. Look at
what share of CHARACTERS a threshold costs, not what share of documents.
"""

import json
import sys

import numpy as np


def length_report(path, cuts=(50, 100, 200, 400, 800, 1600, 3200), limit=None):
    from tqdm.auto import tqdm
    lens = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, unit="doc", file=sys.stdout)):
            if limit and i >= limit:
                break
            lens.append(len(json.loads(line)))
    a = np.array(lens)
    tot = a.sum()

    print(f"\n{len(a):,} docs, {tot/1e9:.2f}B chars, "
          f"mean {a.mean():.0f}, median {np.median(a):.0f}")
    print(f"percentiles: "
          + "  ".join(f"p{p}={np.percentile(a, p):.0f}"
                      for p in (1, 5, 25, 50, 75, 95, 99)))

    print(f"\n{'min_chars':>10}{'docs kept':>12}{'% docs':>9}"
          f"{'chars kept':>13}{'% chars':>9}")
    for c in cuts:
        m = a >= c
        print(f"{c:>10}{m.sum():>12,}{m.mean()*100:>8.1f}%"
              f"{a[m].sum()/1e9:>12.2f}B{a[m].sum()/tot*100:>8.2f}%")
    return a


def sample_short(path, lo, hi, n=8):
    """Read a few documents in a length band, to see what you would drop."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if lo <= len(d) < hi:
                out.append(d)
                if len(out) >= n:
                    break
    for d in out:
        print(f"  [{len(d):>4}]  {d[:150]!r}")
    return out


if __name__ == "__main__":
    import random
    rng = random.Random(0)
    W = "система планета Сонце водень елемент Київ Україна річка область".split()
    with open("/tmp/len.jsonl", "w", encoding="utf-8") as f:
        for _ in range(4000):
            # bimodal: short stubs plus real articles
            n = rng.choice([8, 12, 20]) if rng.random() < 0.3 else rng.randrange(80, 600)
            d = " ".join(rng.choice(W) for _ in range(n))
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    a = length_report("/tmp/len.jsonl", cuts=(50, 100, 200, 400))
    assert len(a) == 4000
    print("\nband 0-200:")
    sample_short("/tmp/len.jsonl", 0, 200, n=3)