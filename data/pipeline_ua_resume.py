"""Extend a corpus without re-cleaning what you already have.

sample_sources is deterministic under a seed: a larger budget yields the
smaller draw as an exact prefix, then more. So extending 7.99B -> 8.8B does
not require re-cleaning the first 7.99B.

`resume` gives the PREVIOUS per-source budget. Documents that the previous
run emitted are counted but not yielded, so only the tail reaches the pool.
Append the result to the existing JSONL.

Boundary rule: the old run yielded documents 1..k where cumulative(k) >=
T_old > cumulative(k-1). So a document belongs to the old run iff its
cumulative total BEFORE it was added is < T_old. That is the exact
complement -- no gap, no overlap.
"""

import glob
import random

import pyarrow.parquet as pq


def sample_sources(sources, seed=0, column="text", resume=None, verbose=True):
    """sources: [(glob_pattern, target_chars), ...]
    resume:    {glob_pattern: previous_target_chars} -- skip what that run
               already emitted, yield only the tail.
    """
    resume = resume or {}
    rng = random.Random(seed)
    for pattern, target in sources:
        skip_to = resume.get(pattern, 0)
        shards = sorted(glob.glob(pattern))
        if not shards:
            raise FileNotFoundError(f"no files match {pattern!r}")

        handles = {p: pq.ParquetFile(p) for p in shards}
        units = [(p, i) for p in shards
                 for i in range(handles[p].metadata.num_row_groups)]
        rng.shuffle(units)                    # same seed -> same order

        total = ndocs = nskip = 0
        done = False
        for path, rg in units:
            if done:
                break
            for b in handles[path].iter_batches(batch_size=2000,
                                                columns=[column],
                                                row_groups=[rg]):
                for d in b.column(column).to_pylist():
                    if not d:
                        continue
                    before = total
                    total += len(d)
                    if before < skip_to:      # the previous run emitted this
                        nskip += 1
                    else:
                        ndocs += 1
                        yield d
                    if total >= target:
                        done = True
                        break
                if done:
                    break

        if verbose:
            note = "" if total >= target else "  <-- EXHAUSTED, under budget"
            msg = (f"{pattern}: {ndocs:,} docs, "
                   f"{(total-skip_to)/1e6:.1f}M new chars "
                   f"({total/target*100:.0f}% of budget){note}")
            if skip_to:
                msg += f"   [skipped {nskip:,} docs already in the corpus]"
            print(msg)

if __name__ == "__main__":
    from data.index import wiki_pattern
    S = wiki_pattern()
    OLD, NEW = 20_000_000, 50_000_000

    old  = list(sample_sources([(S, OLD)], seed=0, verbose=False))
    full = list(sample_sources([(S, NEW)], seed=0, verbose=False))
    tail = list(sample_sources([(S, NEW)], seed=0, resume={S: OLD}, verbose=False))

    print(f"old {len(old):,}  tail {len(tail):,}  full {len(full):,}")
    assert old + tail == full, "resume is not the exact complement"
    assert not (set(old) & set(tail)), "overlap"
    assert list(sample_sources([(S, OLD)], seed=0, resume={S: OLD},
                               verbose=False)) == []
    print("resume verified on real data")