"""Data pipeline: Parquet -> raw docs -> cleaned JSONL -> deduped JSONL.

Order of operations, and why:

  1. sample_sources(SOURCES)        yields RAW text
  2. diagnose(...)                  on RAW -- scrub removes the artifacts it
                                    looks for, so running it on cleaned text
                                    would show nothing
  3. write_sample(...)              cleans in a Pool, writes JSONL
  4. dup_report / dedup_jsonl       on CLEANED -- normalised whitespace and
                                    apostrophes make exact hashing catch
                                    documents differing only in extraction noise
  5. read_sample(...)               feeds the tokenizer trainer

JSONL rather than one-doc-per-line text, so the \\n\\n paragraph breaks that
clean_ws preserves survive escaping. Flatten them and the BPE never learns a
newline-bearing token.
"""

import contextlib
import glob
import json
import random
import sys

import pyarrow.parquet as pq

from data.cleanup_ua_fast import clean_ua_str
from data.index import fineweb_pattern, wiki_pattern


def sample_sources(sources, seed=0, column="text", verbose=True):
    """Sample RAW documents from several Parquet sources, each with its own
    character budget.

        sources = [(glob_pattern, target_chars), ...]

    Yields raw text -- cleaning belongs in the worker pool downstream, not in
    this generator, or the whole pipeline serialises on one core.

    Row groups are drawn without replacement, so no document appears twice.
    Budgets count raw characters; cleaning shrinks that a few percent.
    """
    rng = random.Random(seed)
    for pattern, target in sources:
        shards = sorted(glob.glob(pattern))
        if not shards:
            raise FileNotFoundError(f"no files match {pattern!r}")

        handles = {p: pq.ParquetFile(p) for p in shards}
        units = [(p, i) for p in shards
                 for i in range(handles[p].metadata.num_row_groups)]
        rng.shuffle(units)

        total = ndocs = 0
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
                    total += len(d)
                    ndocs += 1
                    yield d
                    if total >= target:      # per-document, so a small budget
                        done = True          # is not blown by one row group
                        break
                if done:
                    break

        if verbose:
            pct = total / target * 100
            note = "" if total >= target else "  <-- EXHAUSTED, under budget"
            print(f"{pattern}: {ndocs:,} docs, {total/1e6:.2f}M raw chars "
                  f"({pct:.0f}% of budget){note}")


def write_sample(doc_iter, out_path, target_chars=None, clean_fn=clean_ua_str,
                 workers=None, chunksize=64):
    """Clean documents into a JSONL file. Returns the document count.

    target_chars=None means "consume the whole iterator" -- correct when
    sample_sources already owns the per-source budgets. A number here imposes
    a second, global cap on top of them.

    Output order is the input order (pool.imap), so a rebuild is
    deterministic. NOTE: the paper's corpus.jsonl was built with
    imap_unordered, so its line order depended on worker timing. A rebuild
    has the same documents in a different line order; reproduce the paper's
    val split by content with `prepare_corpus.py bins --val-hashes`.
    """
    from tqdm.auto import tqdm
    total = n = 0

    with contextlib.ExitStack() as stack:
        f = stack.enter_context(open(out_path, "w", encoding="utf-8"))
        if workers == 1:
            it = (clean_fn(d) for d in doc_iter)
        else:
            from multiprocessing import Pool
            pool = stack.enter_context(Pool(workers))   # __exit__ terminates
            it = pool.imap(clean_fn, doc_iter, chunksize=chunksize)
        bar = stack.enter_context(tqdm(unit="doc", file=sys.stdout))

        for out in it:
            if not out:
                continue
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
            total += len(out)
            bar.update(1)
            if target_chars is not None and total >= target_chars:
                break

    print(f"{n:,} docs, {total/1e6:.2f}M chars -> {out_path}")
    return n


def read_sample(path):
    """json.loads is C-level, so this feeds the Rust trainer at I/O speed."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def count_lines(path):
    """Document count for tk_train(..., length=n) without re-reading JSON."""
    with open(path, "rb") as f:
        return sum(1 for _ in f)


if __name__ == "__main__":
#    SOURCES = [
#        (fineweb_pattern(), 1_200_000_000),
#        (wiki_pattern(), 300_000_000),
#    ]
#    print(__doc__)
#    print("SOURCES configured for:")
#    for pat, budget in SOURCES:
#        print(f"  {budget/1e6:>8.0f}M chars  {pat}")

    SOURCE = wiki_pattern()
    rng = random.Random(0)
    shards = sorted(glob.glob(SOURCE))
    handles = {p: pq.ParquetFile(p) for p in shards}
    units = [(p, i) for p in shards
                for i in range(handles[p].metadata.num_row_groups)]
    rng.shuffle(units)

    docs = 0
    for path, rg in units:
        for b in handles[path].iter_batches(batch_size=2000,
                                            columns=["text"],
                                            row_groups=[rg]):
            for d in b.column("text").to_pylist():
                if not d:
                    continue
                if (len(d) > 6000*4):
                    print("================================================")
                    print(f"Document length: {len(d)} characters:")
                    print(d)
                    docs+=1
                    if docs >= 10: exit(0)
