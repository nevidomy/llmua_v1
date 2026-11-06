"""Shared inventory, confirmation, and shard naming for the full-corpus
pipeline.

Each parquet file becomes one JSONL shard (same stem, numbered so the
original glob order is reconstructible). Later steps read `manifest.json`
when it exists so they walk shards in that same order -- not `sorted(glob)`,
which would scramble mixed sources.
"""

import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone


def expand_patterns(patterns):
    """patterns: list of glob strings. Returns [(pattern, [abspath, ...]), ...]."""
    groups = []
    for pat in patterns:
        pat = os.path.expanduser(pat)
        files = [os.path.abspath(p) for p in sorted(glob.glob(pat))]
        if not files:
            raise FileNotFoundError(f"no files match {pat!r}")
        groups.append((pat, files))
    return groups


def print_inventory(groups, heading="source files"):
    """Pretty-print paths and sizes. Returns (n_files, total_bytes)."""
    n = total = 0
    print(heading)
    for pat, files in groups:
        print(f"\n  {pat}")
        sub = 0
        for p in files:
            sz = os.path.getsize(p)
            total += sz
            sub += sz
            n += 1
            print(f"    {sz/1e9:7.2f} GB  {os.path.basename(p)}")
        print(f"    {len(files)} files, {sub/1e9:.2f} GB")
    print(f"\n  {n} files, {total/1e9:.2f} GB total")
    return n, total


def confirm(yes, prompt="Proceed with this selection? [y/N] "):
    if yes:
        return
    if not sys.stdin.isatty():
        raise SystemExit("stdin is not a TTY -- pass --yes to run, "
                         "or --dry-run to only list")
    ans = input(prompt).strip().lower()
    if ans not in ("y", "yes"):
        raise SystemExit("aborted")


def shard_jsonl_name(index, parquet_path):
    stem = os.path.splitext(os.path.basename(parquet_path))[0]
    return f"{index:04d}_{stem}.jsonl"


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_manifest(src_dir):
    path = os.path.join(src_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_jsonl(src_dir):
    """JSONL shards in pipeline order. Prefers manifest.json; else glob."""
    man = load_manifest(src_dir)
    if man and man.get("files"):
        out = []
        for rec in man["files"]:
            name = rec.get("jsonl")
            if not name:
                continue
            p = name if os.path.isabs(name) else os.path.join(src_dir, name)
            if os.path.exists(p) and os.path.getsize(p) > 0:
                out.append(os.path.abspath(p))
        if out:
            return out
    files = sorted(glob.glob(os.path.join(src_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"no .jsonl shards in {src_dir}")
    return [os.path.abspath(p) for p in files]


def doc_hash_hex(s):
    """Matches tokenize_replay_val.doc_hash / the stored val_hashes list."""
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()


def load_exclude_hashes(path):
    """val_hashes.json (`val_hashes` list) or a plain JSON list of hex strings."""
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        hashes = obj.get("val_hashes") or obj.get("hashes")
        if hashes is None:
            raise ValueError(f"{path} has no 'val_hashes' list")
    else:
        hashes = obj
    return frozenset(hashes)


def hashes_from_corpus_split(corpus_path, val_frac=0.005, seed=0):
    """Replay tk_convert's per-document pick() on an existing cleaned JSONL."""
    import random
    from tqdm.auto import tqdm
    rng = random.Random(seed)
    out = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in tqdm(f, unit="doc", file=sys.stdout, desc="exclude-corpus"):
            doc = json.loads(line)
            if not doc:
                continue
            if rng.random() < val_frac:
                out.append(doc_hash_hex(doc))
    print(f"  {len(out):,} val hashes from {corpus_path} "
          f"(val_frac={val_frac}, seed={seed})")
    return frozenset(out)
