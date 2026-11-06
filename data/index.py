"""Data-root layout and provenance manifest.

Default root is ``~/data/llmua`` (override with env ``LLMUA_ROOT``).
Everything under it is named, not hardcoded. Raw datasets (FineWeb-2, the
UkrLM wikipedia dump, BrUK) live in a separate sources root, env
``LLMUA_SOURCES``, default ``~/data``:

  {root}/
    tokenizers/{name}.json     HuggingFace-tokenizers JSON (e.g. ua_bpe_32768.json)
    bins/{name}/               pretrain token bins (train.bin or sharded
                               train_XXXX.bin + index.json, val.bin, meta.json)
    sft_data/bins/{name}/      SFT bins (train.bin / val.bin + offsets/lengths
                               + meta.json)
    runs/{model_name}/{run}/   checkpoints, train.log
    clean/                     optional text-pipeline working dir (not a training asset)
      shards/  dedup/  manifest.json

Bins are PER TOKENIZER: meta.json carries vocab_size and the model's embedding
table is sized from it. Mixing a 32k bin with an 8k model produces ids past the
end of the embedding -- a CUDA assert deep in the backward pass, not an
obvious error. One directory per vocab (or an explicit tokenizer field in
meta.json) keeps that impossible.

The clean/ manifest records the cleaning code's hash. When a regex changes --
and three did -- you can tell which corpus predates the fix instead of guessing.
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ENV_ROOT = "LLMUA_ROOT"
DEFAULT_ROOT = os.path.expanduser("~/data/llmua")
ENV_SOURCES = "LLMUA_SOURCES"
DEFAULT_SOURCES = os.path.expanduser("~/data")


def get_root():
    """Absolute data root: ``LLMUA_ROOT`` if set, else ``~/data/llmua``."""
    v = (os.environ.get(ENV_ROOT) or "").strip()
    if v:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(v)))
    return os.path.abspath(DEFAULT_ROOT)


def set_root(path=None):
    """Set process-wide data root (also exported so worker children inherit).

    Empty / None drops the override so ``get_root()`` falls back to
    ``~/data/llmua``.
    """
    if path and str(path).strip():
        os.environ[ENV_ROOT] = os.path.abspath(
            os.path.expanduser(os.path.expandvars(str(path).strip())))
    else:
        os.environ.pop(ENV_ROOT, None)
    return get_root()


def sources_root():
    """Raw datasets root: ``LLMUA_SOURCES`` if set, else ``~/data``."""
    v = (os.environ.get(ENV_SOURCES) or "").strip()
    if v:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(v)))
    return os.path.abspath(DEFAULT_SOURCES)


def fineweb_pattern():
    return os.path.join(sources_root(), "fineweb-2", "data", "ukr_Cyrl",
                        "train", "*.parquet")


def wiki_pattern():
    return os.path.join(sources_root(), "ruvimx", "UkrLM-wiki", "wikipedia",
                        "train-*-of-00012.parquet")


def bruk_good_dir():
    return os.path.join(sources_root(), "bruk", "data", "good")


class _DataRoot(os.PathLike):
    """``from data.index import ROOT`` stays valid after set_root().

    os.path.join / open / f-strings all go through this; they must not
    capture the path at import time.
    """

    def __fspath__(self):
        return get_root()

    def __str__(self):
        return get_root()

    def __repr__(self):
        return f"DataRoot({get_root()!r})"

    def __add__(self, other):
        return get_root() + other

    def __radd__(self, other):
        return other + get_root()


ROOT = _DataRoot()

LAYOUT = ["tokenizers", "bins", "sft_data", "runs", "clean"]


def tokenizers_dir(root=None):
    return os.path.join(root or get_root(), "tokenizers")


def bins_dir(root=None):
    return os.path.join(root or get_root(), "bins")


def sft_bins_dir(root=None):
    return os.path.join(root or get_root(), "sft_data", "bins")


def runs_dir(root=None):
    return os.path.join(root or get_root(), "runs")


def setup(root=None, vocabs=(8192, 16384, 32768, 49152)):
    root = os.path.abspath(os.path.expanduser(os.fspath(root or get_root())))
    for d in LAYOUT:
        os.makedirs(os.path.join(root, d), exist_ok=True)
    os.makedirs(sft_bins_dir(root), exist_ok=True)
    for v in vocabs:
        os.makedirs(os.path.join(root, "bins", str(v)), exist_ok=True)
    print(f"{root}")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > 2:
            continue
        ind = "  " * depth
        if dirpath != root:
            print(f"{ind}{os.path.basename(dirpath)}/")
        for f in sorted(filenames):
            sz = os.path.getsize(os.path.join(dirpath, f))
            print(f"{ind}  {f}  ({sz/1e6:.1f} MB)")
    return root


def _hash_files(paths):
    out = {}
    for p in paths:
        if os.path.exists(p):
            h = hashlib.blake2b(open(p, "rb").read(), digest_size=8).hexdigest()
            out[os.path.basename(p)] = h
    return out


def write_manifest(root, sources, out_name, n_docs, n_chars, seed=0,
                   code_files=("cleanup_ua.py", "cleanup_ua_fast.py",
                               "pipeline_ua.py"),
                   dedup=None, extra=None):
    """Call this right after write_sample / dedup_jsonl."""
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             timeout=5).stdout.strip() or None
    except Exception:
        git = None

    m = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": out_name,
        "sources": [{"pattern": p, "target_chars": c} for p, c in sources],
        "seed": seed,
        "n_docs": n_docs,
        "n_chars": n_chars,
        "chars_per_doc": round(n_chars / max(n_docs, 1), 1),
        "code_hashes": _hash_files(code_files),
        "git": git,
        "python": sys.version.split()[0],
    }
    if dedup:
        m["dedup"] = dedup
    if extra:
        m.update(extra)

    path = os.path.join(root, "clean", "manifest.json")
    hist = json.load(open(path)) if os.path.exists(path) else []
    hist.append(m)
    json.dump(hist, open(path, "w"), indent=2, ensure_ascii=False)
    print(f"manifest -> {path}  ({len(hist)} entries)")
    return m


def check_space(root, chars, cpt=3.56, n_vocabs=1, bytes_per_char=1.803):
    st = os.statvfs(root)
    free = st.f_bavail * st.f_frsize
    jsonl = chars * bytes_per_char
    bins = chars / cpt * 2 * n_vocabs
    peak = 2 * jsonl + bins
    print(f"free at {root}: {free/1e9:.1f} GB")
    print(f"  corpus.jsonl + dedup   {2*jsonl/1e9:>6.1f} GB")
    print(f"  bins x{n_vocabs}              {bins/1e9:>6.1f} GB")
    print(f"  peak during the pass   {peak/1e9:>6.1f} GB"
          f"   {'OK' if free > peak * 1.2 else 'TIGHT -- need 20% headroom'}")
    return free > peak * 1.2


if __name__ == "__main__":
    root = setup("/tmp/llmua_demo")
    print()
    check_space("/tmp", chars=2.5e9, n_vocabs=4)
    print()
    write_manifest(
        "/tmp/llmua_demo",
        sources=[("/data/fineweb-2/.../*.parquet", 2_000_000_000),
                 ("/data/UkrLM-wiki/.../*.parquet", 500_000_000)],
        out_name="corpus_dedup.jsonl",
        n_docs=812_431, n_chars=2_431_002_118, seed=0,
        code_files=("clean_ua.py",),
        dedup={"method": "exact", "min_chars": 100, "removed_pct": 3.2},
    )
    print(json.dumps(json.load(
        open("/tmp/llmua_demo/clean/manifest.json"))[-1], indent=2)[:700])