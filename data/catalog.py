"""Scan the data root for tokenizers and tokenized datasets.

It only reads disk. A directory is an
asset when it has the files training actually opens (meta.json + bins).
"""
import json
import os

from data.index import (bins_dir, get_root, sft_bins_dir, tokenizers_dir)


def _rel(root, path):
    try:
        return os.path.relpath(path, root).replace(os.sep, "/")
    except ValueError:
        return path


def _dir_bytes(path):
    total = 0
    try:
        for name in os.listdir(path):
            p = os.path.join(path, name)
            if os.path.isfile(p):
                total += os.path.getsize(p)
    except OSError:
        return 0
    return total


def _load_json(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def inspect_tokenizer(path):
    """HuggingFace-tokenizers JSON: vocab size from model.vocab."""
    data = _load_json(path)
    if not isinstance(data, dict):
        return None
    vocab = (data.get("model") or {}).get("vocab")
    n = len(vocab) if isinstance(vocab, dict) else None
    added = data.get("added_tokens") or []
    return {
        "name": os.path.splitext(os.path.basename(path))[0],
        "file": os.path.basename(path),
        "path": os.path.abspath(path),
        "vocab_size": n,
        "n_added_tokens": len(added) if isinstance(added, list) else None,
        "bytes": os.path.getsize(path) if os.path.isfile(path) else 0,
    }


def inspect_pretrain(path):
    meta_path = os.path.join(path, "meta.json")
    meta = _load_json(meta_path)
    if not isinstance(meta, dict):
        return None
    sharded = os.path.isfile(os.path.join(path, "index.json"))
    train_bin = os.path.isfile(os.path.join(path, "train.bin"))
    if not sharded and not train_bin:
        return None
    if not os.path.isfile(os.path.join(path, "val.bin")):
        return None
    tok = meta.get("tokenizer")
    if isinstance(tok, str):
        tok = os.path.basename(tok)
    return {
        "name": os.path.basename(os.path.abspath(path)),
        "kind": "pretrain",
        "path": os.path.abspath(path),
        "vocab_size": meta.get("vocab_size"),
        "tokenizer": tok,
        "sharded": sharded,
        "n_tokens_train": meta.get("n_tokens_train"),
        "n_tokens_val": meta.get("n_tokens_val"),
        "n_docs_train": meta.get("n_docs_train"),
        "n_docs_val": meta.get("n_docs_val"),
        "chars_per_token_val": meta.get("chars_per_token_val"),
        "bytes": _dir_bytes(path),
    }


def inspect_sft(path):
    meta_path = os.path.join(path, "meta.json")
    meta = _load_json(meta_path)
    if not isinstance(meta, dict):
        return None
    train = os.path.join(path, "train.bin")
    if not os.path.isfile(train):
        return None
    if not os.path.isfile(os.path.join(path, "train.offsets.npy")):
        return None
    tok = meta.get("tokenizer")
    if isinstance(tok, str):
        tok = os.path.basename(tok)
    has_val = (os.path.isfile(os.path.join(path, "val.bin"))
               and os.path.isfile(os.path.join(path, "val.offsets.npy"))
               and os.path.isfile(os.path.join(path, "val.lengths.npy")))
    return {
        "name": os.path.basename(os.path.abspath(path)),
        "kind": "sft",
        "path": os.path.abspath(path),
        "vocab_size": meta.get("vocab_size"),
        "tokenizer": tok,
        "has_val": has_val,
        "n_examples_train": meta.get("n_examples_train"),
        "n_examples_val": meta.get("n_examples_val"),
        "n_tokens_train": meta.get("n_tokens_train"),
        "n_tokens_val": meta.get("n_tokens_val"),
        "n_answer_tokens": meta.get("n_answer_tokens"),
        "chars_per_token_val": meta.get("chars_per_token_val"),
        "bytes": _dir_bytes(path),
    }


def scan(root=None):
    """List tokenizers + datasets present on disk under `root`."""
    root = os.path.abspath(os.fspath(root or get_root()))
    tokenizers, pretrain, sft = [], [], []

    td = tokenizers_dir(root)
    if os.path.isdir(td):
        for name in sorted(os.listdir(td)):
            if not name.endswith(".json"):
                continue
            info = inspect_tokenizer(os.path.join(td, name))
            if info:
                info["relpath"] = _rel(root, info["path"])
                tokenizers.append(info)

    bd = bins_dir(root)
    if os.path.isdir(bd):
        for name in sorted(os.listdir(bd)):
            p = os.path.join(bd, name)
            if not os.path.isdir(p):
                continue
            info = inspect_pretrain(p)
            if info:
                info["relpath"] = _rel(root, info["path"])
                pretrain.append(info)

    sd = sft_bins_dir(root)
    if os.path.isdir(sd):
        for name in sorted(os.listdir(sd)):
            p = os.path.join(sd, name)
            if not os.path.isdir(p):
                continue
            info = inspect_sft(p)
            if info:
                info["relpath"] = _rel(root, info["path"])
                sft.append(info)

    return {"root": root, "tokenizers": tokenizers,
            "pretrain": pretrain, "sft": sft}
