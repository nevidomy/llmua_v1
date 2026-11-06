"""Shared {question, answer} pair helpers for SFT.

Text files are JSON lists (or {"data": [...]}) of objects with `question`
and `answer` strings. JSONL is accepted too.
"""
import json
import os

from data.cleanup_ua import clean_ua_str
from data.shard_util import write_json


def cap_first(s):
    """Uppercase the first character if it is a lowercase letter."""
    if s and s[0].islower():
        return s[0].upper() + s[1:]
    return s


def clean_part(s):
    """clean_ua_str, then capitalize the first letter if it was lower."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return cap_first(clean_ua_str(s))


def join_question(context, question):
    """Q = "{context} {question}", dropping empty sides so no stray space."""
    parts = [p for p in (context, question) if p]
    return " ".join(parts)


def span_text(item):
    """First SQuAD-style answer: {text, answer_start} or a bare string."""
    if item is None:
        return ""
    if isinstance(item, dict):
        return item.get("text") or ""
    if isinstance(item, str):
        return item
    return str(item)


def first_span(items):
    if not items:
        return ""
    return span_text(items[0])


def load_pairs(path):
    """Yield {question, answer} dicts from a JSON list, {data: [...]}, or JSONL."""
    path = os.path.expanduser(path)
    with open(path, encoding="utf-8") as f:
        start = f.read(1)
        while start and start.isspace():
            start = f.read(1)
        f.seek(0)
        if start == "{":
            obj = json.load(f)
            rows = obj.get("data", obj) if isinstance(obj, dict) else obj
            if isinstance(rows, dict):
                rows = [rows]
            for row in rows:
                yield row
            return
        if start == "[":
            for row in json.load(f):
                yield row
            return
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_pairs(path, pairs):
    path = os.path.expanduser(path)
    write_json(path, list(pairs))
    return path


def drop_old_splits(out_dir, *names):
    """Remove leftover per-split JSON after writing a single source file."""
    for name in names:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.remove(path)
            print(f"  removed {name}")
