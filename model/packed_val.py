"""Pack full BOS…EOS documents for pretrain eval.

Two CE rules, both packing complete documents into T (never split):

  content (mode packed)     body→body only; BOS, EOS, and pad ignored.
                            Reported bpc is mean nats / cpt / ln2.
  uk4b (mode packed_uk4b)   BOS→first body and body→body; drop EOS.
                            Reported bpc is Σ bits / n_chars_val.
"""
import re

import numpy as np
import torch

IGNORE_INDEX = -100

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def custom_loss_key(kind, name, mode="stream"):
    """Checkpoint key for a custom-dataset full val.

    pretrain stream       -> full_val_loss_{name}
    pretrain packed       -> full_val_loss_packed_{name}
    pretrain packed_uk4b  -> full_val_loss_packed_uk4b_{name}
    sft one-per-row       -> full_val_loss_sft_{name}
    sft packed            -> full_val_loss_sft_packed_{name}
    """
    name = str(name or "").strip()
    if not _NAME_RE.match(name):
        raise ValueError(f"bad dataset name: {name!r}")
    kind = (kind or "pretrain").strip().lower()
    mode = (mode or "stream").strip().lower()
    if mode not in ("stream", "packed", "packed_uk4b", ""):
        raise ValueError(
            f"mode must be stream, packed, or packed_uk4b, got {mode!r}")
    if kind == "sft":
        if mode == "packed_uk4b":
            raise ValueError("packed_uk4b is pretrain-only")
        if mode == "packed":
            return f"full_val_loss_sft_packed_{name}"
        return f"full_val_loss_sft_{name}"
    if mode == "packed_uk4b":
        return f"full_val_loss_packed_uk4b_{name}"
    if mode == "packed":
        return f"full_val_loss_packed_{name}"
    return f"full_val_loss_{name}"


def is_custom_loss_key(key):
    return bool(re.fullmatch(
        r"full_val_loss_(?:sft_packed_|packed_uk4b_|sft_|packed_)?"
        r"[A-Za-z0-9][A-Za-z0-9._-]*",
        str(key or "")))


def bos_eos_spans(data, bos_id, eos_id):
    """[start, end) ranges for BOS…EOS documents in a token stream.

    Each BOS is paired with the first EOS before the next BOS. A BOS with
    no EOS in that window is skipped.
    """
    data = np.asarray(data)
    bos = np.flatnonzero(data == int(bos_id))
    n = int(data.shape[0])
    spans = []
    for i, b in enumerate(bos):
        b = int(b)
        hi = int(bos[i + 1]) if i + 1 < len(bos) else n
        rel = np.flatnonzero(data[b:hi] == int(eos_id))
        if rel.size == 0:
            continue
        spans.append((b, b + int(rel[0]) + 1))
    return spans


def collect_docs(data, bos_id, eos_id, max_len):
    """Complete documents that fit in `max_len`. Oversized / no-EOS skipped."""
    docs, n_over, n_no_eos = [], 0, 0
    seen_bos = int((np.asarray(data) == int(bos_id)).sum())
    spans = bos_eos_spans(data, bos_id, eos_id)
    n_no_eos = max(0, seen_bos - len(spans))
    for lo, hi in spans:
        if hi - lo > int(max_len):
            n_over += 1
            continue
        docs.append(np.asarray(data[lo:hi], dtype=np.int64))
    return docs, n_over, n_no_eos


def pack_doc_rows(lengths, T):
    """Group document indices into dense rows. Never splits a document."""
    rows, buf, used = [], [], 0
    T = int(T)
    for i, L in enumerate(lengths):
        L = int(L)
        if L > T:
            raise ValueError(f"document {i} length {L} > context {T}")
        if buf and used + L > T:
            rows.append(buf)
            buf, used = [i], L
        else:
            buf.append(i)
            used += L
    if buf:
        rows.append(buf)
    return rows


def pack_pretrain_xy(ids, T, bos_id, eos_id, pad_id=None,
                     ignore_index=IGNORE_INDEX, rule="content"):
    """Pad one packed row to T.

    rule=content: loss only on content→content (BOS, EOS, pad ignored).
    rule=uk4b: source≠EOS and target∉{BOS,EOS}, so BOS→first body counts.
      `ids` has no pad; x[L:] is already IGNORE. Do not skip pad as
      source when pad_id==bos_id.
    """
    ids = np.asarray(ids, dtype=np.int64)
    L = int(ids.size)
    T = int(T)
    if L > T:
        raise ValueError(f"packed length {L} does not fit context {T}")
    bos_id = int(bos_id)
    eos_id = int(eos_id)
    rule = (rule or "content").strip().lower()
    if rule not in ("content", "uk4b"):
        raise ValueError(f"rule must be content or uk4b, got {rule!r}")
    if pad_id is None:
        pad_id = bos_id
    else:
        pad_id = int(pad_id)
    x = np.full(T, pad_id, dtype=np.int64)
    y = np.full(T, int(ignore_index), dtype=np.int64)
    x[:L] = ids
    if L > 1:
        tgt = ids[1:]
        src = ids[:-1]
        if rule == "uk4b":
            take = (src != eos_id) & (tgt != bos_id) & (tgt != eos_id)
            if pad_id != bos_id:
                take &= (src != pad_id) & (tgt != pad_id)
        else:
            take = np.ones(L - 1, dtype=bool)
            for tok in {bos_id, eos_id, pad_id}:
                take &= src != tok
                take &= tgt != tok
        y[:L - 1][take] = tgt[take]
    return x, y


def packed_doc_attention_mask(token_ids, bos_id):
    """Causal + same-BOS-segment bool mask (SDPA: True = allowed)."""
    B, T = token_ids.shape
    seg = (token_ids == int(bos_id)).cumsum(-1)
    q = torch.arange(T, device=token_ids.device)
    causal = q[None, None, :, None] >= q[None, None, None, :]
    same = seg[:, None, :, None] == seg[:, None, None, :]
    return causal & same
