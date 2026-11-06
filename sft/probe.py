"""Deep probe: per-sequence teacher-forced snapshots for SFT.

When enabled, each train micro-batch row and each val eval row is written
as one JSONL record: the context window up to the last expected END, plus
per-answer-token top-k (and the correct id if it is outside the top-k)
with probabilities.

The log is plain JSONL, read offline.
"""
import json
import os

import torch
import torch.nn.functional as F

from sft.batcher import IGNORE_INDEX

TOP_K = 5
JSONL_NAME = "deep_probe.jsonl"
META_NAME = "deep_probe.meta.json"


def probe_paths(out_dir):
    return (os.path.join(out_dir, JSONL_NAME),
            os.path.join(out_dir, META_NAME))


def load_decoder(data_dir, bos_id=0, usr_id=201, assistant_id=202, end_id=203):
    """Decode one id at a time. SFT specials win over BPE pieces at those ids."""
    tok = None
    meta = {}
    if data_dir:
        mp = os.path.join(data_dir, "meta.json")
        if os.path.isfile(mp):
            meta = json.load(open(mp, encoding="utf-8"))
        tok = _load_tokenizer(data_dir, meta)
    labels = {
        int(bos_id): "<BOS>",
        int(usr_id): "<USR>",
        int(assistant_id): "<ASSISTANT>",
        int(end_id): "<END>",
    }
    return TokenDecoder(tok, labels)


def _load_tokenizer(data_dir, meta):
    name = meta.get("tokenizer") if meta else None
    cands = []
    if name:
        if os.path.isabs(name):
            cands.append(name)
        else:
            from data.index import ROOT
            cands.append(os.path.join(ROOT, "tokenizers", name))
            cands.append(os.path.join(data_dir, name))
    for p in cands:
        if p and os.path.isfile(p):
            from tokenizers import Tokenizer
            return Tokenizer.from_file(p)
    return None


def _visible(s):
    if s is None or s == "":
        return "·"
    return (s.replace(" ", "·").replace("\n", "↵").replace("\t", "⇥")
            .replace("\r", ""))


class TokenDecoder:
    def __init__(self, tok, labels):
        self._hf = tok
        self.labels = {int(k): v for k, v in labels.items()}

    def piece(self, i):
        i = int(i)
        if i in self.labels:
            return self.labels[i]
        if self._hf is None:
            return f"[{i}]"
        raw = self._hf.id_to_token(i)
        try:
            decoded = self._hf.decode([i], skip_special_tokens=False)
        except Exception:
            decoded = raw
        if decoded is None or decoded == "":
            decoded = raw
        if decoded is None or decoded == "":
            return f"[{i}]"
        return _visible(decoded)

    def token(self, i):
        i = int(i)
        spec = None
        t = self.piece(i)
        for sid, lab in self.labels.items():
            if i == sid:
                spec = lab.strip("<>").lower()
                break
        return {"id": i, "t": t, "spec": spec}


def _rank(probs, expected):
    """1-based rank: 1 + number of vocab ids with strictly higher prob."""
    p = probs.gather(-1, expected.unsqueeze(-1)).squeeze(-1)
    return p, 1 + (probs > p.unsqueeze(-1)).sum(dim=-1)


def snapshot_batch(logits, x, y, decoder, split, step, micro=0, epoch=None,
                   k=TOP_K):
    """One record per non-pad row. logits/x/y are (B, T[, V])."""
    with torch.no_grad():
        logits = logits.detach()
        x = x.detach()
        y = y.detach()
        B, T, V = logits.shape
        k = int(min(k, V))
        recs = []
        for b in range(B):
            rec = _snapshot_row(
                logits[b], x[b], y[b], decoder, split, step, micro, b,
                epoch, k)
            if rec is not None:
                recs.append(rec)
        return recs


def _snapshot_row(logits, x, y, decoder, split, step, micro, row, epoch, k):
    valid = y != IGNORE_INDEX
    if not valid.any():
        return None
    pos = valid.nonzero(as_tuple=False).view(-1)
    last = int(pos[-1].item())
    # y[last] is ids[last+1] = last expected token (END). Show through it.
    show = min(int(x.numel()), last + 2)
    sl = logits[pos].float()
    probs = F.softmax(sl, dim=-1)
    expected = y[pos].long()
    p_exp, ranks = _rank(probs, expected)
    top_p, top_i = torch.topk(probs, k, dim=-1)
    pred_i = top_i[:, 0]
    pred_p = top_p[:, 0]
    n = int(pos.numel())
    n_hit = int((pred_i == expected).sum().item())
    ce = F.cross_entropy(sl, expected, reduction="sum").item()
    loss = ce / max(n, 1)

    x_ids = x[:show].to("cpu").tolist()
    y_cpu = y[:show].to("cpu")
    tokens = []
    in_answer = False
    for i, tid in enumerate(x_ids):
        piece = decoder.token(tid)
        spec = piece.get("spec")
        if spec == "bos":
            in_answer = False
            piece["role"] = "prompt"
        elif spec == "usr":
            in_answer = False
            piece["role"] = "prompt"
        elif spec == "assistant":
            piece["role"] = "prompt"
            in_answer = True
        elif in_answer:
            piece["role"] = "answer"
            if i > 0 and int(y_cpu[i - 1].item()) == IGNORE_INDEX:
                piece["mask"] = "lowdiff"
        else:
            piece["role"] = "prompt"
        tokens.append(piece)

    outputs = []
    pos_cpu = pos.to("cpu")
    exp_cpu = expected.to("cpu")
    p_exp_cpu = p_exp.to("cpu")
    ranks_cpu = ranks.to("cpu")
    top_i_cpu = top_i.to("cpu")
    top_p_cpu = top_p.to("cpu")
    pred_i_cpu = pred_i.to("cpu")
    pred_p_cpu = pred_p.to("cpu")
    for j in range(n):
        t = int(pos_cpu[j].item())
        exp_id = int(exp_cpu[j].item())
        rank = int(ranks_cpu[j].item())
        pred_id = int(pred_i_cpu[j].item())
        opts, seen = [], set()
        for r in range(k):
            oid = int(top_i_cpu[j, r].item())
            seen.add(oid)
            piece = decoder.token(oid)
            opts.append({
                "id": oid, "t": piece["t"], "spec": piece["spec"],
                "p": float(top_p_cpu[j, r].item()),
                "k": r + 1, "expected": oid == exp_id,
            })
        if exp_id not in seen:
            piece = decoder.token(exp_id)
            opts.append({
                "id": exp_id, "t": piece["t"], "spec": piece["spec"],
                "p": float(p_exp_cpu[j].item()),
                "k": rank, "expected": True,
            })
        opts.sort(key=lambda o: (-o["p"], o["k"]))
        prev_id = int(x[t].item())
        outputs.append({
            "pos": t,
            "prev": decoder.token(prev_id),
            "expected": decoder.token(exp_id),
            "pred": decoder.token(pred_id),
            "p_expected": float(p_exp_cpu[j].item()),
            "p_pred": float(pred_p_cpu[j].item()),
            "rank": rank,
            "hit": pred_id == exp_id,
            "opts": opts,
        })

    hit_at = {o["pos"] + 1: o["hit"] for o in outputs}
    for i, piece in enumerate(tokens):
        if i in hit_at:
            piece["hit"] = hit_at[i]

    have = {o["pos"] for o in outputs}
    for i, piece in enumerate(tokens):
        if piece.get("mask") != "lowdiff" or i == 0:
            continue
        pos = i - 1
        if pos in have:
            continue
        outputs.append({
            "pos": pos,
            "prev": decoder.token(int(x[pos].item())),
            "expected": decoder.token(int(x[i].item())),
            "mask": "lowdiff",
            "hit": None,
        })
        have.add(pos)
    outputs.sort(key=lambda o: o["pos"])
    n_lowdiff = sum(1 for o in outputs if o.get("mask") == "lowdiff")

    return {
        "split": split,
        "step": int(step),
        "micro": int(micro),
        "row": int(row),
        "epoch": None if epoch is None else int(epoch),
        "loss": float(loss),
        "n_answer": n,
        "n_correct": n_hit,
        "n_lowdiff": n_lowdiff,
        "tokens": tokens,
        "outputs": outputs,
    }


class ProbeLog:
    """Append-only JSONL + a small meta index for O(1) page seeks."""

    def __init__(self, out_dir, decoder, append=False):
        self.out_dir = out_dir
        self.decoder = decoder
        self.jsonl, self.meta_path = probe_paths(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        mode = "ab" if append and os.path.isfile(self.jsonl) else "wb"
        self.f = open(self.jsonl, mode)
        self.items = []
        if append and os.path.isfile(self.meta_path):
            try:
                prev = json.load(open(self.meta_path, encoding="utf-8"))
                self.items = list(prev.get("items") or [])
            except (OSError, ValueError):
                self.items = scan_index(self.jsonl)
        elif append:
            self.items = scan_index(self.jsonl)
        self._write_meta()

    def record(self, logits, x, y, split, step, micro=0, epoch=None, k=TOP_K):
        recs = snapshot_batch(logits, x, y, self.decoder, split, step,
                              micro=micro, epoch=epoch, k=k)
        for rec in recs:
            self._write_rec(rec)
        if recs:
            self._write_meta()
        return recs

    def _write_rec(self, rec):
        raw = (json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
               + "\n").encode("utf-8")
        off = self.f.tell()
        self.f.write(raw)
        self.f.flush()
        self.items.append(_index_entry(len(self.items), off, rec))

    def _write_meta(self):
        n_train = sum(1 for it in self.items if it["split"] == "train")
        n_val = sum(1 for it in self.items if it["split"] == "val")
        payload = {
            "version": 1,
            "n": len(self.items),
            "n_train": n_train,
            "n_val": n_val,
            "items": self.items,
        }
        tmp = self.meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, self.meta_path)

    def close(self):
        if self.f is not None:
            self._write_meta()
            self.f.close()
            self.f = None


def _index_entry(i, off, rec):
    return {
        "i": int(i),
        "off": int(off),
        "split": rec["split"],
        "step": rec["step"],
        "micro": rec["micro"],
        "row": rec["row"],
        "epoch": rec.get("epoch"),
        "n_answer": rec["n_answer"],
        "n_correct": rec["n_correct"],
        "loss": rec["loss"],
    }


def scan_index(jsonl_path):
    items = []
    if not os.path.isfile(jsonl_path):
        return items
    with open(jsonl_path, "rb") as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue
            items.append(_index_entry(len(items), off, rec))
    return items


def load_meta(out_dir):
    jsonl, meta_path = probe_paths(out_dir)
    if not os.path.isfile(jsonl):
        return None
    meta = None
    if os.path.isfile(meta_path):
        try:
            meta = json.load(open(meta_path, encoding="utf-8"))
        except (OSError, ValueError):
            meta = None
    jsonl_mtime = os.path.getmtime(jsonl)
    meta_mtime = os.path.getmtime(meta_path) if os.path.isfile(meta_path) else 0
    if meta is None or meta_mtime + 0.05 < jsonl_mtime:
        items = scan_index(jsonl)
        n_train = sum(1 for it in items if it["split"] == "train")
        n_val = sum(1 for it in items if it["split"] == "val")
        meta = {"version": 1, "n": len(items), "n_train": n_train,
                "n_val": n_val, "items": items}
    return meta


def read_record(out_dir, off):
    jsonl, _ = probe_paths(out_dir)
    with open(jsonl, "rb") as f:
        f.seek(int(off))
        line = f.readline()
    if not line:
        raise IndexError("probe offset past EOF")
    return json.loads(line.decode("utf-8"))


def items_for_split(meta, split):
    items = meta.get("items") or []
    if split in (None, "", "all"):
        return items
    return [it for it in items if it["split"] == split]
