"""Fill per-sentence NLL bits so subset bpc is a re-sum, not a rerun.

    python -m data.bruk_sentence_bits \\
      --dir $LLMUA_ROOT/bins/bruk_uk4b_sentenses \\
      --ckpt /path/to/ckpt_final.pt

bpc(subset) = sum(bits) / sum(n_chars)
Packed-val convention: uk4b rule (BOS→first body scored, EOS/pad ignored).
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from data.index import ROOT
from data.shard_util import write_json
from model.packed_val import (IGNORE_INDEX, pack_pretrain_xy,
                              packed_doc_attention_mask)
from model.transformer import doc_block_mask


def load_rows(strings_path):
    rows = []
    with open(strings_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_rows(strings_path, rows):
    with open(strings_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def subset_bpc(rows, pred):
    bits = chars = n = 0
    for r in rows:
        if not pred(r):
            continue
        if r.get("bits") is None:
            raise RuntimeError("bits not filled; run this script first")
        n += 1
        bits += float(r["bits"])
        chars += int(r["n_chars"])
    return {
        "n": n, "bits": bits, "n_chars": chars,
        "bpc": (bits / chars) if chars else None,
    }


def eval_bits(rows, ckpt_path, tok_path, device=None, batch=32, seq_len=None):
    from sample_generator import load_model
    from tokenizers import Tokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(ckpt_path, device)
    model.eval()
    tok = Tokenizer.from_file(tok_path)
    bos_id = tok.token_to_id("<|bos|>")
    eos_id = tok.token_to_id("<|eos|>")
    cfg = ck.get("cfg") or {}
    T = int(seq_len or cfg.get("seq_len") or 2048)

    texts = [r["text"] for r in rows]
    nll = np.zeros(len(rows), dtype=np.float64)
    n_over = 0
    for i0 in range(0, len(texts), batch):
        chunk = texts[i0:i0 + batch]
        encs = tok.encode_batch(chunk)
        xs, ys, take = [], [], []
        for j, enc in enumerate(encs):
            ids = np.asarray(enc.ids, dtype=np.int64)
            if ids.size > T:
                n_over += 1
                continue
            x, y = pack_pretrain_xy(ids, T, bos_id, eos_id, pad_id=bos_id,
                                    rule="uk4b")
            xs.append(x)
            ys.append(y)
            take.append(i0 + j)
        if not xs:
            continue
        x = torch.from_numpy(np.stack(xs)).to(device)
        y = torch.from_numpy(np.stack(ys)).to(device)
        raw = getattr(model, "_orig_mod", model)
        bm = doc_block_mask(x, bos_id) if getattr(raw, "doc_masking", False) else None
        am = None if bm is not None else packed_doc_attention_mask(x, bos_id)
        with torch.no_grad():
            logits = model(x, block_mask=bm, attention_mask=am)
            if isinstance(logits, tuple):
                logits = logits[0]
            per = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1), ignore_index=IGNORE_INDEX,
                reduction="none").view(len(xs), T)
        valid = y != IGNORE_INDEX
        sums = (per * valid).sum(dim=1).cpu().numpy()
        for k, idx in enumerate(take):
            nll[idx] = float(sums[k])
        if (i0 // batch) % 20 == 0:
            print(f"  bits {i0 + len(chunk):,}/{len(texts):,}", flush=True)
    bits = nll / math.log(2)
    for r, b in zip(rows, bits):
        r["bits"] = float(b)
    return bits, n_over, device, T


def run(args):
    out_dir = os.path.abspath(os.path.expanduser(args.dir))
    strings = os.path.join(out_dir, "strings.jsonl")
    rows = load_rows(strings)
    tok_path = os.path.abspath(os.path.expanduser(
        args.tokenizer or os.path.join(str(ROOT), "tokenizers",
                                       "ua_bpe_32768.json")))
    bits, n_over, device, T = eval_bits(
        rows, args.ckpt, tok_path, device=args.device,
        batch=args.batch, seq_len=args.seq_len)
    write_rows(strings, rows)
    np.save(os.path.join(out_dir, "bits.npy"), bits)

    report_path = os.path.join(out_dir, "contamination_report.json")
    report = json.load(open(report_path, encoding="utf-8")) if os.path.isfile(
        report_path) else {}
    subsets = {
        "all_unique": subset_bpc(rows, lambda r: True),
        "minus_exact": subset_bpc(rows, lambda r: not r.get("exact_hit")),
        "minus_exact_and_minhash": subset_bpc(
            rows, lambda r: not r.get("exact_hit") and not r.get("minhash_hit")),
    }
    report["subsets"] = subsets
    report["bits_ckpt"] = os.path.abspath(args.ckpt)
    report["bits_device"] = device
    report["bits_seq_len"] = T
    report["bits_skipped_oversize"] = n_over
    write_json(report_path, report)
    print(json.dumps({
        "device": device, "n": len(rows), "skipped_oversize": n_over,
        "bpc_all_unique": subsets["all_unique"]["bpc"],
        "bpc_minus_exact": subsets["minus_exact"]["bpc"],
        "bpc_minus_exact_and_minhash": subsets["minus_exact_and_minhash"]["bpc"],
        "delta_mh": (
            None if subsets["minus_exact"]["bpc"] is None
            else subsets["minus_exact_and_minhash"]["bpc"]
            - subsets["minus_exact"]["bpc"]),
    }, indent=2))
    return subsets


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default=f"{ROOT}/bins/bruk_uk4b_sentenses")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
