"""Packed BOS…EOS val helpers + a tiny CPU packed full-val.

Run: python -m model.test_packed_val
"""
import json
import math
import os
import tempfile

import numpy as np
import torch

from model.packed_val import (IGNORE_INDEX, bos_eos_spans, collect_docs,
                              custom_loss_key, is_custom_loss_key,
                              pack_doc_rows, pack_pretrain_xy)
from model.training_loop import TrainConfig, Trainer
from model.transformer import Transformer


BOS, EOS = 0, 1


def check(cond, label):
    assert cond, label
    print(f"  ok: {label}")


def test_keys():
    print("=== custom loss keys ===")
    check(custom_loss_key("pretrain", "15k") == "full_val_loss_15k",
          "stream pretrain")
    check(custom_loss_key("pretrain", "15k", "packed")
          == "full_val_loss_packed_15k", "packed pretrain")
    check(custom_loss_key("pretrain", "bruk_uk4b_sentenses", "packed_uk4b")
          == "full_val_loss_packed_uk4b_bruk_uk4b_sentenses",
          "packed_uk4b pretrain")
    check(custom_loss_key("sft", "ua_instruct") == "full_val_loss_sft_ua_instruct",
          "sft one-per-row")
    check(custom_loss_key("sft", "ua_instruct", "packed")
          == "full_val_loss_sft_packed_ua_instruct", "sft packed")
    check(is_custom_loss_key("full_val_loss_15k"), "is stream key")
    check(is_custom_loss_key("full_val_loss_packed_15k"), "is packed key")
    check(is_custom_loss_key("full_val_loss_packed_uk4b_bruk"),
          "is packed_uk4b key")
    check(is_custom_loss_key("full_val_loss_sft_ua"), "is sft key")
    check(is_custom_loss_key("full_val_loss_sft_packed_ua"), "is sft packed key")
    check(not is_custom_loss_key("full_val"), "native key rejected")
    try:
        custom_loss_key("pretrain", "../x")
        check(False, "bad name should raise")
    except ValueError:
        check(True, "bad name rejected")
    try:
        custom_loss_key("sft", "ua", "packed_uk4b")
        check(False, "sft packed_uk4b should raise")
    except ValueError:
        check(True, "sft packed_uk4b rejected")


def test_spans_and_pack():
    print("=== BOS/EOS spans and pack_xy ===")
    # BOS a a EOS BOS b EOS junk BOS c c c EOS
    data = np.array([BOS, 10, 11, EOS, BOS, 12, EOS, 99, BOS, 13, 14, 15, EOS],
                    dtype=np.int64)
    spans = bos_eos_spans(data, BOS, EOS)
    check(spans == [(0, 4), (4, 7), (8, 13)], f"spans {spans}")
    docs, n_over, n_no = collect_docs(data, BOS, EOS, max_len=4)
    check(len(docs) == 2 and n_over == 1 and n_no == 0,
          f"fit 2, skip oversize 1 (got {len(docs)}/{n_over}/{n_no})")
    rows = pack_doc_rows([len(d) for d in docs], T=8)
    check(rows == [[0, 1]], f"one packed row {rows}")
    ids = np.concatenate(docs)
    x, y = pack_pretrain_xy(ids, 8, BOS, EOS, pad_id=BOS)
    # ids = BOS 10 11 EOS BOS 12 EOS
    # loss only 10→11
    check(int((y != IGNORE_INDEX).sum()) == 1, f"one content transition {y}")
    check(int(y[1]) == 11, "target of 10 is 11")
    check(int(y[0]) == IGNORE_INDEX, "BOS position ignored")
    check(int(y[2]) == IGNORE_INDEX, "predicting EOS ignored")
    check(int(y[3]) == IGNORE_INDEX, "EOS→BOS ignored")
    check(int(y[7]) == IGNORE_INDEX, "pad ignored")
    yu = pack_pretrain_xy(ids, 8, BOS, EOS, pad_id=BOS, rule="uk4b")[1]
    check(int((yu != IGNORE_INDEX).sum()) == 3, f"uk4b BOS→body {yu}")
    check(int(yu[0]) == 10, "uk4b BOS predicts 10")
    check(int(yu[1]) == 11, "uk4b 10→11")
    check(int(yu[2]) == IGNORE_INDEX, "uk4b predicting EOS ignored")
    check(int(yu[4]) == 12, "uk4b BOS predicts 12")
    empty = pack_pretrain_xy(
        np.array([BOS, EOS], dtype=np.int64), 8, BOS, EOS,
        pad_id=BOS, rule="uk4b")[1]
    check(int((empty != IGNORE_INDEX).sum()) == 0, f"uk4b [BOS,EOS] empty {empty}")
    one = pack_pretrain_xy(
        np.array([BOS, 10, EOS], dtype=np.int64), 8, BOS, EOS,
        pad_id=BOS, rule="uk4b")[1]
    check(int((one != IGNORE_INDEX).sum()) == 1 and int(one[0]) == 10,
          f"uk4b one-token sentence {one}")


def write_pretrain(d, docs, vocab=128):
    stream = []
    for content in docs:
        stream.extend([BOS, *content, EOS])
    arr = np.array(stream, dtype=np.uint16)
    arr.tofile(os.path.join(d, "val.bin"))
    arr.tofile(os.path.join(d, "train.bin"))
    json.dump({"vocab_size": vocab, "bos_id": BOS, "eos_id": EOS,
               "chars_per_token_val": 1.0, "n_tokens_val": int(arr.size),
               "n_chars_val": 90},
              open(os.path.join(d, "meta.json"), "w"))
    return arr


def tiny(T=16):
    return Transformer(name="pv_tiny", vocab_size=128, width=T,
                       embeddings=32, blocks=2, headsize=16,
                       no_biases=True, custom_init=True, dropout_p=0.0,
                       bos_id=BOS)


def test_packed_eval():
    print("=== packed full-val on tiny CPU model ===")
    with tempfile.TemporaryDirectory() as d:
        write_pretrain(d, [[10, 11, 12], [13, 14], [15, 16, 17, 18]])
        torch.manual_seed(0)
        model = tiny(T=16)
        cfg = TrainConfig(
            bins_dir=d, out_dir=os.path.join(d, "run"),
            micro_batch=2, seq_len=16, accum_target=1,
            eval_only=True, device="cpu", dtype="float32", compile=False)
        t = Trainer(model, cfg)
        nats, bpc, info = t.evaluate_full_val_packed()
        check(info["docs"] == 3, f"3 docs ({info['docs']})")
        check(info["skipped_oversize"] == 0, "none oversize")
        check(info["tokens"] == 6, f"6 content transitions ({info['tokens']})")
        check(info["n_first"] == 0, f"content n_first {info['n_first']}")
        check(info["mode"] == "packed", f"mode {info['mode']}")
        check(info["bpc_mode"] == "cpt", f"bpc_mode {info['bpc_mode']}")
        check(abs(bpc - nats / t.cpt / math.log(2)) < 1e-12,
              "packed bpc is mean/cpt")
        check(np.isfinite(nats) and nats > 0, f"nats {nats}")
        nats_u, bpc_u, info_u = t.evaluate_full_val_packed(rule="uk4b")
        check(info_u["tokens"] == 9, f"uk4b 9 body targets ({info_u['tokens']})")
        check(info_u["n_first"] == 3, f"uk4b n_first {info_u['n_first']}")
        check(info_u["mode"] == "packed_uk4b", f"uk4b mode {info_u['mode']}")
        check(info_u["bpc_mode"] == "chars", f"uk4b bpc_mode {info_u['bpc_mode']}")
        check(abs(bpc_u - info_u["bits_sum"] / 90) < 1e-12,
              "uk4b bpc is Σ bits / chars")
        check(info_u["bpc_cpt"] is not None, "bpc_cpt sidecar")
        check(np.isfinite(nats_u) and nats_u > 0, f"uk4b nats {nats_u}")
        t.meta.pop("n_chars_val")
        try:
            t.evaluate_full_val_packed(rule="uk4b")
            check(False, "uk4b without n_chars_val should raise")
        except RuntimeError as e:
            check("n_chars_val" in str(e), f"uk4b requires n_chars_val ({e})")


def main():
    test_keys()
    test_spans_and_pack()
    test_packed_eval()
    print("PACKED VAL PASS")


if __name__ == "__main__":
    main()
