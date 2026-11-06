"""Deep probe snapshot + log tests.

Run: python -m sft.test_probe
"""
import json
import os
import tempfile

import numpy as np
import torch

from sft.batcher import IGNORE_INDEX, pack_xy
from sft.probe import (TOP_K, ProbeLog, load_decoder, load_meta, read_record,
                       snapshot_batch)
from sft.tokenize import write_split

BOS, USR, ASST, END = 0, 201, 202, 203


def check(cond, label):
    assert cond, label
    print(f"  ok: {label}")


def _xy():
    ids = np.array([BOS, USR, 10, ASST, 20, 21, END], dtype=np.int64)
    x, y = pack_xy(ids, 16, ASST, BOS, pad_id=BOS)
    return ids, x, y


def test_snapshot_top5_and_correct():
    print("=== snapshot top-k + expected ===")
    ids, x, y = _xy()
    V = 256
    xt = torch.from_numpy(x).unsqueeze(0)
    yt = torch.from_numpy(y).unsqueeze(0)
    logits = torch.zeros(1, 16, V)
    # supervised y positions: 20, 21, END at indices where y != ignore
    pos = (yt[0] != IGNORE_INDEX).nonzero(as_tuple=False).view(-1)
    check(pos.numel() == 3, f"3 answer targets, got {pos.numel()}")
    # first target: expected 20 is argmax
    logits[0, int(pos[0]), 20] = 5.0
    logits[0, int(pos[0]), 7] = 2.0
    # second: expected 21 is rank 6 (outside top-5)
    for r, tok in enumerate([1, 2, 3, 4, 5, 21]):
        logits[0, int(pos[1]), tok] = 10.0 - r
    # third: END expected, tie-ish
    logits[0, int(pos[2]), END] = 1.0
    logits[0, int(pos[2]), 9] = 3.0

    dec = load_decoder(None, BOS, USR, ASST, END)
    recs = snapshot_batch(logits, xt, yt, dec, "train", step=3, micro=0)
    check(len(recs) == 1, "skip nothing; one real row")
    rec = recs[0]
    check(rec["split"] == "train" and rec["step"] == 3, "meta")
    check(rec["tokens"][-1]["id"] == END, "truncated at last expected END")
    check(all(t["id"] != BOS or i == 0 or t.get("role") == "prompt"
              for i, t in enumerate(rec["tokens"])),
          "no right-pad BOS in the window")
    check(rec["n_answer"] == 3, "3 output positions")
    o0, o1, o2 = rec["outputs"]
    check(o0["hit"] and o0["expected"]["id"] == 20, "pos0 argmax hit")
    check(o0["rank"] == 1, f"pos0 rank 1, got {o0['rank']}")
    check(o0["opts"][0]["expected"], "top option is expected when rank 1")
    check(not o1["hit"] and o1["rank"] == 6, f"pos1 rank 6, got {o1['rank']}")
    check(any(o["expected"] and o["id"] == 21 for o in o1["opts"]),
          "expected 21 included outside top-5")
    check(len(o1["opts"]) == TOP_K + 1, "top-5 plus expected")
    ps = [o["p"] for o in o1["opts"]]
    check(ps == sorted(ps, reverse=True), "options ranked by probability")
    check(o2["expected"]["t"] == "<END>", "END labelled")
    check(o0["prev"]["t"] == "<ASSISTANT>", "first pred is after ASSISTANT")
    pad = snapshot_batch(logits, xt, torch.full_like(yt, IGNORE_INDEX),
                         dec, "val", step=0)
    check(pad == [], "pad-only row skipped")
    check(rec.get("n_lowdiff", 0) == 0, "no lowdiff when y is fully supervised")


def test_snapshot_marks_lowdiff():
    print("=== snapshot marks copied tokens ===")
    # prompt 10 11 12 13 14, answer is an exact copy + END
    ids = np.array([BOS, USR, 10, 11, 12, 13, 14, ASST, 10, 11, 12, 13, 14, END],
                   dtype=np.int64)
    x, y = pack_xy(ids, 24, ASST, BOS, pad_id=BOS, usr_id=USR, end_id=END,
                   lowdiff_masking=True, lowdiff_threshold=0.8)
    asst = int(np.flatnonzero(ids == ASST)[0])
    check(y[asst] == IGNORE_INDEX, "copies dropped from y")
    check(y[asst + 5] == END, "END remains a target")
    V = 256
    xt = torch.from_numpy(x).unsqueeze(0)
    yt = torch.from_numpy(y).unsqueeze(0)
    logits = torch.zeros(1, 24, V)
    pos = (yt[0] != IGNORE_INDEX).nonzero(as_tuple=False).view(-1)
    check(pos.numel() == 1, f"only END is a loss target, got {pos.numel()}")
    logits[0, int(pos[0]), END] = 5.0
    dec = load_decoder(None, BOS, USR, ASST, END)
    rec = snapshot_batch(logits, xt, yt, dec, "val", step=1)[0]
    check(rec["n_answer"] == 1, "n_answer counts loss tokens only")
    check(rec["n_lowdiff"] == 5, f"5 copied tokens, got {rec['n_lowdiff']}")
    copied = [t for t in rec["tokens"] if t.get("mask") == "lowdiff"]
    check(len(copied) == 5, "tokens carry mask=lowdiff")
    check(all(t["role"] == "answer" for t in copied),
          "copied tokens stay role=answer")
    check(all(t["id"] in (10, 11, 12, 13, 14) for t in copied),
          "copied ids are the prompt span")
    masked_out = [o for o in rec["outputs"] if o.get("mask") == "lowdiff"]
    check(len(masked_out) == 5, "outputs include copied columns")
    check(all(o.get("hit") is None and "opts" not in o for o in masked_out),
          "copied columns have no top-k / hit")
    check(rec["outputs"][-1]["expected"]["id"] == END, "END column last")


def test_probe_log_roundtrip():
    print("=== probe log seek ===")
    ids, x, y = _xy()
    V = 256
    xt = torch.from_numpy(x).unsqueeze(0)
    yt = torch.from_numpy(y).unsqueeze(0)
    logits = torch.randn(1, 16, V)
    dec = load_decoder(None, BOS, USR, ASST, END)
    with tempfile.TemporaryDirectory() as d:
        log = ProbeLog(d, dec)
        log.record(logits, xt, yt, "train", step=0, micro=0)
        log.record(logits, xt, yt, "val", step=0, micro=0)
        log.close()
        meta = load_meta(d)
        check(meta["n"] == 2, f"n=2 got {meta['n']}")
        check(meta["n_train"] == 1 and meta["n_val"] == 1, "split counts")
        rec = read_record(d, meta["items"][1]["off"])
        check(rec["split"] == "val", "seek second record")
        check("outputs" in rec and rec["outputs"], "payload intact")


def test_tiny_train_writes_log():
    print("=== tiny train writes probe log ===")
    import torch as _torch
    from model.adamw_muon import MuonAdamW, muon_param_groups
    from model.transformer import Transformer
    from sft_training_loop import SFTConfig, SFTTrainer

    def ex(q, a):
        return [BOS, USR, *q, ASST, *a, END]

    with tempfile.TemporaryDirectory() as d:
        data = os.path.join(d, "bins")
        os.makedirs(data)
        train = [ex([50 + i, 11], [60 + i, 21]) for i in range(4)]
        val = [ex([80, 31], [90, 41]) for _ in range(2)]
        write_split(data, "train", train, BOS)
        write_split(data, "val", val, BOS)
        json.dump({"vocab_size": 512, "bos_id": BOS, "assistant_id": ASST,
                   "chars_per_token_val": 1.0},
                  open(os.path.join(data, "meta.json"), "w"))
        out = os.path.join(d, "run")
        _torch.manual_seed(0)
        model = Transformer(name="sft_tiny", vocab_size=512, width=32,
                            embeddings=32, blocks=1, headsize=16,
                            no_biases=True, custom_init=True, dropout_p=0.0)
        cfg = SFTConfig(
            data_dir=data, out_dir=out, epochs=1, lr_ratio=1.0,
            orig_lr=1e-3, warmup=0, schedule="constant",
            micro_batch=2, seq_len=32, accum_target=1,
            eval_every=0, eval_batches=4, probe_every=0,
            ckpt_final=False, device="cpu", dtype="float32",
            compile=False, seed=0, log_file="train.log",
            deep_probe=True, epoch_sampler=False)
        t = SFTTrainer(model, cfg, muon_param_groups=muon_param_groups,
                       MuonAdamW=MuonAdamW)
        t.train()
        meta = load_meta(out)
        check(meta is not None, "probe log exists")
        check(meta["n_train"] > 0, f"train windows {meta['n_train']}")
        check(meta["n_val"] > 0, f"val windows {meta['n_val']}")
        rec = read_record(out, meta["items"][0]["off"])
        check(rec["tokens"][0]["t"] == "<BOS>", "specials decoded")
        check(any(o["opts"] for o in rec["outputs"]), "top-k stored")
        log = open(os.path.join(out, "train.log")).read()
        check("deep_probe" in log, "train.log mentions deep_probe")
        n_train_all = meta["n_train"]
        n_val_all = meta["n_val"]

        out2 = os.path.join(d, "run_sparse")
        _torch.manual_seed(0)
        model2 = Transformer(name="sft_tiny", vocab_size=512, width=32,
                             embeddings=32, blocks=1, headsize=16,
                             no_biases=True, custom_init=True, dropout_p=0.0)
        cfg2 = SFTConfig(
            data_dir=data, out_dir=out2, epochs=1, lr_ratio=1.0,
            orig_lr=1e-3, warmup=0, schedule="constant",
            micro_batch=2, seq_len=32, accum_target=1,
            eval_every=0, eval_batches=4, probe_every=0,
            ckpt_final=False, device="cpu", dtype="float32",
            compile=False, seed=0, log_file="train.log",
            deep_probe=True, deep_probe_every=2, epoch_sampler=False)
        t2 = SFTTrainer(model2, cfg2, muon_param_groups=muon_param_groups,
                        MuonAdamW=MuonAdamW)
        t2.train()
        meta2 = load_meta(out2)
        check(meta2["n_train"] < n_train_all,
              f"sparse train {meta2['n_train']} < dense {n_train_all}")
        check(meta2["n_val"] < n_val_all,
              f"sparse val {meta2['n_val']} < dense {n_val_all}")
        log2 = open(os.path.join(out2, "train.log")).read()
        check("every 2" in log2, "log mentions stride")


def main():
    test_snapshot_top5_and_correct()
    test_snapshot_marks_lowdiff()
    test_probe_log_roundtrip()
    test_tiny_train_writes_log()
    print("PROBE PASS")


if __name__ == "__main__":
    main()
