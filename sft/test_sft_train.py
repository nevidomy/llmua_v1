"""Tiny CPU SFT train: example-aligned loss, epoch ckpt, peak LR.

Run: python -m sft.test_sft_train
"""
import json
import os
import tempfile

import torch

from model.adamw_muon import MuonAdamW, muon_param_groups
from model.transformer import Transformer, _HAS_FLEX
from sft.tokenize import write_split
from sft.batcher import IGNORE_INDEX
from sft_training_loop import SFTConfig, SFTTrainer, answer_tokens_per_step, resolve_sft_cpt

BOS, USR, ASST, END = 0, 201, 202, 203


def ex(q, a):
    return [BOS, USR, *q, ASST, *a, END]


def write_bins(d, n_train=8, n_val=4, T=64):
    train = [ex([50 + i, 11], [60 + i, 21]) for i in range(n_train)]
    val = [ex([80 + i, 31], [90, 41]) for i in range(n_val)]
    assert all(len(x) <= T for x in train + val)
    write_split(d, "train", train, BOS)
    write_split(d, "val", val, BOS)
    json.dump({"vocab_size": 512, "bos_id": BOS, "assistant_id": ASST,
               "chars_per_token_val": 1.0},
              open(os.path.join(d, "meta.json"), "w"))


def check(cond, label):
    assert cond, label
    print(f"  ok: {label}")


def tiny_model(T=64):
    return Transformer(name="sft_tiny", vocab_size=512, width=T,
                       embeddings=32, blocks=2, headsize=16,
                       no_biases=True, custom_init=True, dropout_p=0.0)


def main():
    print("=== tiny SFT train ===")
    with tempfile.TemporaryDirectory() as d:
        data = os.path.join(d, "bins")
        os.makedirs(data)
        write_bins(data)
        out = os.path.join(d, "run")
        torch.manual_seed(0)
        model = tiny_model()
        cfg = SFTConfig(
            data_dir=data, out_dir=out, epochs=2, lr_ratio=0.5,
            orig_lr=1e-3, warmup=0.25, schedule="cosine",
            micro_batch=2, seq_len=64, accum_target=1,
            eval_every=1, eval_batches=4, probe_every=0,
            ckpt_final=True, ckpt_epochs=(1, 2), full_val=True,
            device="cpu", dtype="float32", compile=False, seed=47,
            log_file="train.log", log_every=1, progress=False,
            epoch_sampler=True, dense_batching=False)
        t = SFTTrainer(model, cfg, muon_param_groups=muon_param_groups,
                       MuonAdamW=MuonAdamW)
        check(abs(t.cfg.lr - 5e-4) < 1e-12, f"peak lr = orig×ratio ({t.cfg.lr})")
        check(t.plan["steps"] == t._train_batcher.total_steps, "plan steps")
        check(t.cfg.warmup_steps >= 1, f"warmup resolved ({t.cfg.warmup_steps})")
        # first LR after sched.step() in loop is from-zero at step 0
        f0 = t.sched.factor() if t.sched.last_epoch >= 0 else None
        result = t.train()
        check("val_nats" in result, "train returns val")
        check(os.path.isfile(os.path.join(out, "ckpt_final.pt")), "ckpt_final")
        check(os.path.isfile(os.path.join(out, "ckpt_epoch_1.pt")),
              "ckpt_epoch_1")
        check(os.path.isfile(os.path.join(out, "ckpt_epoch_2.pt")),
              "ckpt_epoch_2")
        log = open(os.path.join(out, "train.log")).read()
        check("SFT" in log, "train.log marked SFT")
        check("EVAL step" in log, "eval lines in the log")
        check(" p0 " in log, "p0 loss in train/eval log")
        check("FULL_VAL" in log, "full val ran at end of train")
        check(result.get("val_p0_nats") is not None, "train returns val_p0")
        check(all("val_p0_nats" in h for h in t.history),
              "history stores val_p0")
        check(isinstance(t.full_val, dict)
              and t.full_val.get("val_p0_nats") is not None,
              "full_val stores p0")
        nats, bpc, p0 = t.evaluate(probe=False)
        tot = n = 0
        t.model.eval()
        with torch.no_grad():
            for (x, y), bm in zip(t.eval_batches, t.eval_block_masks):
                logits = t.model(x, block_mask=bm)
                per = torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    y.reshape(-1), ignore_index=IGNORE_INDEX,
                    reduction="none")
                m = (x == ASST) & (y != IGNORE_INDEX)
                tot += per[m.reshape(-1)].sum().item()
                n += int(m.sum())
        check(n > 0 and abs(p0 - tot / n) < 1e-4,
              f"p0 is mean first-answer CE ({p0:.4f})")
        check("done:" in log, "done sentinel in train.log")
        check("answer-token scale" in log, "logs token-weighted accum scale")
        check(t._tok_scale > 1, f"tok_scale set ({t._tok_scale})")

        ck = torch.load(os.path.join(out, "ckpt_final.pt"),
                        map_location="cpu", weights_only=False)
        check(ck["cfg"]["kind"] == "sft", "ckpt cfg.kind = sft")
        check(ck.get("model_arch") is not None, "model_arch stored for tools")
        check(isinstance(ck.get("full_val"), dict)
              and ck["full_val"].get("val_p0_nats") is not None,
              "ckpt full_val includes p0")
        check(ck["final"].get("val_p0_nats") is not None,
              "ckpt final includes p0")

        print("=== dense packing train ===")
        # flex_attention has no CPU backward; packing is covered by test_batcher
        if not _HAS_FLEX or not torch.cuda.is_available():
            print("  skip: dense train needs CUDA flex_attention backward")
        else:
            out2 = os.path.join(d, "run_dense")
            torch.manual_seed(0)
            model2 = tiny_model()
            cfg2 = SFTConfig(
                data_dir=data, out_dir=out2, epochs=1, lr_ratio=1.0,
                orig_lr=1e-3, warmup=0, schedule="constant",
                micro_batch=2, seq_len=64, accum_target=1,
                eval_every=0, eval_batches=4, probe_every=0,
                ckpt_final=True, device="cuda", dtype="float32",
                compile=False, seed=0, log_file=None,
                dense_batching=True, epoch_sampler=True)
            t2 = SFTTrainer(model2, cfg2, muon_param_groups=muon_param_groups,
                            MuonAdamW=MuonAdamW)
            t2.train()
            check(os.path.isfile(os.path.join(out2, "ckpt_final.pt")),
                  "dense ckpt_final")

    print("SFT TRAIN PASS")


def test_accum_token_weighted():
    print("=== token-weighted accum ===")
    import torch.nn as nn
    import torch.nn.functional as F

    class Cfg:
        assistant_id = 202
        bos_id = 0
        epochs = 2

    scale = answer_tokens_per_step(None, Cfg(), {"steps": 10},
                                   {"n_answer_tokens_train": 100})
    check(abs(scale - 20.0) < 1e-9, "100 ans × 2 ep / 10 steps = 20")

    torch.manual_seed(0)
    lin = nn.Linear(8, 16, bias=False)
    # identical positions so per-token CE (and grad) is the same everywhere
    x = torch.randn(1, 1, 8).expand(1, 12, 8).contiguous()
    acc, n1, n2, tok_scale = 2, 2, 10, 12.0

    def y_of(n):
        y = torch.full((1, 12), IGNORE_INDEX, dtype=torch.long)
        y[0, :n] = 3
        return y

    def contrib(n, how):
        lin.zero_grad()
        logits = lin(x)
        y = y_of(n)
        if how == "mean":
            loss = F.cross_entropy(logits.reshape(-1, 16), y.reshape(-1),
                                   ignore_index=IGNORE_INDEX)
            (loss / acc).backward()
        else:
            ce = F.cross_entropy(logits.reshape(-1, 16), y.reshape(-1),
                                 ignore_index=IGNORE_INDEX, reduction="sum")
            (ce / tok_scale).backward()
        return lin.weight.grad.detach().clone()

    g_thin_m = contrib(n1, "mean")
    g_fat_m = contrib(n2, "mean")
    g_thin_s = contrib(n1, "sum")
    g_fat_s = contrib(n2, "sum")
    r_mean = float(g_fat_m.norm() / g_thin_m.norm())
    r_sum = float(g_fat_s.norm() / g_thin_s.norm())
    check(abs(r_mean - 1.0) < 0.15, f"mean/acc weights micros equally ({r_mean:.2f})")
    check(abs(r_sum - 5.0) < 0.5, f"sum/scale weights 10:2 tokens ({r_sum:.2f})")
    print("  ok: token-weighted accum")


def test_sft_cpt():
    print("=== SFT bpc uses answer chars/token ===")
    import math
    cpt = resolve_sft_cpt(None, None, {"chars_per_token_val": 3.5})
    check(abs(cpt - 3.5) < 1e-12, "meta chars_per_token_val")
    nats = 2.0
    bpc = nats / cpt / math.log(2)
    check(bpc < nats, f"bpc {bpc:.3f} < nats when cpt=3.5")
    bpc1 = nats / 1.0 / math.log(2)
    check(bpc1 > nats, "cpt=1 masquerades as bits/token (bpc > nats)")


def test_p0_multiplier():
    print("=== p0_loss_multiplier scales first-token CE ===")
    x = torch.tensor([[BOS, USR, 10, ASST, 20, 21, END, BOS]])
    y = torch.full((1, 8), IGNORE_INDEX, dtype=torch.long)
    y[0, 3] = 20
    y[0, 4] = 21
    y[0, 5] = END
    torch.manual_seed(0)
    base = torch.randn(1, 8, 256)

    class Stub:
        def __init__(self, m):
            self.cfg = SFTConfig(p0_loss_multiplier=m)

    def grads(m):
        logits = base.clone().detach().requires_grad_(True)
        _per, ce, p0_ce, ce_w, p0m = SFTTrainer._ce_train(Stub(m), logits, x, y)
        ce_w.backward()
        g = logits.grad[0]
        return (float(g[3].abs().sum()), float(g[4].abs().sum()),
                float(g[0].abs().sum()), float(ce.item()), float(p0m.sum()))

    p0_1, a1_1, bos_1, ce_1, n_p0 = grads(1.0)
    p0_3, a1_3, bos_3, ce_3, _ = grads(3.0)
    check(n_p0 == 1, "one p0 position")
    check(bos_1 == 0.0 and bos_3 == 0.0, "prompt position has no grad")
    check(abs(ce_1 - ce_3) < 1e-6, "unweighted CE sum unchanged")
    check(abs(a1_3 / a1_1 - 1.0) < 1e-4, f"other answer grad unchanged ({a1_3/a1_1:.3f})")
    check(abs(p0_3 / p0_1 - 3.0) < 1e-4, f"p0 grad ×3 ({p0_3/p0_1:.3f})")
    try:
        SFTConfig(p0_loss_multiplier=-1)
        check(False, "negative multiplier rejected")
    except AssertionError:
        check(True, "negative multiplier rejected")


def test_lowdiff_config_and_scale():
    print("=== lowdiff recounts answer tokens ===")
    try:
        SFTConfig(lowdiff_threshold=0)
        check(False, "threshold 0 rejected")
    except AssertionError:
        check(True, "threshold 0 rejected")
    try:
        SFTConfig(lowdiff_threshold=1.5)
        check(False, "threshold >1 rejected")
    except AssertionError:
        check(True, "threshold >1 rejected")

    class Cfg:
        assistant_id = ASST
        bos_id = BOS
        usr_id = USR
        end_id = END
        epochs = 1
        lowdiff_masking = True
        lowdiff_threshold = 0.8

    with tempfile.TemporaryDirectory() as d:
        data = os.path.join(d, "bins")
        os.makedirs(data)
        # exact copy: 3 answer tokens + END, all 3 content tokens masked
        train = [ex([10, 11, 12], [10, 11, 12]) for _ in range(4)]
        write_split(data, "train", train, BOS)
        write_split(data, "val", train[:1], BOS)
        from sft.batcher import SFTSplit
        split = SFTSplit(data, "train")
        raw = answer_tokens_per_step(
            split, Cfg(), {"steps": 4},
            {"n_answer_tokens_train": 999})
        # meta is ignored when lowdiff is on: 4 examples × 1 END = 4
        # tokens / 4 steps = 1
        check(abs(raw - 1.0) < 1e-9,
              f"lowdiff ignores meta and drops copies ({raw})")
        Cfg.lowdiff_masking = False
        unmask = answer_tokens_per_step(
            split, Cfg(), {"steps": 4},
            {"n_answer_tokens_train": 16})
        check(abs(unmask - 4.0) < 1e-9, "without lowdiff, meta 16/4 = 4")

        json.dump({"vocab_size": 512, "bos_id": BOS, "assistant_id": ASST,
                   "chars_per_token_val": 1.0},
                  open(os.path.join(data, "meta.json"), "w"))
        out = os.path.join(d, "run")
        torch.manual_seed(0)
        model = tiny_model(T=32)
        cfg = SFTConfig(
            data_dir=data, out_dir=out, epochs=1, lr_ratio=1.0,
            orig_lr=1e-3, warmup=0, schedule="constant",
            micro_batch=2, seq_len=32, accum_target=1,
            eval_every=0, eval_batches=2, probe_every=0,
            ckpt_final=False, device="cpu", dtype="float32",
            compile=False, seed=0, log_file="train.log",
            deep_probe=True, epoch_sampler=False,
            lowdiff_masking=True, lowdiff_threshold=0.8)
        t = SFTTrainer(model, cfg, muon_param_groups=muon_param_groups,
                       MuonAdamW=MuonAdamW)
        t.train()
        log = open(os.path.join(out, "train.log")).read()
        check("lowdiff_masking on" in log, "train.log mentions lowdiff")
        from sft.probe import load_meta, read_record
        meta = load_meta(out)
        rec = read_record(out, meta["items"][0]["off"])
        check(rec.get("n_lowdiff", 0) > 0, "probe records copied tokens")
        check(any(tk.get("mask") == "lowdiff" for tk in rec["tokens"]),
              "probe tokens marked mask=lowdiff")


if __name__ == "__main__":
    test_accum_token_weighted()
    test_sft_cpt()
    test_p0_multiplier()
    test_lowdiff_config_and_scale()
    main()
