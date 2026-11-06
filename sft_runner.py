"""CLI launcher for a single SFT run.

Same process shape as model_runner.py (RUN_DONE / RUN_PAUSED, checkpoints
under {out_dir}/{model.name}/{run}/) but example-aligned batches and masked
answer loss. See sft_training_loop.py.

    python sft_runner.py --run sft0 --data_dir $LLMUA_ROOT/sft_data/bins/32768 \\
        --load_from_file runs/llmua_P1/foo/ckpt_final.pt --epochs 1 --lr_ratio 0.1
"""
import argparse
import gc
import os
import sys

import torch

from data.index import ROOT
from model.adamw_muon import MuonAdamW, muon_param_groups
from model.training_loop import Paused
from model.transformer import (model_P1, model_P2, model_P3, model_P4,
                               model_target, model_test)
from sft.batcher import resolve_data_dir
from sft_training_loop import SFTConfig, SFTTrainer

FACTORIES = {"test": model_test, "P1": model_P1, "P2": model_P2,
             "P3": model_P3, "P4": model_P4, "target": model_target}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--model", default="test", choices=list(FACTORIES))
    p.add_argument("--data_dir", required=True,
                   help="SFT bins: a name under {data_root}/sft_data/bins "
                        "or an absolute directory with train.bin/val.bin "
                        "+ offsets/lengths")
    p.set_defaults(epoch_sampler=True)
    p.add_argument("--epoch_sampler", action="store_true",
                   help="shuffle examples each epoch (default on)")
    p.add_argument("--no_epoch_sampler", dest="epoch_sampler",
                   action="store_false",
                   help="sequential example order each epoch")
    p.add_argument("--dense_batching", action="store_true",
                   help="pack many full prompts per sequence; BOS attention mask")
    p.add_argument("--out_dir", default=f"{ROOT}/runs")
    p.add_argument("--lr_ratio", type=float, default=0.1,
                   help="peak LR = original (per optimizer) × this")
    p.add_argument("--seed", type=int, default=47)
    p.add_argument("--use_muon", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="max gradient L2 norm after accumulation")
    p.add_argument("--weight_decay", type=float, default=0.1,
                   help="decoupled decay on all 2D weights; norms and "
                        "biases always use 0")
    p.add_argument("--beta1", type=float, default=0.9,
                   help="AdamW β1 (Muon groups ignore this)")
    p.add_argument("--beta2", type=float, default=0.95,
                   help="AdamW β2 (Muon groups ignore this)")
    p.add_argument("--schedule", default="cosine",
                   choices=["cosine", "wsd", "constant"])
    p.add_argument("--warmup", type=float, default=0.05,
                   help="fraction of steps if < 1 (0.05 = 5%%), else step count")
    p.add_argument("--decay_frac", type=float, default=0.2)
    p.add_argument("--micro_batch_size", type=int, default=16)
    p.add_argument("--micro_batch_count", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=32)
    p.add_argument("--probe_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=0)
    p.add_argument("--ckpt_at", type=int, nargs="+", default=None)
    p.add_argument("--ckpt_epochs", type=int, nargs="+", default=None,
                   help="snapshot after completing these epochs, e.g. 1 3 5")
    p.add_argument("--ckpt_best", action="store_true")
    p.add_argument("--ckpt_final", action="store_true")
    p.add_argument("--full_val", action="store_true")
    p.add_argument("--deep_probe", action="store_true",
                   help="write per-window train/val top-k snapshots to JSONL")
    p.add_argument("--deep_probe_every", type=int, default=1,
                   help="keep every Nth train/val window (1 = all)")
    p.add_argument("--lowdiff_masking", action="store_true",
                   help="mask answer tokens copied from the prompt when the "
                        "pair is similar (edit alignment)")
    p.add_argument("--lowdiff_threshold", type=float, default=0.8,
                   help="copied-token fraction of the answer that triggers "
                        "lowdiff masking (default 0.8)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--load_from_file", default=None,
                   help="pretrained ckpt (weights + original LRs) or SFT resume")
    p.add_argument("--enable_init", action="store_true")
    p.add_argument("--enable_swiglu", action="store_true")
    p.add_argument("--enable_rms_norm", action="store_true")
    p.add_argument("--enable_qk_norm", action="store_true")
    p.add_argument("--doc_masking", action="store_true")
    p.add_argument("--doc_rope", action="store_true")
    p.add_argument("--no_biases", action="store_true")
    p.add_argument("--custom_init", action="store_true")
    p.add_argument("--residual_init", action="store_true")
    p.add_argument("--bos_id", type=int, default=0)
    p.add_argument("--usr_id", type=int, default=201)
    p.add_argument("--assistant_id", type=int, default=202)
    p.add_argument("--end_id", type=int, default=203)
    p.add_argument("--p0_loss_multiplier", type=float, default=1.0,
                   help="scale first-answer-token CE in the train loss sum "
                        "(1 = off)")
    return p.parse_args()


def main():
    args = parse_args()
    args.data_dir = resolve_data_dir(args.data_dir)
    shuffle = args.epoch_sampler

    torch.set_float32_matmul_precision("high")
    if args.device != "cpu":
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM {free/2**30:.2f} / {total/2**30:.2f} GiB free")
        assert free > 0.7 * total, "something else is holding VRAM"

    torch.manual_seed(args.seed)
    # dense packing needs BOS-block attention; the flag is existing
    # Transformer machinery, not a new architecture.
    doc_masking = args.doc_masking or args.dense_batching
    model = FACTORIES[args.model](enable_init=args.enable_init,
                                  enable_swiglu=args.enable_swiglu,
                                  enable_rms_norm=args.enable_rms_norm,
                                  enable_qk_norm=args.enable_qk_norm,
                                  no_biases=args.no_biases,
                                  custom_init=args.custom_init,
                                  residual_init=args.residual_init,
                                  doc_masking=doc_masking,
                                  doc_rope=args.doc_rope)

    cfg = SFTConfig(
        data_dir=args.data_dir,
        epoch_sampler=shuffle,
        dense_batching=args.dense_batching,
        out_dir=os.path.join(args.out_dir, model.name, args.run),
        lr_ratio=args.lr_ratio,
        use_muon=args.use_muon,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        schedule=args.schedule,
        warmup=args.warmup,
        decay_frac=args.decay_frac,
        min_lr_ratio=0.0,
        micro_batch=args.micro_batch_size,
        seq_len=model.context_width,
        accum_target=args.micro_batch_count,
        epochs=args.epochs,
        bos_id=args.bos_id, usr_id=args.usr_id,
        assistant_id=args.assistant_id, end_id=args.end_id,
        p0_loss_multiplier=args.p0_loss_multiplier,
        eval_every=args.eval_every, eval_batches=args.eval_batches,
        probe_every=args.probe_every, log_every=10,
        ckpt_every=args.ckpt_every,
        ckpt_at=tuple(args.ckpt_at or ()),
        ckpt_epochs=tuple(args.ckpt_epochs or ()),
        ckpt_best=args.ckpt_best, ckpt_final=args.ckpt_final,
        full_val=args.full_val, deep_probe=args.deep_probe,
        deep_probe_every=args.deep_probe_every,
        lowdiff_masking=args.lowdiff_masking,
        lowdiff_threshold=args.lowdiff_threshold,
        log_file="train.log",
        progress=args.progress, overwrite=args.overwrite,
        load_from_file=args.load_from_file,
        device=args.device,
        dtype="bfloat16" if args.device != "cpu" else "float32",
        compile=args.device != "cpu", seed=args.seed)

    t = SFTTrainer(model, cfg, muon_param_groups=muon_param_groups,
                   MuonAdamW=MuonAdamW)

    try:
        t.train()
    except Paused:
        print("RUN_PAUSED")
        sys.exit(42)
    print("RUN_DONE")


if __name__ == "__main__":
    main()
