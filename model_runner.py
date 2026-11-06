"""CLI launcher for a single training run.

Per-run knobs come in as flags; everything else is fixed here to match the
Paper 1 reference setup (warmup 200, eval cadence, bfloat16 + compile,
32k tokenizer bins). Checkpoints and the text log land in
{out_dir}/{model.name}/{run}/. Usually launched by main_runner.py, but
works standalone:

    python model_runner.py --run baseline --train_token_multiplier 20
"""

import argparse
import gc
import os
import sys

import torch

from data.index import ROOT
from model.adamw_muon import MuonAdamW, muon_param_groups
from model.training_loop import Paused, TrainConfig, Trainer, resolve_bins_dir
from model.transformer import (model_P1, model_P2, model_P3, model_P4,
                               model_target, model_test)

FACTORIES = {"test": model_test, "P1": model_P1, "P2": model_P2,
             "P3": model_P3, "P4": model_P4, "target": model_target}

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True,
                   help="run identifier: the out_dir subdirectory name")
    p.add_argument("--model", default="test", choices=list(FACTORIES),
                   help="model factory from transformer.py")
    p.add_argument("--bins_dir", default="32768",
                   help="token bin directory: a name under "
                        "{data_root}/bins (default 32768), or an absolute path")
    p.add_argument("--epoch_sampler", action="store_true",
                   help="exact-coverage EpochSampler over sharded "
                        "train_XXXX.bin; default is random windows on a "
                        "single train.bin")
    p.add_argument("--out_dir", default=f"{ROOT}/runs",
                   help="base output dir; run writes to {out_dir}/model/run")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=47,
                   help="Seed for model initialization and data order")
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
    p.add_argument("--muon_lr", type=float, default=None,
                   help="absolute Muon lr; default lr * 30 (muon_lr_mult)")
    p.add_argument("--schedule", default="cosine",
                   choices=["cosine", "wsd", "constant"])
    p.add_argument("--warmup_steps", type=int, default=200,
                   help="LR warmup steps (ignored for decay branches, "
                        "which force 0)")
    p.add_argument("--decay_frac", type=float, default=0.2,
                   help="fraction of the run spent in LR decay / cooldown "
                        "(ignored for decay branches, which force 1.0)")
    p.add_argument("--micro_batch_size", type=int, default=16)
    p.add_argument("--micro_batch_count", type=int, default=1,
                   help="micro-steps per optimizer step; constant effective "
                        "batch, no ramp")
    p.add_argument("--train_token_multiplier", type=float, default=20.0,
                   help="max_tokens = model.num_params() * this (20 = "
                        "Chinchilla)")
    p.add_argument("--max_steps", type=int, default=None,
                   help="fixed step count; overrides train_token_multiplier")
    p.add_argument("--eval_every", type=int, default=100,
                   help="val eval cadence in steps; 0 = disabled "
                        "(the end-of-run eval still runs)")
    p.add_argument("--eval_batches", type=int, default=32,
                   help="total val sequences per eval; must be divisible "
                        "by micro_batch_size * micro_batch_count")
    p.add_argument("--probe_every", type=int, default=500,
                   help="update-to-param probe cadence in steps; "
                        "0 = disabled")
    p.add_argument("--ckpt_every", type=int, default=0,
                   help="periodic checkpoint cadence in steps; 0 = disabled")
    p.add_argument("--ckpt_at", type=int, nargs="+", default=None,
                   help="one-off checkpoints at these steps; validated "
                        "against the planned run length at startup")
    p.add_argument("--ckpt_at_tokens", nargs="+", default=None,
                   metavar="NAME=TOKENS",
                   help="one-off checkpoints by token count: save NAME.pt at "
                        "the first step reaching TOKENS, e.g. "
                        "--ckpt_at_tokens ckpt_10M=10000000")
    p.add_argument("--cooldown_checkpoints", type=float, nargs="+",
                   default=None, metavar="MULT",
                   help="WSD branch checkpoints by token multiplier, e.g. "
                        "--cooldown_checkpoints 20 40.5 80; saves "
                        "cooldown_<mult>.pt at each branch point "
                        "(M*params minus the decay tail). Fractions ok; "
                        "20 and 20.0 both write cooldown_20.pt")
    p.add_argument("--ckpt_pre_cooldown", action="store_true",
                   help="save pre_cooldown.pt at this run's WSD decay-start "
                        "token count (total minus decay_frac of post-warmup)")
    p.add_argument("--ckpt_best", action="store_true",
                   help="keep best-val checkpoint (ckpt_best.pt)")
    p.add_argument("--ckpt_final", action="store_true",
                   help="save final checkpoint (ckpt_final.pt)")
    p.add_argument("--full_val", action="store_true",
                   help="after the run, evaluate the full val split "
                        "(same as `evaluate.py full-val`) and store "
                        "it on ckpt_final.pt as full_val")
    p.add_argument("--overwrite", action="store_true",
                   help="allow starting into a non-empty out_dir (default: "
                        "refuse, to protect previous runs)")
    p.add_argument("--progress", action="store_true",
                   help="tqdm bar on stderr; main_runner parses it into "
                        "its own display")
    p.add_argument("--device", default="cuda",
                   help="cuda (default) or cpu; cpu forces float32, "
                        "no compile")
    p.add_argument("--load_from_file", default=None,
                   help="resume full training state from this checkpoint file")
    p.add_argument("--extend_schedule", action="store_true",
                   help="after --load_from_file, rebuild the LR schedule for "
                        "this run's plan instead of restoring the saved "
                        "cycle (continue with a bigger budget)")
    p.add_argument("--load_from_file_for_decay", default=None,
                   help="load weights from this stable-phase checkpoint and "
                        "run this run's own WSD decay schedule; forces "
                        "warmup_steps=0, decay_frac=1.0")
    p.add_argument("--enable_init", action="store_true")
    p.add_argument("--enable_swiglu", action="store_true")
    p.add_argument("--enable_rms_norm", action="store_true")
    p.add_argument("--enable_qk_norm", action="store_true")
    p.add_argument("--doc_masking", action="store_true",
                   help="intra-document attention mask (flex_attention)")
    p.add_argument("--doc_rope", action="store_true",
                   help="RoPE positions reset at each BOS")
    # fine-grained init breakdown; enable_init implies all three
    p.add_argument("--no_biases", action="store_true")
    p.add_argument("--custom_init", action="store_true")
    p.add_argument("--residual_init", action="store_true",
                   help="scale proj/lin2 by 1/sqrt(2*blocks); only has "
                        "effect together with custom_init")
    return p.parse_args()


def main():
    args = parse_args()
    args.bins_dir = resolve_bins_dir(args.bins_dir)

    torch.set_float32_matmul_precision("high")
    if args.device != "cpu":
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM {free/2**30:.2f} / {total/2**30:.2f} GiB free")
        assert free > 0.7 * total, "something else is holding VRAM"

    torch.manual_seed(args.seed)
    model = FACTORIES[args.model](enable_init=args.enable_init,
                                  enable_swiglu=args.enable_swiglu,
                                  enable_rms_norm=args.enable_rms_norm,
                                  enable_qk_norm=args.enable_qk_norm,
                                  no_biases=args.no_biases,
                                  custom_init=args.custom_init,
                                  residual_init=args.residual_init,
                                  doc_masking=args.doc_masking,
                                  doc_rope=args.doc_rope)

    # a decay branch is its own short run: load_for_decay() requires no
    # re-warmup and the whole run being the anneal
    decay_branch = args.load_from_file_for_decay is not None
    assert not (decay_branch and args.cooldown_checkpoints), \
        "cooldown_checkpoints mark branch points of a stable run; " \
        "a decay branch takes none"
    if decay_branch:
        print("decay branch: forcing warmup_steps=0, decay_frac=1.0")
    warmup_steps = 0 if decay_branch else args.warmup_steps
    decay_frac = 1.0 if decay_branch else args.decay_frac

    def decay_start_tokens(total):
        toks_per_step = (args.micro_batch_size * args.micro_batch_count
                         * model.context_width)
        warmup_toks = warmup_steps * toks_per_step
        return int(total - decay_frac * (total - warmup_toks))

    ckpt_at_tokens = None
    if args.ckpt_at_tokens:
        ckpt_at_tokens = {}
        for kv in args.ckpt_at_tokens:
            name, eq, v = kv.partition("=")
            if not eq or not name:
                raise SystemExit(
                    f"--ckpt_at_tokens expects NAME=TOKENS, got {kv!r}")
            try:
                ckpt_at_tokens[name] = int(v)
            except ValueError:
                raise SystemExit(
                    f"--ckpt_at_tokens: bad token count in {kv!r}")
    if args.cooldown_checkpoints:
        # branch point of budget M*params = total minus the decay tail
        cd = {}
        for m in args.cooldown_checkpoints:
            cd[f"cooldown_{float(m):g}"] = decay_start_tokens(
                m * model.num_params())
        ckpt_at_tokens = {**(ckpt_at_tokens or {}), **cd}
        print("cooldown checkpoints:",
              {k: f"{v:,}" for k, v in cd.items()})
    if args.ckpt_pre_cooldown:
        if args.schedule != "wsd":
            raise SystemExit("--ckpt_pre_cooldown requires --schedule wsd")
        if decay_branch:
            raise SystemExit("--ckpt_pre_cooldown does not apply to a "
                             "decay branch (the whole run is the cooldown)")
        if args.max_steps:
            toks_per_step = (args.micro_batch_size * args.micro_batch_count
                             * model.context_width)
            total = args.max_steps * toks_per_step
        else:
            total = model.num_params() * args.train_token_multiplier
        t = decay_start_tokens(total)
        ckpt_at_tokens = {**(ckpt_at_tokens or {}), "pre_cooldown": t}
        print(f"pre-cooldown checkpoint: pre_cooldown.pt at {t:,} tokens")

    cfg = TrainConfig(
        # data
        bins_dir=args.bins_dir,
        epoch_sampler=args.epoch_sampler,
        out_dir=os.path.join(args.out_dir, model.name, args.run),

        # optimizer
        lr=args.lr, use_muon=args.use_muon, muon_lr_mult=30.0,
        muon_lr=args.muon_lr, grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),

        # schedule
        schedule=args.schedule, min_lr_ratio=0.1,
        warmup_steps=warmup_steps, decay_frac=decay_frac,

        # batching -- constant effective batch, no ramp
        micro_batch=args.micro_batch_size, seq_len=model.context_width,
        accum_start=args.micro_batch_count,
        accum_target=args.micro_batch_count,
        ramp_steps=0,

        # length (plan() prefers max_steps when set)
        max_steps=args.max_steps,
        max_tokens=int(model.num_params() * args.train_token_multiplier),

        # display
        eval_every=args.eval_every, eval_batches=args.eval_batches,
        probe_every=args.probe_every, log_every=10,
        ckpt_every=args.ckpt_every, ckpt_at=tuple(args.ckpt_at or ()),
        ckpt_at_tokens=ckpt_at_tokens,
        ckpt_best=args.ckpt_best, ckpt_final=args.ckpt_final,
        full_val=args.full_val,
        log_file="train.log", progress=args.progress,
        overwrite=args.overwrite,

        # startup state (mutually exclusive; Trainer asserts)
        load_from_file=args.load_from_file,
        load_from_file_for_decay=args.load_from_file_for_decay,
        extend_schedule=args.extend_schedule,

        # runtime
        device=args.device, dtype="bfloat16" if args.device != "cpu"
                else "float32",
        compile=args.device != "cpu", seed=args.seed)

    t = Trainer(model, cfg, muon_param_groups=muon_param_groups,
                MuonAdamW=MuonAdamW)

    try:
        t.train()
    except Paused:
        # resume with --load_from_file <out_dir>/ckpt_paused.pt --overwrite
        print("RUN_PAUSED")
        sys.exit(42)
    print("RUN_DONE")


if __name__ == "__main__":
    main()
