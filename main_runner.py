"""Ordered multi-run launcher around model_runner.py.

A "run" (runid) is a named list of RunConfigs, executed in order, each in
its own subprocess (fresh CUDA state per step, one crash doesn't kill the
queue). Progress is checkpointed to .runner_ckpts/{runid}.json after every
completed step: re-running this file skips finished steps, and a step whose
config changed since the checkpoint is re-run. A failed step is NOT marked
done, so the next launch resumes at it.

    python main_runner.py --list
    python main_runner.py NAME [NAME ...]

NAME is a run set from experiments/appendix_c.py (the sweeps of Appendix C).

Call reset_checkpoint() before main_runner().run(...) to wipe a run's
checkpoint file (if present) and start it from step 0.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, replace

from tqdm import tqdm

from data.index import ROOT

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, ".runner_ckpts")

_RESET = False
MODEL_DIRS = {"test": "nano_llmua", "target": "llmua"}


def reset_checkpoint():
    """Global flag: every run launched after this call deletes its
    checkpoint file first (if present) and starts from step 0."""
    global _RESET
    _RESET = True


# --------------------------------------------------------------- config
@dataclass
class RunConfig:
    """One model_runner invocation. Defaults mirror model_runner's CLI;
    None means 'use the CLI default'."""
    run: str                     # run name + out_dir subdir
    model: str = "test"          # test | P1 | P2 | P3 | P4 | target
    seed: int = 47               # sets model init AND data order
    bins_dir: str = "32768"
    epoch_sampler: bool = False
    out_dir: str = None
    lr: float = 1e-3
    use_muon: bool = False
    muon_lr: float = None          # absolute Muon lr; None = lr * 30
    schedule: str = "cosine"
    warmup_steps: int = 200      # LR warmup steps
    decay_frac: float = 0.2      # fraction of the run spent in LR decay
    micro_batch_size: int = 16
    micro_batch_count: int = 1   # micro-steps per optimizer step (no ramp)
    train_token_multiplier: float = 20.0
    max_steps: int = None        # fixed steps; overrides token multiplier
    eval_every: int = 100        # val eval cadence in steps; 0 = disabled
    eval_batches: int = 32       # total val sequences; % (micro * count) == 0
    probe_every: int = 500       # update-to-param probe cadence; 0 = disabled
    ckpt_every: int = 0          # 0 = disabled
    ckpt_at: tuple = ()          # one-off checkpoints at these steps
    ckpt_at_tokens: dict = None  # {name: tokens} one-off checkpoints
    cooldown_checkpoints: tuple = None  # token multipliers -> cooldown_M.pt
    ckpt_best: bool = False
    ckpt_final: bool = False     # save ckpt_final.pt at end of training
    full_val: bool = False       # full-split val after the run; on the ckpt
    overwrite: bool = False      # allow starting into a non-empty out_dir
    load_from_file: str = None            # full resume via load()
    load_from_file_for_decay: str = None  # weights-only WSD decay branch
    progress: bool = True        # child tqdm stream -> our progress bar
    enable_init: bool = False
    enable_swiglu: bool = False
    enable_rms_norm: bool = False
    enable_qk_norm: bool = False
    doc_masking: bool = False
    doc_rope: bool = False
    # fine-grained init breakdown; enable_init implies all three
    no_biases: bool = False
    custom_init: bool = False
    residual_init: bool = False

    def to_args(self):
        args = []
        for k, v in asdict(self).items():
            if v is None:
                continue
            if isinstance(v, bool):
                if v:
                    args.append(f"--{k}")
            elif isinstance(v, (list, tuple)):
                if v:
                    args.append(f"--{k}")
                    args += [str(x) for x in v]
            elif isinstance(v, dict):
                if v:
                    args.append(f"--{k}")
                    args += [f"{kk}={vv}" for kk, vv in v.items()]
            else:
                args += [f"--{k}", str(v)]
        return args

    def key(self):
        """Config fingerprint; detects edits between launcher invocations.
        ANY change to RunConfig -- a value, a new field, a changed default
        -- invalidates every stored step. `progress` is a display knob,
        not an experiment parameter."""
        d = asdict(self)
        d.pop("progress", None)
        blob = json.dumps(d, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


# --------------------------------------------------------------- runner
_ITER = re.compile(r"(\d+)/(\d+)")   # tqdm stderr lines: "train:  5%| 100/2000"


def _run_watched(cmd, tag, stderr_log):
    """Run model_runner, folding the child's tqdm stderr stream into one
    bar of our own. Child tqdm emits `step/total` (with its own ETA/speed);
    we mirror position and let our bar recompute ETA across step boundary.

    Child stdout is dropped -- the training loop already writes its log to
    out_dir/train.log. Non-tqdm stderr (warnings, tracebacks) passes
    through live AND is appended to `stderr_log` -- the bar redraw can eat
    lines on screen, the file keeps them. Returns the child's exit code."""
    with tqdm(total=0, desc=tag, unit="step", bar_format=
              "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
              "[{elapsed}<{remaining}, {rate_fmt}{postfix}]") as bar, \
            open(stderr_log, "a") as slog:
        slog.write(f"\n===== {tag} =====\n")
        proc = subprocess.Popen(
            cmd, cwd=HERE, env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.DEVNULL,      # log file has it all
            stderr=subprocess.PIPE, text=True)
        for line in proc.stderr:
            m = _ITER.search(line)
            if m:
                i, tot = int(m.group(1)), int(m.group(2))
                if tot != bar.total:
                    bar.reset(total=tot)
                bar.update(i - bar.n)
            elif line.strip():
                slog.write(line)
                slog.flush()
                tqdm.write(line.rstrip(), file=sys.stderr)
        proc.wait()
    return proc.returncode


def finished(cfg):
    """True when this run's folder already holds a completed run. Several
    sets repeat a run (a seed-47 run reused from a sweep), and model_runner
    refuses a non-empty out_dir."""
    d = os.path.join(cfg.out_dir or f"{ROOT}/runs",
                     MODEL_DIRS.get(cfg.model, f"llmua_{cfg.model}"), cfg.run)
    if os.path.exists(os.path.join(d, "ckpt_final.pt")):
        return True
    log = os.path.join(d, "train.log")
    if not os.path.exists(log):
        return False
    with open(log, encoding="utf-8", errors="replace") as f:
        return any(line.startswith("done:") for line in f)


class MainRunner:
    def __init__(self, reset=False):
        self.reset = reset
        os.makedirs(CKPT_DIR, exist_ok=True)

    @staticmethod
    def _ckpt_path(runid):
        return os.path.join(CKPT_DIR, f"{runid}.json")

    def run(self, runid, configs):
        path = self._ckpt_path(runid)
        if self.reset and os.path.exists(path):
            os.remove(path)
            print(f"[{runid}] checkpoint reset")
        done = json.load(open(path)) if os.path.exists(path) else {}

        n = len(configs)
        for i, cfg in enumerate(configs):
            tag = f"[{runid}] step {i + 1}/{n} ({cfg.run})"
            if done.get(str(i)) == cfg.key():
                print(f"{tag} already done, skipping")
                continue
            if finished(cfg):
                print(f"{tag} finished under another set, skipping")
                continue
            if str(i) in done:
                print(f"{tag} config changed since checkpoint, re-running")
            if not cfg.progress:
                cfg = replace(cfg, progress=True)   # we always watch
            cmd = [sys.executable,
                   os.path.join(HERE, "model_runner.py"), *cfg.to_args()]
            rc = _run_watched(cmd, tag,
                              stderr_log=os.path.join(
                                  CKPT_DIR, f"{runid}.stderr.log"))
            if rc != 0:
                raise RuntimeError(
                    f"{tag} failed with exit code {rc}; fix and re-run "
                    f"-- it is NOT marked done in the checkpoint")
            done[str(i)] = cfg.key()
            with open(path, "w") as f:
                json.dump(done, f, indent=1)
        print(f"[{runid}] all {n} steps complete")


def main_runner():
    return MainRunner(reset=_RESET)


# -------------------------------------------------------- run definitions
def run_test():
    """Quick end-to-end check of the launcher: ~100-200 steps per config,
    a couple of minutes each on a desktop GPU. Re-run fresh with
    reset_checkpoint()."""
    return "test", [
        RunConfig(run="smoke_base", train_token_multiplier=0.05),
        RunConfig(run="smoke_swiglu", train_token_multiplier=0.05,
                  enable_swiglu=True),
    ]

def p1_best_sweep(learning_rates=[4e-3], muon_lrs=[0.01], seeds=[47]):
    return f"P1_best_sweep", [
         RunConfig(
            run=f"P1_best_lr_{learning_rate}_muon_lr_{muon_lr}_seed_{seed}", 
            model="P1", seed=seed,
            schedule="wsd", lr=learning_rate, muon_lr=muon_lr,
            enable_init=False, 
            enable_swiglu=True, enable_rms_norm=True, enable_qk_norm=True, use_muon=True,
            no_biases=True, custom_init=True, residual_init=False,
            ckpt_final=True)
        for seed in seeds
        for learning_rate in learning_rates
        for muon_lr in muon_lrs
    ]

def cooldown_branching(model, seed, learning_rate, muon_lr,
                       points=(20, 40, 80), skip_intial_run=False,
                       add_suffix="", adamw_only=False):
    """1 full-length stable run + one WSD decay branch per earlier point.

    The full run uses multiplier max(points) so its own decay starts exactly
    at cooldown_<max> (it doubles as the largest-branch decay run) and it
    snapshots every branch point via cooldown_checkpoints. Each earlier point
    spawns a decay run that loads cooldown_<m>.pt and anneals for the rest.
    Works for any tuple of points, e.g. (10, 20, 40, 80) -> 4 runs.
    """
    assert points, "need at least one branch point"
    pts = tuple(sorted(points))
    top = pts[-1]
    tag = "_adamw_only" if adamw_only else ""
    base = f"{model}_cooldown_branching{tag}"
    full_name = f"{base}_{top}x_seed_{seed}{add_suffix}"
    shared = dict(model=model, seed=seed, schedule="wsd",
                  lr=learning_rate, muon_lr=muon_lr,
                  enable_init=False, enable_swiglu=True, enable_rms_norm=True,
                  enable_qk_norm=True, use_muon=(not adamw_only),
                  no_biases=True, custom_init=True, residual_init=False,
                  ckpt_final=True)
    full_dir = f"{ROOT}/runs/llmua_{model}/{full_name}"
    runs = ([] if skip_intial_run else [
        RunConfig(run=full_name, train_token_multiplier=float(top),
                  cooldown_checkpoints=pts, **shared)])
    runs += [
        RunConfig(run=f"{base}_{m}x_seed_{seed}{add_suffix}",
                  train_token_multiplier=float(m),
                  load_from_file_for_decay=f"{full_dir}/cooldown_{float(m):g}.pt",
                  **shared)
        for m in pts[:-1]]
    return f"{base}{add_suffix}", runs

def lr_sweep(model, seeds=[47], learning_rates=[4e-3], muon_lrs=[0.01], residual_init=False):
    return f"{model}_lr_sweep", [
        RunConfig(
            run=f"{model}_LR_sweep_lr_{learning_rate}_muon_lr_{muon_lr}_seed{"_resid_on" if residual_init else ""}_{seed}", 
            model=model, seed=seed,
            schedule="wsd", lr=learning_rate, muon_lr=muon_lr,
            enable_init=False, 
            enable_swiglu=True, enable_rms_norm=True, enable_qk_norm=True, use_muon=True,
            no_biases=True, custom_init=True, residual_init=residual_init,
            ckpt_final=True)
        for seed in seeds
        for learning_rate in learning_rates
        for muon_lr in muon_lrs
    ]

def adamw_only(model, seeds=[47], learning_rates=[4e-3], residual_init=False):
    return f"{model}_lr_sweep", [
        RunConfig(
            run=f"{model}_adam_only_lr_{learning_rate}_seed{"_resid_on" if residual_init else ""}_{seed}", 
            model=model, seed=seed,
            schedule="wsd", lr=learning_rate,
            enable_init=False, 
            enable_swiglu=True, enable_rms_norm=True, enable_qk_norm=True, use_muon=False,
            no_biases=True, custom_init=True, residual_init=residual_init,
            ckpt_final=True)
        for seed in seeds
        for learning_rate in learning_rates
    ]

def no_qk_norm(model, seeds=[47], learning_rate=4e-3, muon_lr=0.01):
    return f"{model}_no_qk_norm_sweep", [
        RunConfig(
            run=f"{model}_no_qk_norm_lr_{learning_rate}_s{seed}", 
            model=model, seed=seed,
            schedule="wsd", lr=learning_rate, muon_lr=muon_lr,
            enable_init=False, 
            enable_swiglu=True, enable_rms_norm=True, enable_qk_norm=False, use_muon=True,
            no_biases=True, custom_init=True, residual_init=False,
            ckpt_final=True)
        for seed in seeds
    ]

def run_sets():
    """Name -> function for every run set of experiments/appendix_c.py."""
    # when run as a script this module is "__main__"; appendix_c imports it
    # by name, so point that name at the running module (one RunConfig class)
    sys.modules.setdefault("main_runner", sys.modules[__name__])
    from experiments.appendix_c import RUN_SETS
    return RUN_SETS


if __name__ == "__main__":
    sets = run_sets()
    names = sys.argv[1:]
    if not names or names == ["--list"]:
        for name, fn in sets.items():
            doc = (fn.__doc__ or "").strip().splitlines()
            print(f"{name:34s} {doc[0] if doc else ''}")
        sys.exit(0 if names else 2)
    unknown = [n for n in names if n not in sets]
    if unknown:
        sys.exit(f"unknown run set(s): {', '.join(unknown)} "
                 f"(see python main_runner.py --list)")
    for name in names:
        main_runner().run(*sets[name]())
