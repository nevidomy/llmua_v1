"""Training loop handler.

Three pieces of machinery worth explaining before the code:

MICROBATCHING. `micro_batch` is what fits in memory; `accum` micro-steps are
summed before one optimizer step. Tokens per optimizer step is
`micro_batch * seq_len * accum`. The loss of each micro-step is divided by
`accum` so gradients average rather than sum -- otherwise the effective LR
would scale with `accum` and the batch ramp below would silently be an LR ramp.

BATCH RAMP. McCandlish et al. 2018 (An Empirical Model of Large-Batch
Training): the gradient noise scale grows during training, so the useful batch
size grows too. Early on, small batches are as informative per token and much
cheaper per step. GPT-3 ramps over the first several billion tokens.
Implemented by growing `accum` from `accum_start` to `accum_target` over
`ramp_steps`, which keeps `micro_batch` fixed at whatever memory allows.
This buys time-to-loss, NOT final loss -- do not expect it to show up in a
val-loss A/B.

TOKEN BUDGET. With a ramp, steps and tokens stop being proportional. Give
`max_tokens` (your D = 20N) and the planner solves for the step count,
accounting for the ramp. `max_steps` also works if you prefer.

Grad clipping happens once per optimizer step, after all micro-steps, on the
accumulated gradient -- not per micro-step.
"""

import json
import math
import os
import signal
import time
from dataclasses import dataclass, field, asdict, replace

import numpy as np
import torch
import torch.nn.functional as F

from model.lr_scheduler import WarmupSchedule
from model.packed_val import (IGNORE_INDEX, collect_docs, pack_doc_rows,
                              pack_pretrain_xy, packed_doc_attention_mask)
from model.params_update_probe import UpdateToParamProbe, print_report
from model.transformer import doc_block_mask


def resolve_bins_dir(bins_dir):
    """`32768` -> `{root}/bins/32768`; an existing path is used as-is."""
    from data.index import get_root
    if not bins_dir:
        bins_dir = "32768"
    bins_dir = os.path.expanduser(os.path.expandvars(str(bins_dir)))
    if os.path.isdir(bins_dir):
        return os.path.abspath(bins_dir)
    under = os.path.join(get_root(), "bins", bins_dir)
    if os.path.isdir(under):
        return os.path.abspath(under)
    return os.path.abspath(bins_dir)


@dataclass
class TrainConfig:
    bins_dir: str = "32768"       # name under {data_root}/bins, or an absolute path
    epoch_sampler: bool = False   # False: random windows on a single train.bin
    out_dir: str = "runs/run"

    # optimizer
    lr: float = 1e-3
    use_muon: bool = True
    muon_lr_mult: float = 30.0
    muon_lr: float = None           # absolute; overrides muon_lr_mult if set
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    ns_steps: int = 5

    # schedule
    schedule: str = "cosine"          # cosine | wsd | constant
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1
    decay_frac: float = 0.2           # wsd only
    decay_shape: str = "linear"       # wsd only

    # batching
    micro_batch: int = 16
    seq_len: int = 256
    accum_target: int = 8
    accum_start: int = 1
    ramp_steps: int = 0               # 0 disables the ramp

    # duration -- give exactly one
    max_steps: int = None
    max_tokens: int = None

    # eval and logging
    eval_every: int = 250
    eval_batches: int = 32        # total val sequences; % (micro * accum) == 0
    eval_seed: int = 74
    probe_every: int = 500
    log_every: int = 20
    progress: bool = False          # tqdm bar on stderr; for pipe-watchers
    ckpt_every: int = 0
    ckpt_at: tuple = ()             # one-off checkpoints at these steps
    ckpt_at_tokens: dict = None     # {name: tokens} -> save NAME.pt once past
    ckpt_final: bool = False
    ckpt_best: bool = False
    full_val: bool = False        # full-split val after the run; stored on ckpt
    log_file: str = None        # None -> stdout; relative paths go under out_dir
    overwrite: bool = False     # allow starting into a non-empty out_dir

    # startup state -- at most one may be set
    load_from_file: str = None            # full resume via load()
    load_from_file_for_decay: str = None  # WSD decay branch via load_for_decay()
    extend_schedule: bool = False         # rebuild LR cycle for THIS plan
                                          # after load_from_file (continue)

    # runtime
    device: str = "cuda"
    dtype: str = "bfloat16"           # bfloat16 | float16 | float32
    compile: bool = False
    seed: int = 47
    eval_only: bool = False           # skip opt/sched/out_dir; evaluate() only

    def __post_init__(self):
        self.bins_dir = resolve_bins_dir(self.bins_dir)


def accum_at(step, cfg):
    """Accumulation count at `step`, linear ramp then flat."""
    if not cfg.ramp_steps or step >= cfg.ramp_steps:
        return cfg.accum_target
    t = step / cfg.ramp_steps
    return max(1, int(round(cfg.accum_start
                            + (cfg.accum_target - cfg.accum_start) * t)))


class Paused(Exception):
    """Raised out of train() after a graceful pause: SIGTERM was caught, the
    step in flight completed, and ckpt_paused.pt holds the full state
    (model/opt/sched/step/tokens/RNG). Resume with load_from_file."""


def plan(cfg):
    """Resolve max_steps <-> max_tokens through the ramp. Returns a dict."""
    per_micro = cfg.micro_batch * cfg.seq_len
    if cfg.max_steps is not None:
        steps = cfg.max_steps
        toks = sum(per_micro * accum_at(s, cfg) for s in range(steps))
    else:
        assert cfg.max_tokens, "give max_steps or max_tokens"
        toks, steps = 0, 0
        while toks < cfg.max_tokens:
            toks += per_micro * accum_at(steps, cfg)
            steps += 1
    ramp_toks = sum(per_micro * accum_at(s, cfg)
                    for s in range(min(cfg.ramp_steps, steps)))
    return {"steps": steps, "tokens": toks,
            "tokens_per_step_final": per_micro * cfg.accum_target,
            "ramp_tokens": ramp_toks}


class Trainer:
    def __init__(self, model, cfg: TrainConfig, tokenizer=None,
                 muon_param_groups=None, MuonAdamW=None):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        if not cfg.eval_only:
            assert not (cfg.load_from_file and cfg.load_from_file_for_decay), (
                "load_from_file and load_from_file_for_decay are mutually exclusive")
            assert not (cfg.extend_schedule and not cfg.load_from_file), (
                "extend_schedule requires load_from_file")
        self.tok = tokenizer
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if cfg.eval_only:
            self._init_eval_only()
            return
        # refuse to clobber a previous run unless explicitly allowed; must
        # precede makedirs and the "w" log open below (which truncates)
        clash = (os.path.isdir(cfg.out_dir) and os.listdir(cfg.out_dir)
                 and not cfg.overwrite)
        assert not clash, (
            f"out_dir {cfg.out_dir} already exists and is not empty -- "
            f"refusing to clobber a previous run. Pass overwrite=True "
            f"(CLI --overwrite) or pick another run name.")
        if os.path.isdir(cfg.out_dir) and os.listdir(cfg.out_dir):
            print(f"WARNING: out_dir {cfg.out_dir} is not empty -- "
                  f"overwrite=True, previous contents may be clobbered")
        os.makedirs(cfg.out_dir, exist_ok=True)

        self._logf = None
        if cfg.log_file:
            lp = (cfg.log_file if os.path.isabs(cfg.log_file)
                  else os.path.join(cfg.out_dir, cfg.log_file))
            self._logf = open(lp, "w", buffering=1)  # line-buffered, tail -f

        self.meta = json.load(open(os.path.join(cfg.bins_dir, "meta.json")))
        assert self.meta["vocab_size"] == model.vocab_size, (
            f"bins are vocab {self.meta['vocab_size']}, model is "
            f"{model.vocab_size} -- WRONG BINS FOR THIS MODEL")
        self.cpt = self.meta["chars_per_token_val"]
        self._init_bins()

        groups = muon_param_groups(
            model, lr=cfg.lr, muon_lr=cfg.muon_lr,
            muon_lr_mult=cfg.muon_lr_mult,
            weight_decay=cfg.weight_decay, use_muon=cfg.use_muon)
        self.opt = MuonAdamW(groups, betas=cfg.betas, ns_steps=cfg.ns_steps)

        self.plan = plan(cfg)
        # fail fast: an out-of-range ckpt_at step would silently never save
        self._ckpt_at = frozenset(cfg.ckpt_at or ())
        bad = sorted(s for s in self._ckpt_at
                     if not 0 <= s < self.plan["steps"])
        assert not bad, (f"ckpt_at steps {bad} fall outside the planned "
                         f"run [0, {self.plan['steps']})")
        # same for token targets: cumulative, so validated against the total
        self._ckpt_tok = dict(cfg.ckpt_at_tokens or {})
        far = {n: t for n, t in self._ckpt_tok.items()
               if not 0 < t <= self.plan["tokens"]}
        assert not far, (f"ckpt_at_tokens targets never reached: {far}; "
                         f"planned total is {self.plan['tokens']:,} tokens")
        self.sched = WarmupSchedule(
            self.opt, total_steps=self.plan["steps"],
            warmup_steps=cfg.warmup_steps, schedule=cfg.schedule,
            min_ratio=cfg.min_lr_ratio, decay_frac=cfg.decay_frac,
            decay_shape=cfg.decay_shape)
        self.probe = UpdateToParamProbe(model)

        self.ctx = (torch.autocast(device_type=cfg.device.split(":")[0],
                                   dtype=getattr(torch, cfg.dtype))
                    if cfg.dtype != "float32" and cfg.device != "cpu"
                    else torch.autocast("cpu", enabled=False))
        if cfg.compile:
            print("Compiling model...")
            self.model = torch.compile(self.model)
            print("Model compiled.")

        self._data = {}
        self.eval_batches = self._make_eval_batches()
        self.eval_block_masks = [self._block_mask(x) for x, _ in self.eval_batches]
        self.step = -1                # last COMPLETED step; -1 = fresh
        self.tokens = 0
        self.history = []

        if cfg.load_from_file:
            self.load(cfg.load_from_file)
        elif cfg.load_from_file_for_decay:
            self.load_for_decay(cfg.load_from_file_for_decay)

    def _init_eval_only(self):
        """Bins + autocast. No optimizer, no scheduler, no writes under
        out_dir. Seed-locked eval batches are built lazily on evaluate();
        full-val never builds them."""
        cfg = self.cfg
        self._logf = None
        self.meta = json.load(open(os.path.join(cfg.bins_dir, "meta.json")))
        assert self.meta["vocab_size"] == self.model.vocab_size, (
            f"bins are vocab {self.meta['vocab_size']}, model is "
            f"{self.model.vocab_size} -- WRONG BINS FOR THIS MODEL")
        self.cpt = self.meta["chars_per_token_val"]
        self._init_bins()
        self.opt = self.plan = self.sched = self.probe = None
        self.ctx = (torch.autocast(device_type=cfg.device.split(":")[0],
                                   dtype=getattr(torch, cfg.dtype))
                    if cfg.dtype != "float32" and cfg.device != "cpu"
                    else torch.autocast("cpu", enabled=False))
        self._data = {}
        # lazy: full-val never uses the seed-locked set, and old ckpts
        # store eval_batches=2 (forward count) which would fail the
        # divisibility assert if we built it here
        self.eval_batches = None
        self.eval_block_masks = None
        self.step = -1
        self.tokens = 0
        self.history = []

    def _eval_forward_B(self):
        """One regular-eval forward is a full training batch."""
        return self.cfg.micro_batch * (self.cfg.accum_target or 1)

    def _ensure_eval_batches(self):
        if self.eval_batches is None:
            self.eval_batches = self._make_eval_batches()
            self.eval_block_masks = [
                self._block_mask(x) for x, _ in self.eval_batches]

    # ---------------------------------------------------------------- data
    def _init_bins(self):
        """Default: random windows over a single `{split}.bin`. EpochSampler
        (sharded train_XXXX.bin) is opt-in via cfg.epoch_sampler."""
        bins = self.cfg.bins_dir
        train_bin = os.path.join(bins, "train.bin")
        val_bin = os.path.join(bins, "val.bin")
        index = os.path.join(bins, "index.json")
        assert os.path.isfile(val_bin), f"missing val split {val_bin}"
        self._train = None
        self._sampler = None
        if not self.cfg.epoch_sampler:
            if not self.cfg.eval_only:
                assert not os.path.isfile(index), (
                    f"{index} found; default sampling is single-file only. "
                    f"Set epoch_sampler=True")
                assert os.path.isfile(train_bin), (
                    f"default sampling needs a single-file {train_bin}; "
                    f"for sharded bins set epoch_sampler=True")
            return
        if self.cfg.eval_only:
            return
        from tokens.sharded_bins import EpochSampler, open_split
        self._train = open_split(bins, "train")
        self._sampler = EpochSampler(
            n_tokens=self._train.n_tokens,
            T=self.cfg.seq_len,
            seed=self.cfg.seed)

    def _bin(self, split):
        # reopened per call: one long-lived memmap accumulates page refs
        return np.memmap(os.path.join(self.cfg.bins_dir, f"{split}.bin"),
                         dtype=np.uint16, mode="r")

    def _batch(self, split, B, rng=None):
        T = self.cfg.seq_len
        if split == "train" and self._sampler is not None and rng is None:
            starts = self._sampler.next_starts(B)
            x, y = self._train.windows(starts, T)
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            return self._to_device(x, y)
        data = self._bin(split)
        rng = rng or np.random
        ix = rng.integers(0, len(data) - T - 1, size=B) if hasattr(rng, "integers") \
            else rng.randint(0, len(data) - T - 1, size=B)
        x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if self.cfg.device != "cpu":
            x = x.pin_memory().to(self.cfg.device, non_blocking=True)
            y = y.pin_memory().to(self.cfg.device, non_blocking=True)
        return x, y

    def _to_device(self, x, y):
        if self.cfg.device != "cpu":
            x = x.pin_memory().to(self.cfg.device, non_blocking=True)
            y = y.pin_memory().to(self.cfg.device, non_blocking=True)
        return x, y

    def _make_eval_batches(self):
        """Fixed val set, built once. `eval_batches` is the total number of
        sequences (not a count of forwards). Drawn in one RNG call then
        sliced into full-batch (micro × count) forwards. Resampling every
        eval adds variance large enough to invent regressions that are
        not there."""
        cfg = self.cfg
        total, B = cfg.eval_batches, self._eval_forward_B()
        assert total > 0, "eval_batches must be > 0"
        assert total % B == 0, (
            f"eval_batches ({total}) must be divisible by "
            f"micro_batch_size * micro_batch_count "
            f"({cfg.micro_batch} * {cfg.accum_target or 1} = {B})")
        g = np.random.default_rng(cfg.eval_seed)
        x, y = self._batch("val", total, g)
        return [(x[i:i + B], y[i:i + B]) for i in range(0, total, B)]

    # ---------------------------------------------------------------- eval
    @torch.no_grad()
    def evaluate(self, progress_cb=None):
        self._ensure_eval_batches()
        self.model.eval()
        try:
            tot = 0.0
            n = len(self.eval_batches)
            for i, ((x, y), bm) in enumerate(
                    zip(self.eval_batches, self.eval_block_masks)):
                with self.ctx:
                    loss = self.model(x, y, block_mask=bm)
                    if isinstance(loss, tuple):
                        loss = loss[1]
                tot += loss.item()
                if progress_cb:
                    progress_cb(i + 1, n)
        finally:
            self.model.train()
        nats = tot / len(self.eval_batches)
        return nats, nats / self.cpt / math.log(2)

    @torch.no_grad()
    def evaluate_full_val(self, progress_cb=None):
        """Packed sequential windows over the whole val split.

        Windows of seq_len, batched at the training macro-batch
        (micro_batch * accum_target). A leftover that doesn't fill a
        complete macro-batch is dropped, so every forward is the same
        shape as training eval.
        """
        cfg = self.cfg
        data = self._bin("val")
        T = cfg.seq_len
        B = cfg.micro_batch * cfg.accum_target
        n_windows = (len(data) - 1) // T
        n_drop = n_windows % B
        n = n_windows - n_drop
        n_macros = n // B
        assert n_macros > 0, (
            f"val split too small for one macro-batch "
            f"(windows={n_windows}, B={B}, T={T})")
        if progress_cb:
            progress_cb(0, n_macros)
        self.model.eval()
        try:
            tot = 0.0
            for i, start in enumerate(range(0, n, B)):
                ix = (start + np.arange(B)) * T
                x = np.stack([data[j:j + T] for j in ix]).astype(np.int64)
                y = np.stack([data[j + 1:j + 1 + T] for j in ix]).astype(np.int64)
                x, y = self._to_device(torch.from_numpy(x), torch.from_numpy(y))
                bm = self._block_mask(x)
                with self.ctx:
                    loss = self.model(x, y, block_mask=bm)
                    if isinstance(loss, tuple):
                        loss = loss[1]
                tot += loss.item()
                if progress_cb:
                    progress_cb(i + 1, n_macros)
        finally:
            self.model.train()
        nats = tot / n_macros
        return nats, nats / self.cpt / math.log(2), {
            "batches": n_macros, "batch": B, "seq_len": T,
            "windows": n, "dropped": n_drop, "tokens": n * T,
        }

    @torch.no_grad()
    def evaluate_full_val_packed(self, progress_cb=None, rule="content"):
        """Full documents only (BOS…EOS), packed like SFT dense batching.

        rule=content: CE on body→body; bpc = mean nats / cpt / ln2.
        rule=uk4b: CE includes BOS→first body; bpc = Σ bits / n_chars_val
        (required). No oversize / no-EOS skips. Documents longer than
        seq_len are skipped only under rule=content. Last macro-batch is
        padded (not dropped) so leftover docs still count.
        """
        cfg = self.cfg
        data = self._bin("val")
        T = cfg.seq_len
        B = cfg.micro_batch * cfg.accum_target
        bos_id = int(getattr(self._raw_model(), "bos_id",
                             self.meta.get("bos_id", 0)) or 0)
        eos_id = self.meta.get("eos_id")
        rule = (rule or "content").strip().lower()
        if rule not in ("content", "uk4b"):
            raise ValueError(f"rule must be content or uk4b, got {rule!r}")
        if eos_id is None:
            raise RuntimeError(
                f"{cfg.bins_dir}/meta.json has no eos_id -- "
                "packed val needs BOS/EOS document boundaries")
        eos_id = int(eos_id)
        docs, n_over, n_no_eos = collect_docs(data, bos_id, eos_id, T)
        if rule == "uk4b":
            if n_over or n_no_eos:
                raise RuntimeError(
                    f"{cfg.bins_dir}: uk4b packed val cannot skip documents "
                    f"(oversize={n_over}, no_eos={n_no_eos}); n_chars_val "
                    f"would keep their characters")
            if not self.meta.get("n_chars_val"):
                raise RuntimeError(
                    f"{cfg.bins_dir}/meta.json needs n_chars_val for rule=uk4b")
        assert docs, (
            f"no BOS…EOS document fits in T={T} "
            f"(skipped oversize={n_over}, no_eos={n_no_eos})")
        rows = pack_doc_rows([len(d) for d in docs], T)
        n_rows = len(rows)
        n_pad = (B - (n_rows % B)) % B
        padded = rows + [None] * n_pad
        n_macros = len(padded) // B
        assert n_macros > 0, "val split empty"
        if progress_cb:
            progress_cb(0, n_macros)
        self.model.eval()
        tot = 0.0
        n_tok = 0
        first_nats = 0.0
        n_first = 0
        try:
            for i in range(n_macros):
                chunk = padded[i * B:(i + 1) * B]
                xs, ys = [], []
                for ix in chunk:
                    if ix is None:
                        x = np.full(T, bos_id, np.int64)
                        y = np.full(T, IGNORE_INDEX, np.int64)
                    else:
                        ids = np.concatenate([docs[j] for j in ix])
                        x, y = pack_pretrain_xy(ids, T, bos_id, eos_id,
                                                pad_id=bos_id, rule=rule)
                    xs.append(x)
                    ys.append(y)
                x, y = self._to_device(torch.from_numpy(np.stack(xs)),
                                       torch.from_numpy(np.stack(ys)))
                bm = self._block_mask(x)
                am = None
                if bm is None:
                    am = packed_doc_attention_mask(x, bos_id)
                with self.ctx:
                    logits = self.model(x, block_mask=bm, attention_mask=am)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    per = F.cross_entropy(
                        logits.float().reshape(-1, logits.size(-1)),
                        y.reshape(-1), ignore_index=IGNORE_INDEX,
                        reduction="none")
                valid = y.reshape(-1) != IGNORE_INDEX
                tot += per[valid].sum().item()
                n_tok += int(valid.sum())
                first = (x.reshape(-1) == bos_id) & valid
                n_first += int(first.sum())
                first_nats += per[first].sum().item()
                if progress_cb:
                    progress_cb(i + 1, n_macros)
        finally:
            self.model.train()
        n_scored = n_tok
        n_docs = len(docs)
        n_tok_docs = int(sum(int(d.size) for d in docs))
        per_doc = 2 if rule == "uk4b" else 3
        expect = sum(max(0, int(d.size) - per_doc) for d in docs)
        assert n_scored == expect, (
            f"n_scored {n_scored} != {n_tok_docs}-{per_doc}·{n_docs} "
            f"(got expect {expect}, rule={rule})")
        if n_over == 0 and n_no_eos == 0:
            nt = self.meta.get("n_tokens_val")
            if nt is not None:
                assert n_tok_docs == int(nt), (
                    f"packed docs {n_tok_docs} tokens != meta n_tokens_val {nt}")
        nats = tot / max(n_scored, 1)
        ln2 = math.log(2)
        bits_sum = tot / ln2
        n_chars = self.meta.get("n_chars_val")
        bpc_cpt = nats / self.cpt / ln2 if self.cpt else None
        first_bits_mean = (
            first_nats / n_first / ln2 if n_first else None)
        if rule == "uk4b":
            bpc = bits_sum / float(n_chars)
            bpc_mode = "chars"
            mode = "packed_uk4b"
        else:
            bpc = bpc_cpt
            bpc_mode = "cpt"
            mode = "packed"
        return nats, bpc, {
            "batches": n_macros, "batch": B, "seq_len": T,
            "windows": n_rows, "dropped": 0, "tokens": n_scored,
            "n_scored": n_scored, "bits_sum": bits_sum,
            "n_chars": n_chars, "bpc_mode": bpc_mode, "bpc_cpt": bpc_cpt,
            "first_nats_sum": first_nats, "n_first": n_first,
            "first_bits_mean": first_bits_mean,
            "docs": n_docs, "skipped_oversize": n_over,
            "skipped_no_eos": n_no_eos, "mode": mode, "rule": rule,
        }

    # --------------------------------------------------------------- train
    def _log(self, msg=""):
        if self._logf is not None:
            self._logf.write(msg + "\n")
            self._logf.flush()
        else:
            print(msg)

    def _lrs(self):
        """(adamw_lr, muon_lr) by flag -- group ORDER depends on use_muon,
        so param_groups[-1] is only AdamW by accident."""
        muon = next((g["lr"] for g in self.opt.param_groups
                     if g.get("use_muon")), None)
        adam = next((g["lr"] for g in self.opt.param_groups
                     if not g.get("use_muon")), None)
        return adam, muon

    def train(self):
        """Resumes after `load()` from `self.step + 1`."""
        cfg, p = self.cfg, self.plan
        # token targets a loaded state already passed were saved pre-crash
        self._ckpt_tok = {n: t for n, t in self._ckpt_tok.items()
                          if t > self.tokens}
        # historical best survives resume, so ckpt_best is never overwritten
        # by a worse post-resume eval
        best_nats = min((h["val_nats"] for h in self.history),
                        default=float("inf"))
        self._log(f"steps {p['steps']:,}  tokens {p['tokens']/1e9:.3f}B  "
                  f"tokens/step (final) {p['tokens_per_step_final']:,}")
        if cfg.ramp_steps:
            self._log(f"  ramp: accum {cfg.accum_start} -> {cfg.accum_target} "
                      f"over {cfg.ramp_steps:,} steps "
                      f"({p['ramp_tokens']/1e6:.0f}M tokens)")
        self._log(f"  vocab {self.meta['vocab_size']:,}  cpt {self.cpt:.4f}  "
                  f"muon {'on' if cfg.use_muon else 'off'}")
        if self.step >= 0:
            left = p["steps"] - self.step - 1
            self._log(f"  RESUME: steps 0..{self.step} done "
                      f"({self.tokens/1e6:.1f}M tokens), "
                      f"{left:,} steps left")
        self._log("")

        steps = range(self.step + 1, p["steps"])
        if cfg.progress:
            # stderr, not stdout: the bar doubles as a machine-readable
            # step stream for main_runner to watch. mininterval keeps the
            # pipe small; the last line is emitted on close().
            from tqdm import tqdm
            steps = tqdm(steps, initial=self.step + 1, total=p["steps"],
                         desc="train", unit="step", mininterval=5,
                         file=__import__("sys").stderr)

        self.model.train()
        t0 = time.time()
        # graceful pause: SIGTERM flips a flag; the loop checkpoints at the
        # next step boundary and raises Paused. (Not restored on the Paused
        # path -- a CLI process exits right after anyway.)
        self._pause = False
        prev_term = signal.signal(signal.SIGTERM,
                                  lambda *_: setattr(self, "_pause", True))
        for step in steps:
            self.step = step
            acc = accum_at(step, cfg)

            self.opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(acc):
                x, y = self._batch("train", cfg.micro_batch)
                bm = self._block_mask(x)
                with self.ctx:
                    out = self.model(x, y, block_mask=bm)
                    loss = out[1] if isinstance(out, tuple) else out
                (loss / acc).backward()          # average, do not sum
                loss_sum += loss.item()
            self.tokens += cfg.micro_batch * cfg.seq_len * acc

            gn = torch.nn.utils.clip_grad_norm_(   # once, on the accumulation
                self.model.parameters(), cfg.grad_clip).item()

            probing = cfg.probe_every and step % cfg.probe_every == 0
            if probing:
                self.probe.snapshot()
            self.opt.step()
            self.sched.step()

            adam_lr, muon_lr = self._lrs()

            if cfg.log_every and step % cfg.log_every == 0:
                tr = loss_sum / acc
                lr_s = f"lr {adam_lr:.2e}"
                if muon_lr is not None:
                    lr_s += f"  muon {muon_lr:.2e}"
                extra = ""
                if self._sampler is not None:
                    extra = (f"  ep {self._sampler.epoch} "
                             f"{self._sampler.cursor}/{self._sampler.n_win}")
                self._log(f"step {step:>6}  loss {tr:7.4f}  "
                          f"bpc {tr/self.cpt/math.log(2):6.4f}  "
                          f"{lr_s}  gn {gn:5.2f}  acc {acc:>3}  "
                          f"tok {self.tokens/1e6:8.1f}M  "
                          f"{time.time()-t0:6.0f}s{extra}")

            if probing:
                self._log("  update-to-param:")
                print_report(self.probe.measure(), file=self._logf)

            if cfg.eval_every and (step % cfg.eval_every == 0
                                   or step == p["steps"] - 1):
                nats, bpc = self.evaluate()
                self.history.append({"step": step, "tokens": self.tokens,
                                     "val_nats": nats, "val_bpc": bpc})
                self._log(f"  EVAL step {step:>6}  val {nats:.4f} nats/tok  "
                          f"{bpc:.4f} bits/char")
                if cfg.ckpt_best and nats < best_nats:
                    best_nats = nats
                    self.save("ckpt_best.pt")

            if ((cfg.ckpt_every and step and step % cfg.ckpt_every == 0)
                    or step in self._ckpt_at):
                self.save(f"ckpt_{step:06d}.pt")

            if self._ckpt_tok:      # first step past each target, once
                for name in [n for n, t in self._ckpt_tok.items()
                             if self.tokens >= t]:
                    self.save(name if name.endswith(".pt") else f"{name}.pt")
                    del self._ckpt_tok[name]

            if self._pause:
                self.save("ckpt_paused.pt")
                self._log(f"  PAUSED after step {step} "
                          f"({self.tokens/1e6:.1f}M tokens) -> ckpt_paused.pt")
                raise Paused(f"paused after step {step}")

        signal.signal(signal.SIGTERM, prev_term)
        if cfg.progress:
            steps.close()

        # final eval (and optional full-split val) BEFORE ckpt_final so
        # the checkpoint carries the numbers
        nats, bpc = self.evaluate()
        self.final = {"val_nats": nats, "val_bpc": bpc}
        if cfg.full_val:
            cb = None
            bar = None
            if cfg.progress:
                from tqdm import tqdm
                bar = tqdm(desc="full_val", unit="batch", mininterval=2,
                           file=__import__("sys").stderr)

                def cb(d, t, _bar=bar):
                    _bar.total = t
                    _bar.n = d
                    _bar.refresh()
            fv_nats, fv_bpc, info = self.evaluate_full_val(progress_cb=cb)
            if bar is not None:
                bar.close()
            self.full_val = {"val_nats": fv_nats, "val_bpc": fv_bpc}
            self._log(f"  FULL_VAL val {fv_nats:.4f} nats/tok  "
                      f"{fv_bpc:.4f} bits/char  "
                      f"({info['windows']} windows, drop {info['dropped']})")
        if cfg.ckpt_final:
            self.save("ckpt_final.pt")

        self._log(f"\ndone: {self.tokens/1e9:.3f}B tokens, "
                  f"val {nats:.4f} nats/tok = {bpc:.4f} bits/char, "
                  f"{time.time()-t0:.0f}s")
        if self._logf is not None:
            self._logf.close()
            self._logf = None
        return {"val_nats": nats, "val_bpc": bpc, "tokens": self.tokens,
                "history": self.history}

    # ---------------------------------------------------------------- ckpt
    def _block_mask(self, x):
        """Document BlockMask for this batch, or None. Built here so
        create_block_mask stays outside torch.compile(self.model)."""
        m = self._raw_model()
        if not getattr(m, "doc_masking", False):
            return None
        return doc_block_mask(x, m.bos_id)

    def _raw_model(self):
        # unwrap torch.compile so checkpoints are portable across
        # compiled/uncompiled contexts. _block_mask uses this too:
        # OptimizedModule.__getattr__ would find bos_id, but the
        # explicit unwrap is the same path save/load already take.
        return getattr(self.model, "_orig_mod", self.model)

    def save(self, name):
        path = os.path.join(self.cfg.out_dir, name)
        cfg = asdict(self.cfg)
        cfg.pop("eval_only", None)       # tool-only flag, not part of a run
        torch.save({"model": self._raw_model().state_dict(),
                    "model_arch": getattr(self._raw_model(), "arch", None),
                    "opt": self.opt.state_dict(),
                    "sched": self.sched.state_dict(),
                    "step": self.step, "tokens": self.tokens,
                    "np_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": (torch.cuda.get_rng_state()
                                 if torch.cuda.is_available() else None),
                    "sampler": (None if self._sampler is None
                                else self._sampler.state_dict()),
                    "final": getattr(self, "final", None),
                    "full_val": getattr(self, "full_val", None),
                    "cfg": cfg, "history": self.history}, path)
        return path

    def _load_weights(self, ck):
        """Model and optimizer state only."""
        sd = ck["model"]
        if any(k.startswith("_orig_mod.") for k in sd):   # compiled checkpoint
            sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        self._raw_model().load_state_dict(sd)
        self.opt.load_state_dict(ck["opt"])
    
    
    def _restore_rng(self, ck):
        """Put the data stream back exactly where the checkpoint left off.
    
        .cpu(): map_location=cfg.device moves the saved rng tensors to GPU, but
        the setters require CPU ByteTensors.
        """
        if ck.get("np_rng") is not None:
            np.random.set_state(ck["np_rng"])
        if ck.get("torch_rng") is not None:
            torch.set_rng_state(ck["torch_rng"].cpu())
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu())

    def _restore_sampler(self, ck):
        if self._sampler is not None and ck.get("sampler") is not None:
            self._sampler.load_state_dict(ck["sampler"])
    
    
    def load(self, path, rebuild_schedule=None):
        """Resume a run: weights, optimizer, step/tokens, RNG, sampler.

        Default restores the saved LR-scheduler state (pause/resume of the
        same plan). With rebuild_schedule (or cfg.extend_schedule) the
        scheduler built in __init__ for THIS plan is kept: last_epoch is
        aligned to the loaded step so a bigger-budget continue inherits
        progress but not the old cycle length.
        """
        if rebuild_schedule is None:
            rebuild_schedule = bool(self.cfg.extend_schedule)
        ck = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self._load_weights(ck)
        self.step, self.tokens = ck["step"], ck["tokens"]
        self.history = ck.get("history", [])
        self._restore_rng(ck)
        self._restore_sampler(ck)
        if rebuild_schedule:
            # after completing step S, sched.step() has advanced last_epoch
            # to S+1 (the LR the next opt.step will use). Replay that.
            self.sched.last_epoch = self.step
            self.sched.step()
            f = self.sched.factor()
            self._log(f"  EXTEND: {self.cfg.schedule} schedule rebuilt for "
                      f"{self.plan['steps']:,} steps "
                      f"(factor {f:.4f} at last_epoch {self.sched.last_epoch})")
        else:
            self.sched.load_state_dict(ck["sched"])
        return ck
    
    
    def load_for_decay(self, path, expect_total=None):
        """Load a stable-phase checkpoint to run THIS trainer's decay schedule.

        Unlike load(), the LR-scheduler state is discarded: this job is a
        WSD anneal (warmup_steps=0, decay_frac=1.0) over the remaining
        budget. Step and token counters stay GLOBAL so tqdm, train.log,
        and a later continue all show tokens/steps learned so far
        -- the same totals cooldown planning already used to place the
        branch. The scheduler itself is still sized to the remainder
        (that is the anneal).

        cfg.max_steps / max_tokens mean the TOTAL budget of the finished
        model (stable phase + this decay): a checkpoint at 10M tokens with
        max_tokens=15M runs a 5M-token anneal; a checkpoint with 1200 steps
        done under max_steps=1400 runs 200 more steps, numbered 1200..1399.

        `expect_total`: redundant with total-budget semantics when max_steps
        is used, but still asserted -- catches a checkpoint taken at the
        wrong branch point, which is otherwise silent.
        """
        cfg = self.cfg
        assert cfg.schedule == "wsd", (
            f"decay job needs schedule='wsd', got {cfg.schedule!r}")
        assert cfg.warmup_steps == 0, (
            f"decay job needs warmup_steps=0, got {cfg.warmup_steps} -- a second "
            f"warmup would re-ramp the LR from zero mid-training")
        assert abs(cfg.decay_frac - 1.0) < 1e-12, (
            f"decay job needs decay_frac=1.0 so the whole short run is the "
            f"anneal, got {cfg.decay_frac}")
        assert not cfg.ramp_steps, "no batch ramp in a decay job"

        ck = torch.load(path, map_location=cfg.device, weights_only=False)

        # the stable run must have used the same base LRs, or this trainer's
        # sched.base_lrs (captured at construction) will not match the weights
        prev = ck.get("cfg") or {}
        for key in ("lr", "muon_lr", "muon_lr_mult", "use_muon", "seq_len",
                    "micro_batch", "accum_target"):
            if key in prev and getattr(cfg, key) != prev[key]:
                raise AssertionError(
                    f"{key} differs from the stable run: {prev[key]!r} -> "
                    f"{getattr(cfg, key)!r}. The decay must continue the same run.")

        done = ck["step"] + 1                      # completed steps in the stable phase

        # the configured budget is the TOTAL; the LR schedule covers only
        # the remainder, but self.plan / self.step stay on the global clock
        if cfg.max_steps is not None:
            extra = cfg.max_steps - done
            assert extra > 0, (f"checkpoint has {done} steps done -- budget "
                               f"max_steps={cfg.max_steps} is already covered")
            rem = plan(replace(cfg, max_steps=extra))
        else:
            extra_toks = cfg.max_tokens - ck["tokens"]
            assert extra_toks > 0, (f"checkpoint has {ck['tokens']:,} tokens -- "
                                    f"budget max_tokens={cfg.max_tokens:,} is "
                                    f"already covered")
            rem = plan(replace(cfg, max_tokens=extra_toks))
        self.plan = {
            "steps": done + rem["steps"],
            "tokens": ck["tokens"] + rem["tokens"],
            "tokens_per_step_final": rem["tokens_per_step_final"],
            "ramp_tokens": 0,
        }
        # rebuild the scheduler for the remainder BEFORE _load_weights
        # overwrites group lrs: base_lrs must come from the configured lrs
        self.sched = WarmupSchedule(
            self.opt, total_steps=rem["steps"],
            warmup_steps=cfg.warmup_steps, schedule=cfg.schedule,
            min_ratio=cfg.min_lr_ratio, decay_frac=cfg.decay_frac,
            decay_shape=cfg.decay_shape)
        # ckpt_at is global (same clock as the stable run)
        bad = sorted(s for s in self._ckpt_at
                     if not 0 <= s < self.plan["steps"])
        assert not bad, (f"ckpt_at steps {bad} fall outside the decay run "
                         f"[0, {self.plan['steps']})")

        if expect_total is not None:
            assert self.plan["steps"] == expect_total, (
                f"branch point mismatch: checkpoint has {done} steps done and "
                f"this decay is {rem['steps']} steps = {self.plan['steps']} "
                f"total, but the budget wants {expect_total}. Recompute with "
                f"branch_points().")

        self._load_weights(ck)
        # NOT self.sched.load_state_dict(ck["sched"]) -- see module docstring
        self.step = ck["step"]                     # keep the global counter
        self.tokens = ck["tokens"]
        self.history = list(ck.get("history", []))
        self._restore_rng(ck)
        self._restore_sampler(ck)

        far = {n: t for n, t in self._ckpt_tok.items()
               if t > self.plan["tokens"]}
        assert not far, (f"ckpt_at_tokens targets never reached: {far}; "
                         f"this decay run ends at {self.plan['tokens']:,} tokens")

        # opt.load_state_dict overwrote param_group['lr'] with the stable run's
        # value. Re-apply THIS schedule's step-0 factor so the first decay step
        # starts where the schedule says, not where the checkpoint left it.
        f0 = self.sched.factor()
        for g, base in zip(self.opt.param_groups, self.sched.base_lrs):
            g["lr"] = base * f0

        last = self.plan["steps"] - 1
        self._log(f"  DECAY BRANCH from {path}")
        self._log(f"    stable steps 0..{ck['step']} done "
                  f"({self.tokens/1e6:.1f}M tokens)")
        self._log(f"    decay {rem['steps']:,} steps "
                  f"(global {done}..{last}), factor {f0:.4f} -> "
                  f"{cfg.min_lr_ratio:.4f}")
        return ck
