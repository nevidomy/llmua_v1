"""SFT training loop: example-aligned batches, masked answer loss.

Separate from model/training_loop.py -- pretrain still draws random token
windows. This loop slices whole Q/A examples (offsets/lengths), optionally
packs them densely with BOS-delimited attention, and backprops CE only on
{answer}<END>.
"""
import json
import math
import os
import signal
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from model.lr_scheduler import WarmupSchedule, resolve_warmup
from model.packed_val import packed_doc_attention_mask
from model.params_update_probe import UpdateToParamProbe, print_report
from model.training_loop import Paused
from model.transformer import doc_block_mask
from sft.batcher import (IGNORE_INDEX, LOWDIFF_THRESHOLD, SFTEpochBatcher,
                         SFTSplit, loss_targets, pack_rows, pack_xy,
                         resolve_data_dir, response_p0_mask, supervised_mask)


@dataclass
class SFTConfig:
    kind: str = "sft"
    data_dir: str = ""
    epoch_sampler: bool = True          # shuffle example order each epoch
    dense_batching: bool = False
    out_dir: str = "runs/sft"

    # optimizer -- `lr` / `muon_lr` are PEAKS (original × lr_ratio)
    lr: float = 1e-4
    lr_ratio: float = 0.1
    orig_lr: float = None
    orig_muon_lr: float = None
    use_muon: bool = False
    muon_lr_mult: float = 30.0
    muon_lr: float = None
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    ns_steps: int = 5

    # schedule: warmup starts at 0, decay ends at 0
    schedule: str = "cosine"
    warmup: float = 0.05              # fraction if < 1, else step count
    warmup_steps: int = 0             # resolved from warmup × total_steps
    min_lr_ratio: float = 0.0
    decay_frac: float = 0.2
    decay_shape: str = "linear"

    # batching
    micro_batch: int = 16
    seq_len: int = 256
    accum_target: int = 1
    epochs: int = 1

    # specials (defaults match sft/tokenize.py)
    bos_id: int = 0
    usr_id: int = 201
    assistant_id: int = 202
    end_id: int = 203

    # eval / ckpt
    eval_every: int = 100
    eval_batches: int = 32
    eval_seed: int = 74
    probe_every: int = 500
    log_every: int = 20
    progress: bool = False
    ckpt_every: int = 0
    ckpt_at: tuple = ()
    ckpt_epochs: tuple = ()           # snapshot after completing these epochs
    ckpt_final: bool = False
    ckpt_best: bool = False
    full_val: bool = False
    deep_probe: bool = False          # snapshot train/val windows to JSONL
    deep_probe_every: int = 1         # keep every Nth window (1 = all)
    lowdiff_masking: bool = False     # drop answer tokens copied from prompt
    lowdiff_threshold: float = LOWDIFF_THRESHOLD
    p0_loss_multiplier: float = 1.0   # scale first-answer-token CE in train sum
    log_file: str = None
    overwrite: bool = False

    load_from_file: str = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False
    seed: int = 47
    eval_only: bool = False

    def __post_init__(self):
        if self.data_dir:
            self.data_dir = resolve_data_dir(self.data_dir)
        self.ckpt_at = tuple(self.ckpt_at or ())
        self.ckpt_epochs = tuple(int(x) for x in (self.ckpt_epochs or ()))
        m = float(self.p0_loss_multiplier)
        assert math.isfinite(m) and m >= 0, (
            f"p0_loss_multiplier must be finite and >= 0, got {self.p0_loss_multiplier}")
        self.p0_loss_multiplier = m
        t = float(self.lowdiff_threshold)
        assert math.isfinite(t) and 0.0 < t <= 1.0, (
            f"lowdiff_threshold must be in (0, 1], got {self.lowdiff_threshold}")
        self.lowdiff_threshold = t
        n = int(self.deep_probe_every)
        assert n >= 1, f"deep_probe_every must be >= 1, got {self.deep_probe_every}"
        self.deep_probe_every = n


def sft_eval_trainer(model, stored_cfg, device, data_dir=None,
                     dense_batching=None):
    """eval_only SFTTrainer from a checkpoint cfg (reeval / full_val tools).

    `data_dir` overrides the run's split (custom-dataset full val). A
    pretrain checkpoint has no SFT specials; those come from the
    dataset meta. `dense_batching` overrides packing when the caller
    picked a strategy (one-per-row vs pack).
    """
    fields = SFTConfig.__dataclass_fields__
    kw = {k: v for k, v in stored_cfg.items() if k in fields}
    if data_dir:
        kw["data_dir"] = data_dir
        meta_path = os.path.join(resolve_data_dir(data_dir), "meta.json")
        if os.path.isfile(meta_path):
            meta = json.load(open(meta_path))
            for k in ("bos_id", "usr_id", "assistant_id", "end_id"):
                if meta.get(k) is not None:
                    kw[k] = meta[k]
        if stored_cfg.get("kind") != "sft":
            kw["lowdiff_masking"] = False
    if dense_batching is not None:
        kw["dense_batching"] = bool(dense_batching)
    kw.update(eval_only=True, log_file=None, progress=False,
              load_from_file=None, compile=False, device=device)
    kw.pop("overwrite", None)
    return SFTTrainer(model, SFTConfig(**kw))


def answer_tokens_per_step(split, cfg, plan, meta):
    """Fixed denominator for token-weighted accum.

    Sum-CE / this, not mean-CE / accum: a micro-batch with more answer
    tokens must contribute more gradient. The constant is absorbed by the
    LR; it must not change per micro-batch.
    """
    n_ans = meta.get("n_answer_tokens_train") if meta else None
    if getattr(cfg, "lowdiff_masking", False):
        n_ans = None
    if n_ans is None:
        n_ans = 0
        for i in range(split.n):
            n_ans += int(loss_targets(
                split.example(i), cfg.assistant_id, cfg.bos_id,
                usr_id=getattr(cfg, "usr_id", 201),
                end_id=getattr(cfg, "end_id", 203),
                lowdiff_masking=getattr(cfg, "lowdiff_masking", False),
                lowdiff_threshold=getattr(cfg, "lowdiff_threshold",
                                         LOWDIFF_THRESHOLD)).sum())
    steps = max(int(plan["steps"]), 1)
    return max(1.0, float(n_ans) * cfg.epochs / steps)


def resolve_sft_cpt(split, cfg, meta):
    """Chars per *loss* token: source chars in {answer} / ({answer}+END).

    SFT CE is only on those tokens. Using 1.0 (the old fallback) makes
    reported bpc = nats / ln(2) — bits per token, not per character —
    which is why bpc looked higher than nats.
    """
    raw = (meta or {}).get("chars_per_token_val")
    if raw:
        return float(raw)
    from sft.probe import load_decoder
    dec = load_decoder(cfg.data_dir, cfg.bos_id, cfg.usr_id,
                       cfg.assistant_id, cfg.end_id)
    if dec._hf is None:
        return 1.0
    n_ch = n_tok = 0
    skip = {int(cfg.bos_id), int(cfg.usr_id), int(cfg.assistant_id),
            int(cfg.end_id)}
    for i in range(split.n):
        ids = split.example(i)
        sup = supervised_mask(ids, cfg.assistant_id, cfg.bos_id)
        n_tok += int(sup.sum())
        for j in np.flatnonzero(sup):
            tid = int(ids[j])
            if tid in skip:
                continue
            n_ch += len(dec.piece(tid))
    return max(float(n_ch), 1.0) / max(n_tok, 1)


def _strip_orig_mod(sd):
    if any(k.startswith("_orig_mod.") for k in sd):
        return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    return sd


class SFTTrainer:
    def __init__(self, model, cfg: SFTConfig, muon_param_groups=None,
                 MuonAdamW=None):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        m = self._raw_model()
        m.bos_id = cfg.bos_id
        if getattr(m, "arch", None) is not None:
            m.arch["bos_id"] = cfg.bos_id
        if cfg.dense_batching and not cfg.eval_only:
            m.doc_masking = True
            if getattr(m, "arch", None) is not None:
                m.arch["doc_masking"] = True

        if cfg.eval_only:
            self._init_eval_only()
            return

        clash = (os.path.isdir(cfg.out_dir) and os.listdir(cfg.out_dir)
                 and not cfg.overwrite)
        assert not clash, (
            f"out_dir {cfg.out_dir} already exists and is not empty -- "
            f"refusing to clobber. Pass overwrite=True.")
        if os.path.isdir(cfg.out_dir) and os.listdir(cfg.out_dir):
            print(f"WARNING: out_dir {cfg.out_dir} is not empty -- "
                  f"overwrite=True, previous contents may be clobbered")
        os.makedirs(cfg.out_dir, exist_ok=True)

        self._logf = None
        if cfg.log_file:
            lp = (cfg.log_file if os.path.isabs(cfg.log_file)
                  else os.path.join(cfg.out_dir, cfg.log_file))
            self._logf = open(lp, "w", buffering=1)

        self._load_meta()
        self._init_data()
        self.cpt = resolve_sft_cpt(self._val, cfg, self.meta)
        resume_ck = self._maybe_scale_lrs(muon_param_groups is not None)

        groups = muon_param_groups(
            self._raw_model(), lr=cfg.lr, muon_lr=cfg.muon_lr,
            muon_lr_mult=cfg.muon_lr_mult,
            weight_decay=cfg.weight_decay, use_muon=cfg.use_muon)
        self.opt = MuonAdamW(groups, betas=cfg.betas, ns_steps=cfg.ns_steps)

        self.plan = {"steps": self._train_batcher.total_steps,
                     "tokens": None, "tokens_per_step_final":
                     cfg.micro_batch * cfg.seq_len * cfg.accum_target,
                     "ramp_tokens": 0, "epochs": cfg.epochs,
                     "steps_per_epoch": self._train_batcher.steps_per_epoch}
        self._tok_scale = answer_tokens_per_step(
            self._train_batcher.split, cfg, self.plan, self.meta)
        cfg.warmup_steps = resolve_warmup(cfg.warmup, self.plan["steps"])
        self._ckpt_at = frozenset(cfg.ckpt_at)
        bad = sorted(s for s in self._ckpt_at
                     if not 0 <= s < self.plan["steps"])
        assert not bad, (f"ckpt_at steps {bad} fall outside the planned "
                         f"run [0, {self.plan['steps']})")
        bad_ep = sorted(e for e in cfg.ckpt_epochs
                        if not 1 <= e <= cfg.epochs)
        assert not bad_ep, (
            f"ckpt_epochs {bad_ep} fall outside 1..{cfg.epochs}")

        self.sched = WarmupSchedule(
            self.opt, total_steps=self.plan["steps"],
            warmup_steps=cfg.warmup_steps, schedule=cfg.schedule,
            min_ratio=cfg.min_lr_ratio, decay_frac=cfg.decay_frac,
            decay_shape=cfg.decay_shape, warmup_from_zero=True)
        self.probe = UpdateToParamProbe(self._raw_model())

        self.ctx = (torch.autocast(device_type=cfg.device.split(":")[0],
                                   dtype=getattr(torch, cfg.dtype))
                    if cfg.dtype != "float32" and cfg.device != "cpu"
                    else torch.autocast("cpu", enabled=False))
        if cfg.compile:
            print("Compiling model...")
            self.model = torch.compile(self.model)
            print("Model compiled.")

        self.eval_batches = self._make_eval_batches()
        self.eval_block_masks = [self._block_mask(x) for x, _ in self.eval_batches]
        self.step = -1
        self.tokens = 0
        self.history = []
        self._sampler = self._train_batcher
        self._probe = None

        if resume_ck is not None:
            self._resume(resume_ck)

        if cfg.deep_probe:
            from sft.probe import ProbeLog, load_decoder
            dec = load_decoder(cfg.data_dir, cfg.bos_id, cfg.usr_id,
                               cfg.assistant_id, cfg.end_id)
            self._probe = ProbeLog(cfg.out_dir, dec,
                                   append=resume_ck is not None)
            ev = int(cfg.deep_probe_every)
            self._log(f"  deep_probe {self._probe.jsonl}"
                      + (f"  every {ev}" if ev > 1 else ""))

    def _probe_keep(self, idx):
        """True if this window (0-based) is kept under deep_probe_every."""
        n = int(self.cfg.deep_probe_every)
        return n <= 1 or (int(idx) % n == 0)

    def _init_eval_only(self):
        cfg = self.cfg
        self._logf = None
        self._load_meta()
        self._val = SFTSplit(cfg.data_dir, "val")
        self.cpt = resolve_sft_cpt(self._val, cfg, self.meta)
        self._train_batcher = None
        self.opt = self.plan = self.sched = self.probe = None
        self.ctx = (torch.autocast(device_type=cfg.device.split(":")[0],
                                   dtype=getattr(torch, cfg.dtype))
                    if cfg.dtype != "float32" and cfg.device != "cpu"
                    else torch.autocast("cpu", enabled=False))
        self.eval_batches = None
        self.eval_block_masks = None
        self.step = -1
        self.tokens = 0
        self.history = []
        self._sampler = None
        self._probe = None

    def _load_meta(self):
        meta_path = os.path.join(self.cfg.data_dir, "meta.json")
        self.meta = json.load(open(meta_path)) if os.path.isfile(meta_path) \
            else {}
        vs = self.meta.get("vocab_size")
        if vs is not None:
            assert vs == self._raw_model().vocab_size, (
                f"bins are vocab {vs}, model is "
                f"{self._raw_model().vocab_size} -- WRONG BINS")
        # cpt is resolved after the val split is loaded (may decode).

    def _init_data(self):
        cfg = self.cfg
        train = SFTSplit(cfg.data_dir, "train")
        self._val = SFTSplit(cfg.data_dir, "val")
        self._train_batcher = SFTEpochBatcher(
            train, T=cfg.seq_len, B=cfg.micro_batch, accum=cfg.accum_target,
            dense=cfg.dense_batching, seed=cfg.seed,
            assistant_id=cfg.assistant_id, bos_id=cfg.bos_id,
            shuffle=cfg.epoch_sampler, pad_id=cfg.bos_id,
            epochs=cfg.epochs, usr_id=cfg.usr_id, end_id=cfg.end_id,
            lowdiff_masking=cfg.lowdiff_masking,
            lowdiff_threshold=cfg.lowdiff_threshold)

    def _pack_xy(self, ids, T=None):
        cfg = self.cfg
        return pack_xy(ids, T if T is not None else cfg.seq_len,
                       cfg.assistant_id, cfg.bos_id, pad_id=cfg.bos_id,
                       usr_id=cfg.usr_id, end_id=cfg.end_id,
                       lowdiff_masking=cfg.lowdiff_masking,
                       lowdiff_threshold=cfg.lowdiff_threshold)

    def _maybe_scale_lrs(self, _has_opt_ctor):
        """If load_from_file is a pretrained ckpt, set peak LRs from it.
        If it is an SFT ckpt, return it for resume. None = scratch."""
        path = self.cfg.load_from_file
        if not path:
            orig = self.cfg.orig_lr if self.cfg.orig_lr is not None else 1e-3
            self.cfg.orig_lr = orig
            self.cfg.lr = orig * self.cfg.lr_ratio
            if self.cfg.use_muon:
                orig_m = self.cfg.orig_muon_lr
                if orig_m is None:
                    orig_m = orig * self.cfg.muon_lr_mult
                    self.cfg.orig_muon_lr = orig_m
                self.cfg.muon_lr = orig_m * self.cfg.lr_ratio
            return None
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if (ck.get("cfg") or {}).get("kind") == "sft":
            prev = ck["cfg"]
            self.cfg.lr = prev.get("lr", self.cfg.lr)
            self.cfg.muon_lr = prev.get("muon_lr", self.cfg.muon_lr)
            self.cfg.orig_lr = prev.get("orig_lr", self.cfg.orig_lr)
            self.cfg.orig_muon_lr = prev.get("orig_muon_lr", self.cfg.orig_muon_lr)
            return ck
        prev = ck.get("cfg") or {}
        orig_lr = prev.get("lr")
        if orig_lr is None:
            orig_lr = self.cfg.orig_lr or 1e-3
        orig_muon = prev.get("muon_lr")
        if orig_muon is None and (prev.get("use_muon") or self.cfg.use_muon):
            orig_muon = orig_lr * (prev.get("muon_lr_mult") or 30.0)
        self.cfg.orig_lr = orig_lr
        self.cfg.orig_muon_lr = orig_muon
        self.cfg.lr = orig_lr * self.cfg.lr_ratio
        self.cfg.muon_lr = (orig_muon * self.cfg.lr_ratio
                            if orig_muon is not None else None)
        sd = _strip_orig_mod(ck["model"])
        self._raw_model().load_state_dict(sd)
        return None

    def _resume(self, ck):
        sd = _strip_orig_mod(ck["model"])
        self._raw_model().load_state_dict(sd)
        self.opt.load_state_dict(ck["opt"])
        self.sched.load_state_dict(ck["sched"])
        self.step, self.tokens = ck["step"], ck["tokens"]
        self.history = ck.get("history", [])
        if ck.get("np_rng") is not None:
            np.random.set_state(ck["np_rng"])
        if ck.get("torch_rng") is not None:
            torch.set_rng_state(ck["torch_rng"].cpu())
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        if ck.get("sampler") is not None:
            self._train_batcher.load_state_dict(ck["sampler"])

    def _raw_model(self):
        return getattr(self.model, "_orig_mod", self.model)

    def _block_mask(self, x):
        m = self._raw_model()
        if not getattr(m, "doc_masking", False):
            return None
        return doc_block_mask(x, m.bos_id)

    def _to_device(self, x, y):
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        y = torch.from_numpy(y) if isinstance(y, np.ndarray) else y
        if self.cfg.device != "cpu":
            x = x.pin_memory().to(self.cfg.device, non_blocking=True)
            y = y.pin_memory().to(self.cfg.device, non_blocking=True)
        else:
            x, y = x.to(self.cfg.device), y.to(self.cfg.device)
        return x, y

    def _ce_sum(self, logits, y):
        """Sum CE over non-ignored (answer) positions. 0 if a pad micro-batch."""
        return F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            y.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum")

    def _ce(self, logits, y):
        """Mean CE over non-ignored positions; 0 if a pad micro-batch."""
        n_valid = (y != IGNORE_INDEX).sum()
        if n_valid == 0:
            return logits.sum() * 0.0
        return self._ce_sum(logits, y) / n_valid

    def _p0_sum(self, logits, x, y):
        """Sum CE and count at first-answer-token positions (unweighted)."""
        mask = response_p0_mask(x, y, self.cfg.assistant_id)
        n = int(mask.sum())
        if n == 0:
            return 0.0, 0
        per = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            y.reshape(-1), ignore_index=IGNORE_INDEX, reduction="none")
        return float(per[mask.reshape(-1)].sum().item()), n

    def _ce_train(self, logits, x, y):
        """Per-token CE, unweighted sum, p0 sum, and the train (possibly
        p0-reweighted) sum used for backward.

        Logged train_loss / train_p0_loss stay unweighted. The multiplier
        only scales first-answer-token CE in the backward sum.
        """
        per = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            y.reshape(-1), ignore_index=IGNORE_INDEX, reduction="none")
        ce = per.sum()
        p0m = response_p0_mask(x, y, self.cfg.assistant_id).reshape(-1)
        p0_ce = per[p0m].sum()
        m = self.cfg.p0_loss_multiplier
        ce_w = ce if m == 1.0 else ce + (m - 1.0) * p0_ce
        return per, ce, p0_ce, ce_w, p0m

    def _make_eval_batches(self):
        cfg = self.cfg
        total = cfg.eval_batches
        B = cfg.micro_batch * cfg.accum_target
        assert total > 0 and total % B == 0, (
            f"eval_batches ({total}) must be > 0 and divisible by "
            f"micro_batch * accum ({cfg.micro_batch} * {cfg.accum_target})")
        val = self._val
        g = np.random.default_rng(cfg.eval_seed)
        if cfg.epoch_sampler:
            order = g.permutation(val.n)
        else:
            order = np.arange(val.n)
        rows = pack_rows(val.lengths, order, cfg.seq_len, cfg.dense_batching)
        assert rows, "empty val split"
        tiled = [rows[i % len(rows)] for i in range(total)]
        out = []
        for i in range(0, total, B):
            xs, ys = [], []
            for ex_ix in tiled[i:i + B]:
                ids = np.concatenate([val.example(j) for j in ex_ix])
                x, y = self._pack_xy(ids, cfg.seq_len)
                xs.append(x)
                ys.append(y)
            x, y = self._to_device(np.stack(xs), np.stack(ys))
            out.append((x, y))
        return out

    def _ensure_eval_batches(self):
        if self.eval_batches is None:
            self.eval_batches = self._make_eval_batches()
            self.eval_block_masks = [
                self._block_mask(x) for x, _ in self.eval_batches]

    def _log(self, msg=""):
        if self._logf is not None:
            self._logf.write(msg + "\n")
            self._logf.flush()
        else:
            print(msg)

    def _lrs(self):
        muon = next((g["lr"] for g in self.opt.param_groups
                     if g.get("use_muon")), None)
        adam = next((g["lr"] for g in self.opt.param_groups
                     if not g.get("use_muon")), None)
        return adam, muon

    @torch.no_grad()
    def evaluate(self, progress_cb=None, probe=True):
        """Seed-locked val CE. Returns (nats, bpc, p0_nats).

        p0_nats is mean CE on the first answer token of each response
        (the target at the ASSISTANT position).
        """
        self._ensure_eval_batches()
        self.model.eval()
        try:
            tot = 0.0
            n = 0
            p0_tot = 0.0
            p0_n = 0
            for i, ((x, y), bm) in enumerate(
                    zip(self.eval_batches, self.eval_block_masks)):
                with self.ctx:
                    logits = self.model(x, block_mask=bm)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    per = F.cross_entropy(
                        logits.float().reshape(-1, logits.size(-1)),
                        y.reshape(-1), ignore_index=IGNORE_INDEX,
                        reduction="none")
                if (probe and self._probe is not None
                        and self._probe_keep(i)):
                    self._probe.record(logits, x, y, split="val",
                                       step=self.step, micro=i)
                valid = y.reshape(-1) != IGNORE_INDEX
                n_valid = int(valid.sum())
                if n_valid:
                    tot += per[valid].sum().item()
                    n += n_valid
                p0m = response_p0_mask(x, y, self.cfg.assistant_id).reshape(-1)
                n_p0 = int(p0m.sum())
                if n_p0:
                    p0_tot += per[p0m].sum().item()
                    p0_n += n_p0
                if progress_cb:
                    progress_cb(i + 1, len(self.eval_batches))
        finally:
            self.model.train()
        nats = tot / max(n, 1)
        p0 = p0_tot / max(p0_n, 1)
        return nats, nats / self.cpt / math.log(2), p0

    @torch.no_grad()
    def evaluate_full_val(self, progress_cb=None, isolate_docs=False):
        """Every val example once, same packing as training. Token-weighted.

        `isolate_docs` forces a BOS-segment mask when the model was not
        trained with doc_masking (custom val of a causal pretrain ckpt
        on an SFT set). Native SFT full-val leaves this off.
        """
        cfg = self.cfg
        val = self._val
        T, dense = cfg.seq_len, cfg.dense_batching
        B = cfg.micro_batch * cfg.accum_target
        rows = pack_rows(val.lengths, np.arange(val.n), T, dense)
        n_rows = len(rows)
        n_drop = 0
        n_pad = (B - (n_rows % B)) % B
        padded = rows + [None] * n_pad
        n_macros = len(padded) // B
        assert n_macros > 0, "val split empty"
        if progress_cb:
            progress_cb(0, n_macros)
        self.model.eval()
        tot = 0.0
        n_tok = 0
        p0_tot = 0.0
        p0_n = 0
        try:
            for i in range(n_macros):
                chunk = padded[i * B:(i + 1) * B]
                xs, ys = [], []
                for ex_ix in chunk:
                    if ex_ix is None:
                        x = np.full(T, cfg.bos_id, np.int64)
                        y = np.full(T, IGNORE_INDEX, np.int64)
                    else:
                        ids = np.concatenate([val.example(j) for j in ex_ix])
                        x, y = self._pack_xy(ids, T)
                    xs.append(x)
                    ys.append(y)
                x, y = self._to_device(np.stack(xs), np.stack(ys))
                bm = self._block_mask(x)
                am = None
                if bm is None and isolate_docs:
                    am = packed_doc_attention_mask(x, cfg.bos_id)
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
                p0m = response_p0_mask(x, y, cfg.assistant_id).reshape(-1)
                n_p0 = int(p0m.sum())
                if n_p0:
                    p0_tot += per[p0m].sum().item()
                    p0_n += n_p0
                if progress_cb:
                    progress_cb(i + 1, n_macros)
        finally:
            self.model.train()
        nats = tot / max(n_tok, 1)
        p0 = p0_tot / max(p0_n, 1) if p0_n else None
        p0_bpc = (p0 / self.cpt / math.log(2)) if p0 is not None else None
        return nats, nats / self.cpt / math.log(2), {
            "batches": n_macros, "batch": B, "seq_len": T,
            "windows": n_rows, "dropped": n_drop, "tokens": n_tok,
            "p0_nats": p0, "p0_bpc": p0_bpc, "p0_tokens": p0_n,
        }

    def train(self):
        cfg, p = self.cfg, self.plan
        best_nats = min((h["val_nats"] for h in self.history),
                        default=float("inf"))
        self._log(f"SFT  epochs {cfg.epochs}  steps {p['steps']:,}  "
                  f"dense {cfg.dense_batching}  "
                  f"shuffle {cfg.epoch_sampler}")
        self._log(f"  warmup {cfg.warmup} -> {cfg.warmup_steps} steps "
                  f"(from 0)  {cfg.schedule} to 0  "
                  f"peak lr {cfg.lr:.2e} (ratio {cfg.lr_ratio:g} × "
                  f"orig {cfg.orig_lr})")
        self._log(f"  vocab {self.meta.get('vocab_size', '?')}  "
                  f"cpt {self.cpt:.4f}  muon "
                  f"{'on' if cfg.use_muon else 'off'}")
        self._log(f"  accum {cfg.accum_target}  answer-token scale "
                  f"{self._tok_scale:.1f}/step (sum CE, token-weighted)")
        if cfg.p0_loss_multiplier != 1.0:
            self._log(f"  p0_loss_multiplier {cfg.p0_loss_multiplier:g} "
                      f"(first-answer-token CE in the train sum)")
        if cfg.lowdiff_masking:
            self._log(f"  lowdiff_masking on  threshold "
                      f"{cfg.lowdiff_threshold:g} "
                      f"(mask answer tokens copied from the prompt)")
        if self.step >= 0:
            left = p["steps"] - self.step - 1
            self._log(f"  RESUME: steps 0..{self.step} done, {left:,} left")
        self._log("")

        steps = range(self.step + 1, p["steps"])
        if cfg.progress:
            from tqdm import tqdm
            steps = tqdm(steps, initial=self.step + 1, total=p["steps"],
                         desc="sft", unit="step", mininterval=5,
                         file=__import__("sys").stderr)

        self.model.train()
        t0 = time.time()
        self._pause = False
        prev_term = signal.signal(signal.SIGTERM,
                                  lambda *_: setattr(self, "_pause", True))
        acc = cfg.accum_target
        for step in steps:
            self.step = step
            self.opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            n_valid_sum = 0
            p0_sum = 0.0
            p0_n_sum = 0
            finished_epoch = None
            for micro in range(acc):
                x_np, y_np, done_ep = self._train_batcher.next_micro()
                if done_ep is not None:
                    finished_epoch = done_ep
                x, y = self._to_device(x_np, y_np)
                bm = self._block_mask(x)
                with self.ctx:
                    logits = self.model(x, block_mask=bm)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    _per, ce, p0_ce, ce_w, p0m = self._ce_train(logits, x, y)
                if (self._probe is not None
                        and self._probe_keep(step * acc + micro)):
                    self._probe.record(
                        logits, x, y, split="train", step=step,
                        micro=micro,
                        epoch=self._train_batcher.epoch + 1)
                # token-weighted: sum over answer tokens / fixed per-step
                # scale. Do NOT divide mean-per-micro by acc — that would
                # give every micro-batch equal weight regardless of how
                # many answer tokens it holds. p0_loss_multiplier scales
                # only the first-answer-token terms in ce_w.
                (ce_w / self._tok_scale).backward()
                n_valid = int((y != IGNORE_INDEX).sum())
                if n_valid:
                    loss_sum += ce.item()
                    n_valid_sum += n_valid
                n_p0 = int(p0m.sum())
                if n_p0:
                    p0_sum += p0_ce.item()
                    p0_n_sum += n_p0
            self.tokens += cfg.micro_batch * cfg.seq_len * acc

            gn = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.grad_clip).item()

            probing = cfg.probe_every and step % cfg.probe_every == 0
            if probing:
                self.probe.snapshot()
            self.opt.step()
            self.sched.step()

            train_loss = loss_sum / max(n_valid_sum, 1)
            train_p0 = p0_sum / max(p0_n_sum, 1)
            adam_lr, muon_lr = self._lrs()

            if cfg.log_every and step % cfg.log_every == 0:
                lr_s = f"lr {adam_lr:.2e}"
                if muon_lr is not None:
                    lr_s += f"  muon {muon_lr:.2e}"
                extra = (f"  ep {self._train_batcher.epoch + 1}/"
                         f"{cfg.epochs}")
                self._log(f"step {step:>6}  loss {train_loss:7.4f}  "
                          f"bpc {train_loss/self.cpt/math.log(2):6.4f}  "
                          f"{lr_s}  gn {gn:5.2f}  acc {acc:>3}  "
                          f"tok {self.tokens/1e6:8.1f}M  "
                          f"{time.time()-t0:6.0f}s{extra}  "
                          f"p0 {train_p0:7.4f}")

            if probing:
                self._log("  update-to-param:")
                print_report(self.probe.measure(), file=self._logf)

            if cfg.eval_every and (step % cfg.eval_every == 0
                                   or step == p["steps"] - 1):
                nats, bpc, p0 = self.evaluate()
                p0_bpc = p0 / self.cpt / math.log(2)
                self.history.append({"step": step, "tokens": self.tokens,
                                     "val_nats": nats, "val_bpc": bpc,
                                     "val_p0_nats": p0, "val_p0_bpc": p0_bpc})
                self._log(f"  EVAL step {step:>6}  val {nats:.4f} nats/tok  "
                          f"{bpc:.4f} bits/char  p0 {p0:.4f} nats/tok  "
                          f"{p0_bpc:.4f} bits/char")
                if cfg.ckpt_best and nats < best_nats:
                    best_nats = nats
                    self.save("ckpt_best.pt")

            if ((cfg.ckpt_every and step and step % cfg.ckpt_every == 0)
                    or step in self._ckpt_at):
                self.save(f"ckpt_{step:06d}.pt")

            if finished_epoch is not None and finished_epoch in cfg.ckpt_epochs:
                self.save(f"ckpt_epoch_{finished_epoch}.pt")
                self._log(f"  EPOCH {finished_epoch} checkpoint")

            if self._pause:
                self.save("ckpt_paused.pt")
                self._log(f"  PAUSED after step {step} -> ckpt_paused.pt")
                self._close_probe()
                raise Paused(f"paused after step {step}")

        signal.signal(signal.SIGTERM, prev_term)
        if cfg.progress:
            steps.close()

        nats, bpc, p0 = self.evaluate(probe=not bool(cfg.eval_every))
        p0_bpc = p0 / self.cpt / math.log(2)
        self.final = {"val_nats": nats, "val_bpc": bpc,
                      "val_p0_nats": p0, "val_p0_bpc": p0_bpc}
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
            p0s = ""
            if info.get("p0_nats") is not None:
                self.full_val["val_p0_nats"] = info["p0_nats"]
                self.full_val["val_p0_bpc"] = info["p0_bpc"]
                p0s = (f"  p0 {info['p0_nats']:.4f} nats/tok  "
                       f"{info['p0_bpc']:.4f} bits/char")
            self._log(f"  FULL_VAL val {fv_nats:.4f} nats/tok  "
                      f"{fv_bpc:.4f} bits/char{p0s}  "
                      f"({info['windows']} examples, drop {info['dropped']})")
        if cfg.ckpt_final:
            self.save("ckpt_final.pt")

        self._log(f"\ndone: {cfg.epochs} epoch(s), "
                  f"val {nats:.4f} nats/tok = {bpc:.4f} bits/char, "
                  f"p0 {p0:.4f}, "
                  f"{time.time()-t0:.0f}s")
        self._close_probe()
        if self._logf is not None:
            self._logf.close()
            self._logf = None
        return {"val_nats": nats, "val_bpc": bpc, "val_p0_nats": p0,
                "val_p0_bpc": p0_bpc, "tokens": self.tokens,
                "history": self.history}

    def _close_probe(self):
        if self._probe is not None:
            n = len(self._probe.items)
            self._log(f"  deep_probe wrote {n} windows")
            self._probe.close()
            self._probe = None

    def save(self, name):
        path = os.path.join(self.cfg.out_dir, name)
        cfg = asdict(self.cfg)
        cfg.pop("eval_only", None)
        torch.save({"model": self._raw_model().state_dict(),
                    "model_arch": getattr(self._raw_model(), "arch", None),
                    "opt": self.opt.state_dict(),
                    "sched": self.sched.state_dict(),
                    "step": self.step, "tokens": self.tokens,
                    "np_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": (torch.cuda.get_rng_state()
                                 if torch.cuda.is_available() else None),
                    "sampler": (None if self._train_batcher is None
                                else self._train_batcher.state_dict()),
                    "final": getattr(self, "final", None),
                    "full_val": getattr(self, "full_val", None),
                    "cfg": cfg, "history": self.history}, path)
        return path
