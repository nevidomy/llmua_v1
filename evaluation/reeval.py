"""Re-evaluate a finished run's val loss, exactly as training measured it.

The eval is fully determined by the checkpoint's stored TrainConfig:
Trainer._make_eval_batches() draws from np.random.default_rng(eval_seed),
so the same val windows come out every time. We rebuild a real Trainer
(eval_only: bins + batches + autocast, no optimizer, no out_dir writes)
from that cfg -- then call Trainer.evaluate() and report the fresh
numbers next to the ones stored in the checkpoint's `final`.
"""

import math
import os

import torch


def reevaluate(ckpt_path, device=None, progress_cb=None):
    from sample_generator import load_model
    from model.training_loop import TrainConfig, Trainer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if progress_cb:
        progress_cb(phase="loading model", done=0, total=0)
    model, ck = load_model(ckpt_path, device)

    stored_cfg = ck.get("cfg") or {}
    if not stored_cfg:
        raise RuntimeError(f"{ckpt_path} has no stored cfg -- cannot "
                           f"reproduce the eval setup")
    if stored_cfg.get("kind") == "sft":
        from sft_training_loop import sft_eval_trainer
        if progress_cb:
            progress_cb(phase="building eval batches", done=0, total=0)
        trainer = sft_eval_trainer(model, stored_cfg, device)
        trainer._ensure_eval_batches()
        n = len(trainer.eval_batches)
        prog = (lambda d, t: progress_cb(phase="evaluating", done=d, total=t)
                ) if progress_cb else None
        if prog and n:
            prog(0, n)
        nats, bpc, p0 = trainer.evaluate(progress_cb=prog)
        stored, stored_from = _stored_loss(ck)
        cfg = trainer.cfg
        p0_bpc = p0 / trainer.cpt / math.log(2)
        return {
            "ckpt": os.path.basename(ckpt_path),
            "step": _int(ck.get("step")), "tokens": _int(ck.get("tokens")),
            "stored": stored, "stored_from": stored_from,
            "fresh": {"val_nats": float(nats), "val_bpc": float(bpc),
                      "val_p0_nats": float(p0), "val_p0_bpc": float(p0_bpc)},
            "eval": {"batches": n,
                     "batch": int(cfg.micro_batch),
                     "sequences": int(cfg.eval_batches),
                     "seq_len": int(cfg.seq_len), "seed": int(cfg.eval_seed),
                     "dtype": cfg.dtype, "device": device},
        }
    fields = TrainConfig.__dataclass_fields__
    kw = {k: v for k, v in stored_cfg.items() if k in fields}
    # historic cfg: eval_batches counted (micro * accum) forwards, not
    # total sequences. Translate so reeval still hits the same windows.
    eb, mbs = kw.get("eval_batches"), kw.get("micro_batch")
    acc = kw.get("accum_target") or 1
    if eb is not None and mbs and eb % mbs != 0:
        kw["eval_batches"] = eb * mbs * acc
    kw.update(
        eval_only=True,             # skip opt/sched; don't touch out_dir
        log_file=None,
        progress=False,
        load_from_file=None,        # weights come from load_model above
        load_from_file_for_decay=None,
        extend_schedule=False,      # continue-runs store this; eval doesn't load
        compile=False,              # one-off eval: skip the compile warmup
        device=device)
    if progress_cb:
        progress_cb(phase="building eval batches", done=0, total=0)
    trainer = Trainer(model, TrainConfig(**kw))
    trainer._ensure_eval_batches()

    n = len(trainer.eval_batches)
    prog = (lambda d, t: progress_cb(phase="evaluating", done=d, total=t)
            ) if progress_cb else None
    if prog:
        prog(0, n)
    nats, bpc = trainer.evaluate(progress_cb=prog)

    stored, stored_from = _stored_loss(ck)
    cfg = trainer.cfg
    return {
        "ckpt": os.path.basename(ckpt_path),
        "step": _int(ck.get("step")), "tokens": _int(ck.get("tokens")),
        "stored": stored, "stored_from": stored_from,
        "fresh": {"val_nats": float(nats), "val_bpc": float(bpc)},
        "eval": {"batches": n,
                 "batch": int(cfg.micro_batch),
                 "sequences": int(cfg.eval_batches),
                 "seq_len": int(cfg.seq_len), "seed": int(cfg.eval_seed),
                 "dtype": cfg.dtype, "device": device},
    }


def _float(x):
    return None if x is None else float(x)


def _int(x):
    return None if x is None else int(x)


def _p0_pair(d):
    if not isinstance(d, dict) or d.get("val_p0_nats") is None:
        return {}
    out = {"val_p0_nats": _float(d["val_p0_nats"])}
    if d.get("val_p0_bpc") is not None:
        out["val_p0_bpc"] = _float(d["val_p0_bpc"])
    return out


def _stored_loss(ck):
    """Prefer checkpoint `final` (written at end-of-run); fall back to the
    last history eval so older ckpts still have something to compare."""
    fin = ck.get("final")
    if isinstance(fin, dict) and fin.get("val_nats") is not None:
        out = {"val_nats": _float(fin["val_nats"]),
               "val_bpc": _float(fin.get("val_bpc"))}
        out.update(_p0_pair(fin))
        return out, "final"
    hist = ck.get("history") or []
    if hist and hist[-1].get("val_nats") is not None:
        last = hist[-1]
        out = {"val_nats": _float(last["val_nats"]),
               "val_bpc": _float(last.get("val_bpc"))}
        out.update(_p0_pair(last))
        return out, "history"
    return None, None
