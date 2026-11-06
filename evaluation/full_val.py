"""Full-split val loss: every packed window that fits a complete
macro-batch (micro_batch * accum_target). Leftover windows that don't
fill a batch are dropped.

Unlike reeval, this is NOT the training eval (which is a seed-locked
sample of random windows). It is a second number stored beside `final` as
`full_val` (val_nats / val_bpc). SFT also stores first-answer-token CE as
val_p0_nats / val_p0_bpc on that dict.

Custom-dataset evals pick a strategy from the dataset kind:

    full_val_loss_{name}              pretrain tiled windows (original)
    full_val_loss_packed_{name}       pretrain packed BOS…EOS (body→body; mean/cpt)
    full_val_loss_packed_uk4b_{name}  pretrain packed uk4b (BOS→body; Σ bits/chars)
    full_val_loss_sft_{name}          SFT one example per row
    full_val_loss_sft_packed_{name}   SFT dense pack

Run from the repo root via `python evaluate.py full-val`.
"""

import math
import os

import torch

from model.packed_val import custom_loss_key, is_custom_loss_key


def resolve_eval_dataset(kind, name):
    """Disk inspect for a named (or absolute) eval dataset."""
    from data.catalog import inspect_pretrain, inspect_sft
    from data.index import bins_dir, sft_bins_dir

    kind = (kind or "pretrain").strip().lower()
    if kind not in ("pretrain", "sft"):
        raise ValueError(f"kind must be pretrain or sft, got {kind!r}")
    name = str(name or "").strip()
    if not name:
        raise ValueError("dataset name required")
    inspect = inspect_sft if kind == "sft" else inspect_pretrain
    if os.path.isdir(name):
        info = inspect(os.path.abspath(name))
        if info:
            _require_val(info, kind, name)
            return info
    root = sft_bins_dir() if kind == "sft" else bins_dir()
    path = os.path.join(root, name)
    info = inspect(path) if os.path.isdir(path) else None
    if not info:
        raise FileNotFoundError(f"{kind} dataset not found: {name}")
    _require_val(info, kind, name)
    return info


def _require_val(info, kind, name):
    if kind == "sft" and not info.get("has_val"):
        raise FileNotFoundError(f"SFT dataset {name} has no val split")


def evaluate_full_val(ckpt_path, device=None, progress_cb=None,
                      dataset=None, ds_kind=None, mode="stream"):
    from sample_generator import load_model
    from model.training_loop import TrainConfig, Trainer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mode = (mode or "stream").strip().lower()
    if mode not in ("stream", "packed", "packed_uk4b"):
        raise ValueError(
            f"mode must be stream, packed, or packed_uk4b, got {mode!r}")
    if progress_cb:
        progress_cb(phase="loading model", done=0, total=0)
    model, ck = load_model(ckpt_path, device)

    stored_cfg = ck.get("cfg") or {}
    if not stored_cfg:
        raise RuntimeError(f"{ckpt_path} has no stored cfg -- cannot "
                           "reproduce the eval setup")

    custom = None
    if dataset:
        ds_kind = (ds_kind or "pretrain").strip().lower()
        info = resolve_eval_dataset(ds_kind, dataset)
        custom = {"name": info["name"], "kind": ds_kind, "mode": mode,
                  "path": info["path"],
                  "key": custom_loss_key(ds_kind, info["name"], mode)}

    if (custom and custom["kind"] == "sft") or (
            not custom and stored_cfg.get("kind") == "sft"):
        from sft_training_loop import sft_eval_trainer
        if progress_cb:
            progress_cb(phase="scanning val split", done=0, total=0)
        data_dir = custom["path"] if custom else None
        dense = None
        if custom and custom["kind"] == "sft":
            if custom["mode"] == "packed_uk4b":
                raise ValueError("packed_uk4b is pretrain-only")
            dense = custom["mode"] == "packed"
        trainer = sft_eval_trainer(model, stored_cfg, device,
                                   data_dir=data_dir, dense_batching=dense)
        isolate = bool(custom and stored_cfg.get("kind") != "sft")
        prog = (lambda d, t: progress_cb(phase="evaluating", done=d, total=t)
                ) if progress_cb else None
        nats, bpc, info = trainer.evaluate_full_val(
            progress_cb=prog, isolate_docs=isolate)
        return _result(ckpt_path, ck, trainer.cfg, device, nats, bpc, info,
                       custom)

    fields = TrainConfig.__dataclass_fields__
    kw = {k: v for k, v in stored_cfg.items() if k in fields}
    if custom:
        kw["bins_dir"] = custom["path"]
    kw.update(
        eval_only=True,
        log_file=None,
        progress=False,
        load_from_file=None,
        load_from_file_for_decay=None,
        extend_schedule=False,
        compile=False,
        device=device)
    if progress_cb:
        progress_cb(phase="scanning val split", done=0, total=0)
    trainer = Trainer(model, TrainConfig(**kw))

    prog = (lambda d, t: progress_cb(phase="evaluating", done=d, total=t)
            ) if progress_cb else None
    if custom and custom["mode"] in ("packed", "packed_uk4b"):
        rule = "uk4b" if custom["mode"] == "packed_uk4b" else "content"
        nats, bpc, info = trainer.evaluate_full_val_packed(
            progress_cb=prog, rule=rule)
    else:
        nats, bpc, info = trainer.evaluate_full_val(progress_cb=prog)
    return _result(ckpt_path, ck, trainer.cfg, device, nats, bpc, info,
                   custom)


def _result(ckpt_path, ck, cfg, device, nats, bpc, info, custom):
    stored = (_stored_custom(ck, custom["key"]) if custom
              else _stored_full_val(ck))
    out = {
        "ckpt": os.path.basename(ckpt_path),
        "step": _int(ck.get("step")), "tokens": _int(ck.get("tokens")),
        "stored": stored,
        "fresh": _fresh_full_val(nats, bpc, info),
        "eval": {"batches": info["batches"], "batch": info["batch"],
                 "seq_len": info["seq_len"], "windows": info["windows"],
                 "dropped": info["dropped"], "tokens": info["tokens"],
                 "p0_tokens": info.get("p0_tokens"),
                 "docs": info.get("docs"),
                 "skipped_oversize": info.get("skipped_oversize"),
                 "skipped_no_eos": info.get("skipped_no_eos"),
                 "mode": (custom or {}).get("mode") or info.get("mode")
                 or "stream",
                 "dtype": cfg.dtype, "device": device},
    }
    if custom:
        out["key"] = custom["key"]
        out["dataset"] = custom["name"]
        out["kind"] = custom["kind"]
        out["mode"] = custom["mode"]
        out["fresh"]["key"] = custom["key"]
        out["fresh"]["dataset"] = custom["name"]
        out["fresh"]["kind"] = custom["kind"]
        out["fresh"]["mode"] = custom["mode"]
    return out


def _float(x):
    return None if x is None else float(x)


def _int(x):
    return None if x is None else int(x)


def _ok_loss(x):
    return x is not None and math.isfinite(x) and 0 < x < 100


def _p0_pair(src):
    nats = src.get("val_p0_nats") if isinstance(src, dict) else None
    if nats is None and isinstance(src, dict):
        nats = src.get("p0_nats")
    if nats is None:
        return {}
    bpc = None
    if isinstance(src, dict):
        bpc = src.get("val_p0_bpc")
        if bpc is None:
            bpc = src.get("p0_bpc")
    out = {"val_p0_nats": _float(nats)}
    if bpc is not None:
        out["val_p0_bpc"] = _float(bpc)
    return out


def _fresh_full_val(nats, bpc, info):
    fresh = {"val_nats": float(nats), "val_bpc": float(bpc)}
    fresh.update(_p0_pair(info or {}))
    return fresh


def _loss_rec(fv):
    if not isinstance(fv, dict) or fv.get("val_nats") is None:
        return None
    out = {"val_nats": _float(fv["val_nats"]),
           "val_bpc": _float(fv.get("val_bpc"))}
    out.update(_p0_pair(fv))
    for k in ("dataset", "kind", "mode", "key"):
        if fv.get(k) is not None:
            out[k] = fv[k]
    return out


def _stored_full_val(ck):
    return _loss_rec(ck.get("full_val"))


def _stored_custom(ck, key):
    return _loss_rec(ck.get(key))


def custom_from_ckpt(ck):
    """{key: rec} for every full_val_loss_* dict on a checkpoint."""
    out = {}
    if not isinstance(ck, dict):
        return out
    for k, v in ck.items():
        if is_custom_loss_key(k) and isinstance(v, dict) \
                and v.get("val_nats") is not None:
            rec = _loss_rec(v)
            if rec:
                rec.setdefault("key", k)
                out[k] = rec
    return out


def save_full_val(ckpt_path, val_nats, val_bpc,
                  val_p0_nats=None, val_p0_bpc=None):
    """Write `full_val` onto ckpt_final.pt (tmp+replace). Does not touch
    training `final` / val_nats / val_bpc. p0 is SFT-only and optional."""
    rec = _checked_rec(val_nats, val_bpc, val_p0_nats, val_p0_bpc)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    prev = ck.get("full_val") if isinstance(ck.get("full_val"), dict) else {}
    ck["full_val"] = {**prev, **rec}
    _atomic_save(ck, ckpt_path)
    return float(rec["val_nats"]), float(rec["val_bpc"])


def save_custom_full_val(ckpt_path, key, val_nats, val_bpc,
                         val_p0_nats=None, val_p0_bpc=None,
                         dataset=None, kind=None, mode=None):
    """Write `full_val_loss_*` onto the checkpoint (tmp+replace)."""
    if not is_custom_loss_key(key):
        raise ValueError(f"bad custom val key: {key!r}")
    rec = _checked_rec(val_nats, val_bpc, val_p0_nats, val_p0_bpc)
    rec["key"] = key
    if dataset:
        rec["dataset"] = dataset
    if kind:
        rec["kind"] = kind
    if mode:
        rec["mode"] = mode
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    prev = ck.get(key) if isinstance(ck.get(key), dict) else {}
    ck[key] = {**prev, **rec}
    _atomic_save(ck, ckpt_path)
    return rec


def _checked_rec(val_nats, val_bpc, val_p0_nats=None, val_p0_bpc=None):
    if not (_ok_loss(val_nats) and _ok_loss(val_bpc)):
        raise ValueError("implausible loss values")
    if val_p0_nats is not None and not _ok_loss(val_p0_nats):
        raise ValueError("implausible p0 loss values")
    if val_p0_bpc is not None and not _ok_loss(val_p0_bpc):
        raise ValueError("implausible p0 loss values")
    rec = {"val_nats": float(val_nats), "val_bpc": float(val_bpc)}
    if val_p0_nats is not None:
        rec["val_p0_nats"] = float(val_p0_nats)
        if val_p0_bpc is not None:
            rec["val_p0_bpc"] = float(val_p0_bpc)
    return rec


def _atomic_save(ck, ckpt_path):
    tmp = ckpt_path + ".fullval-tmp"
    torch.save(ck, tmp)
    os.replace(tmp, ckpt_path)
