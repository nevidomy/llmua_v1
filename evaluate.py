"""Evaluation entry points: re-measure a checkpoint's validation loss.

    python evaluate.py full-val --ckpt C.pt [--dataset NAME] [--ds_kind pretrain|sft]
                                [--mode stream|packed|packed_uk4b] [--device D] [--save]
    python evaluate.py subset --ckpt C.pt [--device D]
    python evaluate.py instruments --ckpt C.pt [--device D] [--save]

full-val      every complete macro-batch of the val split (or of a named
              eval dataset under the bins root); prints JSON with the fresh
              number next to the one stored in the checkpoint.
subset        the training-time eval: the seed-locked sample of val windows
              (eval_batches sequences) rebuilt from the stored config.
instruments   the paper's three full-val instruments: full val tiles (no
              dataset, stream), BrUK documents (`bruk`, stream) and BrUK
              sentences under uk4b's rule (`bruk_uk4b_sentenses`,
              packed_uk4b). The spelling "sentenses" is load-bearing: it is
              part of the keys stored in checkpoints.

Read-only by default. `--save` writes the fresh number back onto the
checkpoint (tmp file + replace) through save_full_val / save_custom_full_val.
"""

import argparse
import json
import sys

# (label, dataset, ds_kind, mode)
INSTRUMENTS = [
    ("full val tiles", None, "pretrain", "stream"),
    ("BrUK documents", "bruk", "pretrain", "stream"),
    ("BrUK sentences (uk4b)", "bruk_uk4b_sentenses", "pretrain",
     "packed_uk4b"),
]


def _progress(**fields):
    phase = fields.get("phase", "")
    done, total = fields.get("done", 0), fields.get("total", 0)
    if total:
        print(f"\r{phase}: {done}/{total}", end="", file=sys.stderr,
              flush=True)
        if done >= total:
            print(file=sys.stderr)
    elif phase:
        print(phase, file=sys.stderr, flush=True)


def _save(ckpt, res):
    from evaluation.full_val import save_custom_full_val, save_full_val
    fresh = res["fresh"]
    p0 = dict(val_p0_nats=fresh.get("val_p0_nats"),
              val_p0_bpc=fresh.get("val_p0_bpc"))
    if res.get("key"):
        save_custom_full_val(
            ckpt, res["key"], fresh["val_nats"], fresh["val_bpc"], **p0,
            dataset=res.get("dataset"), kind=res.get("kind"),
            mode=res.get("mode"))
    else:
        save_full_val(ckpt, fresh["val_nats"], fresh["val_bpc"], **p0)


def _full_val(args, dataset, ds_kind, mode):
    from evaluation.full_val import evaluate_full_val
    return evaluate_full_val(
        args.ckpt, device=args.device, dataset=dataset, ds_kind=ds_kind,
        mode=mode, progress_cb=_progress)


def cmd_full_val(args):
    res = _full_val(args, args.dataset, args.ds_kind, args.mode)
    print(json.dumps(res, indent=2))
    if args.save:
        _save(args.ckpt, res)
        print("saved", res.get("key") or "full_val", "->", args.ckpt)


def cmd_subset(args):
    from evaluation.reeval import reevaluate
    res = reevaluate(args.ckpt, device=args.device, progress_cb=_progress)
    print(json.dumps(res, indent=2))


def cmd_instruments(args):
    results = []
    for label, dataset, ds_kind, mode in INSTRUMENTS:
        res = _full_val(args, dataset, ds_kind, mode)
        results.append((label, res))
        if args.save:
            _save(args.ckpt, res)
    print(json.dumps([dict(res, instrument=label) for label, res in results],
                     indent=2))
    print()
    print(f"{'instrument':<24}{'fresh bpc':>12}{'stored bpc':>12}")
    for label, res in results:
        stored = res["stored"]
        s = f"{stored['val_bpc']:.4f}" if stored else "-"
        print(f"{label:<24}{res['fresh']['val_bpc']:>12.4f}{s:>12}")
    if args.save:
        print("saved", len(results), "records ->", args.ckpt)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    fv = sub.add_parser("full-val", help="full-split val loss")
    fv.add_argument("--ckpt", required=True)
    fv.add_argument("--dataset", default=None,
                    help="eval dataset name or dir (omit = the run's own val)")
    fv.add_argument("--ds_kind", default=None, choices=["pretrain", "sft"])
    fv.add_argument("--mode", default="stream",
                    choices=["stream", "packed", "packed_uk4b"])
    fv.add_argument("--device", default=None)
    fv.add_argument("--save", action="store_true",
                    help="write the fresh number onto the checkpoint")
    fv.set_defaults(fn=cmd_full_val)

    ss = sub.add_parser("subset", help="the seed-locked training-time eval")
    ss.add_argument("--ckpt", required=True)
    ss.add_argument("--device", default=None)
    ss.set_defaults(fn=cmd_subset)

    ins = sub.add_parser("instruments", help="the paper's three instruments")
    ins.add_argument("--ckpt", required=True)
    ins.add_argument("--device", default=None)
    ins.add_argument("--save", action="store_true",
                     help="write the fresh numbers onto the checkpoint")
    ins.set_defaults(fn=cmd_instruments)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
