"""Replay runs from experiments/runs.json, one subprocess at a time.

    python experiments/replay.py --list
    python experiments/replay.py --group paper1_final_R2 --dry-run
    python experiments/replay.py --id 92 --id 104
    python experiments/replay.py --id 104 --force

Selected entries run in id order (a branch needs its trunk first, and #104
extends #92). A run whose ckpt_final.pt already exists is skipped unless
--force. Stops on the first non-zero exit; exit code 42 means the run paused.
{ROOT} in the stored argv is replaced by the data root (env LLMUA_ROOT).
"""

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from data.index import get_root  # noqa: E402

PAUSED = 42
MODEL_DIRS = {"P1": "llmua_P1", "P2": "llmua_P2", "P3": "llmua_P3",
              "P4": "llmua_P4", "target": "llmua", "test": "nano_llmua"}


def load():
    with open(os.path.join(REPO, "experiments", "runs.json")) as f:
        return sorted(json.load(f), key=lambda e: e["id"])


def command(entry, root):
    argv = [a.replace("{ROOT}", root) for a in entry["argv"]]
    return [sys.executable, "-u", os.path.join(REPO, entry["runner"]), *argv]


def missing_input(cmd):
    """The --load_from_file* checkpoint of this command, if not on disk."""
    for flag in ("--load_from_file", "--load_from_file_for_decay"):
        if flag in cmd:
            path = cmd[cmd.index(flag) + 1]
            if not os.path.exists(path):
                return path
    return None


def final_ckpt(entry, root):
    a = entry["argv"]
    model = a[a.index("--model") + 1] if "--model" in a else "test"
    return os.path.join(root, "runs", MODEL_DIRS.get(model, f"llmua_{model}"),
                        a[a.index("--run") + 1], "ckpt_final.pt")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="list entries and exit")
    ap.add_argument("--group", action="append", default=[], metavar="G")
    ap.add_argument("--id", action="append", default=[], type=int, metavar="N")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands, run nothing")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if ckpt_final.pt exists")
    args = ap.parse_args()

    entries = load()
    if args.list:
        for e in entries:
            print(f"{e['id']:4d}  {e['group']:36s} {e['kind']:8s} {e['name']}")
        return 0
    groups = {e["group"] for e in entries}
    for g in args.group:
        if g not in groups:
            sys.exit(f"unknown group {g!r} (see --list)")
    ids = {e["id"] for e in entries}
    for i in args.id:
        if i not in ids:
            sys.exit(f"unknown id {i} (see --list)")
    if not args.group and not args.id:
        sys.exit("select runs with --group and/or --id (or use --list)")
    chosen = [e for e in entries if e["group"] in args.group or e["id"] in args.id]

    root = get_root()
    for e in chosen:
        cmd = command(e, root)
        tag = f"#{e['id']} [{e['group']}] {e['name']}"
        if e["note"]:
            print(f"note {tag}: {e['note']}")
        skip = os.path.exists(final_ckpt(e, root)) and not args.force
        if skip and not args.dry_run:
            print(f"skip {tag}: ckpt_final.pt exists")
            continue
        verb = ("would skip" if skip else "would run") if args.dry_run else "run"
        print(f"{verb} {tag}\n  {' '.join(cmd)}")
        if args.dry_run:
            continue
        need = missing_input(cmd)
        if need:
            print(f"stopped: {tag} needs {need}; run the run that makes it first")
            return 1
        rc = subprocess.run(cmd, cwd=REPO).returncode
        if rc == PAUSED:
            print(f"paused: {tag} exited with {PAUSED}; resume it before continuing")
            return rc
        if rc != 0:
            print(f"failed: {tag} exited with {rc}")
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
