"""Original P2 adamw=4e-3 muon_lr sweep (measurement A, P2 only).

s = median over Muon-managed 2D weights of rms(W)*sqrt(fan_in)/0.55.
Also prints/plots the by-role and by-layer breakdown.

For P3 and for s-vs-area see muon_scaling_check_a.py / _b.py.

    python -m analysis.muon_scaling_check
"""
import argparse
import os

import numpy as np

from analysis import muon_scaling as ms
from analysis.muon_scaling_check_a import P2, run_model


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="ckpt_final.pt")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()
    run_model(P2, args.ckpt, args.out_dir, args.no_plot)


if __name__ == "__main__":
    main()
