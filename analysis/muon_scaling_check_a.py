"""Measurement A: exponent a from s vs muon_lr at fixed duration.

s is the median, over Muon-managed 2D weights, of
    rms(W) * sqrt(fan_in) / 0.55
(init scale; s=1 at custom_init). Fit ln(s) ~ a ln(muon_lr).

  * a ≈ 0    decay inert (s independent of muon_lr)
  * a ≈ 0.5  decay controls (s ~ sqrt(lr))

P2: adamw=4e-3, S≈24,253, muon_lr 0.002..0.04  (five points)
P3: adamw=4e-3, S=10,127, muon_lr 2e-3..7e-3  (four points)

    python -m analysis.muon_scaling_check_a
"""
import argparse
import os

import numpy as np

from analysis import muon_scaling as ms

P2 = dict(
    name="P2",
    title="P2 · adamw=4e-3 · S≈24253",
    runs=[
        dict(id=235, x=0.002, dir="llmua_P2/P2_lr_sweep_lr_0.004_muon_lr_0.002_seed_47"),
        dict(id=227, x=0.005, dir="llmua_P2/P2_lr_sweep_lr_0.004_muon_lr_0.005_seed_47"),
        dict(id=219, x=0.01,  dir="llmua_P2/P2_lr_sweep_lr_0.004_muon_lr_0.01_seed_47"),
        dict(id=228, x=0.02,  dir="llmua_P2/P2_lr_sweep_lr_0.004_muon_lr_0.02_seed_47"),
        dict(id=236, x=0.04,  dir="llmua_P2/P2_lr_sweep_lr_0.004_muon_lr_0.04_seed_47"),
    ],
)
P3 = dict(
    name="P3",
    title="P3 · adamw=4e-3 · S=10127",
    runs=[
        dict(id=279, x=0.002,  dir="llmua_P3/P3_LR_sweep_lr_0.004_muon_lr_0.002_seed_47"),
        dict(id=280, x=0.003,  dir="llmua_P3/P3_LR_sweep_lr_0.004_muon_lr_0.003_seed_47"),
        dict(id=290, x=0.0045, dir="llmua_P3/P3_LR_sweep_lr_0.004_muon_lr_0.0045_seed_47"),
        dict(id=291, x=0.007,  dir="llmua_P3/P3_LR_sweep_lr_0.004_muon_lr_0.007_seed_47"),
    ],
)


def run_model(spec, ckpt, out_dir, no_plot):
    print(f"\n=== A  {spec['title']} ===")
    grid = []
    for g in spec["runs"]:
        st = ms.read_run(g["dir"], ckpt)
        grid.append((g, st))
    ms.print_table(grid, "muon_lr")
    xs = np.array([g["x"] for g, _ in grid])
    rows = [st for _, st in grid]
    a = ms.fit_loglog_slope(xs, [r["median"] for r in rows])
    span = max(r["median"] for r in rows) / min(r["median"] for r in rows)
    print(f"a = d ln(s) / d ln(muon_lr) = {a:.3f}    "
          f"(inert 0,  decay-controls 0.5)")
    print(f"span max/min median-s = {span:.3f}x")
    print("role slopes:",
          ", ".join(f"{role} {ms.fit_loglog_slope(xs, [r['roles'][role] for r in rows]):.3f}"
                    for role in ms.ROLES))
    div = ms.role_divergence(rows)
    print(f"role spread (max/min of role-medians, worst point) = {div:.3f}x"
          + ("  -- roles DIVERGE" if div > 1.4 else "  -- roles track"))
    ms.print_detail(grid, "muon_lr")
    if not no_plot:
        os.makedirs(out_dir, exist_ok=True)
        refs = ((0.0, "slope 0 (inert)", "--"),
                (0.5, "slope 0.5 (controls)", ":"))
        ms.plot_loglog(
            xs, rows,
            xlabel="muon_lr",
            ylabel="s  (median rms · √fan_in / 0.55)",
            title=spec["title"],
            out_path=os.path.join(out_dir, f"muon_scaling_a_{spec['name']}.png"),
            ref_slopes=refs,
        )
        ms.plot_detail(
            xs, rows,
            xlabel="muon_lr",
            title=spec["title"],
            out_path=os.path.join(out_dir, f"muon_scaling_a_{spec['name']}_detail.png"),
            ref_slopes=refs,
        )
    return a


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="ckpt_final.pt")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()
    a2 = run_model(P2, args.ckpt, args.out_dir, args.no_plot)
    a3 = run_model(P3, args.ckpt, args.out_dir, args.no_plot)
    print(f"\nA summary:  a_P2 = {a2:.3f}   a_P3 = {a3:.3f}")


if __name__ == "__main__":
    main()
