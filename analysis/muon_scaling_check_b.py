"""Measurement B: exponent b from s vs schedule area at fixed LR.

s is the median, over Muon-managed 2D weights, of
    rms(W) * sqrt(fan_in) / 0.55
(init scale; s=1 at custom_init). Fit ln(s) ~ −b ln(area).

Area is the WSD schedule area (not step count): ratios 2.015 and 4.045
on the P2 branches, not 2 and 4.

  * b ≈ 0    s independent of duration
  * b ≈ 0.5  s ~ 1/sqrt(area)

P2: muon 5e-3, branches 20× / 40× / 80×  (area 5437, 10956, 21995)
P3: muon 3.65e-3, branches 20× / 40× / 80×  (area 9134, 18404, 36781)

    python -m analysis.muon_scaling_check_b
"""
import argparse
import os

import numpy as np

from analysis import muon_scaling as ms

P2 = dict(
    name="P2",
    title="P2 · muon 5e-3 · cooldown branches",
    runs=[
        dict(id=276, x=5437,  dir="llmua_P2/P2_cooldown_branching_20x_seed_47_run2"),
        dict(id=277, x=10956, dir="llmua_P2/P2_cooldown_branching_40x_seed_47_run2"),
        dict(id=269, x=21995, dir="llmua_P2/P2_cooldown_branching_80x_seed_47"),
    ],
)
P3 = dict(
    name="P3",
    title="P3 · muon 3.65e-3 · cooldown branches",
    runs=[
        dict(id=323, x=9134,  dir="llmua_P3/P3_seed47_lr0.003_muon_mlr0.00365_wsd_swiglu_rms_qk_nobias_cinit_cdb_x20"),
        dict(id=324, x=18404, dir="llmua_P3/P3_seed47_lr0.003_muon_mlr0.00365_wsd_swiglu_rms_qk_nobias_cinit_cdb_x40"),
        dict(id=307, x=36781, dir="llmua_P3/P3_seed47_lr0.003_muon_mlr0.00365_wsd_swiglu_rms_qk_nobias_cinit_cdb_x80"),
    ],
)


def run_model(spec, ckpt, out_dir, no_plot):
    print(f"\n=== B  {spec['title']} ===")
    grid = []
    for g in spec["runs"]:
        st = ms.read_run(g["dir"], ckpt)
        grid.append((g, st))
    ms.print_table(grid, "area", xfmt="{:.0f}")
    xs = np.array([g["x"] for g, _ in grid], float)
    rows = [st for _, st in grid]
    slope = ms.fit_loglog_slope(xs, [r["median"] for r in rows])
    b = -slope
    span = max(r["median"] for r in rows) / min(r["median"] for r in rows)
    # area ratios vs naive 2/4 from the multiplier
    r01, r02 = xs[1] / xs[0], xs[2] / xs[0]
    print(f"area ratios  {r01:.3f}, {r02:.3f}  (vs 2, 4 from the × multiplier)")
    print(f"b = − d ln(s) / d ln(area) = {b:.3f}    "
          f"(flat 0,  1/sqrt(area) 0.5)")
    print(f"span max/min median-s = {span:.3f}x")
    print("role b:",
          ", ".join(
              f"{role} {-ms.fit_loglog_slope(xs, [r['roles'][role] for r in rows]):.3f}"
              for role in ms.ROLES))
    div = ms.role_divergence(rows)
    print(f"role spread (max/min of role-medians, worst point) = {div:.3f}x"
          + ("  -- roles DIVERGE" if div > 1.4 else "  -- roles track"))
    ms.print_detail(grid, "area", xfmt="{:.0f}")
    if not no_plot:
        os.makedirs(out_dir, exist_ok=True)
        refs = ((0.0, "slope 0 (b=0)", "--"),
                (-0.5, "slope −0.5 (b=0.5)", ":"))
        ms.plot_loglog(
            xs, rows,
            xlabel="schedule area",
            ylabel="s  (median rms · √fan_in / 0.55)",
            title=spec["title"],
            out_path=os.path.join(out_dir, f"muon_scaling_b_{spec['name']}.png"),
            ref_slopes=refs,
        )
        ms.plot_detail(
            xs, rows,
            xlabel="schedule area",
            title=spec["title"],
            out_path=os.path.join(out_dir, f"muon_scaling_b_{spec['name']}_detail.png"),
            ref_slopes=refs,
        )
    return b


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="ckpt_final.pt")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()
    b2 = run_model(P2, args.ckpt, args.out_dir, args.no_plot)
    b3 = run_model(P3, args.ckpt, args.out_dir, args.no_plot)
    print(f"\nB summary:  b_P2 = {b2:.3f}   b_P3 = {b3:.3f}")


if __name__ == "__main__":
    main()
