"""Shared helpers for muon_scaling_check*.py."""
import os
import re
from collections import defaultdict

import numpy as np
import torch

from data.index import ROOT

INIT_STD = 0.55
ROLES = ("qkv", "o_proj", "gate_up", "down")
_LAYER = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
ROLE_COLORS = dict(qkv="C1", o_proj="C2", gate_up="C3", down="C4")


def is_muon_managed(name):
    name = name.removeprefix("_orig_mod.")
    if name in ("embed.weight", "lmh.weight"):
        return False
    return name.endswith(".weight")


def role_of(name):
    name = name.removeprefix("_orig_mod.")
    if name.endswith(".qkv.weight"):
        return "qkv"
    if name.endswith(".proj.weight"):
        return "o_proj"
    if name.endswith(".lin1.weight"):
        return "gate_up"
    if name.endswith(".lin2.weight"):
        return "down"
    return "other"


def layer_of(name):
    name = name.removeprefix("_orig_mod.")
    m = _LAYER.search(name)
    return int(m.group(1)) if m else None


def scale_s(p):
    rms = p.float().pow(2).mean().sqrt().item()
    return rms * p.shape[1] ** 0.5 / INIT_STD


def _med(v):
    return float(np.median(v)) if v else float("nan")


def read_run(subdir, ckpt="ckpt_final.pt"):
    path = os.path.join(ROOT, "runs", subdir, ckpt)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    by_role = defaultdict(list)
    by_layer = defaultdict(list)
    cell = {}                       # (layer, role) -> s
    all_s = []
    for name, p in sd.items():
        if p.ndim != 2 or not is_muon_managed(name):
            continue
        s = scale_s(p)
        all_s.append(s)
        r, L = role_of(name), layer_of(name)
        by_role[r].append(s)
        if L is not None:
            by_layer[L].append(s)
            cell[L, r] = s
    if not all_s:
        raise RuntimeError(f"no muon-managed 2D weights in {path}")
    a = np.array(all_s)
    layers = sorted(by_layer)
    out = {
        "n": len(a),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p16": float(np.percentile(a, 16)),
        "p84": float(np.percentile(a, 84)),
        "min": float(a.min()),
        "max": float(a.max()),
        "roles": {r: _med(by_role.get(r, [])) for r in ROLES},
        "role_n": {r: len(by_role.get(r, [])) for r in ROLES},
        "by_role": {r: list(by_role[r]) for r in ROLES if r in by_role},
        "by_layer": {i: list(by_layer[i]) for i in layers},
        "layer_med": {i: _med(by_layer[i]) for i in layers},
        "layers": layers,
        "cell": cell,
        "path": path,
    }
    return out


def fit_loglog_slope(xs, ys):
    x, y = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    x = x - x.mean()
    y = y - y.mean()
    den = float((x * x).sum())
    if den == 0:
        return float("nan")
    return float((x * y).sum() / den)


def plot_loglog(xs, rows, *, xlabel, ylabel, title, out_path, ref_slopes=()):
    """Summary: median + 16–84 band + four role medians."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    med = np.array([r["median"] for r in rows])
    lo = np.array([r["p16"] for r in rows])
    hi = np.array([r["p84"] for r in rows])
    ax.errorbar(xs, med, yerr=[med - lo, hi - med], fmt="o-",
                color="C0", zorder=4, capsize=3, label="median (16–84)")
    for role in ROLES:
        ys = [r["roles"][role] for r in rows]
        if any(np.isnan(ys)):
            continue
        ax.plot(xs, ys, "s--", color=ROLE_COLORS[role], alpha=0.85,
                linewidth=1, markersize=4, label=role)

    gx, gy = np.exp(np.mean(np.log(xs))), np.exp(np.mean(np.log(med)))
    xref = np.array([min(xs) / 1.15, max(xs) * 1.15])
    for sl, lab, sty in ref_slopes:
        ax.plot(xref, gy * (xref / gx) ** sl, sty, color="0.45",
                linewidth=1, label=lab)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def plot_detail(xs, rows, *, xlabel, title, out_path, ref_slopes=()):
    """Two-panel detail: left = by role, right = by layer. Plus a
    layer×role heatmap strip, one column per run."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    nrun = len(rows)
    layers = rows[0]["layers"]
    nL = len(layers)
    fig = plt.figure(figsize=(11.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.38, wspace=0.28)
    ax_r = fig.add_subplot(gs[0, 0])
    ax_l = fig.add_subplot(gs[0, 1])
    ax_h = fig.add_subplot(gs[1, :])

    med = np.array([r["median"] for r in rows])
    gx, gy = np.exp(np.mean(np.log(xs))), np.exp(np.mean(np.log(med)))
    xref = np.array([min(xs) / 1.15, max(xs) * 1.15])

    for role in ROLES:
        ys = [r["roles"][role] for r in rows]
        ax_r.plot(xs, ys, "o-", color=ROLE_COLORS[role], label=role)
    ax_r.plot(xs, med, "k--", linewidth=1, alpha=0.5, label="all-median")
    for sl, lab, sty in ref_slopes:
        ax_r.plot(xref, gy * (xref / gx) ** sl, sty, color="0.5", linewidth=1)
    ax_r.set_xscale("log")
    ax_r.set_yscale("log")
    ax_r.set_xlabel(xlabel)
    ax_r.set_ylabel("s")
    ax_r.set_title("by role")
    ax_r.legend(frameon=False, fontsize=8)

    cmap_L = plt.cm.viridis(np.linspace(0.1, 0.9, nL))
    for i, L in enumerate(layers):
        ys = [r["layer_med"][L] for r in rows]
        ax_l.plot(xs, ys, "o-", color=cmap_L[i],
                  label=f"L{L}" if (i == 0 or i == nL - 1 or nL <= 8) else None)
    for sl, lab, sty in ref_slopes:
        ax_l.plot(xref, gy * (xref / gx) ** sl, sty, color="0.5", linewidth=1)
    ax_l.set_xscale("log")
    ax_l.set_yscale("log")
    ax_l.set_xlabel(xlabel)
    ax_l.set_ylabel("s")
    ax_l.set_title("by layer (median of 4 roles)")
    ax_l.legend(frameon=False, fontsize=7, ncol=2)

    # heatmap: rows = layer, cols = run×role blocks
    grid = np.full((nL, nrun * len(ROLES)), np.nan)
    for j, r in enumerate(rows):
        for k, role in enumerate(ROLES):
            for i, L in enumerate(layers):
                grid[i, j * len(ROLES) + k] = r["cell"].get((L, role), np.nan)
    im = ax_h.imshow(grid, aspect="auto", origin="lower",
                     cmap="magma", norm=LogNorm(vmin=np.nanmin(grid),
                                                vmax=np.nanmax(grid)))
    ax_h.set_yticks(range(nL))
    ax_h.set_yticklabels([str(L) for L in layers])
    ax_h.set_ylabel("layer")
    xt = [j * len(ROLES) + 1.5 for j in range(nrun)]
    ax_h.set_xticks(xt)
    ax_h.set_xticklabels([f"{xs[j]:.4g}" for j in range(nrun)])
    ax_h.set_xlabel(xlabel)
    for j in range(1, nrun):
        ax_h.axvline(j * len(ROLES) - 0.5, color="w", linewidth=0.8)
    # tiny role letters under the first block as a legend
    ax_h.set_title("layer × role  (blocks = " + " | ".join(ROLES) + " per x)")
    fig.colorbar(im, ax=ax_h, fraction=0.02, pad=0.01, label="s")
    fig.suptitle(title, y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def print_table(grid, xname, xfmt="{:.4g}"):
    hdr = (f"{'run':>5} {xname:>10} {'n':>4} {'s_med':>8} {'p16':>8} {'p84':>8} "
           f"{'min':>8} {'max':>8}  "
           + " ".join(f"{r:>8}" for r in ROLES))
    print(hdr)
    for g, st in grid:
        x = g["x"]
        rs = st["roles"]
        print(f"{g['id']:5d} {xfmt.format(x):>10} {st['n']:4d} "
              f"{st['median']:8.4f} {st['p16']:8.4f} {st['p84']:8.4f} "
              f"{st['min']:8.4f} {st['max']:8.4f}  "
              + " ".join(f"{rs[r]:8.4f}" for r in ROLES))


def print_detail(grid, xname, xfmt="{:.4g}"):
    """Per-layer medians, then each run's layer×role grid."""
    layers = grid[0][1]["layers"]
    print(f"\n--- by layer (median of 4 roles) ---")
    print(f"{'run':>5} {xname:>10} " + " ".join(f"{'L'+str(L):>7}" for L in layers))
    for g, st in grid:
        print(f"{g['id']:5d} {xfmt.format(g['x']):>10} "
              + " ".join(f"{st['layer_med'][L]:7.4f}" for L in layers))

    for g, st in grid:
        print(f"\n--- by layer × role   run #{g['id']}  {xname}={xfmt.format(g['x'])} ---")
        print(f"{'layer':>6} " + " ".join(f"{r:>8}" for r in ROLES) + f" {'med':>8}")
        for L in layers:
            vals = [st["cell"].get((L, r), float("nan")) for r in ROLES]
            print(f"{L:6d} " + " ".join(f"{v:8.4f}" for v in vals)
                  + f" {st['layer_med'][L]:8.4f}")

    xs = [g["x"] for g, _ in grid]
    print("\n--- log-log slope by layer ---")
    print(f"{'layer':>6} {'slope':>8}  " + " ".join(f"{r:>8}" for r in ROLES))
    for L in layers:
        ys = [st["layer_med"][L] for _, st in grid]
        sl = fit_loglog_slope(xs, ys)
        role_sl = []
        for r in ROLES:
            rv = [st["cell"].get((L, r), float("nan")) for _, st in grid]
            role_sl.append(fit_loglog_slope(xs, rv) if not any(np.isnan(rv)) else float("nan"))
        print(f"{L:6d} {sl:8.3f}  " + " ".join(f"{s:8.3f}" for s in role_sl))


def role_divergence(rows):
    ratios = []
    for st in rows:
        vals = [st["roles"][r] for r in ROLES if not np.isnan(st["roles"][r])]
        if len(vals) >= 2:
            ratios.append(max(vals) / min(vals))
    return max(ratios) if ratios else 1.0
