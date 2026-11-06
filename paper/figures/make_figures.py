"""Main-text figures for LLMUA Paper 1 (rev4). All numbers are the paper's; nothing is fit here
except the three external rays of Fig. 3, refit from the table values with the paper's estimator.

Outputs: fig1_ray.{pdf,png}, fig2a_horizons.{pdf,png}, fig2b_beta.{pdf,png}, fig3_instruments.{pdf,png}
written to $OUT_DIR if set, otherwise to this folder (paper/figures/).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from scipy import optimize

OUT = os.environ.get("OUT_DIR", os.path.dirname(os.path.abspath(__file__))) + os.sep

# ---------- style (print, single/double column) ----------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated slots 1-3 (all-pairs CVD-safe)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "legend.frameon": False,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.4, "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
})
M = 1e6
N = np.array([5326464, 9833664, 16422144, 33826176], float)      # N_total, Table 1
NT = 124278528.0
rung_names = ["P1", "P2", "P3", "P4"]

def ntick(ax, extra=()):
    ticks = [5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8, 1e9]
    ax.set_xscale("log"); ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/M:g}M" if v < 1e9 else "1B"))
    ax.set_xlabel("parameters, N_total")

def despine(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)

def fit_ray(Nv, L):
    f = lambda p, x: p[0] + p[1] * x ** (-p[2]); best = None
    for a0 in [0.5, 0.6, 0.7, 0.8, 1.0, 1.2]:
        for b0 in [0.2, 0.29, 0.35, 0.45]:
            s = optimize.least_squares(lambda p: f(p, Nv) - L, [a0, (L[0] - a0) * Nv[0] ** b0, b0],
                                       max_nfev=20000, xtol=1e-14, ftol=1e-14)
            if best is None or s.cost < best.cost: best = s
    return best.x, f

def save(fig, name):
    fig.savefig(OUT + f"{name}.pdf"); fig.savefig(OUT + f"{name}.png"); plt.close(fig)

# =====================================================================
# Figure 1 — the 20x ray, the four rungs, the LOO range, the target's two 20x points, the Part-2 forecast
# =====================================================================
L20 = np.array([1.50010, 1.36550, 1.27222, 1.15967])            # masked 20x means (Table 3 / §4.1)
ray = lambda n: 0.6803 + 73.16 * n ** (-0.290)                  # registered four-rung ray (§4.2)
loo_t = (1.0022, 1.0266)                                        # leave-one-out range at the target
branch, own, reg = 1.0062, 1.0005, 1.0091
refit = lambda n: 0.6188 + 53.33 * n ** (-0.2650)               # like-for-like five-rung refit (§8)
fc = {5e8: (0.8831, 0.878, 0.902), 1e9: (0.8388, 0.831, 0.862)} # forecast, LOO

fig, ax = plt.subplots(figsize=(3.4, 2.7))
xx = np.logspace(np.log10(4.5e6), np.log10(1.3e9), 300)
ax.plot(xx, ray(xx), color=BLUE, lw=1.2, label="registered four-rung ray", zorder=2)
ax.plot(xx[xx >= 1.2e8], refit(xx[xx >= 1.2e8]), color=BLUE, lw=1.0, ls=(0, (3, 2)), label="five-rung refit (§8 forecast)", zorder=2)
ax.scatter(N, L20, s=22, color=BLUE, zorder=4, label="ladder rungs, 20× (masked)")
for n, l, nm in zip(N, L20, rung_names): ax.annotate(nm, (n, l), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
ax.plot([NT, NT], loo_t, color=MUTED, lw=2.2, solid_capstyle="butt", alpha=0.45, zorder=3)
ax.scatter([NT], [reg], s=30, marker="D", facecolor="white", edgecolor=BLUE, lw=1.0, zorder=5, label="registered prediction 1.0091")
ax.scatter([NT], [branch], s=30, marker="s", facecolor="white", edgecolor=ORANGE, lw=1.1, zorder=6, label="measured, branch of the 80×-rate trunk 1.0062")
ax.scatter([NT], [own], s=34, marker="^", color=ORANGE, zorder=7, label="measured, own 20× rate 1.0005")
for n, (c, lo, hi) in fc.items():
    ax.plot([n, n], [lo, hi], color=MUTED, lw=2.2, alpha=0.45, zorder=3, solid_capstyle="butt")
    ax.scatter([n], [c], s=26, marker="o", facecolor="white", edgecolor=BLUE, lw=1.0, zorder=5)
ax.annotate("124M target (see inset)", (NT, 1.0266), xytext=(9, 3), textcoords="offset points", ha="left", fontsize=7, color=INK2)
ax.annotate("500M / 1B forecast,\nleave-one-out range", (5e8, 0.902), xytext=(4, 5), textcoords="offset points", ha="left", va="bottom", fontsize=7, color=INK2)
ntick(ax); ax.set_xlim(4.5e6, 1.3e9); ax.set_ylim(0.80, 1.56); ax.set_ylabel("validation loss at 20 tokens/parameter (bpc)")
despine(ax)
# inset: the target's three 20x points and the leave-one-out range
zi = ax.inset_axes([0.08, 0.08, 0.30, 0.34])
xz = np.linspace(1.0e8, 1.9e8, 50)
zi.plot(xz, ray(xz), color=BLUE, lw=1.2); zi.plot(xz, refit(xz), color=BLUE, lw=1.0, ls=(0, (3, 2)))
zi.plot([NT, NT], loo_t, color=MUTED, lw=4, alpha=0.4, solid_capstyle="butt")
zi.scatter([NT], [reg], s=34, marker="D", facecolor="white", edgecolor=BLUE, lw=1.0, zorder=5)
zi.scatter([NT], [branch], s=34, marker="s", facecolor="white", edgecolor=ORANGE, lw=1.1, zorder=6)
zi.scatter([NT], [own], s=38, marker="^", color=ORANGE, zorder=7)
zi.annotate("leave-one-out\nrange", (NT, 1.0266), xytext=(6, -9), textcoords="offset points", fontsize=6.2, color=INK2)
zi.set_xlim(1.0e8, 1.8e8); zi.set_ylim(0.994, 1.031); zi.set_xticks([]); zi.set_yticks([1.00, 1.02]); zi.tick_params(labelsize=6, length=1.5)
zi.grid(True, lw=0.4)
for sp in ("top", "right"): zi.spines[sp].set_visible(False)
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.22), handlelength=1.6, columnspacing=1.2)
save(fig, "fig1_ray")

# =====================================================================
# Figure 2a — the target's three horizons: registered vs measured; own-rate 20x; 40x correction; log-linear alternative
# =====================================================================
hz = np.array([20, 40, 80]); regv = [1.0091, 0.9858, 0.9704]; meas = [1.0062, 0.9747, 0.9549]
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(hz, regv, color=BLUE, lw=1.0, ls=(0, (3, 2)), zorder=2)
ax.scatter(hz, regv, s=30, marker="D", facecolor="white", edgecolor=BLUE, lw=1.0, zorder=4, label="registered prediction")
ax.plot(hz, meas, color=ORANGE, lw=1.0, zorder=2)
ax.scatter(hz, meas, s=30, marker="s", facecolor="white", edgecolor=ORANGE, lw=1.1, zorder=5, label="measured, branches of the 80×-rate trunk")
ax.scatter([20], [own], s=36, marker="^", color=ORANGE, zorder=6, label="measured, 20× at its own rate")
ax.plot([40, 40], [0.9727, 0.9737], color=ORANGE, lw=2.2, alpha=0.5, solid_capstyle="butt", zorder=3)
ax.annotate("model-based\n40× correction", (40, 0.9727), xytext=(-6, -16), textcoords="offset points", ha="right", fontsize=6.8, color=INK2)
ax.annotate("−0.3% (branch)", (20, branch), xytext=(8, 2), textcoords="offset points", fontsize=6.8, color=INK2)
ax.annotate("−0.85% (own rate)", (20, own), xytext=(-3, -12), textcoords="offset points", fontsize=6.8, color=INK2)
ax.annotate("−1.1%", (40, 0.9747), xytext=(7, 1), textcoords="offset points", fontsize=6.8, color=INK2)
ax.annotate("−1.6%", (80, 0.9549), xytext=(7, -2), textcoords="offset points", fontsize=6.8, color=INK2)
ax.set_xscale("log", base=2); ax.xaxis.set_major_locator(FixedLocator([20, 40, 80])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}×"))
ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xlim(17, 110); ax.set_ylim(0.945, 1.016)
ax.set_xlabel("tokens per parameter"); ax.set_ylabel("validation loss (bpc)"); despine(ax)
h, l = ax.get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.20), handlelength=1.4)
save(fig, "fig2a_horizons")

# =====================================================================
# Figure 2b — beta per rung with SE; target range; corrected P2/P3 as hollow markers
# =====================================================================
beta = np.array([0.546, 0.605, 0.646, 0.637]); se = np.array([0.028, 0.005, 0.019, 0.021])
fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.errorbar(N, beta, yerr=se, fmt="o", color=BLUE, ms=4.5, capsize=2, lw=1, zorder=4, label="per-rung β, as measured (± seed SE)")
ax.scatter([N[1], N[2]], [0.57, 0.48], s=30, facecolor="white", edgecolor=BLUE, lw=1.0, zorder=5, label="P2, P3 with the 80× branch detuning removed (§4.4)")
ax.plot([NT, NT], [0.52, 0.60], color=ORANGE, lw=5, alpha=0.35, solid_capstyle="butt", zorder=3)
ax.scatter([NT], [0.54], s=34, marker="^", color=ORANGE, zorder=6, label="target: 0.54 (0.52–0.60 across the 40× correction)")
ax.scatter([NT, NT], [0.38, 0.67], s=22, marker="_", color=ORANGE, lw=1.4, zorder=6)
ax.annotate("0.67: three branches as measured", (NT, 0.67), xytext=(-6, 3), textcoords="offset points", ha="right", fontsize=6.8, color=INK2)
ax.annotate("0.38: 40× branch uncorrected", (NT, 0.38), xytext=(-6, 3), textcoords="offset points", ha="right", fontsize=6.8, color=INK2)
for n, b, nm in zip(N, beta, rung_names): ax.annotate(nm, (n, b), xytext=(5, 4), textcoords="offset points", fontsize=7, color=INK2)
ntick(ax); ax.set_xlim(4.5e6, 2.0e8); ax.set_ylim(0.33, 0.72); ax.set_ylabel("data exponent β (20×→80×)"); despine(ax)
h, l = ax.get_legend_handles_labels(); order = [2, 0, 1]
fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.30), handlelength=1.4)
save(fig, "fig2b_beta")

# =====================================================================
# Figure 3 — two panels: (a) three instruments at 80x with rays and target points; (b) the two protocol offsets vs N
# =====================================================================
inst = {
    "val (tiles)":        (np.array([1.44138, 1.30838, 1.21866, 1.11020]), 0.9549, (0.958, 0.995), BLUE, "o"),
    "BrUK documents":     (np.array([1.5395, 1.4131, 1.3271, 1.2234]), 1.0756, (1.078, 1.109), AQUA, "s"),
    "BrUK sentences":     (np.array([1.7189, 1.6096, 1.5372, 1.4478]), 1.3267, (1.318, 1.364), ORANGE, "D"),
}
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8), gridspec_kw=dict(width_ratios=[1.55, 1], wspace=0.32))
xx = np.logspace(np.log10(4.5e6), np.log10(1.0e9), 300)
for name, (L, tgt, loo, col, mk) in inst.items():
    p, f = fit_ray(N, L)
    ax.plot(xx, f(p, xx), color=col, lw=1.1, zorder=2)
    ax.scatter(N, L, s=20, marker=mk, color=col, zorder=4)
    ax.plot([NT, NT], loo, color=col, lw=2.6, alpha=0.4, solid_capstyle="butt", zorder=3)
    ax.scatter([NT], [tgt], s=30, marker=mk, facecolor="white", edgecolor=col, lw=1.1, zorder=5)
    if name == "val (tiles)": ax.annotate(name, (N[1], L[1]), xytext=(-7, -9), textcoords="offset points", ha="right", fontsize=7, color=col)
    else: ax.annotate(name, (N[0], L[0]), xytext=(7, 5), textcoords="offset points", fontsize=7, color=col)
    ax.annotate(f"{100*(tgt-f(p,NT))/f(p,NT):+.2f}%", (NT, tgt), xytext=(7, -2), textcoords="offset points", fontsize=6.8, color=INK2)
ax.scatter([774e6], [1.4665], s=34, marker="*", color=INK2, zorder=6)
ax.annotate("uk4b Large (774M)\non the same sentences: 1.4665", (774e6, 1.4665), xytext=(6, 7), textcoords="offset points", ha="right", va="bottom", fontsize=6.8, color=INK2)
ntick(ax); ax.set_xlim(4.5e6, 1.0e9); ax.set_ylim(0.90, 1.80); ax.set_ylabel("loss at 80 tokens/parameter (bpc)"); despine(ax)
ax.set_title("(a) three instruments, one ray each", fontsize=8, color=INK2, loc="left")
# (b) offsets vs N
Nall = np.append(N, NT)
s_d = np.array([0.1794, 0.1965, 0.2101, 0.2244, 0.2511]); d_v = np.array([0.0981, 0.1047, 0.1084, 0.1132, 0.1207])
ax2.plot(Nall, s_d, color=ORANGE, marker="D", ms=3.5, lw=1.1, label="sentence − document")
ax2.plot(Nall, d_v, color=AQUA, marker="s", ms=3.5, lw=1.1, label="document − val")
ax2.axhline(0.087, color=MUTED, lw=0.8, ls=(0, (2, 2)))
ax2.annotate("first-token cost, 9.2 bits ≈ 0.087 bpc (inside the sentence offset)", (5.3e6, 0.087), xytext=(2, -11), textcoords="offset points", fontsize=6.3, color=INK2)
ax2.annotate("sentence − document", (Nall[-1], s_d[-1]), xytext=(0, 7), textcoords="offset points", ha="right", fontsize=7, color=ORANGE)
ax2.annotate("document − val", (Nall[-1], d_v[-1]), xytext=(0, 7), textcoords="offset points", ha="right", fontsize=7, color=AQUA)
ntick(ax2); ax2.set_xlim(4.5e6, 2.0e8); ax2.set_ylim(0.05, 0.30); ax2.set_ylabel("protocol offset (bpc)"); despine(ax2)
ax2.set_title("(b) offsets between protocols", fontsize=8, color=INK2, loc="left")
save(fig, "fig3_instruments")
print("done")
