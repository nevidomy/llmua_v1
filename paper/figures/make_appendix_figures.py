"""Appendix figures F1, F2, I1 from the three CSV exports in figures/data/ (box exports of 2026-10-08).
Written to $OUT_DIR if set, otherwise to this folder (paper/figures/)."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "legend.frameon": False, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.3, "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight"})
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data") + os.sep; O = os.environ.get("OUT_DIR", HERE) + os.sep
os.makedirs(O, exist_ok=True)
CPT, LN2 = 4.0305, np.log(2)              # validation set characters per token (§2); nats -> bpc = /CPT/ln2
def despine(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
def save(fig, name): fig.savefig(O + name + ".pdf"); fig.savefig(O + name + ".png"); plt.close(fig)

# ---------------- F1: P2 seed 47 trunk with its three cooldowns ----------------
f1 = pd.read_csv(D + "F1_p2_seed47_trunk_cooldowns.csv"); f1["bpc"] = f1.val_loss / CPT / LN2; f1["Mtok"] = f1.tokens / 1e6
trunk = f1[f1.run_id == 45]; b20 = f1[f1.run_id == 46]; b40 = f1[f1.run_id == 47]
fig, ax = plt.subplots(figsize=(3.4, 2.5))
stable = trunk[trunk.Mtok <= 0.8 * 786.8]; cool80 = trunk[trunk.Mtok >= 0.8 * 786.8 - 1]
ax.plot(stable.Mtok, stable.bpc, color=BLUE, lw=1.3, label="trunk, constant rate (Muon 5.0e-3)")
ax.plot(cool80.Mtok, cool80.bpc, color=ORANGE, lw=1.3, label="cooldowns: linear decay to 0.1× over the last 20%")
ax.plot(b20.Mtok, b20.bpc, color=ORANGE, lw=1.3); ax.plot(b40.Mtok, b40.bpc, color=ORANGE, lw=1.3)
for b, lab, h in [(b20, "20×", 196.7), (b40, "40×", 393.4), (cool80, "80×", 786.8)]:
    ax.scatter([b.Mtok.iloc[-1]], [b.bpc.iloc[-1]], s=22, color=ORANGE, zorder=5)
    ax.annotate(f"{lab}: {b.bpc.iloc[-1]:.4f}", (b.Mtok.iloc[-1], b.bpc.iloc[-1]), xytext=(4, -9 if lab != "80×" else 3), textcoords="offset points", fontsize=6.8, color=INK2, ha="left" if lab != "80×" else "right")
for x0 in (0.8 * 196.7, 0.8 * 393.4, 0.8 * 786.8): ax.axvline(x0, color=GRID, lw=0.6)
ax.set_xlim(0, 800); ax.set_ylim(1.32, 1.57); ax.set_xlabel("tokens (M)"); ax.set_ylabel("monitor loss (bpc, 32-window subset)"); despine(ax)
ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.2)
save(fig, "figF1_trunk_cooldowns")

# ---------------- F2: junk test at P4 ----------------
f2 = pd.read_csv(D + "F2_p4_junk_buckets.csv")
labels = ["leading\nsegment"] + [f"{lo}" if lo == hi else (f"{lo}–{hi}" if hi < 100 else f"{lo}–\n{hi}") for lo, hi in zip(f2.junk_lo[1:], f2.junk_hi[1:])]
x = np.arange(len(f2))
fig, ax = plt.subplots(figsize=(6.8, 2.6))
ax.errorbar(x, f2.delta_nats, yerr=f2.se_nats, fmt="o", color=BLUE, ms=3.8, capsize=2, lw=1, zorder=4)
ax.axhline(0, color=AXIS, lw=0.8)
ax.axhspan(0.014, 0.020, color=BLUE, alpha=0.08, lw=0)
ax.annotate("plateau 0.014–0.020 nats/token\n(0.005–0.007 bpc)", (3, 0.020), xytext=(0, 5), textcoords="offset points", fontsize=6.6, color=INK2)
for xi, n in zip(x, f2.n_tokens):
    ax.annotate(f"{n/1e6:.2g}M" if n >= 1e6 else f"{n/1e3:.0f}k", (xi, -0.004), ha="center", fontsize=6.3, color=MUTED)
ax.annotate("tokens scored per bucket", (-0.4, -0.0075), fontsize=6.3, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
ax.set_xlabel("foreign tokens (from previous documents in the tile) before the scored token")
ax.set_ylabel("masked − causal evaluation (nats/token)"); ax.set_ylim(-0.009, 0.037); despine(ax)
ax.set_title("causal-trained P4 (seed 47), full validation set", fontsize=7.5, color=INK2, loc="left")
save(fig, "figF2_junk_test")

# ---------------- I1: hidden-matrix RMS vs Muon learning rate ----------------
i1 = pd.read_csv(D + "I1_rms_vs_muon_lr.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.4))
slopes = {}
for m, col, mk in [("P2", BLUE, "o"), ("P3", AQUA, "s")]:
    d = i1[i1.model == m]; k, b = np.polyfit(np.log(d.muon_lr), np.log(d.s), 1); slopes[m] = k
    xx = np.logspace(np.log10(d.muon_lr.min() / 1.2), np.log10(d.muon_lr.max() * 1.2), 50)
    ax.plot(xx, np.exp(b) * xx ** k, color=col, lw=1.0)
    ax.scatter(d.muon_lr, d.s, s=22, marker=mk, color=col, zorder=4, label=f"{m}: fitted slope {k:.2f}")
ref = 0.5; x0, y0 = 0.004, 0.72; xx = np.array([0.0025, 0.013]); ax.plot(xx, y0 * (xx / x0) ** ref, color=MUTED, lw=0.9, ls=(0, (3, 2)), label="slope 0.5: ‖W‖ ∝ √(η/λ), decay equilibrium")
ax.axvline(5.45e-3, color=BLUE, lw=0.6, ls=(0, (1, 2))); ax.axvline(3.63e-3, color=AQUA, lw=0.6, ls=(0, (1, 2)))
ax.annotate("20× vertices (P3, P2)", (4.4e-3, 2.95), ha="center", va="top", fontsize=6.5, color=INK2)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1.6e-3, 5.2e-2); ax.set_ylim(0.6, 3.2)
ax.xaxis.set_major_locator(FixedLocator([2e-3, 5e-3, 1e-2, 2e-2, 4e-2])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*1e3:g}e-3")); ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_locator(FixedLocator([0.6, 1, 1.5, 2, 3])); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("Muon learning rate"); ax.set_ylabel("hidden-matrix RMS, s (1 = init)"); despine(ax)
ax.legend(loc="lower right", handlelength=1.2, borderaxespad=0.2)
save(fig, "figI1_rms_vs_lr")
print("slopes", slopes)
