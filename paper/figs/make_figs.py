#!/usr/bin/env python3
"""Generate the two paper figures from the numbers in Tables 1/2/5 (main.tex).

Fig 1: the three single-knob toggles at K=32 (conv / transition / decay).
Fig 2: capacity curve (K-sweep) vs. the interference wall and its removal.

All values transcribed from GDN_PAPER.md sections 4.1-4.5 (3-seed means; the
curriculum cells are 1-seed and marked as such in the paper).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": "#666666",
    "axes.linewidth": 0.7,
    "xtick.color": "#666666",
    "ytick.color": "#666666",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "pdf.fonttype": 42,
})

WITHOUT, WITH = "#7FC0AF", "#0E6A5B"     # light/dark single hue: ingredient off/on
INKSOFT = "#555555"

# ------------------------------------------------------------------ figure 1
fig, axes = plt.subplots(
    1, 3, figsize=(6.3, 2.35), width_ratios=[2, 1.15, 1.15], sharey=True)

def toggle_panel(ax, groups, title, floor=0.068):
    """groups: list of (label, without, with_)"""
    xs, w = [], 0.36
    for i, (label, lo, hi) in enumerate(groups):
        x = i * 1.15
        ax.bar(x - w/2 - 0.01, lo, w, color=WITHOUT, zorder=3)
        ax.bar(x + w/2 + 0.01, hi, w, color=WITH, zorder=3)
        for xx, v in ((x - w/2 - 0.01, lo), (x + w/2 + 0.01, hi)):
            ax.text(xx, v + 0.025, f"{v:.2f}".lstrip("0") if v < 1 else "1.0",
                    ha="center", va="bottom", fontsize=7.2, color=INKSOFT)
        d = hi - lo
        ax.annotate(f"{d:+.2f}", (x, max(lo, hi) + 0.115),
                    ha="center", fontsize=8, fontweight="bold",
                    color=WITH if d > 0 else "#8C3B22")
        xs.append((x, label))
    ax.axhline(floor, ls=(0, (3, 2)), lw=0.7, color="#999999", zorder=2)
    ax.set_xticks([x for x, _ in xs])
    ax.set_xticklabels([l for _, l in xs])
    ax.set_title(title, pad=4)
    ax.set_ylim(0, 1.24)
    ax.set_xlim(min(x for x, _ in xs) - 0.75, max(x for x, _ in xs) + 0.75)
    ax.grid(axis="y", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

toggle_panel(axes[0],
             [("delta-rule cell", 0.446, 0.993), ("diagonal cell", 0.172, 0.641)],
             "short conv off $\\rightarrow$ on")
toggle_panel(axes[1], [("RWKV-7 cell", 0.359, 0.639)],
             "rank-1 term off $\\rightarrow$ on")
# decay: 'without' = deltanet 0.769, 'with' = gated_deltanet 0.446
toggle_panel(axes[2], [("delta-rule cell", 0.769, 0.446)],
             "decay off $\\rightarrow$ on")
axes[0].set_ylabel("masked recall @ $K{=}32$")
axes[0].text(1.72, 0.095, "no-binding\nfloor", fontsize=6.2, color="#999999",
             ha="right", va="bottom", linespacing=1.1)

fig.tight_layout(w_pad=1.6)
fig.savefig(OUT / "fig_decomposition.pdf")
plt.close(fig)

# ------------------------------------------------------------------ figure 2
fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.3, 2.55), width_ratios=[1.15, 1])

K = [8, 16, 32]
sweep = [  # label, values, color, marker, right-label y-offset
    ("armed GDN",            [1.000, 1.000, 0.993], "#007A5E", "o",  0.000),
    ("DeltaNet",             [1.000, 1.000, 0.769], "#0062A3", "s",  0.000),
    ("Mamba-2 ref",          [1.000, 0.758, 0.641], "#B77800", "^",  0.012),
    ("Gated DeltaNet",       [1.000, 0.970, 0.446], "#BE4B1A", "D",  0.000),
    ("RWKV-7 diag ablation", [1.000, 0.835, 0.359], "#A5588E", "v",  0.000),
    ("Mamba-2 ref, no conv", [0.582, 0.230, 0.172], "#5E5FA8", "x", -0.012),
]
for label, ys, c, m, dy in sweep:
    axL.plot(K, ys, color=c, lw=1.6, marker=m, ms=3.4,
             markerfacecolor="white", markeredgewidth=1.1, zorder=3)
    axL.annotate(label, (32.8, ys[-1] + dy), fontsize=6.8, color=c,
                 va="center", ha="left")
axL.set_xscale("log", base=2)
axL.set_xticks(K)
axL.set_xticklabels([str(k) for k in K])
axL.minorticks_off()
axL.set_xlim(7.2, 62)
axL.set_ylim(0, 1.06)
axL.set_xlabel("key–value pairs $K$")
axL.set_ylabel("masked recall")
axL.set_title("load: graceful capacity decline", pad=4)
axL.grid(axis="y", lw=0.4, color="#DDDDDD", zorder=0)
axL.set_axisbelow(True)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

regimes = [
    ("naive",           0.021, "#CFE5DE"),
    ("dense\nonly",     0.021, "#9CCCBE"),
    ("curric.\nonly",   0.822, "#4E9483"),
    ("dense+\ncurric.", 1.000, "#0E6A5B"),
]
xs = range(len(regimes))
for x, (lab, v, c) in zip(xs, regimes):
    axR.bar(x, v, 0.62, color=c, zorder=3)
    axR.text(x, v + 0.028, f"{v:.3f}".rstrip("0").rstrip(".") if v < 1 else "1.000",
             ha="center", va="bottom", fontsize=7.2, color=INKSOFT)
axR.bar(len(regimes) + 0.35, 1.000, 0.62, facecolor="white",
        edgecolor="#888888", hatch="////", lw=0.8, zorder=3)
axR.text(len(regimes) + 0.35, 1.028, "1.000", ha="center", va="bottom",
         fontsize=7.2, color=INKSOFT)
axR.axhline(0.019, ls=(0, (3, 2)), lw=0.7, color="#999999", zorder=2)
axR.set_xticks(list(xs) + [len(regimes) + 0.35])
axR.set_xticklabels([lab for lab, _, _ in regimes] + ["attention\n(ref.)"],
                    fontsize=6.6)
axR.set_ylim(0, 1.13)
axR.set_title("distance: the wall and its removal", pad=4)
axR.grid(axis="y", lw=0.4, color="#DDDDDD", zorder=0)
axR.set_axisbelow(True)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

fig.tight_layout(w_pad=2.2)
fig.savefig(OUT / "fig_wall_curriculum.pdf")
print("wrote", OUT / "fig_decomposition.pdf", "and", OUT / "fig_wall_curriculum.pdf")
