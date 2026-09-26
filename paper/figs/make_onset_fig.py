#!/usr/bin/env python3
"""Onset distribution figure, built ONLY from numbers the paper already states.

The paper's central qualitative claim -- that the residual misses trace to onset
timing being heavy-tailed and truncated by a fixed budget -- is currently carried
by prose alone. Per-step EMA/cap telemetry for these runs is not recoverable from
this machine (it lives on the box that ran them), so this figure plots the onset
anchors the text reports rather than a reconstructed curve. Nothing here is
interpolated: each point is a value stated in the paper or its source notes.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 8.5,
    "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.edgecolor": "#666666", "axes.linewidth": 0.7,
    "xtick.color": "#666666", "ytick.color": "#666666", "text.color": "#222222",
    "axes.labelcolor": "#222222",
})

# (label, onset in steps, did it lock in, marker) -- all values from the paper.
# Onsets not stated in the text are omitted rather than guessed.
rows = [
    ("1.7B wikitext",            5_400, True,  "o"),
    ("1.7B pool-128 (late)",    13_000, True,  "o"),
    ("8B seed 1",               11_000, True,  "s"),
    ("8B seed 0",                6_500, True,  "s"),
    ("8B seed 2",               16_000, False, "s"),   # censored: no onset by 16k
    ("8B RWKV-7",               16_000, False, "^"),   # censored: no onset by 16k
]
BUDGET = 16_000

fig, ax = plt.subplots(figsize=(3.4, 2.0))
for i, (lab, x, ok, mk) in enumerate(sorted(rows, key=lambda r: r[1])):
    censored = not ok
    ax.plot([x], [i], marker=mk, ms=5.5, mfc=("#3b6ea5" if ok else "white"),
            mec=("#3b6ea5" if ok else "#b03030"), mew=1.3, ls="none", zorder=3)
    if censored:                      # arrow: onset had not arrived by the budget
        ax.annotate("", xy=(BUDGET + 2_200, i), xytext=(x + 350, i),
                    arrowprops=dict(arrowstyle="->", color="#b03030", lw=1.1))
        ax.text(BUDGET + 2_450, i, "no onset", fontsize=7, color="#b03030",
                ha="left", va="center")
ax.axvline(BUDGET, color="#888888", lw=0.9, ls="--", zorder=1)
ax.text(BUDGET - 350, -0.55, "16k budget", fontsize=7.5,
        color="#666666", ha="right", va="bottom")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in sorted(rows, key=lambda r: r[1])], fontsize=7)
ax.set_xlabel("step of retrieval onset")
ax.set_xlim(0, 19_500)
ax.set_ylim(-0.9, len(rows) - 0.35)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#eeeeee", lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_onset.pdf")
print("wrote", OUT / "fig_onset.pdf")
