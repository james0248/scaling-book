"""Chinchilla-paper plot style, applied globally on import.

rcParams mirror tiny/moe_optimization.ipynb. CURVE_CMAP colours training curves
by model size (their fig. 2), ISO_CMAP colours isoFLOP budgets (their fig. 3).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,  # cmr10 has no glyph for it
        "font.size": 9,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
    }
)

GRAY = "0.45"
RED = "#e34f4f"
TEAL = "#20807d"
FRONTIER_BLUE = "#3573a1"

CURVE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "chinchilla_curves", plt.get_cmap("gnuplot")(np.linspace(0.08, 1.0, 256))
)

ISO_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "chinchilla_iso",
    ["#A7D6A4", "#6DBCA2", "#46989E", "#347399", "#2C4E8A", "#22315E", "#141F3C"],
)


def human_format(x, pos=None):
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            v = x / div
            return f"{v:.0f}{suf}" if v == int(v) else f"{v:.1f}{suf}"
    return f"{x:.0f}"


def budget_label(c: float) -> str:
    return f"{c:.0e}".replace("e+", "e")


def notable_param_ticks(ax, lo: float, hi: float):
    """Human-readable major ticks at round parameter counts only"""
    cands = [2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7]
    ticks = [t for t in cands if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([human_format(t) for t in ticks])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def annotate_optimum(ax, x, y, label, color=TEAL):
    """Paper-style projection: a line from the left spine to (x, y), then down"""
    x0 = ax.get_xlim()[0]
    y0 = ax.get_ylim()[0]
    ax.plot([x0, x, x], [y, y, y0], color=color, lw=0.9, zorder=5)
    ax.annotate(label, (x0, y), xytext=(3, 3), textcoords="offset points",
                fontsize=8, color=color)
