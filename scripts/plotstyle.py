"""Shared publication figure style (single source of truth).

Import and call ``use()`` at the top of every figure script so all panels share
one typographic and colour system.  ``PALETTE`` and ``REG_COLOR`` give the
semantic colours used throughout the paper.
"""
import matplotlib as mpl

# --- cohesive, print-safe palette -----------------------------------------
PALETTE = {
    "ink":      "#1a1a1a",   # text / axes
    "blue":     "#1f5fa8",   # community 1 / primary
    "orange":   "#d2691e",   # community 2 / secondary
    "green":    "#2a7a36",   # efficient / corrective / good news
    "red":      "#b2182b",   # wrong / distortive / bad news
    "purple":   "#6a51a3",   # polarised
    "grey":     "#9e9e9e",   # unresolved / neutral
    "cyan":     "#1a9aa8",   # accent
}
REG_COLOR = {"efficient": PALETTE["green"], "wrong": PALETTE["red"],
             "polarised": PALETTE["purple"], "unresolved": PALETTE["grey"]}


def use():
    """Apply the shared rcParams (idempotent)."""
    mpl.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 400,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        # typography
        "font.family": "serif", "mathtext.fontset": "cm",
        "font.size": 9.5, "axes.titlesize": 10.0, "axes.labelsize": 9.5,
        "axes.titlepad": 6.0, "axes.labelpad": 3.0,
        "legend.fontsize": 8.0, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "figure.titlesize": 10.5,
        # axes / spines
        "axes.edgecolor": PALETTE["ink"], "axes.linewidth": 0.8,
        "axes.labelcolor": PALETTE["ink"], "text.color": PALETTE["ink"],
        "xtick.color": PALETTE["ink"], "ytick.color": PALETTE["ink"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "medium",
        # ticks
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        # grid
        "axes.grid": True, "grid.color": PALETTE["ink"],
        "grid.alpha": 0.14, "grid.linewidth": 0.6,
        # lines / legend
        "lines.linewidth": 1.9, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.4, "legend.labelspacing": 0.3,
        "patch.linewidth": 0.6,
    })


def panel_tag(ax, s, dx=0.0, dy=0.0):
    """Bold panel label in the top-left corner, outside the data area."""
    ax.text(-0.02 + dx, 1.04 + dy, s, transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", va="bottom", ha="right")
