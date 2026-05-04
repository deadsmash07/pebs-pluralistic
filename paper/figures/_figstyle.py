"""Shared figure-style helpers for the P1 Pluralistic figure suite.

Sets a single global rcParams baseline + provides Wong 2011 / Okabe-Ito
colorblind-safe categorical palette, plus small reusable helpers
(despine, panel-label, save_fig).

Reused across all P1 .py figure scripts so the suite has unified
typography, palette, and aspect-ratio discipline.
"""
from __future__ import annotations

import os
import matplotlib.pyplot as plt


# Wong 2011 / Okabe-Ito (8 colors, all colorblind-safe).
WONG = {
    "black":      "#000000",
    "orange":     "#E69F00",   # untrained / control / null
    "sky_blue":   "#56B4E9",
    "blu_green":  "#009E73",   # established / verdict-positive
    "yellow":     "#F0E442",
    "blue":       "#0072B2",   # ours / TRAINED-attribute / proposed
    "vermillion": "#D55E00",   # falsified / sign-flip
    "red_purple": "#CC79A7",
    "grey":       "#606060",   # axes / anchor / non-active
    "light_grey": "#B0B0B0",
}


def set_pub_style() -> None:
    """Apply the publication-quality rcParams baseline.

    All text >=7pt at the final NeurIPS/ICML 5.5in single-column or
    7.0in double-column print size.  Type-42 (TrueType) embedded
    fonts.  Top + right spines off by default.
    """
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset":  "stix",
        "font.size":         8,
        "axes.labelsize":    9,
        "axes.titlesize":    10,
        "axes.titleweight":  "bold",
        "axes.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    WONG["grey"],
        "axes.labelcolor":   "black",
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "xtick.color":       WONG["grey"],
        "ytick.color":       WONG["grey"],
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "legend.fontsize":   7.5,
        "legend.frameon":    False,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.03,
    })


def panel_label(ax, label: str, x: float = -0.10, y: float = 1.04,
                fontsize: int = 11) -> None:
    """Place an upper-left panel label A/B/C... in Nature/Science style."""
    ax.text(x, y, label, transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=fontsize, fontweight="bold", color="black")


def save_fig(fig, path: str) -> None:
    """Save with paper-grade defaults; print confirmation."""
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {path}")


def here(file: str) -> str:
    """Return absolute directory of the calling .py file."""
    return os.path.dirname(os.path.abspath(file))
