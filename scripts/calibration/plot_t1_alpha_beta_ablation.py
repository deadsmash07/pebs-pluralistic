"""Camera-ready Plot 4: T1 alpha-vs-beta ablation.

Two-metric bar chart: each of 4 arms (pop-slope, alpha-only, beta-only, both-full)
gets two bars side-by-side — relative RMSE improvement (%) and user-win rate (%).

The 'alpha-only' bar should clearly carry ~88 % of the full-PEBS lift, i.e.
intercept dominates slope.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
OUT = FIGS / "t1_alpha_beta_ablation.pdf"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
})


def main() -> None:
    data = json.loads((RESULTS / "t1_alpha_vs_beta_ablation.json").read_text())
    arms = data["arms"]

    order = [
        ("pop-slope",           "pop_slope",  "#8c8c8c"),
        ("alpha-only (intercept)", "alpha_only", "#1f77b4"),
        ("beta-only  (slope)",     "beta_only",  "#ff7f0e"),
        ("both (full PEBS)",   "both_full",  "#d62728"),
    ]

    labels  = [o[0] for o in order]
    pct     = [arms[o[1]]["rel_improvement_pct_vs_pop"] for o in order]
    winrate = [arms[o[1]]["user_win_rate"] * 100 for o in order]
    colors  = [o[2] for o in order]

    y = np.arange(len(order))[::-1]
    h = 0.36

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    b1 = ax.barh(y + h/2, pct, height=h, color=colors, edgecolor="white",
                 label="% RMSE improvement vs pop-slope")
    b2 = ax.barh(y - h/2, winrate, height=h, color=colors, edgecolor="white",
                 alpha=0.45, label="User-win rate (%)", hatch="//")

    for yy, v in zip(y + h/2, pct):
        ax.text(v + 0.25, yy, f"{v:.2f}%", va="center", fontsize=9)
    for yy, v in zip(y - h/2, winrate):
        ax.text(v + 0.25, yy, f"{v:.1f}%", va="center", fontsize=9, color="#555")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Metric value (%)")
    ax.set_xlim(0, 85)
    ax.set_title(
        "T1: Per-user $(\\alpha_j, \\beta_j)$ ablation — intercept dominates slope",
    )

    # Intercept-dominance annotation
    ratio = pct[1] / pct[3] * 100 if pct[3] > 0 else 0
    ax.annotate(
        f"alpha-only recovers {ratio:.1f}% of full-PEBS lift\n"
        f"beta-only recovers {pct[2]/pct[3]*100:.1f}%",
        xy=(pct[1] + 0.25, y[1] + h/2),
        xytext=(45, y[1] + 0.6),
        fontsize=9, color="#333",
        arrowprops=dict(arrowstyle="->", color="#333", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff5cc", ec="#d4a017", lw=0.7),
    )

    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="x", ls=":", color="#bbb", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUT, format="pdf", bbox_inches="tight", metadata={"Creator": "PEBS"})
    print(f"wrote: {OUT}")


if __name__ == "__main__":
    main()
