"""Capstone paper figure: Track 3's bias-robustness story in one image.

Two panels:
  1. Simpson's stress (β_true=0): detector FPR bar chart
     MixedLM 0%, NaiveOLS 100%, PageHinkley 100%
  2. Mixed-drift: scatter of ML estimate vs NaiveOLS estimate,
     color-coded by true drift. Diagonal = perfect calibration; horizontal
     line at 0 = zero detected drift. The sign-flip cases (ML positive,
     NaiveOLS negative) cluster in one quadrant.

These two results together form Track 3's bias-robustness argument:
  - Pure composition shift → naive method false-fires 100% (Panel 1)
  - Composition + real positive drift → naive method reports WRONG SIGN
    100% of the time (Panel 2)

References
----------
- Simpson 1951
- Pinheiro & Bates 2000 §2 (random-effects composition control)
- Our prior: track3_simpson_stress_validated.md + track3_mixed_drift_sign_flip.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--simpson-json", default="results/track3_simpson_stress/simpson_stress.json")
    p.add_argument("--mixed-json", default="results/track3_mixed_drift/mixed_drift.json")
    p.add_argument("--output", default="results/figs/track3_bias_robustness_capstone.png")
    return p.parse_args()


def main():
    args = parse_args()
    simpson = json.loads(Path(args.simpson_json).read_text())
    mixed = json.loads(Path(args.mixed_json).read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Simpson's stress FPR
    ax = axes[0]
    detectors = ["MixedLM\n+ permutation", "NaiveOLS\n(monthly aggregate)", "PageHinkley\n(monthly aggregate)"]
    fprs = [simpson["fpr"]["mixedlm_perm"],
            simpson["fpr"]["naive_ols"],
            simpson["fpr"]["pagehinkley"]]
    colors = ["#3b8ea5", "#d97757", "#999999"]
    bars = ax.bar(detectors, [100 * f for f in fprs], color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.6)
    ax.axhline(5.0, color="red", linestyle="--", linewidth=1.2, label="α = 5%")
    for bar, f in zip(bars, fprs):
        lab = f"{100*f:.0f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, 100*f + 2,
                lab, ha="center", va="bottom", fontsize=11, fontweight="bold")
    n_seeds = simpson["config"]["n_seeds"]
    ax.set_ylabel(f"False-positive rate (n={n_seeds} seeds)")
    ax.set_ylim(0, 115)
    ax.set_title("Composition shift, zero true drift:\nwhich detectors false-fire?",
                 fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)
    # Verdict annotation
    ax.text(0.5, 0.50,
            "MixedLM correctly reports null.\n"
            "Naive aggregation detectors fire\n"
            "in every single simulation seed.",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="#333",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8e7",
                      edgecolor="#c07030", alpha=0.85))

    # Panel 2: Mixed drift — ML estimate vs Naive estimate, color by true drift
    ax = axes[1]
    for cell in mixed["cells"]:
        beta_true = cell["true_drift"]
        ml_slopes = cell["mixedlm_slope_estimates"]
        naive_slopes = cell["naive_monthly_slopes"]
        # Filter NaN
        ml_slopes = [m for m in ml_slopes if m is not None and not (isinstance(m, float) and np.isnan(m))]
        naive_slopes = [n for n in naive_slopes if n is not None and not (isinstance(n, float) and np.isnan(n))]
        # Color by true drift
        color = {-0.005: "#d44a4a", 0.0: "#444444", 0.005: "#3b8ea5"}.get(beta_true, "#888")
        label = f"β_true = {beta_true:+.4f}"
        ax.scatter(naive_slopes, ml_slopes, s=70, color=color, alpha=0.75,
                   edgecolors="black", linewidth=0.4, label=label)
        # Annotate means
        mean_ml = float(np.mean(ml_slopes))
        mean_naive = float(np.mean(naive_slopes))
        ax.scatter([mean_naive], [mean_ml], s=200, color=color, marker="*",
                   edgecolors="black", linewidth=1.2)

    # Diagonals + zero lines
    lims = (-0.035, 0.015)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.plot([-0.04, 0.02], [-0.04, 0.02], color="black", linewidth=0.8,
            linestyle="--", label="perfect (ML = naive)")
    ax.set_xlim(lims)
    ax.set_ylim(-0.015, 0.010)
    ax.set_xlabel("NaiveOLS estimate of drift (pts/mo)")
    ax.set_ylabel("MixedLM estimate of drift (pts/mo)")
    ax.set_title("Composition shift + real drift:\nwho gets the direction right?",
                 fontsize=12)
    # Quadrant annotation for the sign-flip region
    ax.text(-0.020, 0.006,
            "NaiveOLS: negative\nML: POSITIVE\n(sign-flip region)",
            fontsize=9, color="#3b8ea5", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f4f8",
                      edgecolor="#3b8ea5", alpha=0.9))
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.suptitle("Track 3: bias-robustness of the MixedLM+permutation detector",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"[save] {args.output}")
    svg_path = Path(args.output).with_suffix(".svg")
    plt.savefig(svg_path, bbox_inches="tight")
    print(f"[save] {svg_path}")


if __name__ == "__main__":
    main()
