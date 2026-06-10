"""Research-grade n=5 forest plot of per-seed BT-LL deltas (PEBS - vanilla)
with paired-t 95% CI and across-seed summary diamond.

Reads the output of aggregate_7b_ppo_results.py (aggregate_n5.json) and writes a
single-panel forest plot as PDF.

Example:
  python scripts/plot_t1_bt_ll_n5_forest.py \
    --aggregate results/track1_ppo_7b/aggregate_n5.json \
    --output-pdf results/track1_ppo_7b/figure_n5_bt_ll_forest.pdf
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as tdist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", required=True)
    ap.add_argument("--output-pdf", required=True)
    args = ap.parse_args()

    agg = json.loads(Path(args.aggregate).read_text())
    per_seed = agg["per_seed"]
    seeds = [row["seed"] for row in per_seed]
    deltas = np.array([row["delta_bt_ll"] for row in per_seed])
    # Per-seed SE not in aggregate; approximate with pooled SE from across-seed
    # spread (upper bound — conservative).  Show dots only + pooled diamond.
    n = len(deltas)
    mean_delta = float(np.mean(deltas))
    sd = float(np.std(deltas, ddof=1)) if n >= 2 else float("nan")
    se = sd / math.sqrt(n) if n >= 2 else float("nan")
    tq = float(tdist.ppf(0.975, df=n - 1)) if n >= 2 else float("nan")
    ci_lo = mean_delta - tq * se
    ci_hi = mean_delta + tq * se
    p = agg.get("paired_t_across_seeds_bt_ll", {}).get("p", float("nan"))
    wp = agg.get("wilcoxon_across_seeds_bt_ll", {}).get("p", float("nan"))

    # Sort seeds by delta for legibility
    order = np.argsort(deltas)
    seeds_s = [seeds[i] for i in order]
    deltas_s = deltas[order]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    y = np.arange(n)
    ax.axvline(0.0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.scatter(deltas_s * 1e4, y, marker="o", s=70, c="#1f4e79", zorder=3,
               edgecolor="white", lw=1.2)
    for yi, d in zip(y, deltas_s):
        ax.plot([0, d * 1e4], [yi, yi], color="#bbbbbb", lw=0.6, zorder=1)

    # Pooled diamond
    dy = n
    ax.plot([ci_lo * 1e4, ci_hi * 1e4], [dy, dy], color="#c0392b", lw=2.2)
    ax.scatter([mean_delta * 1e4], [dy], marker="D", s=110, c="#c0392b",
               zorder=4, edgecolor="white", lw=1.2)

    ax.set_yticks(list(y) + [dy])
    ax.set_yticklabels([f"seed {s}" for s in seeds_s] + ["pooled (n=5)"])
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ BT-LL = PEBS $-$ vanilla (×10$^{-4}$ nats)")
    title = (f"T1.MI BT-LL forest (n={n} seeds, 500 held-out pairs/seed)\n"
             f"pooled $\\Delta$ = {mean_delta:+.2e}, 95% CI [{ci_lo:+.2e}, {ci_hi:+.2e}]\n"
             f"paired-t p = {p:.3f}, Wilcoxon p = {wp:.3f} -- null consistent with "
             r"$\mathcal{T}_1$ monotone invariance")
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    mde_80 = 2.8 * se if n >= 2 else float("nan")
    if math.isfinite(mde_80):
        ax.text(0.02, 0.02, f"MDE@80% = {mde_80:.2e}", transform=ax.transAxes,
                fontsize=7, alpha=0.75, ha="left", va="bottom")

    fig.tight_layout()
    Path(args.output_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf, bbox_inches="tight")
    print(f"[save] {args.output_pdf}")


if __name__ == "__main__":
    main()
