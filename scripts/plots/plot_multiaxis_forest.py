"""Forest plot of per-axis β̂_month with 95% BCa cluster-bootstrap CIs.

Reads results/track3_oasst2_multiaxis/summary.json and produces
figure_t3_multiaxis_drift.pdf + .png. Rows sorted by |β̂| descending.

Color key:
  red    — drifts after Bonferroni correction
  orange — drifts after BH-FDR only
  grey   — not significant (weak or null)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/track3_oasst2_multiaxis/summary.json")
    ap.add_argument("--output-pdf", default="paper/figure_t3_multiaxis_drift.pdf")
    ap.add_argument("--output-png", default="paper/figure_t3_multiaxis_drift.png")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    valid = [r for r in summary["per_axis"] if not r.get("skipped")]
    # Sort by |beta| desc
    valid.sort(key=lambda r: -abs(r["beta_month"]))

    axes_lbl = [r["axis"] for r in valid]
    betas = np.array([r["beta_month"] for r in valid])
    lo = np.array([r["bca_ci_95"][0] for r in valid])
    hi = np.array([r["bca_ci_95"][1] for r in valid])

    colors = []
    for r in valid:
        if r.get("bonferroni_reject_0p05"):
            colors.append("#c0392b")  # red
        elif r.get("bh_fdr_reject_0p05"):
            colors.append("#e67e22")  # orange
        else:
            colors.append("#7f8c8d")  # grey

    fig, ax = plt.subplots(figsize=(8.5, 0.5 + 0.35 * len(valid)))
    y = np.arange(len(valid))[::-1]  # reverse so largest at top
    for i, (b, l_, h, c) in enumerate(zip(betas, lo, hi, colors)):
        yi = y[i]
        ax.errorbar(b, yi, xerr=[[b - l_], [h - b]], fmt="o", color=c,
                    capsize=4, markersize=7, lw=1.8)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(axes_lbl)
    ax.set_xlabel(r"$\hat{\beta}_{\mathrm{month}}$  (points / month, 95% BCa cluster-bootstrap CI)")
    K = summary.get("K_tested", len(valid))
    bonf_alpha = summary.get("bonferroni_alpha_0p05", 0.05 / K) if K else 0.05
    ax.set_title(f"OASST2 multi-axis drift scan — {K} axes, 353 authors, 43.6k messages\n"
                 f"red = Bonferroni significant (α={bonf_alpha:.4f}); orange = BH-FDR only; grey = null/weak")

    # annotate each with perm_p
    xmax = hi.max()
    xmin = lo.min()
    span = xmax - xmin
    for i, r in enumerate(valid):
        yi = y[i]
        p = r["perm_p_phipson_smyth"]
        if p < 1e-3:
            ptxt = f"p<1e-3"
        else:
            ptxt = f"p={p:.3f}"
        ax.text(xmax + 0.02 * span, yi, f" {ptxt}  n={r['n_obs']:,}",
                va="center", fontsize=8, color="#444")

    ax.set_xlim(xmin - 0.05 * span, xmax + 0.30 * span)
    fig.tight_layout()
    Path(args.output_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf, bbox_inches="tight")
    fig.savefig(args.output_png, bbox_inches="tight", dpi=160)
    print(f"[save] {args.output_pdf}")
    print(f"[save] {args.output_png}")


if __name__ == "__main__":
    main()
