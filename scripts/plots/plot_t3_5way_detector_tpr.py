"""Camera-ready Plot 3: T3 5-way drift-detector TPR comparison.

Grouped bar chart: 5 drift levels beta in {0, 0.002, 0.005, 0.010, 0.030} (pts/mo),
five detectors (MixedLM+perm, Naive OLS, PageHinkley, CUSUM, EWMA).

Error bars are the Wilson 95% CIs already stored in the summary.json.
Annotates the headline gap: MixedLM 52% vs Naive OLS 12% at beta=0.002.
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
OUT = FIGS / "t3_5way_detector_tpr.pdf"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
})


def main() -> None:
    data = json.loads((RESULTS / "track3_detector_comparison_extended" / "summary.json").read_text())
    tbl = data["tpr_table"]
    drift = np.array(tbl["drift"])

    detectors = [
        ("MixedLM + perm", "mixedlm_perm", "ci_mixedlm_perm", "#1f77b4"),
        ("Naive OLS",      "naive_ols",    "ci_naive_ols",    "#2ca02c"),
        ("Page-Hinkley",   "pagehinkley",  "ci_pagehinkley",  "#9467bd"),
        ("CUSUM",          "cusum",        "ci_cusum",        "#ff7f0e"),
        ("EWMA",           "ewma",         "ci_ewma",         "#8c564b"),
    ]

    n_det = len(detectors)
    n_grp = len(drift)
    x = np.arange(n_grp)
    width = 0.15

    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    for i, (label, key, cikey, color) in enumerate(detectors):
        tpr = np.array(tbl[key])
        ci = np.array(tbl[cikey])  # shape (n_grp, 2)
        # Wilson CIs can fall very slightly on either side of the point estimate
        # when TPR is 0.0 or 1.0 (clamp to non-negative for matplotlib).
        lo = np.clip(tpr - ci[:, 0], 0.0, None)
        hi = np.clip(ci[:, 1] - tpr, 0.0, None)
        pos = x + (i - (n_det - 1) / 2) * width
        ax.bar(pos, tpr, width=width, color=color, edgecolor="white", label=label)
        ax.errorbar(pos, tpr, yerr=[lo, hi], fmt="none",
                    ecolor="#333", lw=0.9, capsize=2.5)

    # Headline annotation at beta=0.002 (index 1): MixedLM 0.52 vs Naive OLS 0.12
    idx = 1
    mix_pos = x[idx] + (0 - (n_det - 1) / 2) * width
    nai_pos = x[idx] + (1 - (n_det - 1) / 2) * width
    mix_tpr = tbl["mixedlm_perm"][idx]
    nai_tpr = tbl["naive_ols"][idx]
    y_brk = 0.70
    ax.annotate("", xy=(mix_pos, y_brk), xytext=(nai_pos, y_brk),
                arrowprops=dict(arrowstyle="-", color="#d62728", lw=1.1))
    ax.annotate("", xy=(mix_pos, mix_tpr + 0.03), xytext=(mix_pos, y_brk),
                arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.9))
    ax.annotate("", xy=(nai_pos, nai_tpr + 0.03), xytext=(nai_pos, y_brk),
                arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.9))
    ax.text((mix_pos + nai_pos) / 2, y_brk + 0.03,
            f"+{(mix_tpr - nai_tpr)*100:.0f} pp\nat low drift",
            ha="center", va="bottom", color="#d62728",
            fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:.3f}" if b > 0 else "0 (null)" for b in drift])
    ax.set_xlabel(r"True drift rate $\beta$ (quality points / month)")
    ax.set_ylabel("True-positive rate (TPR @ 100 seeds)")
    ax.set_ylim(0, 1.08)
    ax.set_title("T3: 5-way drift-detector comparison (FPR-calibrated; 95% Wilson CIs)")
    ax.axhline(0.05, ls=":", color="#888", lw=0.8)
    ax.text(-0.45, 0.07, "nominal alpha = 0.05", fontsize=7.5, color="#666")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              frameon=False, fontsize=9)
    ax.grid(axis="y", ls=":", color="#bbb", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUT, format="pdf", bbox_inches="tight", metadata={"Creator": "PILSD"})
    print(f"wrote: {OUT}")


if __name__ == "__main__":
    main()
