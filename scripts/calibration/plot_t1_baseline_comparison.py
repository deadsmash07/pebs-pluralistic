"""Camera-ready Plot 1: T1 baseline comparison (horizontal bar chart).

Loads per-user RMSE parquets (Qwen + Skywork arms), computes cluster-level
(per-user) bootstrap 95% CIs on the mean RMSE, and plots 6 arms:
  no-calib, pop-slope (Qwen), naive-OLS (Qwen), PILSD-shrunk (Qwen),
  Skywork pop-slope, Skywork + PILSD-shrunk.

The 8.58% headline improvement (pop-slope Qwen -> PILSD-shrunk Qwen) is
annotated with a bracket.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Config
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
OUT = FIGS / "t1_baseline_comparison.pdf"

N_BOOT = 2000
RNG_SEED = 42

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,  # editable text in PDF
})


def bootstrap_mean_ci(x: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float = 0.05):
    """Percentile bootstrap CI on the mean over users (each user is a cluster)."""
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(x.mean()), float(lo), float(hi)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    # Load per-user RMSE tables
    df_shr = pd.read_parquet(RESULTS / "track1_user_score_mse_shrunk.parquet")
    skywork_path = RESULTS / "track1_skywork_5arm_eval.json"
    sky = json.loads(skywork_path.read_text())

    # Arm data: (label, per-user-array-or-None, paper-number-fallback)
    # For Skywork we only have aggregate means; use paper numbers + mark CI as NA.
    arms = [
        ("No calibration",          df_shr["rmse_no_calib"].to_numpy(),   27.13, "#8c8c8c"),
        ("Pop-slope (Qwen)",        df_shr["rmse_pop_slope"].to_numpy(),  25.52, "#4e79a7"),
        ("Naive OLS (Qwen)",        df_shr["rmse_pilsd_ols"].to_numpy(),  23.73, "#59a14f"),
        ("PILSD shrunk (Qwen)",     df_shr["rmse_pilsd_shrunk"].to_numpy(),23.33, "#e15759"),
        ("Pop-slope (Skywork)",     None,                                  27.17, "#b07aa1"),
        ("PILSD shrunk (Skywork)",  None,                                  25.46, "#edc948"),
    ]

    labels, means, los, his, colors = [], [], [], [], []
    for label, arr, fallback, c in arms:
        if arr is not None and len(arr) > 0:
            m, lo, hi = bootstrap_mean_ci(arr, N_BOOT, rng)
        else:
            m, lo, hi = fallback, fallback, fallback
        labels.append(label)
        means.append(m)
        los.append(lo)
        his.append(hi)
        colors.append(c)

    means = np.array(means); los = np.array(los); his = np.array(his)

    # Plot
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    y = np.arange(len(labels))[::-1]  # top-down
    xerr = np.vstack([means - los, his - means])
    bars = ax.barh(y, means, xerr=xerr, color=colors, edgecolor="white",
                   height=0.66, error_kw=dict(lw=1.1, capsize=4, ecolor="#333"))

    for i, (yy, m, lo, hi) in enumerate(zip(y, means, los, his)):
        if lo == hi:  # no bootstrap CI available
            ax.text(m + 0.12, yy, f"{m:.2f}", va="center", ha="left",
                    fontsize=9, color="#333")
        else:
            ax.text(m + 0.12, yy, f"{m:.2f}  [{lo:.2f},{hi:.2f}]",
                    va="center", ha="left", fontsize=8.5, color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Held-out per-user RMSE (lower is better)")
    ax.set_xlim(21.5, 30.5)
    ax.set_title("T1: Per-user calibration on PRISM (1394 users, 5-fold CV)")

    # Bracket annotation for headline 8.58 % improvement
    y_pop   = y[1]  # Pop-slope Qwen
    y_pilsd = y[3]  # PILSD-shrunk Qwen
    x_pop, x_pilsd = 25.52, 23.33
    x_brk = 22.2
    ax.annotate("", xy=(x_brk, y_pop), xytext=(x_brk, y_pilsd),
                arrowprops=dict(arrowstyle="-", color="#d62728", lw=1.1))
    ax.annotate("", xy=(x_pop, y_pop), xytext=(x_brk, y_pop),
                arrowprops=dict(arrowstyle="-", color="#d62728", lw=1.1))
    ax.annotate("", xy=(x_pilsd, y_pilsd), xytext=(x_brk, y_pilsd),
                arrowprops=dict(arrowstyle="-", color="#d62728", lw=1.1))
    ax.text(x_brk - 0.15, (y_pop + y_pilsd) / 2, "-8.58%\n(headline)",
            ha="right", va="center", fontsize=9.5, color="#d62728",
            fontweight="bold")

    ax.grid(axis="x", ls=":", color="#bbb", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUT, format="pdf", bbox_inches="tight", metadata={"Creator": "PILSD"})
    print(f"wrote: {OUT}")


if __name__ == "__main__":
    main()
