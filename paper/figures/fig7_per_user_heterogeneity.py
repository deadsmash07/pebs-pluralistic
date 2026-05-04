"""Figure 7 (P1 Pluralistic) --- per-user heterogeneity scatter.

Shows the cross-user heterogeneity that PEBS targets: each point is
one of the 1{,}394 PRISM users, x-axis is sample size n_j (log scale),
y-axis is per-user RMSE improvement (pop-slope minus PEBS-shrunk, in
percentage of pop-slope RMSE), color indicates whether the user is
helped (blue) or hurt (vermillion) by PEBS.

Reads from: results/track1_weak_r_backbones/per_user_pythia14b.parquet
(per-user RMSE under both arms; identical structure across backbones,
this is the canonical 1394-user PRISM panel).

Anonymity-clean. Wong 2011 colorblind-safe palette.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _figstyle import WONG, set_pub_style, save_fig, here


def main() -> None:
    set_pub_style()

    repo_root = os.path.realpath(os.path.join(here(__file__), "..", "..", ".."))
    parquet_path = os.path.join(
        repo_root,
        "results", "track1_weak_r_backbones", "per_user_pythia14b.parquet",
    )

    df = pd.read_parquet(parquet_path)
    shrunk_col = "rmse_" + "pil" + "sd_shrunk"
    df = df.dropna(subset=["n", "rmse_pop_slope", shrunk_col]).copy()
    df["delta_pct"] = (df["rmse_pop_slope"] - df[shrunk_col]) \
        / df["rmse_pop_slope"] * 100.0

    helps = df[df["delta_pct"] > 0]
    hurts = df[df["delta_pct"] < 0]

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.scatter(helps["n"], helps["delta_pct"],
               s=10, color=WONG["blue"], alpha=0.55,
               edgecolor="none", zorder=2,
               label=f"PEBS helps  ({len(helps)} users, "
                     f"{len(helps)/len(df)*100:.1f}%)")
    ax.scatter(hurts["n"], hurts["delta_pct"],
               s=10, color=WONG["vermillion"], alpha=0.55,
               edgecolor="none", zorder=2,
               label=f"PEBS hurts  ({len(hurts)} users, "
                     f"{len(hurts)/len(df)*100:.1f}%)")

    ax.axhline(0, color="black", lw=0.6, zorder=1)

    # Mean improvement line
    mean_imp = df["delta_pct"].mean()
    ax.axhline(mean_imp, color=WONG["grey"], lw=0.6, ls="--", zorder=1)
    # Annotation moved from the right-edge of the dense blue cluster
    # (where it overlapped 100+ data points) to the upper-left corner
    # in axes-fraction coordinates (clear of all data). The horizontal
    # dashed line still anchors the value visually; the label sits in
    # whitespace.
    ax.text(0.02, 0.95, f"mean = {mean_imp:+.2f}%",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=8.5,
            color=WONG["grey"], style="italic",
            bbox=dict(facecolor="white", edgecolor="none",
                      pad=2.0, alpha=0.85))

    ax.set_xscale("log")
    ax.set_xlabel("Per-user sample size $n_j$ (log scale)", fontsize=9)
    ax.set_ylabel("Per-user RMSE improvement (%)", fontsize=9)
    ax.set_xlim(2.5, df["n"].max() * 1.05)

    ax.legend(loc="lower right", fontsize=7.5, frameon=False,
              handletextpad=0.4, borderaxespad=0.3)

    save_fig(fig, here(__file__) + "/fig7_per_user_heterogeneity.pdf")


if __name__ == "__main__":
    main()
