"""Figure 4 (P1 Pluralistic) --- F19' multi-seed cross-seed scatter.

UPDATED 2026-04-28 22:21 IST per user repeat critique
("shapes very overlapping, legend in between figure"):
  - legend moved BELOW the axes (loc='upper center', ncol=5) and
    placed via bbox_to_anchor outside the axes box so the legend
    column no longer competes for horizontal space.  Previous fix
    moved legend to the right side outside but the right-column
    legend was still creating visual balance issues at column-width
    print sizes.
  - increased x-jitter +/-15 -> +/-35 units so the 4 UNTRAINED
    markers at each seed are clearly separated in x.  At the
    previous +/-15 the helpfulness/correctness markers overlapped
    wherever their values were close (e.g. seed 137 helpfulness=9.2
    + correctness=18.5 still overlapped horizontally).
  - bumped UNTRAINED marker size 28 -> 36 + reduced alpha 0.65 ->
    0.55 + black edge thickened 0.5 -> 0.7.  Markers individually
    legible at column-width.
  - kept marker shape mapping (o/s/^/v) for UNTRAINED, diamond for
    TRAINED.

PRIOR REGENERATION 2026-04-28 21:10 IST: legend moved outside,
alpha=0.65, jitter +/-15.  User reported overlap and legend issue
still hampered understanding.

Anonymity-clean. Wong 2011 colorblind-safe palette.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _figstyle import WONG, set_pub_style, save_fig, here


def main() -> None:
    set_pub_style()

    fig, ax = plt.subplots(figsize=(4.8, 2.7))

    seeds = [42, 137, 271, 314, 1729]
    seeds_arr = np.array(seeds, dtype=float)
    f19_anchor = 43.18
    trained_coh = np.array([42.55, 43.21, 41.98, 42.54, 43.18])
    untrained = {
        "helpfulness": np.array([14.5, 9.2, 18.0, 11.7, 15.3]),
        "correctness": np.array([22.0, 18.5, 15.2, 19.8, 20.3]),
        "complexity":  np.array([8.5, 12.1, 5.4, 10.2, 7.8]),
        "verbosity":   np.array([-32.6, -28.5, -35.1, -30.2, -33.4]),
    }

    # Phi-3 in-family anchor band (+/- 5pp). Yellow = ESTABLISHED-WITHIN-FAMILY-MAGNITUDE.
    ax.axhspan(f19_anchor - 5, f19_anchor + 5, color=WONG["yellow"],
               alpha=0.22, zorder=0)
    ax.axhline(f19_anchor, color=WONG["grey"], lw=0.8, ls="--",
               alpha=0.7, zorder=1)

    # All 4 UNTRAINED attributes plotted in the SAME light-grey color
    # (Tufte: "the data should drive the visual variation").  Marker
    # shape distinguishes them; color does not.
    # x-jitter so 4 markers at the same seed do not perfectly overlap.
    untrained_markers = {"helpfulness": "o", "correctness": "s",
                         "complexity":  "^", "verbosity":  "v"}
    # Jitter range expanded +/-22.5 -> +/-52.5 so the four UNTRAINED
    # markers at each seed are clearly separated in x; no horizontal
    # overlap regardless of y-overlap.
    jitter_offsets = {"helpfulness": -52.5, "correctness": -17.5,
                      "complexity":  +17.5, "verbosity":  +52.5}
    for name, vals in untrained.items():
        ax.scatter(seeds_arr + jitter_offsets[name], vals,
                   marker=untrained_markers[name], s=38,
                   color=WONG["light_grey"], edgecolor="black",
                   linewidth=0.7, alpha=0.55, zorder=2,
                   label=f"untrained: {name}")

    # TRAINED-coh: bold blue diamonds (single emphasized class).
    ax.scatter(seeds, trained_coh, marker="D", s=58, color=WONG["blue"],
               edgecolor="black", linewidth=0.7, alpha=0.95, zorder=3,
               label="trained: coherence")

    ax.axhline(0, color="black", lw=0.5, zorder=1)

    ax.set_xticks(seeds)
    ax.set_xticklabels([str(s) for s in seeds], fontsize=8)
    ax.set_xlabel(r"Seed", fontsize=9)
    ax.set_ylabel("Per-attribute gain (%)", fontsize=9)
    ax.set_xlim(min(seeds) - 110, max(seeds) + 110)
    ax.tick_params(axis="y", labelsize=8)
    # Anchor band annotation positioned in upper-left corner of the
    # axes (axes-fraction; survives any data-coord changes).
    ax.text(0.02, 0.96, r"Phi-3 in-family anchor: $+43.18\%\,\pm\,5$pp (yellow band)",
            transform=ax.transAxes, fontsize=7,
            color=WONG["grey"], ha="left", va="top", style="italic")

    # Legend BELOW the axes box; ncol=5 horizontal layout so legend
    # column does not compete for horizontal space inside the figure.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              fontsize=7, ncol=5, handletextpad=0.3,
              columnspacing=0.8, borderpad=0.3, frameon=False)

    save_fig(fig, here(__file__) + "/fig4_f19_prime_cross_seed_scatter.pdf")


if __name__ == "__main__":
    main()
