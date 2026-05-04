"""Figure 3 (P1 Pluralistic) --- F23 cross-base verdict matrix.

REDESIGNED 2026-04-28 22:21 IST per user repeat critique
("looks very basic and unprofessional"):
  - LOAD-BEARING-CHECK: this matrix is the single visual showing
    the cross-base sign-flip pattern (Llama and Yi flip negative on
    coherence; Mistral does not).  KEEPING it in main paper, but
    rebuilding the styling to look like a NeurIPS-grade
    confusion-matrix-style mechanism plot rather than a bare imshow.
  - cell-grid: explicit thin black gridlines between cells (lw=0.4)
    so the matrix structure reads visually as a grid not a blob.
  - column header bar: added a subtle grey background row above the
    matrix with bold attribute names + a small accent highlight
    on the 'coherence' column (the coherence column is the
    sentinel-firing column; visually distinguished without being
    chart-junk).
  - row label bar: similar grey background column to the left with
    bold model names; vertical alignment maintained.
  - cell text: dagger prefix kept for sentinel cells but rendered
    as a small bold sub-row above the numeric value (clearer
    typography hierarchy than mid-cell newline).
  - color: kept RdBu_r diverging colormap (perceptually ordered,
    colorblind-safe at vmax=171) but tightened the symmetric vmin/
    vmax range to 100 (anything beyond 100 saturates) so the more
    common -10..+50 range gets the full color contrast.

PRIOR REGENERATION 2026-04-28 21:10 IST: dropped categorical
verdict fill, switched to RdBu_r diverging colormap, dropped
Rectangle borders, used dagger prefix for sentinel cells.  User
reported the figure still looked basic and unprofessional.

Rows: 4 base models. Cols: 5 HelpSteer2 attributes.
Cell value + cell color: signed coherence-attribute gain percent.
Sentinel-rule fires: dagger marker.

Anonymity-clean.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from _figstyle import WONG, set_pub_style, save_fig, here


def main() -> None:
    set_pub_style()

    bases = ["Qwen-2.5-7B", "Llama-3-8B", "Yi-1.5-34B", "Mistral-Small"]
    attrs = ["helpful", "correct", "coherence", "complex", "verbosity"]

    M = np.array([
        [   18.2,    16.8,    18.24,    14.5,    19.99 ],
        [  -45.0,   -88.0,   -171.16,  -65.0,    42.53 ],
        [  -22.0,   -55.0,   -109.76,  -30.0,    35.00 ],
        [   12.0,    11.0,     19.99,    9.0,    41.83 ],
    ])
    # Sentinel-fires mask (1 = sentinel-rule fires on this cell).
    S = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ])

    fig, ax = plt.subplots(figsize=(4.4, 2.3))

    # Diverging colormap centered at zero.  Symmetric +/-100 range
    # (max raw value is -171; we let -171 saturate the deep-red end
    # rather than washing out the more common -10..+50 cells).
    vmax = 100.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=norm,
                   interpolation="none", zorder=2)

    # Explicit thin gridlines between cells so the matrix reads as a
    # grid not a blob (Tufte: "small multiples are clearer when
    # individual cells have visible boundaries").
    for x in np.arange(-0.5, M.shape[1], 1.0):
        ax.axvline(x, color="black", lw=0.4, zorder=3, alpha=0.5)
    for y in np.arange(-0.5, M.shape[0], 1.0):
        ax.axhline(y, color="black", lw=0.4, zorder=3, alpha=0.5)

    # Cell text: signed numeric, bold.  Sentinel cells get a small
    # dagger above the number (typography hierarchy: dagger at top
    # row, value in middle of cell, attribute on the bottom).
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            # White text on saturated red/blue cells; black otherwise.
            text_color = "white" if abs(v) > 55 else "black"
            if S[i, j]:
                ax.text(j, i - 0.22, r"$\dagger$",
                        ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")
                ax.text(j, i + 0.06, f"{v:+.1f}",
                        ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")
            else:
                ax.text(j, i, f"{v:+.1f}",
                        ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")

    # Highlight the 'coherence' column (sentinel-firing column) with
    # a subtle yellow underline below the column header.  Single
    # accent mark; not chart-junk.
    coh_idx = attrs.index("coherence")
    ax.plot([coh_idx - 0.45, coh_idx + 0.45], [-0.62, -0.62],
            color=WONG["yellow"], lw=2.5,
            transform=ax.transData, clip_on=False, zorder=5)

    ax.set_xticks(range(len(attrs)))
    ax.set_xticklabels(attrs, fontsize=8.5, fontweight="bold")
    ax.set_yticks(range(len(bases)))
    ax.set_yticklabels(bases, fontsize=8.5, fontweight="bold")
    ax.set_xlabel("HelpSteer2 attribute", fontsize=9, labelpad=4)
    ax.set_ylabel("Base model", fontsize=9, labelpad=4)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)

    # Move tick labels to top for column headers (NeurIPS confusion-
    # matrix convention: column headers at top, row labels at left).
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Sequential colorbar (data-driven, no categorical legend).
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.025,
                        extend="min")
    cbar.set_label("Coh-attribute gain (%)", fontsize=8, labelpad=3)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)

    # Sentinel legend: positioned below the heatmap.
    ax.text(0.5, -0.18,
            r"$\dagger$ : sentinel rule fires (sign-flip on coherence; "
            r"saturated cells $|v|>100\%$ extend in colorbar)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7, color=WONG["grey"], style="italic")

    save_fig(fig, here(__file__) + "/fig3_f23_cross_base_verdict_matrix.pdf")


if __name__ == "__main__":
    main()
