"""Figure 1 (P1 Pluralistic) --- PEBS headline forest plot.

UPDATED 2026-04-28 22:21 IST per user repeat critique
("text is garbled, models and family are on top of each other"):
  - figure width 7.2 -> 8.4 in so the left margin has room for the
    full row-labels + the italic lineage-group annotations without
    horizontal overlap.
  - lineage-group annotations moved from data-coordinate x=-360 to
    axes-fraction x=-0.30 with explicit ha='right'.  Previously the
    data-coord -360 collided with the row-label text region at the
    column-width print size; axes-fraction is robust to figsize
    changes.
  - row-label fontsize bumped 8.5 -> 9pt (matches axis-label scale).
  - increased subplots_adjust left margin to 0.30 so the row-label
    column is fully visible.
  - shortened "Phi-3-medium" -> "Phi-3-14B" consistently and dropped
    parenthetical attribute hints in favor of a separate column header
    (lineage column shows attribute via prefix in italic group label).

PRIOR REGENERATION 2026-04-28 21:10 IST: increased figsize
3.4 -> 4.6 in vertical, shortened abbreviations, increased row-label
font 7.5 -> 8.5pt.  User reported labels still garbled at column
width.

PRIOR REGENERATION 2026-04-28 per FIGURE_CRITIQUE: dropped the
3-panel A/B/C concept-cramming layout (panels A/B/C had no shared
axes).  This figure now shows the single coherent argument the
paper makes visually: 5 numerical anchor points + their 95% CIs
in a forest plot, ordered by base-model lineage (Qwen-family
ESTABLISHED -> Phi-3 exploratory -> F23 cross-base sentinel
firings -> verbosity-only counterfactuals).

Color-codes by verdict class (Wong palette):
  Blue (#0072B2)   = Qwen-family ESTABLISHED
  BluGreen (...73) = within-family ESTABLISHED replication
  Vermillion (..00)= F23 sentinel sign-flip (FALSIFIED)
  Orange  (..00)   = NULL / verdict straddles 0
  Grey  (#606060)  = anchor / reference

The schematic of EB shrinkage that previously occupied panel A and
the adapter-inspection horizontal-bar plot that occupied panel C are
moved to fig2 (omega curve already covers the shrinkage idea
mechanically) and a future Inkscape mechanism schematic
(deferred per skill anti-pattern: matplotlib is the wrong tool for
schematics).

Anonymity-clean. Wong 2011 colorblind-safe palette.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _figstyle import WONG, set_pub_style, save_fig, here


def main() -> None:
    set_pub_style()

    # Anchored on PAPER_CLAIMS + body-text values; preserved bit-exact.
    # Row labels: shortened "model + family" prefix so the left
    # column does not overlap (per user critique 21:10 IST: prior
    # version had garbled / overlapping family labels).
    rows = [
        # label, point, lo, hi, verdict_color, verdict_text
        ("Qwen-2.5-7B (4-seed)",            +18.24, +17.97, +18.52, WONG["blue"],       "ESTABLISHED"),
        ("Phi-3-14B (in-family)",           +43.23, +31.69, +51.21, WONG["blu_green"],  "EXPLORATORY"),
        ("Llama-3-8B (coherence-only)",    -171.16, -256.40, -82.33, WONG["vermillion"], "SIGN-FLIP"),
        ("Yi-1.5-34B (coherence-only)",    -109.76, -170.11, -47.92, WONG["vermillion"], "SIGN-FLIP"),
        ("Mistral-22B (coherence-only)",    +19.99, -17.22, +42.86, WONG["orange"],     "NULL"),
        ("Llama-3-8B (verbosity-only)",     +42.53, +32.40, +51.86, WONG["blu_green"],  "ESTABLISHED-NULL"),
        ("Mistral-22B (verbosity-only)",    +41.83, +32.71, +49.24, WONG["blu_green"],  "ESTABLISHED-NULL"),
        ("Phi-3-14B (verbosity-only)",      +43.18, +33.02, +50.65, WONG["blu_green"],  "(coh-anchor)"),
    ]

    n = len(rows)
    # Figure size 8.4 x 4.6 in: width bumped 7.2 -> 8.4 per user
    # repeat critique 22:13 IST ("text garbled, models and family on
    # top of each other").  Wider canvas + axes-fraction lineage
    # labels eliminates horizontal overlap.
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    fig.subplots_adjust(left=0.30, right=0.97, top=0.93, bottom=0.18)

    y = np.arange(n)[::-1]   # top-to-bottom = first row at top
    for yi, (lbl, p, lo, hi, c, vt) in zip(y, rows):
        ax.errorbar(p, yi, xerr=[[p - lo], [hi - p]], fmt="none",
                    ecolor=WONG["grey"], elinewidth=1.0, capsize=2.5,
                    zorder=2)
        ax.plot(p, yi, "o", markersize=6.5, color=c,
                markeredgecolor="black", markeredgewidth=0.5, zorder=3)
        # inline numeric to the right of the CI
        ax.text(hi + 8, yi, f"{p:+.1f}%  [{lo:+.1f}, {hi:+.1f}]",
                ha="left", va="center", fontsize=7.5,
                color=c)

    # Reference: zero line (no effect)
    ax.axvline(0, color="black", lw=0.6, zorder=1)
    # Reference: Phi-3 in-family anchor band (43.18% +/- 5pp; pre-registered gate).
    ax.axvspan(43.18 - 5, 43.18 + 5, color=WONG["yellow"], alpha=0.18,
               zorder=0)
    ax.text(43.18, n - 0.4, "Phi-3 in-family anchor\n($\\pm$5pp)",
            fontsize=6.5, color=WONG["grey"], ha="center", va="bottom",
            style="italic")

    # Group separators (lineage)
    for sep_y in [n - 1.5, n - 5.5]:   # between Qwen+Phi3 / coherence-only / verbosity-only
        ax.axhline(sep_y, color=WONG["light_grey"], lw=0.4, ls=":",
                   zorder=0)

    # Lineage group labels: positioned in axes-fraction space (x=-0.30
    # of axes width, well outside the row-label column).  Previously
    # at data-coord x=-360 the labels collided with the model+family
    # row-labels.  ha='right' anchors text right-edge at x=-0.30 so
    # there is reliable whitespace between group-label and row-label
    # at every figsize.  Less internal-jargon language: "ESTABLISHED"
    # rephrased as "in-family replication" etc.
    groups = [
        (n - 1, "Qwen-family\nbaseline"),
        (n - 2, "Phi-3 (in-family\nreplication)"),
        (n - 4, "Cross-base\nsentinel firings"),
        (n - 7, "Verbosity-only\ncounterfactual"),
    ]
    for yi, txt in groups:
        ax.text(-0.30, yi, txt, ha="right", va="center",
                transform=ax.get_yaxis_transform(),
                fontsize=7.5, color=WONG["grey"], style="italic")

    # Legend (academic prose; no ALL-CAPS verdict labels per skill §M25).
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=WONG["blue"], markeredgecolor="black",
                   markeredgewidth=0.5, markersize=6.5,
                   label="Qwen-family anchor"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=WONG["blu_green"], markeredgecolor="black",
                   markeredgewidth=0.5, markersize=6.5,
                   label="Within-family / verbosity-control"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=WONG["vermillion"], markeredgecolor="black",
                   markeredgewidth=0.5, markersize=6.5,
                   label="Cross-family sign-flip"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=WONG["orange"], markeredgecolor="black",
                   markeredgewidth=0.5, markersize=6.5,
                   label="Null (CI straddles 0)"),
    ]
    ax.legend(handles=handles, loc="lower right",
              bbox_to_anchor=(1.0, -0.32), ncol=2,
              fontsize=7.5, frameon=False)

    ax.set_yticks(y)
    # Row-label fontsize bumped 8.5 -> 9 (matches axes-label scale).
    # The wider 8.4in canvas + 0.30 left-margin reservation eliminates
    # the overlap with lineage-group italic labels.
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("Coherence-attribute PEBS gain (%)", fontsize=9)
    ax.set_xlim(-340, 130)
    ax.set_ylim(-0.7, n - 0.2)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8)

    save_fig(fig, here(__file__) + "/fig1_pebs_overview.pdf")


if __name__ == "__main__":
    main()
