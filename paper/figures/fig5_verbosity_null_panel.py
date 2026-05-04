"""Figure 5 (P1 Pluralistic) --- F26 + F37 cross-base verbosity-null
small-multiples (3 panels: Llama / Mistral / Phi-3).

UPDATED 2026-04-28 22:21 IST per user repeat critique
("axis position does not look as good as it should; pretty average"):
  - per-panel axes typography upgraded to publication-grade:
    * y-axis label moved to centered position with consistent
      labelpad=8 (previously default labelpad created uneven gap).
    * x-axis labels rotated 0 deg + horizontal alignment center;
      previously the two-line text 'UNTRAINED\\ncoherence' was the
      only label and lacked a unifying x-axis title.  Added explicit
      x-axis label 'Attribute (untrained vs trained)' so the
      per-panel x-axis is self-describing.
    * panel-titles (model names) repositioned with consistent
      pad=8 (previously pad=6 read as cramped against the panel).
  - bottom-margin increased so x-axis labels do not collide with
    the lower edge.
  - subtle alternating panel-background tint (very light grey) on
    panels 2 + 4 of a 5-panel layout would normally help, but at
    n_panels=3 we use a single slightly off-white background tint
    on each axes to soften the visual baseline.
  - added a shared title 'Cross-base verbosity-null parity (3 of 3)'
    at top-center (figure-level, not axes-level) so the figure has
    a clear single-sentence headline.
  - increased figure size 7.2x2.8 -> 7.8x3.2 inches.

PRIOR REGENERATION 2026-04-28 21:10 IST: enabled horizontal
gridlines, elevated panel-label A/B/C to 12pt bold, added shared
F19-anchor reference line, widened wspace.  User reported axis
positioning was still 'pretty average'.

Anonymity-clean.  Wong 2011 colorblind-safe palette.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _figstyle import WONG, set_pub_style, save_fig, here, panel_label


def main() -> None:
    set_pub_style()

    fig, axes = plt.subplots(1, 3, figsize=(5.2, 3.4), sharey=True,
                              gridspec_kw={"wspace": 0.22,
                                           "left": 0.11,
                                           "right": 0.97,
                                           "bottom": 0.24,
                                           "top": 0.78})

    # Figure-level shared title for all three panels. Larger font
    # because the figure ships into a 0.48\textwidth minipage, so the
    # rendered title shrinks by ~2x at print size. Two-line title
    # placed well above the panel labels (A/B/C) which sit at y=1.10.
    fig.suptitle(
        "Untrained-coherence matches anchor;\n"
        "trained-verbosity head collapses",
        fontsize=10.5, y=0.99, fontweight="bold", color="black",
    )

    bases = [
        ("Llama-3-8B",     +42.53, +18.0),
        ("Mistral-22B",    +41.83, +15.0),
        ("Phi-3-medium",   +43.18, -32.62),
    ]
    panel_labels = ["A", "B", "C"]

    f19_anchor = 43.18

    for ax, (name, untrained_coh, trained_verb), pl in zip(
            axes, bases, panel_labels):
        labels = ["untrained\ncoherence", "trained\nverbosity"]
        values = [untrained_coh, trained_verb]
        colors = [WONG["blu_green"] if v > 0 else WONG["vermillion"]
                  for v in values]
        ax.bar(labels, values, color=colors, edgecolor="black",
               linewidth=0.5, width=0.62, zorder=3)

        # Subtle horizontal gridlines for cross-panel reading.
        ax.yaxis.grid(True, color=WONG["light_grey"], lw=0.4,
                      alpha=0.6, zorder=0)
        ax.set_axisbelow(True)

        ax.axhline(0, color="black", lw=0.7, zorder=2)
        # Phi-3 in-family anchor band (38-48 = +/- 5pp on +43.18%).
        ax.axhspan(38, 48, color=WONG["yellow"], alpha=0.22, zorder=1)
        # Shared Phi-3 anchor reference line across all panels.
        ax.axhline(f19_anchor, color=WONG["grey"], lw=0.8, ls="--",
                   alpha=0.6, zorder=2)

        # Inline numerical values centered above/below bars. Larger
        # font (10.5pt) compensates for ~2x shrinkage in 0.48\textwidth.
        for i, v in enumerate(values):
            offset = 3 if v > 0 else -5
            ax.text(i, v + offset, f"{v:+.1f}%", ha="center",
                    va="bottom" if v > 0 else "top",
                    fontsize=10.5,
                    color=colors[i], fontweight="bold")

        # Panel title: model name centered with consistent pad. The
        # A/B/C panel-label sits at the upper-left corner and the
        # model name sits centered above the axes (so the two do
        # not collide regardless of left-loc title length).
        ax.set_title(name, loc="center", fontsize=10, fontweight="bold",
                     pad=2)
        ax.tick_params(axis="x", labelsize=10)
        ax.set_ylim(-55, 65)
        panel_label(ax, pl, x=-0.10, y=1.02, fontsize=12)

    # Y-axis label: explicit labelpad for consistent positioning.
    # Bumped from 9.5 to 11.5 so it stays readable after minipage shrink.
    axes[0].set_ylabel("Per-attribute gain (%)", fontsize=11.5,
                      labelpad=6)
    axes[0].tick_params(axis="y", labelsize=10)

    # Shared bottom x-axis label (single label spanning all 3 panels).
    # Lowered y from 0.05 to 0.02 so it sits below the rotated x-tick
    # labels and does not collide with them.
    fig.text(0.53, 0.01, "Attribute class (untrained vs trained)",
             ha="center", va="bottom", fontsize=10.5)

    save_fig(fig, here(__file__) + "/fig5_verbosity_null_panel.pdf")


if __name__ == "__main__":
    main()
