"""Figure 6 (P1 Pluralistic) --- PILSD 3-layer architecture schematic.

Per-skill (research-paper-writing-icml-neurips §M16) matplotlib is the
wrong tool for schematics in general; TikZ or Inkscape are preferred.
We use matplotlib here for time-budget reasons and keep the schematic
intentionally simple: three labelled boxes connected by arrows, with
inline math under each box describing the layer's role. No pixel
art, no shading, no overlapping.

Anonymity-clean. Wong 2011 colorblind-safe palette.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from _figstyle import WONG, set_pub_style, save_fig, here


def main() -> None:
    set_pub_style()

    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    # Box geometry. Boxes wider + larger gaps so titles don't overflow.
    box_w, box_h = 28, 16
    y0 = 13
    centers_x = [16, 50, 84]
    titles = [
        "Layer 1\nReward model $r_\\phi(x)$",
        "Layer 2\nPer-rater calibrator\n$(\\hat\\alpha_j,\\hat\\beta_j)$",
        "Layer 3\nEB shrinkage\n$\\omega_j = \\tau^2 / (\\tau^2 + V_j)$",
    ]
    colors = [WONG["sky_blue"], WONG["blu_green"], WONG["blue"]]
    subtitles = [
        "Standard RLHF.\nFrozen at deployment.",
        "Linear correction\nfrom OLS per user.",
        "Trust-knob blend toward\npopulation mean.",
    ]

    # Draw boxes
    for cx, title, c, sub in zip(centers_x, titles, colors, subtitles):
        bx = cx - box_w / 2
        by = y0
        box = FancyBboxPatch(
            (bx, by), box_w, box_h,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            linewidth=1.0,
            edgecolor=c,
            facecolor="white",
            zorder=2,
        )
        ax.add_patch(box)
        # Title (centered, bold)
        ax.text(cx, by + box_h * 0.70, title,
                ha="center", va="center", fontsize=7.5, color=c,
                weight="bold", zorder=3, linespacing=1.3)
        # Subtitle (smaller, grey)
        ax.text(cx, by + box_h * 0.18, sub,
                ha="center", va="center", fontsize=6.5,
                color=WONG["grey"], style="italic", zorder=3,
                linespacing=1.3)

    # Arrows between boxes
    for i in range(len(centers_x) - 1):
        x_from = centers_x[i] + box_w / 2
        x_to = centers_x[i + 1] - box_w / 2
        arrow = FancyArrowPatch(
            (x_from, y0 + box_h / 2),
            (x_to, y0 + box_h / 2),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.0,
            color=WONG["grey"],
            zorder=1,
        )
        ax.add_patch(arrow)
        # Arrow label
        labels_arrow = ["scalar score\nper item", "shrunk to\npopulation"]
        ax.text((x_from + x_to) / 2, y0 + box_h / 2 + 3.3, labels_arrow[i],
                ha="center", va="bottom", fontsize=6.5,
                color=WONG["grey"], style="italic")

    # Bottom annotation: deployment-time prediction formula
    formula = (
        r"At deployment, annotator $j$'s predicted rating on item $x$:  "
        r"$\hat y_{ji} = \hat\alpha_j^{\,\mathrm{shrunk}} \cdot r_\phi(x) "
        r"+ \hat\beta_j^{\,\mathrm{shrunk}}$"
    )
    ax.text(50, 4, formula, ha="center", va="center", fontsize=8.0,
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=WONG["yellow"],
                      edgecolor=WONG["grey"], linewidth=0.5, alpha=0.5))

    # Top labels: input / output
    ax.text(centers_x[0], y0 + box_h + 2.5, "Pooled human preference data",
            ha="center", va="bottom", fontsize=7.0,
            color=WONG["grey"], style="italic")
    ax.text(centers_x[-1], y0 + box_h + 2.5, "Per-annotator-calibrated reward",
            ha="center", va="bottom", fontsize=7.0,
            color=WONG["grey"], style="italic")

    save_fig(fig, here(__file__) + "/fig6_pilsd_schematic.pdf")


if __name__ == "__main__":
    main()
