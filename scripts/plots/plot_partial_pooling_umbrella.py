"""Figure 2 -- the partial-pooling umbrella with two distinct instantiations.

This figure is a direct response to the adversarial review's "cosmetic
unification" critique. It replaces the narrative-stretch framing
("one principle, three tracks") with the honest reframe:

    Partial-pooling UMBRELLA with TWO distinct instantiations:
      (a) BLUP / EB shrinkage   -> Tracks 1 + 3 (per-annotator scale
          calibration; per-author longitudinal drift)
      (b) V-trace importance-weighted correction -> Track 2 (delayed-
          reward forward injection across optimizer steps)
    Both are random-effect-aware corrections to naive aggregation, but
    they are *mathematically distinct* estimators and should not be
    conflated.

Panels
------
A. Abstract random-effect diagram. Cluster variable j with observations
   y_{j,i}; partial-pooling arrow collapses per-cluster estimates toward
   a pooled anchor. Decorated with the shrinkage formula and the
   identity-limit annotations (omega->0: pool; omega->1: local).
B. Dichotomy matrix (track x estimator property). Rows: T1, T2, T3.
   Columns: (1) estimator form, (2) which random effect is modelled,
   (3) shrinkage target. Cells are colored by which of the two
   instantiation families they belong to (BLUP vs V-trace IS).

The figure is self-explanatory: a reader who skips to Figure 2 will
immediately see that T2 is NOT Eq. (1) under a literal BLUP reading;
instead, it is a second instantiation of the partial-pooling umbrella
that shares the random-effect-awareness without sharing the estimator
form.

No seaborn; matplotlib only.
"""
from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PNG = REPO_ROOT / "3_PEBS_Standalone/paper/figure2.png"
DEFAULT_OUTPUT_SVG = REPO_ROOT / "3_PEBS_Standalone/paper/figure2.svg"


# Colors
COLOR_BLUP = "#17becf"    # teal = BLUP / shrinkage (T1, T3)
COLOR_VTRACE = "#ff7f0e"  # orange = V-trace IS clip (T2)
COLOR_POOLED = "#888888"  # gray = pooled anchor
COLOR_LOCAL = "#d62728"   # red = per-cluster local estimate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the PEBS partial-pooling umbrella figure."
    )
    p.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    p.add_argument("--output-svg", type=Path, default=DEFAULT_OUTPUT_SVG)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Panel A : abstract random-effect diagram
# ---------------------------------------------------------------------------
def plot_panel_a(ax) -> None:
    """Abstract partial-pooling schematic.

    Five clusters j in {1..5}. Each cluster has a local estimate
    theta_j^local (red). A pooled anchor theta_pool (gray) sits in the
    middle. Arrows from each local estimate point toward the pooled
    anchor, annotated with shrinkage weight omega_j.
    """
    import numpy as np

    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(-0.2, 5.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Cluster x-positions
    xs = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    # Local estimates (spread around pool)
    thetas = np.array([4.3, 2.5, 3.8, 1.8, 4.6])
    pool = 3.2

    # Pooled anchor band (horizontal)
    ax.axhline(pool, color=COLOR_POOLED, linestyle="--", linewidth=1.5,
               alpha=0.65, zorder=1)
    ax.text(10.25, pool + 0.05,
            r"$\hat\theta_{\mathrm{pool}}$",
            color=COLOR_POOLED, fontsize=12, va="bottom", ha="right")

    # For each cluster: local scatter + arrow toward pooled
    for j, (x, th) in enumerate(zip(xs, thetas), start=1):
        # A few within-cluster observations as light dots
        rng = np.random.default_rng(7 * j)
        yobs = th + rng.normal(0, 0.22, size=6)
        xobs = x + rng.uniform(-0.22, 0.22, size=6)
        ax.scatter(xobs, yobs, s=11, color=COLOR_LOCAL, alpha=0.45, zorder=2)
        # Local estimate point
        ax.scatter([x], [th], s=70, color=COLOR_LOCAL, zorder=4,
                   edgecolor="black", linewidth=0.6)
        # Shrinkage arrow toward pool
        omega = 0.35 + 0.12 * (j - 3) ** 2 / 4.0
        shrunk = omega * th + (1.0 - omega) * pool
        ax.annotate(
            "",
            xy=(x, shrunk),
            xytext=(x, th),
            arrowprops=dict(arrowstyle="->", color=COLOR_BLUP,
                            lw=1.5, alpha=0.9),
            zorder=3,
        )
        # Shrunk estimate
        ax.scatter([x], [shrunk], s=55, color=COLOR_BLUP, zorder=5,
                   edgecolor="black", linewidth=0.6)
        # Label cluster j
        ax.text(x, -0.05, f"j={j}", ha="center", va="top",
                fontsize=10, color="#333")
        # Label omega on the arrow midpoint
        ax.text(x + 0.28, 0.5 * (th + shrunk),
                rf"$\omega_{{{j}}}$",
                color=COLOR_BLUP, fontsize=9.5, ha="left", va="center")

    # Legend entries (manual patches to avoid clutter)
    ax.scatter([], [], s=70, color=COLOR_LOCAL, edgecolor="black",
               linewidth=0.6,
               label=r"local  $\hat\theta_j^{\mathrm{local}}$")
    ax.scatter([], [], s=55, color=COLOR_BLUP, edgecolor="black",
               linewidth=0.6,
               label=r"partial-pooled  $\hat\theta_j^{\mathrm{PP}}$")
    ax.plot([], [], color=COLOR_POOLED, linestyle="--", linewidth=1.5,
            label=r"pool  $\hat\theta_{\mathrm{pool}}$")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95,
              bbox_to_anchor=(0.00, 1.04))

    # Shrinkage formula
    ax.text(
        5.0, 4.85,
        r"$\hat\theta_j^{\mathrm{PP}} \;=\; "
        r"\omega_j\,\hat\theta_j^{\mathrm{local}} + "
        r"(1-\omega_j)\,\hat\theta_{\mathrm{pool}}, "
        r"\quad \omega_j=\frac{\tau^2}{\tau^2+V(\hat\theta_j^{\mathrm{local}})}$",
        ha="center", va="bottom", fontsize=10.5,
        bbox=dict(facecolor="white", edgecolor="#bbbbbb",
                  boxstyle="round,pad=0.3", alpha=0.95),
    )

    # Annotation: limits
    ax.text(0.0, 1.05,
            r"$\omega\!\to\!0$: complete pool   $\vert$   "
            r"$\omega\!\to\!1$: no pool",
            transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, color="#444",
            style="italic")

    ax.set_title(
        "A. Partial-pooling umbrella  (abstract random-effect schematic)",
        fontsize=12, fontweight="bold", loc="left",
    )


# ---------------------------------------------------------------------------
# Panel B : dichotomy matrix
# ---------------------------------------------------------------------------
def plot_panel_b(ax) -> None:
    """Track x property matrix, colored by which family the cell belongs to.

    Families
    --------
    BLUP / EB shrinkage family  (teal)  -> Tracks 1, 3
    V-trace IS family           (orange) -> Track 2
    """
    import matplotlib.patches as mpatches

    ax.set_xlim(-0.1, 4.6)
    ax.set_ylim(-0.2, 3.9)
    ax.axis("off")

    # Headers
    headers = [
        "Track",
        "Estimator form",
        "Random effect\nbeing modelled",
        "Shrinkage /\ncorrection target",
    ]
    # Column x-centres
    col_x = [0.45, 1.65, 2.85, 4.05]
    col_w = [1.1, 1.1, 1.1, 1.1]

    for x, label in zip(col_x, headers):
        ax.text(x, 3.50, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="black")

    # Horizontal separator under header
    ax.plot([-0.05, 4.60], [3.18, 3.18], color="black", linewidth=1.2)

    # Row data: (row label, family, cells...)
    rows = [
        (
            "T1 (PRISM)",
            "blup",
            r"EB closed-form:" "\n" r"$\omega \cdot \hat\alpha^{\mathrm{OLS}}_j + (1-\omega)\alpha_{\mathrm{pop}}$",
            "per-annotator\n" r"$(\alpha_j,\beta_j)$",
            r"population slope $\alpha_{\mathrm{pop}}$",
        ),
        (
            "T2 (MDP RAC)",
            "vtrace",
            r"V-trace IS clip + forward-inject:" "\n" r"$\rho^{\mathrm{clip}} = \min(\bar\rho,\frac{\pi_t}{\pi_{t-\Delta}})$",
            "behaviour-policy\nversion at delay $\Delta$",
            r"current-policy gradient $\nabla J(\pi_{\theta_t})$",
        ),
        (
            "T3 (OASST2 /\nMultiPref)",
            "blup",
            r"REML BLUP + within-author" "\n" r"permutation test",
            r"per-author random" "\n" r"intercept $\alpha_j$",
            r"population intercept $\mu_\alpha$",
        ),
    ]

    # Draw each row
    row_ys = [2.45, 1.55, 0.55]
    row_h = 0.86
    for (y, rdata) in zip(row_ys, rows):
        label, family, est_form, re_mod, shrink_target = rdata
        fill = COLOR_BLUP if family == "blup" else COLOR_VTRACE

        # Row background
        rect = mpatches.FancyBboxPatch(
            (-0.05, y - row_h / 2), 4.65, row_h,
            boxstyle="round,pad=0.02",
            linewidth=0.6, edgecolor="#888888",
            facecolor=fill, alpha=0.18,
        )
        ax.add_patch(rect)

        # Left label bar (family color)
        left = mpatches.Rectangle(
            (-0.05, y - row_h / 2), 0.08, row_h,
            linewidth=0.0, facecolor=fill, alpha=0.9,
        )
        ax.add_patch(left)

        # Track name
        ax.text(col_x[0], y, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="black")
        # Estimator form
        ax.text(col_x[1], y, est_form, ha="center", va="center",
                fontsize=8.5, color="#222")
        # Random effect
        ax.text(col_x[2], y, re_mod, ha="center", va="center",
                fontsize=8.5, color="#222")
        # Shrinkage target
        ax.text(col_x[3], y, shrink_target, ha="center", va="center",
                fontsize=8.5, color="#222")

    # Family legend
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_BLUP, alpha=0.6,
                       label="Family (a): BLUP / EB shrinkage "
                             "(T1 + T3)"),
        mpatches.Patch(facecolor=COLOR_VTRACE, alpha=0.6,
                       label="Family (b): V-trace IS correction (T2)"),
    ]
    ax.legend(handles=legend_handles,
              loc="lower center", fontsize=9,
              ncol=2, bbox_to_anchor=(0.5, -0.06),
              framealpha=0.95)

    ax.set_title(
        "B. Two distinct instantiations, NOT a single estimator",
        fontsize=12, fontweight="bold", loc="left",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.4),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    fig.subplots_adjust(wspace=0.12)

    plot_panel_a(axes[0])
    plot_panel_b(axes[1])

    fig.suptitle(
        "Partial-pooling umbrella with two mathematically distinct instantiations",
        fontsize=13, fontweight="bold", y=1.005,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=args.dpi,
                bbox_inches="tight", facecolor="white")
    fig.savefig(args.output_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Figure 2 written to:\n  PNG: {args.output_png}\n  SVG: {args.output_svg}")


if __name__ == "__main__":
    main()
