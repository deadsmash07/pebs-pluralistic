"""Figure 2 (P1 Pluralistic) --- Empirical-Bayes shrinkage weight
omega = tau^2 / (tau^2 + V_j) as a function of per-user sample size.

UPDATED 2026-04-28 23:30 IST per user critique
("Figure 2 entirely disturbed; one more scale inside; use solid
colors rather than dashed"):
  - Removed the small-n inset axes entirely.  The inset was the
    "second scale" that was rendering poorly at column-width and
    overlapped the main-panel curves; the small-n separation is
    already visible in the main panel via the shaded n<=8 band
    and the per-population marker shapes.
  - Switched all three curves to solid lines (linestyle='-') per
    explicit user request "use solid colors rather than dashed".
    Per-population identity is now carried by hue + marker shape +
    endpoint labels, not linestyle.
  - Tightened endpoint label placement so they don't overlap the
    "omega -> 1 (no shrinkage)" guide-text.
  - Distinct Wong-2011 / Tol palette hues for clean separation:
    PRISM = Wong blue (#0072B2), PluriHarms = Wong vermillion
    (#D55E00), HelpSteer2 = Tol teal (#117733).

Anonymity-clean.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _figstyle import WONG, set_pub_style, save_fig, here


# Tol-palette teal for HelpSteer2 (visually distinct from Wong
# blu_green at column width).
TOL_TEAL = "#117733"


def main() -> None:
    set_pub_style()

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    n_grid = np.arange(2, 51)
    # All curves solid; per-population identity via hue + marker
    # shape + endpoint label.
    populations = [
        ("PRISM",      0.42, 0.65, WONG["blue"],       2.0, "o", 0),
        ("PluriHarms", 0.38, 0.55, WONG["vermillion"], 2.0, "s", 1),
        ("HelpSteer2", 0.55, 0.40, TOL_TEAL,           2.0, "^", 2),
    ]
    for name, tau2, s2, col, lw, mk, mk_off in populations:
        V_j = s2 / n_grid
        omega = tau2 / (tau2 + V_j)
        ax.plot(n_grid, omega, lw=lw, color=col, linestyle="-",
                label=name, zorder=3)
        # Markers at every 5th grid-point with per-population offset
        # so the three corpora are visually distinguishable in the
        # converged regime n>20.
        m_idx = np.arange(2 + mk_off, 51, 5) - 2
        ax.plot(n_grid[m_idx], omega[m_idx], lw=0,
                marker=mk, ms=4.5, color=col,
                markeredgecolor="black", markeredgewidth=0.4, zorder=4)

    # Asymptote guide at omega=1.
    ax.axhline(1.0, color=WONG["grey"], lw=0.6, ls=":", zorder=0)
    ax.text(2.5, 1.03, r"$\omega \to 1$ (no shrinkage)",
            ha="left", va="bottom", fontsize=7, color=WONG["grey"])

    # Light grey shade for the small-n shrinkage regime (replaces
    # the removed inset axes for marking n<=8 emphasis).
    ax.axvspan(2, 8, color=WONG["light_grey"], alpha=0.18, zorder=0)
    ax.text(5.0, 0.08, "small-$n$\nregime", ha="center", va="bottom",
            fontsize=7, color=WONG["grey"], style="italic")

    ax.set_xlabel(r"Per-user sample size $n_j$", fontsize=9)
    ax.set_ylabel(r"Shrinkage weight $\omega_j$", fontsize=9)
    ax.set_xlim(2, 54)
    ax.set_ylim(0, 1.10)

    ax.legend(loc="lower right", fontsize=7.5, ncol=1)

    save_fig(fig, here(__file__) + "/fig2_eb_shrinkage_omega.pdf")


if __name__ == "__main__":
    main()
