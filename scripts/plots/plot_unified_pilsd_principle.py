"""Cross-track unification figure for the PILSD 3-track RLHF paper.

This script builds the single capstone figure that communicates the paper's
ONE-LINE thesis:

    PILSD = partial-pooling / random-effect accounting dominates naive
    aggregation across three RLHF settings:
      (A) reward-model calibration              (Track 1, PRISM)
      (B) asynchronous credit assignment        (Track 2, 3-state MDP)
      (C) longitudinal drift detection          (Track 3, reviewer cohort)

Panels
------
A. Track 1 learning curve: mean held-out RMSE vs per-user budget k.
   Lines: naive OLS (red, with IQR band) and EB-shrunk (teal). Horizontal
   references: pop-slope (black dotted), no-calib (gray dotted). Log-x scale.
   Annotations mark the break-even budgets (k=5 shrunk, k=20 naive).

B. Track 2 gradient-bias bars: ||E[grad] - grad_true|| across delay horizons
   Delta in {5, 20, 50}. Bars: naive r_fast-only (red), RAC (teal).
   Log-y scale. Mean reduction factor (47.9x) annotated.

C. Track 3 composition-shift diagnostics. Left sub-group: FPR on pure
   composition shift (beta_true = 0). Right sub-group: fraction of seeds
   recovering correct positive sign under composition + genuine +0.005/mo
   drift. Grouped bars per detector {MixedLM+perm (teal), NaiveOLS (red),
   PageHinkley (gray)}.

All three panels use the consistent color code:
    red  (#d62728) = naive aggregation, OLS, r_fast-only, monthly aggregate
    teal (#17becf) = PILSD (partial-pooling, EB shrinkage, RAC, MixedLM)
    gray (#7f7f7f) = PageHinkley (third detector)
    black dotted   = population-slope baseline
    gray  dotted   = no-calibration baseline

References
----------
- Gelman & Hill 2007, "Data Analysis Using Regression and Multilevel / Hierarchical
  Models", Cambridge UP, Chapter 12 (partial pooling / random-effects).
- Pinheiro & Bates 2000, "Mixed-Effects Models in S and S-PLUS", Springer,
  Chapter 2 (linear mixed-effects; random intercept and slope).
- Simpson 1951, "The Interpretation of Interaction in Contingency Tables",
  J. Royal Statistical Society 13(2):238-241 (Simpson's paradox origin).
- Our Theorem A3 (delay-aware consistency of RAC's forward-injected
  correction under (U-i)); see phase2_final plan and theorem_review
  round-1/2/3 notes in MEMORY.md.
- Our merged-paper Theorem A2 (Pinsker KL-TV bound governing RAC's
  bias-variance tradeoff under IS-clipping).

Data inputs (paths via argparse; all JSON)
------------------------------------------
- Track 1 naive-OLS cold start:
  1_Causal_RLHF/results/track1_coldstart_curve_randomk.json
- Track 1 shrunk cold start:
  1_Causal_RLHF/results/track1_coldstart_shrinkage.json
- Track 1 H2e 4-arm comparison:
  1_Causal_RLHF/results/track1_user_score_mse_shrunk.json
- Track 2 closed-form MDP validation:
  2_Delay_Aware_RLHF/results/track2_rac_gradient_validation/validation.json
- Track 3 Simpson stress (pure composition shift, true drift = 0):
  3_PILSD_Standalone/results/track3_simpson_stress/simpson_stress.json
- Track 3 mixed drift (composition + genuine drift):
  3_PILSD_Standalone/results/track3_mixed_drift/mixed_drift.json

Outputs
-------
- 3_PILSD_Standalone/results/figs/unified_pilsd_principle.png  (150 DPI)
- 3_PILSD_Standalone/results/figs/unified_pilsd_principle.svg  (vector)
- 3_PILSD_Standalone/results/unified_pilsd_summary.json       (machine-readable)

Usage
-----
    python3 scripts/plot_unified_pilsd_principle.py

All paths default to repo-relative defaults; override via --track1-* / --track2-* /
--track3-* / --output-* flags.

No seaborn dependency (keeps the pinned numpy<2 / torch 2.5 / transformers
4.46.3 / trl 0.12.2 / peft 0.14.0 environment lean). Matplotlib only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


# -----------------------------------------------------------------------------
# Default paths (repo-relative from the current workspace)
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRACK1_NAIVE = REPO_ROOT / "1_Causal_RLHF/results/track1_coldstart_curve_randomk.json"
DEFAULT_TRACK1_SHRUNK = REPO_ROOT / "1_Causal_RLHF/results/track1_coldstart_shrinkage.json"
DEFAULT_TRACK1_H2E = REPO_ROOT / "1_Causal_RLHF/results/track1_user_score_mse_shrunk.json"
DEFAULT_TRACK2 = REPO_ROOT / "2_Delay_Aware_RLHF/results/track2_rac_gradient_validation/validation.json"
DEFAULT_TRACK3_SIMPSON = REPO_ROOT / "3_PILSD_Standalone/results/track3_simpson_stress/simpson_stress.json"
DEFAULT_TRACK3_MIXED = REPO_ROOT / "3_PILSD_Standalone/results/track3_mixed_drift/mixed_drift.json"

DEFAULT_OUTPUT_PNG = REPO_ROOT / "3_PILSD_Standalone/results/figs/unified_pilsd_principle.png"
DEFAULT_OUTPUT_SVG = REPO_ROOT / "3_PILSD_Standalone/results/figs/unified_pilsd_principle.svg"
DEFAULT_SUMMARY_JSON = REPO_ROOT / "3_PILSD_Standalone/results/unified_pilsd_summary.json"

# -----------------------------------------------------------------------------
# Consistent color palette across panels
# -----------------------------------------------------------------------------
COLOR_NAIVE = "#d62728"   # strong red         -> naive aggregation / OLS / r_fast only
COLOR_PILSD = "#17becf"   # strong teal        -> partial pooling / shrinkage / RAC / MixedLM
COLOR_POP = "#000000"     # black              -> population-slope baseline
COLOR_NOCAL = "#888888"   # mid-gray           -> no-calibration baseline
COLOR_PH = "#7f7f7f"      # gray               -> PageHinkley third detector


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the cross-track unified PILSD principle figure.",
    )
    p.add_argument("--track1-naive", type=Path, default=DEFAULT_TRACK1_NAIVE)
    p.add_argument("--track1-shrunk", type=Path, default=DEFAULT_TRACK1_SHRUNK)
    p.add_argument("--track1-h2e", type=Path, default=DEFAULT_TRACK1_H2E)
    p.add_argument("--track2", type=Path, default=DEFAULT_TRACK2)
    p.add_argument("--track3-simpson", type=Path, default=DEFAULT_TRACK3_SIMPSON)
    p.add_argument("--track3-mixed", type=Path, default=DEFAULT_TRACK3_MIXED)
    p.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    p.add_argument("--output-svg", type=Path, default=DEFAULT_OUTPUT_SVG)
    p.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Panel A : Track 1 cold-start learning curve
# -----------------------------------------------------------------------------
def plot_panel_a(ax, naive: Dict[str, Any], shrunk: Dict[str, Any]) -> Dict[str, Any]:
    """Held-out RMSE vs labelled-utterances budget k.

    Curves
    ------
    - naive OLS     (red,    solid, with IQR band)
    - EB shrunk     (teal,   solid, with IQR band)
    - pop slope     (black,  dotted horizontal reference)
    - no calib      (gray,   dotted horizontal reference)

    Annotations
    -----------
    - vertical dashed line + label for naive break-even k=20
    - vertical dashed line + label for shrunk break-even k=5
    """
    import numpy as np

    budgets = naive["curve"]["budgets"]
    per_k_naive = naive["curve"]["per_k"]
    per_k_shrunk = shrunk["per_k"]

    pop_slope_rmse = naive["holdout_rmse_baselines"]["pop_slope_mean"]
    no_calib_rmse = naive["holdout_rmse_baselines"]["no_calib_mean"]

    # Extract per-k stats
    naive_mean = np.array([per_k_naive[str(k)]["rmse_pilsd"]["mean"] for k in budgets])
    naive_median = np.array([per_k_naive[str(k)]["rmse_pilsd"]["median"] for k in budgets])
    naive_p25 = np.array([per_k_naive[str(k)]["rmse_pilsd"]["p25"] for k in budgets])
    naive_p75 = np.array([per_k_naive[str(k)]["rmse_pilsd"]["p75"] for k in budgets])

    shrunk_mean = np.array([per_k_shrunk[str(k)]["shrunk_mean"] for k in budgets])
    shrunk_median = np.array([per_k_shrunk[str(k)]["shrunk_median"] for k in budgets])

    # We use MEDIAN as the robust centre (means blow up at k=2 for OLS: 135
    # because a small fraction of users hit singular design matrices and the
    # OLS slope is essentially unbounded -- this is precisely the failure
    # mode shrinkage corrects). Plot median; show a shaded IQR band with the
    # p25/p75 of the naive curve to make the variance explosion visible.
    ax.fill_between(
        budgets, naive_p25, naive_p75,
        color=COLOR_NAIVE, alpha=0.15, linewidth=0,
        label="Naive OLS per-user IQR",
    )
    ax.plot(
        budgets, naive_median,
        color=COLOR_NAIVE, linewidth=2.2, marker="o", markersize=7,
        label="Naive OLS (median)",
    )
    ax.plot(
        budgets, shrunk_median,
        color=COLOR_PILSD, linewidth=2.4, marker="s", markersize=7,
        label="EB-shrunk (median)  [PILSD]",
    )

    # Horizontal baselines
    ax.axhline(pop_slope_rmse, color=COLOR_POP, linestyle=":", linewidth=1.4,
               label=f"Pop-slope baseline ({pop_slope_rmse:.2f})")
    ax.axhline(no_calib_rmse, color=COLOR_NOCAL, linestyle=":", linewidth=1.2,
               label=f"No-calib baseline ({no_calib_rmse:.2f})")

    # Break-even annotations
    ax.axvline(5, color=COLOR_PILSD, linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axvline(20, color=COLOR_NAIVE, linestyle="--", linewidth=1.0, alpha=0.6)
    ax.annotate(
        "shrunk break-even\nk=5",
        xy=(5, pop_slope_rmse), xytext=(2.4, pop_slope_rmse - 5.2),
        fontsize=9, color=COLOR_PILSD,
        arrowprops=dict(arrowstyle="->", color=COLOR_PILSD, lw=1.0),
        ha="center",
    )
    ax.annotate(
        "naive break-even\nk=20",
        xy=(20, pop_slope_rmse), xytext=(14.5, pop_slope_rmse + 4.0),
        fontsize=9, color=COLOR_NAIVE,
        arrowprops=dict(arrowstyle="->", color=COLOR_NAIVE, lw=1.0),
        ha="center",
    )

    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(k) for k in budgets])
    ax.set_xlabel("Budget k  (labelled utterances per held-out user)", fontsize=10)
    ax.set_ylabel("Held-out RMSE  (user score, 0-100 scale)", fontsize=10)
    ax.set_title("A. Track 1 - Reward-model calibration on PRISM\n"
                 "Shrinkage cuts break-even from k=20 to k=5",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(17, 58)
    ax.grid(True, which="both", alpha=0.25, linestyle="-", linewidth=0.4)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    return {
        "break_even_k_naive": 20,
        "break_even_k_shrunk": 5,
        "pop_slope_rmse": pop_slope_rmse,
        "no_calib_rmse": no_calib_rmse,
        "naive_median_k5": float(naive_median[budgets.index(5)]),
        "shrunk_median_k5": float(shrunk_median[budgets.index(5)]),
    }


# -----------------------------------------------------------------------------
# Panel B : Track 2 gradient-bias bars
# -----------------------------------------------------------------------------
def plot_panel_b(ax, track2: Dict[str, Any]) -> Dict[str, Any]:
    """Policy-gradient bias ||E[grad] - grad_true||  vs delay horizon Delta.

    Bars
    ----
    - naive r_fast-only        (red)
    - RAC forward-injected      (teal)

    Log-y axis. Annotate each pair with the per-Delta reduction factor and
    the global mean reduction (47.9x).
    """
    import numpy as np

    deltas = track2["config"]["deltas"]
    per_delta = track2["per_delta"]
    bias_naive = np.array([per_delta[str(d)]["bias_naive"] for d in deltas])
    bias_rac = np.array([per_delta[str(d)]["bias_rac"] for d in deltas])
    reduction = np.array([per_delta[str(d)]["reduction_factor"] for d in deltas])
    vif_per_delta = np.array([per_delta[str(d)]["vif"] for d in deltas])

    mean_reduction = track2["aggregate"]["mean_reduction_factor"]
    min_reduction = track2["aggregate"]["min_reduction_factor"]
    mean_vif = float(np.mean(vif_per_delta))
    max_vif = track2["aggregate"]["max_vif"]

    x = np.arange(len(deltas), dtype=float)
    bar_w = 0.36
    b1 = ax.bar(x - bar_w / 2, bias_naive, width=bar_w,
                color=COLOR_NAIVE, edgecolor="black", linewidth=0.6,
                label="Naive r_fast-only PG")
    b2 = ax.bar(x + bar_w / 2, bias_rac, width=bar_w,
                color=COLOR_PILSD, edgecolor="black", linewidth=0.6,
                label="RAC forward-injected  [PILSD]")

    # Annotate each pair with the reduction factor, placed above the red bar
    for xi, bn, br, rf in zip(x, bias_naive, bias_rac, reduction):
        top = bn * 1.35
        ax.annotate(
            f"{rf:.1f}x",
            xy=(xi, top),
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            color="black",
        )

    # Annotate bar heights (naive gets the big number; rac gets the small one)
    for rect, val in zip(b1.patches, bias_naive):
        ax.text(rect.get_x() + rect.get_width() / 2,
                val * 1.02, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, color=COLOR_NAIVE)
    for rect, val in zip(b2.patches, bias_rac):
        ax.text(rect.get_x() + rect.get_width() / 2,
                val * 1.08, f"{val:.4f}",
                ha="center", va="bottom", fontsize=8, color="#087a88")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Delta = {d}" for d in deltas])
    ax.set_xlabel("Delay horizon  Delta  (optimizer steps)", fontsize=10)
    ax.set_ylabel(r"Gradient-bias  $\|\mathbb{E}[\widehat\nabla J] - \nabla J_{\mathrm{true}}\|_2$",
                  fontsize=10)
    ax.set_title("B. Track 2 - Async credit on 3-state MDP\n"
                 f"RAC reduces bias {mean_reduction:.1f}x on average (min {min_reduction:.1f}x)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(1e-4, 2e-1)

    # Shaded reduction arrow summary
    ax.text(
        0.03, 0.02,
        f"mean reduction = {mean_reduction:.1f}x   |   mean VIF = {mean_vif:.2f}",
        transform=ax.transAxes,
        fontsize=9, color="black",
        bbox=dict(facecolor="white", edgecolor="#bbbbbb", boxstyle="round,pad=0.3", alpha=0.92),
    )

    ax.grid(True, which="both", axis="y", alpha=0.3, linestyle="-", linewidth=0.4)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    return {
        "mean_bias_reduction_factor": mean_reduction,
        "min_bias_reduction_factor": min_reduction,
        "mean_vif": mean_vif,
        "max_vif": max_vif,
        "bias_naive_mean": float(np.mean(bias_naive)),
        "bias_rac_mean": float(np.mean(bias_rac)),
    }


# -----------------------------------------------------------------------------
# Panel C : Track 3 composition-shift diagnostics
# -----------------------------------------------------------------------------
def plot_panel_c(ax, simpson: Dict[str, Any], mixed: Dict[str, Any]) -> Dict[str, Any]:
    """Two grouped bar-blocks:

    Left block  (pure composition shift, true drift = 0):
        y = detector false-positive rate (fraction of seeds that reject
        H0 = "no drift" at alpha = 0.05).  Lower is better.

    Right block (composition + true drift = +0.005/mo):
        y = fraction of seeds with *correctly-signed* estimate (positive).
        Higher is better. NaiveOLS gets sign wrong 10/10 times;
        MixedLM gets it right 8/10.

    Three detectors per block, color-coded as
        MixedLM+perm  = teal   (#17becf)
        NaiveOLS      = red    (#d62728)
        PageHinkley   = gray   (#7f7f7f)
    """
    import numpy as np

    fpr = simpson["fpr"]
    fpr_ml = fpr["mixedlm_perm"]
    fpr_naive = fpr["naive_ols"]
    fpr_ph = fpr["pagehinkley"]

    # Find the cell with true_drift = +0.005
    pos_cell = next(c for c in mixed["cells"] if abs(c["true_drift"] - 0.005) < 1e-9)
    n_seeds = mixed["config"]["n_seeds"]
    sign_ml = pos_cell["mixedlm_correct_sign"]
    sign_naive = pos_cell["naive_correct_sign"]
    # Track3 mixed_drift file only records sign for mixedlm/naive; PageHinkley
    # is a change-point test and doesn't emit a signed estimate, so we treat
    # PH sign-correctness as NaN and render it as a hatched "N/A" bar.
    # (It is still plotted for completeness of the detector trio.)

    x_fpr = np.arange(3, dtype=float) - 0.20
    x_sign = np.arange(3, dtype=float) + 3.20

    block_w = 0.30

    colors = [COLOR_PILSD, COLOR_NAIVE, COLOR_PH]
    detectors = ["MixedLM+perm", "NaiveOLS", "PageHinkley"]
    fpr_vals = [fpr_ml, fpr_naive, fpr_ph]
    sign_vals = [sign_ml / n_seeds, sign_naive / n_seeds, float("nan")]  # NaN -> hatch

    # Left block (FPR)
    bars_fpr = ax.bar(x_fpr, fpr_vals, width=block_w,
                      color=colors, edgecolor="black", linewidth=0.6)
    # Right block (sign-correctness). Plot real bars for ML and naive, hatched for PH
    for xi, val, clr, det in zip(x_sign, sign_vals, colors, detectors):
        if np.isnan(val):
            # Hatched placeholder for sign-undefined detector
            ax.bar(xi, 1.0, width=block_w,
                   color="white", edgecolor="black", linewidth=0.6,
                   hatch="//")
            ax.text(xi, 0.50, "N/A\n(no signed\nestimate)",
                    ha="center", va="center", fontsize=8, color="#444444")
        else:
            ax.bar(xi, val, width=block_w,
                   color=clr, edgecolor="black", linewidth=0.6)

    # Numeric labels on top of each bar (in data coords, always inside the plot area)
    for xi, val in zip(x_fpr, fpr_vals):
        ax.text(xi, val + 0.025, f"{val*100:.0f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    for xi, val in zip(x_sign[:2], sign_vals[:2]):
        ax.text(xi, val + 0.025, f"{val*100:.0f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Block separator labels placed BELOW the x-axis tick labels (negative y
    # in data coords -> below x-axis); use a small gap so they're clearly
    # the block descriptors.
    ax.text(np.mean(x_fpr), 1.48,
            "Pure composition shift\n(true beta = 0)  -  lower FPR better",
            ha="center", va="top", fontsize=8.5, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="#bbbbbb",
                      boxstyle="round,pad=0.2", alpha=0.95))
    ax.text(np.mean(x_sign), 1.48,
            "Composition + genuine drift\n(true beta = +0.005/mo)  -  higher sign-correct better",
            ha="center", va="top", fontsize=8.5, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="#bbbbbb",
                      boxstyle="round,pad=0.2", alpha=0.95))

    # Vertical separator between blocks
    ax.axvline(2.5, color="#cccccc", linestyle="-", linewidth=1.0)

    # x-tick labels per detector
    xticks = list(x_fpr) + list(x_sign)
    xticklabels = detectors + detectors
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=8.5, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.70)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel("Rate across seeds  (0 - 1)", fontsize=10)
    ax.set_title("C. Track 3 - Longitudinal drift detection\n"
                 "MixedLM 0% FPR; naive reports wrong sign under real drift",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linestyle="-", linewidth=0.4)

    # Custom legend: three detectors. Anchor in the inner middle between
    # the two bar groups (above the dividing vertical line) at y=1.08; this
    # region is empty (no bars cross y=1.0) so no overlap is possible.
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=COLOR_PILSD, label="MixedLM+perm  [PILSD]"),
        mpatches.Patch(color=COLOR_NAIVE, label="NaiveOLS  (monthly aggregate)"),
        mpatches.Patch(color=COLOR_PH, label="PageHinkley  (change-point)"),
    ]
    ax.legend(handles=legend_handles, loc="center", fontsize=8,
              framealpha=0.95, ncol=1, bbox_to_anchor=(0.5, 1.08),
              bbox_transform=ax.transData)

    return {
        "fpr_mixedlm": fpr_ml,
        "fpr_naive": fpr_naive,
        "fpr_ph": fpr_ph,
        "sign_correct_mixedlm": sign_ml,
        "sign_correct_naive": sign_naive,
        "n_seeds_mixed": n_seeds,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # --- Load inputs ----------------------------------------------------------
    naive = _load_json(args.track1_naive)
    shrunk = _load_json(args.track1_shrunk)
    h2e = _load_json(args.track1_h2e)
    track2 = _load_json(args.track2)
    simpson = _load_json(args.track3_simpson)
    mixed = _load_json(args.track3_mixed)

    # --- Plot -----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Set a consistent sans-serif font that ships with mpl so the figure is
    # deterministic across macOS / Linux / headless CI.
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    })

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.0))
    # A tiny extra horizontal spacing lets Panel C's block titles breathe
    # without the vector artists colliding with adjacent panels.
    fig.subplots_adjust(wspace=0.32)

    a_summary = plot_panel_a(axes[0], naive, shrunk)
    b_summary = plot_panel_b(axes[1], track2)
    c_summary = plot_panel_c(axes[2], simpson, mixed)

    # Figure title (centered, 14pt bold, single line)
    fig.suptitle(
        "PILSD: partial-pooling / random-effect accounting dominates naive aggregation "
        "across three RLHF settings",
        fontsize=14, fontweight="bold", y=0.995,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.955])

    # --- Save PNG + SVG -------------------------------------------------------
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- Paper-ready JSON summary --------------------------------------------
    improvement = h2e["relative_improvement_vs_pop_pct"]
    summary = {
        "track1": {
            "break_even_k_naive": a_summary["break_even_k_naive"],
            "break_even_k_shrunk": a_summary["break_even_k_shrunk"],
            "H2e_improvement_shrunk_pct": round(improvement["shrunk"], 2),
            "H2e_improvement_naive_pct": round(improvement["naive_ols"], 2),
            "H2e_n_users": h2e["n_users"],
            "H2e_k_folds": h2e["k_folds"],
            "pop_slope_rmse": round(a_summary["pop_slope_rmse"], 3),
            "no_calib_rmse": round(a_summary["no_calib_rmse"], 3),
            "wilcoxon_p_shrunk_vs_pop": h2e["comparisons"]["shrunk_vs_pop"]["wilcoxon_p"],
            "wilcoxon_p_shrunk_vs_ols": h2e["comparisons"]["shrunk_vs_ols"]["wilcoxon_p"],
        },
        "track2": {
            "mean_bias_reduction_factor": round(b_summary["mean_bias_reduction_factor"], 1),
            "min_bias_reduction_factor": round(b_summary["min_bias_reduction_factor"], 1),
            "mean_vif": round(b_summary["mean_vif"], 2),
            "max_vif": round(b_summary["max_vif"], 2),
            "deltas": track2["config"]["deltas"],
            "n_trials": track2["config"]["n_trials"] * len(track2["config"]["seeds"]),
            "trajectory_len": track2["config"]["trajectory_len"],
            "bias_naive_mean": round(b_summary["bias_naive_mean"], 5),
            "bias_rac_mean": round(b_summary["bias_rac_mean"], 5),
        },
        "track3": {
            "fpr_mixedlm": c_summary["fpr_mixedlm"],
            "fpr_naive": c_summary["fpr_naive"],
            "fpr_ph": c_summary["fpr_ph"],
            "sign_correct_naive": c_summary["sign_correct_naive"],
            "sign_correct_mixedlm": c_summary["sign_correct_mixedlm"],
            "n_seeds_simpson": simpson["config"]["n_seeds"],
            "n_seeds_mixed": c_summary["n_seeds_mixed"],
            "true_drift_mixed": 0.005,
        },
        "paper_caption": (
            "Figure N. PILSD's partial-pooling principle dominates naive "
            "aggregation across three RLHF settings. "
            "(A) On PRISM held-out-user score prediction, empirical-Bayes "
            "shrinkage (tau^2/(tau^2+V) blend of per-user OLS with population "
            "slope) reduces the break-even sample size from k=20 to k=5 "
            "labeled utterances and improves the within-user held-out RMSE "
            "headline from 7.02% to 8.58% over a pop-slope baseline "
            "(N=1394 users, k=5 CV, Wilcoxon p<10^-108). "
            "(B) On a 3-state closed-form MDP with delayed slow rewards, RAC "
            "reduces policy-gradient bias by 47.9x on average over naive "
            "r_fast-only PG across delay horizons Delta in {5, 20, 50} "
            "(3000 MC trials per cell). "
            "(C) On a synthetic reviewer-cohort DGP with composition shift "
            "and zero true drift, MixedLM+within-author permutation null "
            "maintains 0% FPR while NaiveOLS and PageHinkley on monthly "
            "aggregates false-fire 100% of the time (20 seeds); under "
            "composition + genuine +0.005/mo drift, NaiveOLS reports "
            "inverted-sign (negative) in 10/10 seeds while MixedLM correctly "
            "recovers positive sign in 8/10."
        ),
        "sources": {
            "track1_naive": str(args.track1_naive),
            "track1_shrunk": str(args.track1_shrunk),
            "track1_h2e": str(args.track1_h2e),
            "track2": str(args.track2),
            "track3_simpson": str(args.track3_simpson),
            "track3_mixed": str(args.track3_mixed),
        },
        "outputs": {
            "png": str(args.output_png),
            "svg": str(args.output_svg),
        },
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w") as f:
        json.dump(summary, f, indent=2)

    # Short console report
    print("=" * 72)
    print("Unified PILSD principle figure written to:")
    print(f"  PNG : {args.output_png}")
    print(f"  SVG : {args.output_svg}")
    print(f"  JSON: {args.summary_json}")
    print("=" * 72)
    print("Track 1 :  break-even k   shrunk={}   naive={}   "
          "H2e shrunk={:.2f}%   naive={:.2f}%".format(
              summary["track1"]["break_even_k_shrunk"],
              summary["track1"]["break_even_k_naive"],
              summary["track1"]["H2e_improvement_shrunk_pct"],
              summary["track1"]["H2e_improvement_naive_pct"],
          ))
    print("Track 2 :  mean reduction = {}x    min = {}x    mean VIF = {}".format(
        summary["track2"]["mean_bias_reduction_factor"],
        summary["track2"]["min_bias_reduction_factor"],
        summary["track2"]["mean_vif"],
    ))
    print("Track 3 :  FPR  ml={}  naive={}  ph={}    sign-correct  ml={}/{}  naive={}/{}".format(
        summary["track3"]["fpr_mixedlm"],
        summary["track3"]["fpr_naive"],
        summary["track3"]["fpr_ph"],
        summary["track3"]["sign_correct_mixedlm"],
        summary["track3"]["n_seeds_mixed"],
        summary["track3"]["sign_correct_naive"],
        summary["track3"]["n_seeds_mixed"],
    ))


if __name__ == "__main__":
    main()
