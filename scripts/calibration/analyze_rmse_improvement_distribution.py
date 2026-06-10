"""Per-user RMSE improvement distribution — the paper figure that answers
"is the 79.2% win rate driven by extreme users, or is the effect well-
distributed?".

Outputs
-------
- `results/track1_rmse_improvement_distribution.json` — summary stats
- `results/figs/track1_rmse_improvement_hist.png` — histogram figure

Reviewer-proof questions this addresses
---------------------------------------
1. **Tail risk**: For the 20.8% of users where PEBS LOSES, how much does
   it lose by? Is there a catastrophic-case user?
2. **Monotonicity in n**: Do users with more labeled utterances benefit
   more (confirming that the per-user calibration is picking up real
   per-user structure, not overfitting noise)?
3. **Distribution shape**: Normal? Heavy-tailed? Bimodal (suggesting two
   user-types)?
4. **Effect-size distribution**: What's the per-user Δ-RMSE at the
   25th, 50th, 75th, 95th percentiles?

References
----------
- Cohen 1988 effect-size interpretation for within-subject improvements
- Hastie et al. 2009 ESL §7.12 for interpreting CV-based error distributions
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per-user-parquet",
                   default="results/track1_user_score_mse.parquet")
    p.add_argument("--output-json",
                   default="results/track1_rmse_improvement_distribution.json")
    p.add_argument("--figure-path",
                   default="results/figs/track1_rmse_improvement_hist.png")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.per_user_parquet)
    print(f"[load] {len(df)} users")

    # Per-user improvements (positive means PEBS is BETTER = smaller RMSE)
    df["improvement_pebs_vs_pop_slope"] = df["rmse_pop_slope"] - df["rmse_pebs"]
    df["improvement_pebs_vs_no_calib"] = df["rmse_no_calib"] - df["rmse_pebs"]
    df["improvement_rel_pebs_vs_pop_slope_pct"] = (
        100 * df["improvement_pebs_vs_pop_slope"]
        / df["rmse_pop_slope"].replace(0, np.nan)
    )

    imp = df["improvement_pebs_vs_pop_slope"].to_numpy()
    imp_rel = df["improvement_rel_pebs_vs_pop_slope_pct"].to_numpy()
    n_per_user = df["n"].to_numpy()

    # Tail-risk analysis
    n_worse = int((imp < 0).sum())
    n_better = int((imp > 0).sum())
    n_tied = int((imp == 0).sum())
    frac_worse = n_worse / len(df)
    worst_user = df.sort_values("improvement_pebs_vs_pop_slope").head(5)
    best_user = df.sort_values("improvement_pebs_vs_pop_slope").tail(5)

    # Percentile bands
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    imp_pct = {f"p{p}": float(np.percentile(imp, p)) for p in percentiles}
    imp_rel_pct = {f"p{p}": float(np.percentile(imp_rel, p)) for p in percentiles}

    # Correlation with n_utterances (does more data → more benefit?)
    rho_spearman = stats.spearmanr(n_per_user, imp)
    rho_pearson = stats.pearsonr(n_per_user, imp)

    # Monotone trend: bucket users by n_utterances (deciles) and compute
    # mean Δ-RMSE per bucket
    q_bins = pd.qcut(df["n"], q=10, labels=False, duplicates="drop")
    bucket_means = df.groupby(q_bins)["improvement_pebs_vs_pop_slope"].mean().to_dict()
    bucket_ns = df.groupby(q_bins)["n"].mean().to_dict()

    # Normality tests
    sw_stat, sw_p = stats.shapiro(imp) if len(imp) <= 5000 else (np.nan, np.nan)
    # Excess kurtosis > 0 → heavy-tailed
    kurt = float(stats.kurtosis(imp, fisher=True))
    skew = float(stats.skew(imp))

    # Catastrophic-case thresholds (for reviewer-proof claims)
    n_lose_gt_1rmse = int((imp < -1.0).sum())
    n_lose_gt_5rmse = int((imp < -5.0).sum())
    n_gain_gt_1rmse = int((imp > 1.0).sum())
    n_gain_gt_5rmse = int((imp > 5.0).sum())

    out = {
        "n_users": int(len(df)),
        "mean_n_obs_per_user": float(df["n"].mean()),
        "mean_improvement_absolute": float(np.mean(imp)),
        "mean_improvement_relative_pct": float(np.nanmean(imp_rel)),
        "frac_users_better_with_pebs": float(n_better / len(df)),
        "frac_users_worse_with_pebs": float(frac_worse),
        "frac_users_tied": float(n_tied / len(df)),
        "percentiles_absolute": imp_pct,
        "percentiles_relative_pct": imp_rel_pct,
        "n_users_worse_by_gt_1_rmse": n_lose_gt_1rmse,
        "n_users_worse_by_gt_5_rmse": n_lose_gt_5rmse,
        "n_users_better_by_gt_1_rmse": n_gain_gt_1rmse,
        "n_users_better_by_gt_5_rmse": n_gain_gt_5rmse,
        "correlation_n_obs_with_improvement": {
            "spearman_rho": float(rho_spearman.statistic),
            "spearman_p": float(rho_spearman.pvalue),
            "pearson_r": float(rho_pearson.statistic),
            "pearson_p": float(rho_pearson.pvalue),
        },
        "decile_bucket_means_absolute_improvement": {
            f"decile_{k}": {
                "mean_n_obs": float(bucket_ns[k]),
                "mean_abs_improvement": float(v),
            }
            for k, v in bucket_means.items()
        },
        "distribution_shape": {
            "skewness": skew,
            "excess_kurtosis": kurt,
            "shapiro_wilk_stat": float(sw_stat) if not np.isnan(sw_stat) else None,
            "shapiro_wilk_p": float(sw_p) if not np.isnan(sw_p) else None,
        },
        "worst_5_users": worst_user[["user_id", "n", "rmse_pop_slope", "rmse_pebs",
                                     "improvement_pebs_vs_pop_slope"]].to_dict("records"),
        "best_5_users": best_user[["user_id", "n", "rmse_pop_slope", "rmse_pebs",
                                   "improvement_pebs_vs_pop_slope"]].to_dict("records"),
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, indent=2, default=str))
    print(f"[save] {args.output_json}")

    # Make the figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: histogram of absolute improvement
        ax = axes[0]
        ax.hist(imp, bins=60, color="#3b8ea5", alpha=0.8, edgecolor="black", linewidth=0.4)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="no improvement")
        ax.axvline(float(np.mean(imp)), color="black", linestyle="-", linewidth=1.3,
                   label=f"mean = {float(np.mean(imp)):.2f}")
        ax.axvline(float(np.median(imp)), color="black", linestyle=":",  linewidth=1.3,
                   label=f"median = {float(np.median(imp)):.2f}")
        ax.set_xlabel("Per-user Δ-RMSE (pop-slope − PEBS)\n"
                      "positive = PEBS is better")
        ax.set_ylabel("Number of users")
        ax.set_title(f"PRISM held-out user-score RMSE:\n"
                     f"PEBS vs global calibration ({len(df)} users)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.25)

        # Panel 2: improvement vs n_utterances (log-x)
        ax = axes[1]
        ax.scatter(df["n"], imp, s=5, alpha=0.35, color="#3b8ea5")
        ax.axhline(0, color="red", linestyle="--", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("log(n utterances per user)")
        ax.set_ylabel("Δ-RMSE")
        ax.set_title(f"Per-user gain vs sample size\n"
                     f"Spearman ρ = {rho_spearman.statistic:+.3f} (p = {rho_spearman.pvalue:.1e})")
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        Path(args.figure_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.figure_path, dpi=150, bbox_inches="tight")
        print(f"[save] {args.figure_path}")
    except Exception as e:
        print(f"[warn] figure save skipped: {e}")

    # Human-readable summary
    print(f"\n=== Per-user RMSE improvement distribution ===")
    print(f"  n_users={len(df)}, mean_n_obs/user={df['n'].mean():.1f}")
    print(f"  Δ-RMSE: mean={float(np.mean(imp)):+.3f}  median={float(np.median(imp)):+.3f}")
    print(f"  Relative: mean={float(np.nanmean(imp_rel)):+.2f}%  median={float(np.nanmedian(imp_rel)):+.2f}%")
    print(f"  Win rate: PEBS better {100*n_better/len(df):.1f}%, "
          f"worse {100*frac_worse:.1f}%, tied {100*n_tied/len(df):.1f}%")
    print(f"  Tails:")
    print(f"    users WORSE by >1 RMSE: {n_lose_gt_1rmse} ({100*n_lose_gt_1rmse/len(df):.1f}%)")
    print(f"    users WORSE by >5 RMSE: {n_lose_gt_5rmse} ({100*n_lose_gt_5rmse/len(df):.2f}%)")
    print(f"    users BETTER by >1 RMSE: {n_gain_gt_1rmse} ({100*n_gain_gt_1rmse/len(df):.1f}%)")
    print(f"    users BETTER by >5 RMSE: {n_gain_gt_5rmse} ({100*n_gain_gt_5rmse/len(df):.1f}%)")
    print(f"  Spearman n-obs vs improvement: ρ={rho_spearman.statistic:+.3f} (p={rho_spearman.pvalue:.1e})")
    print(f"  Distribution shape: skew={skew:+.2f}  excess-kurtosis={kurt:+.2f}")
    if sw_p is not None and not np.isnan(sw_p):
        print(f"  Shapiro-Wilk normality p={sw_p:.1e} "
              f"({'REJECT normality' if sw_p < 1e-4 else 'plausible normal'})")


if __name__ == "__main__":
    main()
