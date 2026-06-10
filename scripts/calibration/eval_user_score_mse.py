"""Within-user held-out MSE — the proper metric for PEBS linear calibration.

Last iteration's finding: pair accuracy is MONOTONE-INVARIANT within user, so
arm C (PEBS-corrected) can't improve over arm B on pair ranking by
construction. The metric doesn't test what linear calibration does.

This script tests what it DOES: predict a user's held-out score magnitude
from the RM score via per-user α_j, β_j. Compare three predictors:

  No-calib (baseline 1): user_score_hat = global β_0 + global β_1 · rm_score
  Pop-slope (baseline 2): same global slope + global intercept, applied to everyone
  PEBS (ours): user_score_hat = β̂_j + α̂_j · rm_score  (per-user fit)

Metric: RMSE on within-user held-out utterances via k-fold CV. RMSE IS
sensitive to scale calibration (pair accuracy is not).

Paper claim if PEBS wins: "Per-user linear calibration reduces held-out
user-score prediction RMSE by X% over a pooled-slope baseline on PRISM."

References:
  - Hastie et al. 2009 ESL §7.10 k-fold CV for prediction error
  - Shao 1993 k-fold consistency for CV-based model selection
  - Pinheiro & Bates 2000 §2 random slope + intercept
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
    p.add_argument("--scored-parquet", default="data/prism_rm_scored.parquet")
    p.add_argument("--min-obs-per-user", type=int, default=6,
                   help="Users with fewer labeled utterances get skipped (need room for k-fold).")
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-path", default="results/track1_user_score_mse.json")
    return p.parse_args()


def kfold_split(n: int, k: int, rng: np.random.Generator):
    """Deterministic k-fold index split."""
    idx = np.arange(n)
    rng.shuffle(idx)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        stop = (i + 1) * fold_size if i < k - 1 else n
        test_idx = idx[start:stop]
        train_idx = np.concatenate([idx[:start], idx[stop:]])
        folds.append((train_idx, test_idx))
    return folds


def ols_intercept_slope(x: np.ndarray, y: np.ndarray):
    """Returns (intercept, slope) via np.polyfit degree-1. Handles <2 points by
    falling back to (mean(y), 0).
    """
    if len(x) < 2 or np.var(x) < 1e-12:
        return float(np.mean(y)) if len(y) else 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.scored_parquet)
    df = df.dropna(subset=["score_user"]).reset_index(drop=True)
    print(f"[load] {len(df)} utterances with user scores, {df.user_id.nunique()} users")

    # Global calibration (baseline): one (α, β) for everyone
    global_int, global_slope = ols_intercept_slope(
        df["rm_score"].to_numpy(), df["score_user"].to_numpy().astype(float)
    )
    print(f"[global] intercept={global_int:.3f}  slope={global_slope:.3f}")

    # Per-user k-fold CV
    per_user_rmse = []
    per_user_n = []
    kept_users = 0
    skipped_low_obs = 0

    for uid, grp in df.groupby("user_id"):
        n = len(grp)
        if n < args.min_obs_per_user:
            skipped_low_obs += 1
            continue
        kept_users += 1
        x = grp["rm_score"].to_numpy()
        y = grp["score_user"].to_numpy().astype(float)
        folds = kfold_split(n, args.k_folds, rng)

        fold_no_calib_sq = []     # predict global mean of y_train each fold
        fold_pop_slope_sq = []    # predict global_int + global_slope · x_test
        fold_pebs_sq = []        # predict user-own OLS(α_j, β_j) fit on train

        for train_idx, test_idx in folds:
            x_tr, y_tr = x[train_idx], y[train_idx]
            x_te, y_te = x[test_idx], y[test_idx]
            if len(x_te) == 0:
                continue
            # No-calib: predict train mean for all test points
            y_hat_nc = np.full_like(y_te, fill_value=float(np.mean(y_tr)))
            fold_no_calib_sq.extend(((y_hat_nc - y_te) ** 2).tolist())

            # Pop-slope: global (intercept, slope) applied to test x
            y_hat_ps = global_int + global_slope * x_te
            fold_pop_slope_sq.extend(((y_hat_ps - y_te) ** 2).tolist())

            # PEBS: per-user (α, β) fit on train fold only
            alpha_j, beta_j = ols_intercept_slope(x_tr, y_tr)
            y_hat_pebs = alpha_j + beta_j * x_te
            # NOTE: ols_intercept_slope returns (intercept, slope), so this is β̂_j + α̂_j·x
            # We already used the same variable names consistently.
            fold_pebs_sq.extend(((y_hat_pebs - y_te) ** 2).tolist())

        per_user_rmse.append({
            "user_id": uid,
            "n": n,
            "rmse_no_calib": float(np.sqrt(np.mean(fold_no_calib_sq))),
            "rmse_pop_slope": float(np.sqrt(np.mean(fold_pop_slope_sq))),
            "rmse_pebs": float(np.sqrt(np.mean(fold_pebs_sq))),
        })
        per_user_n.append(n)

    per_user_df = pd.DataFrame(per_user_rmse)
    print(f"\n=== k-fold CV results (k={args.k_folds}, min_obs={args.min_obs_per_user}) ===")
    print(f"  kept users: {kept_users}, skipped (low obs): {skipped_low_obs}")
    print(f"  mean n/user among kept: {np.mean(per_user_n):.1f}")

    # Summary stats per arm
    for arm in ["rmse_no_calib", "rmse_pop_slope", "rmse_pebs"]:
        vals = per_user_df[arm]
        print(f"  {arm}: mean={vals.mean():.3f}  median={vals.median():.3f}  "
              f"p25={vals.quantile(0.25):.3f}  p75={vals.quantile(0.75):.3f}")

    # Paired tests on per-user RMSE
    def paired(a_col: str, b_col: str):
        a = per_user_df[a_col].to_numpy()
        b = per_user_df[b_col].to_numpy()
        diff = a - b
        stat = stats.wilcoxon(a, b, alternative="two-sided")
        return {
            "mean_diff_a_minus_b": float(diff.mean()),
            "median_diff": float(np.median(diff)),
            "wilcoxon_stat": float(stat.statistic),
            "wilcoxon_p": float(stat.pvalue),
            "frac_a_larger": float((a > b).mean()),
            "frac_a_smaller": float((a < b).mean()),
        }

    out = {
        "n_users_kept": int(kept_users),
        "n_users_skipped_low_obs": int(skipped_low_obs),
        "mean_obs_per_user": float(np.mean(per_user_n)),
        "k_folds": int(args.k_folds),
        "global_calibration": {"intercept": global_int, "slope": global_slope},
        "aggregate_rmse": {
            "no_calib": {"mean": float(per_user_df.rmse_no_calib.mean()),
                        "median": float(per_user_df.rmse_no_calib.median())},
            "pop_slope": {"mean": float(per_user_df.rmse_pop_slope.mean()),
                         "median": float(per_user_df.rmse_pop_slope.median())},
            "pebs": {"mean": float(per_user_df.rmse_pebs.mean()),
                      "median": float(per_user_df.rmse_pebs.median())},
        },
        "wilcoxon_tests": {
            "pop_slope_vs_no_calib": paired("rmse_pop_slope", "rmse_no_calib"),
            "pebs_vs_no_calib": paired("rmse_pebs", "rmse_no_calib"),
            "pebs_vs_pop_slope": paired("rmse_pebs", "rmse_pop_slope"),
        },
    }

    # Relative improvement (PEBS vs pop_slope baseline)
    rel = 100.0 * (per_user_df.rmse_pop_slope.mean() - per_user_df.rmse_pebs.mean()) \
          / max(per_user_df.rmse_pop_slope.mean(), 1e-9)
    out["pebs_vs_pop_slope_relative_improvement_pct"] = float(rel)

    # Per-user-bootstrap CI for the improvement: resample users, recompute mean
    n_boot = 1000
    users = per_user_df.index.to_numpy()
    rng_ci = np.random.default_rng(args.seed + 1)
    boot_rel = []
    for _ in range(n_boot):
        sub = rng_ci.choice(users, size=len(users), replace=True)
        a = per_user_df.iloc[sub].rmse_pop_slope.mean()
        b = per_user_df.iloc[sub].rmse_pebs.mean()
        boot_rel.append(100.0 * (a - b) / max(a, 1e-9))
    out["pebs_vs_pop_slope_rel_improvement_95CI"] = [
        float(np.percentile(boot_rel, 2.5)),
        float(np.percentile(boot_rel, 97.5)),
    ]

    # Save
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    per_user_df.to_parquet(out_path.with_suffix(".parquet"))
    print(f"\n[save] {out_path} + .parquet")

    # Verdict
    print(f"\n=== Verdict ===")
    print(f"  PEBS vs pop-slope relative improvement: {rel:.2f}%")
    print(f"  95% CI: [{out['pebs_vs_pop_slope_rel_improvement_95CI'][0]:.2f}%, "
          f"{out['pebs_vs_pop_slope_rel_improvement_95CI'][1]:.2f}%]")
    p = out["wilcoxon_tests"]["pebs_vs_pop_slope"]["wilcoxon_p"]
    print(f"  Wilcoxon paired p (pebs < pop_slope): {p:.2e}")
    if rel > 0 and p < 0.05:
        print("  ✓ PEBS significantly improves user-score prediction vs global calibration")
    else:
        print("  ✗ PEBS does NOT improve RMSE over global calibration on this test")


if __name__ == "__main__":
    main()
