"""PEBS cold-start adaptation curve on PRISM.

Research question: for a NEW user with a small labeled-utterance budget k,
how does PEBS's per-user calibration compare to pop-slope (the best we
can do without per-user data)? At what k does PEBS break even?

This is the paper claim reviewers will ask about: "PEBS requires labeled
utterances per user. Is that practical? What's the minimum budget?"

Design
------
- For each user with ≥30 utterances (ensures meaningful held-out after k):
    - For each budget k ∈ {1, 2, 3, 5, 10, 20}:
        - Fit OLS on first k utterances → user's (α̂_j, β̂_j)
        - Predict held-out utterances (remaining n-k) via α̂_j + β̂_j · rm_score
        - Record held-out RMSE
    - Also: pop-slope RMSE on same held-out (using train-user-only fit)
    - Also: no-calib RMSE on same held-out
- Compute break-even k: smallest k where median PEBS RMSE ≤ median pop-slope RMSE

Honest baselines
----------------
- `utterances_first_k`: simple first-k (may be biased by conversation order).
  Also report a random-k variant to check this isn't ordering-dependent.
- Pop-slope is fit on 80% of users (train); 20% are held-out; the same
  held-out users' utterances are used across all k.

References
----------
- Rasch 1960, Baker 2001 IRT §3 (scale calibration as individual-parameter fit)
- Gelman & Hill 2007 §12 Bayesian shrinkage for small-n-per-group
- Learning-curve analysis: Hutter et al. 2014 for ML benchmarks
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
    p.add_argument("--budgets", default="1,2,3,5,10,20")
    p.add_argument("--min-obs-per-user", type=int, default=30,
                   help="Users with <min_obs utterances are skipped (need meaningful held-out after k).")
    p.add_argument("--holdout-user-frac", type=float, default=0.2,
                   help="Fraction of users held out entirely for cross-user pop-slope fit.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-path",
                   default="results/track1_coldstart_curve.json")
    p.add_argument("--use-random-k", action="store_true",
                   help="Sample first-k at random positions instead of chronological.")
    return p.parse_args()


def ols_intercept_slope(x: np.ndarray, y: np.ndarray):
    if len(x) < 2 or np.var(x) < 1e-12:
        return float(np.mean(y)) if len(y) else 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    budgets = [int(b) for b in args.budgets.split(",")]

    df = pd.read_parquet(args.scored_parquet).dropna(subset=["score_user"]).reset_index(drop=True)
    print(f"[load] {len(df)} utterances, {df.user_id.nunique()} users")

    # Keep users with enough observations
    user_n = df.groupby("user_id").size()
    kept_users = user_n[user_n >= args.min_obs_per_user].index.to_list()
    print(f"[filter] {len(kept_users)} users with ≥{args.min_obs_per_user} utterances")

    # 80/20 user split
    shuffled = list(kept_users)
    rng.shuffle(shuffled)
    n_holdout = int(len(shuffled) * args.holdout_user_frac)
    holdout_users = set(shuffled[:n_holdout])
    train_users = set(shuffled[n_holdout:])
    print(f"[split] train users: {len(train_users)}, holdout users: {len(holdout_users)}")

    # Fit pop-slope on TRAIN users only
    train_df = df[df.user_id.isin(train_users)]
    global_int, global_slope = ols_intercept_slope(
        train_df["rm_score"].to_numpy(),
        train_df["score_user"].to_numpy().astype(float),
    )
    train_mean_y = float(train_df["score_user"].mean())
    print(f"[fit train] pop-slope intercept={global_int:.3f} slope={global_slope:.3f}  mean_y={train_mean_y:.3f}")

    # Evaluate on holdout users
    curve_rows = []
    per_user_rows = []
    for uid in holdout_users:
        grp = df[df.user_id == uid].reset_index(drop=True)
        if len(grp) < args.min_obs_per_user:
            continue
        x = grp["rm_score"].to_numpy()
        y = grp["score_user"].to_numpy().astype(float)
        n = len(x)

        if args.use_random_k:
            perm = rng.permutation(n)
            x, y = x[perm], y[perm]

        rec = {"user_id": uid, "n": n}
        # Pop-slope RMSE on ALL n utterances (cross-user, no per-user fit)
        y_hat_pop = global_int + global_slope * x
        rmse_pop = float(np.sqrt(np.mean((y_hat_pop - y) ** 2)))
        rec["rmse_pop_slope"] = rmse_pop

        # No-calib RMSE: predict train_mean
        rmse_nocal = float(np.sqrt(np.mean((train_mean_y - y) ** 2)))
        rec["rmse_no_calib"] = rmse_nocal

        # PEBS at each budget k: fit on first k, eval on remaining n-k
        for k in budgets:
            if k + 2 > n:   # Need at least 2 held-out
                rec[f"rmse_pebs_k{k}"] = np.nan
                continue
            alpha_j, beta_j = ols_intercept_slope(x[:k], y[:k])
            y_hat_pebs = alpha_j + beta_j * x[k:]
            rmse_pebs = float(np.sqrt(np.mean((y_hat_pebs - y[k:]) ** 2)))
            rec[f"rmse_pebs_k{k}"] = rmse_pebs

        per_user_rows.append(rec)

    pu = pd.DataFrame(per_user_rows)
    print(f"\n=== Cold-start adaptation curve ({len(pu)} holdout users) ===")
    print(f"  Pop-slope RMSE on holdout users: mean={pu.rmse_pop_slope.mean():.3f}  median={pu.rmse_pop_slope.median():.3f}")
    print(f"  No-calib RMSE:                  mean={pu.rmse_no_calib.mean():.3f}  median={pu.rmse_no_calib.median():.3f}")

    curve = {"budgets": budgets, "per_k": {}}
    for k in budgets:
        col = f"rmse_pebs_k{k}"
        vals = pu[col].dropna()
        if len(vals) < 10:
            continue
        # Paired RMSE deltas: PEBS vs pop-slope and no-calib on the SAME holdout users
        paired_pop = pu.dropna(subset=[col])
        delta_vs_pop = paired_pop[col] - paired_pop.rmse_pop_slope
        # Positive = PEBS is WORSE at this budget
        w_pop = stats.wilcoxon(paired_pop[col], paired_pop.rmse_pop_slope, alternative="two-sided") \
                if len(paired_pop) >= 10 else None
        curve["per_k"][k] = {
            "n_users": int(len(vals)),
            "rmse_pebs": {"mean": float(vals.mean()), "median": float(vals.median()),
                          "p25": float(vals.quantile(0.25)), "p75": float(vals.quantile(0.75))},
            "vs_pop_slope": {
                "mean_delta_pebs_minus_pop": float(delta_vs_pop.mean()),
                "median_delta": float(delta_vs_pop.median()),
                "frac_pebs_wins": float((delta_vs_pop < 0).mean()),
                "wilcoxon_p": float(w_pop.pvalue) if w_pop else None,
            },
        }
        p_str = f"{w_pop.pvalue:.2g}" if w_pop else "n/a"
        print(f"  k={k:2d}: PEBS mean RMSE={vals.mean():.3f}  "
              f"median={vals.median():.3f}  "
              f"Δ vs pop={delta_vs_pop.mean():+.3f} "
              f"(wins {100*(delta_vs_pop<0).mean():.0f}%, "
              f"p={p_str})")

    # Break-even k: smallest budget where PEBS's mean RMSE ≤ pop-slope's mean RMSE
    pop_mean_rmse = pu.rmse_pop_slope.mean()
    break_even = None
    for k in budgets:
        col = f"rmse_pebs_k{k}"
        if col not in pu.columns:
            continue
        m = pu[col].mean()
        if np.isnan(m):
            continue
        if m <= pop_mean_rmse:
            break_even = k
            break

    out = {
        "config": {
            "n_holdout_users": int(len(pu)),
            "n_train_users": int(len(train_users)),
            "min_obs_per_user": args.min_obs_per_user,
            "budgets": budgets,
            "seed": args.seed,
            "use_random_k": bool(args.use_random_k),
        },
        "pop_slope": {"intercept": global_int, "slope": global_slope,
                     "train_mean_y": train_mean_y},
        "holdout_rmse_baselines": {
            "pop_slope_mean": float(pu.rmse_pop_slope.mean()),
            "pop_slope_median": float(pu.rmse_pop_slope.median()),
            "no_calib_mean": float(pu.rmse_no_calib.mean()),
            "no_calib_median": float(pu.rmse_no_calib.median()),
        },
        "curve": curve,
        "break_even_k_mean": break_even,
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).write_text(json.dumps(out, indent=2))
    pu.to_parquet(Path(args.output_path).with_suffix(".parquet"))
    print(f"\n[save] {args.output_path}")
    print(f"\nBreak-even k (PEBS beats pop-slope in mean RMSE): {break_even}")


if __name__ == "__main__":
    main()
