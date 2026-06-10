"""PEBS gain as a function of cohort size N.

Reviewer question: "Your 8.58% gain is on N=1394 users. Would a smaller
cohort (e.g., N=100) still benefit from EB shrinkage? At what N does the
benefit stabilize?"

Method: subsample PRISM users to N ∈ {100, 200, 400, 800, 1400}, refit
MoM τ² on that sub-cohort, compute PEBS-shrunk gain vs pop-OLS.
30 random subsample seeds per N for stable averages.

Output: results/track1_gain_vs_cohort_size/summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

T1 = (Path(__file__).resolve().parents[2] / "1_Causal_RLHF")
OUT = (Path(__file__).resolve().parents[2] / "3_PEBS_Standalone/results/track1_gain_vs_cohort_size")
N_SEEDS = 30
RNG = 20260420


def fit_ols(x, y):
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan
    x_mean, y_mean = x.mean(), y.mean()
    var_x = np.sum((x - x_mean) ** 2)
    if var_x < 1e-10:
        return np.nan, np.nan, np.nan, np.nan
    beta = float(np.sum((x - x_mean) * (y - y_mean)) / var_x)
    alpha = float(y_mean - beta * x_mean)
    n = len(x)
    if n <= 2:
        return alpha, beta, np.nan, np.nan
    resid = y - (alpha + beta * x)
    mse = np.sum(resid ** 2) / (n - 2)
    se_beta = float(np.sqrt(mse / var_x))
    se_alpha = float(np.sqrt(mse * (1.0 / n + x_mean ** 2 / var_x)))
    return alpha, beta, se_alpha, se_beta


def run_subcohort(df_scored, users, rng):
    """Compute PEBS-shrunk gain on the cohort defined by `users`."""
    sub = df_scored[df_scored["user_id"].isin(users)].copy()
    if sub["user_id"].nunique() < 10:
        return None

    # Per-user OLS (on all that user's data for simplicity — this measures
    # "how well does PEBS calibrate within this cohort" given each user's
    # data is available)
    per_user = {}
    for uid, g in sub.groupby("user_id"):
        a, b, sa, sb = fit_ols(g["rm_score"].to_numpy(), g["score_user"].to_numpy())
        if np.all(np.isfinite([a, b, sa, sb])):
            per_user[uid] = {"alpha": a, "beta": b, "se_a": sa, "se_b": sb,
                             "n": len(g)}

    if len(per_user) < 10:
        return None

    alphas = np.array([v["alpha"] for v in per_user.values()])
    betas = np.array([v["beta"] for v in per_user.values()])
    se_as = np.array([v["se_a"] for v in per_user.values()])
    se_bs = np.array([v["se_b"] for v in per_user.values()])

    alpha_pop = float(alphas.mean())
    beta_pop = float(betas.mean())
    tau2_a = max(0.0, float(np.var(alphas, ddof=1) - np.mean(se_as ** 2)))
    tau2_b = max(0.0, float(np.var(betas, ddof=1) - np.mean(se_bs ** 2)))

    # For each user compute PEBS-shrunk predictions vs pop-OLS predictions on
    # within-user held-out 20% (time-shuffled split)
    gains = []
    for uid, v in per_user.items():
        g = sub[sub["user_id"] == uid].copy()
        if len(g) < 10:
            continue
        # 80/20 random split (not the LOCO level — we want just the "shrinkage
        # benefit at cohort size N" signal, not the prompt-leakage question)
        n_test = max(2, len(g) // 5)
        idx = rng.permutation(len(g))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        g_train = g.iloc[train_idx]
        g_test = g.iloc[test_idx]
        if len(g_train) < 3:
            continue
        a_u, b_u, sa_u, sb_u = fit_ols(g_train["rm_score"].to_numpy(),
                                        g_train["score_user"].to_numpy())
        if not np.all(np.isfinite([a_u, b_u, sa_u, sb_u])):
            continue
        w_a = tau2_a / (tau2_a + sa_u ** 2 + 1e-12)
        w_b = tau2_b / (tau2_b + sb_u ** 2 + 1e-12)
        a_s = w_a * a_u + (1 - w_a) * alpha_pop
        b_s = w_b * b_u + (1 - w_b) * beta_pop

        x_test = g_test["rm_score"].to_numpy(dtype=np.float64)
        y_test = g_test["score_user"].to_numpy(dtype=np.float64)

        pred_pop = alpha_pop + beta_pop * x_test
        pred_pebs = a_s + b_s * x_test

        rmse_pop = float(np.sqrt(np.mean((y_test - pred_pop) ** 2)))
        rmse_pebs = float(np.sqrt(np.mean((y_test - pred_pebs) ** 2)))
        if rmse_pop > 1e-6:
            gain = 100.0 * (rmse_pop - rmse_pebs) / rmse_pop
            gains.append(gain)
    if not gains:
        return None
    return {"n_users": len(per_user), "mean_gain_pct": float(np.mean(gains)),
            "median_gain_pct": float(np.median(gains)),
            "tau2_a": tau2_a, "tau2_b": tau2_b}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG)

    df = pd.read_parquet(T1 / "data/prism_rm_scored.parquet").dropna(
        subset=["score_user", "rm_score"]
    )
    all_users = df["user_id"].unique()
    print(f"[load] {len(df)} scored utt × {len(all_users)} users")

    results = []
    for N in [100, 200, 400, 800, 1400]:
        cell = []
        for seed in tqdm(range(N_SEEDS), desc=f"N={N}", leave=False):
            sub_rng = np.random.default_rng(RNG + seed)
            sel = sub_rng.choice(all_users, size=min(N, len(all_users)), replace=False)
            r = run_subcohort(df, set(sel.tolist()), sub_rng)
            if r is not None:
                cell.append(r)
        gains = [c["mean_gain_pct"] for c in cell]
        tau2_as = [c["tau2_a"] for c in cell]
        print(f"N={N}  n_seeds={len(cell)}  mean_gain={np.mean(gains):.2f}%  "
              f"SD={np.std(gains, ddof=1):.2f}  min={min(gains):.2f}  max={max(gains):.2f}"
              f"  tau2_a_mean={np.mean(tau2_as):.1f}")
        results.append({
            "N": N, "n_seeds_completed": len(cell),
            "mean_gain_pct": float(np.mean(gains)),
            "sd_gain_pct": float(np.std(gains, ddof=1)),
            "ci95_lo": float(np.percentile(gains, 2.5)),
            "ci95_hi": float(np.percentile(gains, 97.5)),
            "per_seed_gains": gains,
            "mean_tau2_alpha": float(np.mean(tau2_as)),
        })

    summary = {
        "n_seeds_per_N": N_SEEDS,
        "cells": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT}/summary.json")


if __name__ == "__main__":
    main()
