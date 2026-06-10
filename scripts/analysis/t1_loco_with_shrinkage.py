"""LOCO CV extension: measure PEBS-SHRUNK (not just per-user OLS) gain.

The earlier LOCO analysis reported per-user OLS vs pop-OLS
(+4.22%). The paper's actual claim is about PEBS-shrunk — a convex blend
of per-user OLS with population via EB weight ω = τ²/(τ² + σ²/n_j).

This script: for each held-out conversation within a user,
  1. Fit per-user OLS on training folds to get (α_j_LOCO, β_j_LOCO)
  2. Compute population mean (α_pop, β_pop) across OTHER users
  3. Estimate τ² from the LOCO-population per-user OLS distribution
  4. Compute ω_j and PEBS-SHRUNK per-user calibrator
  5. Evaluate on held-out conversation RMSE

If PEBS-shrunk beats per-user OLS in LOCO, shrinkage is the mechanism
(not memorization). If they tie, per-user is enough for this slice.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

T1 = (Path(__file__).resolve().parents[2] / "1_Causal_RLHF")
OUT = (Path(__file__).resolve().parents[2] / "3_PEBS_Standalone/results/track1_loco_cv_shrunk")
N_BOOT = 2000
RNG = 20260420
MIN_CONV_PER_USER = 2
MIN_UTT_PER_USER = 10


def fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan
    x_mean, y_mean = x.mean(), y.mean()
    var_x = np.sum((x - x_mean) ** 2)
    if var_x < 1e-10:
        return np.nan, np.nan, np.nan, np.nan
    beta = float(np.sum((x - x_mean) * (y - y_mean)) / var_x)
    alpha = float(y_mean - beta * x_mean)
    y_hat = alpha + beta * x
    resid = y - y_hat
    sse = float(np.sum(resid ** 2))
    n = len(x)
    if n <= 2:
        return alpha, beta, np.nan, np.nan
    mse = sse / (n - 2)
    se_beta = float(np.sqrt(mse / var_x))
    se_alpha = float(np.sqrt(mse * (1.0 / n + x_mean ** 2 / var_x)))
    return alpha, beta, se_alpha, se_beta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG)

    df = pd.read_parquet(T1 / "data/prism_rm_scored.parquet").dropna(
        subset=["score_user", "rm_score", "conversation_id"]
    )
    # Filter: users with ≥ 2 conversations AND ≥ 10 utterances
    conv_count = df.groupby("user_id")["conversation_id"].nunique()
    utt_count = df.groupby("user_id").size()
    keep = conv_count[(conv_count >= MIN_CONV_PER_USER) & (utt_count >= MIN_UTT_PER_USER)].index
    df = df[df["user_id"].isin(keep)].copy()
    print(f"[filter] {len(df)} utt × {df['user_id'].nunique()} users")

    # Pre-compute each user's FULL per-user OLS (on all their data) for τ² estimation
    full_ols = {}
    for uid, g in df.groupby("user_id"):
        a, b, se_a, se_b = fit_ols(g["rm_score"].to_numpy(), g["score_user"].to_numpy())
        full_ols[uid] = {"alpha_full": a, "beta_full": b, "se_alpha_full": se_a, "se_beta_full": se_b}

    alphas = np.array([v["alpha_full"] for v in full_ols.values() if np.isfinite(v["alpha_full"])])
    betas = np.array([v["beta_full"] for v in full_ols.values() if np.isfinite(v["beta_full"])])
    se_as = np.array([v["se_alpha_full"] for v in full_ols.values() if np.isfinite(v["se_alpha_full"])])
    se_bs = np.array([v["se_beta_full"] for v in full_ols.values() if np.isfinite(v["se_beta_full"])])
    alpha_pop, beta_pop = float(alphas.mean()), float(betas.mean())
    tau2_a = max(0.0, float(np.var(alphas, ddof=1) - np.mean(se_as ** 2)))
    tau2_b = max(0.0, float(np.var(betas, ddof=1) - np.mean(se_bs ** 2)))
    print(f"[global MoM] α_pop={alpha_pop:.2f}  β_pop={beta_pop:.4f}  τ²_α={tau2_a:.2f}  τ²_β={tau2_b:.4f}")

    per_user_results = []
    for uid, u_df in tqdm(df.groupby("user_id"), desc="LOCO per user"):
        conv_ids = u_df["conversation_id"].unique().tolist()
        if len(conv_ids) < MIN_CONV_PER_USER:
            continue

        sq_naive = []
        sq_user_ols = []
        sq_pebs_shrunk = []

        for held_conv in conv_ids:
            train = u_df[u_df["conversation_id"] != held_conv]
            test = u_df[u_df["conversation_id"] == held_conv]
            if len(train) < 3 or len(test) < 1:
                continue
            a_u, b_u, se_a_u, se_b_u = fit_ols(train["rm_score"].to_numpy(), train["score_user"].to_numpy())

            x_test = test["rm_score"].to_numpy(dtype=np.float64)
            y_test = test["score_user"].to_numpy(dtype=np.float64)

            pred_naive = alpha_pop + beta_pop * x_test
            pred_user_ols = a_u + b_u * x_test if np.isfinite(a_u) else pred_naive

            # PEBS-shrunk: ω blend of (a_u, b_u) with (α_pop, β_pop)
            if np.isfinite(a_u) and np.isfinite(se_a_u) and np.isfinite(b_u) and np.isfinite(se_b_u):
                w_a = tau2_a / (tau2_a + se_a_u ** 2 + 1e-12)
                w_b = tau2_b / (tau2_b + se_b_u ** 2 + 1e-12)
                a_s = w_a * a_u + (1 - w_a) * alpha_pop
                b_s = w_b * b_u + (1 - w_b) * beta_pop
                pred_pebs = a_s + b_s * x_test
            else:
                pred_pebs = pred_naive

            sq_naive.append(float(np.sum((y_test - pred_naive) ** 2)))
            sq_user_ols.append(float(np.sum((y_test - pred_user_ols) ** 2)))
            sq_pebs_shrunk.append(float(np.sum((y_test - pred_pebs) ** 2)))

        n_utt = len(u_df)
        if n_utt < MIN_UTT_PER_USER:
            continue
        per_user_results.append({
            "user_id": uid,
            "n_utt": int(n_utt),
            "n_conv": int(len(conv_ids)),
            "rmse_naive_pop": float(np.sqrt(sum(sq_naive) / n_utt)),
            "rmse_user_ols_loco": float(np.sqrt(sum(sq_user_ols) / n_utt)),
            "rmse_pebs_shrunk_loco": float(np.sqrt(sum(sq_pebs_shrunk) / n_utt)),
        })

    per_user = pd.DataFrame(per_user_results)
    per_user["gain_ols_pct"] = 100.0 * (per_user["rmse_naive_pop"] - per_user["rmse_user_ols_loco"]) / per_user["rmse_naive_pop"]
    per_user["gain_pebs_pct"] = 100.0 * (per_user["rmse_naive_pop"] - per_user["rmse_pebs_shrunk_loco"]) / per_user["rmse_naive_pop"]
    per_user["delta_pebs_vs_ols_pct"] = per_user["gain_pebs_pct"] - per_user["gain_ols_pct"]

    # Cluster-bootstrap
    def boot_mean_ci(v: np.ndarray) -> tuple[float, float, float]:
        n = len(v)
        point = float(v.mean())
        boots = np.array([np.random.default_rng(RNG + i).choice(v, n, replace=True).mean()
                          for i in range(N_BOOT)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return point, float(lo), float(hi)

    p_ols, lo_ols, hi_ols = boot_mean_ci(per_user["gain_ols_pct"].to_numpy())
    p_pebs, lo_pebs, hi_pebs = boot_mean_ci(per_user["gain_pebs_pct"].to_numpy())
    p_delta, lo_delta, hi_delta = boot_mean_ci(per_user["delta_pebs_vs_ols_pct"].to_numpy())

    print(f"\n[LOCO CV — per-user OLS vs pop]       gain = {p_ols:+.2f}%  CI [{lo_ols:+.2f}, {hi_ols:+.2f}]")
    print(f"[LOCO CV — PEBS-shrunk vs pop]       gain = {p_pebs:+.2f}%  CI [{lo_pebs:+.2f}, {hi_pebs:+.2f}]")
    print(f"[paired Δ — PEBS-shrunk vs user OLS] gain = {p_delta:+.2f}%  CI [{lo_delta:+.2f}, {hi_delta:+.2f}]"
          f"  {'(significant)' if lo_delta > 0 or hi_delta < 0 else '(null)'}")

    summary = {
        "n_users": int(len(per_user)),
        "tau2_alpha": tau2_a,
        "tau2_beta": tau2_b,
        "alpha_pop": alpha_pop,
        "beta_pop": beta_pop,
        "loco_ols_gain_pct":          {"mean": p_ols,   "ci95": [lo_ols,   hi_ols]},
        "loco_pebs_shrunk_gain_pct": {"mean": p_pebs, "ci95": [lo_pebs, hi_pebs]},
        "delta_shrunk_vs_ols_pct":    {"mean": p_delta, "ci95": [lo_delta, hi_delta]},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    per_user.to_parquet(OUT / "per_user.parquet", index=False)
    print(f"\nWrote {OUT}/summary.json")


if __name__ == "__main__":
    main()
