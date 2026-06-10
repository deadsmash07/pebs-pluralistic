"""T1 cold-start break-even under LOCO-strict protocol.

Paper claims PEBS drops the cold-start break-even from k=20 to k=5 labelled
utterances per user (4× data efficiency). That claim was made under random-
fold CV. Re-verify under the LOCO protocol: for each user
with ≥k utterances, fit per-user OLS on first-k-utterances-excluding-held-out-
conversation and compare PEBS-shrunk RMSE on held-out conversation vs pop-
slope baseline, varying k ∈ {2, 5, 10, 15, 20, 30}.

Break-even is the smallest k at which PEBS-shrunk outperforms pop-slope.

Output: results/track1_cold_start_loco/summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

T1 = (Path(__file__).resolve().parents[2] / "1_Causal_RLHF")
OUT = (Path(__file__).resolve().parents[2] / "3_PEBS_Standalone/results/track1_cold_start_loco")
N_BOOT = 500
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
    mse = float(np.sum(resid ** 2)) / (n - 2)
    se_beta = float(np.sqrt(mse / var_x))
    se_alpha = float(np.sqrt(mse * (1.0 / n + x_mean ** 2 / var_x)))
    return alpha, beta, se_alpha, se_beta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG)

    df = pd.read_parquet(T1 / "data/prism_rm_scored.parquet").dropna(
        subset=["score_user", "rm_score", "conversation_id"]
    )
    conv_count = df.groupby("user_id")["conversation_id"].nunique()
    # Need users with ≥2 conversations for LOCO
    keep = conv_count[conv_count >= 2].index
    df = df[df["user_id"].isin(keep)].copy()

    # Global MoM τ² from full-data per-user OLS
    full_by_user = {}
    for uid, g in df.groupby("user_id"):
        a, b, sa, sb = fit_ols(g["rm_score"].to_numpy(), g["score_user"].to_numpy())
        if np.all(np.isfinite([a, b, sa, sb])):
            full_by_user[uid] = (a, b, sa, sb)
    alphas = np.array([v[0] for v in full_by_user.values()])
    betas = np.array([v[1] for v in full_by_user.values()])
    se_as = np.array([v[2] for v in full_by_user.values()])
    se_bs = np.array([v[3] for v in full_by_user.values()])
    alpha_pop = float(alphas.mean())
    beta_pop = float(betas.mean())
    tau2_a = max(0.0, float(np.var(alphas, ddof=1) - np.mean(se_as ** 2)))
    tau2_b = max(0.0, float(np.var(betas, ddof=1) - np.mean(se_bs ** 2)))
    print(f"[global MoM] α_pop={alpha_pop:.2f}  β_pop={beta_pop:.4f}  τ²_α={tau2_a:.2f}  τ²_β={tau2_b:.4f}")

    K_GRID = [2, 5, 10, 15, 20, 30]

    # For each k, per-user LOCO-strict cold-start evaluation
    per_k = {}
    for k in K_GRID:
        per_user_gains = []
        for uid, u_df in df.groupby("user_id"):
            conv_ids = u_df["conversation_id"].unique().tolist()
            if len(conv_ids) < 2:
                continue
            # Held-out = last conversation; train = first-k-utterances from the rest
            other_conv = u_df[u_df["conversation_id"] != conv_ids[-1]]
            test = u_df[u_df["conversation_id"] == conv_ids[-1]]
            if len(other_conv) < k or len(test) < 1:
                continue
            # Take first k utterances sorted by turn
            other_sorted = other_conv.sort_values(["conversation_id", "turn", "within_turn_id"]).head(k)
            a_u, b_u, sa_u, sb_u = fit_ols(
                other_sorted["rm_score"].to_numpy(dtype=np.float64),
                other_sorted["score_user"].to_numpy(dtype=np.float64),
            )
            x_test = test["rm_score"].to_numpy(dtype=np.float64)
            y_test = test["score_user"].to_numpy(dtype=np.float64)
            pred_pop = alpha_pop + beta_pop * x_test
            if np.all(np.isfinite([a_u, b_u, sa_u, sb_u])):
                # PEBS shrunk blend
                w_a = tau2_a / (tau2_a + sa_u ** 2 + 1e-12)
                w_b = tau2_b / (tau2_b + sb_u ** 2 + 1e-12)
                a_s = w_a * a_u + (1 - w_a) * alpha_pop
                b_s = w_b * b_u + (1 - w_b) * beta_pop
                pred_pebs = a_s + b_s * x_test
            else:
                pred_pebs = pred_pop
            rmse_pop = float(np.sqrt(np.mean((y_test - pred_pop) ** 2)))
            rmse_pebs = float(np.sqrt(np.mean((y_test - pred_pebs) ** 2)))
            if rmse_pop > 1e-6:
                per_user_gains.append(100.0 * (rmse_pop - rmse_pebs) / rmse_pop)

        if not per_user_gains:
            continue
        gains = np.array(per_user_gains)
        mean = float(gains.mean())
        boots = np.array([np.random.default_rng(RNG + k + i).choice(gains, len(gains), replace=True).mean()
                          for i in range(N_BOOT)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        per_k[k] = {
            "k": k, "n_users": int(len(gains)), "mean_gain_pct": mean,
            "ci95_lo": float(lo), "ci95_hi": float(hi),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
        }
        print(f"  k={k:>2d}  n={len(gains):>4d}  gain={mean:+.3f}%  CI [{lo:+.3f}, {hi:+.3f}]"
              f"  {'*' if (lo > 0 or hi < 0) else ''}")

    # Break-even = smallest k with CI > 0 (gain significantly positive)
    break_even = None
    for k in K_GRID:
        if k in per_k and per_k[k]["ci95_lo"] > 0:
            break_even = k
            break

    print(f"\n[cold-start LOCO break-even] k* = {break_even}")

    summary = {
        "protocol": "LOCO-strict cold-start: train on first-k utterances from all but last conversation, eval on last conversation",
        "tau2_alpha": tau2_a, "tau2_beta": tau2_b,
        "per_k": per_k,
        "break_even_k": break_even,
        "paper_claim_random_fold": "k=5 break-even (PEBS) vs k=20 (per-user OLS)",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT}/summary.json")


if __name__ == "__main__":
    main()
