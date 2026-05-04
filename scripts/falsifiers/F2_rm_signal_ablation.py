"""F2 — RM-signal-ablation baseline.

Pre-registered (iter+N+290) falsifier for the claim that PILSD's +8.58%
gain on PRISM is driven by RM-signal-informed slope inference (not
exclusively by intercept shrinkage per the 1973 Efron-Morris baseline).

Design
------
Take the real PRISM scored data (1394 users, ~49 utt/user, 7B Qwen RM).
Replace the rm_score column with THREE null variants:
    (i)   iid N(0, 1) per row (seeded)
    (ii)  iid N(0, 1) calibrated to match the empirical rm_score sd
    (iii) global random permutation of rm_score across ALL rows

Re-run PILSD (same LOCO-conversation hold-out as neighbor_head_to_head.py)
and measure RMSE gain vs pop-slope.

Pre-registered criterion (iter+N+290):
  CONFIRMING:  each variant returns gain <= 6.5 pp
               (consistent with the Efron-Morris intercept-only +6.04% ceiling
                documented in PAPER_INSERT_pythia_stress.tex at r ~ 0)
  FALSIFYING:  any variant returns gain > 6.5 pp
               (would mean +8.58% headline is mostly intercept-shrinkage)

Outputs
-------
results/falsifiers_iter290/F2_rm_signal_ablation.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
T1 = ROOT.parent / "1_Causal_RLHF"
SCORED = T1 / "data/prism_rm_scored.parquet"
OUT = ROOT / "results/falsifiers_iter290"

MIN_CONV_PER_USER = 2
MIN_UTT_PER_USER = 10
N_BOOT = 2000
RNG_BASE = 20260420


def fit_ols_with_se(x, y):
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan
    xm = float(np.mean(x))
    ssx = float(np.sum((x - xm) ** 2))
    if ssx < 1e-10:
        return float(np.mean(y)), 0.0, np.inf, np.inf
    beta = float(np.sum((x - xm) * (y - np.mean(y))) / ssx)
    alpha = float(np.mean(y) - beta * xm)
    resid = y - (alpha + beta * x)
    n = len(x)
    if n <= 2:
        return alpha, beta, np.inf, np.inf
    mse = float(np.sum(resid ** 2) / (n - 2))
    se_alpha = float(np.sqrt(mse * (1.0 / n + xm ** 2 / ssx)))
    se_beta = float(np.sqrt(mse / ssx))
    return alpha, beta, se_alpha, se_beta


def method_pilsd_shrunk(x_tr, y_tr, x_te, alpha_pop, beta_pop, tau2_a, tau2_b):
    a, b, se_a, se_b = fit_ols_with_se(x_tr, y_tr)
    if not np.isfinite(a):
        return alpha_pop + beta_pop * x_te
    w_a = tau2_a / (tau2_a + se_a ** 2 + 1e-12) if np.isfinite(se_a) else 0.0
    w_b = tau2_b / (tau2_b + se_b ** 2 + 1e-12) if np.isfinite(se_b) else 0.0
    a_s = w_a * a + (1 - w_a) * alpha_pop
    b_s = w_b * b + (1 - w_b) * beta_pop
    return a_s + b_s * x_te


def evaluate_pilsd(df: pd.DataFrame, label: str):
    """Full PILSD LOCO pipeline. df columns: user_id, conversation_id, rm_score, score_user."""
    slope_pop, intercept_pop = np.polyfit(df["rm_score"].to_numpy(),
                                          df["score_user"].to_numpy().astype(float), 1)
    alpha_pop, beta_pop = float(intercept_pop), float(slope_pop)

    user_stats = []
    for uid, g in df.groupby("user_id"):
        a, b, sa, sb = fit_ols_with_se(g["rm_score"].to_numpy(),
                                        g["score_user"].to_numpy().astype(float))
        user_stats.append((a, b, sa, sb))
    alphas = np.array([s[0] for s in user_stats if np.isfinite(s[0])])
    betas = np.array([s[1] for s in user_stats if np.isfinite(s[1])])
    sas = np.array([s[2] for s in user_stats if np.isfinite(s[2])])
    sbs = np.array([s[3] for s in user_stats if np.isfinite(s[3])])
    tau2_a = max(0.0, float(np.var(alphas, ddof=1) - np.mean(sas ** 2)))
    tau2_b = max(0.0, float(np.var(betas, ddof=1) - np.mean(sbs ** 2)))
    r_obs = float(np.corrcoef(df["rm_score"], df["score_user"])[0, 1])
    print(f"[{label}] alpha_pop={alpha_pop:.3f}  beta_pop={beta_pop:.5f}  "
          f"tau2_a={tau2_a:.2f}  tau2_b={tau2_b:.4f}  |r|={abs(r_obs):.3f}")

    per_user = []
    for uid, g in df.groupby("user_id"):
        conv_ids = g["conversation_id"].unique().tolist()
        if len(conv_ids) < MIN_CONV_PER_USER:
            continue
        sq_pop, sq_pilsd = [], []
        n_utt_used = 0
        for hc in conv_ids:
            tr = g[g["conversation_id"] != hc]
            te = g[g["conversation_id"] == hc]
            if len(tr) < 3 or len(te) < 1:
                continue
            x_tr = tr["rm_score"].to_numpy(dtype=np.float64)
            y_tr = tr["score_user"].to_numpy(dtype=np.float64)
            x_te = te["rm_score"].to_numpy(dtype=np.float64)
            y_te = te["score_user"].to_numpy(dtype=np.float64)
            pred_pop = alpha_pop + beta_pop * x_te
            pred_pilsd = method_pilsd_shrunk(x_tr, y_tr, x_te,
                                              alpha_pop, beta_pop, tau2_a, tau2_b)
            sq_pop.extend(((pred_pop - y_te) ** 2).tolist())
            sq_pilsd.extend(((pred_pilsd - y_te) ** 2).tolist())
            n_utt_used += len(y_te)
        if n_utt_used < MIN_UTT_PER_USER:
            continue
        per_user.append({
            "user_id": uid, "n_utt": n_utt_used,
            "rmse_pop": float(np.sqrt(np.mean(sq_pop))),
            "rmse_pilsd": float(np.sqrt(np.mean(sq_pilsd))),
        })
    pu = pd.DataFrame(per_user)
    pu["gain_pct"] = 100.0 * (pu["rmse_pop"] - pu["rmse_pilsd"]) / pu["rmse_pop"]

    rng = np.random.default_rng(RNG_BASE)
    g_vals = pu["gain_pct"].to_numpy()
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, len(g_vals), size=len(g_vals))
        boots[b] = float(g_vals[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n_users": int(len(pu)),
        "alpha_pop": alpha_pop,
        "beta_pop": beta_pop,
        "tau2_a": tau2_a,
        "tau2_b": tau2_b,
        "abs_r_pearson": abs(r_obs),
        "rmse_pop_mean": float(pu["rmse_pop"].mean()),
        "rmse_pilsd_mean": float(pu["rmse_pilsd"].mean()),
        "gain_pct_mean": float(g_vals.mean()),
        "gain_pct_ci95": [float(lo), float(hi)],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet(SCORED).dropna(subset=["score_user", "rm_score", "conversation_id"])
    conv_count = df.groupby("user_id")["conversation_id"].nunique()
    utt_count = df.groupby("user_id").size()
    keep = conv_count[(conv_count >= MIN_CONV_PER_USER) & (utt_count >= MIN_UTT_PER_USER)].index
    df = df[df["user_id"].isin(keep)].reset_index(drop=True)
    print(f"[filter] {len(df)} utt x {df['user_id'].nunique()} users")

    sd_rm = float(df["rm_score"].std())
    print(f"[feat] rm_score sd={sd_rm:.4f}")

    results = {}

    # Reference: real RM signal
    print("\n[REFERENCE] real Qwen-7B rm_score")
    results["real_rm"] = evaluate_pilsd(df, label="real_rm")

    # Variant (i): iid N(0, 1)
    print("\n[VARIANT i] iid N(0,1) null RM")
    rng = np.random.default_rng(RNG_BASE + 1)
    df_null1 = df.copy()
    df_null1["rm_score"] = rng.normal(0.0, 1.0, len(df_null1))
    results["null_n01"] = evaluate_pilsd(df_null1, label="null_n01")

    # Variant (ii): iid N(0, sd_rm)
    print("\n[VARIANT ii] iid N(0, sd_rm) null RM")
    rng = np.random.default_rng(RNG_BASE + 2)
    df_null2 = df.copy()
    df_null2["rm_score"] = rng.normal(0.0, sd_rm, len(df_null2))
    results["null_n0sdrm"] = evaluate_pilsd(df_null2, label="null_n0sdrm")

    # Variant (iii): random permutation
    print("\n[VARIANT iii] permuted rm_score")
    rng = np.random.default_rng(RNG_BASE + 3)
    df_null3 = df.copy()
    df_null3["rm_score"] = rng.permutation(df["rm_score"].to_numpy())
    results["null_permuted"] = evaluate_pilsd(df_null3, label="null_permuted")

    # Pre-registered threshold: each null variant's gain must be <= 6.5 pp
    threshold = 6.5
    null_gains = {k: results[k]["gain_pct_mean"] for k in ("null_n01", "null_n0sdrm", "null_permuted")}
    any_above = any(g > threshold for g in null_gains.values())
    branch = "falsifying" if any_above else "confirming"
    summary = {
        "config": {
            "threshold_pp": threshold,
            "criterion": "All 3 null variants must return gain <= 6.5pp",
            "n_bootstrap": N_BOOT,
            "rng_base": RNG_BASE,
        },
        "results": results,
        "null_gains_pp": null_gains,
        "branch_disposition": {
            "branch": branch,
            "threshold_pp": threshold,
            "any_null_above_threshold": any_above,
        },
        "runtime_seconds": float(time.time() - t0),
    }
    (OUT / "F2_rm_signal_ablation.json").write_text(json.dumps(summary, indent=2))

    print("\n=== F2 SUMMARY ===")
    print(f"Real RM: gain = {results['real_rm']['gain_pct_mean']:+.3f}pp  "
          f"CI [{results['real_rm']['gain_pct_ci95'][0]:+.2f}, {results['real_rm']['gain_pct_ci95'][1]:+.2f}]  "
          f"|r|={results['real_rm']['abs_r_pearson']:.3f}")
    for k in ("null_n01", "null_n0sdrm", "null_permuted"):
        print(f"{k:16s}: gain = {results[k]['gain_pct_mean']:+.3f}pp  "
              f"CI [{results[k]['gain_pct_ci95'][0]:+.2f}, {results[k]['gain_pct_ci95'][1]:+.2f}]  "
              f"|r|={results[k]['abs_r_pearson']:.3f}")
    print(f"\nBranch: {branch.upper()} (threshold={threshold}pp)")


if __name__ == "__main__":
    main()
