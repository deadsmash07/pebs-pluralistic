"""Toy validation: is the PRISM α vs β PILSD decomposition a direct
consequence of the (σ_α, σ_β) variance ratio, or does it require any
mechanism beyond that?

CLAIM under test (iter+N+93 / commit 86e29bb on PRISM):
    pop_slope  : baseline
    α_only     : +7.58% RMSE improvement
    β_only     : +0.96%
    both_full  : +8.58%
    7.58 + 0.96 ≈ 8.54 ≈ 8.58  -> "approximately additive"
    α_only / both_full ≈ 88%   -> "intercept carries ~88% of gain"

PRISM empirical calibrator variance:
    σ_α = sqrt(τ_α²_only) = sqrt(111.60) = 10.57
    σ_β = sqrt(τ_β²_only) = sqrt(28.00)  = 5.29  (paper 4.82 w/ full pooling)
    ρ(α,β) ≈ 0.09

We simulate 3 DGPs with the SAME generative form used by PILSD on
PRISM (y = α_j + β_j · x + ε) and run the same 4-arm ablation + EB
shrinkage. If ONLY the variance ratio matters, PRISM_RATIO DGP should
reproduce the 88%/12% split.

DGPs
----
1. ONLY_ALPHA : σ_α=10.57, σ_β=0   -> expect α_only ≈ full, β_only ≈ 0
2. ONLY_BETA  : σ_α=0,     σ_β=4.82 -> expect α_only ≈ 0, β_only ≈ full
3. PRISM_RATIO: σ_α=10.57, σ_β=4.82, ρ=0.09  -> expect ~88%/12%

Pipeline per (DGP, seed)
------------------------
- Sample N=1000 users with calibrators (α_j, β_j) from the target MVN.
- For each user, sample K=30 (x, y) pairs: x ~ N(0, σ_x²); y = α_j + β_j x + ε.
- Global fit: α_pop, β_pop via OLS on all pooled (x, y).
- Per-user train: first k=5 points (the point of PILSD cold-start) or all-but-holdout;
  here we use the same 5-fold CV pattern PRISM uses on users.
- 4 arms on held-out split within each user (k-fold):
    pop_slope  : y_hat = α_pop + β_pop x
    α_only     : y_hat = α_j^shrunk + β_pop x             (β forced to pop)
    β_only     : y_hat = α_pop + β_j^shrunk x             (α forced to pop)
    both_full  : y_hat = α_j^shrunk + β_j^shrunk x
- EB shrinkage uses (τ_α², τ_β²) estimated from train-fold per-user OLS
  (same formula as eval_shrinkage_coldstart.py).
- Report mean_rmse, rel_improvement_vs_pop, additivity, α-share = α_only/both.

Expected empirical check
------------------------
- ONLY_ALPHA  : α-share close to 100%, β_only ≈ 0
- ONLY_BETA   : α-share close to 0%,   β_only ≈ full
- PRISM_RATIO : α-share close to 88% if decomp is purely arithmetic from
                variance ratios; otherwise the paper's claim has extra
                structure.

Runtime: CPU only, <5 min for 5 seeds × 3 DGPs × 1000 users × 30 items.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Numerics
# ------------------------------------------------------------

def ols_intercept_slope_with_variance(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, float, float]:
    """OLS (α̂, β̂) + sampling variance (V(α̂), V(β̂)).

    Matches eval_shrinkage_coldstart.py so the toy and PRISM pipelines
    are mechanically identical at this step.
    """
    k = len(x)
    if k < 2 or np.var(x) < 1e-12:
        return (float(np.mean(y)) if k else 0.0, 0.0, np.inf, np.inf)
    x_bar = float(x.mean())
    Sxx = float(((x - x_bar) ** 2).sum())
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = intercept + slope * x
    residuals = y - y_pred
    if k >= 3:
        sigma_hat_sq = float((residuals ** 2).sum() / (k - 2))
    else:
        sigma_hat_sq = float(np.var(y)) if np.var(y) > 0 else 1.0
    V_alpha = sigma_hat_sq * (1.0 / k + x_bar ** 2 / max(Sxx, 1e-12))
    V_beta = sigma_hat_sq / max(Sxx, 1e-12)
    return float(intercept), float(slope), float(V_alpha), float(V_beta)


def estimate_tau(train_cal: pd.DataFrame) -> Tuple[float, float]:
    """Parametric-EB τ² estimator (Morris 1983 moment estimator)."""
    V_alpha_total = float(train_cal["alpha_hat"].var())
    V_beta_total = float(train_cal["beta_hat"].var())
    Va_series = train_cal["V_alpha_hat"].replace(
        [np.inf, -np.inf], np.nan).dropna()
    Vb_series = train_cal["V_beta_hat"].replace(
        [np.inf, -np.inf], np.nan).dropna()
    mean_samp_V_alpha = float(np.mean(Va_series)) if len(Va_series) else 0.0
    mean_samp_V_beta = float(np.mean(Vb_series)) if len(Vb_series) else 0.0
    tau_a_sq = max(0.0, V_alpha_total - mean_samp_V_alpha)
    tau_b_sq = max(0.0, V_beta_total - mean_samp_V_beta)
    return tau_a_sq, tau_b_sq


def shrink(alpha_hat, beta_hat, V_alpha, V_beta,
           alpha_pop, beta_pop, tau_a_sq, tau_b_sq):
    """EB shrinkage: ω = τ²/(τ²+V)."""
    omega_a = tau_a_sq / (tau_a_sq + V_alpha) if np.isfinite(V_alpha) else 0.0
    omega_b = tau_b_sq / (tau_b_sq + V_beta) if np.isfinite(V_beta) else 0.0
    a_s = omega_a * alpha_hat + (1 - omega_a) * alpha_pop
    b_s = omega_b * beta_hat + (1 - omega_b) * beta_pop
    return a_s, b_s


# ------------------------------------------------------------
# DGPs
# ------------------------------------------------------------

@dataclass
class DGPConfig:
    name: str
    sigma_alpha: float
    sigma_beta: float
    rho_ab: float  # correlation between α and β
    mu_alpha: float = 66.44   # matches PRISM pop α
    mu_beta: float = 14.34    # matches PRISM pop β
    sigma_x: float = 1.0      # RM score std
    sigma_eps: float = 15.0   # residual noise std (PRISM-like)
    N: int = 1000
    K: int = 30


def sample_calibrators(cfg: DGPConfig, rng: np.random.Generator):
    """Sample N per-user (α_j, β_j) from target MVN."""
    cov_ab = cfg.rho_ab * cfg.sigma_alpha * cfg.sigma_beta
    Sigma = np.array(
        [[cfg.sigma_alpha ** 2, cov_ab],
         [cov_ab, cfg.sigma_beta ** 2]], dtype=float)
    mean = np.array([cfg.mu_alpha, cfg.mu_beta])
    # For degenerate cases (σ=0) rely on the cholesky degrading gracefully
    if cfg.sigma_alpha == 0 and cfg.sigma_beta == 0:
        alphas = np.full(cfg.N, cfg.mu_alpha)
        betas = np.full(cfg.N, cfg.mu_beta)
    else:
        # Handle σ=0 edge case by sampling then zeroing variance
        draws = rng.multivariate_normal(mean, Sigma + 1e-12 * np.eye(2),
                                        size=cfg.N)
        alphas = draws[:, 0]
        betas = draws[:, 1]
        if cfg.sigma_alpha == 0:
            alphas = np.full(cfg.N, cfg.mu_alpha)
        if cfg.sigma_beta == 0:
            betas = np.full(cfg.N, cfg.mu_beta)
    return alphas, betas


def sample_data(cfg: DGPConfig, alphas, betas, rng: np.random.Generator):
    """Return a long-form DataFrame: user_id, rm_score, score_user."""
    users = np.arange(cfg.N).repeat(cfg.K)
    x = rng.normal(0.0, cfg.sigma_x, size=cfg.N * cfg.K)
    alpha_rep = np.repeat(alphas, cfg.K)
    beta_rep = np.repeat(betas, cfg.K)
    eps = rng.normal(0.0, cfg.sigma_eps, size=cfg.N * cfg.K)
    y = alpha_rep + beta_rep * x + eps
    return pd.DataFrame({"user_id": users, "rm_score": x, "score_user": y})


# ------------------------------------------------------------
# 4-arm ablation via K-fold CV within each user
# ------------------------------------------------------------

def run_ablation_one_fold(df: pd.DataFrame,
                           train_mask: np.ndarray,
                           rng: np.random.Generator):
    """Run one fold: fit calibrators on train_mask, RMSE on held-out mask.

    Returns dict of arm -> per-user RMSE array.
    """
    train = df[train_mask].copy()
    test = df[~train_mask].copy()

    # Pop-slope on train (pooled across users)
    x_tr_all = train.rm_score.to_numpy()
    y_tr_all = train.score_user.to_numpy()
    slope_pop, intercept_pop = np.polyfit(x_tr_all, y_tr_all, 1)
    alpha_pop = float(intercept_pop)
    beta_pop = float(slope_pop)

    # Per-user OLS on train fold
    rows = []
    for uid, grp in train.groupby("user_id"):
        if len(grp) < 3:
            continue
        a, b, Va, Vb = ols_intercept_slope_with_variance(
            grp.rm_score.to_numpy(),
            grp.score_user.to_numpy().astype(float),
        )
        rows.append(dict(user_id=uid, alpha_hat=a, beta_hat=b,
                         V_alpha_hat=Va, V_beta_hat=Vb, n=len(grp)))
    train_cal = pd.DataFrame(rows)
    if len(train_cal) == 0:
        return None
    tau_a_sq, tau_b_sq = estimate_tau(train_cal)

    # Build shrunk calibrator table
    shrunk_rows = []
    for _, r in train_cal.iterrows():
        a_s, b_s = shrink(r.alpha_hat, r.beta_hat, r.V_alpha_hat, r.V_beta_hat,
                          alpha_pop, beta_pop, tau_a_sq, tau_b_sq)
        shrunk_rows.append(dict(user_id=r.user_id, alpha_s=a_s, beta_s=b_s))
    shrunk = pd.DataFrame(shrunk_rows).set_index("user_id")

    # Compute held-out RMSE per arm per user
    per_user_rmse = {"pop_slope": [], "alpha_only": [],
                     "beta_only": [], "both_full": []}
    for uid, grp in test.groupby("user_id"):
        if uid not in shrunk.index:
            continue
        x = grp.rm_score.to_numpy()
        y = grp.score_user.to_numpy().astype(float)
        a_s = shrunk.loc[uid].alpha_s
        b_s = shrunk.loc[uid].beta_s
        y_pop = alpha_pop + beta_pop * x
        y_a = a_s + beta_pop * x
        y_b = alpha_pop + b_s * x
        y_ab = a_s + b_s * x
        per_user_rmse["pop_slope"].append(
            float(np.sqrt(np.mean((y_pop - y) ** 2))))
        per_user_rmse["alpha_only"].append(
            float(np.sqrt(np.mean((y_a - y) ** 2))))
        per_user_rmse["beta_only"].append(
            float(np.sqrt(np.mean((y_b - y) ** 2))))
        per_user_rmse["both_full"].append(
            float(np.sqrt(np.mean((y_ab - y) ** 2))))
    return per_user_rmse


def kfold_user_stratified(df: pd.DataFrame, k_folds: int,
                           rng: np.random.Generator):
    """Yield boolean train masks for each of k_folds folds.

    Each user is split into k_folds approximately-equal parts; fold f
    holds out user's f-th part as test and uses the rest as train.
    """
    # Assign fold id per row (within-user round-robin shuffled)
    fold_id = np.zeros(len(df), dtype=int)
    for uid, idx in df.groupby("user_id").indices.items():
        permuted = rng.permutation(idx)
        folds = np.arange(len(permuted)) % k_folds
        fold_id[permuted] = folds
    for f in range(k_folds):
        yield fold_id != f


def run_dgp(cfg: DGPConfig, seed: int, k_folds: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    alphas, betas = sample_calibrators(cfg, rng)
    df = sample_data(cfg, alphas, betas, rng)

    arm_rmses = {"pop_slope": [], "alpha_only": [],
                 "beta_only": [], "both_full": []}
    fold_rng = np.random.default_rng(seed + 1)
    for train_mask in kfold_user_stratified(df, k_folds, fold_rng):
        fold_res = run_ablation_one_fold(df, train_mask, rng)
        if fold_res is None:
            continue
        for arm, vals in fold_res.items():
            arm_rmses[arm].extend(vals)

    # Summarize
    result = {"dgp": cfg.name, "seed": seed,
              "sigma_alpha": cfg.sigma_alpha,
              "sigma_beta": cfg.sigma_beta,
              "rho_ab": cfg.rho_ab,
              "N": cfg.N, "K": cfg.K, "k_folds": k_folds}
    pop_mean = float(np.mean(arm_rmses["pop_slope"])) if arm_rmses["pop_slope"] else np.nan
    for arm in ["pop_slope", "alpha_only", "beta_only", "both_full"]:
        v = np.array(arm_rmses[arm])
        mean_rmse = float(np.mean(v)) if len(v) else np.nan
        rel = 100.0 * (pop_mean - mean_rmse) / pop_mean if pop_mean else np.nan
        result[f"{arm}_rmse"] = mean_rmse
        result[f"{arm}_rel_pct"] = rel
    # Derived
    gain_full = result["both_full_rel_pct"]
    gain_a = result["alpha_only_rel_pct"]
    gain_b = result["beta_only_rel_pct"]
    result["sum_a_plus_b_pct"] = gain_a + gain_b
    result["additive_ratio"] = (
        (gain_a + gain_b) / gain_full if gain_full else np.nan)
    result["alpha_share_pct"] = (
        100.0 * gain_a / gain_full if gain_full else np.nan)
    result["beta_share_pct"] = (
        100.0 * gain_b / gain_full if gain_full else np.nan)
    return result


# ------------------------------------------------------------
# Driver
# ------------------------------------------------------------

def main():
    t0 = time.time()
    # PRISM calibrator-variance targets (alpha_only config)
    # τ_α²=111.60 -> σ_α=10.57; τ_β²=28.00 -> σ_β=5.29.
    # Paper claims σ_β=4.82 via full-pooling; prompt says 4.82 — use 4.82
    # for PRISM_RATIO to match the prompt exactly.
    sigma_alpha_prism = 10.57
    sigma_beta_prism = 4.82
    rho_prism = 0.09

    dgps = [
        DGPConfig(name="ONLY_ALPHA",
                  sigma_alpha=sigma_alpha_prism, sigma_beta=0.0, rho_ab=0.0),
        DGPConfig(name="ONLY_BETA",
                  sigma_alpha=0.0, sigma_beta=sigma_beta_prism, rho_ab=0.0),
        DGPConfig(name="PRISM_RATIO",
                  sigma_alpha=sigma_alpha_prism, sigma_beta=sigma_beta_prism,
                  rho_ab=rho_prism),
    ]
    seeds = [17, 42, 101, 2026, 9001]

    all_rows = []
    for cfg in dgps:
        for seed in seeds:
            r = run_dgp(cfg, seed, k_folds=5)
            all_rows.append(r)
            print(f"[{cfg.name:12s} seed={seed}] "
                  f"pop={r['pop_slope_rmse']:6.2f}  "
                  f"α={r['alpha_only_rel_pct']:6.2f}%  "
                  f"β={r['beta_only_rel_pct']:6.2f}%  "
                  f"full={r['both_full_rel_pct']:6.2f}%  "
                  f"α-share={r['alpha_share_pct']:6.1f}%")

    # Aggregate across seeds per DGP
    df_all = pd.DataFrame(all_rows)
    summary = {}
    for dgp_name in df_all.dgp.unique():
        sub = df_all[df_all.dgp == dgp_name]
        summary[dgp_name] = {
            "n_seeds": int(len(sub)),
            "alpha_only_rel_pct": {
                "mean": float(sub.alpha_only_rel_pct.mean()),
                "std": float(sub.alpha_only_rel_pct.std()),
            },
            "beta_only_rel_pct": {
                "mean": float(sub.beta_only_rel_pct.mean()),
                "std": float(sub.beta_only_rel_pct.std()),
            },
            "both_full_rel_pct": {
                "mean": float(sub.both_full_rel_pct.mean()),
                "std": float(sub.both_full_rel_pct.std()),
            },
            "sum_a_plus_b_pct": {
                "mean": float(sub.sum_a_plus_b_pct.mean()),
            },
            "additive_ratio": {
                "mean": float(sub.additive_ratio.mean()),
            },
            "alpha_share_pct": {
                "mean": float(sub.alpha_share_pct.mean()),
                "std": float(sub.alpha_share_pct.std()),
            },
            "beta_share_pct": {
                "mean": float(sub.beta_share_pct.mean()),
            },
        }

    # Print aggregate table
    print("\n" + "=" * 85)
    print(f"{'DGP':14s} {'α-only %':>12s} {'β-only %':>12s} {'full %':>10s} "
          f"{'sum':>9s} {'α-share':>10s} {'β-share':>10s}")
    print("-" * 85)
    for dgp_name, s in summary.items():
        print(f"{dgp_name:14s} "
              f"{s['alpha_only_rel_pct']['mean']:11.2f}  "
              f"{s['beta_only_rel_pct']['mean']:11.2f}  "
              f"{s['both_full_rel_pct']['mean']:9.2f}  "
              f"{s['sum_a_plus_b_pct']['mean']:8.2f}  "
              f"{s['alpha_share_pct']['mean']:9.1f}%  "
              f"{s['beta_share_pct']['mean']:9.1f}%")
    print("=" * 85)

    print("\nPRISM paper numbers (for reference):")
    print(f"{'PRISM_EMPIRICAL':14s} "
          f"{7.58:11.2f}  {0.96:11.2f}  {8.58:9.2f}  "
          f"{8.54:8.2f}  {88.3:9.1f}%  {11.2:9.1f}%")

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "t1_toy_alpha_beta_validation.json"
    payload = {
        "per_seed": all_rows,
        "summary_by_dgp": summary,
        "prism_reference": {
            "alpha_only_pct": 7.58, "beta_only_pct": 0.96,
            "both_full_pct": 8.58, "sum_a_plus_b_pct": 8.54,
            "alpha_share_pct": 88.3, "beta_share_pct": 11.2,
        },
        "sigma_alpha_prism": sigma_alpha_prism,
        "sigma_beta_prism": sigma_beta_prism,
        "rho_prism": rho_prism,
        "elapsed_s": time.time() - t0,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[save] {out_path}")
    print(f"[time] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
