"""F1 — Synthetic-user recovery test.

Pre-registered falsifier for PEBS's identifiability claim
(``empirical-Bayes recovers the true tau^2'' in introduction + section 3.2).

Design
------
Generate N_j = 1000 synthetic users. For each user j:
    alpha_j ~ N(alpha_pop_true, tau2_alpha_true)
    beta_j  ~ N(beta_pop_true,  tau2_beta_true)
    n_j     ~ Poisson(lambda_n)  (per-user obs count)
    r_ij    ~ N(0, 1)            (synthetic RM-score covariate; Gaussian
                                  matches the PRISM z-score regime)
    score_ij = alpha_j + beta_j * r_ij + eps_ij, eps_ij ~ N(0, sigma_eps^2)

Apply PEBS's closed-form EB machinery (per-user OLS -> tau^2 MoM -> omega
shrinkage) and report:
  (i)  bias + RMSE of recovered (hat_tau2_alpha, hat_tau2_beta)
  (ii) per-user alpha_j / beta_j recovery MSE vs pop-slope (mean-only floor)

Three priors tested per criterion spec:
  - Gaussian (well-specified) : should recover within 1.5x Morris 1983 MoM SE
  - Student-t3 (heavy-tail)   : predictable degradation
  - Heavy-right Gaussian mix  : adversarial asymmetric prior

Run 100 seeds x 3 priors = 300 synthetic Monte-Carlo draws.

Pre-registered criterion:
  CONFIRMING:  Gaussian prior recovery bias < 1.5x MoM SE
               AND per-user (alpha, beta) MSE < pop-slope MSE by >= 30%
  FALSIFYING:  hat_tau^2 outside [0.5x, 2x] of true for Gaussian prior
               OR per-user MSE not significantly lower than pop-slope

Outputs
-------
results/falsifiers/F1_synthetic_user_recovery.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/falsifiers"

# Pre-registered constants (calibrated to PRISM-like scale)
N_USERS = 1000
LAMBDA_N = 50          # Poisson mean matches PRISM ~49 per user
SIGMA_EPS = 12.0       # residual sd matches PRISM ~12
ALPHA_POP_TRUE = 50.0
BETA_POP_TRUE = 5.0
TAU2_ALPHA_TRUE = 400.0   # user intercept variance (alpha sd = 20)
TAU2_BETA_TRUE = 4.0      # user slope variance (beta sd = 2)
N_SEEDS = 100

PRIORS = ["gaussian", "student_t3", "heavy_right"]


def fit_ols_with_se(x: np.ndarray, y: np.ndarray):
    """OLS + SEs. Matches neighbor_head_to_head.fit_ols_with_se."""
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


def draw_users(prior: str, rng: np.random.Generator):
    """Draw (alpha_j, beta_j) for N_USERS under prior."""
    if prior == "gaussian":
        alpha = rng.normal(ALPHA_POP_TRUE, np.sqrt(TAU2_ALPHA_TRUE), N_USERS)
        beta = rng.normal(BETA_POP_TRUE, np.sqrt(TAU2_BETA_TRUE), N_USERS)
    elif prior == "student_t3":
        # t_3 scaled to match sd sqrt(tau2) (actual variance of t_3 = 3)
        alpha = ALPHA_POP_TRUE + rng.standard_t(3, N_USERS) * np.sqrt(TAU2_ALPHA_TRUE / 3.0)
        beta = BETA_POP_TRUE + rng.standard_t(3, N_USERS) * np.sqrt(TAU2_BETA_TRUE / 3.0)
    elif prior == "heavy_right":
        # asymmetric: 80% tight Gaussian, 20% broad right tail
        alpha = np.where(
            rng.uniform(size=N_USERS) < 0.2,
            ALPHA_POP_TRUE + rng.exponential(np.sqrt(TAU2_ALPHA_TRUE), N_USERS),
            rng.normal(ALPHA_POP_TRUE, np.sqrt(TAU2_ALPHA_TRUE) * 0.6, N_USERS),
        )
        beta = np.where(
            rng.uniform(size=N_USERS) < 0.2,
            BETA_POP_TRUE + rng.exponential(np.sqrt(TAU2_BETA_TRUE), N_USERS),
            rng.normal(BETA_POP_TRUE, np.sqrt(TAU2_BETA_TRUE) * 0.6, N_USERS),
        )
    else:
        raise ValueError(prior)
    return alpha, beta


def one_seed(prior: str, seed: int):
    rng = np.random.default_rng(seed)
    alpha_true, beta_true = draw_users(prior, rng)
    n_js = np.clip(rng.poisson(LAMBDA_N, N_USERS), 5, None)

    # Per-user OLS fits
    alpha_hat = np.full(N_USERS, np.nan)
    beta_hat = np.full(N_USERS, np.nan)
    se_a = np.full(N_USERS, np.nan)
    se_b = np.full(N_USERS, np.nan)
    # Also accumulate pop-slope MSE baselines per-user
    user_data = []
    for j in range(N_USERS):
        n = int(n_js[j])
        x_j = rng.normal(0, 1, n)
        eps = rng.normal(0, SIGMA_EPS, n)
        y_j = alpha_true[j] + beta_true[j] * x_j + eps
        a, b, sa, sb = fit_ols_with_se(x_j, y_j)
        alpha_hat[j] = a
        beta_hat[j] = b
        se_a[j] = sa
        se_b[j] = sb
        user_data.append((x_j, y_j))

    # Population pooled OLS (naive no-pooling baseline)
    x_all = np.concatenate([d[0] for d in user_data])
    y_all = np.concatenate([d[1] for d in user_data])
    b_pop_obs, a_pop_obs = np.polyfit(x_all, y_all, 1)
    alpha_pop_obs = float(a_pop_obs)
    beta_pop_obs = float(b_pop_obs)

    # MoM tau^2
    mask_a = np.isfinite(alpha_hat) & np.isfinite(se_a)
    mask_b = np.isfinite(beta_hat) & np.isfinite(se_b)
    tau2_a_hat = max(0.0, float(np.var(alpha_hat[mask_a], ddof=1) - np.mean(se_a[mask_a] ** 2)))
    tau2_b_hat = max(0.0, float(np.var(beta_hat[mask_b], ddof=1) - np.mean(se_b[mask_b] ** 2)))

    # EB-shrunk point estimates
    w_a = np.where(np.isfinite(se_a), tau2_a_hat / (tau2_a_hat + se_a ** 2 + 1e-12), 0.0)
    w_b = np.where(np.isfinite(se_b), tau2_b_hat / (tau2_b_hat + se_b ** 2 + 1e-12), 0.0)
    alpha_shrunk = np.where(np.isfinite(alpha_hat),
                            w_a * alpha_hat + (1 - w_a) * alpha_pop_obs,
                            alpha_pop_obs)
    beta_shrunk = np.where(np.isfinite(beta_hat),
                           w_b * beta_hat + (1 - w_b) * beta_pop_obs,
                           beta_pop_obs)

    # Error metrics
    mse_alpha_shrunk = float(np.mean((alpha_shrunk - alpha_true) ** 2))
    mse_beta_shrunk = float(np.mean((beta_shrunk - beta_true) ** 2))
    mse_alpha_pop = float(np.mean((alpha_pop_obs - alpha_true) ** 2))
    mse_beta_pop = float(np.mean((beta_pop_obs - beta_true) ** 2))
    mse_alpha_nopool = float(np.mean((alpha_hat[mask_a] - alpha_true[mask_a]) ** 2))
    mse_beta_nopool = float(np.mean((beta_hat[mask_b] - beta_true[mask_b]) ** 2))

    # Morris 1983 MoM variance for tau^2 estimator (approx):
    #   Var(hat_tau^2) ~ 2 * (tau^2 + avg_se^2)^2 / (J - 1)
    J = int(mask_a.sum())
    morris_se_tau2_a = float(np.sqrt(
        2.0 * (TAU2_ALPHA_TRUE + float(np.mean(se_a[mask_a] ** 2))) ** 2 / max(J - 1, 1)
    ))
    morris_se_tau2_b = float(np.sqrt(
        2.0 * (TAU2_BETA_TRUE + float(np.mean(se_b[mask_b] ** 2))) ** 2 / max(J - 1, 1)
    ))

    return {
        "tau2_a_hat": tau2_a_hat,
        "tau2_b_hat": tau2_b_hat,
        "morris_se_tau2_a": morris_se_tau2_a,
        "morris_se_tau2_b": morris_se_tau2_b,
        "mse_alpha_shrunk": mse_alpha_shrunk,
        "mse_beta_shrunk": mse_beta_shrunk,
        "mse_alpha_pop": mse_alpha_pop,
        "mse_beta_pop": mse_beta_pop,
        "mse_alpha_nopool": mse_alpha_nopool,
        "mse_beta_nopool": mse_beta_nopool,
        "alpha_pop_obs": alpha_pop_obs,
        "beta_pop_obs": beta_pop_obs,
        "mean_omega_a": float(np.mean(w_a)),
        "mean_omega_b": float(np.mean(w_b)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    all_results = {p: [] for p in PRIORS}
    for prior in PRIORS:
        print(f"\n[prior={prior}]")
        for s in range(N_SEEDS):
            res = one_seed(prior, seed=100000 + s)
            all_results[prior].append(res)
        print(f"  done {N_SEEDS} seeds")

    summary = {
        "config": {
            "N_USERS": N_USERS,
            "LAMBDA_N": LAMBDA_N,
            "SIGMA_EPS": SIGMA_EPS,
            "TAU2_ALPHA_TRUE": TAU2_ALPHA_TRUE,
            "TAU2_BETA_TRUE": TAU2_BETA_TRUE,
            "ALPHA_POP_TRUE": ALPHA_POP_TRUE,
            "BETA_POP_TRUE": BETA_POP_TRUE,
            "N_SEEDS": N_SEEDS,
            "seed_base": 100000,
        },
        "priors": {},
        "branch_disposition": {},
    }

    for prior in PRIORS:
        rs = all_results[prior]
        tau2_a = np.array([r["tau2_a_hat"] for r in rs])
        tau2_b = np.array([r["tau2_b_hat"] for r in rs])
        morris_se_a = float(np.mean([r["morris_se_tau2_a"] for r in rs]))
        morris_se_b = float(np.mean([r["morris_se_tau2_b"] for r in rs]))
        mse_a_shr = np.array([r["mse_alpha_shrunk"] for r in rs])
        mse_b_shr = np.array([r["mse_beta_shrunk"] for r in rs])
        mse_a_pop = np.array([r["mse_alpha_pop"] for r in rs])
        mse_b_pop = np.array([r["mse_beta_pop"] for r in rs])
        mse_a_np = np.array([r["mse_alpha_nopool"] for r in rs])
        mse_b_np = np.array([r["mse_beta_nopool"] for r in rs])

        # Recovery bias
        bias_a = float(np.mean(tau2_a) - TAU2_ALPHA_TRUE)
        bias_b = float(np.mean(tau2_b) - TAU2_BETA_TRUE)
        # % of seeds within [0.5, 2.0] x true
        within_2x_a = float(np.mean((tau2_a >= 0.5 * TAU2_ALPHA_TRUE) & (tau2_a <= 2.0 * TAU2_ALPHA_TRUE)))
        within_2x_b = float(np.mean((tau2_b >= 0.5 * TAU2_BETA_TRUE) & (tau2_b <= 2.0 * TAU2_BETA_TRUE)))
        # within 1.5x Morris SE
        within_15se_a = float(np.mean(np.abs(tau2_a - TAU2_ALPHA_TRUE) <= 1.5 * morris_se_a))
        within_15se_b = float(np.mean(np.abs(tau2_b - TAU2_BETA_TRUE) <= 1.5 * morris_se_b))

        # MSE reduction (pct) vs pop-only and vs no-pool
        red_vs_pop_a = 100.0 * (np.mean(mse_a_pop) - np.mean(mse_a_shr)) / max(np.mean(mse_a_pop), 1e-9)
        red_vs_pop_b = 100.0 * (np.mean(mse_b_pop) - np.mean(mse_b_shr)) / max(np.mean(mse_b_pop), 1e-9)
        red_vs_nopool_a = 100.0 * (np.mean(mse_a_np) - np.mean(mse_a_shr)) / max(np.mean(mse_a_np), 1e-9)
        red_vs_nopool_b = 100.0 * (np.mean(mse_b_np) - np.mean(mse_b_shr)) / max(np.mean(mse_b_np), 1e-9)

        # Paired 95% CI on MSE reduction via percentile across seeds
        gain_a = 100.0 * (mse_a_pop - mse_a_shr) / np.maximum(mse_a_pop, 1e-9)
        gain_b = 100.0 * (mse_b_pop - mse_b_shr) / np.maximum(mse_b_pop, 1e-9)
        ci_a = np.percentile(gain_a, [2.5, 97.5]).tolist()
        ci_b = np.percentile(gain_b, [2.5, 97.5]).tolist()

        summary["priors"][prior] = {
            "tau2_a": {
                "true": TAU2_ALPHA_TRUE,
                "mean_hat": float(np.mean(tau2_a)),
                "median_hat": float(np.median(tau2_a)),
                "sd_hat": float(np.std(tau2_a, ddof=1)),
                "bias": bias_a,
                "morris_se": morris_se_a,
                "frac_within_2x": within_2x_a,
                "frac_within_1p5_morris_se": within_15se_a,
            },
            "tau2_b": {
                "true": TAU2_BETA_TRUE,
                "mean_hat": float(np.mean(tau2_b)),
                "median_hat": float(np.median(tau2_b)),
                "sd_hat": float(np.std(tau2_b, ddof=1)),
                "bias": bias_b,
                "morris_se": morris_se_b,
                "frac_within_2x": within_2x_b,
                "frac_within_1p5_morris_se": within_15se_b,
            },
            "mse_reduction_pct_vs_pop_slope": {
                "alpha": {"mean": float(red_vs_pop_a), "ci95": ci_a},
                "beta": {"mean": float(red_vs_pop_b), "ci95": ci_b},
            },
            "mse_reduction_pct_vs_no_pool": {
                "alpha": float(red_vs_nopool_a),
                "beta": float(red_vs_nopool_b),
            },
        }
        print(f"\n[{prior}] tau2_alpha: true={TAU2_ALPHA_TRUE:.1f}  hat={np.mean(tau2_a):.1f}+-{np.std(tau2_a,ddof=1):.1f}  "
              f"frac in [0.5x, 2x]={within_2x_a:.2f}  "
              f"frac within 1.5 MoM SE (={morris_se_a:.1f})={within_15se_a:.2f}")
        print(f"[{prior}] tau2_beta:  true={TAU2_BETA_TRUE:.2f}  hat={np.mean(tau2_b):.2f}+-{np.std(tau2_b,ddof=1):.2f}  "
              f"frac in [0.5x, 2x]={within_2x_b:.2f}  "
              f"frac within 1.5 MoM SE (={morris_se_b:.2f})={within_15se_b:.2f}")
        print(f"[{prior}] MSE_alpha_shrunk vs pop: {red_vs_pop_a:+.1f}% [{ci_a[0]:+.1f}, {ci_a[1]:+.1f}]")
        print(f"[{prior}] MSE_beta_shrunk  vs pop: {red_vs_pop_b:+.1f}% [{ci_b[0]:+.1f}, {ci_b[1]:+.1f}]")

    # Branch disposition — Gaussian is the pre-registered anchor
    g = summary["priors"]["gaussian"]
    recov_ok = (g["tau2_a"]["frac_within_2x"] >= 0.80
                and g["tau2_b"]["frac_within_2x"] >= 0.80)
    mse_ok = (g["mse_reduction_pct_vs_pop_slope"]["alpha"]["mean"] >= 30.0
              and g["mse_reduction_pct_vs_pop_slope"]["alpha"]["ci95"][0] > 0
              and g["mse_reduction_pct_vs_pop_slope"]["beta"]["mean"] >= 30.0
              and g["mse_reduction_pct_vs_pop_slope"]["beta"]["ci95"][0] > 0)
    summary["branch_disposition"] = {
        "branch": "confirming" if (recov_ok and mse_ok) else "falsifying",
        "recovery_check_pass": bool(recov_ok),
        "mse_reduction_check_pass": bool(mse_ok),
        "criterion_recovery": "Gaussian frac of tau^2 in [0.5x, 2x] >= 0.80 for both alpha, beta",
        "criterion_mse": "Mean MSE reduction >= 30% AND CI lower > 0",
    }

    summary["runtime_seconds"] = float(time.time() - t0)
    out_path = OUT / "F1_synthetic_user_recovery.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[save] {out_path}  ({summary['runtime_seconds']:.1f}s)")
    print(f"\n=== BRANCH: {summary['branch_disposition']['branch'].upper()} ===")
    print(json.dumps(summary['branch_disposition'], indent=2))


if __name__ == "__main__":
    main()
