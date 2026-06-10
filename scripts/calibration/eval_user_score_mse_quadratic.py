"""Linear vs QUADRATIC per-user calibration — within-user held-out RMSE.

Paper §5.3 future-work: does adding a quadratic term (rm_score^2) to the
per-user calibrator yield real predictive gain over the linear form?

Arms (evaluated on SAME 5-fold CV splits, matching eval_user_score_mse_shrunk.py):

  no_calib              predict train-fold mean of y
  pop_slope_linear      global α₀ + β₀ · x           (linear baseline)
  pebs_linear_ols      per-user linear OLS on train fold (naive)
  pebs_linear_shrunk   EB shrunk linear (ω = τ²/(τ²+V))
  pop_slope_quadratic   global α₀ + β₀ · x + γ₀ · x² (quadratic pop baseline)
  pebs_quadratic_ols   per-user quadratic OLS on train fold
  pebs_quadratic_shrunk EB shrunk quadratic (ω applied per coefficient)

The shrinkage form is identical in structure: for each coefficient c ∈ {α,β,γ}
we compute ω_c = τ_c² / (τ_c² + V_c), where τ_c² is the between-user variance
of the coefficient (after removing mean within-user sampling variance) and
V_c is the within-fold sampling variance of the coefficient's OLS estimate.

Key comparison: Wilcoxon paired on rmse_pebs_quadratic_shrunk vs
rmse_pebs_linear_shrunk, and a cluster-bootstrap 95% CI on the mean delta
(positive -> quadratic is better).

Refs: Hastie et al. 2009 ESL §7.10; Pinheiro & Bates 2000 §2;
Gelman & Hill 2007 §12.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scored-parquet", default="data/prism_rm_scored.parquet")
    p.add_argument("--min-obs-per-user", type=int, default=6)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42,
                   help="MUST match eval_user_score_mse_shrunk.py (default=42) "
                        "to reproduce identical fold indices.")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--output-path", default="results/track1_user_score_mse_quadratic.json")
    return p.parse_args()


def kfold_split(n: int, k: int, rng: np.random.Generator):
    """Same deterministic k-fold routine as eval_user_score_mse_shrunk.py."""
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


def ols_linear_with_V(x: np.ndarray, y: np.ndarray):
    """OLS linear fit: y = intercept + slope * x. Returns (int, slope, V_int, V_slope)."""
    k = len(x)
    if k < 2 or np.var(x) < 1e-12:
        return float(np.mean(y)) if k else 0.0, 0.0, np.inf, np.inf
    x_bar = x.mean()
    Sxx = ((x - x_bar) ** 2).sum()
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = intercept + slope * x
    sigma_hat_sq = ((y - y_pred) ** 2).sum() / max(k - 2, 1)
    V_int = sigma_hat_sq * (1.0 / k + x_bar ** 2 / max(Sxx, 1e-12))
    V_slope = sigma_hat_sq / max(Sxx, 1e-12)
    return float(intercept), float(slope), float(V_int), float(V_slope)


def ols_quadratic_with_V(x: np.ndarray, y: np.ndarray):
    """OLS quadratic fit: y = intercept + lin * x + quad * x^2.

    Returns (intercept, lin_slope, quad_coef, V_int, V_lin, V_quad).
    Uses the design-matrix covariance (X'X)^-1 * sigma_hat^2 to extract the
    three diagonal variances for EB shrinkage.
    """
    k = len(x)
    # Need at least 3 points to identify 3 coefficients. Fall back to linear
    # (or mean) when not enough support.
    if k < 3 or np.var(x) < 1e-12:
        if k >= 2 and np.var(x) >= 1e-12:
            a, b, Va, Vb = ols_linear_with_V(x, y)
            return a, b, 0.0, Va, Vb, np.inf
        return float(np.mean(y)) if k else 0.0, 0.0, 0.0, np.inf, np.inf, np.inf
    X = np.column_stack([np.ones(k), x, x ** 2])
    try:
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        a, b, Va, Vb = ols_linear_with_V(x, y)
        return a, b, 0.0, Va, Vb, np.inf
    beta_hat = XtX_inv @ X.T @ y
    y_pred = X @ beta_hat
    resid = y - y_pred
    sigma_hat_sq = (resid @ resid) / max(k - 3, 1)
    V = sigma_hat_sq * np.diag(XtX_inv)
    return (float(beta_hat[0]), float(beta_hat[1]), float(beta_hat[2]),
            float(V[0]), float(V[1]), float(V[2]))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.scored_parquet).dropna(subset=["score_user"]).reset_index(drop=True)
    print(f"[load] {len(df)} utterances, {df.user_id.nunique()} users")

    # -------------------- Pop-level baselines --------------------
    # Linear: global α₀ + β₀·x (match eval_user_score_mse_shrunk.py convention)
    slope_pop_lin, intercept_pop_lin = np.polyfit(df.rm_score, df.score_user, 1)
    pop_int_lin = float(intercept_pop_lin)
    pop_slope_lin = float(slope_pop_lin)
    print(f"[pop-lin]  α₀={pop_int_lin:.3f}  β₀={pop_slope_lin:.3f}")

    # Quadratic population baseline
    q, m, c = np.polyfit(df.rm_score, df.score_user, 2)  # returns [quad, lin, int]
    pop_int_qd = float(c)
    pop_lin_qd = float(m)
    pop_quad_qd = float(q)
    print(f"[pop-qd]  γ₀={pop_int_qd:.3f}  β₀={pop_lin_qd:.3f}  α₀={pop_quad_qd:.3f}")

    # -------------------- Pre-pass for EB τ² --------------------
    # Fit per-user OLS (linear + quadratic) on ALL data to estimate τ² for
    # each coefficient (matching the main eval's formula).
    user_stats_lin = []
    user_stats_qd = []
    for uid, grp in df.groupby("user_id"):
        if len(grp) < args.min_obs_per_user:
            continue
        x = grp.rm_score.to_numpy()
        y = grp.score_user.to_numpy().astype(float)
        a, b, Va, Vb = ols_linear_with_V(x, y)
        user_stats_lin.append({"user_id": uid, "a": a, "b": b, "Va": Va, "Vb": Vb})
        aq, bq, cq, Vaq, Vbq, Vcq = ols_quadratic_with_V(x, y)
        user_stats_qd.append({"user_id": uid, "a": aq, "b": bq, "c": cq,
                              "Va": Vaq, "Vb": Vbq, "Vc": Vcq})

    us_lin = pd.DataFrame(user_stats_lin)
    us_qd = pd.DataFrame(user_stats_qd)

    # Linear τ²
    tau_a_lin = max(float(us_lin.a.var()) -
                    float(us_lin.Va.replace([np.inf, -np.inf], np.nan).dropna().mean()),
                    1e-6)
    tau_b_lin = max(float(us_lin.b.var()) -
                    float(us_lin.Vb.replace([np.inf, -np.inf], np.nan).dropna().mean()),
                    1e-6)
    print(f"[EB-lin] τ_α²={tau_a_lin:.3f}  τ_β²={tau_b_lin:.3f}")

    # Quadratic τ² (one per coefficient: intercept, linear slope, quad coef)
    tau_a_qd = max(float(us_qd.a.var()) -
                   float(us_qd.Va.replace([np.inf, -np.inf], np.nan).dropna().mean()),
                   1e-6)
    tau_b_qd = max(float(us_qd.b.var()) -
                   float(us_qd.Vb.replace([np.inf, -np.inf], np.nan).dropna().mean()),
                   1e-6)
    tau_c_qd = max(float(us_qd.c.var()) -
                   float(us_qd.Vc.replace([np.inf, -np.inf], np.nan).dropna().mean()),
                   1e-6)
    print(f"[EB-qd]  τ_intercept²={tau_a_qd:.3f}  τ_lin²={tau_b_qd:.3f}  τ_quad²={tau_c_qd:.3f}")

    # -------------------- Per-user k-fold CV with 7 arms --------------------
    arm_names = [
        "no_calib",
        "pop_slope_linear",
        "pebs_linear_ols",
        "pebs_linear_shrunk",
        "pop_slope_quadratic",
        "pebs_quadratic_ols",
        "pebs_quadratic_shrunk",
    ]

    per_user_rows = []
    t0 = time.time()
    total_users = 0
    for uid, grp in df.groupby("user_id"):
        n = len(grp)
        if n < args.min_obs_per_user:
            continue
        total_users += 1
        x = grp.rm_score.to_numpy()
        y = grp.score_user.to_numpy().astype(float)
        folds = kfold_split(n, args.k_folds, rng)
        squared = {arm: [] for arm in arm_names}

        for train_idx, test_idx in folds:
            x_tr, y_tr = x[train_idx], y[train_idx]
            x_te, y_te = x[test_idx], y[test_idx]
            if len(x_te) == 0:
                continue

            # (1) No-calib
            squared["no_calib"].extend(((float(np.mean(y_tr)) - y_te) ** 2).tolist())

            # (2) Pop linear
            yhat = pop_int_lin + pop_slope_lin * x_te
            squared["pop_slope_linear"].extend(((yhat - y_te) ** 2).tolist())

            # (3) PEBS linear OLS
            a, b, Va, Vb = ols_linear_with_V(x_tr, y_tr)
            yhat = a + b * x_te
            squared["pebs_linear_ols"].extend(((yhat - y_te) ** 2).tolist())

            # (4) PEBS linear shrunk (coefficient-wise EB)
            wa = tau_a_lin / (tau_a_lin + Va) if np.isfinite(Va) else 0.0
            wb = tau_b_lin / (tau_b_lin + Vb) if np.isfinite(Vb) else 0.0
            a_s = wa * a + (1 - wa) * pop_int_lin
            b_s = wb * b + (1 - wb) * pop_slope_lin
            yhat = a_s + b_s * x_te
            squared["pebs_linear_shrunk"].extend(((yhat - y_te) ** 2).tolist())

            # (5) Pop quadratic
            yhat = pop_int_qd + pop_lin_qd * x_te + pop_quad_qd * x_te ** 2
            squared["pop_slope_quadratic"].extend(((yhat - y_te) ** 2).tolist())

            # (6) PEBS quadratic OLS
            aq, bq, cq, Vaq, Vbq, Vcq = ols_quadratic_with_V(x_tr, y_tr)
            yhat = aq + bq * x_te + cq * x_te ** 2
            squared["pebs_quadratic_ols"].extend(((yhat - y_te) ** 2).tolist())

            # (7) PEBS quadratic shrunk — coefficient-wise ω toward pop quadratic
            waq = tau_a_qd / (tau_a_qd + Vaq) if np.isfinite(Vaq) else 0.0
            wbq = tau_b_qd / (tau_b_qd + Vbq) if np.isfinite(Vbq) else 0.0
            wcq = tau_c_qd / (tau_c_qd + Vcq) if np.isfinite(Vcq) else 0.0
            a_sq = waq * aq + (1 - waq) * pop_int_qd
            b_sq = wbq * bq + (1 - wbq) * pop_lin_qd
            c_sq = wcq * cq + (1 - wcq) * pop_quad_qd
            yhat = a_sq + b_sq * x_te + c_sq * x_te ** 2
            squared["pebs_quadratic_shrunk"].extend(((yhat - y_te) ** 2).tolist())

        row = {"user_id": uid, "n": n,
               **{f"rmse_{arm}": float(np.sqrt(np.mean(squared[arm])))
                  for arm in arm_names}}
        per_user_rows.append(row)

    t_cv = time.time() - t0
    per_user_time = t_cv / max(total_users, 1)
    print(f"\n[cv] total users={total_users}  wall={t_cv:.1f}s  per-user={per_user_time*1000:.1f}ms")

    pu = pd.DataFrame(per_user_rows)

    # -------------------- Aggregate --------------------
    print(f"\n=== 7-arm within-user CV (n_users={len(pu)}, k={args.k_folds}) ===")
    agg_mean = {}
    agg_median = {}
    for arm in arm_names:
        col = f"rmse_{arm}"
        agg_mean[arm] = float(pu[col].mean())
        agg_median[arm] = float(pu[col].median())
        print(f"  {col:<34} mean={agg_mean[arm]:.4f}  median={agg_median[arm]:.4f}")

    # -------------------- Paired tests --------------------
    def paired(a_col, b_col):
        a, b = pu[a_col].to_numpy(), pu[b_col].to_numpy()
        w = stats.wilcoxon(a, b, alternative="two-sided")
        return {
            "mean_delta_a_minus_b": float((a - b).mean()),
            "median_delta_a_minus_b": float(np.median(a - b)),
            "frac_a_smaller": float((a < b).mean()),
            "wilcoxon_stat": float(w.statistic),
            "wilcoxon_p": float(w.pvalue),
        }

    comparisons = {
        # Headline comparison
        "quadratic_shrunk_vs_linear_shrunk": paired("rmse_pebs_quadratic_shrunk",
                                                    "rmse_pebs_linear_shrunk"),
        "quadratic_ols_vs_linear_ols":       paired("rmse_pebs_quadratic_ols",
                                                    "rmse_pebs_linear_ols"),
        "pop_quadratic_vs_pop_linear":       paired("rmse_pop_slope_quadratic",
                                                    "rmse_pop_slope_linear"),
        # Shrunk vs OLS within each family
        "quadratic_shrunk_vs_quadratic_ols": paired("rmse_pebs_quadratic_shrunk",
                                                    "rmse_pebs_quadratic_ols"),
        "linear_shrunk_vs_linear_ols":       paired("rmse_pebs_linear_shrunk",
                                                    "rmse_pebs_linear_ols"),
    }
    print(f"\n=== Paired Wilcoxon (negative mean_delta -> FIRST is better) ===")
    for name, d in comparisons.items():
        sign = "↓" if d["mean_delta_a_minus_b"] < 0 else "↑"
        print(f"  {name:<38} Δ={d['mean_delta_a_minus_b']:+.4f} {sign}  "
              f"frac_first_smaller={d['frac_a_smaller']:.1%}  p={d['wilcoxon_p']:.3e}")

    # -------------------- Bootstrap CI on key delta --------------------
    # Headline: is RMSE_linear_shrunk - RMSE_quadratic_shrunk > 0?
    rng_ci = np.random.default_rng(args.seed + 1)
    n_users = len(pu)
    idx_arr = np.arange(n_users)

    def boot_ci(a_vals, b_vals):
        deltas = []
        for _ in range(args.n_boot):
            sub = rng_ci.choice(idx_arr, size=n_users, replace=True)
            deltas.append(float(a_vals[sub].mean() - b_vals[sub].mean()))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        return float(lo), float(hi), float(np.mean(deltas)), deltas

    a_lin = pu.rmse_pebs_linear_shrunk.to_numpy()
    a_qd = pu.rmse_pebs_quadratic_shrunk.to_numpy()
    lo, hi, m, _ = boot_ci(a_lin, a_qd)
    print(f"\n=== Headline: mean(RMSE_linear_shrunk) - mean(RMSE_quadratic_shrunk) ===")
    print(f"  point estimate: {agg_mean['pebs_linear_shrunk'] - agg_mean['pebs_quadratic_shrunk']:+.5f}")
    print(f"  cluster-bootstrap mean: {m:+.5f}")
    print(f"  95% CI: [{lo:+.5f}, {hi:+.5f}]  (n_boot={args.n_boot})")
    includes_zero = lo <= 0.0 <= hi
    print(f"  CI includes 0? {includes_zero}")

    # Relative improvements vs pop_slope_linear (matches paper headline frame)
    rel_lin = 100 * (agg_mean["pop_slope_linear"] - agg_mean["pebs_linear_shrunk"]) / agg_mean["pop_slope_linear"]
    rel_qd = 100 * (agg_mean["pop_slope_linear"] - agg_mean["pebs_quadratic_shrunk"]) / agg_mean["pop_slope_linear"]
    rel_qd_vs_lin = 100 * (agg_mean["pebs_linear_shrunk"] - agg_mean["pebs_quadratic_shrunk"]) / agg_mean["pebs_linear_shrunk"]

    print(f"\n=== Relative improvement vs pop_slope_linear (paper-style) ===")
    print(f"  pebs_linear_shrunk:    {rel_lin:+.3f}%")
    print(f"  pebs_quadratic_shrunk: {rel_qd:+.3f}%")
    print(f"  quadratic over linear (of linear mean): {rel_qd_vs_lin:+.3f}%")

    # -------------------- Verdict --------------------
    verdict = None
    if (lo > 0) and comparisons["quadratic_shrunk_vs_linear_shrunk"]["wilcoxon_p"] < 0.05 and \
       comparisons["quadratic_shrunk_vs_linear_shrunk"]["mean_delta_a_minus_b"] < 0:
        verdict = "QUADRATIC_BETTER"
    elif (hi < 0) or (comparisons["quadratic_shrunk_vs_linear_shrunk"]["mean_delta_a_minus_b"] > 0 and
                      comparisons["quadratic_shrunk_vs_linear_shrunk"]["wilcoxon_p"] < 0.05):
        verdict = "QUADRATIC_WORSE_OVERFIT"
    else:
        verdict = "HONEST_NULL"
    print(f"\n=== VERDICT: {verdict} ===")

    # -------------------- Save --------------------
    out = {
        "n_users": int(len(pu)),
        "total_users_kept": int(total_users),
        "k_folds": int(args.k_folds),
        "wallclock_seconds_cv": float(t_cv),
        "per_user_fit_ms": float(per_user_time * 1000),
        "seed": int(args.seed),
        "eb_linear": {"tau_alpha_sq": tau_a_lin, "tau_beta_sq": tau_b_lin},
        "eb_quadratic": {"tau_intercept_sq": tau_a_qd,
                         "tau_linear_sq": tau_b_qd,
                         "tau_quadratic_sq": tau_c_qd},
        "pop_linear": {"intercept": pop_int_lin, "slope": pop_slope_lin},
        "pop_quadratic": {"intercept": pop_int_qd, "linear": pop_lin_qd,
                          "quadratic": pop_quad_qd},
        "aggregate_rmse_mean": agg_mean,
        "aggregate_rmse_median": agg_median,
        "comparisons": comparisons,
        "bootstrap_delta_linear_minus_quadratic_shrunk": {
            "point_estimate": float(agg_mean["pebs_linear_shrunk"] - agg_mean["pebs_quadratic_shrunk"]),
            "bootstrap_mean": float(m),
            "ci95_lo": float(lo),
            "ci95_hi": float(hi),
            "includes_zero": bool(includes_zero),
            "n_boot": int(args.n_boot),
        },
        "relative_improvement_vs_pop_slope_linear_pct": {
            "pebs_linear_shrunk": float(rel_lin),
            "pebs_quadratic_shrunk": float(rel_qd),
            "quadratic_over_linear_of_linear_mean_pct": float(rel_qd_vs_lin),
        },
        "verdict": verdict,
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    pu.to_parquet(out_path.with_suffix(".parquet"))
    print(f"\n[save] {out_path} + .parquet")


if __name__ == "__main__":
    main()
