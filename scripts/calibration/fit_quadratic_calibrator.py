"""Per-user OLS quadratic calibrator fit WITH γ_j significance diagnostic.

This is the ICLR-grade falsifiability-scope companion to
`fit_user_calibrators_quadratic.py`. Whereas that script runs a pooled
MixedLM (REML) to harvest Empirical-Bayes-shrunk coefficients, this
script runs *per-user* OLS independently on each user's data, extracts
the raw γ_j (quadratic coefficient), and tests H0: γ_j = 0 under a
Wald t-test with design-matrix variance.

This answers the specific scientific question raised by Prop T1.MI's
falsifiable extension:

    "For what fraction of PRISM annotators is the quadratic term
     statistically distinguishable from zero?"

If γ_j is non-zero for a large share of users, the 'r' = α_j + β_j·r +
γ_j·r²' calibrator is genuinely non-affine per user, and the affine
monotone-invariance premise of Prop T1.MI is violated for those users.

Protocol (matches eval_user_score_mse_quadratic.py)
---------------------------------------------------
1. Standardise rm_score to rm_z (z-score over the full corpus).
2. Centre rm_z^2 by subtracting its corpus mean (reduces collinearity).
3. For each user j with ≥ 6 observations, fit:
       score_user = γ_j + β_j · rm_z + α_j · rm_z_sq_c  + ε
   via ordinary least squares, with heteroskedasticity-robust (HC1)
   standard errors for the Wald t-test on α_j (the quadratic coef).
4. Also run k=5 CV within-user to report held-out RMSE per user for
   three arms (pop linear, PILSD linear OLS, PILSD quadratic OLS).
5. Save per-user parquet + summary.json under
   `results/track1_quadratic_calibrator/`.

Implementation notes
--------------------
- We use HC1 robust variance because PRISM scores are ordinal 0-100
  and residuals are heteroskedastic near the boundaries.
- For users with <3 observations in a fold we fall back to the linear
  OLS estimate (the ols_quadratic_with_V path in the existing eval).
- Significance fraction is reported at α ∈ {0.05, 0.01, 0.001} and
  with Bonferroni correction for n_users tests.

Refs: White 1980 (HC1); Hastie et al. 2009 ESL §7.10; Pinheiro & Bates
2000 §2; Gelman & Hill 2007 §4.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scored-parquet", default="data/prism_rm_scored.parquet")
    p.add_argument("--min-obs-per-user", type=int, default=6)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--output-dir", default="results/track1_quadratic_calibrator")
    return p.parse_args()


def kfold_split(n: int, k: int, rng: np.random.Generator):
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


def ols_quadratic_hc1(x_raw: np.ndarray, y: np.ndarray, sq_mean: float):
    """OLS quadratic fit with HC1 robust standard errors.

    Design: X = [1, x, (x^2 - sq_mean)]  (centered quadratic)
    Returns dict with point estimates, HC1 SE, t, p (two-sided), n.
    """
    n = len(x_raw)
    if n < 4 or np.var(x_raw) < 1e-12:
        return None
    x = x_raw
    xsq_c = x * x - sq_mean
    X = np.column_stack([np.ones(n), x, xsq_c])
    # statsmodels OLS with White 1980 HC1 robust covariance;
    # fallback to None on singular design (matches prior np.linalg.inv guard).
    try:
        res_hc1 = sm.OLS(y, X).fit(cov_type="HC1")
    except (np.linalg.LinAlgError, ValueError):
        return None
    res_cls = sm.OLS(y, X).fit()  # classical homoskedastic cov for comparison
    beta = np.asarray(res_hc1.params)
    k = X.shape[1]
    se = np.asarray(res_hc1.bse)
    se_cls = np.asarray(res_cls.bse)
    t = np.asarray(res_hc1.tvalues)
    p = np.asarray(res_hc1.pvalues)  # two-sided t(df=n-k) under HC1
    return {
        "gamma_j": float(beta[0]),   # intercept  (naming per theorem: r' = alpha + beta*r + gamma*r^2; but per-paper convention intercept=γ, linear=β, quad=α)
        "beta_j": float(beta[1]),
        "alpha_j": float(beta[2]),   # quadratic (per-paper convention matches quadratic calibrator memo)
        "se_intercept_hc1": float(se[0]),
        "se_linear_hc1":    float(se[1]),
        "se_quadratic_hc1": float(se[2]),
        "se_intercept_cls": float(se_cls[0]),
        "se_linear_cls":    float(se_cls[1]),
        "se_quadratic_cls": float(se_cls[2]),
        "t_intercept": float(t[0]),
        "t_linear":    float(t[1]),
        "t_quadratic": float(t[2]),
        "p_intercept": float(p[0]),
        "p_linear":    float(p[1]),
        "p_quadratic": float(p[2]),
        "n": int(n),
        "df": int(max(n - k, 1)),
    }


def ols_linear_fit(x: np.ndarray, y: np.ndarray):
    """Linear OLS y = a + b·x on train fold. Returns (a, b)."""
    if len(x) < 2 or np.var(x) < 1e-12:
        return float(np.mean(y)) if len(x) else 0.0, 0.0
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def ols_quad_fit(x: np.ndarray, y: np.ndarray, sq_mean: float):
    """Quadratic OLS y = γ + β x + α (x^2 - sq_mean). Returns (γ, β, α)."""
    n = len(x)
    if n < 3 or np.var(x) < 1e-12:
        a, b = ols_linear_fit(x, y)
        return a, b, 0.0
    xsq_c = x * x - sq_mean
    X = np.column_stack([np.ones(n), x, xsq_c])
    try:
        beta = np.linalg.solve(X.T @ X, X.T @ y)
    except np.linalg.LinAlgError:
        a, b = ols_linear_fit(x, y)
        return a, b, 0.0
    return float(beta[0]), float(beta[1]), float(beta[2])


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.scored_parquet).dropna(subset=["score_user"]).reset_index(drop=True)
    print(f"[load] {len(df)} utterances, {df.user_id.nunique()} users")

    rm_mean = float(df.rm_score.mean())
    rm_std = float(df.rm_score.std())
    # z-scored rm for coefficient stability
    df["rm_z"] = (df.rm_score - rm_mean) / max(rm_std, 1e-9)
    sq_mean = float((df["rm_z"] ** 2).mean())
    print(f"[feat] rm_mean={rm_mean:.3f} rm_std={rm_std:.3f} sq_mean={sq_mean:.4f}")

    # Pop coefficients (for baseline + shrinkage fallback)
    slope_pop_lin, intercept_pop_lin = np.polyfit(df.rm_z, df.score_user, 1)
    pop_a_lin = float(intercept_pop_lin)
    pop_b_lin = float(slope_pop_lin)
    # Quadratic pop coefficients (y = c + b*x + a*(x^2 - sq_mean))
    X_pop = np.column_stack([np.ones(len(df)), df.rm_z.to_numpy(),
                             df.rm_z.to_numpy() ** 2 - sq_mean])
    beta_pop = np.linalg.solve(X_pop.T @ X_pop, X_pop.T @ df.score_user.to_numpy())
    pop_c_qd = float(beta_pop[0])
    pop_b_qd = float(beta_pop[1])
    pop_a_qd = float(beta_pop[2])
    print(f"[pop] lin: intercept={pop_a_lin:.3f} slope={pop_b_lin:.3f}")
    print(f"[pop] qd:  intercept={pop_c_qd:.3f} linear={pop_b_qd:.3f} quadratic={pop_a_qd:.3f}")

    # ---------- Per-user OLS fits with HC1 t-tests on γ_j ----------
    t0 = time.time()
    per_user_fits = []
    skipped = 0
    for uid, grp in df.groupby("user_id"):
        n = len(grp)
        if n < args.min_obs_per_user:
            skipped += 1
            continue
        x_z = grp.rm_z.to_numpy()
        y = grp.score_user.to_numpy().astype(float)
        fit = ols_quadratic_hc1(x_z, y, sq_mean)
        if fit is None:
            skipped += 1
            continue
        fit["user_id"] = str(uid)
        # Also get linear-only fit for per-user baseline
        a_lin, b_lin = ols_linear_fit(x_z, y)
        fit["intercept_j_linear_only"] = a_lin
        fit["slope_j_linear_only"] = b_lin
        per_user_fits.append(fit)
    print(f"[fit] per-user OLS done for {len(per_user_fits)} users  wall={time.time() - t0:.1f}s  skipped={skipped}")

    fits_df = pd.DataFrame(per_user_fits)

    # ---------- Significance diagnostics on γ_j (the quadratic coef) ----------
    n_users = len(fits_df)
    p_quad = fits_df["p_quadratic"].to_numpy()
    # Bonferroni corrected
    bonf_threshold = 0.05 / max(n_users, 1)

    sig_counts = {
        "alpha_0p05":    int((p_quad < 0.05).sum()),
        "alpha_0p01":    int((p_quad < 0.01).sum()),
        "alpha_0p001":   int((p_quad < 0.001).sum()),
        "bonferroni_0p05": int((p_quad < bonf_threshold).sum()),
    }
    sig_fracs = {k: v / n_users for k, v in sig_counts.items()}
    print(f"[sig] γ_j significance (n_users={n_users}):")
    for k, v in sig_fracs.items():
        print(f"  {k}: {sig_counts[k]}/{n_users} = {v:.3%}")

    # ---------- γ_j distribution ----------
    alpha_stats = {
        "mean":   float(fits_df["alpha_j"].mean()),
        "median": float(fits_df["alpha_j"].median()),
        "std":    float(fits_df["alpha_j"].std()),
        "q05":    float(fits_df["alpha_j"].quantile(0.05)),
        "q25":    float(fits_df["alpha_j"].quantile(0.25)),
        "q75":    float(fits_df["alpha_j"].quantile(0.75)),
        "q95":    float(fits_df["alpha_j"].quantile(0.95)),
        "fraction_negative": float((fits_df["alpha_j"] < 0).mean()),
        "fraction_positive": float((fits_df["alpha_j"] > 0).mean()),
    }
    print(f"[dist] γ_j (quadratic) mean={alpha_stats['mean']:.3f}  median={alpha_stats['median']:.3f}  "
          f"q[5,25,75,95]=[{alpha_stats['q05']:.3f}, {alpha_stats['q25']:.3f}, "
          f"{alpha_stats['q75']:.3f}, {alpha_stats['q95']:.3f}]")
    print(f"[dist] γ_j frac_neg={alpha_stats['fraction_negative']:.3%}  "
          f"frac_pos={alpha_stats['fraction_positive']:.3%}")

    # ---------- k=5 CV RMSE per user (affine vs quadratic OLS) ----------
    print(f"[cv] k={args.k_folds} CV per-user RMSE")
    t0 = time.time()
    per_user_cv = []
    for uid, grp in df.groupby("user_id"):
        n = len(grp)
        if n < args.min_obs_per_user:
            continue
        x_z = grp.rm_z.to_numpy()
        y = grp.score_user.to_numpy().astype(float)
        folds = kfold_split(n, args.k_folds, rng)
        sq_lin, sq_quad, sq_pop_lin, sq_pop_quad = [], [], [], []
        for train_idx, test_idx in folds:
            x_tr, y_tr = x_z[train_idx], y[train_idx]
            x_te, y_te = x_z[test_idx], y[test_idx]
            if len(x_te) == 0:
                continue
            # pop linear
            yhat_pl = pop_a_lin + pop_b_lin * x_te
            sq_pop_lin.extend(((yhat_pl - y_te) ** 2).tolist())
            # pop quadratic
            yhat_pq = pop_c_qd + pop_b_qd * x_te + pop_a_qd * (x_te ** 2 - sq_mean)
            sq_pop_quad.extend(((yhat_pq - y_te) ** 2).tolist())
            # per-user linear OLS
            a_u, b_u = ols_linear_fit(x_tr, y_tr)
            yhat_lin = a_u + b_u * x_te
            sq_lin.extend(((yhat_lin - y_te) ** 2).tolist())
            # per-user quadratic OLS
            c_u, b_u_q, a_u_q = ols_quad_fit(x_tr, y_tr, sq_mean)
            yhat_quad = c_u + b_u_q * x_te + a_u_q * (x_te ** 2 - sq_mean)
            sq_quad.extend(((yhat_quad - y_te) ** 2).tolist())
        if not sq_lin:
            continue
        per_user_cv.append({
            "user_id": str(uid),
            "n": n,
            "rmse_pop_linear":    float(np.sqrt(np.mean(sq_pop_lin))),
            "rmse_pop_quadratic": float(np.sqrt(np.mean(sq_pop_quad))),
            "rmse_pilsd_linear_ols":    float(np.sqrt(np.mean(sq_lin))),
            "rmse_pilsd_quadratic_ols": float(np.sqrt(np.mean(sq_quad))),
        })
    cv_df = pd.DataFrame(per_user_cv)
    t_cv = time.time() - t0
    print(f"[cv] done wall={t_cv:.1f}s per-user={t_cv / max(len(cv_df), 1) * 1000:.1f}ms")

    # Join per-user OLS coefs + CV RMSEs
    merged = fits_df.merge(cv_df, on="user_id", how="inner")
    per_user_path = out_dir / "per_user.parquet"
    merged.to_parquet(per_user_path)
    print(f"[save] {per_user_path}  rows={len(merged)}  cols={len(merged.columns)}")

    # ---------- Aggregate RMSE ----------
    rmse_cols = [c for c in cv_df.columns if c.startswith("rmse_")]
    agg_mean = {c: float(cv_df[c].mean()) for c in rmse_cols}
    agg_median = {c: float(cv_df[c].median()) for c in rmse_cols}
    print(f"\n=== k=5 CV aggregate RMSE (per-user OLS, n_users={len(cv_df)}) ===")
    for c in rmse_cols:
        print(f"  {c:<32} mean={agg_mean[c]:.4f}  median={agg_median[c]:.4f}")

    # ---------- Paired Wilcoxon quadratic_ols vs linear_ols ----------
    w = stats.wilcoxon(cv_df["rmse_pilsd_quadratic_ols"].to_numpy(),
                       cv_df["rmse_pilsd_linear_ols"].to_numpy(),
                       alternative="two-sided")
    d_pairs = (cv_df["rmse_pilsd_quadratic_ols"] - cv_df["rmse_pilsd_linear_ols"]).to_numpy()
    quad_vs_lin_ols = {
        "mean_delta_quad_minus_lin": float(np.mean(d_pairs)),
        "median_delta":              float(np.median(d_pairs)),
        "frac_quad_smaller":         float((d_pairs < 0).mean()),
        "wilcoxon_stat":             float(w.statistic),
        "wilcoxon_p":                float(w.pvalue),
    }
    print(f"[paired] quadratic_ols vs linear_ols  mean Δ={quad_vs_lin_ols['mean_delta_quad_minus_lin']:+.4f}  "
          f"frac_quad<lin={quad_vs_lin_ols['frac_quad_smaller']:.3%}  p={quad_vs_lin_ols['wilcoxon_p']:.3e}")

    # ---------- Cluster bootstrap on mean(RMSE_lin - RMSE_quad) ----------
    rng_b = np.random.default_rng(args.seed + 1)
    n_u = len(cv_df)
    lin = cv_df["rmse_pilsd_linear_ols"].to_numpy()
    qd = cv_df["rmse_pilsd_quadratic_ols"].to_numpy()
    deltas = []
    for _ in range(args.n_boot):
        sub = rng_b.choice(n_u, size=n_u, replace=True)
        deltas.append(float(lin[sub].mean() - qd[sub].mean()))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    boot_summary = {
        "point_estimate": float(lin.mean() - qd.mean()),
        "bootstrap_mean": float(np.mean(deltas)),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "includes_zero": bool(lo <= 0 <= hi),
        "n_boot": int(args.n_boot),
    }
    print(f"[boot] mean(RMSE_lin) - mean(RMSE_quad) = {boot_summary['point_estimate']:+.4f}  "
          f"CI95=[{lo:+.4f}, {hi:+.4f}]  CI_incl_0={boot_summary['includes_zero']}")

    # ---------- Save summary.json ----------
    summary = {
        "n_users": int(n_users),
        "n_utterances": int(len(df)),
        "rm_feature_mean": rm_mean,
        "rm_feature_std": rm_std,
        "rm_z_sq_mean": sq_mean,
        "pop_coefficients": {
            "linear":    {"intercept": pop_a_lin, "slope": pop_b_lin},
            "quadratic": {"intercept": pop_c_qd, "linear": pop_b_qd, "quadratic": pop_a_qd},
        },
        "gamma_significance": {
            "bonferroni_threshold": float(bonf_threshold),
            "counts": sig_counts,
            "fractions": sig_fracs,
        },
        "gamma_distribution": alpha_stats,
        "aggregate_rmse_mean":   agg_mean,
        "aggregate_rmse_median": agg_median,
        "quadratic_vs_linear_ols_paired": quad_vs_lin_ols,
        "bootstrap_linear_minus_quadratic_rmse": boot_summary,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[save] {summary_path}")

    # ---------- Key verdict banner ----------
    key_frac = sig_fracs["alpha_0p05"]
    thresh = 0.30
    print("\n" + "=" * 72)
    print(f"FALSIFIABILITY VERDICT (γ_j signif at α=0.05)")
    print(f"  fraction significant = {key_frac:.3%}  (threshold for violation = {thresh:.1%})")
    if key_frac >= thresh:
        print(f"  ✓ >= {thresh:.0%} of users show γ_j ≠ 0 ⇒ affine-calibrator premise is")
        print(f"    violated for a substantial subset of PRISM annotators.")
        print(f"    Prop T1.MI's scope claim is empirically challenged for this subset.")
    else:
        print(f"  ✗ < {thresh:.0%} of users show γ_j ≠ 0 ⇒ affine-calibrator premise holds")
        print(f"    for the majority of PRISM annotators. Prop T1.MI scope empirically intact.")
    print("=" * 72)


if __name__ == "__main__":
    main()
