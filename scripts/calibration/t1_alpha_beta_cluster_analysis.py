"""
T1 alpha-beta cluster analysis.

Tests whether the fitted per-user (alpha_j, beta_j) calibrators from PILSD on
PRISM come from a single Gaussian (validates hierarchical Gaussian prior), a
mixture (suggests mixture prior future work), or a continuous but
non-Gaussian distribution (heavy tails / skew).

Outputs:
  - marginal Gaussianity tests (Shapiro-Wilk, Anderson-Darling, Jarque-Bera)
  - joint Gaussianity (Mardia skew + kurt, Henze-Zirkler)
  - GMM fit K in {1..5}, best by BIC (also reports AIC)
  - per-cluster centroids + sizes + demographic interpretation
  - KDE 2D grid written to CSV for plotting
  - text summary printed to stdout

CPU-only. <5 min. Uses sklearn.mixture.GaussianMixture + scipy.stats.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture


def gaussianity_marginal(x: np.ndarray, name: str) -> dict:
    """Run Shapiro-Wilk (up to 5000), Anderson-Darling, Jarque-Bera, D'Agostino K^2."""
    out: dict = {"variable": name, "n": int(len(x))}

    # Shapiro-Wilk (subsample if >5000 because scipy caps)
    if len(x) <= 5000:
        sw = stats.shapiro(x)
        out["shapiro_W"] = float(sw.statistic)
        out["shapiro_p"] = float(sw.pvalue)
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(x), 5000, replace=False)
        sw = stats.shapiro(x[idx])
        out["shapiro_W"] = float(sw.statistic)
        out["shapiro_p"] = float(sw.pvalue)
        out["shapiro_subsampled"] = True

    # Anderson-Darling -- compare to critical at 1%
    ad = stats.anderson(x, dist="norm")
    out["ad_statistic"] = float(ad.statistic)
    out["ad_critical_1pct"] = float(ad.critical_values[-1])
    out["ad_reject_1pct"] = bool(ad.statistic > ad.critical_values[-1])

    # Jarque-Bera
    jb = stats.jarque_bera(x)
    out["jb_statistic"] = float(jb.statistic)
    out["jb_p"] = float(jb.pvalue)

    # D'Agostino
    k2 = stats.normaltest(x)
    out["dagostino_K2"] = float(k2.statistic)
    out["dagostino_p"] = float(k2.pvalue)

    # Descriptive
    out["mean"] = float(np.mean(x))
    out["std"] = float(np.std(x, ddof=1))
    out["skew"] = float(stats.skew(x))
    out["kurtosis_excess"] = float(stats.kurtosis(x))
    out["min"] = float(np.min(x))
    out["max"] = float(np.max(x))
    return out


def mardia_multinormality(X: np.ndarray) -> dict:
    """Mardia's test for multivariate normality.

    b_1 = (1/n^2) sum_{i,j} [ (x_i - mu)' S^{-1} (x_j - mu) ]^3   (skewness)
    b_2 = (1/n) sum_i    [ (x_i - mu)' S^{-1} (x_i - mu) ]^2      (kurtosis)
    Under H0 multinormal:
        A = n*b_1/6  ~  chi2( p(p+1)(p+2)/6 )
        B = (b_2 - p(p+2)) / sqrt(8 p(p+2)/n)  ~  N(0, 1)
    """
    n, p = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu
    S = np.cov(Xc, rowvar=False, bias=False)
    S_inv = np.linalg.inv(S)
    D = Xc @ S_inv @ Xc.T  # n x n quadratic form
    b1 = (D ** 3).sum() / (n ** 2)
    b2 = np.diag(D) ** 2
    b2 = b2.mean()
    A = n * b1 / 6
    df_skew = p * (p + 1) * (p + 2) / 6
    p_skew = 1 - stats.chi2.cdf(A, df_skew)
    B = (b2 - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
    p_kurt = 2 * (1 - stats.norm.cdf(abs(B)))
    return {
        "mardia_skew_stat": float(A),
        "mardia_skew_df": float(df_skew),
        "mardia_skew_p": float(p_skew),
        "mardia_kurt_z": float(B),
        "mardia_kurt_p": float(p_kurt),
        "b1": float(b1),
        "b2": float(b2),
    }


def gmm_selection(X: np.ndarray, max_k: int = 5, n_init: int = 10, seed: int = 0) -> dict:
    """Fit GMM K in {1..max_k} with full covariance; return BIC/AIC table + best."""
    rows = []
    fits = {}
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=n_init,
            max_iter=500,
            random_state=seed,
            reg_covar=1e-6,
        )
        gmm.fit(X)
        bic = float(gmm.bic(X))
        aic = float(gmm.aic(X))
        rows.append({"K": k, "bic": bic, "aic": aic, "converged": bool(gmm.converged_)})
        fits[k] = gmm
    table = pd.DataFrame(rows)
    k_best_bic = int(table.loc[table.bic.idxmin(), "K"])
    k_best_aic = int(table.loc[table.aic.idxmin(), "K"])
    return {
        "table": table,
        "k_best_bic": k_best_bic,
        "k_best_aic": k_best_aic,
        "fits": fits,
    }


def describe_clusters(X: np.ndarray, gmm: GaussianMixture, dim_labels=("alpha_j", "beta_j")) -> pd.DataFrame:
    labels = gmm.predict(X)
    rows = []
    for k in range(gmm.n_components):
        mask = labels == k
        cov = gmm.covariances_[k]
        rows.append({
            "cluster": k,
            "n": int(mask.sum()),
            "weight": float(gmm.weights_[k]),
            f"{dim_labels[0]}_mean": float(gmm.means_[k, 0]),
            f"{dim_labels[1]}_mean": float(gmm.means_[k, 1]),
            f"{dim_labels[0]}_sd": float(np.sqrt(cov[0, 0])),
            f"{dim_labels[1]}_sd": float(np.sqrt(cov[1, 1])),
            "corr": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
            f"{dim_labels[0]}_emp_mean": float(X[mask, 0].mean()) if mask.any() else float("nan"),
            f"{dim_labels[1]}_emp_mean": float(X[mask, 1].mean()) if mask.any() else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def main(args):
    calibs_path = Path(args.calibrators)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng_seed = args.seed

    df = pd.read_parquet(calibs_path)
    print(f"[load] {calibs_path}  shape={df.shape}  cols={list(df.columns)}")

    # Select columns
    alpha = df["alpha_j"].to_numpy(dtype=float)
    beta = df["beta_j"].to_numpy(dtype=float)

    # Drop NaNs if any
    keep = np.isfinite(alpha) & np.isfinite(beta)
    if keep.sum() < len(df):
        print(f"[warn] dropping {(~keep).sum()} non-finite rows")
    alpha = alpha[keep]
    beta = beta[keep]
    X = np.column_stack([alpha, beta])
    print(f"[data] n={len(X)} alpha mean={alpha.mean():.3f} sd={alpha.std(ddof=1):.3f}"
          f"  beta mean={beta.mean():.3f} sd={beta.std(ddof=1):.3f}"
          f"  corr={np.corrcoef(alpha, beta)[0, 1]:.4f}")

    # Marginal Gaussianity
    print("\n[marginal gaussianity]")
    marg_alpha = gaussianity_marginal(alpha, "alpha_j")
    marg_beta = gaussianity_marginal(beta, "beta_j")
    for m in (marg_alpha, marg_beta):
        print(
            f"  {m['variable']:10s}  Shapiro p={m['shapiro_p']:.2e}"
            f"  JB p={m['jb_p']:.2e}  K2 p={m['dagostino_p']:.2e}"
            f"  AD reject@1%={m['ad_reject_1pct']}"
            f"  skew={m['skew']:+.3f} kurt_ex={m['kurtosis_excess']:+.3f}"
        )

    # Joint Gaussianity (Mardia)
    print("\n[joint gaussianity]")
    mardia = mardia_multinormality(X)
    print(f"  Mardia skew stat={mardia['mardia_skew_stat']:.2f}  df={mardia['mardia_skew_df']:.0f}  p={mardia['mardia_skew_p']:.2e}")
    print(f"  Mardia kurt z={mardia['mardia_kurt_z']:+.3f}  p={mardia['mardia_kurt_p']:.2e}")

    # GMM selection
    print("\n[GMM K in 1..5]")
    gmm_res = gmm_selection(X, max_k=5, n_init=10, seed=rng_seed)
    table = gmm_res["table"]
    print(table.to_string(index=False))
    k_best = gmm_res["k_best_bic"]
    print(f"\n[BIC pick] K_best = {k_best}  (AIC pick K={gmm_res['k_best_aic']})")

    # For K=1 BIC baseline (pure Gaussian assumption), compare ΔBIC
    bic_k1 = float(table.loc[table.K == 1, "bic"].iloc[0])
    dbic = table.assign(delta_bic=table.bic - bic_k1)
    print("\n[ΔBIC vs K=1]")
    print(dbic[["K", "bic", "delta_bic"]].to_string(index=False))
    print("  (Kass-Raftery: ΔBIC > 10 => very strong evidence against K=1; >6 => strong; >2 => positive)")

    # Cluster description -- describe both K_best and a forced K=2 comparison
    print("\n[clusters at K_best]")
    clust_best = describe_clusters(X, gmm_res["fits"][k_best])
    print(clust_best.to_string(index=False))

    if k_best != 2:
        print("\n[clusters at K=2 (forced, for comparison)]")
        clust_2 = describe_clusters(X, gmm_res["fits"][2])
        print(clust_2.to_string(index=False))

    # Robustness: compare with shrunk vs naive if naive columns available
    robustness = None
    if "alpha_naive_ols" in df.columns and "beta_naive_ols" in df.columns:
        a0 = df["alpha_naive_ols"].to_numpy(dtype=float)
        b0 = df["beta_naive_ols"].to_numpy(dtype=float)
        keep0 = np.isfinite(a0) & np.isfinite(b0)
        X0 = np.column_stack([a0[keep0], b0[keep0]])
        gmm_naive = gmm_selection(X0, max_k=5, n_init=10, seed=rng_seed)
        robustness = {
            "naive_table": gmm_naive["table"].to_dict(orient="records"),
            "naive_k_best": gmm_naive["k_best_bic"],
        }
        print("\n[robustness: naive OLS calibrators]")
        print(gmm_naive["table"].to_string(index=False))
        print(f"  K_best(naive) = {gmm_naive['k_best_bic']}")

    # Save all findings
    findings = {
        "calibrators_path": str(calibs_path),
        "n": int(len(X)),
        "marginal_alpha": marg_alpha,
        "marginal_beta": marg_beta,
        "mardia": mardia,
        "gmm_table": table.to_dict(orient="records"),
        "k_best_bic": k_best,
        "k_best_aic": gmm_res["k_best_aic"],
        "delta_bic_vs_K1": dbic[["K", "delta_bic"]].to_dict(orient="records"),
        "clusters_at_k_best": clust_best.to_dict(orient="records"),
        "robustness": robustness,
    }
    if k_best != 2:
        findings["clusters_at_K2_forced"] = clust_2.to_dict(orient="records")
    with open(out_dir / "t1_alpha_beta_cluster_findings.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)
    print(f"\n[saved] {out_dir/'t1_alpha_beta_cluster_findings.json'}")

    # KDE grid for optional plotting
    try:
        kde = stats.gaussian_kde(X.T)
        g_alpha = np.linspace(alpha.min() - 1, alpha.max() + 1, 60)
        g_beta = np.linspace(beta.min() - 1, beta.max() + 1, 60)
        GA, GB = np.meshgrid(g_alpha, g_beta)
        grid_points = np.vstack([GA.ravel(), GB.ravel()])
        densities = kde(grid_points).reshape(GA.shape)
        kde_df = pd.DataFrame(densities, index=g_beta, columns=g_alpha)
        kde_df.index.name = "beta_j"
        kde_df.columns.name = "alpha_j"
        kde_df.to_csv(out_dir / "t1_alpha_beta_kde_grid.csv")
        print(f"[saved] {out_dir/'t1_alpha_beta_kde_grid.csv'}")
    except Exception as e:
        print(f"[warn] KDE skipped: {e}")

    # Cluster labels per user (for downstream demographic cross-tab)
    labels_best = gmm_res["fits"][k_best].predict(X)
    user_ids = df.loc[keep, "user_id"].to_numpy()
    pd.DataFrame({"user_id": user_ids, "cluster_kbest": labels_best}).to_parquet(
        out_dir / "t1_alpha_beta_cluster_labels.parquet", index=False
    )
    print(f"[saved] {out_dir/'t1_alpha_beta_cluster_labels.parquet'}")

    # Return findings so callers can inspect
    return findings


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Cluster analysis of PILSD (alpha_j, beta_j)")
    p.add_argument(
        "--calibrators",
        default="data/prism_user_calibrators_shrunk.parquet",
        help="path to fitted calibrator parquet",
    )
    p.add_argument("--out-dir", default="results/t1_alpha_beta_clusters")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)
