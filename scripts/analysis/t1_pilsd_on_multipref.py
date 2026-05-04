"""Cross-dataset generalization test: does PILSD's 8.58% RMSE improvement
transfer from PRISM to AI2 MultiPref?

Design (direct PRISM → MultiPref analog)
----------------------------------------
PRISM setup:
  - target: per-user reward score (continuous, 0-1)
  - feature x: trained-7B-RM score on the same (prompt, response) (shared ref)
  - per-user calibrator:  y_ij = α_j + β_j · x_ij + ε_ij
  - empirical-Bayes shrink toward pop-slope with ω = τ² / (τ² + V)
  - 5-fold CV on held-out pairs within user → RMSE
  - Headline: 8.58% relative RMSE improvement over pop-slope baseline

MultiPref analog (each comparison has 4 evaluators → LOO crowd consensus):
  - target: evaluator's own pref_score ∈ {0, .25, .5, .75, 1}
      (mapped from {A-clear, A-slight, Tie, B-slight, B-clear})
  - feature x: leave-one-out mean of OTHER 3 evaluators' pref_score on the
      same comparison_id — a "crowd consensus" signal (shared ref, by design
      independent of this user's own rating)
  - per-user calibrator: same α_j + β_j · x_ij form
  - same EB shrinkage and 5-fold held-out CV

This is a cleaner cross-dataset test than PRISM's "user vs RM" because the
crowd-consensus reference is ground-truth-free (no trained model required),
yet still reveals per-user calibration heterogeneity.

Runs CPU-only; expected <2 min for 226 users × 5 folds.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats


PREF_TO_SCORE = {
    "A-is-clearly-better": 1.0,
    "A-is-slightly-better": 0.75,
    "Tie": 0.5,
    "B-is-slightly-better": 0.25,
    "B-is-clearly-better": 0.0,
}


def load_and_prepare(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["pref_score"] = df["overall_pref"].map(PREF_TO_SCORE)
    df = df.dropna(subset=["pref_score"]).reset_index(drop=True)

    # Leave-one-out crowd consensus per comparison
    g = df.groupby("comparison_id")["pref_score"]
    df["cmp_sum"] = g.transform("sum")
    df["cmp_n"] = g.transform("count")
    df = df[df["cmp_n"] >= 2].copy()  # need at least 2 annotators
    df["loo_consensus"] = (df["cmp_sum"] - df["pref_score"]) / (df["cmp_n"] - 1)
    df = df.drop(columns=["cmp_sum", "cmp_n"])
    return df


def per_user_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (alpha, beta, resid_var). Handles degenerate x (var=0) by falling
    back to alpha = mean(y), beta = 0."""
    xm, ym = x.mean(), y.mean()
    xvar = float(((x - xm) ** 2).sum())
    if xvar < 1e-12:
        return float(ym), 0.0, float(((y - ym) ** 2).mean())
    beta = float(((x - xm) * (y - ym)).sum() / xvar)
    alpha = float(ym - beta * xm)
    pred = alpha + beta * x
    resid_var = float(((y - pred) ** 2).mean())
    return alpha, beta, resid_var


def shrinkage_omega(per_user_betas: np.ndarray, per_user_resid_vars: np.ndarray,
                    n_per_user: np.ndarray, x_var_per_user: np.ndarray) -> float:
    """Estimate tau² (between-user variance of β) via method-of-moments:

        Var(β̂_j) = τ² + E[V_j]       where V_j = σ²_j / (n_j · var_j(x))

    ω = τ² / (τ² + V̄)  is the shrinkage weight (1 = trust user entirely,
    0 = collapse to population slope).
    """
    # Per-user sampling variance of β̂_j
    with np.errstate(divide="ignore", invalid="ignore"):
        V_j = per_user_resid_vars / np.clip(n_per_user * x_var_per_user, 1e-12, None)
    V_mean = float(np.nanmean(V_j))
    beta_var = float(np.nanvar(per_user_betas, ddof=1))
    tau2 = max(beta_var - V_mean, 1e-6)  # floor
    omega = tau2 / (tau2 + V_mean)
    return float(np.clip(omega, 0.0, 1.0))


def run_cv(df: pd.DataFrame, n_folds: int = 5, min_per_user: int = 5,
           seed: int = 42) -> dict:
    """5-fold within-user CV. For each fold:
       - Fit pop-slope (α_0, β_0) on training rows across all users
       - For each user ≥ min_per_user training rows: fit (α_j, β_j) via OLS
       - Compute EB shrinkage ω from training user-level stats
       - For each held-out row, evaluate three predictors:
           naive:    pop-slope only (α_0 + β_0 · x)
           raw_user: per-user OLS only (α_j + β_j · x)
           shrunk:   α_shrink + β_shrink · x
                     where (α_shrink, β_shrink) = ω*(α_j, β_j) + (1-ω)*(α_0, β_0)
    """
    rng = np.random.default_rng(seed)
    users = df["user_id"].unique()

    # Only consider users with enough data for meaningful CV folds
    users = [u for u in users if (df["user_id"] == u).sum() >= min_per_user * n_folds // (n_folds - 1)]

    # Preassign each row a fold 0..n_folds-1 within its user
    df = df.copy()
    df["fold"] = -1
    for uid in users:
        mask = df["user_id"] == uid
        idx = df.index[mask].to_numpy()
        rng.shuffle(idx)
        df.loc[idx, "fold"] = np.arange(len(idx)) % n_folds

    df = df[df["fold"] >= 0].reset_index(drop=True)
    print(f"[cv] n_users={len(users)}, n_rows={len(df)}")

    sq_err_naive, sq_err_raw, sq_err_shrunk = [], [], []
    fold_stats = []

    for fold in range(n_folds):
        train = df[df["fold"] != fold]
        test = df[df["fold"] == fold]

        # Population slope on training
        a0, b0, _ = per_user_ols(
            train["loo_consensus"].to_numpy(), train["pref_score"].to_numpy()
        )

        # Per-user fits on training
        user_fits: dict[str, tuple[float, float, float, int, float]] = {}
        for uid, g in train.groupby("user_id"):
            if len(g) < min_per_user:
                continue
            x = g["loo_consensus"].to_numpy()
            y = g["pref_score"].to_numpy()
            alpha, beta, rv = per_user_ols(x, y)
            xvar = float(np.var(x, ddof=0))
            user_fits[uid] = (alpha, beta, rv, len(g), xvar)

        # EB ω from training user stats
        betas = np.array([v[1] for v in user_fits.values()])
        rvs = np.array([v[2] for v in user_fits.values()])
        ns = np.array([v[3] for v in user_fits.values()])
        xvs = np.array([v[4] for v in user_fits.values()])
        omega = shrinkage_omega(betas, rvs, ns, xvs)
        fold_stats.append({
            "fold": fold,
            "pop_slope": b0,
            "pop_intercept": a0,
            "omega": omega,
            "n_users_with_fit": len(user_fits),
            "per_user_beta_mean": float(betas.mean()),
            "per_user_beta_std": float(betas.std(ddof=1)),
        })

        # Evaluate held-out
        for _, row in test.iterrows():
            uid = row["user_id"]
            x = row["loo_consensus"]
            y = row["pref_score"]
            # Naive
            y_naive = a0 + b0 * x
            # Raw per-user (fall back to pop if not fit)
            if uid in user_fits:
                au, bu = user_fits[uid][0], user_fits[uid][1]
                y_raw = au + bu * x
                # Shrunk
                a_sh = omega * au + (1 - omega) * a0
                b_sh = omega * bu + (1 - omega) * b0
                y_sh = a_sh + b_sh * x
            else:
                y_raw = y_naive
                y_sh = y_naive
            sq_err_naive.append((y - y_naive) ** 2)
            sq_err_raw.append((y - y_raw) ** 2)
            sq_err_shrunk.append((y - y_sh) ** 2)

    sq_err_naive = np.asarray(sq_err_naive)
    sq_err_raw = np.asarray(sq_err_raw)
    sq_err_shrunk = np.asarray(sq_err_shrunk)

    rmse_naive = float(np.sqrt(sq_err_naive.mean()))
    rmse_raw = float(np.sqrt(sq_err_raw.mean()))
    rmse_shrunk = float(np.sqrt(sq_err_shrunk.mean()))

    # Paired Wilcoxon of (naive - shrunk) squared errors
    w_stat, w_p = sstats.wilcoxon(
        sq_err_naive, sq_err_shrunk, alternative="greater", zero_method="zsplit"
    )

    # Cluster bootstrap (by user) on rel-improvement: (rmse_naive - rmse_shrunk)/rmse_naive
    rng_b = np.random.default_rng(seed + 1)
    uids = df["user_id"].to_numpy()
    err_by_uid_naive: dict[str, list[float]] = {}
    err_by_uid_shrunk: dict[str, list[float]] = {}
    for i, uid in enumerate(uids):
        err_by_uid_naive.setdefault(uid, []).append(sq_err_naive[i])
        err_by_uid_shrunk.setdefault(uid, []).append(sq_err_shrunk[i])
    uid_keys = list(err_by_uid_naive.keys())

    boot_rel = []
    for _ in range(2000):
        pick = rng_b.choice(uid_keys, size=len(uid_keys), replace=True)
        sn, ss = [], []
        for k in pick:
            sn.extend(err_by_uid_naive[k])
            ss.extend(err_by_uid_shrunk[k])
        rn = np.sqrt(np.mean(sn))
        rs = np.sqrt(np.mean(ss))
        boot_rel.append((rn - rs) / max(rn, 1e-12) * 100.0)
    boot_rel = np.asarray(boot_rel)

    return {
        "n_users": int(df["user_id"].nunique()),
        "n_observations": int(len(df)),
        "rmse_naive_popslope": rmse_naive,
        "rmse_raw_peruser": rmse_raw,
        "rmse_shrunk_pilsd": rmse_shrunk,
        "abs_improvement": rmse_naive - rmse_shrunk,
        "rel_improvement_pct": (rmse_naive - rmse_shrunk) / rmse_naive * 100.0,
        "rel_improvement_ci95": [
            float(np.percentile(boot_rel, 2.5)),
            float(np.percentile(boot_rel, 97.5)),
        ],
        "rel_improvement_boot_mean": float(boot_rel.mean()),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p_onesided": float(w_p),
        "fold_stats": fold_stats,
        "mean_omega": float(np.mean([f["omega"] for f in fold_stats])),
        "mean_per_user_beta_std": float(np.mean([f["per_user_beta_std"] for f in fold_stats])),
    }


def heterogeneity_anova(df: pd.DataFrame) -> dict:
    """One-way ANOVA on pref_score by user_id — is there heterogeneity?"""
    groups = [g["pref_score"].to_numpy() for _, g in df.groupby("user_id") if len(g) >= 5]
    f_stat, p_val = sstats.f_oneway(*groups)
    # Also: how much variance is between-user?
    overall_mean = df["pref_score"].mean()
    total_ss = float(((df["pref_score"] - overall_mean) ** 2).sum())
    between_ss = 0.0
    for uid, g in df.groupby("user_id"):
        if len(g) >= 5:
            between_ss += len(g) * (g["pref_score"].mean() - overall_mean) ** 2
    eta_sq = between_ss / total_ss if total_ss > 0 else float("nan")
    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_val),
        "eta_squared": float(eta_sq),
        "n_groups": len(groups),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-parquet", default="data/multipref_evaluator_quality.parquet")
    ap.add_argument("--min-per-user", type=int, default=5)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/t1_pilsd_on_multipref.json")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.cohort_parquet}")
    df = load_and_prepare(Path(args.cohort_parquet))
    print(f"[load] {len(df)} rows, {df['user_id'].nunique()} evaluators")
    print(f"[load] pref_score mean={df['pref_score'].mean():.3f} std={df['pref_score'].std():.3f}")
    print(f"[load] loo_consensus mean={df['loo_consensus'].mean():.3f} std={df['loo_consensus'].std():.3f}")

    # Heterogeneity characterization (analog to PRISM's per-user variance test)
    print("\n[het] one-way ANOVA on pref_score by user_id …")
    het = heterogeneity_anova(df)
    print(f"[het] F={het['f_statistic']:.2f}, p={het['p_value']:.4g}, η²={het['eta_squared']:.4f}, n_groups={het['n_groups']}")

    # Per-user correlation of own-pref vs LOO-consensus (is the signal there?)
    corrs = []
    for uid, g in df.groupby("user_id"):
        if len(g) >= 10 and g["loo_consensus"].std() > 0.01:
            r, _ = sstats.pearsonr(g["loo_consensus"], g["pref_score"])
            corrs.append(r)
    print(f"[corr] Per-user r(pref vs LOO): n={len(corrs)}, median={np.median(corrs):.3f}, IQR=[{np.percentile(corrs,25):.3f}, {np.percentile(corrs,75):.3f}]")

    # PILSD CV
    print(f"\n[cv] running {args.n_folds}-fold within-user CV …")
    t0 = time.time()
    res = run_cv(df, n_folds=args.n_folds, min_per_user=args.min_per_user, seed=args.seed)
    print(f"[cv] done in {time.time()-t0:.1f}s")

    print(f"\n=== PILSD cross-dataset result (MultiPref) ===")
    print(f"n_users              = {res['n_users']}")
    print(f"n_observations       = {res['n_observations']}")
    print(f"mean EB omega        = {res['mean_omega']:.3f}  (1.0 = no shrinkage, 0.0 = pop-slope)")
    print(f"per-user β std       = {res['mean_per_user_beta_std']:.3f}")
    print(f"")
    print(f"RMSE naive (pop)     = {res['rmse_naive_popslope']:.5f}")
    print(f"RMSE raw (per-user)  = {res['rmse_raw_peruser']:.5f}")
    print(f"RMSE PILSD (shrunk)  = {res['rmse_shrunk_pilsd']:.5f}")
    print(f"Δ abs                = {res['abs_improvement']:+.5f}")
    print(f"Δ rel                = {res['rel_improvement_pct']:+.3f} %  "
          f"CI95 [{res['rel_improvement_ci95'][0]:+.2f}, {res['rel_improvement_ci95'][1]:+.2f}]")
    print(f"Wilcoxon one-sided p = {res['wilcoxon_p_onesided']:.4g}")

    # Cross-dataset verdict
    pct = res["rel_improvement_pct"]
    ci_lo = res["rel_improvement_ci95"][0]
    prism_target = 8.58
    if ci_lo > 0:
        verdict = "POSITIVE (PILSD generalizes)"
    elif pct > 0:
        verdict = "DIRECTIONAL (positive but CI straddles zero)"
    else:
        verdict = "NULL (no cross-dataset generalization)"

    print(f"\n[verdict] {verdict}")
    print(f"[verdict] PRISM headline was {prism_target}% improvement; MultiPref is {pct:+.2f}% "
          f"(ratio {pct/prism_target:+.2f}×).")

    out = {
        "dataset": "allenai/multipref",
        "cohort_parquet": str(args.cohort_parquet),
        "config": {"min_per_user": args.min_per_user, "n_folds": args.n_folds, "seed": args.seed},
        "heterogeneity": het,
        "per_user_signal_corr": {
            "n_users": len(corrs),
            "median": float(np.median(corrs)),
            "q25": float(np.percentile(corrs, 25)),
            "q75": float(np.percentile(corrs, 75)),
        },
        "pilsd": res,
        "verdict": verdict,
        "prism_reference_pct": prism_target,
        "multipref_vs_prism_ratio": pct / prism_target,
    }
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
