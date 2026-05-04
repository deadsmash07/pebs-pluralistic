"""Run MixedLM + within-evaluator permutation drift detector on MultiPref.

This is the direct analog of `scripts/test_reviewer_drift_mixedlm.py` for the
AI2 MultiPref dataset (`allenai/multipref`). MultiPref exposes per-annotator
IDs and per-annotation UTC timestamps, so the same identification strategy
applies — except the corpus spans ~29 days (May 2024), so we bucket by DAY
and estimate `β = d(quality)/d(day)` with evaluator random effects.

Model
-----
    quality_ij  =  β₀ + β₁·day_num_i + β₂·is_expert_i + u_j + ε_ij
    u_j ~ N(0, σ²_u)        (per-evaluator random intercept)
    ε_ij ~ N(0, σ²_ε)

H₀ : β₁ = 0  (no monotone intra-month drift in review-quality proxy)
H₁ : β₁ ≠ 0  (calibration shift as the annotation wave progresses)

Quality proxy
-------------
`mean_conf` — mean of the 4 per-aspect confidence Likert answers
    (absolutely=1.0, fairly=0.66, not=0.33). Higher ⇒ more careful review.
A `--quality-col time_spent` switch lets you re-run on effort-seconds as a
second proxy (values are in seconds — no need to log-transform; MixedLM is
robust to scale).

References
----------
  - Miranda et al. 2024 arXiv:2410.19133 "Hybrid Preferences …" (dataset)
  - Pinheiro & Bates 2000 Mixed-Effects Models in S and S-PLUS (MixedLM)
  - Good 2006 Permutation, Parametric, and Bootstrap Tests §11.4
    (within-group permutation respects cluster random-effects under H₀)

Run
---
    python3 scripts/run_drift_on_multipref.py \
        --cohort-parquet data/multipref_evaluator_quality.parquet \
        --cohort-filter   data/multipref_evaluator_cohort.parquet \
        --n-permutations  500 \
        --output-dir      results/track3_multipref
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def fit_mixedlm(
    df: pd.DataFrame,
    quality_col: str = "quality",
    time_col: str = "day_num",
    exog_extra: str = "is_expert",
    shuffle_within_author: bool = False,
    seed: int | None = None,
):
    """Fit `quality ~ time + exog + (1|user_id)`; return (time_coef, pval, ll, converged).

    Permutation: shuffle `time_col` within each user's rows (preserves per-user
    volume + marginal day distribution — Good 2006 §11.4).
    """
    data = df.copy()
    if shuffle_within_author:
        rng = np.random.default_rng(seed)

        def _shuffle(s):
            arr = s.to_numpy(copy=True)
            rng.shuffle(arr)
            return pd.Series(arr, index=s.index)

        data[time_col] = data.groupby("user_id", sort=False)[time_col].transform(_shuffle)

    try:
        formula = f"{quality_col} ~ {time_col}"
        if exog_extra and exog_extra in data.columns:
            formula += f" + {exog_extra}"
        md = smf.mixedlm(formula, data=data, groups=data["user_id"])
        res = md.fit(method="lbfgs", maxiter=200, disp=False)
        coef = float(res.params[time_col])
        pval = float(res.pvalues[time_col])
        return coef, pval, float(res.llf), bool(res.converged), res
    except Exception as e:
        print(f"[fit] failure: {e}")
        return float("nan"), float("nan"), float("nan"), False, None


def run_naive_ols(df: pd.DataFrame, quality_col: str, time_col: str):
    """Naive daily-mean OLS: aggregate to day → regress day_mean on day_num.

    Per `memory/track3_mixed_drift_sign_flip.md`, this is the known-biased
    comparator that inverts sign under composition shift. Reported for the
    bias-robustness table only.
    """
    daily = (
        df.groupby(df[time_col].astype(int))
        .agg(q=(quality_col, "mean"), n=(quality_col, "size"))
        .reset_index(names=[time_col])
    )
    x = daily[time_col].to_numpy(float)
    y = daily["q"].to_numpy(float)
    if len(x) < 3:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se2 = resid @ resid / max(len(x) - 2, 1)
    var = se2 * np.linalg.inv(X.T @ X)[1, 1]
    tstat = beta[1] / np.sqrt(max(var, 1e-30))
    from scipy.stats import t as tdist
    p = 2 * (1 - tdist.cdf(abs(tstat), df=max(len(x) - 2, 1)))
    return float(beta[1]), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-parquet", default="data/multipref_evaluator_quality.parquet")
    ap.add_argument("--cohort-filter", default="data/multipref_evaluator_cohort.parquet")
    ap.add_argument("--quality-col", default="quality",
                    help="Column to regress: default 'quality' (=mean_conf); "
                         "try 'time_spent' or 'overall_conf' as alternates.")
    ap.add_argument("--time-col", default="day_num")
    ap.add_argument("--exog-extra", default="is_expert")
    ap.add_argument("--n-permutations", type=int, default=500)
    ap.add_argument("--output-dir", default="results/track3_multipref")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full-cohort", action="store_true",
                    help="Use ALL evaluators (not just power cohort).")
    ap.add_argument("--also-run-naive-ols", action="store_true", default=True,
                    help="Run naive daily-mean OLS for bias-robustness comparison.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.cohort_parquet)

    cohort_users = set(pd.read_parquet(args.cohort_filter).index)
    if not args.full_cohort:
        df = df[df["user_id"].isin(cohort_users)].reset_index(drop=True)
        print(f"[data] restricted to power cohort: "
              f"{df['user_id'].nunique()} evaluators, {len(df)} rows")
    else:
        print(f"[data] FULL: {df['user_id'].nunique()} evaluators, {len(df)} rows")

    # Data hygiene
    df = df.dropna(subset=[args.quality_col, args.time_col]).reset_index(drop=True)
    print(f"[feat] {args.time_col} range: "
          f"{df[args.time_col].min():.2f} → {df[args.time_col].max():.2f}")
    print(f"[feat] {args.quality_col} mean={df[args.quality_col].mean():.4f} "
          f"std={df[args.quality_col].std():.4f}")
    if args.exog_extra in df.columns:
        print(f"[feat] {args.exog_extra} mean={df[args.exog_extra].mean():.3f}")

    # Primary MixedLM fit
    print("\n[fit] primary MixedLM (cluster-robust via author random intercept)")
    t = time.time()
    coef_obs, p_obs, ll_obs, conv_obs, res = fit_mixedlm(
        df, args.quality_col, args.time_col, args.exog_extra,
    )
    day_range = float(df[args.time_col].max() - df[args.time_col].min())
    print(f"[fit] {args.time_col} coef = {coef_obs:+.6f} per day  "
          f"(Wald p = {p_obs:.4g}, converged={conv_obs}, ll={ll_obs:.1f})")
    print(f"[fit] total-window drift estimate: {day_range * coef_obs:+.5f} "
          f"over {day_range:.1f} days")
    print(f"[fit] elapsed {time.time()-t:.1f}s")

    # Naive OLS comparator (bias-robustness table)
    naive_coef, naive_p = float("nan"), float("nan")
    if args.also_run_naive_ols:
        naive_coef, naive_p = run_naive_ols(df, args.quality_col, args.time_col)
        print(f"\n[naive] daily-mean OLS slope = {naive_coef:+.6f}/day  (p={naive_p:.4g})")
        sign_flip = (np.sign(naive_coef) != np.sign(coef_obs)) and abs(coef_obs) > 1e-8
        if sign_flip:
            print("[naive] SIGN FLIP vs MixedLM — matches memory/track3_mixed_drift_sign_flip pattern")

    # Permutation null
    print(f"\n[perm] running {args.n_permutations} within-evaluator permutations")
    t = time.time()
    null_coefs: list[float] = []
    for i in range(args.n_permutations):
        coef_null, _, _, _, _ = fit_mixedlm(
            df, args.quality_col, args.time_col, args.exog_extra,
            shuffle_within_author=True, seed=args.seed + i,
        )
        if not np.isnan(coef_null):
            null_coefs.append(coef_null)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t
            print(f"[perm] {i+1}/{args.n_permutations}  "
                  f"rate={(i+1)/elapsed:.1f}/s  "
                  f"null|coef| mean={np.mean(np.abs(null_coefs)):.6f}")
    null_arr = np.asarray(null_coefs)
    perm_p = float((np.abs(null_arr) >= abs(coef_obs)).mean()) if len(null_arr) else float("nan")
    print(f"[perm] null |coef| mean={np.abs(null_arr).mean():.6f} "
          f"std={null_arr.std():.6f}")
    print(f"[perm] permutation p-value (two-sided) = {perm_p:.4f}")

    report = {
        "dataset": "allenai/multipref",
        "quality_col": args.quality_col,
        "time_col": args.time_col,
        "exog_extra": args.exog_extra,
        "n_authors": int(df["user_id"].nunique()),
        "n_observations": int(len(df)),
        "day_range": [float(df[args.time_col].min()), float(df[args.time_col].max())],
        "observed": {
            "coef_per_day": coef_obs,
            "wald_p": p_obs,
            "window_drift": day_range * coef_obs,
            "converged": conv_obs,
            "loglik": ll_obs,
        },
        "permutation_null": {
            "n_permutations": int(len(null_arr)),
            "null_abs_coef_mean": float(np.abs(null_arr).mean()) if len(null_arr) else None,
            "null_coef_std": float(null_arr.std()) if len(null_arr) else None,
            "two_sided_p_value": perm_p,
        },
        "naive_ols_comparator": {
            "coef_per_day": naive_coef,
            "wald_p": naive_p,
            "sign_matches_mixedlm":
                bool(np.sign(naive_coef) == np.sign(coef_obs))
                if not (np.isnan(naive_coef) or np.isnan(coef_obs))
                else None,
        },
        "verdict": (
            "H3a_REJECTED_null" if perm_p < 0.05 else "H3a_NOT_REJECTED"
        ),
    }
    out_path = out_dir / "reviewer_drift_mixedlm.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[save] {out_path}")
    print(f"[verdict] {report['verdict']}  "
          f"(perm p={perm_p:.4f}, Wald p={p_obs:.4g})")


if __name__ == "__main__":
    main()
