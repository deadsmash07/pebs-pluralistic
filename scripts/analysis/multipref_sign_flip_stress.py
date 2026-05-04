"""Simpson's-paradox / sign-flip stress test on MultiPref.

Analog of `scripts/mixed_drift_composition_test.py` for MultiPref. Tests
whether MixedLM retains correct sign under adversarial evaluator-composition
shift, while naive daily-mean OLS inverts.

Procedure
---------
For each seed:
  1. Take the real MultiPref cohort (148 evaluators, ~34k annotations).
  2. Resample evaluators WITH REPLACEMENT, but bias sampling toward
     LATE-JOINING evaluators in the second half of the time window
     (`p_late = 0.7` vs natural `0.5`). This synthesizes composition
     drift without altering any individual evaluator's true trajectory.
  3. OPTIONALLY plant a true per-evaluator drift of β_true quality_units/day.
  4. Fit MixedLM + naive OLS. Record sign agreement with β_true.

References
----------
  - memory/track3_mixed_drift_sign_flip.md  (the OASST2 analog)
  - Pinheiro & Bates 2000; Simpson 1951

Run
---
    python3 scripts/multipref_sign_flip_stress.py \
        --quality-col quality --n-seeds 10 --beta-true 0.002
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")  # convergence / singular REs are expected in noisy subsamples

import statsmodels.formula.api as smf


def fit_mixedlm_quiet(data, quality_col, time_col, exog_extra):
    try:
        formula = f"{quality_col} ~ {time_col}"
        if exog_extra in data.columns:
            formula += f" + {exog_extra}"
        md = smf.mixedlm(formula, data=data, groups=data["user_id"])
        res = md.fit(method="lbfgs", maxiter=200, disp=False)
        return float(res.params[time_col])
    except Exception:
        return float("nan")


def fit_naive(data, quality_col, time_col):
    daily = (
        data.groupby(data[time_col].astype(int))
        .agg(q=(quality_col, "mean"))
        .reset_index(names=[time_col])
    )
    x = daily[time_col].to_numpy(float)
    y = daily["q"].to_numpy(float)
    if len(x) < 3:
        return float("nan")
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])


def resample_with_composition_shift(
    df: pd.DataFrame,
    rng: np.random.Generator,
    p_late: float = 0.7,
    time_col: str = "day_num",
):
    """Resample evaluators with late-biased selection to create composition drift."""
    median_day = df[time_col].median()
    ev_mean_day = df.groupby("user_id")[time_col].mean()
    late_ids = ev_mean_day[ev_mean_day >= median_day].index.to_numpy()
    early_ids = ev_mean_day[ev_mean_day < median_day].index.to_numpy()
    n_ev = df["user_id"].nunique()
    n_late = int(round(p_late * n_ev))
    n_early = n_ev - n_late
    # sample WITH replacement from each pool
    chosen = np.concatenate([
        rng.choice(late_ids, size=n_late, replace=True),
        rng.choice(early_ids, size=n_early, replace=True),
    ])
    # gather rows; note: duplicate evaluators produce multiple user_id groups
    # we need to give each sampled evaluator a unique user_id (re-label) so
    # MixedLM sees independent clusters
    parts = []
    for i, ev in enumerate(chosen):
        sub = df[df["user_id"] == ev].copy()
        sub["user_id"] = f"{ev}__dup{i}"
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def plant_drift(df: pd.DataFrame, beta_true: float,
                quality_col: str = "quality", time_col: str = "day_num"):
    out = df.copy()
    out[quality_col] = out[quality_col] + beta_true * out[time_col]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-parquet", default="data/multipref_evaluator_quality.parquet")
    ap.add_argument("--cohort-filter", default="data/multipref_evaluator_cohort.parquet")
    ap.add_argument("--quality-col", default="quality")
    ap.add_argument("--time-col", default="day_num")
    ap.add_argument("--exog-extra", default="is_expert")
    ap.add_argument("--beta-true", type=float, default=0.002,
                    help="Planted per-evaluator drift (quality units / day)")
    ap.add_argument("--p-late", type=float, default=0.7)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--output-dir", default="results/track3_multipref")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.cohort_parquet)
    cohort_users = set(pd.read_parquet(args.cohort_filter).index)
    df = df[df["user_id"].isin(cohort_users)].reset_index(drop=True)
    df = df.dropna(subset=[args.quality_col, args.time_col]).reset_index(drop=True)
    print(f"[data] base cohort: {df['user_id'].nunique()} evaluators, {len(df)} rows")

    results = []
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        shifted = resample_with_composition_shift(df, rng, args.p_late, args.time_col)
        shifted = plant_drift(shifted, args.beta_true, args.quality_col, args.time_col)

        ml_coef = fit_mixedlm_quiet(shifted, args.quality_col, args.time_col, args.exog_extra)
        nv_coef = fit_naive(shifted, args.quality_col, args.time_col)

        ml_sign_ok = (np.sign(ml_coef) == np.sign(args.beta_true)) if args.beta_true != 0 else None
        nv_sign_ok = (np.sign(nv_coef) == np.sign(args.beta_true)) if args.beta_true != 0 else None

        results.append({
            "seed": seed,
            "mixedlm_coef": ml_coef,
            "naive_coef": nv_coef,
            "ml_sign_correct": bool(ml_sign_ok) if ml_sign_ok is not None else None,
            "nv_sign_correct": bool(nv_sign_ok) if nv_sign_ok is not None else None,
        })
        print(f"[seed={seed:02d}] ML={ml_coef:+.6f}  naive={nv_coef:+.6f}  "
              f"ML_ok={ml_sign_ok}  naive_ok={nv_sign_ok}")

    rdf = pd.DataFrame(results)
    n = len(rdf)
    ml_wins = rdf["ml_sign_correct"].sum()
    nv_wins = rdf["nv_sign_correct"].sum()
    print(f"\n=== Composition-shift + β_true={args.beta_true:+.4f}/day ===")
    print(f"MixedLM    correct sign: {ml_wins}/{n}")
    print(f"Naive OLS  correct sign: {nv_wins}/{n}")
    print(f"MixedLM mean β̂: {rdf['mixedlm_coef'].mean():+.6f}")
    print(f"Naive   mean β̂: {rdf['naive_coef'].mean():+.6f}")

    out = {
        "dataset": "allenai/multipref",
        "beta_true": args.beta_true,
        "p_late": args.p_late,
        "n_seeds": n,
        "mixedlm_sign_correct_rate": float(ml_wins) / n if n else None,
        "naive_sign_correct_rate": float(nv_wins) / n if n else None,
        "mixedlm_mean_coef": float(rdf["mixedlm_coef"].mean()),
        "naive_mean_coef": float(rdf["naive_coef"].mean()),
        "per_seed": results,
    }
    out_path = out_dir / f"sign_flip_stress_beta_{args.beta_true:+.4f}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
