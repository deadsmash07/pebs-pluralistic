"""Mixed-scenario stress test: real per-author drift + cohort composition shift.

The prior iteration (iter+N+8 Simpson's stress) showed that NaiveOLS and
PageHinkley both fire 100% of the time on pure composition shift. But the
harder question — the one a reviewer will ask — is:

    "Can MixedLM STILL detect a genuine per-author drift when it's
    MASKED by a confounding composition shift? And can the naive methods
    get the SIGN right, or do they report opposite-signed drift?"

This test adds a constant per-author drift β_true ∈ {+0.005, -0.005, 0}
on top of the composition-shift DGP. Expected:

  β_true = +0.005  (true positive drift)
    - MixedLM: detects positive drift with correct magnitude
    - NaiveOLS: likely reports NEGATIVE drift (composition dominates)
    - PageHinkley: fires but gives no sign information

  β_true = -0.005  (true negative drift, same direction as composition)
    - MixedLM: detects correct magnitude
    - NaiveOLS: fires with over-magnitude (composition + real stack)
    - PageHinkley: fires

  β_true = 0  (from prior iteration)
    - MixedLM: 0% FPR
    - NaiveOLS + PageHinkley: 100% FPR

References
----------
- Simpson 1951 paradox
- Pinheiro & Bates 2000 random-effects as composition control
- Our prior result: `track3_simpson_stress_validated.md`
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent))
from simpson_paradox_stress_test import (
    detector_mixedlm_perm,
    detector_naive_ols,
    detector_pagehinkley,
)


def simulate_mixed_cohort(
    span_months: float = 10.0,
    authors_per_month_start: int = 15,
    authors_per_month_end: int = 35,
    msgs_per_author_mean: int = 40,
    author_active_months: float = 3.0,
    theta_high: float = 0.75,
    theta_low: float = 0.55,
    true_drift: float = 0.005,
    residual_sd: float = 0.18,
    seed: int = 0,
) -> pd.DataFrame:
    """Composition shift + REAL per-author drift superimposed.

    Same cohort setup as simpson_paradox_stress_test but with
    `q = theta_j + true_drift * (t - t_j_start) + noise` — so each
    author drifts at the same rate from their own baseline.
    """
    rng = np.random.default_rng(seed)
    rows = []
    aid = 0
    for m_start in np.arange(0, span_months, 1.0):
        frac = m_start / span_months
        n_new = int(round(authors_per_month_start + frac * (authors_per_month_end - authors_per_month_start)))
        for _ in range(n_new):
            theta_j = theta_high - frac * (theta_high - theta_low)
            theta_j += rng.normal(0, 0.05)
            n_msgs = max(int(rng.poisson(msgs_per_author_mean)), 2)
            t_msg = rng.uniform(m_start, min(m_start + author_active_months, span_months), size=n_msgs)
            # True per-author drift baked in — quality grows linearly
            # with elapsed months since author joined (constant-rate drift
            # across all authors = population-level drift).
            q = theta_j + true_drift * t_msg + rng.normal(0, residual_sd, size=n_msgs)
            q = np.clip(q, 0.0, 1.0)
            for t, qq in zip(t_msg, q):
                rows.append({"user_id": f"u{aid}", "month_num": float(t), "quality": float(qq)})
            aid += 1
    return pd.DataFrame(rows)


def fit_ols_slope(df: pd.DataFrame) -> float | float:
    """Recover the slope estimate each detector implies (for sign comparison)."""
    # NaiveOLS on monthly means (what the naive detector sees)
    m = df.copy()
    m["month_bucket"] = np.floor(m.month_num).astype(int)
    monthly = m.groupby("month_bucket").quality.mean().sort_index()
    if len(monthly) < 3 or monthly.var() < 1e-10:
        return np.nan
    return float(np.polyfit(monthly.index.astype(float), monthly.values, 1)[0])


def fit_mixedlm_slope(df: pd.DataFrame) -> float:
    try:
        md = smf.mixedlm("quality ~ month_num", data=df, groups=df.user_id).fit(
            method="lbfgs", maxiter=200, disp=False)
        return float(md.params["month_num"])
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift-grid", default="-0.005,0.0,0.005")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-perms", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--output-path",
                    default="results/track3_mixed_drift/mixed_drift.json")
    ap.add_argument("--base-seed", type=int, default=222)
    args = ap.parse_args()

    drifts = [float(x) for x in args.drift_grid.split(",")]
    cells = []
    t0 = time.time()

    for true_drift in drifts:
        cell = {"true_drift": true_drift,
                "mixedlm_slope_estimates": [],
                "naive_monthly_slopes": [],
                "mixedlm_reject": 0, "naive_reject": 0, "ph_reject": 0,
                "mixedlm_correct_sign": 0, "naive_correct_sign": 0}
        for s in range(args.n_seeds):
            df = simulate_mixed_cohort(true_drift=true_drift,
                                        seed=args.base_seed + s)
            # Point estimates
            ml_slope = fit_mixedlm_slope(df)
            naive_slope = fit_ols_slope(df)
            cell["mixedlm_slope_estimates"].append(ml_slope)
            cell["naive_monthly_slopes"].append(naive_slope)
            # Detector firings
            p_ml = detector_mixedlm_perm(df, n_perms=args.n_perms, seed=args.base_seed + s)
            p_ols = detector_naive_ols(df)
            p_ph = detector_pagehinkley(df)
            if not np.isnan(p_ml) and p_ml < args.alpha: cell["mixedlm_reject"] += 1
            if not np.isnan(p_ols) and p_ols < args.alpha: cell["naive_reject"] += 1
            if not np.isnan(p_ph) and p_ph < args.alpha: cell["ph_reject"] += 1
            # Sign correctness (only defined for nonzero true drift)
            if true_drift != 0:
                if not np.isnan(ml_slope) and (np.sign(ml_slope) == np.sign(true_drift)):
                    cell["mixedlm_correct_sign"] += 1
                if not np.isnan(naive_slope) and (np.sign(naive_slope) == np.sign(true_drift)):
                    cell["naive_correct_sign"] += 1

        cell["mixedlm_mean_slope"] = float(np.nanmean(cell["mixedlm_slope_estimates"]))
        cell["naive_mean_slope"] = float(np.nanmean(cell["naive_monthly_slopes"]))
        cells.append(cell)
        elapsed = time.time() - t0
        print(f"[true β={true_drift:+.4f}]  "
              f"ML est={cell['mixedlm_mean_slope']:+.5f}  "
              f"naive est={cell['naive_mean_slope']:+.5f}  "
              f"ML reject={cell['mixedlm_reject']}/{args.n_seeds}  "
              f"naive reject={cell['naive_reject']}/{args.n_seeds}  "
              f"elapsed={elapsed:.0f}s")

    print(f"\n=== Mixed-scenario sign-correctness table ===")
    print(f"{'β_true':>8} | {'ML est':>10} | {'naive est':>10} | {'ML sign':>10} | {'naive sign':>12}")
    print("-" * 75)
    for c in cells:
        if c["true_drift"] == 0:
            ml_sign = "N/A (null)"; naive_sign = "N/A (null)"
        else:
            ml_sign = f"{c['mixedlm_correct_sign']}/{args.n_seeds}"
            naive_sign = f"{c['naive_correct_sign']}/{args.n_seeds}"
        print(f"{c['true_drift']:>+8.4f} | {c['mixedlm_mean_slope']:>+10.5f} | "
              f"{c['naive_mean_slope']:>+10.5f} | {ml_sign:>10} | {naive_sign:>12}")

    out = {"config": vars(args), "cells": cells}
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).write_text(json.dumps(out, indent=2))
    print(f"\n[save] {args.output_path}")


if __name__ == "__main__":
    main()
