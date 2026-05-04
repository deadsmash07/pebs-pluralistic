"""Track 3 robustness: detector FPR under TIME-VARYING residual noise σ_ε(t).

The Simpson's-stress and mixed-drift paper capstones assume σ_ε is
TIME-CONSTANT. A savvy reviewer will ask:

    "What if reviewer noise ALSO drifts over time? E.g., early-stage
    reviewers are careful, late-stage reviewers are noisier due to
    fatigue. Does your 0% FPR under composition shift survive that?"

This script answers. DGP: Simpson composition shift (low-θ authors enter
later) + σ_ε(t) = σ₀ · (1 + amp·sin(2π·t/T)) with β_true = 0.

Hypothesis:
  - MixedLM + within-author permutation null STAYS CALIBRATED because
    the permutation destroys any time-signal within each author's
    contribution window — the permutation distribution still centers
    at zero even if residuals are scaled heteroskedastically.
  - NaiveOLS on monthly aggregates already fires (Simpson stress).
    Heteroskedasticity either leaves FPR at 100% or pushes it higher.

References:
  - Pinheiro & Bates 2000 §4 heteroskedastic mixed-effects models
  - Good 2006 §11.4 within-author permutation robustness
  - Our capstone: `memory/track3_simpson_stress_validated.md`
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Reuse detectors from the committed simpson_paradox_stress_test.py
sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location(
    "simpson_stress", Path(__file__).parent / "simpson_paradox_stress_test.py")
sim = importlib.util.module_from_spec(spec)
sys.modules["simpson_stress"] = sim
spec.loader.exec_module(sim)


def simulate_heteroskedastic_cohort(
    amp: float = 0.5,
    span_months: float = 10.0,
    authors_per_month_start: int = 15,
    authors_per_month_end: int = 35,
    msgs_per_author_mean: int = 40,
    author_active_months: float = 3.0,
    theta_high: float = 0.75,
    theta_low: float = 0.55,
    sigma_0: float = 0.18,
    seed: int = 0,
) -> pd.DataFrame:
    """Simpson composition DGP + time-varying σ_ε(t) = σ₀·(1+amp·sin(2π·t/T)).

    True per-author drift is ZERO. Only the residual noise amplitude
    varies across time. If MixedLM's permutation null stays calibrated,
    FPR should be ≤ α=0.05 (Wilson CI tolerance).
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
            sigma_t = sigma_0 * (1.0 + amp * np.sin(2 * np.pi * t_msg / span_months))
            q = theta_j + rng.normal(0, sigma_t, size=n_msgs)
            q = np.clip(q, 0.0, 1.0)
            for t, qq in zip(t_msg, q):
                rows.append({"user_id": f"u{aid}", "month_num": float(t), "quality": float(qq)})
            aid += 1
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amps", default="0.0,0.5,1.0")
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--n-perms", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--output-path",
                    default="results/track3_heteroskedastic/heteroskedastic_robustness.json")
    ap.add_argument("--base-seed", type=int, default=7777)
    args = ap.parse_args()

    amps = [float(x) for x in args.amps.split(",")]
    results = {"config": vars(args), "cells": []}
    t0 = time.time()

    print(f"=== Heteroskedastic-σ_ε robustness (β_true=0, amp∈{amps}) ===")
    print(f"  n_seeds={args.n_seeds}  n_perms={args.n_perms}  α={args.alpha}")
    for amp in amps:
        cell = {"amp": amp, "rejections_ml": 0, "rejections_naive": 0,
                "ml_ps": [], "naive_ps": []}
        for s in range(args.n_seeds):
            df = simulate_heteroskedastic_cohort(amp=amp, seed=args.base_seed + s)
            p_ml = sim.detector_mixedlm_perm(df, n_perms=args.n_perms, seed=args.base_seed + s)
            p_naive = sim.detector_naive_ols(df)
            cell["ml_ps"].append(p_ml)
            cell["naive_ps"].append(p_naive)
            if not np.isnan(p_ml) and p_ml < args.alpha:
                cell["rejections_ml"] += 1
            if not np.isnan(p_naive) and p_naive < args.alpha:
                cell["rejections_naive"] += 1
        cell["fpr_ml"] = cell["rejections_ml"] / args.n_seeds
        cell["fpr_naive"] = cell["rejections_naive"] / args.n_seeds
        cell["mean_ml_p"] = float(np.nanmean(cell["ml_ps"]))
        cell["mean_naive_p"] = float(np.nanmean(cell["naive_ps"]))
        results["cells"].append(cell)
        elapsed = time.time() - t0
        # Wilson 95% CI for FPR
        from math import sqrt
        k, n = cell["rejections_ml"], args.n_seeds
        z = 1.96
        if n > 0:
            p = k / n
            denom = 1 + z**2/n
            center = (p + z**2/(2*n))/denom
            hw = z*sqrt(p*(1-p)/n + z**2/(4*n**2))/denom
            ci_lo, ci_hi = max(0, center-hw), min(1, center+hw)
        else:
            ci_lo, ci_hi = 0, 1
        print(f"  [amp={amp:.2f}]  FPR_ml={cell['fpr_ml']:.2%} "
              f"[CI {ci_lo:.2%}, {ci_hi:.2%}]  FPR_naive={cell['fpr_naive']:.2%}  "
              f"elapsed={elapsed:.0f}s")

    # Verdict
    max_ml_fpr = max(c["fpr_ml"] for c in results["cells"])
    max_naive_fpr = max(c["fpr_naive"] for c in results["cells"])
    results["verdict"] = {
        "max_ml_fpr": max_ml_fpr,
        "max_naive_fpr": max_naive_fpr,
        "ml_robust_to_heteroskedasticity": max_ml_fpr <= 0.15,
        "naive_vulnerable_to_composition": max_naive_fpr >= 0.80,
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).write_text(json.dumps(results, indent=2))
    print(f"\n[save] {args.output_path}")

    print(f"\n=== Verdict ===")
    robust_tag = "✓ ROBUST to heteroskedasticity" if max_ml_fpr <= 0.15 else "✗ FRAGILE"
    print(f"  MixedLM+perm max FPR across amp grid: {max_ml_fpr:.2%}  {robust_tag}")
    vuln_tag = "✗ VULNERABLE to composition (expected)" if max_naive_fpr >= 0.80 else "△ unexpected"
    print(f"  NaiveOLS max FPR across amp grid:     {max_naive_fpr:.2%}  {vuln_tag}")


if __name__ == "__main__":
    main()
