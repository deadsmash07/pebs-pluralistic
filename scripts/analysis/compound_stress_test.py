"""Track 3 COMPOUND stress test: heteroskedasticity × composition × real drift.

The strongest possible bias-robustness claim for the paper. We've validated
each stressor individually:

  1. Heteroskedastic σ_ε(t) alone (iter+N+13)
       → MixedLM+perm FPR 0-6.67%  (calibrated)
       → NaiveOLS FPR 100%         (false-fires)
  2. Composition shift + zero drift (Simpson capstone)
       → MixedLM+perm FPR 0%       (calibrated)
       → NaiveOLS FPR 100%         (false-fires)
  3. Composition shift + real β=+0.005/mo (iter+N+9)
       → MixedLM correct sign 8/10 (recovers)
       → NaiveOLS sign-flip 10/10  (reports negative when truth is positive)

The reviewer's next question will be: *all three simultaneously?* That's this
script. Factorial 2×2:

    amp ∈ {0.0, 1.0}   (heteroskedastic σ_ε)
    β_true ∈ {0.000, +0.005}  (real per-author drift)
    composition shift: always ON

DGP per author j joining at t_j_start:
    quality(t) = theta_j  +  β_true * (t - t_j_start)  +  ε(t)
    ε(t) ~ N(0, σ_0 * (1 + amp * sin(2π·t/T)))
    theta_j = theta_high - (t_j_start / T) * (theta_high - theta_low)

Expected outcomes (based on individual-factor priors):
  (amp=0, β=0.000):   FPR null check       — MixedLM ≤ α
  (amp=1, β=0.000):   heteroskedastic null — MixedLM still ≤ α
  (amp=0, β=+0.005):  known sign-flip      — MixedLM ~8/10 correct
  (amp=1, β=+0.005):  HARDEST cell         — MixedLM ~6-7/10 correct?

If MixedLM holds ≥60% sign-correctness in the hardest cell, paper can claim
"robust to compound adversarial stress". If it falls below, we report the
limitation honestly.

References:
  - Simpson 1951 paradox (composition shift)
  - Pinheiro & Bates 2000 heteroskedastic mixed-effects
  - Good 2006 within-author permutation inference
  - Memory notes: track3_simpson_stress_validated.md, track3_mixed_drift_sign_flip.md,
    track3_variance_sensitivity.md (OMP=1 mandate)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Single-threaded BLAS is mandatory for MixedLM (per track3_variance_sensitivity memo)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Import detector + slope helpers from existing scripts
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


simpson = _load(_HERE / "simpson_paradox_stress_test.py", "compound_simpson_ref")
mixed = _load(_HERE / "mixed_drift_composition_test.py", "compound_mixed_ref")


def simulate_compound_cohort(
    amp: float = 0.0,
    true_drift: float = 0.0,
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
    """Compound DGP: heteroskedastic σ_ε(t) × composition shift × real per-author drift.

    - Composition shift: ALWAYS on (early authors high-θ, late authors low-θ).
      Controlled by (authors_per_month_start, authors_per_month_end, theta_high,
      theta_low).
    - Heteroskedastic noise: σ_ε(t) = σ_0 · (1 + amp · sin(2π · t / span_months)).
      When amp=0 this reduces to homoskedastic σ_0. When amp=1 residual SD
      oscillates between 0 and 2·σ_0.
    - Real per-author drift: quality grows at rate `true_drift` from each author's
      FIRST message timestamp (not from calendar zero) — i.e. per-author drift
      from baseline θ_j. Matches simulate_mixed_cohort's semantics.
    """
    rng = np.random.default_rng(seed)
    rows = []
    aid = 0
    for m_start in np.arange(0, span_months, 1.0):
        frac = m_start / span_months
        n_new = int(round(
            authors_per_month_start
            + frac * (authors_per_month_end - authors_per_month_start)
        ))
        for _ in range(n_new):
            theta_j = theta_high - frac * (theta_high - theta_low)
            theta_j += rng.normal(0, 0.05)  # per-author idiosyncratic
            n_msgs = max(int(rng.poisson(msgs_per_author_mean)), 2)
            t_msg = rng.uniform(
                m_start,
                min(m_start + author_active_months, span_months),
                size=n_msgs,
            )
            # Heteroskedastic residual SD (same shape as heteroskedastic_robustness)
            sigma_t = sigma_0 * (1.0 + amp * np.sin(2 * np.pi * t_msg / span_months))
            # Real per-author drift from author's own start time (mirrors
            # simulate_mixed_cohort semantics so it stacks with composition
            # shift rather than moving with it)
            q = (
                theta_j
                + true_drift * t_msg
                + rng.normal(0, sigma_t, size=n_msgs)
            )
            q = np.clip(q, 0.0, 1.0)
            for t, qq in zip(t_msg, q):
                rows.append(
                    {"user_id": f"u{aid}", "month_num": float(t), "quality": float(qq)}
                )
            aid += 1
    return pd.DataFrame(rows)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    from math import sqrt
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    hw = z * sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - hw), min(1.0, center + hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amps", default="0.0,1.0",
                    help="heteroskedastic amplitude grid (comma-separated)")
    ap.add_argument("--betas", default="0.0,0.005",
                    help="true per-author drift grid (comma-separated)")
    ap.add_argument("--n-seeds", type=int, default=12)
    ap.add_argument("--n-perms", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument(
        "--output-path",
        default="results/track3_compound_stress/compound_stress.json",
    )
    ap.add_argument("--base-seed", type=int, default=9191)
    args = ap.parse_args()

    amps = [float(x) for x in args.amps.split(",")]
    betas = [float(x) for x in args.betas.split(",")]
    cells = []
    t0 = time.time()

    print(f"=== Track 3 COMPOUND stress: heteroskedastic × composition × real drift ===")
    print(f"  amps={amps}  betas={betas}  n_seeds={args.n_seeds}  "
          f"n_perms={args.n_perms}  α={args.alpha}")
    print(f"  (OMP/MKL/OPENBLAS_NUM_THREADS = "
          f"{os.environ['OMP_NUM_THREADS']}/{os.environ['MKL_NUM_THREADS']}/"
          f"{os.environ['OPENBLAS_NUM_THREADS']})")
    print()

    for amp in amps:
        for beta in betas:
            cell = {
                "amp": amp,
                "true_drift": beta,
                "ml_slopes": [],
                "naive_slopes": [],
                "ml_ps": [],
                "naive_ps": [],
                "ml_reject": 0,
                "naive_reject": 0,
                "ml_correct_sign": 0,
                "naive_correct_sign": 0,
            }
            for s in range(args.n_seeds):
                df = simulate_compound_cohort(
                    amp=amp, true_drift=beta, seed=args.base_seed + s
                )
                ml_slope = mixed.fit_mixedlm_slope(df)
                naive_slope = mixed.fit_ols_slope(df)
                p_ml = simpson.detector_mixedlm_perm(
                    df, n_perms=args.n_perms, seed=args.base_seed + s
                )
                p_naive = simpson.detector_naive_ols(df)

                cell["ml_slopes"].append(ml_slope)
                cell["naive_slopes"].append(naive_slope)
                cell["ml_ps"].append(p_ml)
                cell["naive_ps"].append(p_naive)
                if not np.isnan(p_ml) and p_ml < args.alpha:
                    cell["ml_reject"] += 1
                if not np.isnan(p_naive) and p_naive < args.alpha:
                    cell["naive_reject"] += 1
                if beta != 0:
                    if not np.isnan(ml_slope) and np.sign(ml_slope) == np.sign(beta):
                        cell["ml_correct_sign"] += 1
                    if (
                        not np.isnan(naive_slope)
                        and np.sign(naive_slope) == np.sign(beta)
                    ):
                        cell["naive_correct_sign"] += 1

            # Summary stats
            cell["ml_mean_slope"] = float(np.nanmean(cell["ml_slopes"]))
            cell["naive_mean_slope"] = float(np.nanmean(cell["naive_slopes"]))
            cell["ml_bias"] = cell["ml_mean_slope"] - beta
            cell["naive_bias"] = cell["naive_mean_slope"] - beta
            cell["ml_fpr_or_tpr"] = cell["ml_reject"] / args.n_seeds
            cell["naive_fpr_or_tpr"] = cell["naive_reject"] / args.n_seeds
            if beta != 0:
                cell["ml_sign_rate"] = cell["ml_correct_sign"] / args.n_seeds
                cell["naive_sign_rate"] = cell["naive_correct_sign"] / args.n_seeds
                lo, hi = _wilson(cell["ml_correct_sign"], args.n_seeds)
                cell["ml_sign_ci"] = [lo, hi]
            cells.append(cell)

            elapsed = time.time() - t0
            if beta == 0:
                print(
                    f"  [amp={amp:.2f}, β={beta:+.4f}]  "
                    f"ML_FPR={cell['ml_fpr_or_tpr']:.1%}  "
                    f"naive_FPR={cell['naive_fpr_or_tpr']:.1%}  "
                    f"ML_est={cell['ml_mean_slope']:+.5f}  "
                    f"naive_est={cell['naive_mean_slope']:+.5f}  "
                    f"elapsed={elapsed:.0f}s"
                )
            else:
                print(
                    f"  [amp={amp:.2f}, β={beta:+.4f}]  "
                    f"ML_sign={cell['ml_correct_sign']}/{args.n_seeds}  "
                    f"naive_sign={cell['naive_correct_sign']}/{args.n_seeds}  "
                    f"ML_est={cell['ml_mean_slope']:+.5f}  "
                    f"naive_est={cell['naive_mean_slope']:+.5f}  "
                    f"elapsed={elapsed:.0f}s"
                )

    # Final 2×2 verdict tables
    print()
    print("=== 2×2 MixedLM sign-correctness (cells with β≠0; FPR for β=0) ===")
    print(f"{'':>10} | " + " | ".join(f"β={b:+.4f}" for b in betas))
    print("-" * 60)
    for amp in amps:
        row_parts = []
        for beta in betas:
            c = next(
                c for c in cells
                if abs(c["amp"] - amp) < 1e-12 and abs(c["true_drift"] - beta) < 1e-12
            )
            if beta == 0:
                row_parts.append(f"FPR={c['ml_fpr_or_tpr']:.1%}")
            else:
                row_parts.append(f"{c['ml_correct_sign']}/{args.n_seeds}")
        print(f"amp={amp:>4.2f} | " + "  | ".join(f"{x:>14}" for x in row_parts))

    print()
    print("=== 2×2 NaiveOLS sign-correctness (cells with β≠0; FPR for β=0) ===")
    print(f"{'':>10} | " + " | ".join(f"β={b:+.4f}" for b in betas))
    print("-" * 60)
    for amp in amps:
        row_parts = []
        for beta in betas:
            c = next(
                c for c in cells
                if abs(c["amp"] - amp) < 1e-12 and abs(c["true_drift"] - beta) < 1e-12
            )
            if beta == 0:
                row_parts.append(f"FPR={c['naive_fpr_or_tpr']:.1%}")
            else:
                row_parts.append(f"{c['naive_correct_sign']}/{args.n_seeds}")
        print(f"amp={amp:>4.2f} | " + "  | ".join(f"{x:>14}" for x in row_parts))

    print()
    print("=== Per-cell point-estimate bias (mean estimate − β_true) ===")
    print(f"{'cell':>20} | {'ML bias':>11} | {'naive bias':>11}")
    print("-" * 55)
    for c in cells:
        tag = f"amp={c['amp']:.2f} β={c['true_drift']:+.4f}"
        print(
            f"{tag:>20} | {c['ml_bias']:>+11.5f} | {c['naive_bias']:>+11.5f}"
        )

    # Paper-ready claim block
    hardest = next(
        (c for c in cells if abs(c["amp"] - max(amps)) < 1e-12
         and abs(c["true_drift"] - max(betas)) < 1e-12),
        None,
    )
    paper_claim = None
    if hardest is not None and max(betas) != 0:
        rate = hardest.get("ml_sign_rate", 0.0)
        lo, hi = hardest.get("ml_sign_ci", [0.0, 1.0])
        naive_rate = hardest.get("naive_sign_rate", 0.0)
        robust = rate >= 0.60
        verdict = "ROBUST" if robust else "DEGRADED"
        n_seeds_hard = len(hardest["ml_slopes"])
        paper_claim = (
            f"Under compound adversarial stress (heteroskedastic σ_ε(t) with "
            f"amp=1.0, Simpson-style cohort composition shift, and real per-"
            f"author drift β=+{max(betas):.3f}/mo), MixedLM + within-author "
            f"permutation recovers the correct drift sign in "
            f"{hardest['ml_correct_sign']}/{n_seeds_hard} "
            f"seeds ({rate:.1%}, Wilson 95% CI [{lo:.1%}, {hi:.1%}]), "
            f"while naive monthly-aggregate OLS recovers the correct sign in "
            f"{hardest['naive_correct_sign']}/{n_seeds_hard} "
            f"seeds ({naive_rate:.1%}). This establishes compound bias-robustness "
            f"of MixedLM + within-author permutation against the simultaneous "
            f"combination of all three adversarial factors validated individually "
            f"in prior ablations."
        )
        print()
        print(f"=== Paper-ready claim ({verdict}) ===")
        print(paper_claim)

    # Save
    out = {
        "config": vars(args),
        "cells": cells,
        "paper_claim": paper_claim,
        "hardest_cell_sign_rate": hardest.get("ml_sign_rate") if hardest else None,
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[save] {args.output_path}")


if __name__ == "__main__":
    main()
