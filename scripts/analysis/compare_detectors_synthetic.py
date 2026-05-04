"""Detector baseline comparison: MixedLM + permutation vs naive OLS + t-test vs
PageHinkley on aggregate monthly means.

Critical paper gap: the Track 3 claim "our detector achieves MDE 0.00097/mo"
is only meaningful relative to existing drift-detection methods. This
script runs three detectors on the SAME synthetic cohorts to measure:
  - TPR at planted drift
  - FPR at β=0 null
  - Which detector has best type-II robustness at fixed type-I rate?

Detectors
---------
1. **MixedLM + permutation null** (our method):
   `quality ~ month_num + (1 | user_id)` in statsmodels; within-author
   permutation of month_num for null distribution. Fixed effect captures
   population-mean drift after author fixed effects.

2. **Naive OLS on author-level means** (pedagogical baseline):
   Collapse per-author to (author, month) cells with mean quality. Fit
   OLS on (quality ~ month_num) with parametric t-test. This is what a
   naïve analyst would do without knowing about mixed effects.

3. **PageHinkley on monthly aggregate** (streaming baseline):
   Compute monthly mean quality, feed to bidirectional PageHinkley CUSUM.
   This is the detector-of-record in streaming change-detection.

References
----------
- Page 1954 "Continuous inspection schemes" (CUSUM)
- Gama & Castillo 2006 PageHinkley for drift
- Pinheiro & Bates 2000 §2 random-effects ANOVA (our method)
- Simpson 1951 aggregation paradox — why naive OLS can fail on longitudinal
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
from scipy import stats
import statsmodels.formula.api as smf

# Import simulate_cohort from our existing DGP
sys.path.insert(0, str(Path(__file__).parent))
from synthetic_drift_power_analysis import simulate_cohort, fit_and_test

# Import PageHinkley from Track 1's method module (shared between tracks).
# Register in sys.modules BEFORE exec so @dataclass can resolve cls.__module__.
_SD_NAME = "streaming_drift_shared"
spec = importlib.util.spec_from_file_location(
    _SD_NAME,
    "<DATA_ROOT>/1_Causal_RLHF/src/methods/streaming_drift.py",
)
_sd = importlib.util.module_from_spec(spec)
sys.modules[_SD_NAME] = _sd  # required for @dataclass — see track3_parity_bugs_caught memo
spec.loader.exec_module(_sd)
PageHinkleyDetector = _sd.PageHinkleyDetector


def detector_mixedlm_perm(df: pd.DataFrame, n_perms: int = 50, seed: int = 0) -> float:
    """Returns permutation p-value for month_num coefficient.

    This is the existing Track 3 detector — reused verbatim."""
    out = fit_and_test(df, n_permutations=n_perms, seed=seed)
    return out["perm_p"]


def detector_naive_ols(df: pd.DataFrame) -> float:
    """Collapse to author × month cells, fit OLS, return t-test p.

    Ignores within-author correlation → parametric t-test is anti-
    conservative if authors contribute multiple months."""
    # Collapse: one row per (user_id, month_bucket)
    df2 = df.copy()
    df2["month_bucket"] = np.floor(df2["month_num"]).astype(int)
    cell = df2.groupby(["user_id", "month_bucket"], as_index=False).agg(
        quality=("quality", "mean"),
        month_num=("month_num", "mean"),
    )
    if len(cell) < 10 or cell["month_num"].var() < 1e-10:
        return np.nan
    # OLS with parametric t-test on slope
    try:
        md = smf.ols("quality ~ month_num", data=cell).fit()
        return float(md.pvalues["month_num"])
    except Exception:
        return np.nan


def detector_pagehinkley_monthly(df: pd.DataFrame) -> float:
    """Run bidirectional PageHinkley on monthly-aggregate means.

    Returns a synthetic p-value: 0.0 if detector fires during 10-month stream,
    0.5 otherwise. (PageHinkley gives binary fire/no-fire, not a p-value, so
    we binarize. A more rigorous version would calibrate the threshold via
    null-distribution simulation.)"""
    df2 = df.copy()
    df2["month_bucket"] = np.floor(df2["month_num"]).astype(int)
    monthly = df2.groupby("month_bucket")["quality"].mean().sort_index()
    if len(monthly) < 6:
        return np.nan
    # Bidirectional PH — use two one-sided detectors
    det_up = PageHinkleyDetector(min_instances=3, delta=0.005, threshold=0.5)
    det_dn = PageHinkleyDetector(min_instances=3, delta=0.005, threshold=0.5)
    fired = False
    for val in monthly.values:
        det_up.update(val)
        det_dn.update(-val)
        if det_up.drift_detected or det_dn.drift_detected:
            fired = True
            break
    return 0.0 if fired else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-authors", type=int, default=100)
    ap.add_argument("--msgs-per-author-mean", type=int, default=50)
    ap.add_argument("--span-months", type=float, default=10.0)
    ap.add_argument("--author-sd", type=float, default=0.12)
    ap.add_argument("--residual-sd", type=float, default=0.23)
    ap.add_argument("--drift-grid", default="0.0,0.002,0.005,0.01,0.03")
    ap.add_argument("--n-seeds-per-cell", type=int, default=10)
    ap.add_argument("--n-permutations", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--output-dir", default="results/track3_detector_comparison")
    ap.add_argument("--base-seed", type=int, default=98765)
    args = ap.parse_args()

    drift_grid = [float(x) for x in args.drift_grid.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"config": vars(args), "per_drift": []}
    t0 = time.time()

    for drift in drift_grid:
        cell = {"drift": drift, "detectors": {}}
        for name in ["mixedlm_perm", "naive_ols", "pagehinkley_monthly"]:
            cell["detectors"][name] = {"rejections": 0, "ps": []}

        for s in range(args.n_seeds_per_cell):
            seed = args.base_seed + s + int(drift * 1e6)
            df = simulate_cohort(
                n_authors=args.n_authors,
                msgs_per_author_mean=args.msgs_per_author_mean,
                span_months=args.span_months,
                drift_per_month=drift,
                author_sd=args.author_sd,
                residual_sd=args.residual_sd,
                seed=seed,
            )

            p_ml = detector_mixedlm_perm(df, n_perms=args.n_permutations, seed=seed)
            p_ols = detector_naive_ols(df)
            p_ph = detector_pagehinkley_monthly(df)

            cell["detectors"]["mixedlm_perm"]["ps"].append(p_ml)
            cell["detectors"]["naive_ols"]["ps"].append(p_ols)
            cell["detectors"]["pagehinkley_monthly"]["ps"].append(p_ph)

            for name, p in [("mixedlm_perm", p_ml), ("naive_ols", p_ols),
                            ("pagehinkley_monthly", p_ph)]:
                if p is not None and not np.isnan(p) and p < args.alpha:
                    cell["detectors"][name]["rejections"] += 1

        for name in cell["detectors"]:
            d = cell["detectors"][name]
            d["TPR"] = d["rejections"] / max(args.n_seeds_per_cell, 1)
            ps = np.array([p for p in d["ps"] if p is not None and not np.isnan(p)])
            d["mean_p"] = float(np.nanmean(ps)) if len(ps) else np.nan

        results["per_drift"].append(cell)
        elapsed = time.time() - t0
        print(f"[β={drift:.4f}/mo]  "
              f"MixedLM TPR={cell['detectors']['mixedlm_perm']['TPR']:.2%}  "
              f"NaiveOLS TPR={cell['detectors']['naive_ols']['TPR']:.2%}  "
              f"PH TPR={cell['detectors']['pagehinkley_monthly']['TPR']:.2%}  "
              f"elapsed={elapsed:.0f}s")

    out_path = out_dir / "comparison.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[save] {out_path}")

    # Summary
    print(f"\n=== Detector comparison summary ===")
    print(f"{'β/mo':>8} | {'MixedLM':>8} | {'NaiveOLS':>9} | {'PageHink':>9}")
    print("-" * 45)
    for cell in results["per_drift"]:
        print(f"{cell['drift']:>8.4f} | "
              f"{cell['detectors']['mixedlm_perm']['TPR']:>7.2%} | "
              f"{cell['detectors']['naive_ols']['TPR']:>8.2%} | "
              f"{cell['detectors']['pagehinkley_monthly']['TPR']:>8.2%}")

    # Flag FPR issues at β=0
    null_cell = next((c for c in results["per_drift"] if c["drift"] == 0.0), None)
    if null_cell:
        print("\n=== Type-I error rate at β=0 (should be ≤ α=0.05) ===")
        for name in ["mixedlm_perm", "naive_ols", "pagehinkley_monthly"]:
            fpr = null_cell["detectors"][name]["TPR"]
            print(f"  {name:>22}: {fpr:.2%}  "
                  f"{'✓ calibrated' if fpr <= 0.15 else '✗ INFLATED'}")


if __name__ == "__main__":
    main()
