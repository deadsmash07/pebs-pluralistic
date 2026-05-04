"""5-way detector comparison: MixedLM+perm vs NaiveOLS vs PageHinkley
vs CUSUM vs EWMA on the reviewer-cohort drift DGP.

Extends `compare_detectors_synthetic.py` (3-way) with two canonical
streaming-SQC detectors:
  - CUSUM (Page 1954, two-sided, k = σ_mo/2)
  - EWMA  (Roberts 1959, λ = 0.25, L tuned on stationary null)

Both are FPR-calibrated on the same stationary-null DGP used by the
rest of Track 3 before the drift-grid sweep. The comparison then
reports TPR for each detector at every β ∈ {0, 0.002, 0.005, 0.01, 0.03}
with 100 seeds per cell (configurable).

Paper context: §4.3 already has a 4-detector CUSUM table on three
Simpson-stress scenarios. This script does the complementary
stationary-null-plus-uniform-drift sweep so the paper can report both
axes: robustness (Simpson) and raw-power (here).

Streaming-aggregate convention
------------------------------
CUSUM, EWMA, and PageHinkley all consume **monthly-mean quality** — the
same stream the three existing baselines see. MixedLM+perm operates on
per-message data. This is the fair "detector-on-same-aggregate"
comparison — otherwise MixedLM's per-message visibility would be
confounded with its better statistical treatment.

Usage
-----
$ python scripts/detector_comparison_extended.py \\
    --n-seeds-per-cell 100 \\
    --n-permutations 30 \\
    --output-dir results/track3_detector_comparison_extended

References
----------
- Page 1954 Biometrika, "Continuous Inspection Schemes" (CUSUM)
- Roberts 1959 Technometrics, "Control Chart Tests Based on
  Geometric Moving Averages" (EWMA)
- Gama & Castillo 2006 (PageHinkley for drift)
- Pinheiro & Bates 2000 (random-effects ANOVA; our method)
- Montgomery 2013 §9 (modern two-sided CUSUM / EWMA textbook)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Pin BLAS to single-thread BEFORE numpy / statsmodels import in parent
# (and re-pin inside workers) so parallel seeds don't oversubscribe.
for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_k] = "1"

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))


def _load(path: Path, name: str):
    """Load a module by file path, registering it in sys.modules first
    so @dataclass can resolve cls.__module__ (see track3_parity_bugs_caught
    memo for why this matters)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Load primitives from sibling scripts to avoid re-implementation drift.
# (Imported in both parent and each worker process.)
# ---------------------------------------------------------------------------
def _import_shared():
    _cusum = _load(
        _ROOT / "src" / "detectors" / "cusum.py", "cusum_for_ext_cmp"
    )
    _ewma = _load(
        _ROOT / "src" / "detectors" / "ewma.py", "ewma_for_ext_cmp"
    )
    _sd_power = _load(
        _HERE / "synthetic_drift_power_analysis.py",
        "sd_power_for_ext_cmp",
    )

    spec = importlib.util.spec_from_file_location(
        "streaming_drift_for_ext_cmp",
        "<DATA_ROOT>/"
        "1_Causal_RLHF/src/methods/streaming_drift.py",
    )
    _ph = importlib.util.module_from_spec(spec)
    sys.modules["streaming_drift_for_ext_cmp"] = _ph
    spec.loader.exec_module(_ph)

    return {
        "CUSUMDetector": _cusum.CUSUMDetector,
        "run_cusum_stream": _cusum.run_cusum_stream,
        "EWMADetector": _ewma.EWMADetector,
        "run_ewma_stream": _ewma.run_ewma_stream,
        "simulate_cohort": _sd_power.simulate_cohort,
        "fit_and_test": _sd_power.fit_and_test,
        "PageHinkleyDetector": _ph.PageHinkleyDetector,
    }


_SHARED = _import_shared()
CUSUMDetector = _SHARED["CUSUMDetector"]
run_cusum_stream = _SHARED["run_cusum_stream"]
EWMADetector = _SHARED["EWMADetector"]
run_ewma_stream = _SHARED["run_ewma_stream"]
simulate_cohort = _SHARED["simulate_cohort"]
fit_and_test = _SHARED["fit_and_test"]
PageHinkleyDetector = _SHARED["PageHinkleyDetector"]


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def monthly_means(df: pd.DataFrame) -> np.ndarray:
    m = df.copy()
    m["month_bucket"] = np.floor(m["month_num"]).astype(int)
    return (
        m.groupby("month_bucket")["quality"].mean().sort_index().to_numpy()
    )


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    hw = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - hw), min(1.0, center + hw)


# ---------------------------------------------------------------------------
# Detector wrappers — each returns a p-value in [0, 1] (with 0.0 meaning
# "fire" and 0.5 meaning "did not fire" for binary streaming detectors),
# matching the convention used by compare_detectors_synthetic.py.
# ---------------------------------------------------------------------------
def detector_mixedlm_perm(
    df: pd.DataFrame, n_perms: int = 30, seed: int = 0
) -> float:
    out = fit_and_test(df, n_permutations=n_perms, seed=seed)
    return out["perm_p"]


def detector_naive_ols(df: pd.DataFrame) -> float:
    df2 = df.copy()
    df2["month_bucket"] = np.floor(df2["month_num"]).astype(int)
    cell = df2.groupby(["user_id", "month_bucket"], as_index=False).agg(
        quality=("quality", "mean"),
        month_num=("month_num", "mean"),
    )
    if len(cell) < 10 or cell["month_num"].var() < 1e-10:
        return np.nan
    try:
        md = smf.ols("quality ~ month_num", data=cell).fit()
        return float(md.pvalues["month_num"])
    except Exception:
        return np.nan


def detector_pagehinkley(
    df: pd.DataFrame,
    min_instances: int = 3,
    delta: float = 0.005,
    threshold: float = 0.5,
) -> float:
    monthly = monthly_means(df)
    if len(monthly) < 6:
        return np.nan
    det_up = PageHinkleyDetector(
        min_instances=min_instances, delta=delta, threshold=threshold
    )
    det_dn = PageHinkleyDetector(
        min_instances=min_instances, delta=delta, threshold=threshold
    )
    fired = False
    for val in monthly:
        det_up.update(val)
        det_dn.update(-val)
        if det_up.drift_detected or det_dn.drift_detected:
            fired = True
            break
    return 0.0 if fired else 0.5


def detector_cusum(
    df: pd.DataFrame, k: float, h: float, min_instances: int = 3
) -> tuple[float, str]:
    monthly = monthly_means(df)
    if len(monthly) < min_instances + 2:
        return np.nan, ""
    out = run_cusum_stream(
        monthly, k=k, h=h, min_instances=min_instances
    )
    return (0.0 if out["fired"] else 0.5), out["direction"]


def detector_ewma(
    df: pd.DataFrame, lam: float, L: float, min_instances: int = 3
) -> tuple[float, str]:
    monthly = monthly_means(df)
    if len(monthly) < min_instances + 2:
        return np.nan, ""
    out = run_ewma_stream(
        monthly, lam=lam, L=L, min_instances=min_instances
    )
    return (0.0 if out["fired"] else 0.5), out["direction"]


# ---------------------------------------------------------------------------
# Parallel worker: run ALL 5 detectors on ONE synthetic cohort seed.
# ---------------------------------------------------------------------------
def _seed_task(args_tuple):
    """Run one seed for one β cell with all 5 detectors. Returns dict.

    The MixedLM+perm detector dominates the runtime (~0.5s); the other
    four are near-instant. Rather than parallelise at the permutation
    level, we parallelise across (seed × β) cells — 500 units across 16
    cores → ~30 units per worker.
    """
    (
        drift,
        seed,
        base_seed,
        n_authors,
        msgs_per_author_mean,
        span_months,
        author_sd,
        residual_sd,
        n_permutations,
        cusum_k,
        cusum_h,
        ewma_lam,
        ewma_L,
    ) = args_tuple

    # Re-pin BLAS in worker (idempotent — parent already set env, but
    # threadpoolctl-managed pools may have been pre-seeded at fork time).
    for _k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[_k] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    warnings.filterwarnings("ignore")

    df = simulate_cohort(
        n_authors=n_authors,
        msgs_per_author_mean=msgs_per_author_mean,
        span_months=span_months,
        drift_per_month=drift,
        author_sd=author_sd,
        residual_sd=residual_sd,
        seed=base_seed + seed + int(drift * 1e6),
    )

    p_ml = detector_mixedlm_perm(
        df, n_perms=n_permutations, seed=base_seed + seed
    )
    p_ols = detector_naive_ols(df)
    p_ph = detector_pagehinkley(df)
    p_cu, dir_cu = detector_cusum(df, k=cusum_k, h=cusum_h)
    p_ew, dir_ew = detector_ewma(df, lam=ewma_lam, L=ewma_L)

    return {
        "drift": drift,
        "seed": seed,
        "p_mixedlm_perm": p_ml,
        "p_naive_ols": p_ols,
        "p_pagehinkley": p_ph,
        "p_cusum": p_cu,
        "dir_cusum": dir_cu,
        "p_ewma": p_ew,
        "dir_ewma": dir_ew,
    }


# ---------------------------------------------------------------------------
# FPR calibration (CUSUM + EWMA) on stationary null
# ---------------------------------------------------------------------------
def _probe_sigma_monthly(
    n_probe: int,
    base_seed: int,
    author_sd: float,
    residual_sd: float,
    span_months: float,
    n_authors: int,
    msgs_per_author_mean: int,
) -> float:
    probe_vals = []
    for s in range(n_probe):
        df = simulate_cohort(
            n_authors=n_authors,
            msgs_per_author_mean=msgs_per_author_mean,
            span_months=span_months,
            drift_per_month=0.0,
            author_sd=author_sd,
            residual_sd=residual_sd,
            seed=base_seed + 9999 + s,
        )
        probe_vals.append(monthly_means(df))
    return float(np.nanstd(np.concatenate(probe_vals)))


def _generate_null_monthly_streams(
    n_seeds: int,
    base_seed: int,
    author_sd: float,
    residual_sd: float,
    span_months: float,
    n_authors: int,
    msgs_per_author_mean: int,
    seed_offset: int,
) -> list:
    """Generate n_seeds stationary-null monthly-mean streams ONCE, so the
    calibration grid can reuse them across (k, h) / L sweeps."""
    streams = []
    for s in range(n_seeds):
        df = simulate_cohort(
            n_authors=n_authors,
            msgs_per_author_mean=msgs_per_author_mean,
            span_months=span_months,
            drift_per_month=0.0,
            author_sd=author_sd,
            residual_sd=residual_sd,
            seed=base_seed + seed_offset + s,
        )
        streams.append(monthly_means(df))
    return streams


def calibrate_cusum(
    n_seeds: int,
    base_seed: int,
    sigma_monthly: float,
    author_sd: float,
    residual_sd: float,
    span_months: float,
    n_authors: int,
    msgs_per_author_mean: int,
    target_fpr: float = 0.05,
) -> dict:
    """Sweep (k, h) grid on pre-generated null streams; return closest FPR."""
    streams = _generate_null_monthly_streams(
        n_seeds=n_seeds,
        base_seed=base_seed,
        author_sd=author_sd,
        residual_sd=residual_sd,
        span_months=span_months,
        n_authors=n_authors,
        msgs_per_author_mean=msgs_per_author_mean,
        seed_offset=77777,
    )
    k_grid = [0.0, 0.5 * sigma_monthly, 1.0 * sigma_monthly]
    h_mult_grid = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]
    sweep = []
    for k in k_grid:
        for h_mult in h_mult_grid:
            h = h_mult * sigma_monthly
            fires = 0
            for stream in streams:
                out = run_cusum_stream(stream, k=k, h=h, min_instances=3)
                fires += int(out["fired"])
            fpr = fires / n_seeds
            sweep.append(
                {
                    "k": float(k),
                    "k_mult": float(k / sigma_monthly)
                    if sigma_monthly > 0
                    else 0.0,
                    "h": float(h),
                    "h_mult": float(h_mult),
                    "fpr": fpr,
                }
            )
    eligible = [s for s in sweep if s["fpr"] <= target_fpr + 0.02]
    if not eligible:
        eligible = sweep
    eligible.sort(key=lambda s: (abs(s["fpr"] - target_fpr), -s["h"]))
    best = eligible[0]
    return {
        "k": best["k"],
        "h": best["h"],
        "k_mult": best["k_mult"],
        "h_mult": best["h_mult"],
        "achieved_fpr": best["fpr"],
        "sweep": sweep,
        "sigma_monthly": sigma_monthly,
    }


def calibrate_ewma(
    n_seeds: int,
    base_seed: int,
    sigma_monthly: float,
    author_sd: float,
    residual_sd: float,
    span_months: float,
    n_authors: int,
    msgs_per_author_mean: int,
    target_fpr: float = 0.05,
    lam_fixed: float = 0.25,
) -> dict:
    """Sweep L grid at fixed λ on pre-generated null streams."""
    streams = _generate_null_monthly_streams(
        n_seeds=n_seeds,
        base_seed=base_seed,
        author_sd=author_sd,
        residual_sd=residual_sd,
        span_months=span_months,
        n_authors=n_authors,
        msgs_per_author_mean=msgs_per_author_mean,
        seed_offset=55555,
    )
    L_grid = [
        2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 4.0, 4.5, 5.0, 5.5, 6.0,
        7.0, 8.0, 10.0, 12.0, 15.0,
    ]
    sweep = []
    for L in L_grid:
        fires = 0
        for stream in streams:
            out = run_ewma_stream(
                stream, lam=lam_fixed, L=L, min_instances=3
            )
            fires += int(out["fired"])
        fpr = fires / n_seeds
        sweep.append({"lam": lam_fixed, "L": float(L), "fpr": fpr})

    eligible = [s for s in sweep if s["fpr"] <= target_fpr + 0.02]
    if not eligible:
        eligible = sweep
    eligible.sort(key=lambda s: (abs(s["fpr"] - target_fpr), -s["L"]))
    best = eligible[0]
    return {
        "lam": best["lam"],
        "L": best["L"],
        "achieved_fpr": best["fpr"],
        "sweep": sweep,
        "sigma_monthly": sigma_monthly,
    }


# ---------------------------------------------------------------------------
# Cell-level aggregation
# ---------------------------------------------------------------------------
def _aggregate_cell(drift: float, rows: list, alpha: float) -> dict:
    cell = {"drift": drift, "detectors": {}}
    detector_names = [
        "mixedlm_perm",
        "naive_ols",
        "pagehinkley",
        "cusum",
        "ewma",
    ]
    key_map = {
        "mixedlm_perm": "p_mixedlm_perm",
        "naive_ols": "p_naive_ols",
        "pagehinkley": "p_pagehinkley",
        "cusum": "p_cusum",
        "ewma": "p_ewma",
    }
    dir_map = {"cusum": "dir_cusum", "ewma": "dir_ewma"}

    for name in detector_names:
        key = key_map[name]
        ps = [r[key] for r in rows]
        ps_clean = np.array(
            [p for p in ps if p is not None and not np.isnan(p)]
        )
        rejections = int(
            sum(
                1
                for p in ps
                if p is not None and not np.isnan(p) and p < alpha
            )
        )
        n_valid = len(ps_clean)
        d = {
            "rejections": rejections,
            "n_valid": n_valid,
            "ps": [None if p is None or (isinstance(p, float) and np.isnan(p)) else float(p) for p in ps],
            "TPR": rejections / n_valid if n_valid > 0 else float("nan"),
            "mean_p": (
                float(np.nanmean(ps_clean)) if len(ps_clean) else np.nan
            ),
        }
        lo, hi = wilson_ci(rejections, n_valid)
        d["TPR_CI"] = [lo, hi]
        if name in dir_map:
            d["directions"] = [r[dir_map[name]] for r in rows]
        cell["detectors"][name] = d
    return cell


# ---------------------------------------------------------------------------
# MAIN: 5-way comparison
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-authors", type=int, default=100)
    ap.add_argument("--msgs-per-author-mean", type=int, default=50)
    ap.add_argument("--span-months", type=float, default=10.0)
    ap.add_argument("--author-sd", type=float, default=0.12)
    ap.add_argument("--residual-sd", type=float, default=0.23)
    ap.add_argument("--drift-grid", default="0.0,0.002,0.005,0.01,0.03")
    ap.add_argument("--n-seeds-per-cell", type=int, default=100)
    ap.add_argument("--n-permutations", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-calib-seeds", type=int, default=200)
    ap.add_argument(
        "--output-dir",
        default="results/track3_detector_comparison_extended",
    )
    ap.add_argument("--base-seed", type=int, default=98765)
    ap.add_argument("--max-workers", type=int, default=0,
                    help="0 = auto (min(nproc, n_total_tasks))")
    ap.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Use canonical defaults (CUSUM k=σ/2,h=4σ; EWMA λ=0.25,L=3.0)",
    )
    args = ap.parse_args()

    drift_grid = [float(x) for x in args.drift_grid.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------
    # Step 1: Probe σ_monthly on null
    # -----------------------------------------------------------
    t0 = time.time()
    sigma_mo = _probe_sigma_monthly(
        n_probe=10,
        base_seed=args.base_seed,
        author_sd=args.author_sd,
        residual_sd=args.residual_sd,
        span_months=args.span_months,
        n_authors=args.n_authors,
        msgs_per_author_mean=args.msgs_per_author_mean,
    )
    print(
        f"[probe] σ_monthly (null-stream) = {sigma_mo:.4f} "
        f"(residual_sd={args.residual_sd}, author_sd={args.author_sd})"
    )

    # -----------------------------------------------------------
    # Step 2: Calibrate CUSUM + EWMA on null
    # -----------------------------------------------------------
    if args.skip_calibration:
        cusum_cal = {
            "k": 0.5 * sigma_mo,
            "h": 4.0 * sigma_mo,
            "achieved_fpr": None,
            "sweep": None,
            "sigma_monthly": sigma_mo,
            "k_mult": 0.5,
            "h_mult": 4.0,
        }
        ewma_cal = {
            "lam": 0.25,
            "L": 3.0,
            "achieved_fpr": None,
            "sweep": None,
            "sigma_monthly": sigma_mo,
        }
        print("[calibrate] skipped; using canonical defaults")
    else:
        print(
            f"[calibrate] CUSUM grid sweep "
            f"(k∈{{0,σ/2,σ}}, h_mult∈{{2…8}}, "
            f"{args.n_calib_seeds} seeds)"
        )
        cusum_cal = calibrate_cusum(
            n_seeds=args.n_calib_seeds,
            base_seed=args.base_seed,
            sigma_monthly=sigma_mo,
            author_sd=args.author_sd,
            residual_sd=args.residual_sd,
            span_months=args.span_months,
            n_authors=args.n_authors,
            msgs_per_author_mean=args.msgs_per_author_mean,
            target_fpr=args.alpha,
        )
        print(
            f"[calibrate] CUSUM: k={cusum_cal['k']:.4f} "
            f"(={cusum_cal['k_mult']:.2f}σ), "
            f"h={cusum_cal['h']:.4f} "
            f"(={cusum_cal['h_mult']:.2f}σ), "
            f"achieved FPR={cusum_cal['achieved_fpr']:.2%}"
        )

        print(
            f"[calibrate] EWMA grid sweep "
            f"(λ=0.25, L∈{{2.5…6.0}}, {args.n_calib_seeds} seeds)"
        )
        ewma_cal = calibrate_ewma(
            n_seeds=args.n_calib_seeds,
            base_seed=args.base_seed,
            sigma_monthly=sigma_mo,
            author_sd=args.author_sd,
            residual_sd=args.residual_sd,
            span_months=args.span_months,
            n_authors=args.n_authors,
            msgs_per_author_mean=args.msgs_per_author_mean,
            target_fpr=args.alpha,
        )
        print(
            f"[calibrate] EWMA: λ={ewma_cal['lam']}, "
            f"L={ewma_cal['L']}, "
            f"achieved FPR={ewma_cal['achieved_fpr']:.2%}"
        )

    # -----------------------------------------------------------
    # Step 3: Parallel grid sweep (drift × seed)
    # -----------------------------------------------------------
    tasks = []
    for drift in drift_grid:
        for s in range(args.n_seeds_per_cell):
            tasks.append(
                (
                    drift,
                    s,
                    args.base_seed,
                    args.n_authors,
                    args.msgs_per_author_mean,
                    args.span_months,
                    args.author_sd,
                    args.residual_sd,
                    args.n_permutations,
                    cusum_cal["k"],
                    cusum_cal["h"],
                    ewma_cal["lam"],
                    ewma_cal["L"],
                )
            )
    print(
        f"[schedule] {len(tasks)} tasks total "
        f"({len(drift_grid)} drift cells × {args.n_seeds_per_cell} seeds)"
    )

    max_workers = args.max_workers
    if max_workers <= 0:
        max_workers = min(len(tasks), os.cpu_count() or 4)

    rows_by_drift: dict = {d: [] for d in drift_grid}
    t_start = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_seed_task, t): t for t in tasks}
        for fut in as_completed(futures):
            row = fut.result()
            rows_by_drift[row["drift"]].append(row)
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                elapsed = time.time() - t_start
                rate = completed / max(elapsed, 1e-6)
                eta = (len(tasks) - completed) / max(rate, 1e-6)
                print(
                    f"[progress] {completed}/{len(tasks)} "
                    f"({100 * completed / len(tasks):.1f}%) "
                    f"rate={rate:.2f}/s ETA={eta:.0f}s"
                )

    # -----------------------------------------------------------
    # Step 4: Aggregate per-cell
    # -----------------------------------------------------------
    results = {
        "config": vars(args),
        "sigma_monthly": sigma_mo,
        "cusum_calibration": {
            k: v for k, v in cusum_cal.items() if k != "sweep"
        },
        "ewma_calibration": {
            k: v for k, v in ewma_cal.items() if k != "sweep"
        },
        "cusum_sweep": cusum_cal.get("sweep"),
        "ewma_sweep": ewma_cal.get("sweep"),
        "per_drift": [],
    }
    for drift in drift_grid:
        rows = rows_by_drift[drift]
        results["per_drift"].append(
            _aggregate_cell(drift, rows, args.alpha)
        )

    out_path = out_dir / "comparison.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[save] {out_path}")

    # -----------------------------------------------------------
    # Step 5: Summary table
    # -----------------------------------------------------------
    print("\n=== 5-way detector TPR table ===")
    hdr = (
        f"{'β/mo':>8} | {'MixedLM':>9} | {'NaiveOLS':>9} | "
        f"{'PageHink':>9} | {'CUSUM':>9} | {'EWMA':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for cell in results["per_drift"]:
        row = (
            f"{cell['drift']:>8.4f} | "
            f"{cell['detectors']['mixedlm_perm']['TPR']:>8.2%} | "
            f"{cell['detectors']['naive_ols']['TPR']:>8.2%} | "
            f"{cell['detectors']['pagehinkley']['TPR']:>8.2%} | "
            f"{cell['detectors']['cusum']['TPR']:>8.2%} | "
            f"{cell['detectors']['ewma']['TPR']:>8.2%}"
        )
        print(row)

    null_cell = next(
        (c for c in results["per_drift"] if c["drift"] == 0.0), None
    )
    if null_cell:
        print(f"\n=== Type-I (FPR) at β=0 (target ≤ α={args.alpha}) ===")
        for name in [
            "mixedlm_perm",
            "naive_ols",
            "pagehinkley",
            "cusum",
            "ewma",
        ]:
            fpr = null_cell["detectors"][name]["TPR"]
            lo, hi = null_cell["detectors"][name]["TPR_CI"]
            tag = "CALIBRATED" if fpr <= args.alpha + 0.05 else "INFLATED"
            print(
                f"  {name:>15}: {fpr:>6.2%} "
                f"[{lo:.2%}, {hi:.2%}]  {tag}"
            )

    # Slimmer summary file for the paper
    summary = {
        "cusum_calibration": results["cusum_calibration"],
        "ewma_calibration": results["ewma_calibration"],
        "tpr_table": {
            "drift": [c["drift"] for c in results["per_drift"]],
            "mixedlm_perm": [
                c["detectors"]["mixedlm_perm"]["TPR"]
                for c in results["per_drift"]
            ],
            "naive_ols": [
                c["detectors"]["naive_ols"]["TPR"]
                for c in results["per_drift"]
            ],
            "pagehinkley": [
                c["detectors"]["pagehinkley"]["TPR"]
                for c in results["per_drift"]
            ],
            "cusum": [
                c["detectors"]["cusum"]["TPR"]
                for c in results["per_drift"]
            ],
            "ewma": [
                c["detectors"]["ewma"]["TPR"]
                for c in results["per_drift"]
            ],
            "ci_mixedlm_perm": [
                c["detectors"]["mixedlm_perm"]["TPR_CI"]
                for c in results["per_drift"]
            ],
            "ci_naive_ols": [
                c["detectors"]["naive_ols"]["TPR_CI"]
                for c in results["per_drift"]
            ],
            "ci_pagehinkley": [
                c["detectors"]["pagehinkley"]["TPR_CI"]
                for c in results["per_drift"]
            ],
            "ci_cusum": [
                c["detectors"]["cusum"]["TPR_CI"]
                for c in results["per_drift"]
            ],
            "ci_ewma": [
                c["detectors"]["ewma"]["TPR_CI"]
                for c in results["per_drift"]
            ],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] {out_dir / 'summary.json'}")

    print(f"\n[done] total elapsed = {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
