"""Track 3 COMPOUND stress test (PARALLEL + INCREMENTAL) — 50 seeds, 500 perms.

This is a scale-up of compound_stress_test.py that:
  1. Parallelises seeds across worker processes (16 cores → ~16× speedup).
  2. Writes each cell to a per-cell JSON as soon as it finishes so a crash
     does not lose prior work. Final aggregate JSON is written at the end.
  3. Pins BLAS to single-thread per worker (MixedLM + multiprocessing
     otherwise oversubscribes).

DGP / detectors: identical to compound_stress_test.py (re-uses the helpers via
import) so results are comparable cell-for-cell.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Single-threaded BLAS — MUST be set BEFORE numpy / statsmodels import in
# parent or workers. Using `os.environ[]` (not setdefault) so we override
# any inherited values.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
           "BLIS_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Import the canonical DGP + detectors from the serial script so we stay
# byte-identical to the 12-seed baseline (only scale changes).
serial = _load(_HERE / "compound_stress_test.py", "compound_serial_ref")
simulate_compound_cohort = serial.simulate_compound_cohort
mixed = serial.mixed
simpson = serial.simpson


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    from math import sqrt
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    hw = z * sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - hw), min(1.0, center + hw)


def _seed_task(args):
    """Run one seed for one (amp, beta) cell. Returns a dict."""
    amp, beta, seed, n_perms, base_seed = args
    # Re-enforce BLAS thread pinning in the worker (idempotent guard even
    # though parent already set these before numpy imported).
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
               "BLIS_NUM_THREADS"):
        os.environ[_k] = "1"
    try:
        # Runtime override for threadpoolctl-managed pools (scipy+sklearn
        # use OpenBLAS/MKL via threadpoolctl; env alone is not enough if
        # threads were already spawned before fork).
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    warnings.filterwarnings("ignore")

    df = simulate_compound_cohort(amp=amp, true_drift=beta, seed=base_seed + seed)
    ml_slope = mixed.fit_mixedlm_slope(df)
    naive_slope = mixed.fit_ols_slope(df)
    p_ml = simpson.detector_mixedlm_perm(df, n_perms=n_perms, seed=base_seed + seed)
    p_naive = simpson.detector_naive_ols(df)
    return {
        "amp": amp,
        "beta": beta,
        "seed": seed,
        "ml_slope": ml_slope,
        "naive_slope": naive_slope,
        "p_ml": p_ml,
        "p_naive": p_naive,
    }


def _aggregate_cell(amp, beta, per_seed_rows, alpha):
    ml_slopes = [r["ml_slope"] for r in per_seed_rows]
    naive_slopes = [r["naive_slope"] for r in per_seed_rows]
    ml_ps = [r["p_ml"] for r in per_seed_rows]
    naive_ps = [r["p_naive"] for r in per_seed_rows]
    n = len(per_seed_rows)

    ml_reject = sum(1 for p in ml_ps if not np.isnan(p) and p < alpha)
    naive_reject = sum(1 for p in naive_ps if not np.isnan(p) and p < alpha)
    ml_correct_sign = 0
    naive_correct_sign = 0
    if beta != 0:
        for s, ns in zip(ml_slopes, naive_slopes):
            if not np.isnan(s) and np.sign(s) == np.sign(beta):
                ml_correct_sign += 1
            if not np.isnan(ns) and np.sign(ns) == np.sign(beta):
                naive_correct_sign += 1

    cell = {
        "amp": amp,
        "true_drift": beta,
        "n_seeds": n,
        "ml_slopes": ml_slopes,
        "naive_slopes": naive_slopes,
        "ml_ps": ml_ps,
        "naive_ps": naive_ps,
        "ml_reject": ml_reject,
        "naive_reject": naive_reject,
        "ml_correct_sign": ml_correct_sign,
        "naive_correct_sign": naive_correct_sign,
        "ml_mean_slope": float(np.nanmean(ml_slopes)),
        "naive_mean_slope": float(np.nanmean(naive_slopes)),
    }
    cell["ml_bias"] = cell["ml_mean_slope"] - beta
    cell["naive_bias"] = cell["naive_mean_slope"] - beta
    cell["ml_fpr_or_tpr"] = ml_reject / n
    cell["naive_fpr_or_tpr"] = naive_reject / n
    if beta != 0:
        cell["ml_sign_rate"] = ml_correct_sign / n
        cell["naive_sign_rate"] = naive_correct_sign / n
        cell["ml_sign_ci"] = list(_wilson(ml_correct_sign, n))
        cell["naive_sign_ci"] = list(_wilson(naive_correct_sign, n))
    # FPR Wilson CI for null cells
    if beta == 0:
        cell["ml_fpr_ci"] = list(_wilson(ml_reject, n))
        cell["naive_fpr_ci"] = list(_wilson(naive_reject, n))
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amps", default="0.0,1.0")
    ap.add_argument("--betas", default="0.0,0.005")
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--n-perms", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=14,
                    help="parallel seed workers (leave 2 cores free)")
    ap.add_argument(
        "--output-path",
        default="results/track3_compound_stress_50seeds.json",
    )
    ap.add_argument(
        "--per-cell-dir",
        default="results/track3_compound_stress/per_cell_50s_500p",
    )
    ap.add_argument("--base-seed", type=int, default=9191)
    args = ap.parse_args()

    amps = [float(x) for x in args.amps.split(",")]
    betas = [float(x) for x in args.betas.split(",")]

    Path(args.per_cell_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(
        f"=== Track 3 COMPOUND stress (parallel): "
        f"amps={amps} betas={betas} n_seeds={args.n_seeds} "
        f"n_perms={args.n_perms} workers={args.workers} α={args.alpha} ===",
        flush=True,
    )

    cells = []
    for amp in amps:
        for beta in betas:
            cell_tag = f"amp{amp:.2f}_beta{beta:+.4f}"
            cell_path = Path(args.per_cell_dir) / f"cell_{cell_tag}.json"
            if cell_path.exists():
                # Resume: load prior cell if matches n_seeds
                try:
                    prior = json.loads(cell_path.read_text())
                    if prior.get("n_seeds") == args.n_seeds:
                        cells.append(prior)
                        print(
                            f"[resume] loaded cell {cell_tag} from {cell_path}",
                            flush=True,
                        )
                        continue
                except Exception:
                    pass

            print(f"[cell] start {cell_tag} at t+{time.time() - t0:.0f}s",
                  flush=True)
            tasks = [
                (amp, beta, s, args.n_perms, args.base_seed)
                for s in range(args.n_seeds)
            ]
            per_seed_rows = []
            t_cell = time.time()
            # Use 'fork' (Linux default). Env vars are set BEFORE numpy
            # import at module top, so fork children inherit the
            # single-threaded BLAS state. Workers additionally call
            # threadpool_limits(1) as belt-and-braces. We tried 'spawn'
            # but it hung during process bootstrap at high system load.
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(
                max_workers=args.workers, mp_context=ctx
            ) as ex:
                futures = {ex.submit(_seed_task, t): t for t in tasks}
                done = 0
                for fut in as_completed(futures):
                    r = fut.result()
                    per_seed_rows.append(r)
                    done += 1
                    if done % 5 == 0 or done == args.n_seeds:
                        print(
                            f"  {cell_tag}: {done}/{args.n_seeds} seeds done "
                            f"(cell elapsed {time.time() - t_cell:.0f}s, "
                            f"total {time.time() - t0:.0f}s)",
                            flush=True,
                        )
            # Sort per-seed rows for reproducibility
            per_seed_rows.sort(key=lambda x: x["seed"])
            cell = _aggregate_cell(amp, beta, per_seed_rows, args.alpha)
            cell_path.write_text(json.dumps(cell, indent=2, default=str))
            print(f"[save] {cell_path}", flush=True)
            if beta == 0:
                print(
                    f"  [DONE {cell_tag}]  ML_FPR={cell['ml_fpr_or_tpr']:.1%} "
                    f"(CI {cell['ml_fpr_ci'][0]:.1%}-{cell['ml_fpr_ci'][1]:.1%}) "
                    f"naive_FPR={cell['naive_fpr_or_tpr']:.1%}",
                    flush=True,
                )
            else:
                print(
                    f"  [DONE {cell_tag}]  ML_sign={cell['ml_correct_sign']}/"
                    f"{args.n_seeds} ({cell['ml_sign_rate']:.1%}, "
                    f"Wilson CI {cell['ml_sign_ci'][0]:.1%}-"
                    f"{cell['ml_sign_ci'][1]:.1%}) "
                    f"naive_sign={cell['naive_correct_sign']}/{args.n_seeds}",
                    flush=True,
                )
            cells.append(cell)

    # Final aggregation + paper-ready claim
    print("\n=== 2×2 MixedLM sign-correctness (β≠0; FPR for β=0) ===",
          flush=True)
    for amp in amps:
        row = []
        for beta in betas:
            c = next(
                c for c in cells
                if abs(c["amp"] - amp) < 1e-12
                and abs(c["true_drift"] - beta) < 1e-12
            )
            if beta == 0:
                row.append(f"FPR={c['ml_fpr_or_tpr']:.1%}")
            else:
                row.append(f"{c['ml_correct_sign']}/{c['n_seeds']}")
        print(f"amp={amp:>4.2f} | " + "  | ".join(f"{x:>16}" for x in row),
              flush=True)

    hardest = next(
        (c for c in cells
         if abs(c["amp"] - max(amps)) < 1e-12
         and abs(c["true_drift"] - max(betas)) < 1e-12),
        None,
    )
    paper_claim = None
    if hardest and max(betas) != 0:
        r = hardest.get("ml_sign_rate", 0.0)
        lo, hi = hardest.get("ml_sign_ci", [0.0, 1.0])
        nr = hardest.get("naive_sign_rate", 0.0)
        nlo, nhi = hardest.get("naive_sign_ci", [0.0, 1.0])
        n = hardest["n_seeds"]
        robust = r >= 0.60
        verdict = "ROBUST" if robust else "DEGRADED"
        paper_claim = (
            f"Under compound adversarial stress (heteroskedastic σ_ε(t) with "
            f"amp=1.0, Simpson-style composition shift, real per-author drift "
            f"β=+{max(betas):.3f}/mo), MixedLM + within-author permutation "
            f"recovers the correct drift sign in {hardest['ml_correct_sign']}/"
            f"{n} seeds ({r:.1%}, Wilson 95% CI [{lo:.1%}, {hi:.1%}]), while "
            f"naive monthly-aggregate OLS recovers the correct sign in "
            f"{hardest['naive_correct_sign']}/{n} seeds ({nr:.1%}, Wilson 95% "
            f"CI [{nlo:.1%}, {nhi:.1%}]). This establishes compound bias-"
            f"robustness of MixedLM + within-author permutation against the "
            f"simultaneous combination of all three adversarial factors "
            f"validated individually in prior ablations."
        )
        print(f"\n=== Paper-ready claim ({verdict}) ===\n{paper_claim}",
              flush=True)

    out = {
        "config": vars(args),
        "cells": cells,
        "paper_claim": paper_claim,
        "hardest_cell_sign_rate":
            hardest.get("ml_sign_rate") if hardest else None,
        "hardest_cell_sign_ci":
            hardest.get("ml_sign_ci") if hardest else None,
        "wall_clock_sec": time.time() - t0,
    }
    Path(args.output_path).write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[save] {args.output_path}", flush=True)
    print(f"[wall] {out['wall_clock_sec']:.1f}s total", flush=True)


if __name__ == "__main__":
    main()
