"""Robust M-estimator cross-corpus replication on MultiPref.

Iter+N+199 established on OASST1+OASST2 14-month union that β̂ drift is
robust to OLS/Huber/median. This script replicates the methodology on
MultiPref (227 evaluators × 29-day span, daily time unit due to short
span) as a second real-corpus check.

Dependent variable: `quality` (mean confidence per evaluation ∈ [0,1]).
Time variable: `day_num` (continuous day index since corpus start).
Power-author filter: ≥10 evaluations per evaluator.

Outputs: results/track3_multipref_robust/summary.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

DATA = (Path(__file__).resolve().parents[2] / "3_PILSD_Standalone/data/multipref_evaluator_quality.parquet")
OUT_DIR = (Path(__file__).resolve().parents[2] / "3_PILSD_Standalone/results/track3_multipref_robust")
N_BOOT = 500
RNG_SEED = 20260420


def load_and_filter() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    df = df.dropna(subset=["quality", "day_num", "evaluator"]).copy()
    counts = df.groupby("evaluator").size()
    keep = counts[counts >= 10].index
    return df[df["evaluator"].isin(keep)].copy().reset_index(drop=True)


def demean(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = df["quality"].to_numpy(dtype=np.float64)
    x = df["day_num"].to_numpy(dtype=np.float64)
    uidx, _ = pd.factorize(df["evaluator"], sort=False)
    counts = np.bincount(uidx)
    y_mean = np.bincount(uidx, weights=y) / np.maximum(counts, 1)
    x_mean = np.bincount(uidx, weights=x) / np.maximum(counts, 1)
    return y - y_mean[uidx], x - x_mean[uidx], uidx


def ols(y, x):
    return float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))


def huber(y, x):
    return float(sm.RLM(y, x.reshape(-1, 1), M=sm.robust.norms.HuberT(t=1.345)).fit().params[0])


def median(y, x):
    return float(QuantReg(y, x.reshape(-1, 1)).fit(q=0.5, max_iter=2000).params[0])


def cluster_bootstrap(y, x, uidx, fit_fn, n_boot, rng):
    n_u = uidx.max() + 1
    groups = {u: np.where(uidx == u)[0] for u in range(n_u)}
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        s = rng.integers(0, n_u, size=n_u)
        rows = np.concatenate([groups[u] for u in s])
        try:
            out[b] = fit_fn(y[rows], x[rows])
        except Exception:
            out[b] = np.nan
    return out


def ci(samples, alpha=0.05):
    finite = samples[np.isfinite(samples)]
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    t0 = time.time()
    df = load_and_filter()
    print(f"[load] {len(df)} evals from {df['evaluator'].nunique()} power-evaluators "
          f"(≥10 evals each) over {df['day_num'].max() - df['day_num'].min():.2f} day span")

    y, x, uidx = demean(df)

    b_o = ols(y, x)
    b_h = huber(y, x)
    b_m = median(y, x)

    print(f"\n[OLS]    β̂/day = {b_o:+.4e}")
    print(f"[Huber]  β̂/day = {b_h:+.4e}")
    print(f"[Median] β̂/day = {b_m:+.4e}")

    print(f"\nCluster-bootstrap {N_BOOT} reps per estimator ...")
    boot_o = cluster_bootstrap(y, x, uidx, ols, N_BOOT, np.random.default_rng(RNG_SEED))
    boot_h = cluster_bootstrap(y, x, uidx, huber, N_BOOT, np.random.default_rng(RNG_SEED + 1))
    boot_m = cluster_bootstrap(y, x, uidx, median, N_BOOT, np.random.default_rng(RNG_SEED + 2))

    ci_o = ci(boot_o); ci_h = ci(boot_h); ci_m = ci(boot_m)

    summary = {
        "corpus": "MultiPref-eval-quality",
        "n_evals": int(len(df)),
        "n_evaluators": int(df["evaluator"].nunique()),
        "day_span": float(df["day_num"].max() - df["day_num"].min()),
        "n_bootstrap": N_BOOT,
        "rng_seed": RNG_SEED,
        "estimators": {
            "OLS":    {"beta_per_day": b_o, "ci95": list(ci_o)},
            "Huber":  {"beta_per_day": b_h, "ci95": list(ci_h)},
            "Median": {"beta_per_day": b_m, "ci95": list(ci_m)},
        },
        "all_three_ci_exclude_zero": bool(
            (ci_o[0] > 0 and ci_h[0] > 0 and ci_m[0] > 0) or
            (ci_o[1] < 0 and ci_h[1] < 0 and ci_m[1] < 0)
        ),
        "wall_s": round(time.time() - t0, 1),
    }

    print("\n=== MULTIPREF ROBUST RE-ANALYSIS SUMMARY ===")
    for name in ("OLS", "Huber", "Median"):
        d = summary["estimators"][name]
        print(f"  {name:7s}  β̂/day = {d['beta_per_day']:+.4e}  CI95 = [{d['ci95'][0]:+.4e}, {d['ci95'][1]:+.4e}]")
    print(f"  All three CIs same-sided away from zero: {summary['all_three_ci_exclude_zero']}")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    np.save(OUT_DIR / "boot_ols.npy", boot_o)
    np.save(OUT_DIR / "boot_huber.npy", boot_h)
    np.save(OUT_DIR / "boot_median.npy", boot_m)
    print(f"\nWrote {OUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
