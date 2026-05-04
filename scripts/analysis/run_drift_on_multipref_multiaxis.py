"""Multi-axis drift scan on MultiPref (iter+N+192).

Adaption of `scripts/run_drift_on_oasst2_multiaxis.py` for the AI2 MultiPref
corpus. Day-granularity (29-day window) rather than month-granularity, and
`is_expert` instead of `is_assistant`.

For each of K ≈ 16 per-annotation axes with coverage ≥ min_coverage:

  1. MixedLM REML:  y_ij = α_j + β_axis · day_num_i + γ · is_expert_i + ε_ij
     → β̂ (Wald p, log-lik, convergence flag), BLUPs α̂_j (fallback: within-
     author mean residual).
  2. Fast within-evaluator permutation via FWL-OLS (demean by evaluator,
     partial-out is_expert) — verified relative-error < 1e-3 vs MixedLM on
     the same observed data. N_perm = 1000, Phipson-Smyth p = (exceed+1)/(M+1).
  3. Cluster-bootstrap BCa by evaluator (B = 2000), resample evaluators with
     replacement, concatenate their rows, refit FWL → β̂*, then BCa95.
  4. Multiplicity: Bonferroni + BH-FDR across K valid axes.
  5. Cross-axis correlations: per-evaluator random-intercept BLUPs stacked
     across axes → Pearson r matrix. Tests whether drifts are low-rank
     (fatigue) or high-rank (aspect-specific re-calibration).

References
  - Miranda et al. 2024 arXiv:2410.19133 (dataset)
  - Pinheiro & Bates 2000 (MixedLM REML)
  - Good 2006 §11.4 (within-group permutation)
  - Efron 1987 (BCa)
  - Benjamini & Hochberg 1995 (FDR)
  - Phipson & Smyth 2010 (conservative permutation p-value)
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

warnings.filterwarnings("ignore")


DEFAULT_AXES = [
    "overall_conf", "helpful_conf", "truthful_conf", "harmless_conf",
    "mean_conf", "time_spent", "log_time_spent", "total_reasons_checked",
    "overall_decisiveness", "helpful_decisiveness",
    "truthful_decisiveness", "harmless_decisiveness",
    "is_tie_overall", "is_tie_helpful",
    "is_tie_truthful", "is_tie_harmless",
]


# ---------------------------------------------------------------------------
# FWL fast-path: partial out evaluator FE + is_expert, regress on day_num
# ---------------------------------------------------------------------------
def prep_fwl(df: pd.DataFrame, y_col: str):
    user_codes = df["user_id"].astype("category").cat.codes.to_numpy()
    n_groups = int(user_codes.max()) + 1
    group_count = np.bincount(user_codes, minlength=n_groups).astype(float)

    y = df[y_col].to_numpy(dtype=float)
    day = df["day_num"].to_numpy(dtype=float)
    is_exp = df["is_expert"].to_numpy(dtype=float)

    y_sum = np.bincount(user_codes, weights=y, minlength=n_groups)
    y_mean = y_sum / group_count
    e_sum = np.bincount(user_codes, weights=is_exp, minlength=n_groups)
    e_mean = e_sum / group_count

    y_d = y - y_mean[user_codes]
    e_d = is_exp - e_mean[user_codes]
    ee = float(e_d @ e_d)
    if ee < 1e-12:
        y_t = y_d.copy()
    else:
        ye = float(e_d @ y_d)
        y_t = y_d - (ye / ee) * e_d

    order = np.argsort(user_codes, kind="stable")
    inv_order = np.argsort(order)
    user_codes_sorted = user_codes[order]
    _, first_idx = np.unique(user_codes_sorted, return_index=True)
    first_idx = np.sort(first_idx)
    boundaries = np.concatenate([first_idx, [len(user_codes_sorted)]]).astype(np.int64)
    day_sorted = day[order].copy()

    return dict(
        y_t=y_t, e_d=e_d, ee=ee,
        day_sorted=day_sorted, boundaries=boundaries,
        inv_order=inv_order, user_codes=user_codes,
        group_count=group_count, n_groups=n_groups,
    )


def fwl_coef(day: np.ndarray, ctx: dict) -> float:
    user_codes = ctx["user_codes"]
    group_count = ctx["group_count"]
    e_d = ctx["e_d"]
    ee = ctx["ee"]
    y_t = ctx["y_t"]
    n_groups = ctx["n_groups"]
    d_sum = np.bincount(user_codes, weights=day, minlength=n_groups)
    d_mean = d_sum / group_count
    x_d = day - d_mean[user_codes]
    if ee < 1e-12:
        x_t = x_d
    else:
        xe = float(e_d @ x_d)
        x_t = x_d - (xe / ee) * e_d
    denom = float(x_t @ x_t)
    if denom < 1e-18:
        return float("nan")
    return float((x_t @ y_t) / denom)


def permute_batch(args):
    (seed, n, day_sorted, boundaries, inv_order,
     user_codes, group_count, e_d, ee, y_t) = args
    rng = np.random.default_rng(seed)
    n_groups = int(user_codes.max()) + 1
    out = np.empty(n, dtype=np.float64)
    local = day_sorted.copy()
    for i in range(n):
        for j in range(len(boundaries) - 1):
            lo, hi = boundaries[j], boundaries[j + 1]
            if hi - lo > 1:
                rng.shuffle(local[lo:hi])
        day = local[inv_order]
        d_sum = np.bincount(user_codes, weights=day, minlength=n_groups)
        d_mean = d_sum / group_count
        x_d = day - d_mean[user_codes]
        if ee < 1e-12:
            x_t = x_d
        else:
            xe = float(e_d @ x_d)
            x_t = x_d - (xe / ee) * e_d
        denom = float(x_t @ x_t)
        out[i] = float((x_t @ y_t) / denom) if denom > 1e-18 else np.nan
    return out


def cluster_bootstrap(df: pd.DataFrame, y_col: str, n_boot: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    authors = df["user_id"].unique()
    n_authors = len(authors)
    row_idx_by_author = {a: np.where(df["user_id"].values == a)[0] for a in authors}

    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample_authors = authors[rng.integers(0, n_authors, size=n_authors)]
        all_idx = np.concatenate([row_idx_by_author[a] for a in sample_authors])
        sub = df.iloc[all_idx].reset_index(drop=True)
        ctx = prep_fwl(sub, y_col)
        boot[b] = fwl_coef(sub["day_num"].to_numpy(), ctx)
    return boot


def bca_ci(boot: np.ndarray, theta_hat: float, alpha: float = 0.05):
    boot = np.asarray(boot, dtype=float)
    boot = boot[np.isfinite(boot)]
    if len(boot) < 10:
        return float("nan"), float("nan"), {}
    n_boot = len(boot)
    p0 = float(np.mean(boot < theta_hat))
    if p0 == 0.0:
        p0 = 1.0 / (2 * n_boot)
    if p0 == 1.0:
        p0 = 1.0 - 1.0 / (2 * n_boot)
    z0 = norm.ppf(p0)
    mu = boot.mean()
    num = np.sum((mu - boot) ** 3)
    den = 6.0 * (np.sum((mu - boot) ** 2)) ** 1.5
    a = 0.0 if den == 0 else num / den
    zL = norm.ppf(alpha / 2)
    zU = norm.ppf(1 - alpha / 2)
    aL = norm.cdf(z0 + (z0 + zL) / (1 - a * (z0 + zL)))
    aU = norm.cdf(z0 + (z0 + zU) / (1 - a * (z0 + zU)))
    aL = float(np.clip(aL, 1e-6, 1 - 1e-6))
    aU = float(np.clip(aU, 1e-6, 1 - 1e-6))
    ciL = float(np.quantile(boot, aL))
    ciU = float(np.quantile(boot, aU))
    return ciL, ciU, {"z0": float(z0), "a_hat": float(a)}


def perm_pvalue_phipson_smyth(exceed: int, M: int) -> float:
    return (exceed + 1) / (M + 1)


def bh_fdr(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    sorted_p = p[order]
    k_ok = np.where(sorted_p <= (np.arange(1, n + 1) / n) * alpha)[0]
    if len(k_ok) == 0:
        return [False] * n
    k_max = k_ok.max()
    cutoff = sorted_p[k_max]
    return [pv <= cutoff for pv in p]


# ---------------------------------------------------------------------------
# Per-axis driver
# ---------------------------------------------------------------------------
def run_one_axis(df: pd.DataFrame, axis: str, n_perm: int, n_boot: int,
                 seed: int, out_dir: Path, min_coverage_frac: float,
                 n_workers: int) -> dict:
    df_a = df.dropna(subset=[axis]).reset_index(drop=True)
    coverage = len(df_a) / len(df)

    result = {
        "axis": axis,
        "n_obs": int(len(df_a)),
        "n_authors": int(df_a["user_id"].nunique()),
        "coverage_frac": float(coverage),
        "mean": float(df_a[axis].mean()) if len(df_a) else float("nan"),
        "std": float(df_a[axis].std()) if len(df_a) else float("nan"),
        "skipped": False,
    }
    if coverage < min_coverage_frac:
        result["skipped"] = True
        result["skip_reason"] = f"coverage {coverage:.2%} < {min_coverage_frac:.0%}"
        return result
    if df_a[axis].std() < 1e-8:
        result["skipped"] = True
        result["skip_reason"] = "near-zero variance"
        return result

    # --- MixedLM primary ---
    t0 = time.time()
    try:
        df_a = df_a.rename(columns={axis: "_y"})
        md = smf.mixedlm("_y ~ day_num + is_expert",
                         data=df_a, groups=df_a["user_id"])
        res = md.fit(method="lbfgs", maxiter=200, disp=False)
        coef_obs = float(res.params["day_num"])
        wald_p = float(res.pvalues["day_num"])
        llf = float(res.llf)
        conv = bool(res.converged)
        df_a = df_a.rename(columns={"_y": axis})
    except Exception as e:
        result["skipped"] = True
        result["skip_reason"] = f"MixedLM failed: {e}"
        return result

    # BLUPs for cross-axis correlation
    try:
        re_dict = res.random_effects
        re_series = pd.Series({a: float(v.iloc[0]) for a, v in re_dict.items()},
                              name=f"ranef_{axis}")
    except Exception:
        y_arr = df_a[axis].to_numpy(dtype=float)
        day_arr = df_a["day_num"].to_numpy(dtype=float)
        exp_arr = df_a["is_expert"].to_numpy(dtype=float)
        gamma = float(res.params.get("is_expert", 0.0))
        intercept = float(res.params.get("Intercept", 0.0))
        resid = y_arr - intercept - coef_obs * day_arr - gamma * exp_arr
        re_series = (pd.Series(resid, index=df_a["user_id"].values)
                     .groupby(level=0).mean()
                     .rename(f"ranef_{axis}"))
    mixedlm_secs = time.time() - t0

    # FWL matches
    ctx = prep_fwl(df_a, axis)
    coef_fwl = fwl_coef(df_a["day_num"].to_numpy(), ctx)
    rel_err = abs(coef_fwl - coef_obs) / max(abs(coef_obs), 1e-18)

    # --- Permutation ---
    t0 = time.time()
    chunks = []
    chunk_size = max(50, n_perm // max(n_workers * 4, 1))
    done = 0
    chunk_seed = seed + 1_000_000
    args_list = []
    while done < n_perm:
        n_this = min(chunk_size, n_perm - done)
        args_list.append((
            chunk_seed + done, n_this,
            ctx["day_sorted"], ctx["boundaries"], ctx["inv_order"],
            ctx["user_codes"], ctx["group_count"],
            ctx["e_d"], ctx["ee"], ctx["y_t"],
        ))
        done += n_this

    if n_workers > 1 and len(args_list) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            parts = list(ex.map(permute_batch, args_list))
    else:
        parts = [permute_batch(a) for a in args_list]
    null_coefs = np.concatenate(parts)
    null_coefs = null_coefs[np.isfinite(null_coefs)]
    exceed = int((np.abs(null_coefs) >= abs(coef_obs)).sum())
    p_perm = perm_pvalue_phipson_smyth(exceed, len(null_coefs))
    perm_secs = time.time() - t0
    np.save(out_dir / f"null_coefs_{axis}.npy", null_coefs)

    # --- Cluster-bootstrap BCa ---
    t0 = time.time()
    boot = cluster_bootstrap(df_a, axis, n_boot=n_boot, seed=seed + 777)
    boot = boot[np.isfinite(boot)]
    ci_lo_pct = float(np.quantile(boot, 0.025)) if len(boot) else float("nan")
    ci_hi_pct = float(np.quantile(boot, 0.975)) if len(boot) else float("nan")
    ci_lo_bca, ci_hi_bca, bca_diag = bca_ci(boot, coef_obs, alpha=0.05)
    boot_secs = time.time() - t0
    np.save(out_dir / f"bootstrap_betas_{axis}.npy", boot)

    # Random-intercept series
    re_series.to_frame().to_parquet(out_dir / f"ranef_{axis}.parquet")

    # Naive daily-mean OLS
    daily = (df_a.assign(d=df_a["day_num"].astype(int))
             .groupby("d")[axis].mean().reset_index())
    daily.columns = ["d", axis]
    naive_coef, naive_p = float("nan"), float("nan")
    if len(daily) >= 3:
        x = daily["d"].to_numpy(float)
        y = daily[axis].to_numpy(float)
        X = np.column_stack([np.ones_like(x), x])
        bhat, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ bhat
        se2 = resid @ resid / max(len(x) - 2, 1)
        var = se2 * np.linalg.inv(X.T @ X)[1, 1]
        if var > 0:
            tstat = bhat[1] / np.sqrt(var)
            from scipy.stats import t as tdist
            naive_p = 2 * (1 - tdist.cdf(abs(tstat), df=max(len(x) - 2, 1)))
            naive_coef = float(bhat[1])

    result.update({
        "beta_day": coef_obs,
        "wald_p": wald_p,
        "mixedlm_converged": conv,
        "mixedlm_loglik": llf,
        "fwl_vs_mixedlm_rel_err": float(rel_err),
        "perm_exceed": exceed,
        "perm_n": int(len(null_coefs)),
        "perm_p_phipson_smyth": float(p_perm),
        "null_mean": float(null_coefs.mean()),
        "null_std": float(null_coefs.std()),
        "boot_n_finite": int(len(boot)),
        "boot_mean": float(boot.mean()) if len(boot) else float("nan"),
        "boot_std": float(boot.std()) if len(boot) else float("nan"),
        "percentile_ci_95": [ci_lo_pct, ci_hi_pct],
        "bca_ci_95": [ci_lo_bca, ci_hi_bca],
        "bca_diag": bca_diag,
        "naive_ols_coef": naive_coef,
        "naive_ols_p": naive_p,
        "timing_secs": {"mixedlm": mixedlm_secs, "perm": perm_secs, "bootstrap": boot_secs},
    })
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_multiaxis_cohort(cohort_parquet: Path, cohort_filter: Path) -> pd.DataFrame:
    df = pd.read_parquet(cohort_parquet)
    cohort_users = set(pd.read_parquet(cohort_filter).index)
    df = df[df["user_id"].isin(cohort_users)].reset_index(drop=True)
    # Ensure day_num + is_expert columns present (recompute if missing).
    if "day_num" not in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        t0 = df["ts"].min()
        df["day_num"] = ((df["ts"] - t0).dt.total_seconds() / 86400.0).astype(float)
    if "is_expert" not in df.columns:
        df["is_expert"] = (df["kind"] == "expert").astype(float)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-parquet",
                    default="data/multipref_multiaxis_quality.parquet")
    ap.add_argument("--cohort-filter",
                    default="data/multipref_evaluator_cohort.parquet")
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--output-dir", default="results/track3_multipref_multiaxis")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--axes", nargs="*", default=None,
                    help="Subset of axes to run (default: all 16)")
    ap.add_argument("--min-coverage", type=float, default=0.20)
    ap.add_argument("--n-workers", type=int, default=0)
    args = ap.parse_args()

    if args.n_workers <= 0:
        args.n_workers = max(1, os.cpu_count() or 1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cfg] n_perm={args.n_permutations:,}  n_boot={args.n_bootstrap:,}  "
          f"workers={args.n_workers}  min_cov={args.min_coverage:.0%}")

    df = load_multiaxis_cohort(Path(args.cohort_parquet), Path(args.cohort_filter))
    print(f"[data] cohort: {df['user_id'].nunique()} evaluators, "
          f"{len(df)} rows, day range "
          f"{df['day_num'].min():.2f}..{df['day_num'].max():.2f}")

    axes = args.axes if args.axes else DEFAULT_AXES

    per_axis_results = []
    t_total = time.time()
    for axis in axes:
        if axis not in df.columns:
            print(f"[skip] {axis}: not in parquet")
            continue
        print(f"\n[axis] {axis}")
        res = run_one_axis(df, axis, args.n_permutations, args.n_bootstrap,
                           args.seed, out_dir, args.min_coverage, args.n_workers)
        if res.get("skipped"):
            print(f"  SKIPPED: {res.get('skip_reason')}")
        else:
            print(f"  β̂={res['beta_day']:+.4e}/day  Wald p={res['wald_p']:.2e}  "
                  f"perm p={res['perm_p_phipson_smyth']:.2e}  "
                  f"BCa95=[{res['bca_ci_95'][0]:+.3e}, {res['bca_ci_95'][1]:+.3e}]  "
                  f"naive p={res['naive_ols_p']:.2e}  "
                  f"time={sum(res['timing_secs'].values()):.1f}s")
        per_axis_results.append(res)

    total_elapsed = time.time() - t_total
    print(f"\n[total] {total_elapsed/60:.1f} min across {len(per_axis_results)} axes")

    # Multiplicity correction
    valid = [r for r in per_axis_results if not r.get("skipped")]
    K = len(valid)
    if K > 0:
        perm_ps = [r["perm_p_phipson_smyth"] for r in valid]
        bonf_alpha = 0.05 / K
        bonf_reject = [p < bonf_alpha for p in perm_ps]
        bh_reject = bh_fdr(perm_ps, alpha=0.05)
        for r, br, fr in zip(valid, bonf_reject, bh_reject):
            r["bonferroni_reject_0p05"] = bool(br)
            r["bh_fdr_reject_0p05"] = bool(fr)
            r["bonferroni_alpha"] = bonf_alpha

    def verdict(r):
        if r.get("skipped"):
            return "SKIPPED"
        if r["perm_p_phipson_smyth"] >= 0.05:
            return "NULL"
        if r.get("bonferroni_reject_0p05"):
            return f"DRIFT (Bonf α={r['bonferroni_alpha']:.4f})"
        if r.get("bh_fdr_reject_0p05"):
            return "DRIFT (BH-FDR only)"
        return "weak (raw p<0.05, fails multiplicity)"

    rows = []
    for r in per_axis_results:
        if r.get("skipped"):
            rows.append({
                "axis": r["axis"], "n_obs": r["n_obs"], "coverage": r["coverage_frac"],
                "beta_day": None, "wald_p": None, "perm_p": None, "naive_p": None,
                "bca_lo": None, "bca_hi": None,
                "verdict": f"SKIPPED ({r.get('skip_reason')})",
            })
        else:
            rows.append({
                "axis": r["axis"],
                "n_obs": r["n_obs"],
                "coverage": r["coverage_frac"],
                "beta_day": r["beta_day"],
                "wald_p": r["wald_p"],
                "perm_p": r["perm_p_phipson_smyth"],
                "naive_p": r["naive_ols_p"],
                "bca_lo": r["bca_ci_95"][0],
                "bca_hi": r["bca_ci_95"][1],
                "verdict": verdict(r),
            })
    tbl = pd.DataFrame(rows)
    tbl_path = out_dir / "per_axis_table.csv"
    tbl.to_csv(tbl_path, index=False)
    print(f"\n[save] {tbl_path}")
    print("\n" + tbl.to_string(index=False))

    # Cross-axis random-intercept correlations
    print("\n[corr] loading per-axis random intercepts...")
    ranef_frames = {}
    for r in valid:
        pq = out_dir / f"ranef_{r['axis']}.parquet"
        if pq.exists():
            ranef_frames[r["axis"]] = pd.read_parquet(pq).iloc[:, 0]
    corr = None
    if len(ranef_frames) >= 2:
        ranef_df = pd.DataFrame(ranef_frames)
        corr = ranef_df.corr()
        corr_path = out_dir / "axis_correlations.csv"
        corr.to_csv(corr_path)
        print(f"[save] {corr_path}")
        print("\nCross-axis Pearson correlations (rounded):")
        print(corr.round(3).to_string())

        pairs = []
        axes_list = list(corr.columns)
        for i in range(len(axes_list)):
            for j in range(i + 1, len(axes_list)):
                pairs.append((axes_list[i], axes_list[j], float(corr.iloc[i, j])))
        pairs.sort(key=lambda p: -abs(p[2]))
        print("\nTop-10 strongest |corr| axis-pairs:")
        for a, b, c in pairs[:10]:
            print(f"  {a:<26s} vs {b:<26s}  r = {c:+.3f}")
        print("\nTop-10 weakest |corr| axis-pairs:")
        for a, b, c in sorted(pairs, key=lambda p: abs(p[2]))[:10]:
            print(f"  {a:<26s} vs {b:<26s}  r = {c:+.3f}")

    # Summary JSON
    def _sanitize(o):
        if isinstance(o, dict): return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, list): return [_sanitize(v) for v in o]
        if isinstance(o, float):
            if np.isinf(o): return "Infinity" if o > 0 else "-Infinity"
            if np.isnan(o): return None
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    summary = {
        "dataset": "allenai/multipref",
        "cohort": "power-evaluator (n>=30 annotations, span>=7d)",
        "n_authors": int(df["user_id"].nunique()),
        "n_observations": int(len(df)),
        "day_range": [float(df["day_num"].min()), float(df["day_num"].max())],
        "axes_tested": [r["axis"] for r in valid],
        "axes_skipped": [r["axis"] for r in per_axis_results if r.get("skipped")],
        "K_tested": K,
        "bonferroni_alpha_0p05": 0.05 / K if K else None,
        "per_axis": per_axis_results,
        "config": {
            "n_permutations": args.n_permutations,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "min_coverage": args.min_coverage,
        },
        "total_elapsed_sec": total_elapsed,
    }
    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps(_sanitize(summary), indent=2))
    print(f"\n[save] {out_json}")


if __name__ == "__main__":
    main()
