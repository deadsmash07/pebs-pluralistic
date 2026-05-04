"""Run MixedLM + within-user permutation drift detector on PRISM (5th real-data corpus).

PRISM (Kirk et al. 2024) exposes:
  * per-utterance `score_user` (1-100 integer) from the human rater who owns
    the conversation,
  * per-conversation `generated_datetime` (UTC, 29-day collection window
    2023-11-22 -> 2023-12-22),
  * stable `user_id` across conversations.

Per the `track1_temporal_cv_robustness` memo:
  * 1394 users, 68,371 utterances,
  * per-user span median ~43 min, p90 ~1.5 h, so almost all within-user
    variation is *within-session* rather than long-horizon.

We therefore run TWO complementary axes:

  (a) `day_num` across the 29-day collection window -- symmetric to MultiPref
      and OASST2, asks "is there a cross-cohort maturation effect in how
      PRISM annotators score responses as the 29-day collection wave
      progresses?". Identified off the ~348 users with span >= 1 day.

  (b) `hour_within_user` relative to each user's first utterance -- asks
      "does an individual annotator's numeric score drift monotonically over
      the course of their own session (fatigue / anchoring / learning)?"

For both axes we fit a random-intercept-by-user MixedLM, run within-user
permutations (Good 2006 Sec. 11.4) for the primary p-value, and cluster-
bootstrap by user for the BCa 95% CI on beta-hat.

Model
-----
    score_user_ij  =  beta_0 + beta_1 * time_ij + b_j + eps_ij
    b_j ~ N(0, sigma_u^2)   (per-user random intercept)

Primary statistic equals the author-demeaned OLS slope (FWL), which allows
an exact ~300 perms/s fast-path; we sanity-check FWL vs MixedLM on 200
random permutations before running the full 2,000-perm block.

Honest caveat
-------------
Kalman-filter time-varying calibration ALREADY reported a null on PRISM
(memory `negT1_KALMAN` / paper Appx H.4): per-user sessions are too short
relative to the noise floor for any session-interior slope to rise out of
noise. We expect a null here too; if so, that is a consistency check on
the Kalman result at a different model class.

Run
---
    python3 scripts/run_drift_on_prism.py \\
        --rm-parquet ../1_Causal_RLHF/data/prism_rm_scored.parquet \\
        --ts-parquet ../1_Causal_RLHF/data/prism_conversation_timestamps.parquet \\
        --n-permutations 2000 --n-bootstrap 2000 \\
        --output-dir results/track3_prism_temporal
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

# Force line-buffered stdout so progress prints show up in nohup log
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def load_prism(rm_path: Path, ts_path: Path) -> pd.DataFrame:
    rm = pd.read_parquet(rm_path)
    ts = pd.read_parquet(ts_path)
    df = rm.merge(ts, on="conversation_id", how="left", validate="many_to_one")
    assert df["generated_datetime"].notna().all(), "timestamp join lost rows"
    df["score_user"] = df["score_user"].astype(float)
    df["if_chosen"] = df["if_chosen"].astype(float)
    df = df.dropna(subset=["score_user"]).reset_index(drop=True)
    # Day axis: days since global t0
    t0 = df["generated_datetime"].min()
    df["day_num"] = (
        (df["generated_datetime"] - t0).dt.total_seconds() / 86400.0
    ).astype(float)
    # Hour-within-user axis: hours since this user's first utterance
    per_user_t0 = df.groupby("user_id")["generated_datetime"].transform("min")
    df["hour_within_user"] = (
        (df["generated_datetime"] - per_user_t0).dt.total_seconds() / 3600.0
    ).astype(float)
    return df


# ---------------------------------------------------------------------------
# FWL fast-path primitives (identical algebra to OASST2 100k-perm script).
# Here `a_d` is the within-user-demeaned if_chosen covariate (controls for
# chosen-vs-rejected score asymmetry that would otherwise leak into the time
# slope via composition shift).
# ---------------------------------------------------------------------------
def prep_fwl(df: pd.DataFrame, time_col: str, exog_col: str = "if_chosen"):
    user_codes = df["user_id"].astype("category").cat.codes.to_numpy()
    n_groups = int(user_codes.max()) + 1
    group_count = np.bincount(user_codes, minlength=n_groups).astype(float)

    y = df["score_user"].to_numpy(dtype=float)
    x = df[time_col].to_numpy(dtype=float)
    a = df[exog_col].to_numpy(dtype=float)

    q_mean = np.bincount(user_codes, weights=y, minlength=n_groups) / group_count
    a_mean = np.bincount(user_codes, weights=a, minlength=n_groups) / group_count

    y_d = y - q_mean[user_codes]
    a_d = a - a_mean[user_codes]
    aa = float(a_d @ a_d)
    if aa < 1e-12:
        # if_chosen is constant within every user (shouldn't happen on PRISM) -> skip FWL
        y_t = y_d
        a_d = None
        aa = None
    else:
        ya = float(a_d @ y_d)
        y_t = y_d - (ya / aa) * a_d

    # pre-sorted within-group layout for fast shuffle
    order = np.argsort(user_codes, kind="stable")
    inv_order = np.argsort(order)
    user_codes_sorted = user_codes[order]
    _, first_idx = np.unique(user_codes_sorted, return_index=True)
    first_idx = np.sort(first_idx)
    boundaries = np.concatenate([first_idx, [len(user_codes_sorted)]]).astype(np.int64)
    x_sorted = x[order].copy()

    return dict(
        y_t=y_t, a_d=a_d, aa=aa,
        x_sorted=x_sorted, boundaries=boundaries, inv_order=inv_order,
        user_codes=user_codes, group_count=group_count, n_groups=n_groups,
    )


def fwl_coef(x: np.ndarray, ctx: dict) -> float:
    user_codes = ctx["user_codes"]
    group_count = ctx["group_count"]
    a_d = ctx["a_d"]
    aa = ctx["aa"]
    y_t = ctx["y_t"]
    n_groups = ctx["n_groups"]
    m_sum = np.bincount(user_codes, weights=x, minlength=n_groups)
    x_d = x - (m_sum / group_count)[user_codes]
    if a_d is not None:
        xa = float(a_d @ x_d)
        x_t = x_d - (xa / aa) * a_d
    else:
        x_t = x_d
    denom = float(x_t @ x_t)
    if denom < 1e-18:
        return float("nan")
    return float((x_t @ y_t) / denom)


def permute_batch(ctx: dict, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x_sorted = ctx["x_sorted"]
    boundaries = ctx["boundaries"]
    inv_order = ctx["inv_order"]
    user_codes = ctx["user_codes"]
    group_count = ctx["group_count"]
    a_d = ctx["a_d"]
    aa = ctx["aa"]
    y_t = ctx["y_t"]
    n_groups = ctx["n_groups"]
    out = np.empty(n, dtype=np.float64)
    local = x_sorted.copy()
    for i in range(n):
        for j in range(len(boundaries) - 1):
            lo, hi = boundaries[j], boundaries[j + 1]
            if hi - lo > 1:
                rng.shuffle(local[lo:hi])
        x = local[inv_order]
        m_sum = np.bincount(user_codes, weights=x, minlength=n_groups)
        x_d = x - (m_sum / group_count)[user_codes]
        if a_d is not None:
            xa = float(a_d @ x_d)
            x_t = x_d - (xa / aa) * a_d
        else:
            x_t = x_d
        denom = float(x_t @ x_t)
        out[i] = float((x_t @ y_t) / denom) if denom > 1e-18 else np.nan
    return out


def cluster_bootstrap(df: pd.DataFrame, time_col: str, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    authors = df["user_id"].unique()
    n_authors = len(authors)
    row_idx_by_author = {a: np.where(df["user_id"].values == a)[0] for a in authors}

    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        samp = authors[rng.integers(0, n_authors, size=n_authors)]
        all_idx = np.concatenate([row_idx_by_author[a] for a in samp])
        sub = df.iloc[all_idx].reset_index(drop=True)
        ctx = prep_fwl(sub, time_col=time_col)
        boot[b] = fwl_coef(sub[time_col].to_numpy(dtype=float), ctx)
    return boot


def bca_ci(boot: np.ndarray, theta_hat: float, alpha: float = 0.05):
    from scipy.stats import norm
    boot = np.asarray(boot, dtype=float)
    boot = boot[np.isfinite(boot)]
    if len(boot) < 10:
        return float("nan"), float("nan"), {"z0": None, "a_hat": None}
    boot_sorted = np.sort(boot)
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
    return (
        float(np.quantile(boot_sorted, aL)),
        float(np.quantile(boot_sorted, aU)),
        {"z0": float(z0), "a_hat": float(a), "alpha_L": aL, "alpha_U": aU},
    )


def run_naive_ols(df: pd.DataFrame, time_col: str, bin_unit: float = 1.0):
    """Bin-average then OLS on bins. For day_num bin_unit=1 (per-day mean);
    for hour_within_user bin_unit=0.25 (15-min buckets)."""
    bin_ = np.floor(df[time_col].to_numpy(float) / bin_unit).astype(int)
    tmp = pd.DataFrame({"b": bin_, "y": df["score_user"].to_numpy(float)})
    daily = tmp.groupby("b").agg(y=("y", "mean"), n=("y", "size")).reset_index()
    if len(daily) < 3:
        return float("nan"), float("nan")
    x = daily["b"].to_numpy(float) * bin_unit
    y = daily["y"].to_numpy(float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se2 = resid @ resid / max(len(x) - 2, 1)
    var = se2 * np.linalg.inv(X.T @ X)[1, 1]
    tstat = beta[1] / np.sqrt(max(var, 1e-30))
    from scipy.stats import t as tdist
    p = 2 * (1 - tdist.cdf(abs(tstat), df=max(len(x) - 2, 1)))
    return float(beta[1]), float(p)


# ---------------------------------------------------------------------------
# Per-axis driver
# ---------------------------------------------------------------------------
def run_axis(df: pd.DataFrame, axis_name: str, time_col: str,
             n_perm: int, n_boot: int, seed: int, naive_bin: float,
             min_span: float = 0.0, n_mixedlm_check: int = 200) -> dict:
    sub = df[df.groupby("user_id")[time_col].transform(lambda s: s.max() - s.min()) >= min_span].copy()
    sub = sub.reset_index(drop=True)
    print(f"\n=== axis={axis_name}  time_col={time_col}  min_span={min_span} ===")
    print(f"[data] users={sub['user_id'].nunique()} rows={len(sub)} "
          f"{time_col} range {sub[time_col].min():.3f}..{sub[time_col].max():.3f}")

    # Primary MixedLM
    t = time.time()
    md = smf.mixedlm(f"score_user ~ {time_col} + if_chosen",
                     data=sub, groups=sub["user_id"])
    res = md.fit(method="lbfgs", maxiter=200, disp=False)
    coef_obs = float(res.params[time_col])
    wald_p = float(res.pvalues[time_col])
    llf = float(res.llf)
    conv = bool(res.converged)
    print(f"[mixedlm] beta_{time_col} = {coef_obs:+.6e}  Wald p = {wald_p:.3e}  "
          f"converged={conv} ll={llf:.1f}  elapsed={time.time()-t:.1f}s")

    # FWL equivalence sanity on observed data
    ctx = prep_fwl(sub, time_col=time_col)
    coef_fwl = fwl_coef(sub[time_col].to_numpy(dtype=float), ctx)
    rel_err = abs(coef_fwl - coef_obs) / max(abs(coef_obs), 1e-18)
    print(f"[sanity] FWL beta = {coef_fwl:+.6e}  rel_err vs MixedLM = {rel_err:.2e}")

    # FWL vs MixedLM random-perm correlation (tail agreement proxy)
    mixedlm_corr = None
    exc_mix = exc_fwl = None
    if n_mixedlm_check > 0:
        rng_check = np.random.default_rng(seed + 12345)
        df_chk = sub.copy()
        mix_c, fwl_c = [], []
        t = time.time()
        for _ in range(n_mixedlm_check):
            arr = df_chk.groupby("user_id", sort=False)[time_col].transform(
                lambda s: pd.Series(rng_check.permutation(s.to_numpy()), index=s.index)
            )
            df_chk["_perm"] = arr
            try:
                r = smf.mixedlm(f"score_user ~ _perm + if_chosen",
                                data=df_chk, groups=df_chk["user_id"]).fit(
                    method="lbfgs", maxiter=80, disp=False)
                mix_c.append(float(r.params["_perm"]))
            except Exception:
                mix_c.append(np.nan)
            fwl_c.append(fwl_coef(df_chk["_perm"].to_numpy(dtype=float), ctx))
        mix_c = np.array(mix_c); fwl_c = np.array(fwl_c)
        mask = np.isfinite(mix_c) & np.isfinite(fwl_c)
        mixedlm_corr = float(np.corrcoef(mix_c[mask], fwl_c[mask])[0, 1])
        exc_mix = int(np.sum(np.abs(mix_c[mask]) >= abs(coef_obs)))
        exc_fwl = int(np.sum(np.abs(fwl_c[mask]) >= abs(coef_obs)))
        print(f"[sanity] MixedLM vs FWL on {mask.sum()} perms: "
              f"corr={mixedlm_corr:.4f}  exceed_mix={exc_mix} exceed_fwl={exc_fwl} "
              f"elapsed={time.time()-t:.1f}s")

    # Within-user permutation null
    t = time.time()
    null_coefs = permute_batch(ctx, n_perm, seed + 777_000)
    finite = np.isfinite(null_coefs)
    exceed = int((np.abs(null_coefs[finite]) >= abs(coef_obs)).sum())
    perm_p_emp = (exceed / finite.sum()) if finite.sum() else float("nan")
    perm_p_ps = (exceed + 1) / (finite.sum() + 1)
    print(f"[perm] {n_perm} permutations  exceed={exceed}  "
          f"p_empirical={perm_p_emp:.4f}  p_Phipson-Smyth={perm_p_ps:.4f}  "
          f"null mean={null_coefs[finite].mean():+.3e} std={null_coefs[finite].std():.3e}  "
          f"elapsed={time.time()-t:.1f}s")

    # Cluster-bootstrap BCa CI
    t = time.time()
    boot = cluster_bootstrap(sub, time_col, n_boot=n_boot, seed=seed + 999)
    boot_f = boot[np.isfinite(boot)]
    ci_lo_pct = float(np.quantile(boot_f, 0.025))
    ci_hi_pct = float(np.quantile(boot_f, 0.975))
    ci_lo_bca, ci_hi_bca, bca_diag = bca_ci(boot_f, theta_hat=coef_obs)
    print(f"[boot] B={n_boot}  percentile CI [{ci_lo_pct:+.4e}, {ci_hi_pct:+.4e}]  "
          f"BCa CI [{ci_lo_bca:+.4e}, {ci_hi_bca:+.4e}]  "
          f"z0={bca_diag['z0']}  a={bca_diag['a_hat']}  "
          f"elapsed={time.time()-t:.1f}s")

    # Naive aggregate OLS
    naive_beta, naive_p = run_naive_ols(sub, time_col=time_col, bin_unit=naive_bin)
    print(f"[naive] bin={naive_bin}  beta={naive_beta:+.4e}  p={naive_p:.4g}")

    # Per-user random intercept snapshot (MixedLM random-effects dict).
    # Can fail with singular cov_re on very within-user-variance-heavy fits;
    # fall back to per-user OLS-demeaned intercept in that case.
    try:
        re_dict = res.random_effects
        per_user_re = pd.DataFrame({
            "user_id": list(re_dict.keys()),
            "random_intercept": [float(v.iloc[0]) for v in re_dict.values()],
        })
        per_user_re["source"] = "mixedlm_BLUP"
    except Exception as e:
        print(f"[re] MixedLM random_effects extraction failed ({e}); "
              f"using per-user mean-centered intercept as fallback")
        per_user_grp = sub.groupby("user_id")["score_user"].mean()
        global_mean = float(sub["score_user"].mean())
        per_user_re = pd.DataFrame({
            "user_id": per_user_grp.index.tolist(),
            "random_intercept": (per_user_grp.values - global_mean).astype(float),
            "source": "ols_demean_fallback",
        })

    # BCa verdict: straddles zero?
    straddles = (ci_lo_bca <= 0.0 <= ci_hi_bca)

    return {
        "axis": axis_name,
        "time_col": time_col,
        "n_users": int(sub["user_id"].nunique()),
        "n_rows": int(len(sub)),
        "time_range": [float(sub[time_col].min()), float(sub[time_col].max())],
        "primary_mixedlm": {
            "beta": coef_obs, "wald_p": wald_p, "loglik": llf, "converged": conv,
            "window_drift_total": coef_obs * (sub[time_col].max() - sub[time_col].min()),
        },
        "fwl_equivalence": {"beta_fwl": coef_fwl, "rel_err": rel_err,
                            "mixedlm_fwl_corr": mixedlm_corr,
                            "exceed_mixedlm": exc_mix, "exceed_fwl": exc_fwl},
        "permutation_test": {
            "n_permutations": int(n_perm),
            "n_finite": int(finite.sum()),
            "exceedance": exceed,
            "p_empirical": perm_p_emp,
            "p_phipson_smyth": perm_p_ps,
            "null_mean": float(null_coefs[finite].mean()),
            "null_std": float(null_coefs[finite].std()),
        },
        "cluster_bootstrap": {
            "n_bootstrap": int(n_boot),
            "n_finite": int(len(boot_f)),
            "boot_mean": float(boot_f.mean()),
            "boot_std": float(boot_f.std()),
            "percentile_ci_95": [ci_lo_pct, ci_hi_pct],
            "bca_ci_95": [ci_lo_bca, ci_hi_bca],
            "bca_diagnostics": bca_diag,
            "bca_straddles_zero": bool(straddles),
        },
        "naive_ols": {"bin_unit": naive_bin, "beta": naive_beta, "p": naive_p},
        "verdict": ("H3a_REJECTED"
                    if perm_p_ps < 0.05 and not straddles
                    else ("H3a_WALD_ONLY" if wald_p < 0.05 and straddles
                          else "H3a_NOT_REJECTED")),
        "_null_coefs": null_coefs,
        "_boot_betas": boot,
        "_per_user_re": per_user_re,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rm-parquet",
                    default="../1_Causal_RLHF/data/prism_rm_scored.parquet")
    ap.add_argument("--ts-parquet",
                    default="../1_Causal_RLHF/data/prism_conversation_timestamps.parquet")
    ap.add_argument("--n-permutations", type=int, default=2000)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="results/track3_prism_temporal")
    ap.add_argument("--mixedlm-check-k", type=int, default=100)
    ap.add_argument("--skip-hour", action="store_true")
    ap.add_argument("--skip-day", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_prism(Path(args.rm_parquet), Path(args.ts_parquet))
    print(f"[data] merged: {len(df)} utterances, {df['user_id'].nunique()} users")
    print(f"[data] day range: {df['day_num'].min():.2f}..{df['day_num'].max():.2f}")
    print(f"[data] hour_within_user range: "
          f"{df['hour_within_user'].min():.3f}..{df['hour_within_user'].max():.2f}")

    axes = []
    if not args.skip_day:
        axes.append(("day", "day_num", 1.0, 0.0))
    if not args.skip_hour:
        # filter to users with span >= 0.25 h so the axis has non-zero variance
        axes.append(("hour_within_user", "hour_within_user", 0.25, 0.25))

    all_reports = {}
    for (name, col, naive_bin, min_span) in axes:
        rpt = run_axis(
            df, axis_name=name, time_col=col,
            n_perm=args.n_permutations, n_boot=args.n_bootstrap,
            seed=args.seed, naive_bin=naive_bin, min_span=min_span,
            n_mixedlm_check=args.mixedlm_check_k,
        )
        # Persist heavy arrays separately (not in JSON).
        np.save(out_dir / f"null_coefs_{name}.npy", rpt.pop("_null_coefs"))
        np.save(out_dir / f"bootstrap_betas_{name}.npy", rpt.pop("_boot_betas"))
        rpt.pop("_per_user_re").to_parquet(
            out_dir / f"per_user_random_intercepts_{name}.parquet", index=False)
        all_reports[name] = rpt

    summary = {
        "dataset": "PRISM (Kirk et al. 2024)  per-user score_user + conversation timestamps",
        "n_users_total": int(df["user_id"].nunique()),
        "n_utterances_total": int(len(df)),
        "collection_window_days": float(df["day_num"].max() - df["day_num"].min()),
        "axes": all_reports,
        "notes": (
            "PRISM per-user span is short (median ~43 min). The 'day' axis is identified "
            "off the 348 users with span >= 1 day. The 'hour_within_user' axis is "
            "identified off 1329 users with span >= 15 min. A NULL on either is the "
            "honest expected outcome given the Kalman-filter negative result for "
            "time-varying PRISM calibration (Appx H.4)."
        ),
    }

    # Strict-JSON
    def _s(o):
        if isinstance(o, dict): return {k: _s(v) for k, v in o.items()}
        if isinstance(o, list): return [_s(v) for v in o]
        if isinstance(o, float):
            if np.isinf(o): return "Infinity" if o > 0 else "-Infinity"
            if np.isnan(o): return None
        return o

    (out_dir / "summary.json").write_text(json.dumps(_s(summary), indent=2))
    print(f"\n[save] {out_dir/'summary.json'}")
    for name, rpt in all_reports.items():
        print(f"[verdict] axis={name}  beta={rpt['primary_mixedlm']['beta']:+.3e} "
              f"perm p_PS={rpt['permutation_test']['p_phipson_smyth']:.3g} "
              f"BCa 95% CI=[{rpt['cluster_bootstrap']['bca_ci_95'][0]:+.3e}, "
              f"{rpt['cluster_bootstrap']['bca_ci_95'][1]:+.3e}]  "
              f"straddles_zero={rpt['cluster_bootstrap']['bca_straddles_zero']}  "
              f"VERDICT={rpt['verdict']}")


if __name__ == "__main__":
    main()
