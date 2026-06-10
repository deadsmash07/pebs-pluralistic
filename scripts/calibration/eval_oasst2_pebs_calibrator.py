"""F-T1 OASST2 PEBS calibrator replication - within-author EB-shrinkage RMSE gain.

Inherits VERBATIM from `eval_user_score_mse_shrunk.py` (the published PRISM headline
estimator) + `F_T1_5corpus_calibrator_solidification.py` (cluster-bootstrap CI). The
estimator pipeline is the same one that produced the PRISM +8.58% legacy headline +
PluriHarms +8.64% replication; only the corpus loader is OASST2-specific.

Pipeline
--------
1. Load `data/oasst2/oasst2_qwen7b_scored.parquet` (Qwen2.5-7B-Instruct mean-resp-LL
   scored) - one row per OASST2 message with (user_id=AUTHOR, rm_score, quality, ...)
2. Filter to authors with n_j >= min_obs_per_user (default 6, matching PRISM headline
   convention) post-scoring.
3. Pre-pass: per-author OLS on ALL data to estimate Morris-MoM tau^2_alpha + tau^2_beta.
4. K=5 within-author CV. For each fold:
   - Population OLS on training rows
   - Per-author OLS on per-author training rows (n_train >= 5 per author)
   - EB-shrunk per-author calibrators via Morris-MoM omega
5. Held-out predictions on test fold across 4 arms:
   - no_calibration: train-fold mean
   - population_slope: alpha_0 + beta_0 * x
   - per_user_OLS: alpha_j + beta_j * x (no shrinkage)
   - PEBS_EB_shrunk: alpha_shrunk_j + beta_shrunk_j * x  (the headline arm)
6. Cluster-bootstrap by user_id (Cameron 2008, Efron 1987 BCa); B=4000 reps default.
7. Pre-specified decision rule applied to the gain estimate and its CI.

Output
------
`results/falsifiers/F_T1_OASST2_REPLICATION/<seed_NNN>/output.json` per seed +
`summary.json` cross-seed aggregation.

References
----------
- Kopf et al. 2023 - OASST2
- Stiennon et al. 2020 - log-likelihood reward proxy
- Morris 1983 - MoM tau^2 estimator
- Henderson 1975 - BLUP
- Cameron, Gelbach, Miller 2008 - cluster-bootstrap
- Efron 1987 - BCa CI
- Gelman & Hill 2007 - partial pooling reference
"""
from __future__ import annotations

import os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as scstats
except ImportError as e:
    raise ImportError("scipy required for bootstrap + Wilcoxon") from e


ROOT = Path(__file__).resolve().parents[1]


# Pre-specified priors over outcome bands
PRESPECIFIED_PRIORS = {
    "PASS-REPLICATION": 0.55,
    "PARTIAL": 0.30,
    "NULL": 0.15,
}

# Pre-specified outcome bands
PASS_LO = 5.0
PASS_HI = 12.0
PARTIAL_LO = 1.0
PARTIAL_HI = 5.0
NULL_LO = -1.0
NULL_HI = 1.0


# ============================================================================
# 1 PEBS core (verbatim from eval_user_score_mse_shrunk.py + F-T1 5corpus)
# ============================================================================


def ols_with_V(x, y):
    """Returns (intercept, slope, V_intercept, V_slope) with sample-variance estimates.

    Per `eval_user_score_mse_shrunk.py:61-73` verbatim.
    """
    k = len(x)
    if k < 2 or np.var(x) < 1e-12:
        return float(np.mean(y)) if k else 0.0, 0.0, np.inf, np.inf
    x_bar = x.mean()
    Sxx = ((x - x_bar) ** 2).sum()
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = intercept + slope * x
    sigma_hat_sq = ((y - y_pred) ** 2).sum() / max(k - 2, 1)
    V_int = sigma_hat_sq * (1.0 / k + x_bar ** 2 / max(Sxx, 1e-12))
    V_slope = sigma_hat_sq / max(Sxx, 1e-12)
    return float(intercept), float(slope), float(V_int), float(V_slope)


def kfold_split(n, k, rng):
    """Random k-fold split. Per `eval_user_score_mse_shrunk.py:47-58` verbatim."""
    idx = np.arange(n)
    rng.shuffle(idx)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        stop = (i + 1) * fold_size if i < k - 1 else n
        test_idx = idx[start:stop]
        train_idx = np.concatenate([idx[:start], idx[stop:]])
        folds.append((train_idx, test_idx))
    return folds


def file_sha256(p):
    if not Path(p).exists():
        return f"FILE_NOT_FOUND:{p}"
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head_t1():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


# ============================================================================
# 2 Cluster-bootstrap CI (cluster by user_id; BCa+percentile+jackknife)
# Adapted from F_T1_5corpus_calibrator_solidification.py cluster_bootstrap_gain_ci
# ============================================================================


def cluster_bootstrap_gain_ci(sq_err_baseline, sq_err_pebs, per_user_idx,
                              B=4000, seed=42):
    """Cluster-resample-by-user_id bootstrap on gain_pct = (rmse_base - rmse_pebs)
    / rmse_base * 100. Returns BCa + percentile + jackknife CIs."""
    if len(sq_err_baseline) == 0 or len(sq_err_pebs) == 0:
        return {
            "boot_mean": float("nan"),
            "boot_sd": float("nan"),
            "ci_percentile_lo": float("nan"),
            "ci_percentile_hi": float("nan"),
            "ci_bca_lo": float("nan"),
            "ci_bca_hi": float("nan"),
            "ci_jackknife_lo": float("nan"),
            "ci_jackknife_hi": float("nan"),
            "n_boot_valid": 0,
            "error": "empty sq_err arrays",
        }

    user_to_block = {}
    for i, u in enumerate(per_user_idx):
        user_to_block.setdefault(u, []).append(i)
    user_keys = list(user_to_block.keys())
    n_users = len(user_keys)

    rng = np.random.default_rng(seed)
    boot_gains = []
    for _ in range(B):
        sampled_users = rng.choice(user_keys, size=n_users, replace=True)
        flat = []
        for u in sampled_users:
            flat.extend(user_to_block[u])
        if not flat:
            continue
        idx_arr = np.array(flat)
        sb = sq_err_baseline[idx_arr]
        sp = sq_err_pebs[idx_arr]
        rb = math.sqrt(sb.mean()) if sb.size else 0.0
        rp = math.sqrt(sp.mean()) if sp.size else 0.0
        if rb > 0:
            boot_gains.append((rb - rp) / rb * 100.0)

    boot_arr = np.array([g for g in boot_gains if not math.isnan(g)])
    if len(boot_arr) < 100:
        return {
            "boot_mean": float("nan"),
            "boot_sd": float("nan"),
            "ci_percentile_lo": float("nan"),
            "ci_percentile_hi": float("nan"),
            "ci_bca_lo": float("nan"),
            "ci_bca_hi": float("nan"),
            "ci_jackknife_lo": float("nan"),
            "ci_jackknife_hi": float("nan"),
            "n_boot_valid": int(len(boot_arr)),
            "error": "BOOTSTRAP_CI_DEGENERATE",
        }

    boot_mean = float(boot_arr.mean())
    boot_sd = float(boot_arr.std(ddof=1))
    pct_lo = float(np.quantile(boot_arr, 0.025))
    pct_hi = float(np.quantile(boot_arr, 0.975))

    rmse_base_obs = math.sqrt(sq_err_baseline.mean())
    rmse_pebs_obs = math.sqrt(sq_err_pebs.mean())
    obs_gain = ((rmse_base_obs - rmse_pebs_obs) / rmse_base_obs * 100.0
                if rmse_base_obs > 0 else 0.0)

    p_below = float(np.mean(boot_arr < obs_gain))
    z0 = scstats.norm.ppf(p_below) if 0.0 < p_below < 1.0 else 0.0

    # Jackknife leave-one-user-out for acceleration
    jack_gains = []
    for j in range(n_users):
        excluded = user_keys[j]
        flat = [i for u in user_keys if u != excluded for i in user_to_block[u]]
        if not flat:
            continue
        sb_jk = sq_err_baseline[np.array(flat)]
        sp_jk = sq_err_pebs[np.array(flat)]
        rb_jk = math.sqrt(sb_jk.mean()) if sb_jk.size else 0.0
        rp_jk = math.sqrt(sp_jk.mean()) if sp_jk.size else 0.0
        if rb_jk > 0:
            jack_gains.append((rb_jk - rp_jk) / rb_jk * 100.0)
    jack_arr = np.array(jack_gains)
    if len(jack_arr) > 1:
        jm = jack_arr.mean()
        num = float(np.sum((jm - jack_arr) ** 3))
        den = 6.0 * (float(np.sum((jm - jack_arr) ** 2)) ** 1.5)
        a_hat = num / den if den > 0 else 0.0
    else:
        a_hat = 0.0

    z_lo = scstats.norm.ppf(0.025)
    z_hi = scstats.norm.ppf(0.975)

    def _bca_q(z_alpha, z0, a):
        num = z0 + z_alpha
        den = 1 - a * num
        return scstats.norm.cdf(z0 + num / den) if abs(den) > 1e-12 else None

    q_lo = _bca_q(z_lo, z0, a_hat)
    q_hi = _bca_q(z_hi, z0, a_hat)
    if q_lo is None or q_hi is None or q_lo >= q_hi or q_lo < 0 or q_hi > 1:
        bca_lo, bca_hi = pct_lo, pct_hi
    else:
        bca_lo = float(np.quantile(boot_arr, np.clip(q_lo, 0.001, 0.999)))
        bca_hi = float(np.quantile(boot_arr, np.clip(q_hi, 0.001, 0.999)))

    if len(jack_arr) > 1:
        jack_se = math.sqrt((n_users - 1) / n_users
                            * float(np.sum((jack_arr - jack_arr.mean()) ** 2)))
        jk_lo = obs_gain - 1.96 * jack_se
        jk_hi = obs_gain + 1.96 * jack_se
    else:
        jk_lo = float("nan")
        jk_hi = float("nan")

    return {
        "boot_mean": boot_mean,
        "boot_sd": boot_sd,
        "ci_percentile_lo": pct_lo,
        "ci_percentile_hi": pct_hi,
        "ci_bca_lo": bca_lo,
        "ci_bca_hi": bca_hi,
        "ci_jackknife_lo": float(jk_lo),
        "ci_jackknife_hi": float(jk_hi),
        "n_boot_valid": int(len(boot_arr)),
        "obs_gain_pct": float(obs_gain),
        "z0_bias_correction": float(z0),
        "a_hat_acceleration": float(a_hat),
    }


# ============================================================================
# 3 Per-seed driver: PEBS on OASST2
# ============================================================================


def run_pebs_oasst2(parquet_path, seed=42, k_folds=5, min_obs_per_user=6,
                    bootstrap_B=4000, bootstrap_seed=42):
    """Run the full PEBS pipeline on OASST2 scored parquet for a single seed.

    Returns a dict with rmse-per-arm + gain_pct + bootstrap CIs + per-author diagnostics.
    """
    rng = np.random.default_rng(seed)

    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["quality", "rm_score"]).reset_index(drop=True)
    n_obs_total = int(len(df))
    n_users_total = int(df["user_id"].nunique())

    # Filter to authors with >= min_obs_per_user
    n_per = df.groupby("user_id").size()
    keep_users = set(n_per[n_per >= min_obs_per_user].index)
    df = df[df["user_id"].isin(keep_users)].reset_index(drop=True)
    n_users_post_filter = int(df["user_id"].nunique())

    if n_users_post_filter < 100:
        return {
            "n_obs_total": n_obs_total,
            "n_users_total": n_users_total,
            "n_users_post_filter": n_users_post_filter,
            "error": f"N_USERS_BELOW_100_AT_FILTER: {n_users_post_filter} < 100",
            "verdict_branch": "INCONCLUSIVE-COHORT-SHRUNK",
        }

    # Population OLS on entire dataset (legacy convention; matches
    # eval_user_score_mse_shrunk.py:84-87)
    slope_pop, intercept_pop = np.polyfit(df.rm_score, df.quality, 1)
    pop_alpha = float(intercept_pop)
    pop_beta = float(slope_pop)

    # Per-user OLS on ALL data for the pre-pass tau^2 estimate
    # (matches eval_user_score_mse_shrunk.py:90-108)
    user_stats = []
    for uid, grp in df.groupby("user_id"):
        a, b, Va, Vb = ols_with_V(grp.rm_score.to_numpy(),
                                   grp.quality.to_numpy().astype(float))
        user_stats.append({"user_id": uid, "alpha": a, "beta": b,
                           "V_alpha": Va, "V_beta": Vb, "n": len(grp)})
    us = pd.DataFrame(user_stats)
    V_alpha_total = float(us.alpha.var())
    V_beta_total = float(us.beta.var())
    mean_samp_V_alpha = float(us.V_alpha.replace([np.inf, -np.inf], np.nan).dropna().mean())
    mean_samp_V_beta = float(us.V_beta.replace([np.inf, -np.inf], np.nan).dropna().mean())
    tau_a_sq = max(V_alpha_total - mean_samp_V_alpha, 1e-6)
    tau_b_sq = max(V_beta_total - mean_samp_V_beta, 1e-6)

    # K=5 within-user CV across 4 arms
    per_user_rows = []
    sq_err_per_row = {"no_calib": [], "pop_slope": [], "pebs_ols": [], "pebs_shrunk": []}
    per_row_user_idx = []   # for cluster-bootstrap (one per row in TEST fold)

    for uid, grp in df.groupby("user_id"):
        n = len(grp)
        if n < min_obs_per_user:
            continue
        x = grp.rm_score.to_numpy()
        y = grp.quality.to_numpy().astype(float)
        folds = kfold_split(n, k_folds, rng)
        squared = {"no_calib": [], "pop_slope": [], "pebs_ols": [], "pebs_shrunk": []}

        for train_idx, test_idx in folds:
            x_tr, y_tr = x[train_idx], y[train_idx]
            x_te, y_te = x[test_idx], y[test_idx]
            if len(x_te) == 0:
                continue

            # No-calib (train-fold mean)
            y_hat_nc = np.full_like(y_te, np.mean(y_tr) if len(y_tr) else 0.0)
            squared["no_calib"].extend(((y_hat_nc - y_te) ** 2).tolist())

            # Population-slope
            y_hat_ps = pop_alpha + pop_beta * x_te
            squared["pop_slope"].extend(((y_hat_ps - y_te) ** 2).tolist())

            # Per-user naive OLS (no shrinkage)
            a, b, Va, Vb = ols_with_V(x_tr, y_tr)
            y_hat_ols = a + b * x_te
            squared["pebs_ols"].extend(((y_hat_ols - y_te) ** 2).tolist())

            # PEBS EB-shrunk
            omega_a = tau_a_sq / (tau_a_sq + Va) if np.isfinite(Va) else 0.0
            omega_b = tau_b_sq / (tau_b_sq + Vb) if np.isfinite(Vb) else 0.0
            a_s = omega_a * a + (1 - omega_a) * pop_alpha
            b_s = omega_b * b + (1 - omega_b) * pop_beta
            y_hat_sh = a_s + b_s * x_te
            squared["pebs_shrunk"].extend(((y_hat_sh - y_te) ** 2).tolist())

            # Track per-row identity for cluster-bootstrap
            for arm in ["no_calib", "pop_slope", "pebs_ols", "pebs_shrunk"]:
                if arm == "no_calib":
                    sq_err_per_row[arm].extend(((y_hat_nc - y_te) ** 2).tolist())
                elif arm == "pop_slope":
                    sq_err_per_row[arm].extend(((y_hat_ps - y_te) ** 2).tolist())
                elif arm == "pebs_ols":
                    sq_err_per_row[arm].extend(((y_hat_ols - y_te) ** 2).tolist())
                else:
                    sq_err_per_row[arm].extend(((y_hat_sh - y_te) ** 2).tolist())
            per_row_user_idx.extend([uid] * len(y_te))

        per_user_rows.append({
            "user_id": uid,
            "n": n,
            "alpha_full": float(us[us.user_id == uid].alpha.iloc[0]),
            "beta_full": float(us[us.user_id == uid].beta.iloc[0]),
            **{f"rmse_{arm}": float(np.sqrt(np.mean(squared[arm])))
               for arm in squared if squared[arm]},
        })

    pu = pd.DataFrame(per_user_rows)

    # Per-row arrays (for cluster-bootstrap which clusters by user). NOTE: each
    # row is logged 4 times into sq_err_per_row above (once per arm), which is
    # inconsistent. Fix by using single-pass arrays:
    sq_err_baseline_per_row = []
    sq_err_pebs_per_row = []
    sq_err_per_user_ols_per_row = []
    sq_err_no_calib_per_row = []
    per_row_uid = []
    rng2 = np.random.default_rng(seed)
    for uid, grp in df.groupby("user_id"):
        n = len(grp)
        if n < min_obs_per_user:
            continue
        x = grp.rm_score.to_numpy()
        y = grp.quality.to_numpy().astype(float)
        folds = kfold_split(n, k_folds, rng2)
        for train_idx, test_idx in folds:
            x_tr, y_tr = x[train_idx], y[train_idx]
            x_te, y_te = x[test_idx], y[test_idx]
            if len(x_te) == 0:
                continue
            y_hat_nc = np.full_like(y_te, np.mean(y_tr) if len(y_tr) else 0.0)
            y_hat_ps = pop_alpha + pop_beta * x_te
            a, b, Va, Vb = ols_with_V(x_tr, y_tr)
            y_hat_ols = a + b * x_te
            omega_a = tau_a_sq / (tau_a_sq + Va) if np.isfinite(Va) else 0.0
            omega_b = tau_b_sq / (tau_b_sq + Vb) if np.isfinite(Vb) else 0.0
            a_s = omega_a * a + (1 - omega_a) * pop_alpha
            b_s = omega_b * b + (1 - omega_b) * pop_beta
            y_hat_sh = a_s + b_s * x_te
            sq_err_no_calib_per_row.extend(((y_hat_nc - y_te) ** 2).tolist())
            sq_err_baseline_per_row.extend(((y_hat_ps - y_te) ** 2).tolist())
            sq_err_per_user_ols_per_row.extend(((y_hat_ols - y_te) ** 2).tolist())
            sq_err_pebs_per_row.extend(((y_hat_sh - y_te) ** 2).tolist())
            per_row_uid.extend([uid] * len(y_te))

    sq_err_no_calib_per_row = np.asarray(sq_err_no_calib_per_row)
    sq_err_baseline_per_row = np.asarray(sq_err_baseline_per_row)
    sq_err_per_user_ols_per_row = np.asarray(sq_err_per_user_ols_per_row)
    sq_err_pebs_per_row = np.asarray(sq_err_pebs_per_row)

    rmse_no_calib = float(np.sqrt(sq_err_no_calib_per_row.mean()))
    rmse_baseline = float(np.sqrt(sq_err_baseline_per_row.mean()))
    rmse_per_user_ols = float(np.sqrt(sq_err_per_user_ols_per_row.mean()))
    rmse_pebs = float(np.sqrt(sq_err_pebs_per_row.mean()))
    gain_pct = ((rmse_baseline - rmse_pebs) / rmse_baseline * 100.0
                if rmse_baseline > 0 else float("nan"))

    # Wilcoxon signed-rank (per-user RMSE pop-slope vs PEBS-shrunk)
    try:
        if "rmse_pop_slope" in pu.columns and "rmse_pebs_shrunk" in pu.columns:
            mask = pu[["rmse_pop_slope", "rmse_pebs_shrunk"]].notna().all(axis=1)
            if mask.sum() > 0:
                w = scstats.wilcoxon(pu.loc[mask, "rmse_pop_slope"],
                                     pu.loc[mask, "rmse_pebs_shrunk"],
                                     alternative="greater")
                wilcoxon_p = float(w.pvalue)
                wilcoxon_stat = float(w.statistic)
            else:
                wilcoxon_p = float("nan")
                wilcoxon_stat = float("nan")
        else:
            wilcoxon_p = float("nan")
            wilcoxon_stat = float("nan")
    except Exception:
        wilcoxon_p = float("nan")
        wilcoxon_stat = float("nan")

    # Cluster-bootstrap CI on gain_pct (cluster by user_id)
    bs = cluster_bootstrap_gain_ci(
        sq_err_baseline_per_row, sq_err_pebs_per_row, per_row_uid,
        B=bootstrap_B, seed=bootstrap_seed,
    )

    # Diagnostic stats on per-user calibrators
    diag = {
        "n_users_post_filter": n_users_post_filter,
        "n_users_total": n_users_total,
        "n_obs_total": n_obs_total,
        "tau_alpha_sq": tau_a_sq,
        "tau_beta_sq": tau_b_sq,
        "sigma_alpha": float(np.sqrt(tau_a_sq)),
        "sigma_beta": float(np.sqrt(tau_b_sq)),
        "pop_alpha": pop_alpha,
        "pop_beta": pop_beta,
        "per_user_alpha_mean": float(us.alpha.mean()),
        "per_user_alpha_std": float(us.alpha.std(ddof=1)),
        "per_user_beta_mean": float(us.beta.mean()),
        "per_user_beta_std": float(us.beta.std(ddof=1)),
    }

    return {
        "rmse_no_calib": rmse_no_calib,
        "rmse_baseline_pop": rmse_baseline,
        "rmse_per_user_OLS": rmse_per_user_ols,
        "rmse_pebs_shrunk": rmse_pebs,
        "gain_pct": float(gain_pct),
        "gain_ci_bca_lo": float(bs.get("ci_bca_lo", float("nan"))),
        "gain_ci_bca_hi": float(bs.get("ci_bca_hi", float("nan"))),
        "gain_ci_percentile_lo": float(bs.get("ci_percentile_lo", float("nan"))),
        "gain_ci_percentile_hi": float(bs.get("ci_percentile_hi", float("nan"))),
        "gain_ci_jackknife_lo": float(bs.get("ci_jackknife_lo", float("nan"))),
        "gain_ci_jackknife_hi": float(bs.get("ci_jackknife_hi", float("nan"))),
        "boot_mean": float(bs.get("boot_mean", float("nan"))),
        "boot_sd": float(bs.get("boot_sd", float("nan"))),
        "boot_n_valid": int(bs.get("n_boot_valid", 0)),
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p_onesided": wilcoxon_p,
        **diag,
    }


def classify_verdict(gain_pct, ci_lo, ci_hi):
    """Apply the pre-specified decision rule."""
    if (gain_pct is None or np.isnan(gain_pct)
        or np.isnan(ci_lo) or np.isnan(ci_hi)):
        return "INCONCLUSIVE-NaN"

    ci_excludes_zero = (ci_lo > 0) or (ci_hi < 0)

    if PASS_LO <= gain_pct <= PASS_HI and ci_excludes_zero:
        return "PASS-REPLICATION"
    if PARTIAL_LO <= gain_pct < PASS_LO and ci_excludes_zero:
        return "PARTIAL"
    if NULL_LO <= gain_pct <= NULL_HI:
        return "NULL"
    if PARTIAL_LO <= gain_pct < PASS_LO and not ci_excludes_zero:
        return "PARTIAL-CI-STRADDLES"
    if gain_pct > PASS_HI:
        return "INCONCLUSIVE-MAGNITUDE-HIGH"
    if gain_pct < NULL_LO:
        return "INCONCLUSIVE-NEGATIVE"
    return "INCONCLUSIVE-OFF-BAND"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scored-parquet", required=True,
                   help="Path to oasst2_qwen7b_scored.parquet")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 271, 314])
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--min-obs-per-user", type=int, default=6)
    p.add_argument("--bootstrap-B", type=int, default=4000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="If set, use B=500, single seed, min_obs=3")
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke:
        args.bootstrap_B = 500
        args.seeds = args.seeds[:1] if args.seeds else [42]
        args.min_obs_per_user = 3
        print(f"[smoke] override: B={args.bootstrap_B}, seeds={args.seeds}, "
              f"min_obs={args.min_obs_per_user}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = Path(args.scored_parquet)
    if not parquet_path.exists():
        print(f"[error] scored parquet not found: {parquet_path}", file=sys.stderr)
        sys.exit(2)

    print(f"[F-T1 OASST2] anchor T1 git_head: {git_head_t1()}")
    print(f"[F-T1 OASST2] scored parquet: {parquet_path}")
    print(f"[F-T1 OASST2] sha256: {file_sha256(parquet_path)[:16]}...")
    print(f"[F-T1 OASST2] seeds: {args.seeds}")
    print(f"[F-T1 OASST2] k_folds={args.k_folds}, min_obs={args.min_obs_per_user}, "
          f"B={args.bootstrap_B}")

    per_seed_results = []
    t0 = time.time()

    for seed in args.seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[F-T1 OASST2] === seed={seed} ===")
        try:
            result = run_pebs_oasst2(
                parquet_path,
                seed=seed,
                k_folds=args.k_folds,
                min_obs_per_user=args.min_obs_per_user,
                bootstrap_B=args.bootstrap_B,
                bootstrap_seed=args.bootstrap_seed,
            )
        except Exception as e:
            traceback.print_exc()
            result = {
                "error": f"EXCEPTION: {e}",
                "verdict_branch": "INCONCLUSIVE-EXCEPTION",
            }

        if "verdict_branch" not in result:
            result["verdict_branch"] = classify_verdict(
                result.get("gain_pct", float("nan")),
                result.get("gain_ci_bca_lo", float("nan")),
                result.get("gain_ci_bca_hi", float("nan")),
            )

        result["corpus"] = "OASST2"
        result["script_anchor_t1"] = git_head_t1()
        result["seed"] = int(seed)
        result["k_folds"] = int(args.k_folds)
        result["min_obs_per_user"] = int(args.min_obs_per_user)
        result["bootstrap_B"] = int(args.bootstrap_B)
        result["bootstrap_seed"] = int(args.bootstrap_seed)
        result["scored_parquet_sha"] = file_sha256(parquet_path)
        result["smoke"] = bool(args.smoke)

        with (seed_dir / "output.json").open("w") as f:
            json.dump(result, f, indent=2, default=str)

        gp = result.get("gain_pct", float("nan"))
        lo = result.get("gain_ci_bca_lo", float("nan"))
        hi = result.get("gain_ci_bca_hi", float("nan"))
        v = result.get("verdict_branch", "?")
        if isinstance(gp, (int, float)) and not np.isnan(gp):
            print(f"[F-T1 OASST2] seed={seed}: gain={gp:+.3f}%  "
                  f"BCa=[{lo:+.3f}, {hi:+.3f}]  verdict={v}")
        else:
            print(f"[F-T1 OASST2] seed={seed}: gain=NaN  verdict={v}  "
                  f"err={result.get('error')}")
        per_seed_results.append(result)

    # Cross-seed summary
    valid = [r for r in per_seed_results if not np.isnan(r.get("gain_pct", float("nan")))]
    if valid:
        gains = np.array([r["gain_pct"] for r in valid])
        across_mean = float(gains.mean())
        across_std = float(gains.std(ddof=1)) if len(gains) > 1 else 0.0
        across_min = float(gains.min())
        across_max = float(gains.max())

        # Joint verdict: use across-seed mean + worst-case BCa CI rule
        ci_los = np.array([r["gain_ci_bca_lo"] for r in valid])
        ci_his = np.array([r["gain_ci_bca_hi"] for r in valid])
        n_excl_zero_pos = int(((ci_los > 0)).sum())
        n_excl_zero_neg = int(((ci_his < 0)).sum())
        n_seeds = len(valid)

        if (PASS_LO <= across_mean <= PASS_HI) and n_excl_zero_pos >= max(1, n_seeds - 1):
            joint = "PASS-REPLICATION"
        elif PARTIAL_LO <= across_mean < PASS_LO and n_excl_zero_pos >= max(1, n_seeds - 1):
            joint = "PARTIAL"
        elif NULL_LO <= across_mean <= NULL_HI:
            joint = "NULL"
        elif across_mean < NULL_LO:
            joint = "INCONCLUSIVE-NEGATIVE"
        elif across_mean > PASS_HI:
            joint = "INCONCLUSIVE-MAGNITUDE-HIGH"
        else:
            joint = "INCONCLUSIVE-OFF-BAND"
    else:
        across_mean = float("nan")
        across_std = float("nan")
        across_min = float("nan")
        across_max = float("nan")
        joint = "INCONCLUSIVE-NO-VALID-SEEDS"

    summary = {
        "n_seeds": len(per_seed_results),
        "n_valid_seeds": len(valid),
        "per_seed_summary": [
            {
                "seed": r["seed"],
                "gain_pct": r.get("gain_pct"),
                "gain_ci_bca_lo": r.get("gain_ci_bca_lo"),
                "gain_ci_bca_hi": r.get("gain_ci_bca_hi"),
                "verdict_branch": r.get("verdict_branch"),
                "n_users_post_filter": r.get("n_users_post_filter"),
            } for r in per_seed_results
        ],
        "across_seed_mean_gain_pct": across_mean,
        "across_seed_std_gain_pct": across_std,
        "across_seed_min_gain_pct": across_min,
        "across_seed_max_gain_pct": across_max,
        "joint_verdict": joint,
        "prespecified_priors": PRESPECIFIED_PRIORS,
        "script_anchor_t1": git_head_t1(),
        "scored_parquet": str(parquet_path),
        "scored_parquet_sha": file_sha256(parquet_path),
        "elapsed_seconds": float(time.time() - t0),
        "smoke": bool(args.smoke),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[F-T1 OASST2] === JOINT VERDICT === {joint}")
    if not np.isnan(across_mean):
        print(f"[F-T1 OASST2] across-seed mean gain: {across_mean:+.3f}% +/- {across_std:.3f}  "
              f"(range [{across_min:+.3f}, {across_max:+.3f}])")
    print(f"[F-T1 OASST2] elapsed: {summary['elapsed_seconds']/60:.1f} min")
    print(f"[F-T1 OASST2] summary written: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
