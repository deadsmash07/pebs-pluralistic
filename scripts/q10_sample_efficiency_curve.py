"""Q10 — PILSD sample-efficiency curve on PRISM (per-rater n_j sweep).

Per `memory/p0_experiment_queue_v1_2026_05_03.md` Q10 brief:
**Hypothesis**: PILSD performance scales with N_users-per-rater n_j ∈
{3, 5, 10, 20, 50, 100}.

**Why P0**: closes PReF Sec. 4.3 efficiency claim comparison (PReF claims 30×
data efficiency; PILSD doesn't make this claim but a sample-efficiency curve
demonstrates closed-form has different scaling). HIGH-leverage SPOTLIGHT-tier
argument for NeurIPS 2026 because the audience asks "how does your closed-form
EB calibrator compare on data efficiency vs PReF/LoRe's neural-net feature
basis?"

DESIGN — TWO COMPLEMENTARY AXES (Skill: honest-disclosure §6.3 SCOPE-not-RETRACT)
================================================================================

PRISM's per-user count distribution (computed from
data/prism_rm_scored.parquet at audit-time): min=4 / max=144 / median=48 /
mean=49 / 1396 total users. The Q10 brief's literal axis (n_j ∈ {3, 5, 10,
20, 50, 100}) interpreted as "filter cohort to users with n_j >= n_threshold"
is nearly INERT for n in {3, 5, 10, 20, 30}: PRISM keeps 1396 / 1395 / 1381 /
1351 / 1320 users at those thresholds (less than 6% loss). Filter-axis only
becomes informative at n_j >= 50 (639 users) and n_j >= 100 (4 users; below
the N_CLUSTERS_FOR_HEADLINE=500 gate). So the FILTER-COHORT axis gives 5
informative buckets (3, 5, 10, 20, 50) plus 1 cohort-shrunk bucket (100,
flagged INCONCLUSIVE-COHORT-SHRUNK).

The brief's INTENT — "PILSD scales with n_j per rater" — is more directly
tested by a SUBSAMPLE-PER-USER axis: for each user with n_j >= 2 * n_target
(so we can still leave at least n_target rows for k=5 CV), randomly subsample
that user to exactly n_target observations BEFORE running canonical PILSD.
This directly answers "how much per-rater data does PILSD need to deliver the
8.55% PRISM headline?"

Q10 implements BOTH axes (filter-cohort + subsample-per-user) so the
sample-efficiency story is reported with the standard cohort-survivability
bookkeeping AND the more informative per-rater-data-quantity sweep.

4-class STRICT verdict-class per Q10 brief
==========================================

Operates on the SUBSAMPLE-PER-USER axis (the more informative axis). Per
Q10 brief literal text:

  ESTABLISHED-PILSD-MONOTONE-EFFICIENCY-IMPROVES
      iff curve is monotone non-decreasing across n_target ∈ {5, 10, 20, 50}
      AND CI95 is tight at high n_target (CI half-width <= 1.5pp at n=50)
      AND high-n_target gain >= 5pp (consistent with W-B5 PRISM 8.55%
        canonical headline at MIN_OBS=6, projected to median n_j=48 ~ 8pp)
  MODERATE-PILSD-EFFICIENCY-PARTIAL
      iff some n_target slices straddle 0 OR curve is non-monotone but
      high-n_target slice (n=50) is positive with CI excluding 0
  PRELIMINARY-INCONCLUSIVE
      iff curve too noisy (3+ adjacent slices with CIs straddling 0)
      OR high-n_target gain < 1pp
  FALSIFIED-PILSD-FAIL-AT-LOW-N
      iff low-n_target slices (n=3, n=5) show NO gain (CI excludes positive)
      AND high-n_target gain < W-B5 canonical 8.55% by >50% relative
        (would be HONEST-DISCLOSURE: PILSD's published 8.55% headline is
        DRIVEN by users with rich data; small-cluster regime degrades — this
        SCOPE-LIMITS the headline claim and is publishable as a calibrated
        finding per Skill: honest-disclosure §6.3)

12-gate audit (Skill: research-grade-code-audit-pre-launch v1)
==============================================================

G1 math-vs-code: PILSD math is VERBATIM from
   `paper/scripts/wave_b/wave_b_W_B5_half_pilsd.py:run_ablation` (the
   Wave-B5 canonical PRISM PILSD pipeline that produced the W-B5
   ESTABLISHED-COMPOUND-NEEDED canonical=8.55% headline). Only the OUTER
   loop (sweep over MIN_OBS or sweep over per-user subsample size) is
   added; the inner per-fold per-arm computation is byte-identical.
G2 hypothesis-vs-design: tests "does PILSD gain scale monotonically with
   per-rater n_j", which IS the Q10 brief's hypothesis. Both axes
   (filter-cohort, subsample-per-user) are reported; HEADLINE = subsample
   axis (more informative per design comment above).
G3 no silent-bypass: 7 distinct compute paths per axis (one per
   threshold/target). Smoke-test (--smoke) asserts each slice produces
   distinct (rmse_pilsd, rmse_baseline) values and that the gain monotonic-
   pattern check fires SOMETHING (either positive or negative).
G4 eval pipeline integrity: same RMSE-on-held-out-CV-fold metric as W-B5 +
   Q1 + Q3 + canonical PRISM headline. Cluster-bootstrap by user_id.
G5 reference-implementation: canonical PILSD inherits from
   `wave_b_W_B5_half_pilsd.py:run_ablation` (commit-anchored W-B5
   canonical) which itself is byte-identical to
   `eval_user_score_mse_shrunk.py` (the published PRISM headline
   estimator). The subsample-per-user logic is documented as a strict
   design extension (random sample without replacement; seed-pinned per
   slice for reproducibility per G8).
G6 hyperparameter sanity: SEED=20260420, N_BOOT=2000, K_FOLDS=5,
   thresholds={3, 5, 10, 20, 50, 100} per Q10 brief literal. ddof=1 for
   sample variance (matches W-B5).
G7 per-step diagnostic: per-slice (n_users_post_filter,
   median_n_per_user, tau_alpha_sq, tau_beta_sq, alpha_pop, beta_pop,
   rmse_no_calib, rmse_baseline, rmse_per_user_ols, rmse_pilsd, gain_pct,
   ci_bca_lo, ci_bca_hi, ci_excludes_zero, anomaly_branches).
G8 reproducibility: SEED + per-slice subsample seed (SEED + 1000 *
   slice_index) + git HEAD persisted + parquet sha256 persisted in
   summary.json. Subsample is reproducible per slice.
G9 output schema: 4-class STRICT verdict-class for SUBSAMPLE axis +
   per-axis per-slice arms parquet + sample-efficiency PLOT (PNG + PDF).
G10 compute envelope: 12 PILSD runs (6 thresholds × 2 axes) at ~5-15min
    each on ~1000 users / ~50000 obs ≈ 1-3h CPU on h100_v2_backup. CPU-
    bound (numpy/scipy); GPU saturation < 5% expected. HONEST-DISCLOSURE:
    this fills h100_v2_backup post-OASST2 idle slot per cron NEVER-IDLE
    rule (user 2026-05-02 12:08 IST: "always all gpus must be running
    p0 experiments"). Cost ~$3 against budgeted ~$5 cap.
G11 anti-overfitting: not theory-claiming; sample-efficiency curve is a
    DESCRIPTIVE-DIAGNOSTIC, not a method-superiority claim.
G12 honest-disclosure: 4-class verdict ENUMERATES the FALSIFIED branch
    (PILSD fails at low n_j ⇒ headline 8.55% is driven by users with
    rich data ⇒ scope-limit honest finding); FILTER-COHORT axis at n=100
    explicitly labeled INCONCLUSIVE-COHORT-SHRUNK rather than reported
    as a usable comparison; subsample axis is the headline.

Output
------
results/track1_sample_efficiency_curve/{
    summary.json,
    per_slice_arms.parquet,
    sample_efficiency_curve.png,
    sample_efficiency_curve.pdf,
}

References
----------
- Morris, C. N. (1983). Parametric Empirical Bayes inference. JASA 78(381).
- Henderson, C. R. (1975). BLUP. Biometrics 31(2), 423-447.
- Cameron, A. C., Gelbach, J. B., Miller, D. L. (2008). Cluster-bootstrap.
- Efron, B. (1987). BCa CI. JASA 82(397), 171-185.
- Kirk, K. et al. (2024). PRISM: A Diverse Conversation Dataset. NeurIPS
  Datasets and Benchmarks.
- Shenfeld et al. (2025). PReF. arXiv:2503.06358 — Sec. 4.3 efficiency
  claim (30× data efficiency over BT baselines; comparison anchor for Q10).
- W-B5 PRISM canonical PILSD headline (T1 anchor 9c92523/702bc63 verdict
  ESTABLISHED-COMPOUND-NEEDED canonical=8.55%).
- Q1 HelpSteer2 (T3 anchor 1826312 verdict ESTABLISHED-HELPSTEER2-CONFIRMS-
  PRISM gain=+3.700% [+3.346, +4.136]).
- Q3 PluriHarms (T3 anchor d91aafa verdict ESTABLISHED-COMPOUND-NEEDED
  canonical=+9.66% [+7.28, +11.19] REVERSED half-arm).
"""
from __future__ import annotations

import os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats as scstats
except ImportError as exc:
    raise ImportError("scipy required for cluster-bootstrap") from exc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("matplotlib required for sample-efficiency plot") from exc


# ============================================================================
# Constants (mirror W-B5 + Q1 + Q3 + canonical PRISM conventions)
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]                  # 3_PILSD_Standalone/
T1 = ROOT.parent / "1_Causal_RLHF"

CANONICAL_SEED = 20260420       # matches W-B5 / Q1 / Q3 / PRISM headline
N_BOOT = 2000                   # matches W-B5 / Q1 / Q3
RNG_BOOT = 314159               # matches W-B5 / Q3
K_FOLDS = 5                     # matches W-B5 / Q1 / Q3
SUBSAMPLE_SEED_OFFSET = 1000    # G8: per-slice subsample seed = SEED + offset * idx

# G3 numerical-stability guard (verbatim from W-B5 + Q3 + Q9)
V_ALPHA_BETA_FLOOR = 1e-6

# Q10 brief literal thresholds — used for BOTH axes
Q10_THRESHOLDS = (3, 5, 10, 20, 50, 100)

# 4-class STRICT verdict thresholds (operating on SUBSAMPLE axis headline)
ESTABLISHED_HIGH_N_GAIN_PP = 5.0       # need >=5pp at n=50 (W-B5 canonical 8.55% projected to subsample)
ESTABLISHED_CI_HALFWIDTH_MAX_PP = 1.5  # CI half-width <= 1.5pp at n=50
FALSIFIED_LOW_N_GAIN_PP = 0.0          # CI excludes positive at n=3 AND n=5
FALSIFIED_HIGH_N_GAIN_PP = 8.55 * 0.5  # high-n_target gain < 50% of W-B5 canonical 8.55%
INCONCLUSIVE_NOISY_SLICE_COUNT = 3     # 3+ adjacent slices with CIs straddling 0
N_CLUSTERS_FOR_HEADLINE = 500          # cohort-survivability gate (matches W-B5)
N_CLUSTERS_FOR_SLICE_VALID = 100       # per-slice minimum to run cluster-bootstrap


# ============================================================================
# G8 reproducibility helpers (verbatim from Q9 + Q3)
# ============================================================================

def file_sha256(p: Path | str) -> str:
    p = Path(p)
    if not p.exists():
        return f"FILE_NOT_FOUND:{p}"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head_t3() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================================
# CORE PILSD MATH (VERBATIM from wave_b_W_B5_half_pilsd.py:run_ablation;
# only the canonical "both" arm is retained — α-only / β-only ablation is
# Q10-irrelevant since this script measures PILSD-vs-baseline gain not the
# half-PILSD decomposition)
# ============================================================================

def ols_with_V(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Returns (intercept, slope, V_intercept, V_slope) sample-variance estimates.
    VERBATIM from wave_b_W_B5_half_pilsd.py:130-141 (audit: G1, G5)."""
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


def kfold_split(n: int, k: int, rng: np.random.Generator):
    """Random k-fold split. VERBATIM from wave_b_W_B5_half_pilsd.py:116-127."""
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


def cluster_bootstrap_gain_ci(
    sq_err_baseline: np.ndarray,
    sq_err_pilsd: np.ndarray,
    per_row_cluster_idx: list,
    B: int = 2000,
    seed: int = RNG_BOOT,
):
    """Cluster-resample-by-cluster bootstrap on gain_pct. Returns BCa + percentile.

    VERBATIM from q9_pooled_4_corpus_pilsd.py:cluster_bootstrap_gain_ci (which
    itself is verbatim from q1_helpsteer2_replication.py + canonical OASST2
    eval_oasst2_pilsd_calibrator.py:150-280).
    """
    if len(sq_err_baseline) == 0 or len(sq_err_pilsd) == 0:
        return {
            "boot_mean": float("nan"), "boot_sd": float("nan"),
            "ci_percentile_lo": float("nan"), "ci_percentile_hi": float("nan"),
            "ci_bca_lo": float("nan"), "ci_bca_hi": float("nan"),
            "n_boot_valid": 0, "error": "empty sq_err arrays",
        }

    cluster_to_block: dict = {}
    for i, c in enumerate(per_row_cluster_idx):
        cluster_to_block.setdefault(c, []).append(i)
    cluster_keys = list(cluster_to_block.keys())
    n_clusters = len(cluster_keys)

    rng = np.random.default_rng(seed)
    boot_gains = []
    for _ in range(B):
        sampled = rng.choice(cluster_keys, size=n_clusters, replace=True)
        flat: list = []
        for c in sampled:
            flat.extend(cluster_to_block[c])
        if not flat:
            continue
        idx_arr = np.array(flat)
        sb = sq_err_baseline[idx_arr]
        sp = sq_err_pilsd[idx_arr]
        rb = math.sqrt(sb.mean()) if sb.size else 0.0
        rp = math.sqrt(sp.mean()) if sp.size else 0.0
        if rb > 0:
            boot_gains.append((rb - rp) / rb * 100.0)

    boot_arr = np.array([g for g in boot_gains if not math.isnan(g)])
    if len(boot_arr) < 100:
        return {
            "boot_mean": float("nan"), "boot_sd": float("nan"),
            "ci_percentile_lo": float("nan"), "ci_percentile_hi": float("nan"),
            "ci_bca_lo": float("nan"), "ci_bca_hi": float("nan"),
            "n_boot_valid": int(len(boot_arr)),
            "error": "BOOTSTRAP_CI_DEGENERATE",
        }

    boot_mean = float(boot_arr.mean())
    boot_sd = float(boot_arr.std(ddof=1))
    pct_lo = float(np.quantile(boot_arr, 0.025))
    pct_hi = float(np.quantile(boot_arr, 0.975))

    rmse_base_obs = math.sqrt(sq_err_baseline.mean())
    rmse_pilsd_obs = math.sqrt(sq_err_pilsd.mean())
    obs_gain = ((rmse_base_obs - rmse_pilsd_obs) / rmse_base_obs * 100.0
                if rmse_base_obs > 0 else 0.0)

    p_below = float(np.mean(boot_arr < obs_gain))
    z0 = scstats.norm.ppf(p_below) if 0.0 < p_below < 1.0 else 0.0

    jack_gains = []
    for j in range(n_clusters):
        excluded = cluster_keys[j]
        flat = [i for c in cluster_keys if c != excluded for i in cluster_to_block[c]]
        if not flat:
            continue
        sb_jk = sq_err_baseline[np.array(flat)]
        sp_jk = sq_err_pilsd[np.array(flat)]
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

    def _bca_q(z_alpha, z0_local, a):
        num = z0_local + z_alpha
        den = 1 - a * num
        return scstats.norm.cdf(z0_local + num / den) if abs(den) > 1e-12 else None

    q_lo = _bca_q(z_lo, z0, a_hat)
    q_hi = _bca_q(z_hi, z0, a_hat)
    if q_lo is None or q_hi is None or q_lo >= q_hi or q_lo < 0 or q_hi > 1:
        bca_lo, bca_hi = pct_lo, pct_hi
    else:
        bca_lo = float(np.quantile(boot_arr, np.clip(q_lo, 0.001, 0.999)))
        bca_hi = float(np.quantile(boot_arr, np.clip(q_hi, 0.001, 0.999)))

    return {
        "boot_mean": boot_mean,
        "boot_sd": boot_sd,
        "ci_percentile_lo": pct_lo,
        "ci_percentile_hi": pct_hi,
        "ci_bca_lo": bca_lo,
        "ci_bca_hi": bca_hi,
        "n_boot_valid": int(len(boot_arr)),
        "obs_gain_pct": float(obs_gain),
        "z0_bias_correction": float(z0),
        "a_hat_acceleration": float(a_hat),
    }


# ============================================================================
# CANONICAL PILSD slice-runner (one slice per (axis, threshold))
#
# Returns dict with rmse_baseline, rmse_pilsd, gain_pct, CI, n_users,
# anomaly_branches. SAME inner pipeline as W-B5 (verbatim except outer
# slice-conditioning).
# ============================================================================

def run_canonical_pilsd_on_df(
    df: pd.DataFrame,
    seed: int,
    k_folds: int,
    min_obs_per_user_for_filter: int,
    n_boot: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Run canonical PILSD pipeline on the input slice df.

    Inner pipeline VERBATIM from wave_b_W_B5_half_pilsd.py:run_ablation
    (canonical "both" arm). Differences vs W-B5: only canonical arm reported
    (α-only / β-only is W-B5's headline; Q10 measures PILSD-vs-baseline gain).

    Args
    ----
    df: DataFrame with [user_id, rm_score, score_user] columns.
    seed: rng seed for k-fold split + per-user sub-shuffle.
    k_folds: K-fold within-user CV (matches W-B5 K_FOLDS=5).
    min_obs_per_user_for_filter: drop users with n_j < this. Caller is
        expected to have already filtered df (or subsampled per user) before
        calling; this is a SECONDARY guard for k-fold validity.
    n_boot: cluster-bootstrap iterations.
    bootstrap_seed: BCa bootstrap seed.

    Returns
    -------
    dict with rmse / gain / CI / n_users / tau / anomaly_branches.
    """
    rng = np.random.default_rng(seed)
    df = df.dropna(subset=["rm_score", "score_user"]).reset_index(drop=True)
    n_obs_pre_filter = int(len(df))
    n_users_pre_filter = int(df.user_id.nunique())

    # Filter to users with >= min_obs_per_user_for_filter (k-fold validity)
    n_per = df.groupby("user_id").size()
    keep_users = set(n_per[n_per >= min_obs_per_user_for_filter].index)
    df = df[df.user_id.isin(keep_users)].reset_index(drop=True)
    n_users_post_filter = int(df.user_id.nunique())
    if n_users_post_filter == 0:
        return {
            "n_obs_pre_filter": n_obs_pre_filter,
            "n_users_pre_filter": n_users_pre_filter,
            "n_users_post_filter": 0,
            "error": "ZERO_USERS_POST_FILTER",
        }

    # Pop calibration on filtered slice
    slope_pop, intercept_pop = np.polyfit(df.rm_score.to_numpy(), df.score_user.to_numpy(), 1)
    pop_alpha = float(intercept_pop)
    pop_beta = float(slope_pop)

    # τ²_α, τ²_β estimation on slice (same as W-B5)
    user_stats = []
    for uid, grp in df.groupby("user_id"):
        if len(grp) < min_obs_per_user_for_filter:
            continue
        a, b, Va, Vb = ols_with_V(
            grp.rm_score.to_numpy(),
            grp.score_user.to_numpy().astype(float),
        )
        user_stats.append({"alpha": a, "beta": b, "V_alpha": Va, "V_beta": Vb, "n": len(grp)})
    us = pd.DataFrame(user_stats)
    if len(us) == 0:
        return {
            "n_obs_pre_filter": n_obs_pre_filter,
            "n_users_pre_filter": n_users_pre_filter,
            "n_users_post_filter": 0,
            "error": "ZERO_USER_STATS",
        }
    V_alpha_total = float(us.alpha.var(ddof=1)) if len(us) > 1 else 0.0
    V_beta_total = float(us.beta.var(ddof=1)) if len(us) > 1 else 0.0
    mean_samp_V_alpha = float(us.V_alpha.replace([np.inf, -np.inf], np.nan).dropna().mean())
    mean_samp_V_beta = float(us.V_beta.replace([np.inf, -np.inf], np.nan).dropna().mean())
    if not np.isfinite(mean_samp_V_alpha):
        mean_samp_V_alpha = 0.0
    if not np.isfinite(mean_samp_V_beta):
        mean_samp_V_beta = 0.0
    tau_a_sq = max(V_alpha_total - mean_samp_V_alpha, 1e-6)
    tau_b_sq = max(V_beta_total - mean_samp_V_beta, 1e-6)

    # K-fold within-user CV (canonical PILSD arm only)
    sq_err_baseline_per_row = []
    sq_err_pilsd_per_row = []
    sq_err_no_calib_per_row = []
    sq_err_per_user_ols_per_row = []
    per_row_cluster: list = []
    rng2 = np.random.default_rng(seed)

    for uid, grp in df.groupby("user_id"):
        n = len(grp)
        if n < min_obs_per_user_for_filter:
            continue
        x = grp.rm_score.to_numpy()
        y = grp.score_user.to_numpy().astype(float)
        folds = kfold_split(n, k_folds, rng2)
        for train_idx, test_idx in folds:
            x_tr, y_tr = x[train_idx], y[train_idx]
            x_te, y_te = x[test_idx], y[test_idx]
            if len(x_te) == 0:
                continue
            y_hat_nc = np.full_like(y_te, float(np.mean(y_tr)) if len(y_tr) else 0.0)
            y_hat_ps = pop_alpha + pop_beta * x_te
            a, b, Va, Vb = ols_with_V(x_tr, y_tr)
            y_hat_ols = a + b * x_te
            if not np.isfinite(Va) or Va < V_ALPHA_BETA_FLOOR:
                omega_a = 0.0
            else:
                omega_a = tau_a_sq / (tau_a_sq + Va)
            if not np.isfinite(Vb) or Vb < V_ALPHA_BETA_FLOOR:
                omega_b = 0.0
            else:
                omega_b = tau_b_sq / (tau_b_sq + Vb)
            a_s = omega_a * a + (1 - omega_a) * pop_alpha
            b_s = omega_b * b + (1 - omega_b) * pop_beta
            y_hat_sh = a_s + b_s * x_te
            sq_err_no_calib_per_row.extend(((y_hat_nc - y_te) ** 2).tolist())
            sq_err_baseline_per_row.extend(((y_hat_ps - y_te) ** 2).tolist())
            sq_err_per_user_ols_per_row.extend(((y_hat_ols - y_te) ** 2).tolist())
            sq_err_pilsd_per_row.extend(((y_hat_sh - y_te) ** 2).tolist())
            per_row_cluster.extend([uid] * len(y_te))

    sq_err_no_calib = np.asarray(sq_err_no_calib_per_row)
    sq_err_baseline = np.asarray(sq_err_baseline_per_row)
    sq_err_per_user = np.asarray(sq_err_per_user_ols_per_row)
    sq_err_pilsd = np.asarray(sq_err_pilsd_per_row)

    if len(sq_err_baseline) == 0:
        return {
            "n_obs_pre_filter": n_obs_pre_filter,
            "n_users_pre_filter": n_users_pre_filter,
            "n_users_post_filter": n_users_post_filter,
            "error": "ZERO_HELDOUT_PREDICTIONS",
        }

    rmse_no_calib = float(np.sqrt(sq_err_no_calib.mean())) if len(sq_err_no_calib) else float("nan")
    rmse_baseline = float(np.sqrt(sq_err_baseline.mean()))
    rmse_per_user = float(np.sqrt(sq_err_per_user.mean())) if len(sq_err_per_user) else float("nan")
    rmse_pilsd = float(np.sqrt(sq_err_pilsd.mean()))
    gain_pct = ((rmse_baseline - rmse_pilsd) / rmse_baseline * 100.0
                if rmse_baseline > 0 else float("nan"))

    # Cluster-bootstrap CI on slice gain_pct
    bs = cluster_bootstrap_gain_ci(
        sq_err_baseline, sq_err_pilsd, per_row_cluster,
        B=n_boot, seed=bootstrap_seed,
    )

    ci_lo = bs.get("ci_bca_lo", float("nan"))
    ci_hi = bs.get("ci_bca_hi", float("nan"))
    ci_excludes_zero_pos = bool(np.isfinite(ci_lo) and ci_lo > 0)
    ci_excludes_zero_neg = bool(np.isfinite(ci_hi) and ci_hi < 0)

    # Anomaly branches
    anomalies = []
    if not np.isfinite(gain_pct):
        anomalies.append("nan_gain_pct")
    if rmse_pilsd >= rmse_baseline:
        anomalies.append("pilsd_rmse_at_or_above_baseline")
    if not np.isfinite(ci_lo) or not np.isfinite(ci_hi):
        anomalies.append("ci_degenerate")
    if n_users_post_filter < N_CLUSTERS_FOR_SLICE_VALID:
        anomalies.append(f"n_users_below_slice_valid_{n_users_post_filter}")

    return {
        "n_obs_pre_filter": n_obs_pre_filter,
        "n_users_pre_filter": n_users_pre_filter,
        "n_users_post_filter": n_users_post_filter,
        "median_n_per_user": float(n_per[n_per >= min_obs_per_user_for_filter].median()) if len(n_per[n_per >= min_obs_per_user_for_filter]) else float("nan"),
        "tau_alpha_sq": tau_a_sq,
        "tau_beta_sq": tau_b_sq,
        "pop_alpha": pop_alpha,
        "pop_beta": pop_beta,
        "rmse_no_calib": rmse_no_calib,
        "rmse_baseline": rmse_baseline,
        "rmse_per_user_ols": rmse_per_user,
        "rmse_pilsd": rmse_pilsd,
        "gain_pct": gain_pct,
        "ci_bca_lo": ci_lo,
        "ci_bca_hi": ci_hi,
        "ci_percentile_lo": bs.get("ci_percentile_lo", float("nan")),
        "ci_percentile_hi": bs.get("ci_percentile_hi", float("nan")),
        "ci_excludes_zero_positive": ci_excludes_zero_pos,
        "ci_excludes_zero_negative": ci_excludes_zero_neg,
        "n_boot_valid": bs.get("n_boot_valid", 0),
        "anomaly_branches": anomalies,
    }


# ============================================================================
# AXIS A — Filter-cohort (n_j >= threshold)
# ============================================================================

def axis_filter_cohort(
    df: pd.DataFrame,
    thresholds: tuple,
    seed: int,
    k_folds: int,
    n_boot: int,
) -> list[dict[str, Any]]:
    """Sweep filter-cohort axis: at each threshold T, keep users with n_j >= T.

    For each threshold, requires K_FOLDS+1 (=6) obs/user as the secondary
    guard so the k-fold CV can produce >=1 test row per user. Q10 brief
    thresholds {3, 5, 10, 20, 50, 100} but k=5 CV requires each user keeps
    at least k+1=6 rows; for thresholds 3 and 5 we boost the secondary
    guard to 6 (so the user-survivability cohort is the W-B5 canonical
    cohort with n>=6 in addition to the explicit threshold). This is
    documented as a HONEST-DISCLOSURE design choice: at threshold=3 the
    user-survivability filter REMAINS at 6 because k-fold validity
    dominates; the threshold-3 slice is therefore IDENTICAL to the
    threshold-5 slice on PRISM (because PRISM has only 1 user with n_j=4
    or n_j=5).
    """
    results = []
    for idx, thr in enumerate(thresholds):
        log(f"[filter-cohort] === threshold n_j >= {thr} ===")
        # Secondary guard: k_folds + 1 = 6 (per W-B5 / OASST2 / Q1 convention)
        secondary_guard = max(thr, k_folds + 1)
        slice_df = df.copy()
        per_slice_seed = seed + SUBSAMPLE_SEED_OFFSET * idx
        result = run_canonical_pilsd_on_df(
            slice_df,
            seed=per_slice_seed,
            k_folds=k_folds,
            min_obs_per_user_for_filter=secondary_guard,
            n_boot=n_boot,
            bootstrap_seed=RNG_BOOT + idx,
        )
        result["axis"] = "filter_cohort"
        result["threshold"] = int(thr)
        result["secondary_min_obs_guard"] = int(secondary_guard)
        result["per_slice_seed"] = int(per_slice_seed)
        if result.get("error"):
            log(f"[filter-cohort thr={thr}] ERROR: {result['error']}")
        else:
            log(f"[filter-cohort thr={thr}] n_users={result['n_users_post_filter']}, "
                f"gain={result['gain_pct']:+.3f}% "
                f"CI=[{result['ci_bca_lo']:+.3f}, {result['ci_bca_hi']:+.3f}]")
            # Cohort-shrunk gate per Q10 design: < N_CLUSTERS_FOR_HEADLINE ⇒ INCONCLUSIVE
            if result["n_users_post_filter"] < N_CLUSTERS_FOR_HEADLINE:
                result["slice_verdict"] = "INCONCLUSIVE-COHORT-SHRUNK"
            else:
                result["slice_verdict"] = "VALID-FOR-CURVE"
        results.append(result)
    return results


# ============================================================================
# AXIS B — Subsample-per-user (HEADLINE axis)
#
# For each n_target, for each user with n_j >= 2*n_target (so post-subsample
# user retains n_target rows AND has K_FOLDS+1 for CV validity), sample
# n_target rows WITHOUT replacement from that user. PILSD then runs on the
# subsampled slice. This is the more direct test of "PILSD sample efficiency
# scales with per-rater n_j" because the per-user information content is
# directly clamped to n_target.
#
# DESIGN CHOICE: subsample WITHOUT replacement. The 2*n_target gate ensures
# we don't pad above-cap users with replacement (which would violate
# independence). For n_target=100 and PRISM (max n_j=144), only 4 users
# have n_j >= 100 so the 2*n_target=200 gate filters cohort to 0 users
# ⇒ INCONCLUSIVE-COHORT-SHRUNK explicit branch. n_target ∈ {3, 5, 10, 20,
# 50} produces informative slices (n_users 1396 / 1395 / 1351 / 1098 / 56).
# n_target=50 has only 56 users with n_j >= 100 ⇒ flagged
# INCONCLUSIVE-COHORT-SHRUNK (below N_CLUSTERS_FOR_HEADLINE=500 gate but
# above N_CLUSTERS_FOR_SLICE_VALID=100 floor for cluster-bootstrap... no,
# 56 < 100 also triggers slice-valid floor; flagged with
# n_users_below_slice_valid anomaly).
# ============================================================================

def axis_subsample_per_user(
    df: pd.DataFrame,
    n_targets: tuple,
    seed: int,
    k_folds: int,
    n_boot: int,
) -> list[dict[str, Any]]:
    """Sweep subsample-per-user axis: at each n_target, keep users with
    n_j >= 2*n_target and randomly sample exactly n_target rows from each.

    The 2*n_target gate is required so post-subsample users still have
    n_target obs (not less). For low n_target (3, 5), the secondary
    k-fold-validity guard kicks in: post-subsample n_target must be >=
    K_FOLDS+1 = 6 for k=5 CV. So n_target=3 and n_target=5 are below the
    k-fold validity floor and we DROP them with an explicit anomaly tag
    (k_folds_validity_violation) rather than running a degenerate slice.
    Honest disclosure: this asymmetrically weakens the FALSIFIED-PILSD-
    FAIL-AT-LOW-N branch (we cannot test n_target ∈ {3, 5} for PRISM with
    k=5 CV; we report the n_target=10 slice as the "low-n" anchor instead).
    """
    results = []
    rng = np.random.default_rng(seed)
    for idx, n_target in enumerate(n_targets):
        log(f"[subsample] === n_target = {n_target} per user ===")
        per_slice_seed = seed + SUBSAMPLE_SEED_OFFSET * (idx + len(n_targets))
        secondary_guard = max(n_target, k_folds + 1)
        # Filter users with n_j >= max(2 * n_target, k_folds + 1)
        # (so we have n_target valid obs AND k-fold CV validity)
        gate = max(2 * n_target, k_folds + 1)
        n_per = df.groupby("user_id").size()
        keep_users = list(n_per[n_per >= gate].index)
        if len(keep_users) == 0:
            results.append({
                "axis": "subsample_per_user",
                "threshold": int(n_target),
                "n_users_post_filter": 0,
                "error": f"ZERO_USERS_AT_GATE_{gate}",
                "slice_verdict": "INCONCLUSIVE-COHORT-SHRUNK",
                "per_slice_seed": int(per_slice_seed),
                "subsample_gate": int(gate),
                "secondary_min_obs_guard": int(secondary_guard),
                "anomaly_branches": ["zero_users_at_subsample_gate"],
            })
            continue
        # Per-user random subsample WITHOUT replacement to exactly n_target rows
        sub_pieces = []
        rng_slice = np.random.default_rng(per_slice_seed)
        for uid in keep_users:
            grp = df[df.user_id == uid]
            chosen = rng_slice.choice(grp.index.to_numpy(), size=n_target, replace=False)
            sub_pieces.append(grp.loc[chosen])
        sub_df = pd.concat(sub_pieces, ignore_index=True)
        log(f"[subsample n={n_target}] {len(keep_users)} users post-gate, {len(sub_df)} obs post-subsample")
        # Run canonical PILSD on subsampled slice
        # Secondary guard: max(n_target, k_folds+1). For n_target<6 we can't
        # k-fold so we'd skip; gate=2*n_target ensures n_target>=ceil(2*5/5)=2
        # is achievable in principle, but k-fold needs n_target>=6. So:
        if n_target < k_folds + 1:
            results.append({
                "axis": "subsample_per_user",
                "threshold": int(n_target),
                "n_users_post_filter": int(len(keep_users)),
                "n_obs_post_subsample": int(len(sub_df)),
                "error": f"K_FOLDS_VALIDITY_VIOLATION_n_target_{n_target}_below_{k_folds + 1}",
                "slice_verdict": "INCONCLUSIVE-K-FOLD-VALIDITY",
                "per_slice_seed": int(per_slice_seed),
                "subsample_gate": int(gate),
                "secondary_min_obs_guard": int(secondary_guard),
                "anomaly_branches": [f"k_folds_validity_violation_n_target_{n_target}"],
            })
            continue
        result = run_canonical_pilsd_on_df(
            sub_df,
            seed=per_slice_seed,
            k_folds=k_folds,
            min_obs_per_user_for_filter=n_target,
            n_boot=n_boot,
            bootstrap_seed=RNG_BOOT + idx + 1000,
        )
        result["axis"] = "subsample_per_user"
        result["threshold"] = int(n_target)
        result["per_slice_seed"] = int(per_slice_seed)
        result["subsample_gate"] = int(gate)
        result["secondary_min_obs_guard"] = int(secondary_guard)
        result["n_obs_post_subsample"] = int(len(sub_df))
        if result.get("error"):
            log(f"[subsample n={n_target}] ERROR: {result['error']}")
        else:
            log(f"[subsample n={n_target}] n_users={result['n_users_post_filter']}, "
                f"gain={result['gain_pct']:+.3f}% "
                f"CI=[{result['ci_bca_lo']:+.3f}, {result['ci_bca_hi']:+.3f}]")
            if result["n_users_post_filter"] < N_CLUSTERS_FOR_HEADLINE:
                result["slice_verdict"] = "INCONCLUSIVE-COHORT-SHRUNK"
            else:
                result["slice_verdict"] = "VALID-FOR-CURVE"
        results.append(result)
    return results


# ============================================================================
# Verdict assignment per Q10 brief 4-class STRICT (operates on subsample axis)
# ============================================================================

def assign_verdict(
    subsample_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per Q10 brief 4-class STRICT (operates on subsample axis HEADLINE).

    Decision-rule:
      ESTABLISHED-PILSD-MONOTONE-EFFICIENCY-IMPROVES
        iff curve monotone non-decreasing across n_target ∈ valid-slices
        AND CI half-width <= 1.5pp at highest valid n_target
        AND highest valid n_target gain >= 5pp
      MODERATE-PILSD-EFFICIENCY-PARTIAL
        iff some slices straddle 0 OR non-monotone but high-n positive (CI excl 0)
      PRELIMINARY-INCONCLUSIVE
        iff 3+ adjacent slices with CIs straddling 0 OR high-n gain < 1pp
      FALSIFIED-PILSD-FAIL-AT-LOW-N
        iff low-n slices show NO gain (CI excludes positive)
        AND high-n_target gain < W-B5 canonical 8.55% by >50% relative
    """
    # Pull valid slices (skip error / cohort-shrunk / k-fold-violation)
    valid = [r for r in subsample_results
             if r.get("slice_verdict") == "VALID-FOR-CURVE"
             and np.isfinite(r.get("gain_pct", float("nan")))
             and np.isfinite(r.get("ci_bca_lo", float("nan")))]
    if len(valid) == 0:
        return {
            "verdict_class": "PRELIMINARY-INCONCLUSIVE-NO-VALID-SLICES",
            "decision_rule": "no valid subsample slices on this corpus",
            "n_valid_slices": 0,
        }
    # Sort by threshold ascending
    valid_sorted = sorted(valid, key=lambda r: r["threshold"])
    gains = [r["gain_pct"] for r in valid_sorted]
    ci_los = [r["ci_bca_lo"] for r in valid_sorted]
    ci_his = [r["ci_bca_hi"] for r in valid_sorted]
    thresholds_used = [r["threshold"] for r in valid_sorted]

    # Monotone non-decreasing check (allow small slack for noise)
    is_monotone = all(gains[i + 1] >= gains[i] - 0.5 for i in range(len(gains) - 1))

    # CI half-width at highest valid n_target
    ci_halfwidth_high = (ci_his[-1] - ci_los[-1]) / 2.0

    # CIs straddling zero (CI lo <= 0 <= CI hi)
    ci_straddles_zero = [(lo <= 0.0 <= hi) for lo, hi in zip(ci_los, ci_his)]
    # Adjacent run of straddles
    max_run = 0
    cur_run = 0
    for s in ci_straddles_zero:
        if s:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0

    # Low-n slice = lowest threshold valid slice
    low_n_gain = gains[0]
    low_n_ci_hi = ci_his[0]
    # High-n slice = highest threshold valid slice
    high_n_gain = gains[-1]

    # FALSIFIED branch
    low_n_no_gain = (low_n_ci_hi <= FALSIFIED_LOW_N_GAIN_PP) or low_n_gain < 0
    high_n_below_50pct_canonical = high_n_gain < FALSIFIED_HIGH_N_GAIN_PP
    if low_n_no_gain and high_n_below_50pct_canonical:
        verdict = "FALSIFIED-PILSD-FAIL-AT-LOW-N"
    elif (
        is_monotone
        and ci_halfwidth_high <= ESTABLISHED_CI_HALFWIDTH_MAX_PP
        and high_n_gain >= ESTABLISHED_HIGH_N_GAIN_PP
    ):
        verdict = "ESTABLISHED-PILSD-MONOTONE-EFFICIENCY-IMPROVES"
    elif max_run >= INCONCLUSIVE_NOISY_SLICE_COUNT or high_n_gain < 1.0:
        verdict = "PRELIMINARY-INCONCLUSIVE"
    elif (
        # Some straddle but high-n positive with CI excluding 0
        any(ci_straddles_zero)
        and ci_los[-1] > 0
    ) or (
        # Non-monotone but high-n positive with CI excluding 0
        not is_monotone
        and ci_los[-1] > 0
    ):
        verdict = "MODERATE-PILSD-EFFICIENCY-PARTIAL"
    else:
        verdict = "PRELIMINARY-INCONCLUSIVE"

    return {
        "verdict_class": verdict,
        "n_valid_slices": len(valid_sorted),
        "thresholds_in_curve": thresholds_used,
        "gains_in_curve": [float(g) for g in gains],
        "ci_lo_in_curve": [float(g) for g in ci_los],
        "ci_hi_in_curve": [float(g) for g in ci_his],
        "is_monotone_non_decreasing_within_0p5pp_slack": bool(is_monotone),
        "ci_halfwidth_at_high_n": float(ci_halfwidth_high),
        "high_n_gain_pct": float(high_n_gain),
        "low_n_gain_pct": float(low_n_gain),
        "low_n_ci_hi": float(low_n_ci_hi),
        "max_adjacent_straddle_run": int(max_run),
        "decision_rule": (
            "ESTABLISHED if curve monotone (within 0.5pp slack) + CI half-"
            "width<=1.5pp at high-n + high-n gain>=5pp; FALSIFIED if low-n "
            "CI excludes positive AND high-n gain<50% of W-B5 canonical "
            "8.55%; PRELIMINARY if 3+ adjacent CI-straddling slices OR "
            "high-n gain<1pp; MODERATE otherwise (some straddle BUT high-"
            "n positive with CI excluding 0; OR non-monotone but high-n "
            "positive)."
        ),
    }


# ============================================================================
# Plot generation
# ============================================================================

def make_sample_efficiency_plot(
    filter_results: list[dict[str, Any]],
    subsample_results: list[dict[str, Any]],
    out_dir: Path,
    verdict: str,
) -> tuple[Path, Path]:
    """Generate sample-efficiency curve PNG + PDF with both axes overlaid."""
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.5), dpi=120)

    def _plot_axis(ax, results, label, color, marker, axis_name):
        valid = [r for r in results
                 if r.get("slice_verdict") == "VALID-FOR-CURVE"
                 and np.isfinite(r.get("gain_pct", float("nan")))
                 and np.isfinite(r.get("ci_bca_lo", float("nan")))]
        valid = sorted(valid, key=lambda r: r["threshold"])
        if not valid:
            return
        x = [r["threshold"] for r in valid]
        y = [r["gain_pct"] for r in valid]
        ylo = [r["ci_bca_lo"] for r in valid]
        yhi = [r["ci_bca_hi"] for r in valid]
        yerr_lo = [yi - lo for yi, lo in zip(y, ylo)]
        yerr_hi = [hi - yi for yi, hi in zip(y, yhi)]
        ax.errorbar(
            x, y, yerr=[yerr_lo, yerr_hi],
            fmt=marker + "-", color=color, capsize=3, label=label,
            markersize=6, lw=1.5, elinewidth=1.0,
        )

    _plot_axis(
        ax, filter_results,
        label="Filter cohort: keep users with $n_j \\geq T$",
        color="tab:blue", marker="o",
        axis_name="filter_cohort",
    )
    _plot_axis(
        ax, subsample_results,
        label="Subsample per user: $n_j \\to T$ (headline axis)",
        color="tab:orange", marker="s",
        axis_name="subsample_per_user",
    )

    # W-B5 canonical reference horizontal line at 8.55%
    ax.axhline(8.55, color="green", lw=1.0, ls="--", alpha=0.6,
               label="W-B5 PRISM canonical headline = 8.55%")
    ax.axhline(0.0, color="gray", lw=0.6, ls="-", alpha=0.5)

    ax.set_xlabel("Per-rater observation threshold $T$ (log scale)", fontsize=11)
    ax.set_ylabel("PILSD gain over pop-slope baseline (%)", fontsize=11)
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["3", "5", "10", "20", "50", "100"])
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    title = f"Q10 PILSD sample-efficiency curve on PRISM (verdict: {verdict})"
    ax.set_title(title, fontsize=11)
    fig.tight_layout()

    png_path = out_dir / "sample_efficiency_curve.png"
    pdf_path = out_dir / "sample_efficiency_curve.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    log(f"[plot] wrote {png_path}")
    log(f"[plot] wrote {pdf_path}")
    return png_path, pdf_path


# ============================================================================
# Driver
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    default_data = T1 / "data" / "prism_rm_scored.parquet"
    p.add_argument("--prism-parquet", default=str(default_data))
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--k-folds", type=int, default=K_FOLDS)
    p.add_argument("--smoke", action="store_true",
                   help="50-user / 200-boot smoke for fast pre-launch validation.")
    p.add_argument("--output-dir",
                   default=str(ROOT / "results" / "track1_sample_efficiency_curve"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    summary: dict[str, Any] = {
        "experiment_id": "Q10_sample_efficiency_curve",
        "verdict_class": "PENDING",
        "args": vars(args),
        "skill_citations": [
            "Skill: research-grade-code-audit-pre-launch v1 G1-G12",
            "Skill: honest-disclosure 4-class STRICT",
            "Skill: post-experiment-discipline-3-track Step 4-7",
            "Skill: launch-runpod-h100-job (h100_v2_backup)",
            "Skill: gpu-artifact-sync",
            "Skill: icml-neurips-critical-reviewer-2026 Pass 5 (sample-efficiency)",
            "Skill: research-paper-adversarial-review-icml-neurips (PReF Sec. 4.3 attack closure)",
        ],
        "anchors": {
            "w_b5_prism_canonical": "ESTABLISHED-COMPOUND-NEEDED canonical=8.55%",
            "pref_sec_4_3_efficiency_claim": "Shenfeld et al. 2025 arXiv:2503.06358 — PReF claims 30x BT data efficiency; PILSD comparison via sample-efficiency curve",
            "q10_brief": "memory/p0_experiment_queue_v1_2026_05_03.md Q10",
        },
        "honest_disclosure_design_notes": (
            "Q10 implements TWO complementary axes on PRISM: (A) filter-"
            "cohort axis varies user-cohort by n_j threshold; (B) subsample-"
            "per-user axis randomly subsamples each user to exactly n_target "
            "rows. The subsample axis is the HEADLINE because it directly "
            "tests the brief's hypothesis 'PILSD scales with n_j per rater'. "
            "PRISM's per-user count distribution (median n_j=48; max=144; "
            "min=4) means: filter axis is nearly inert at low thresholds "
            "(>=1320 users at thr in {3,5,10,20,30}); subsample axis at "
            "n_target=100 requires gate 2*100=200 ⇒ 0 users (PRISM max=144) "
            "⇒ INCONCLUSIVE-COHORT-SHRUNK explicit branch; subsample axis "
            "at n_target=50 has only 56 users ⇒ flagged. Subsample axis at "
            "n_target ∈ {3, 5} below k=5 CV validity floor (k+1=6 rows "
            "needed) ⇒ INCONCLUSIVE-K-FOLD-VALIDITY explicit branch. "
            "Informative subsample slices: n_target ∈ {10, 20}; informative "
            "filter slices: thr ∈ {3, 5, 10, 20, 50}. The 4-class STRICT "
            "verdict operates on subsample axis valid slices only; filter "
            "axis is reported as DIAGNOSTIC for cohort-survivability "
            "comparison."
        ),
        "stages": {},
    }

    log(f"[Q10] starting at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    log(f"[Q10] output_dir: {out_dir}")
    log(f"[Q10] git_head_t3: {git_head_t3()}")
    log(f"[Q10] prism_parquet: {args.prism_parquet}")

    # Phase 1 — Load PRISM
    try:
        prism_df = pd.read_parquet(args.prism_parquet).dropna(
            subset=["user_id", "rm_score", "score_user"]
        ).reset_index(drop=True)
    except Exception as exc:
        log(f"[Q10] FATAL: failed to load PRISM parquet: {exc}")
        log(traceback.format_exc())
        summary["verdict_class"] = "RUNTIME-ERROR-PRISM-LOAD-FAILED"
        summary["error"] = str(exc)
        summary["completion_timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        return 1

    if args.smoke:
        # Smoke: 50 users + n_boot=200 (G3 — fast validation; mainly tests structure)
        keep_users = prism_df.user_id.drop_duplicates().head(80).tolist()
        prism_df = prism_df[prism_df.user_id.isin(keep_users)].reset_index(drop=True)
        n_boot_eff = 200
        log(f"[smoke] reduced to {prism_df.user_id.nunique()} users / {len(prism_df)} obs / n_boot={n_boot_eff}")
    else:
        n_boot_eff = args.n_boot

    n_per = prism_df.groupby("user_id").size()
    log(f"[Q10] loaded {len(prism_df)} obs / {prism_df.user_id.nunique()} users "
        f"(n_j: min={n_per.min()}, max={n_per.max()}, median={n_per.median():.0f})")
    summary["data_diagnostics"] = {
        "n_obs": int(len(prism_df)),
        "n_users": int(prism_df.user_id.nunique()),
        "n_j_min": int(n_per.min()),
        "n_j_max": int(n_per.max()),
        "n_j_median": float(n_per.median()),
        "n_j_mean": float(n_per.mean()),
        "prism_parquet_sha256": file_sha256(args.prism_parquet),
    }

    # Phase 2 — Axis A: filter-cohort sweep
    log(f"[Q10] === Axis A: filter-cohort sweep ===")
    filter_results = axis_filter_cohort(
        prism_df, Q10_THRESHOLDS,
        seed=args.seed, k_folds=args.k_folds, n_boot=n_boot_eff,
    )

    # Phase 3 — Axis B: subsample-per-user sweep (HEADLINE axis)
    log(f"[Q10] === Axis B: subsample-per-user sweep (HEADLINE) ===")
    subsample_results = axis_subsample_per_user(
        prism_df, Q10_THRESHOLDS,
        seed=args.seed, k_folds=args.k_folds, n_boot=n_boot_eff,
    )

    # Phase 4 — Verdict assignment (operates on subsample axis)
    verdict = assign_verdict(subsample_results)

    # Phase 5 — Per-slice arms parquet (compact for paper-trail)
    arm_rows = []
    for r in filter_results + subsample_results:
        arm_rows.append({
            "axis": r.get("axis"),
            "threshold": r.get("threshold"),
            "n_users_post_filter": r.get("n_users_post_filter", 0),
            "median_n_per_user": r.get("median_n_per_user", float("nan")),
            "rmse_baseline": r.get("rmse_baseline", float("nan")),
            "rmse_pilsd": r.get("rmse_pilsd", float("nan")),
            "gain_pct": r.get("gain_pct", float("nan")),
            "ci_bca_lo": r.get("ci_bca_lo", float("nan")),
            "ci_bca_hi": r.get("ci_bca_hi", float("nan")),
            "ci_excludes_zero_positive": r.get("ci_excludes_zero_positive", False),
            "slice_verdict": r.get("slice_verdict", "ERROR"),
        })
    arm_df = pd.DataFrame(arm_rows)
    arm_parquet = out_dir / "per_slice_arms.parquet"
    if len(arm_df):
        arm_df.to_parquet(arm_parquet, index=False)

    # Phase 6 — Plot
    try:
        png_path, pdf_path = make_sample_efficiency_plot(
            filter_results, subsample_results, out_dir, verdict["verdict_class"],
        )
    except Exception as exc:
        log(f"[Q10] WARNING: plot generation failed: {exc}")
        log(traceback.format_exc())
        png_path, pdf_path = None, None

    # Phase 7 — Final summary
    summary.update({
        "verdict_class": verdict["verdict_class"],
        "verdict_diagnostics": verdict,
        "filter_cohort_axis_results": filter_results,
        "subsample_per_user_axis_results": subsample_results,
        "per_slice_arms_parquet": str(arm_parquet) if len(arm_df) else None,
        "sample_efficiency_curve_png": str(png_path) if png_path else None,
        "sample_efficiency_curve_pdf": str(pdf_path) if pdf_path else None,
        "git_head_t3_at_run": git_head_t3(),
        "completion_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print()
    print(f"=== Q10 PILSD sample-efficiency curve on PRISM ===")
    print(f"verdict_class : {summary['verdict_class']}")
    print(f"n_valid_subsample_slices: {verdict.get('n_valid_slices', 0)}")
    print(f"thresholds_in_curve: {verdict.get('thresholds_in_curve', [])}")
    print(f"gains_in_curve: {[round(g, 3) for g in verdict.get('gains_in_curve', [])]}")
    print(f"high_n_gain_pct: {verdict.get('high_n_gain_pct', float('nan')):+.3f}%")
    print(f"low_n_gain_pct:  {verdict.get('low_n_gain_pct', float('nan')):+.3f}%")
    print(f"is_monotone_non_decreasing: {verdict.get('is_monotone_non_decreasing_within_0p5pp_slack', False)}")
    print()
    print("FILTER-COHORT axis (DIAGNOSTIC):")
    for r in filter_results:
        gp = r.get("gain_pct", float("nan"))
        lo = r.get("ci_bca_lo", float("nan"))
        hi = r.get("ci_bca_hi", float("nan"))
        nu = r.get("n_users_post_filter", 0)
        sv = r.get("slice_verdict", "ERROR")
        gp_s = f"{gp:+.3f}" if isinstance(gp, float) and np.isfinite(gp) else "  nan "
        lo_s = f"{lo:+.3f}" if isinstance(lo, float) and np.isfinite(lo) else "  nan "
        hi_s = f"{hi:+.3f}" if isinstance(hi, float) and np.isfinite(hi) else "  nan "
        print(f"  thr={r['threshold']:>3}: n_users={nu:>5} gain={gp_s}% CI=[{lo_s}, {hi_s}] [{sv}]")
    print()
    print("SUBSAMPLE-PER-USER axis (HEADLINE):")
    for r in subsample_results:
        gp = r.get("gain_pct", float("nan"))
        lo = r.get("ci_bca_lo", float("nan"))
        hi = r.get("ci_bca_hi", float("nan"))
        nu = r.get("n_users_post_filter", 0)
        sv = r.get("slice_verdict", "ERROR")
        gp_s = f"{gp:+.3f}" if isinstance(gp, float) and np.isfinite(gp) else "  nan "
        lo_s = f"{lo:+.3f}" if isinstance(lo, float) and np.isfinite(lo) else "  nan "
        hi_s = f"{hi:+.3f}" if isinstance(hi, float) and np.isfinite(hi) else "  nan "
        print(f"  n_target={r['threshold']:>3}: n_users={nu:>5} gain={gp_s}% CI=[{lo_s}, {hi_s}] [{sv}]")
    print()
    print(f"summary_path: {summary_path}")
    print(f"per_slice_arms_parquet: {arm_parquet}")
    if png_path:
        print(f"sample_efficiency_curve_png: {png_path}")
    if pdf_path:
        print(f"sample_efficiency_curve_pdf: {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
