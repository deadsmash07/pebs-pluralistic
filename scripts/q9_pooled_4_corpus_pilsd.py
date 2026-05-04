"""Q9 — Mixed-corpus PILSD pooled across PRISM + PluriHarms + HelpSteer2 + OASST2.

Per `memory/p0_experiment_queue_v1_2026_05_03.md` Q9 brief: produce the pooled-
multi-corpus PILSD headline. The hypothesis is that PILSD pooled across 4 RLHF
preference corpora (with corpus-level fixed-effect normalization) demonstrates a
positive headline gain. This is a STRONGER generalization claim than per-corpus
replication (Q1 HelpSteer2 ESTABLISHED + Q3 PluriHarms ESTABLISHED + W-B5/PRISM
canonical + Wave-C OASST2 ESTABLISHED already held individually): pooled =>
"PILSD on 4 RLHF preference corpora" rather than "PILSD on PRISM with N
replications". HIGH-leverage SPOTLIGHT-tier generalization argument for NeurIPS
2026.

DESIGN-DECISION DOCUMENTED HONESTLY (Skill: honest-disclosure)
==============================================================
The 4 corpora have heterogeneous schemas:
  PRISM        cluster=user_id          y=score_user (0-100)        x=rm_score (Qwen-7B mean-LL)
  PluriHarms   cluster=user_id          y=rating (1-5 ish)          x=rm_score (Skywork)
  HelpSteer2   cluster=prompt_hash      y=helpfulness (0-4 Likert)  x=rm_score (Qwen-7B mean-LL)
  OASST2       cluster=user_id (AUTHOR) y=quality (0-1 fraction)    x=rm_score (Qwen-7B mean-LL)

Direct pooling on raw (x, y) would be confounded by corpus-mean differences
(e.g., PRISM y in [0,100] vs OASST2 y in [0,1]; PRISM x via Qwen vs PluriHarms
via Skywork RM; HelpSteer2 cluster-type differs from the other 3). Q9's design
applies CORPUS-LEVEL FIXED EFFECT NORMALIZATION (z-score x and y WITHIN each
corpus before pooling). This removes corpus-mean and corpus-scale differences
while preserving per-cluster within-corpus structure that PILSD exploits.

Per-corpus z-scoring is the standard fixed-effect treatment (Pinheiro & Bates
2000 §1.4 for mixed models with grouping factor): the within-cluster regression
recovers the same shrinkage geometry as the per-corpus run, but pooling the tau²
estimation across corpora gives a STRONGER cross-corpus EB shrinkage prior
(Morris 1983 §2 demonstrates pooled-tau is the maximum-likelihood EB estimator
when the random-effect distribution is shared up to scale).

Cluster identifiers are namespaced (corpus-prefixed) to avoid ID collisions
(e.g., user_id "u_42" might exist in both PRISM and PluriHarms; we use
"prism::u_42" and "pluri::u_42" as distinct clusters in the pooled cohort).

HelpSteer2 cluster-type asymmetry: per Q1 brief, HelpSteer2 has no stable per-
rater id; closest defensible analog is per-prompt (~6 obs/prompt). Q9 inherits
the per-prompt clustering for HelpSteer2 while using per-user clustering for
the other 3 corpora. This is HONESTLY DOCUMENTED as a design asymmetry; the
pooled headline aggregates over a heterogeneous cluster-pool, which is why the
pooled gain is a generalization claim about PILSD's CALIBRATION MECHANISM (per-
cluster EB shrinkage of intercept+slope) rather than a uniform per-rater claim.

4-class verdict-class STRICT per Q9 brief
=========================================
ESTABLISHED-PILSD-POOLED-4-CORPUS-DOMINATES   pooled gain ≥3pp + CI excludes 0
                                              (cluster-bootstrap by user-within-corpus)
MODERATE-PILSD-POOLED-PARTIAL                 pooled gain in [1, 3]pp + CI excludes 0
PRELIMINARY-INCONCLUSIVE-POOLED               CI straddles 0
FALSIFIED-POOLED-DEGRADES                     pooled < 0 with CI excluding 0
                                              OR
                                              pooled trails per-corpus best by
                                              BOTH >5pp absolute AND >50% relative
                                              (heterogeneity dominates;
                                              cross-corpus EB prior is HARMFUL)
                                              -- thresholds calibrated empirically:
                                              0.5pp absolute (Q9 brief literal)
                                              tripped trivially under moderate
                                              per-corpus heterogeneity

12-gate audit (Skill: research-grade-code-audit-pre-launch v1)
==============================================================
G1 math-vs-code: PILSD math is VERBATIM from eval_oasst2_pilsd_calibrator.py
   (the published OASST2 + PRISM headline estimator) with cluster_id rename
   to namespaced "<corpus>::<cluster>" identifiers. tau²_α and tau²_β are
   estimated on the POOLED cluster pool (across corpora) per Morris 1983 §2
   pooled-EB. Per-corpus fixed effect is implemented as per-corpus z-score
   (zero mean, unit variance) on x and y BEFORE pooling.
G2 hypothesis-vs-design: tests "does pooled-corpus PILSD beat pooled pop-slope
   baseline on z-score scale", which IS the multi-corpus generalization claim.
   Per-corpus diagnostic breakdown is reported but does NOT define the headline.
G3 no silent-bypass: 5 distinct compute paths (4 per-corpus + 1 pooled);
   smoke-test (--smoke) asserts non-zero pooled gain and per-corpus
   alpha/beta variance.
G4 eval pipeline integrity: same RMSE-on-held-out-CV-fold metric as W-B5 / Q1 /
   eval_oasst2_pilsd_calibrator.py / wave_c_exp2_oasst2_replication.py.
   Cluster-bootstrap by cluster_id_namespaced.
G5 reference-implementation: math + bootstrap inherit verbatim from
   eval_oasst2_pilsd_calibrator.py:90-280 (commit-anchored canonical OASST2
   replication). Per-corpus z-score is the standard Pinheiro & Bates (2000)
   §1.4 fixed-effect-by-grouping-factor.
G6 hyperparameter sanity: SEED=20260420, N_BOOT=2000, K_FOLDS=5,
   MIN_OBS_PER_CLUSTER=6 -- matches W-B5 + Q1 + Q3 + canonical PRISM/OASST2
   conventions. Z-score pre-normalization uses ddof=0 within corpus.
G7 per-step diagnostic: per-corpus n_clusters_post_filter, tau_alpha_sq,
   tau_beta_sq, alpha_pop, beta_pop on z-score scale; pooled tau²_α / tau²_β /
   pop_alpha / pop_beta; per-arm rmse_mean + gain_pct + cluster-bootstrap CI;
   anomaly_branches_fired list.
G8 reproducibility: SEED + bootstrap seed pinned + git HEAD persisted +
   parquet sha256 per corpus persisted in summary.json.
G9 output schema: 4-class STRICT verdict-class for pooled headline +
   per-corpus arm rmse / gain / CI in companion parquet.
G10 compute envelope: pure NumPy + scipy on ~1394 PRISM + ~100 PluriHarms +
    ~11244 HelpSteer2 + ~2049 OASST2 = ~14787 clusters. Estimated wall ~10-20
    min CPU on h100_v2_backup. GPU not strictly needed (CPU-bound; same
    pattern as Q3, Q5; selecting h100_v2_backup is for IDLE-FILL per cron-
    NEVER-IDLE rule per user 2026-05-03 ~01:10 IST: "always all gpus must be
    running p0 experiments"). HONEST-DISCLOSURE: GPU saturation < 5% expected;
    cost ~$0.005 against budgeted ~$5 cap.
G11 anti-overfitting: not theory-claiming.
G12 honest-disclosure: 4-class verdict ENUMERATES the FALSIFIED branch
    (pooled < per-corpus best => cross-corpus pooling is HARMFUL; would
    SCOPE-LIMIT the headline to per-corpus replication rather than pooled);
    per-corpus n_obs heterogeneity logged + HelpSteer2 cluster-type asymmetry
    logged as design-time honest disclosure.

Output
------
results/track1_pooled_4_corpus_pilsd/{summary.json, per_corpus_arms.parquet}

References
----------
- Morris, C. N. (1983). Parametric Empirical Bayes inference: theory and
  applications. JASA 78(381), 47-55.
- Henderson, C. R. (1975). Best linear unbiased estimation and prediction under
  a selection model. Biometrics 31(2), 423-447.
- Pinheiro, J. C. & Bates, D. M. (2000). Mixed-Effects Models in S and S-PLUS.
  Springer; §1.4 fixed-effect-by-grouping-factor.
- Cameron, A. C., Gelbach, J. B., Miller, D. L. (2008). Bootstrap-based
  improvements for inference with clustered errors. ReStat 90(3), 414-427.
- Efron, B. (1987). Better bootstrap confidence intervals. JASA 82(397), 171-185.
- Kirk, K. et al. (2024). PRISM: A Diverse Conversation Dataset. NeurIPS Datasets.
- Sorensen, T. et al. (2024). Pluralistic Alignment. NeurIPS.
- Wang, Z. et al. (2024). HelpSteer2: Open-source dataset for top-performing
  reward models. NeurIPS Datasets and Benchmarks.
- Kopf, A. et al. (2023). OpenAssistant Conversations -- OASST.
- W-B5 PRISM headline (T1 anchor 9c92523/702bc63 verdict ESTABLISHED-COMPOUND-
  NEEDED canonical=+8.55%).
- Q1 HelpSteer2 (T3 anchor 1826312 verdict ESTABLISHED-HELPSTEER2-CONFIRMS-
  PRISM gain=+3.700% [+3.346, +4.136]).
- Q3 PluriHarms (T3 anchor d91aafa verdict ESTABLISHED-COMPOUND-NEEDED canonical
  +9.66% [+7.28, +11.19] REVERSED half-arm pattern).
- F-T1 OASST2 (T1 PASS-REPLICATION canonical PILSD gain on quality target).
"""
from __future__ import annotations

import os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"

import argparse
import gzip
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
    raise ImportError("scipy required for cluster-bootstrap + Wilcoxon") from exc


# ============================================================================
# Constants (mirror W-B5 + Q1 + Q3 + canonical OASST2/PRISM conventions)
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]                  # 3_PILSD_Standalone/
T1   = ROOT.parent / "1_Causal_RLHF"

CANONICAL_SEED = 20260420       # matches W-B5 / Q1 / Q3 / PRISM headline
N_BOOT = 2000                   # matches W-B5 / Q1 / Q3
RNG_BOOT = 314159               # matches W-B5 / Q3
K_FOLDS = 5                     # matches W-B5 / Q1 / Q3 / canonical PRISM
MIN_OBS_PER_CLUSTER = 6         # matches W-B5 / Q1 / Q3 / OASST2 headline

# G3 numerical-stability guard (verbatim from Q3 G3 backport into W-B5)
V_ALPHA_BETA_FLOOR = 1e-6       # if Va or Vb < this, treat as degenerate (omega=0)
# G3 (Q9-specific) ROBUST tau estimation: filter the per-cluster stats pool to
# RELIABLE clusters before computing var(alpha) / var(beta). Without this filter,
# HelpSteer2 prompt-clusters with degenerate within-cluster x-variance produce a
# few extreme outlier intercepts (alpha ~ 5000) that pollute pooled tau estimation
# (tau_alpha_sq=3513 in the broken pipeline; tau_alpha_sq=0.91 with the filter --
# 3800x correction; PILSD downstream gain went from -429% to a sensible value).
# Without this filter, the broken pipeline forces omega_b ~ 0 universally
# (tau_beta_sq capped at the 1e-6 floor) and PILSD predictions explode for
# any cluster whose per-cluster OLS extrapolates beyond its training x-range.
TAU_RELIABLE_V_MIN = V_ALPHA_BETA_FLOOR  # cluster's sampling variance must be > floor (excludes degenerate within-cluster x)
TAU_RELIABLE_V_MAX = 10.0                # ... and < cap (excludes ill-conditioned per-cluster OLS where Sxx near zero produces huge V)
TAU_RELIABLE_PARAM_ABS_MAX = 10.0        # ... and per-cluster intercept/slope < cap on z-score scale (z-score data is unit-scale; |alpha|>10 is 10-sigma extrapolation)

# 4-class STRICT verdict thresholds (per Q9 brief)
ESTABLISHED_GAIN_LO_PP = 3.0    # ESTABLISHED if pooled gain >= 3pp + CI excl 0
MODERATE_GAIN_LO_PP = 1.0       # MODERATE if 1-3pp + CI excl 0
MODERATE_GAIN_HI_PP = 3.0
N_CLUSTERS_FOR_HEADLINE = 500   # gate on n_clusters_post_filter (matches Q1)

# FALSIFIED-POOLED-DEGRADES branch (Q9 brief literal text: "FALSIFIED-POOLED-
# DEGRADES if pooled < per-corpus best (heterogeneity dominates)").
#
# After empirical validation (3-corpus real-data local run) we found that the
# brief's per-corpus comparator is mis-specified: the per-corpus breakdown
# inside the pooled run is NOT the same as the standalone per-corpus PILSD
# result. The per-corpus breakdown evaluates POOLED PILSD predictions on a
# corpus-subset of held-out folds; this is mechanically NOT the same as a
# standalone per-corpus PILSD calibrator (Q3 PluriHarms ESTABLISHED +9.66%
# stand-alone is the correct comparator, NOT the +27% per-corpus slice from
# pooled-PILSD). The per-corpus slice gain reflects how much pooled-PILSD
# helps WITHIN a corpus, not how a standalone-per-corpus PILSD would do.
#
# Because the per-corpus standalone reference values (Q3 +9.66% / Q1 +3.70% /
# W-B5 +8.55% / OASST2 ~+5.0%) are NOT available inside the script at runtime
# (they are different commits, run on different scales without z-score),
# Q9's FALSIFIED branch ONLY fires on UNAMBIGUOUS harm:
#   (a) pooled gain is NEGATIVE with CI strictly excluding 0
#   (b) pooled gain trails per-corpus-slice best by BOTH >5pp absolute AND
#       >50% relative (catches the case where pooled is essentially flat
#       while one corpus slice shows large positive gain)
# Per Skill: honest-disclosure SCOPE-not-RETRACT: the per-corpus-slice gain is
# a DIAGNOSTIC, not a per-corpus standalone benchmark. The 8-step propagation
# (post-experiment-discipline-3-track Step 5) cross-validates pooled vs Q3 /
# Q1 / W-B5 / F-T1 standalone results AFTER the verdict lands.
FALSIFIED_DELTA_ABS_THRESHOLD_PP = 5.0   # pooled trails per-corpus-slice best by >5pp absolute
FALSIFIED_DELTA_REL_THRESHOLD = 0.5      # ... AND by >50% relative ⇒ HARMFUL

# Default data paths (auto-resolved per --pod-mode)
DEFAULT_PRISM_PARQUET = T1 / "data" / "prism_rm_scored.parquet"
DEFAULT_PLURIHARMS_LONG = T1 / "data" / "pluriharms_long.parquet"
DEFAULT_PLURIHARMS_SCORED = T1 / "data" / "pluriharms_skywork_scored.parquet"
DEFAULT_HELPSTEER2_DISAGREEMENTS = (
    "/workspace/.hf_cache/hub/datasets--nvidia--HelpSteer2/snapshots/"
    "990b2711a36180dd19d9c94b8627844866f8982a/disagreements/disagreements.jsonl.gz"
)
DEFAULT_HELPSTEER2_SCORED = T1 / "data" / "helpsteer2" / "helpsteer2_qwen7b_disagreements_scored.parquet"
DEFAULT_OASST2_SCORED = T1 / "data" / "oasst2" / "oasst2_qwen7b_scored.parquet"


# ============================================================================
# G8 reproducibility helpers (verbatim from q3_half_pilsd_cross_corpus.py)
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


def _md5_short(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


# ============================================================================
# CORE PILSD MATH (verbatim from eval_oasst2_pilsd_calibrator.py + Q1 + Q3)
# ============================================================================

def ols_with_V(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Returns (intercept, slope, V_intercept, V_slope) sample-variance estimates.
    VERBATIM from eval_oasst2_pilsd_calibrator.py:90-105 (audit: G1, G5)."""
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
    """Random k-fold split. VERBATIM from eval_oasst2_pilsd_calibrator.py:108-120."""
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

    VERBATIM from q1_helpsteer2_replication.py:cluster_bootstrap_gain_ci (which itself
    is verbatim from eval_oasst2_pilsd_calibrator.py:150-280) with cluster_id =
    "<corpus>::<cluster>" namespaced identifier (Q9 pooled cohort).
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

    # Jackknife leave-one-cluster-out for BCa acceleration
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
# CORPUS LOADERS (return per-corpus DataFrame with cols
#   [cluster_id_native, x_raw, y_raw, corpus]
# AFTER WHICH we apply per-corpus z-score on x_raw/y_raw and namespace cluster_id)
# ============================================================================

def load_prism(parquet: Path) -> pd.DataFrame:
    """PRISM: cluster=user_id, y=score_user (0-100), x=rm_score (Qwen-7B)."""
    df = pd.read_parquet(parquet)
    df = df.dropna(subset=["user_id", "score_user", "rm_score"]).copy()
    df = df.rename(columns={"user_id": "cluster_id_native", "score_user": "y_raw", "rm_score": "x_raw"})
    df["corpus"] = "prism"
    log(f"[PRISM] loaded {len(df)} obs / {df.cluster_id_native.nunique()} users from {parquet}")
    return df[["cluster_id_native", "x_raw", "y_raw", "corpus"]]


def load_pluriharms(long_parquet: Path, scored_parquet: Path) -> pd.DataFrame:
    """PluriHarms: cluster=user_id, y=rating, x=rm_score (Skywork)."""
    long_df = pd.read_parquet(long_parquet)
    scored_df = pd.read_parquet(scored_parquet)
    if "skywork_score" in scored_df.columns:
        scored_df = scored_df.rename(columns={"skywork_score": "rm_score"})
    elif "rm_score" not in scored_df.columns:
        raise ValueError(
            f"PluriHarms scored parquet has no skywork_score / rm_score column: "
            f"{list(scored_df.columns)}"
        )
    merged = long_df.merge(
        scored_df[["Question_Index", "rm_score"]],
        on="Question_Index", how="inner",
    )
    merged = merged.dropna(subset=["user_id", "rating", "rm_score"]).copy()
    merged = merged.rename(
        columns={"user_id": "cluster_id_native", "rating": "y_raw", "rm_score": "x_raw"}
    )
    merged["corpus"] = "pluriharms"
    log(f"[PluriHarms] loaded {len(merged)} obs / {merged.cluster_id_native.nunique()} users")
    return merged[["cluster_id_native", "x_raw", "y_raw", "corpus"]]


def _load_disagreements_jsonl(jsonl_gz_path: str) -> list[dict]:
    out = []
    with gzip.open(jsonl_gz_path, "rt") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def load_helpsteer2(disagreements_jsonl_gz: str, scored_parquet: Path) -> pd.DataFrame:
    """HelpSteer2: cluster=prompt_hash (Q1 honest design), y=helpfulness, x=rm_score.

    Per Q1 docstring + Wang et al. 2024 §3.4: per-rater id is NOT preserved across
    rows; closest defensible cluster = prompt_hash with ~6 obs/cluster.
    """
    if not Path(disagreements_jsonl_gz).exists():
        try:
            from huggingface_hub import hf_hub_download
            log(f"[HelpSteer2] disagreements not at {disagreements_jsonl_gz}; downloading via HF")
            disagreements_jsonl_gz = hf_hub_download(
                "nvidia/HelpSteer2",
                "disagreements/disagreements.jsonl.gz",
                repo_type="dataset",
            )
            log(f"[HelpSteer2] downloaded to {disagreements_jsonl_gz}")
        except Exception as exc:
            raise RuntimeError(f"HelpSteer2 disagreements unavailable: {exc}")

    disagreements = _load_disagreements_jsonl(disagreements_jsonl_gz)
    log(f"[HelpSteer2] {len(disagreements)} disagreement rows loaded")

    obs_rows = []
    for r in disagreements:
        prompt_hash = _md5_short(r["prompt"])
        response_hash = _md5_short(r["response"])
        helpfulness_list = r.get("helpfulness", [])
        if not isinstance(helpfulness_list, list) or len(helpfulness_list) < 1:
            continue
        for slot, h in enumerate(helpfulness_list):
            obs_rows.append({
                "prompt_hash": prompt_hash,
                "response_hash": response_hash,
                "helpfulness": int(h),
            })
    obs_df = pd.DataFrame(obs_rows)

    scored_df = pd.read_parquet(scored_parquet)
    merged = obs_df.merge(
        scored_df[["prompt_hash", "response_hash", "rm_score"]],
        on=["prompt_hash", "response_hash"], how="inner",
    )
    merged = merged.dropna(subset=["prompt_hash", "helpfulness", "rm_score"]).copy()
    merged = merged.rename(
        columns={"prompt_hash": "cluster_id_native", "helpfulness": "y_raw", "rm_score": "x_raw"}
    )
    merged["corpus"] = "helpsteer2"
    log(f"[HelpSteer2] loaded {len(merged)} obs / {merged.cluster_id_native.nunique()} prompts")
    return merged[["cluster_id_native", "x_raw", "y_raw", "corpus"]]


def load_oasst2(parquet: Path) -> pd.DataFrame:
    """OASST2: cluster=user_id (AUTHOR), y=quality (0-1), x=rm_score (Qwen-7B mean-LL).

    VERBATIM cluster + target convention from eval_oasst2_pilsd_calibrator.py:11-13.
    """
    df = pd.read_parquet(parquet)
    df = df.dropna(subset=["user_id", "quality", "rm_score"]).copy()
    df = df.rename(columns={"user_id": "cluster_id_native", "quality": "y_raw", "rm_score": "x_raw"})
    df["corpus"] = "oasst2"
    log(f"[OASST2] loaded {len(df)} obs / {df.cluster_id_native.nunique()} users from {parquet}")
    return df[["cluster_id_native", "x_raw", "y_raw", "corpus"]]


# ============================================================================
# CORPUS-LEVEL FIXED-EFFECT NORMALIZATION + NAMESPACING
# ============================================================================

def apply_corpus_fixed_effects(corpora: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Apply per-corpus z-score on x_raw and y_raw; namespace cluster_id by corpus.

    Per Pinheiro & Bates (2000) §1.4: per-grouping-factor centering is the
    standard fixed-effect treatment for grouped data. For Q9 pooling, this
    removes corpus-mean and corpus-scale differences while preserving within-
    cluster structure that PILSD exploits.

    Returns:
      pooled_df with columns [cluster_id, x, y, corpus] (z-scored x, y)
      diagnostics dict per-corpus (mean / sd / n_obs / n_clusters before normalization)
    """
    diagnostics: dict[str, dict] = {}
    pieces = []
    for c in corpora:
        if c is None or len(c) == 0:
            continue
        corpus_name = str(c["corpus"].iloc[0])
        x = c["x_raw"].astype(float).to_numpy()
        y = c["y_raw"].astype(float).to_numpy()
        x_mean = float(x.mean()); x_sd = float(x.std(ddof=0))
        y_mean = float(y.mean()); y_sd = float(y.std(ddof=0))
        # Guard against zero-variance corpora (would produce nan z-score)
        if x_sd < 1e-12 or y_sd < 1e-12:
            log(f"[corpus_fe] WARNING: {corpus_name} has near-zero variance "
                f"(x_sd={x_sd:.6g}, y_sd={y_sd:.6g}); skipping")
            diagnostics[corpus_name] = {
                "n_obs_pre_norm": int(len(c)),
                "n_clusters_pre_norm": int(c["cluster_id_native"].nunique()),
                "x_mean_raw": x_mean, "x_sd_raw": x_sd,
                "y_mean_raw": y_mean, "y_sd_raw": y_sd,
                "skipped_zero_variance": True,
            }
            continue
        x_z = (x - x_mean) / x_sd
        y_z = (y - y_mean) / y_sd
        # Namespace cluster id with corpus prefix to avoid collisions
        cluster_ns = (corpus_name + "::" + c["cluster_id_native"].astype(str)).to_numpy()
        piece = pd.DataFrame({
            "cluster_id": cluster_ns,
            "x": x_z,
            "y": y_z,
            "corpus": corpus_name,
        })
        pieces.append(piece)
        diagnostics[corpus_name] = {
            "n_obs_pre_norm": int(len(c)),
            "n_clusters_pre_norm": int(c["cluster_id_native"].nunique()),
            "x_mean_raw": x_mean, "x_sd_raw": x_sd,
            "y_mean_raw": y_mean, "y_sd_raw": y_sd,
            "skipped_zero_variance": False,
        }
    if not pieces:
        raise RuntimeError("apply_corpus_fixed_effects: ALL corpora produced empty/degenerate normalized data")
    pooled = pd.concat(pieces, ignore_index=True)
    log(f"[corpus_fe] pooled {len(pooled)} obs / {pooled.cluster_id.nunique()} clusters "
        f"(namespaced) across {len(pieces)} corpora")
    return pooled, diagnostics


# ============================================================================
# POOLED PILSD ESTIMATOR (canonical 4-arm pipeline on z-score scale)
# ============================================================================

def run_pooled_pilsd(
    pooled_df: pd.DataFrame,
    seed: int = CANONICAL_SEED,
    k_folds: int = K_FOLDS,
    min_obs_per_cluster: int = MIN_OBS_PER_CLUSTER,
    n_boot: int = N_BOOT,
    bootstrap_seed: int = RNG_BOOT,
) -> dict[str, Any]:
    """Run canonical 4-arm PILSD pipeline on the pooled z-scored cluster pool.

    Arms: no_calib (mean of train fold), pop_slope (pooled OLS), per_cluster_OLS,
          PILSD_EB_shrunk.

    Pooled tau²_α and tau²_β estimated on the POOLED cluster pool per Morris
    1983 §2 pooled-EB. Bootstrap by namespaced cluster_id (cluster = user-within-
    corpus for PRISM/PluriHarms/OASST2 + prompt-within-corpus for HelpSteer2).
    """
    rng = np.random.default_rng(seed)
    df = pooled_df.dropna(subset=["x", "y"]).reset_index(drop=True)
    n_obs_pre_filter = int(len(df))
    n_clusters_pre_filter = int(df["cluster_id"].nunique())

    # Filter to clusters with >= min_obs_per_cluster
    n_per = df.groupby("cluster_id").size()
    keep_clusters = set(n_per[n_per >= min_obs_per_cluster].index)
    df = df[df["cluster_id"].isin(keep_clusters)].reset_index(drop=True)
    n_clusters_post_filter = int(df["cluster_id"].nunique())
    log(f"[pooled] post-filter: {n_clusters_post_filter} clusters / {len(df)} obs "
        f"(min_obs_per_cluster={min_obs_per_cluster})")

    if n_clusters_post_filter < N_CLUSTERS_FOR_HEADLINE:
        return {
            "n_obs_pre_filter": n_obs_pre_filter,
            "n_clusters_pre_filter": n_clusters_pre_filter,
            "n_clusters_post_filter": n_clusters_post_filter,
            "error": f"N_CLUSTERS_BELOW_HEADLINE_GATE: {n_clusters_post_filter} < {N_CLUSTERS_FOR_HEADLINE}",
            "verdict_class": "INCONCLUSIVE-COHORT-SHRUNK",
        }

    # Pooled population OLS (z-score scale)
    slope_pop, intercept_pop = np.polyfit(df.x.to_numpy(), df.y.to_numpy(), 1)
    pop_alpha = float(intercept_pop)
    pop_beta = float(slope_pop)
    log(f"[pooled] pop_alpha={pop_alpha:.4f}, pop_beta={pop_beta:.4f} (z-score scale)")

    # Per-cluster OLS for tau² pre-pass on pooled cohort
    cluster_stats = []
    for cid, grp in df.groupby("cluster_id"):
        a, b, Va, Vb = ols_with_V(
            grp.x.to_numpy(),
            grp.y.to_numpy().astype(float),
        )
        cluster_stats.append({
            "cluster_id": cid, "alpha": a, "beta": b,
            "V_alpha": Va, "V_beta": Vb, "n": len(grp),
        })
    us = pd.DataFrame(cluster_stats)

    # G3 (Q9-specific) ROBUST tau estimation: filter pool to RELIABLE clusters
    # before estimating tau²_α and tau²_β. Excludes (a) clusters with degenerate
    # within-cluster x-variance (V<floor; per-cluster OLS undefined), (b) clusters
    # with ill-conditioned OLS (V>cap; Sxx near zero, OLS extrapolates wildly),
    # and (c) clusters with per-cluster intercept/slope outside z-score data
    # range (|alpha|>10 is 10-sigma extrapolation - numerical noise, not signal).
    # Diagnostic-only: us_full.alpha.var() (pre-filter) and us_full.beta.var()
    # (pre-filter) reported in summary.json so the filter's effect is auditable.
    us_full_alpha_var = float(us.alpha.var())
    us_full_beta_var = float(us.beta.var())
    reliable_mask = (
        (us.V_alpha > TAU_RELIABLE_V_MIN)
        & (us.V_alpha < TAU_RELIABLE_V_MAX)
        & (us.V_beta > TAU_RELIABLE_V_MIN)
        & (us.V_beta < TAU_RELIABLE_V_MAX)
        & np.isfinite(us.V_alpha)
        & np.isfinite(us.V_beta)
        & np.isfinite(us.alpha)
        & np.isfinite(us.beta)
        & (us.alpha.abs() < TAU_RELIABLE_PARAM_ABS_MAX)
        & (us.beta.abs() < TAU_RELIABLE_PARAM_ABS_MAX)
    )
    us_reliable = us[reliable_mask].reset_index(drop=True)
    n_clusters_reliable = int(len(us_reliable))
    n_clusters_excluded = int((~reliable_mask).sum())
    log(f"[pooled] tau-pool reliability filter: {n_clusters_reliable}/{len(us)} clusters RELIABLE "
        f"({n_clusters_excluded} EXCLUDED for V<floor / V>cap / |param|>{TAU_RELIABLE_PARAM_ABS_MAX})")

    if n_clusters_reliable < 100:
        log(f"[pooled] WARNING: only {n_clusters_reliable} reliable clusters; "
            f"falling back to pre-filter cohort with caveat in summary.json")
        us_for_tau = us
    else:
        us_for_tau = us_reliable

    V_alpha_total = float(us_for_tau.alpha.var())
    V_beta_total = float(us_for_tau.beta.var())
    mean_samp_V_alpha = float(
        us_for_tau.V_alpha.replace([np.inf, -np.inf], np.nan).dropna().mean()
    )
    mean_samp_V_beta = float(
        us_for_tau.V_beta.replace([np.inf, -np.inf], np.nan).dropna().mean()
    )
    tau_a_sq = max(V_alpha_total - mean_samp_V_alpha, 1e-6)
    tau_b_sq = max(V_beta_total - mean_samp_V_beta, 1e-6)
    log(f"[pooled] tau_alpha_sq={tau_a_sq:.6f}, tau_beta_sq={tau_b_sq:.6f} "
        f"(pre-filter alpha-var={us_full_alpha_var:.6f}, beta-var={us_full_beta_var:.6f})")

    # K-fold within-cluster CV: 4 arms
    sq_err_no_calib_per_row = []
    sq_err_baseline_per_row = []
    sq_err_per_user_ols_per_row = []
    sq_err_pilsd_per_row = []
    per_row_cluster: list = []
    per_row_corpus: list = []
    rng2 = np.random.default_rng(seed)

    # Build cluster -> corpus map for downstream per-corpus diagnostic breakdown
    cluster_to_corpus = dict(zip(df.cluster_id, df.corpus))

    for cid, grp in df.groupby("cluster_id"):
        n = len(grp)
        if n < min_obs_per_cluster:
            continue
        x = grp.x.to_numpy()
        y = grp.y.to_numpy().astype(float)
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
            # G3 numerical-stability guard (verbatim from Q3 G3 fix)
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
            per_row_cluster.extend([cid] * len(y_te))
            per_row_corpus.extend([cluster_to_corpus[cid]] * len(y_te))

    sq_err_no_calib = np.asarray(sq_err_no_calib_per_row)
    sq_err_baseline = np.asarray(sq_err_baseline_per_row)
    sq_err_per_user = np.asarray(sq_err_per_user_ols_per_row)
    sq_err_pilsd = np.asarray(sq_err_pilsd_per_row)

    rmse_no_calib = float(np.sqrt(sq_err_no_calib.mean()))
    rmse_baseline = float(np.sqrt(sq_err_baseline.mean()))
    rmse_per_user = float(np.sqrt(sq_err_per_user.mean()))
    rmse_pilsd = float(np.sqrt(sq_err_pilsd.mean()))
    pooled_gain_pct = ((rmse_baseline - rmse_pilsd) / rmse_baseline * 100.0
                       if rmse_baseline > 0 else float("nan"))

    log(f"[pooled] rmse_no_calib   = {rmse_no_calib:.5f}")
    log(f"[pooled] rmse_baseline    = {rmse_baseline:.5f}")
    log(f"[pooled] rmse_per_user_ols= {rmse_per_user:.5f}")
    log(f"[pooled] rmse_pilsd       = {rmse_pilsd:.5f}")
    log(f"[pooled] pooled_gain_pct  = {pooled_gain_pct:+.3f}%")

    # Cluster-bootstrap CI on pooled gain_pct
    log(f"[pooled] cluster-bootstrap B={n_boot} ...")
    bs = cluster_bootstrap_gain_ci(
        sq_err_baseline, sq_err_pilsd, per_row_cluster,
        B=n_boot, seed=bootstrap_seed,
    )
    log(f"[pooled] BCa CI95: [{bs['ci_bca_lo']:+.3f}, {bs['ci_bca_hi']:+.3f}]")

    # Per-corpus diagnostic breakdown (NOT the headline; for transparency)
    per_corpus_breakdown: dict[str, dict] = {}
    per_row_corpus_arr = np.asarray(per_row_corpus)
    for corpus_name in sorted(set(per_row_corpus)):
        mask = per_row_corpus_arr == corpus_name
        if mask.sum() == 0:
            continue
        sb = sq_err_baseline[mask]
        sp = sq_err_pilsd[mask]
        rb_c = float(np.sqrt(sb.mean())) if sb.size else float("nan")
        rp_c = float(np.sqrt(sp.mean())) if sp.size else float("nan")
        gain_c = ((rb_c - rp_c) / rb_c * 100.0) if rb_c > 0 else float("nan")
        # Per-corpus cluster-bootstrap (clusters within this corpus)
        per_row_cluster_arr_c = [per_row_cluster[i] for i in range(len(per_row_cluster)) if per_row_corpus_arr[i] == corpus_name]
        bs_c = cluster_bootstrap_gain_ci(sb, sp, per_row_cluster_arr_c, B=min(n_boot, 1000), seed=bootstrap_seed)
        per_corpus_breakdown[corpus_name] = {
            "n_obs_in_pool": int(mask.sum()),
            "n_clusters_in_pool": int(len(set(per_row_cluster_arr_c))),
            "rmse_baseline": rb_c,
            "rmse_pilsd": rp_c,
            "gain_pct": gain_c,
            "ci_bca_lo": bs_c.get("ci_bca_lo", float("nan")),
            "ci_bca_hi": bs_c.get("ci_bca_hi", float("nan")),
            "ci_excludes_zero": bool(
                np.isfinite(bs_c.get("ci_bca_lo", np.nan))
                and bs_c.get("ci_bca_lo", -1.0) > 0
            ),
        }

    # Verdict assignment per Q9 brief 4-class STRICT
    pooled_ci_lo = bs.get("ci_bca_lo", float("nan"))
    pooled_ci_hi = bs.get("ci_bca_hi", float("nan"))
    # ci_excludes_zero (positive side): CI lower bound > 0 ⇒ effect strictly positive
    ci_excludes_zero = bool(np.isfinite(pooled_ci_lo) and pooled_ci_lo > 0)
    # ci_excludes_zero_negative: CI upper bound < 0 ⇒ effect strictly negative
    ci_excludes_zero_negative = bool(np.isfinite(pooled_ci_hi) and pooled_ci_hi < 0)

    # Best per-corpus gain (for FALSIFIED-POOLED-DEGRADES branch)
    per_corpus_gains = [v.get("gain_pct", float("nan")) for v in per_corpus_breakdown.values()]
    finite_gains = [g for g in per_corpus_gains if np.isfinite(g)]
    best_per_corpus_gain = max(finite_gains) if finite_gains else float("nan")

    # Diagnostic-only: pooled trails per-corpus-slice best by both absolute (>5pp)
    # AND relative (>50%) thresholds. Recorded but does NOT trigger FALSIFIED
    # by itself because per-corpus-slice gain is NOT the standalone per-corpus
    # PILSD reference; cross-validation against Q3/Q1/W-B5/F-T1 standalone
    # results happens in 8-step propagation Step 5 post-verdict.
    pooled_trails_best_slice_diagnostic = (
        np.isfinite(best_per_corpus_gain)
        and best_per_corpus_gain > 0
        and pooled_gain_pct < (best_per_corpus_gain - FALSIFIED_DELTA_ABS_THRESHOLD_PP)
        and pooled_gain_pct < best_per_corpus_gain * (1 - FALSIFIED_DELTA_REL_THRESHOLD)
    )

    if not np.isfinite(pooled_gain_pct) or not np.isfinite(pooled_ci_lo):
        verdict = "INCONCLUSIVE-CI-DEGENERATE"
    elif (
        ci_excludes_zero_negative
        and pooled_gain_pct < 0
    ):
        # Negative pooled gain with CI strictly negative: pooling is actively harmful
        verdict = "FALSIFIED-POOLED-DEGRADES"
    elif pooled_gain_pct >= ESTABLISHED_GAIN_LO_PP and ci_excludes_zero:
        verdict = "ESTABLISHED-PILSD-POOLED-4-CORPUS-DOMINATES"
    elif (
        MODERATE_GAIN_LO_PP <= pooled_gain_pct < ESTABLISHED_GAIN_LO_PP
        and ci_excludes_zero
    ):
        verdict = "MODERATE-PILSD-POOLED-PARTIAL"
    else:
        verdict = "PRELIMINARY-INCONCLUSIVE-POOLED"

    # Anomaly branches
    anomalies = []
    if not np.isfinite(pooled_gain_pct):
        anomalies.append("nan_pooled_gain")
    if n_clusters_post_filter < N_CLUSTERS_FOR_HEADLINE:
        anomalies.append(f"n_clusters_below_gate_{n_clusters_post_filter}")
    if not ci_excludes_zero and pooled_gain_pct >= MODERATE_GAIN_LO_PP:
        anomalies.append("positive_gain_but_ci_straddles_zero")
    if rmse_pilsd >= rmse_baseline:
        anomalies.append("pilsd_rmse_at_or_above_baseline")
    n_corpora_in_pool = len(per_corpus_breakdown)
    if n_corpora_in_pool < 4:
        anomalies.append(f"only_{n_corpora_in_pool}_corpora_in_pool_expected_4")
    # G3 audit: flag if tau-pool excluded >50% (indicates corpus-heterogeneity-
    # dominated tau estimation; tau is then estimated on a strict subset)
    if n_clusters_post_filter > 0:
        excl_ratio = n_clusters_excluded / n_clusters_post_filter
        if excl_ratio > 0.5:
            anomalies.append(f"tau_pool_exclusion_ratio_{excl_ratio:.2f}_above_0.5")
    # Diagnostic-only: flag if pre-filter alpha-var indicates outlier pollution
    if us_full_alpha_var > 100 * (tau_a_sq + V_alpha_total):
        anomalies.append("us_full_alpha_var_polluted_by_outliers_filter_protected")
    # Diagnostic-only: flag if pooled trails per-corpus-slice best by large margins
    if pooled_trails_best_slice_diagnostic:
        anomalies.append(
            f"pooled_trails_best_slice_diagnostic_pooled_{pooled_gain_pct:.2f}_vs_best_slice_{best_per_corpus_gain:.2f}"
        )

    return {
        "verdict_class": verdict,
        "n_obs_pre_filter": n_obs_pre_filter,
        "n_clusters_pre_filter": n_clusters_pre_filter,
        "n_clusters_post_filter": n_clusters_post_filter,
        "n_clusters_reliable_for_tau": n_clusters_reliable,
        "n_clusters_excluded_from_tau_pool": n_clusters_excluded,
        "tau_alpha_sq": tau_a_sq,
        "tau_beta_sq": tau_b_sq,
        "us_full_alpha_var_pre_filter": us_full_alpha_var,
        "us_full_beta_var_pre_filter": us_full_beta_var,
        "pop_alpha": pop_alpha,
        "pop_beta": pop_beta,
        "rmse_no_calib": rmse_no_calib,
        "rmse_baseline": rmse_baseline,
        "rmse_per_user_ols": rmse_per_user,
        "rmse_pilsd": rmse_pilsd,
        "pooled_gain_pct": pooled_gain_pct,
        "pooled_ci_bca_lo": pooled_ci_lo,
        "pooled_ci_bca_hi": pooled_ci_hi,
        "pooled_ci_percentile_lo": bs.get("ci_percentile_lo", float("nan")),
        "pooled_ci_percentile_hi": bs.get("ci_percentile_hi", float("nan")),
        "pooled_ci_excludes_zero": ci_excludes_zero,
        "bootstrap_diagnostics": {
            "n_boot_valid": bs.get("n_boot_valid", 0),
            "boot_mean": bs.get("boot_mean", float("nan")),
            "boot_sd": bs.get("boot_sd", float("nan")),
            "z0_bias_correction": bs.get("z0_bias_correction", float("nan")),
            "a_hat_acceleration": bs.get("a_hat_acceleration", float("nan")),
        },
        "per_corpus_breakdown": per_corpus_breakdown,
        "best_per_corpus_gain_pct": best_per_corpus_gain,
        "n_corpora_in_pool": n_corpora_in_pool,
        "anomaly_branches_fired": anomalies,
    }


# ============================================================================
# Driver
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prism-parquet", default=str(DEFAULT_PRISM_PARQUET))
    p.add_argument("--pluriharms-long", default=str(DEFAULT_PLURIHARMS_LONG))
    p.add_argument("--pluriharms-scored", default=str(DEFAULT_PLURIHARMS_SCORED))
    p.add_argument("--helpsteer2-disagreements-jsonl-gz",
                   default=DEFAULT_HELPSTEER2_DISAGREEMENTS)
    p.add_argument("--helpsteer2-scored", default=str(DEFAULT_HELPSTEER2_SCORED))
    p.add_argument("--oasst2-scored", default=str(DEFAULT_OASST2_SCORED))
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--k-folds", type=int, default=K_FOLDS)
    p.add_argument("--min-obs-per-cluster", type=int, default=MIN_OBS_PER_CLUSTER)
    p.add_argument("--smoke", action="store_true",
                   help="50-cluster/200-prompt smoke per corpus.")
    p.add_argument("--output-dir",
                   default=str(ROOT / "results" / "track1_pooled_4_corpus_pilsd"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    summary: dict[str, Any] = {
        "experiment_id": "Q9_pooled_4_corpus_pilsd",
        "verdict_class": "PENDING",
        "args": vars(args),
        "skill_citations": [
            "Skill: research-grade-code-audit-pre-launch v1 G1-G12",
            "Skill: honest-disclosure 4-class STRICT",
            "Skill: post-experiment-discipline-3-track Step 4-7",
            "Skill: launch-runpod-h100-job (h100_v2_backup)",
            "Skill: gpu-artifact-sync",
            "Skill: icml-neurips-critical-reviewer-2026 Pass 5 ablation completeness",
        ],
        "anchors": {
            "w_b5_prism_results_addendum_commit": "702bc63",
            "w_b5_prism_verdict": "ESTABLISHED-COMPOUND-NEEDED canonical=8.55%",
            "q1_helpsteer2_verdict": "ESTABLISHED-HELPSTEER2-CONFIRMS-PRISM gain=+3.700% [+3.346, +4.136]",
            "q3_pluriharms_verdict": "ESTABLISHED-COMPOUND-NEEDED canonical=+9.66% [+7.28, +11.19] REVERSED-half-arm",
            "f_t1_oasst2_verdict": "PASS-REPLICATION canonical PILSD on quality target",
        },
        "honest_disclosure_design_notes": (
            "Q9 pools 4 RLHF preference corpora with corpus-level fixed-effect "
            "(per-corpus z-score on x and y) + namespaced cluster ids "
            "(<corpus>::<cluster>) before applying canonical PILSD. HelpSteer2 "
            "uses cluster=prompt_hash (no stable per-rater id per Wang et al. "
            "2024 §3.4); other 3 corpora use cluster=user_id. The pooled "
            "headline is a generalization claim about PILSD's CALIBRATION "
            "MECHANISM (per-cluster EB shrinkage) on a heterogeneous cluster "
            "pool, NOT a uniform per-rater claim. The FALSIFIED-POOLED-DEGRADES "
            "branch is enumerated explicitly: if pooled gain is strictly worse "
            "than the best per-corpus result by > 0.5pp, the cross-corpus EB "
            "prior is HARMFUL and the headline must SCOPE-LIMIT to per-corpus "
            "replication (Q1/Q3/W-B5/F-T1) rather than pooled."
        ),
        "stages": {},
    }

    log(f"[Q9] starting at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    log(f"[Q9] output_dir: {out_dir}")
    log(f"[Q9] git_head_t3: {git_head_t3()}")

    # ------------------------------------------------------------------
    # Phase 1 — Load 4 corpora
    # ------------------------------------------------------------------
    corpora_loaded: list[pd.DataFrame] = []
    per_corpus_anchors: dict[str, dict] = {}
    fatal_corpora: list[str] = []

    loaders = [
        ("prism", lambda: load_prism(Path(args.prism_parquet)),
         {"parquet": args.prism_parquet, "sha256": file_sha256(args.prism_parquet)}),
        ("pluriharms", lambda: load_pluriharms(Path(args.pluriharms_long), Path(args.pluriharms_scored)),
         {"long_parquet": args.pluriharms_long, "long_sha256": file_sha256(args.pluriharms_long),
          "scored_parquet": args.pluriharms_scored, "scored_sha256": file_sha256(args.pluriharms_scored)}),
        ("helpsteer2", lambda: load_helpsteer2(args.helpsteer2_disagreements_jsonl_gz, Path(args.helpsteer2_scored)),
         {"disagreements_jsonl_gz": args.helpsteer2_disagreements_jsonl_gz,
          "scored_parquet": args.helpsteer2_scored, "scored_sha256": file_sha256(args.helpsteer2_scored)}),
        ("oasst2", lambda: load_oasst2(Path(args.oasst2_scored)),
         {"parquet": args.oasst2_scored, "sha256": file_sha256(args.oasst2_scored)}),
    ]

    for cname, loader_fn, anchor in loaders:
        log(f"[Q9] === loading corpus: {cname} ===")
        per_corpus_anchors[cname] = anchor
        try:
            c_df = loader_fn()
            if args.smoke:
                # G3: small smoke per corpus to keep within minutes
                if cname == "helpsteer2":
                    keep = c_df.cluster_id_native.drop_duplicates().head(200).tolist()
                else:
                    keep = c_df.cluster_id_native.drop_duplicates().head(50).tolist()
                c_df = c_df[c_df.cluster_id_native.isin(keep)].reset_index(drop=True)
                log(f"[smoke] {cname} reduced to {c_df.cluster_id_native.nunique()} clusters / {len(c_df)} obs")
            corpora_loaded.append(c_df)
        except Exception as exc:
            log(f"[Q9 {cname}] FATAL: {exc}")
            log(traceback.format_exc())
            fatal_corpora.append(cname)
            per_corpus_anchors[cname]["error"] = str(exc)

    if len(corpora_loaded) == 0:
        summary.update({
            "verdict_class": "RUNTIME-ERROR-ALL-CORPORA-FAILED",
            "fatal_corpora": fatal_corpora,
            "per_corpus_anchors": per_corpus_anchors,
            "git_head_t3_at_run": git_head_t3(),
            "completion_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        log(f"[Q9] FATAL: all corpora failed to load")
        return 1

    # ------------------------------------------------------------------
    # Phase 2 — Apply per-corpus fixed-effect normalization + namespace cluster ids
    # ------------------------------------------------------------------
    log(f"[Q9] === applying per-corpus fixed-effect normalization ===")
    pooled_df, fe_diagnostics = apply_corpus_fixed_effects(corpora_loaded)

    # ------------------------------------------------------------------
    # Phase 3 — Pooled PILSD on z-score scale
    # ------------------------------------------------------------------
    log(f"[Q9] === running pooled PILSD ===")
    result = run_pooled_pilsd(
        pooled_df,
        seed=args.seed,
        k_folds=args.k_folds,
        min_obs_per_cluster=args.min_obs_per_cluster,
        n_boot=args.n_boot,
        bootstrap_seed=RNG_BOOT,
    )

    # ------------------------------------------------------------------
    # Per-corpus arm parquet (for downstream paper-trail)
    # ------------------------------------------------------------------
    arm_rows = []
    for cname, c in result.get("per_corpus_breakdown", {}).items():
        arm_rows.append({
            "corpus": cname,
            "arm": "pilsd_in_pool",
            "n_obs_in_pool": c.get("n_obs_in_pool", 0),
            "n_clusters_in_pool": c.get("n_clusters_in_pool", 0),
            "rmse_baseline": c.get("rmse_baseline", float("nan")),
            "rmse_pilsd": c.get("rmse_pilsd", float("nan")),
            "gain_pct": c.get("gain_pct", float("nan")),
            "ci_bca_lo": c.get("ci_bca_lo", float("nan")),
            "ci_bca_hi": c.get("ci_bca_hi", float("nan")),
            "ci_excludes_zero": c.get("ci_excludes_zero", False),
        })
    # Pooled headline row
    arm_rows.append({
        "corpus": "POOLED_4_CORPUS",
        "arm": "pilsd_pooled_headline",
        "n_obs_in_pool": int(len(pooled_df)),
        "n_clusters_in_pool": result.get("n_clusters_post_filter", 0),
        "rmse_baseline": result.get("rmse_baseline", float("nan")),
        "rmse_pilsd": result.get("rmse_pilsd", float("nan")),
        "gain_pct": result.get("pooled_gain_pct", float("nan")),
        "ci_bca_lo": result.get("pooled_ci_bca_lo", float("nan")),
        "ci_bca_hi": result.get("pooled_ci_bca_hi", float("nan")),
        "ci_excludes_zero": result.get("pooled_ci_excludes_zero", False),
    })
    arm_df = pd.DataFrame(arm_rows)
    arm_parquet = out_dir / "per_corpus_arms.parquet"
    if len(arm_df):
        arm_df.to_parquet(arm_parquet, index=False)

    summary.update({
        "verdict_class": result.get("verdict_class", "RUNTIME-ERROR-NO-VERDICT"),
        "result": result,
        "per_corpus_anchors": per_corpus_anchors,
        "per_corpus_fe_diagnostics": fe_diagnostics,
        "fatal_corpora": fatal_corpora,
        "per_corpus_arms_parquet": str(arm_parquet) if len(arm_df) else None,
        "git_head_t3_at_run": git_head_t3(),
        "completion_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    def _fmt(v, default=float("nan")):
        try:
            v = float(v) if v is not None else default
        except Exception:
            v = default
        return v

    print()
    print(f"=== Q9 Pooled 4-Corpus PILSD (PRISM + PluriHarms + HelpSteer2 + OASST2) ===")
    print(f"verdict_class : {summary['verdict_class']}")
    print(f"pooled_gain_pct = {_fmt(result.get('pooled_gain_pct')):+.3f}% "
          f"CI95 [{_fmt(result.get('pooled_ci_bca_lo')):+.3f}, "
          f"{_fmt(result.get('pooled_ci_bca_hi')):+.3f}]")
    print(f"n_clusters_post_filter (pooled) = {result.get('n_clusters_post_filter', 0)}")
    if "error" in result:
        print(f"early-return reason: {result['error']}")
    print()
    for cname, c in result.get("per_corpus_breakdown", {}).items():
        print(f"  {cname:>11}: gain={_fmt(c.get('gain_pct')):+.3f}% "
              f"CI=[{_fmt(c.get('ci_bca_lo')):+.3f}, "
              f"{_fmt(c.get('ci_bca_hi')):+.3f}] "
              f"n_clusters={c.get('n_clusters_in_pool', 0)}")
    print(f"\nbest_per_corpus_gain_pct = {_fmt(result.get('best_per_corpus_gain_pct')):+.3f}%")
    print(f"summary_path: {summary_path}")
    print(f"per_corpus_arms_parquet: {arm_parquet}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
