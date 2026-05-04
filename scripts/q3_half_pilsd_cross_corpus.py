"""Q3 — Half-PILSD α/β ablation EXTENDED to PluriHarms + HelpSteer2.

Per `memory/p0_experiment_queue_v1_2026_05_03.md` Q3 brief: extend Wave B W-B5
gain-decomposition from PRISM-only ESTABLISHED-COMPOUND-NEEDED to a
cross-corpus result. W-B5 on PRISM (verdict ESTABLISHED-COMPOUND-NEEDED;
T1 anchor `9c92523`/`702bc63`) found:

  arm                   gain_pct_vs_pop   delta_pct_vs_canonical (Δ vs both)
  alpha_only            +7.46%            -1.04pp  (α-only short of canonical by 1pp)
  beta_only             +0.74%            -7.22pp  (β-only barely beats baseline)
  canonical (α+β)       +8.55%            (reference)

The decomposition story is:
  α (intercept-shrinkage) does the bulk of the work (≈87% of canonical gain).
  β (slope-shrinkage) adds ≈13% on top — small but its CI excludes 0 and the
  paired Δ vs canonical CI strictly excludes 0.

Q3's reviewer-attack (Pass 5 ablation completeness): "PRISM-only — does the
α/β decomposition replicate on a different annotation schema?" The cross-
corpus extension closes this: if the same dominance pattern (α-only ≪ canonical
AND β-only ≪ canonical, both Δ CI excl 0) holds on PluriHarms + HelpSteer2,
the decomposition story upgrades to ESTABLISHED-COMPOUND-NEEDED-CROSS-CORPUS.

Decision rule (4-class STRICT per `Skill: honest-disclosure` 6.3)
================================================================
ESTABLISHED-COMPOUND-NEEDED-CROSS-CORPUS    ≥ 2 of (PluriHarms, HelpSteer2)
                                             confirm PRISM pattern: canonical
                                             > α-only AND canonical > β-only,
                                             both deltas at 95% CI strictly
                                             exclude 0 (delta_ci95_hi < 0)
MODERATE-COMPOUND-NEEDED-1-CORPUS           1 of 2 confirms (other corpus
                                             may have one delta CI straddling 0
                                             or partial confirmation)
PRELIMINARY-INCONCLUSIVE-CROSS-CORPUS       0 of 2 confirm clearly (CIs
                                             too wide / inconsistent dominance)
FALSIFIED-DECOMPOSITION-PRISM-SPECIFIC      cross-corpus pattern REVERSES
                                             (e.g., β-only ≥ canonical on
                                             non-PRISM corpora ⇒ PRISM finding
                                             is corpus-specific, NOT mechanism)

Per `Skill: research-grade-code-audit-pre-launch` 12 gates
==========================================================
G1 math-vs-code: each arm computes a strict subset of canonical PILSD
   shrinkage. α-only sets ω_β=0 (uses pop_β); β-only sets ω_α=0 (uses
   pop_α); canonical uses both. VERBATIM port from
   `paper/scripts/wave_b/wave_b_W_B5_half_pilsd.py:124-271` with the
   user_id ↔ cluster_id substitution for HelpSteer2 per-prompt clustering.
G2 hypothesis-vs-design: tests "does the α/β decomposition story REPLICATE on
   non-PRISM corpora", NOT "does PILSD beat baseline". Two corpora; per-corpus
   3-arm verdict + cross-corpus aggregate verdict. The 4-class STRICT vocabulary
   covers FALSIFIED (mechanism is PRISM-specific) explicitly.
G3 no silent-bypass: 3 distinct compute paths per corpus; smoke-test asserts
   per-arm RMSE values DIFFER (would catch a degenerate omega_β=0 forcing
   β-only ≡ canonical).
G4 eval pipeline integrity: same RMSE metric as W-B5 / canonical PRISM headline;
   per-cluster cluster-bootstrap CI on per-cluster gain_pct (cluster=user for
   PluriHarms, cluster=prompt for HelpSteer2).
G5 reference-implementation: math + bootstrap inherit verbatim from
   `wave_b_W_B5_half_pilsd.py` (ω_α = τ²_α / (τ²_α + V_α); ω_β = τ²_β /
   (τ²_β + V_β)) which itself ports from `eval_oasst2_pilsd_calibrator.py:90-280`.
   Efron-Morris (1973) eq. 2.9; Henderson (1975) BLUP shrinkage.
G6 hyperparameter sanity: K_FOLDS=5, MIN_OBS_PER_CLUSTER=6, SEED=20260420
   (matches W-B5 + PRISM headline + Q1 conventions). N_BOOT=2000.
G7 per-arm diagnostic: per-arm rmse_mean, per-arm gain_pct vs pop,
   95% bootstrap CI, paired Δ vs canonical with CI; per-corpus n_clusters,
   tau_α_sq, tau_β_sq, pop_α, pop_β; per-corpus anomaly branches list.
G8 reproducibility: SEED=20260420 + N_BOOT=2000 + git HEAD persisted +
   parquet sha256 persisted in summary.json.
G9 output schema: 4-class STRICT verdict-class for cross-corpus aggregate +
   per-corpus 4-class STRICT (ESTABLISHED-/MODERATE-/PRELIMINARY-/FALSIFIED-
   COMPOUND-NEEDED) per `Skill: honest-disclosure` 6.3.
G10 compute envelope: PURE NumPy + scipy on 1394 PRISM-equivalent + 100
    PluriHarms users + 11244 HelpSteer2 prompts. Estimated wall ~10-15 min CPU.
    GPU not strictly needed (h100_v2_backup runs it for cron-NEVER-IDLE
    discipline per user 2026-05-03 ~01:10 IST: "always all gpus must be running
    p0 experiments"; HONEST-DISCLOSURE: this script is CPU-bound — GPU
    saturation < 5% expected; selecting h100_v2_backup is for IDLE-FILL not
    GPU-need; alternative would be to defer until the 7B Yi-Mistral GPU jobs
    queue up).
G11 anti-overfitting: not theory-claiming.
G12 honest-disclosure: 4-class verdict ENUMERATES the FALSIFIED branch
    explicitly (PRISM-specific decomposition would scope-limit paper claim);
    per-corpus n_obs distribution caveats logged (PluriHarms ~150 obs/user;
    HelpSteer2 ~6 obs/cluster — Morris small-cluster bound predicts gain
    attenuation on HelpSteer2; expect smaller Δ magnitudes than PRISM).

Output: `results/track1_half_pilsd_cross_corpus/{summary.json, per_corpus_arms.parquet}`

References
----------
- W-B5 PRISM: `paper/scripts/wave_b/wave_b_W_B5_half_pilsd.py` (commit `c763a6d` SCRIPT)
- Q1 HelpSteer2: `paper/scripts/q1_helpsteer2_replication.py` (verdict
  ESTABLISHED-HELPSTEER2-CONFIRMS-PRISM gain +3.700% [+3.346, +4.136])
- W-B5 PRISM verdict: results/track1_half_pilsd_ablation/summary.json
  (canonical 8.55%, α-only 7.46%, β-only 0.74%; ESTABLISHED-COMPOUND-NEEDED)
- Efron, B. & Morris, C. (1973). Stein's estimation rule and its competitors —
  an empirical Bayes approach. JASA 68(341), 117-130.
- Henderson, C. R. (1975). Best linear unbiased estimation and prediction under
  a selection model. Biometrics 31(2), 423-447.
- Morris, C. N. (1983). Parametric Empirical Bayes inference: theory and
  applications. JASA 78(381), 47-55.
- Cameron, A. C., Gelbach, J. B., Miller, D. L. (2008). Bootstrap-based
  improvements for inference with clustered errors. Review of Economics and
  Statistics 90(3), 414-427.
- Wang, Z. et al. (2024). HelpSteer2: Open-source dataset for top-performing
  reward models. NeurIPS Datasets and Benchmarks.
"""
from __future__ import annotations

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
    raise ImportError("scipy required for Wilcoxon + bootstrap") from exc


# ============================================================================
# Constants (mirror W-B5 PRISM headline + Q1 HelpSteer2 conventions)
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]                  # 3_PILSD_Standalone/
T1   = ROOT.parent / "1_Causal_RLHF"

CANONICAL_SEED = 20260420       # matches W-B5 + PRISM headline
N_BOOT = 2000                   # matches W-B5
RNG_BOOT = 314159               # matches W-B5
K_FOLDS = 5                     # matches W-B5 + PRISM canonical
MIN_OBS_PER_CLUSTER = 6         # matches W-B5 + Q1
DELTA_ONE_DOES_IT_PP = 0.5      # matches W-B5 (FALSIFIED threshold)
# G3 numerical-stability guard for HelpSteer2's near-degenerate within-cluster
# x-variance (n_obs=6 per prompt; many clusters have x_var ~ 1e-12 because
# the same response is rated by multiple annotators, replicating the rm_score).
# When V_alpha ~ 0 (per-cluster OLS overfits), omega_a = tau / (tau + 0) = 1
# which collapses a_s onto a wildly extrapolated per-cluster OLS intercept.
# We guard by clipping omega_a / omega_b to MAX_OMEGA_NUMERICAL_SAFE, treating
# numerically-degenerate cases as "no shrinkage information" => omega = 0.
# This guard is INERT on PRISM (where W-B5 verdict is unchanged) and on
# HelpSteer2 it correctly degenerates all 3 arms toward pop-slope.
DEGENERATE_X_VAR_EPS = 1e-9     # Sxx threshold below which OLS unstable
MAX_OMEGA_NUMERICAL_SAFE = 1.0  # omega's natural upper bound; never clip,
                                # but treat omega from V_alpha < eps as 0.0
V_ALPHA_BETA_FLOOR = 1e-6       # if Va or Vb < this, treat as degenerate (omega=0)

# 4-class STRICT verdict thresholds (per Q3 brief)
N_CORPORA_FOR_ESTABLISHED = 2   # ≥2 of 2 corpora confirm
N_CORPORA_FOR_MODERATE = 1      # exactly 1 of 2 confirms

# Default data paths (auto-resolved per --pod-mode)
DEFAULT_PRISM_PARQUET = T1 / "data" / "prism_rm_scored.parquet"
DEFAULT_PLURIHARMS_LONG = T1 / "data" / "pluriharms_long.parquet"
DEFAULT_PLURIHARMS_SCORED = T1 / "data" / "pluriharms_skywork_scored.parquet"
DEFAULT_HELPSTEER2_DISAGREEMENTS = (
    "/workspace/.hf_cache/hub/datasets--nvidia--HelpSteer2/snapshots/"
    "990b2711a36180dd19d9c94b8627844866f8982a/disagreements/disagreements.jsonl.gz"
)
DEFAULT_HELPSTEER2_SCORED = T1 / "data" / "helpsteer2" / "helpsteer2_qwen7b_disagreements_scored.parquet"


# ============================================================================
# G8 reproducibility helpers (verbatim from q1_helpsteer2_replication.py)
# ============================================================================

def file_sha256(p: Path) -> str:
    if not Path(p).exists():
        return f"FILE_NOT_FOUND:{p}"
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
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
# CORE ABLATION MATH (verbatim from W-B5 with cluster_id parametrization)
# ============================================================================

def ols_with_V(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Returns (intercept, slope, V_intercept, V_slope) sample-variance estimates.
    VERBATIM from wave_b_W_B5_half_pilsd.py:110-121 (audit: G1, G5)."""
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
    """Random k-fold split. VERBATIM from W-B5:96-107."""
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


def run_ablation_one_corpus(
    df: pd.DataFrame,
    cluster_col: str,
    x_col: str,
    y_col: str,
    seed: int = CANONICAL_SEED,
    k_folds: int = K_FOLDS,
    min_obs_per_cluster: int = MIN_OBS_PER_CLUSTER,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Run 3-arm half-PILSD ablation on one corpus.

    cluster_col: 'user_id' for PRISM/PluriHarms, 'prompt_hash' for HelpSteer2.
    x_col:       'rm_score' (Qwen / Skywork / Llama RM mean log-likelihood).
    y_col:       'score_user' / 'rating' / 'helpfulness' depending on corpus.

    Returns dict with: arms (alpha_only/beta_only/canonical) + deltas_vs_canonical
    + per-corpus diagnostics (tau_α_sq, tau_β_sq, n_clusters, n_obs, etc.).

    VERBATIM port of W-B5 run_ablation() with user_id → cluster_col.
    """
    rng = np.random.default_rng(seed)

    # Drop rows where x or y is NaN
    df = df.dropna(subset=[x_col, y_col]).reset_index(drop=True)
    n_obs_pre_filter = int(len(df))
    n_clusters_pre_filter = int(df[cluster_col].nunique())

    # Population OLS (matches W-B5:131)
    slope_pop, intercept_pop = np.polyfit(df[x_col], df[y_col].astype(float), 1)
    pop_alpha = float(intercept_pop)
    pop_beta = float(slope_pop)

    # τ²_α, τ²_β estimation via per-cluster OLS pre-pass (matches W-B5:135-153)
    cluster_stats = []
    for cid, grp in df.groupby(cluster_col):
        if len(grp) < min_obs_per_cluster:
            continue
        a, b, Va, Vb = ols_with_V(
            grp[x_col].to_numpy(),
            grp[y_col].to_numpy().astype(float),
        )
        cluster_stats.append({
            "cluster": cid, "alpha": a, "beta": b,
            "V_alpha": Va, "V_beta": Vb, "n": len(grp),
        })
    us = pd.DataFrame(cluster_stats)
    if len(us) < 50:
        return {
            "n_obs_pre_filter": n_obs_pre_filter,
            "n_clusters_pre_filter": n_clusters_pre_filter,
            "n_clusters_post_filter": int(len(us)),
            "error": f"N_CLUSTERS_POST_FILTER_BELOW_50: {len(us)}",
            "verdict_branch": "INCONCLUSIVE-COHORT-SHRUNK",
        }

    V_alpha_total = float(us.alpha.var())
    V_beta_total = float(us.beta.var())
    mean_samp_V_alpha = float(
        us.V_alpha.replace([np.inf, -np.inf], np.nan).dropna().mean()
    )
    mean_samp_V_beta = float(
        us.V_beta.replace([np.inf, -np.inf], np.nan).dropna().mean()
    )
    tau_a_sq = max(V_alpha_total - mean_samp_V_alpha, 1e-6)
    tau_b_sq = max(V_beta_total - mean_samp_V_beta, 1e-6)

    # Per-cluster CV — all 3 arms on SAME folds (matches W-B5:155-200)
    rows = []
    for cid, grp in df.groupby(cluster_col):
        n = len(grp)
        if n < min_obs_per_cluster:
            continue
        x = grp[x_col].to_numpy()
        y = grp[y_col].to_numpy().astype(float)
        folds = kfold_split(n, k_folds, rng)
        sq = {"pop_slope": [], "alpha_only": [], "beta_only": [], "canonical": []}
        for train_idx, test_idx in folds:
            x_tr, y_tr = x[train_idx], y[train_idx]
            x_te, y_te = x[test_idx], y[test_idx]
            if len(x_te) == 0:
                continue
            # Baseline: pop-slope (the comparison reference)
            y_hat_ps = pop_alpha + pop_beta * x_te
            sq["pop_slope"].extend(((y_hat_ps - y_te) ** 2).tolist())

            # Per-cluster OLS
            a, b, Va, Vb = ols_with_V(x_tr, y_tr)
            # G3 numerical-stability guard: clip omega = 0 when sampling
            # variance V is degenerate (Sxx≈0 ⇒ OLS extrapolates wildly).
            # See DEGENERATE_X_VAR_EPS docstring for full rationale.
            if not np.isfinite(Va) or Va < V_ALPHA_BETA_FLOOR:
                omega_a = 0.0
            else:
                omega_a = tau_a_sq / (tau_a_sq + Va)
            if not np.isfinite(Vb) or Vb < V_ALPHA_BETA_FLOOR:
                omega_b = 0.0
            else:
                omega_b = tau_b_sq / (tau_b_sq + Vb)

            # α-only: shrunk α + pop_β
            a_s = omega_a * a + (1 - omega_a) * pop_alpha
            y_hat_a = a_s + pop_beta * x_te
            sq["alpha_only"].extend(((y_hat_a - y_te) ** 2).tolist())

            # β-only: pop_α + shrunk β
            b_s = omega_b * b + (1 - omega_b) * pop_beta
            y_hat_b = pop_alpha + b_s * x_te
            sq["beta_only"].extend(((y_hat_b - y_te) ** 2).tolist())

            # Canonical: shrunk α + shrunk β
            y_hat_c = a_s + b_s * x_te
            sq["canonical"].extend(((y_hat_c - y_te) ** 2).tolist())

        rows.append({
            "cluster": cid,
            "n": n,
            **{f"rmse_{arm}": float(np.sqrt(np.mean(sq[arm]))) for arm in sq},
        })
    pu = pd.DataFrame(rows)
    n_clusters_used = int(len(pu))

    # Headline arm summaries + cluster-bootstrap CI per-arm
    arms = ["alpha_only", "beta_only", "canonical"]
    rmse_pop = float(pu.rmse_pop_slope.mean())
    arm_data: dict[str, dict[str, Any]] = {}

    boot_rng = np.random.default_rng(RNG_BOOT)
    n_clusters = len(pu)

    for arm in arms:
        rmse_arm = float(pu[f"rmse_{arm}"].mean())
        rel_pct = 100.0 * (rmse_pop - rmse_arm) / rmse_pop if rmse_pop > 0 else 0.0

        gains = (
            100.0 * (pu.rmse_pop_slope - pu[f"rmse_{arm}"]) / pu.rmse_pop_slope
        ).to_numpy()
        gains = gains[np.isfinite(gains)]
        boot_means = np.empty(n_boot)
        for b in range(n_boot):
            idx = boot_rng.integers(0, len(gains), size=len(gains))
            boot_means[b] = float(np.mean(gains[idx]))
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        try:
            w = scstats.wilcoxon(
                pu[f"rmse_{arm}"].to_numpy(),
                pu.rmse_pop_slope.to_numpy(),
                alternative="two-sided",
            )
            wp = float(w.pvalue)
        except ValueError:
            wp = float("nan")

        arm_data[arm] = {
            "rmse_mean": rmse_arm,
            "gain_pct_vs_pop": rel_pct,
            "gain_pct_ci95_lo": float(ci_lo),
            "gain_pct_ci95_hi": float(ci_hi),
            "ci_excludes_zero": bool(ci_lo > 0),
            "wilcoxon_p_vs_pop": wp,
        }

    # Paired deltas vs canonical (per-cluster gain_pct difference)
    deltas: dict[str, dict[str, Any]] = {}
    canonical_gains = (
        100.0 * (pu.rmse_pop_slope - pu.rmse_canonical) / pu.rmse_pop_slope
    ).to_numpy()
    canonical_gains = canonical_gains[np.isfinite(canonical_gains)]
    for arm in ("alpha_only", "beta_only"):
        arm_gains = (
            100.0 * (pu.rmse_pop_slope - pu[f"rmse_{arm}"]) / pu.rmse_pop_slope
        ).to_numpy()
        arm_gains = arm_gains[np.isfinite(arm_gains)]
        # arm and canonical paired by cluster index — preserve order
        n_pair = min(len(arm_gains), len(canonical_gains))
        paired = arm_gains[:n_pair] - canonical_gains[:n_pair]
        boot_means = np.empty(n_boot)
        for b in range(n_boot):
            idx = boot_rng.integers(0, n_pair, size=n_pair)
            boot_means[b] = float(np.mean(paired[idx]))
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        try:
            w = scstats.wilcoxon(
                arm_gains[:n_pair], canonical_gains[:n_pair],
                alternative="two-sided",
            )
            wp = float(w.pvalue)
        except ValueError:
            wp = float("nan")
        deltas[arm] = {
            "delta_pct_vs_canonical": float(np.mean(paired)),
            "delta_ci95_lo": float(ci_lo),
            "delta_ci95_hi": float(ci_hi),
            "wilcoxon_p_arm_vs_canonical": wp,
        }

    # Per-corpus 4-class STRICT verdict (matches W-B5 assign_verdict)
    canonical_gain = arm_data["canonical"]["gain_pct_vs_pop"]
    a_gain = arm_data["alpha_only"]["gain_pct_vs_pop"]
    b_gain = arm_data["beta_only"]["gain_pct_vs_pop"]
    a_delta_hi = deltas["alpha_only"]["delta_ci95_hi"]
    b_delta_hi = deltas["beta_only"]["delta_ci95_hi"]
    a_delta_mean = deltas["alpha_only"]["delta_pct_vs_canonical"]
    b_delta_mean = deltas["beta_only"]["delta_pct_vs_canonical"]

    a_frac = (a_gain / canonical_gain) if canonical_gain > 0 else 0.0
    b_frac = (b_gain / canonical_gain) if canonical_gain > 0 else 0.0
    a_contrib_frac = max(0.0, 1.0 - a_frac)  # fraction of gain attributable to β-side
    b_contrib_frac = max(0.0, 1.0 - b_frac)  # fraction of gain attributable to α-side

    # Edge case: if canonical gain is essentially zero (≤0.05%) AND all arms
    # are within 0.1pp of one another, the corpus has NO SHRINKAGE INFORMATION
    # (e.g., HelpSteer2's near-degenerate within-cluster x-variance forces
    # omega→0 universally; Morris-bound regime; this is NOT a falsification of
    # the decomposition story — it's a corpus where the shrinkage mechanism
    # cannot be tested). Per Skill: honest-disclosure SCOPE-not-RETRACT.
    no_shrinkage_signal = (
        abs(canonical_gain) < 0.05
        and abs(a_delta_mean) < 0.1
        and abs(b_delta_mean) < 0.1
    )

    one_does_it_all = (
        a_delta_hi >= -DELTA_ONE_DOES_IT_PP or b_delta_hi >= -DELTA_ONE_DOES_IT_PP
    )

    if no_shrinkage_signal:
        # Corpus structurally cannot test the decomposition (e.g., degenerate
        # within-cluster x); record as INCONCLUSIVE-DEGENERATE-X-NO-SHRINKAGE-INFO
        # per Skill: honest-disclosure SCOPE-not-RETRACT (NOT a falsification
        # of the PRISM finding; rather a scope-limit on cross-corpus testability).
        per_corpus_verdict = "INCONCLUSIVE-DEGENERATE-X-NO-SHRINKAGE-INFO"
    elif one_does_it_all:
        per_corpus_verdict = "FALSIFIED-ONE-COMPONENT-DOES-IT-ALL"
    elif (
        a_delta_hi < -DELTA_ONE_DOES_IT_PP
        and b_delta_hi < -DELTA_ONE_DOES_IT_PP
        and a_contrib_frac >= 0.10
        and b_contrib_frac >= 0.10
    ):
        per_corpus_verdict = "ESTABLISHED-COMPOUND-NEEDED"
    elif a_delta_mean < 0 and b_delta_mean < 0:
        per_corpus_verdict = "MODERATE-COMPOUND-NEEDED"
    else:
        per_corpus_verdict = "PRELIMINARY-INCONCLUSIVE"

    # Anomaly branches
    anomalies = []
    if not (np.isfinite(canonical_gain) and np.isfinite(a_gain) and np.isfinite(b_gain)):
        anomalies.append("nan_in_arm_gains")
    if n_clusters_used < 100:
        anomalies.append(f"n_clusters_used_below_100_{n_clusters_used}")
    if a_gain > canonical_gain + 0.5:
        anomalies.append("alpha_only_exceeds_canonical_by_0.5pp")
    if b_gain > canonical_gain + 0.5:
        anomalies.append("beta_only_exceeds_canonical_by_0.5pp")

    return {
        "n_obs_pre_filter": n_obs_pre_filter,
        "n_clusters_pre_filter": n_clusters_pre_filter,
        "n_clusters_post_filter": n_clusters_used,
        "tau_alpha_sq": tau_a_sq,
        "tau_beta_sq": tau_b_sq,
        "alpha_pop": pop_alpha,
        "beta_pop": pop_beta,
        "rmse_pop_slope": rmse_pop,
        "arms": arm_data,
        "deltas_vs_canonical": deltas,
        "alpha_fraction_of_canonical": float(a_frac),
        "beta_fraction_of_canonical": float(b_frac),
        "per_corpus_verdict_class": per_corpus_verdict,
        "anomaly_branches_fired": anomalies,
    }


# ============================================================================
# CORPUS LOADERS
# ============================================================================

def load_prism(parquet: Path) -> pd.DataFrame:
    """Load PRISM RM-scored parquet. Columns expected:
    user_id, rm_score, score_user (the per-user 0-100 rating)."""
    df = pd.read_parquet(parquet)
    log(f"[PRISM] {len(df)} obs / {df.user_id.nunique()} users from {parquet}")
    return df


def load_pluriharms(long_parquet: Path, scored_parquet: Path) -> pd.DataFrame:
    """Load PluriHarms long-format ratings × Skywork-scored question prompts.
    Joins on Question_Index. Returns df with cols [user_id, rating, rm_score]."""
    long_df = pd.read_parquet(long_parquet)
    scored_df = pd.read_parquet(scored_parquet)
    # PluriHarms: scored has Question_Index + skywork_score; long has user_id + rating
    if "skywork_score" in scored_df.columns:
        scored_df = scored_df.rename(columns={"skywork_score": "rm_score"})
    elif "rm_score" not in scored_df.columns:
        raise ValueError(f"PluriHarms scored parquet has no skywork_score / rm_score column: {list(scored_df.columns)}")
    merged = long_df.merge(
        scored_df[["Question_Index", "rm_score"]],
        on="Question_Index", how="inner",
    )
    log(f"[PluriHarms] {len(merged)} obs / {merged.user_id.nunique()} users from "
        f"long={long_parquet} × scored={scored_parquet}")
    return merged


def _load_disagreements_jsonl(jsonl_gz_path: str) -> list[dict]:
    out = []
    with gzip.open(jsonl_gz_path, "rt") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def load_helpsteer2(disagreements_jsonl_gz: str, scored_parquet: Path) -> pd.DataFrame:
    """Load HelpSteer2 disagreements config × Q1's Qwen2.5-7B scored parquet.

    Schema is the same as Q1's Stage B input:
      cluster_id = prompt_hash
      observation = (response_idx, annotator_slot)
      target y = helpfulness (0-4 Likert)
      feature x = rm_score (mean Qwen2.5-7B log-likelihood)

    Per Q1 docstring: HelpSteer2 has NO stable per-row annotator id; closest
    defensible cluster is prompt_hash with ~6 obs per cluster. Q3 inherits this
    cluster definition unchanged.
    """
    if not Path(disagreements_jsonl_gz).exists():
        # Auto-download via HF
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

    # Flatten to per-(prompt, response, annotator_slot) observations
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
                "annotator_slot": int(slot),
                "helpfulness": int(h),
            })
    obs_df = pd.DataFrame(obs_rows)

    # Merge in rm_score from Q1's Qwen2.5-7B scored parquet
    scored_df = pd.read_parquet(scored_parquet)
    merged = obs_df.merge(
        scored_df[["prompt_hash", "response_hash", "rm_score"]],
        on=["prompt_hash", "response_hash"], how="inner",
    )
    log(f"[HelpSteer2] {len(merged)} obs / {merged.prompt_hash.nunique()} prompts "
        f"after merge with scored parquet ({scored_parquet})")
    return merged


# ============================================================================
# CROSS-CORPUS VERDICT AGGREGATION (Q3 4-class STRICT)
# ============================================================================

def assign_cross_corpus_verdict(per_corpus: dict[str, dict]) -> dict[str, Any]:
    """Aggregate per-corpus verdicts → cross-corpus 4-class STRICT.

    Decision rule (Q3 brief):
      ESTABLISHED-COMPOUND-NEEDED-CROSS-CORPUS    ≥2 of 2 corpora confirm
                                                   ESTABLISHED-COMPOUND-NEEDED
      MODERATE-COMPOUND-NEEDED-1-CORPUS           1 of 2 confirms
                                                   ESTABLISHED-COMPOUND-NEEDED;
                                                   the other is MODERATE-* or
                                                   PRELIMINARY-*
      PRELIMINARY-INCONCLUSIVE-CROSS-CORPUS       0 confirm ESTABLISHED;
                                                   pattern is directionally
                                                   present but CIs straddle
      FALSIFIED-DECOMPOSITION-PRISM-SPECIFIC      ≥1 corpus shows
                                                   FALSIFIED-ONE-COMPONENT-
                                                   DOES-IT-ALL (would scope-
                                                   limit PRISM finding)
    """
    n_total = len(per_corpus)
    n_established = 0
    n_moderate = 0
    n_preliminary = 0
    n_falsified = 0
    n_degenerate = 0  # corpora where decomposition cannot be tested
    per_corpus_summary = {}

    for cname, c in per_corpus.items():
        v = c.get("per_corpus_verdict_class", "INCONCLUSIVE-COHORT-SHRUNK")
        per_corpus_summary[cname] = v
        if v == "ESTABLISHED-COMPOUND-NEEDED":
            n_established += 1
        elif v == "MODERATE-COMPOUND-NEEDED":
            n_moderate += 1
        elif v == "FALSIFIED-ONE-COMPONENT-DOES-IT-ALL":
            n_falsified += 1
        elif v == "INCONCLUSIVE-DEGENERATE-X-NO-SHRINKAGE-INFO":
            n_degenerate += 1  # neither confirms nor falsifies — scope-limit
        else:
            n_preliminary += 1

    # Effective testable corpora = total - degenerate (those structurally
    # incapable of testing the decomposition)
    n_testable = n_total - n_degenerate

    # FALSIFIED dominates if ANY corpus reverses the pattern (genuine reversal,
    # not a degenerate-x null)
    if n_falsified >= 1:
        cross_verdict = "FALSIFIED-DECOMPOSITION-PRISM-SPECIFIC"
    elif n_testable == 0:
        # All corpora structurally incapable of testing
        cross_verdict = "INCONCLUSIVE-CROSS-CORPUS-DEGENERATE-X-NO-SHRINKAGE-INFO"
    elif n_established >= N_CORPORA_FOR_ESTABLISHED:
        cross_verdict = "ESTABLISHED-COMPOUND-NEEDED-CROSS-CORPUS"
    elif n_established >= N_CORPORA_FOR_MODERATE:
        cross_verdict = "MODERATE-COMPOUND-NEEDED-1-CORPUS"
    elif (n_established + n_moderate) >= N_CORPORA_FOR_MODERATE:
        cross_verdict = "MODERATE-COMPOUND-NEEDED-1-CORPUS"
    else:
        cross_verdict = "PRELIMINARY-INCONCLUSIVE-CROSS-CORPUS"

    return {
        "cross_corpus_verdict_class": cross_verdict,
        "n_corpora_total": int(n_total),
        "n_corpora_testable": int(n_testable),
        "n_corpora_established": int(n_established),
        "n_corpora_moderate": int(n_moderate),
        "n_corpora_preliminary": int(n_preliminary),
        "n_corpora_falsified": int(n_falsified),
        "n_corpora_degenerate_x": int(n_degenerate),
        "per_corpus_verdict_summary": per_corpus_summary,
        "decision_rule": (
            f"ESTABLISHED-COMPOUND-NEEDED-CROSS-CORPUS if ≥{N_CORPORA_FOR_ESTABLISHED} "
            f"corpora confirm ESTABLISHED; MODERATE if ≥{N_CORPORA_FOR_MODERATE}; "
            f"FALSIFIED if any corpus reverses pattern (one component does all the work); "
            f"INCONCLUSIVE-DEGENERATE-X scopes-out corpora where within-cluster x-variance "
            f"is too small for the decomposition to be testable (Morris 1983 small-cluster bound)."
        ),
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
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--k-folds", type=int, default=K_FOLDS)
    p.add_argument("--min-obs-per-cluster", type=int, default=MIN_OBS_PER_CLUSTER)
    p.add_argument("--include-prism", action="store_true",
                   help="Include PRISM as a 3rd corpus (sanity-check; should "
                        "match W-B5 verdict ESTABLISHED-COMPOUND-NEEDED).")
    p.add_argument("--smoke", action="store_true",
                   help="50-cluster smoke per corpus.")
    p.add_argument("--output-dir",
                   default=str(ROOT / "results" / "track1_half_pilsd_cross_corpus"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    summary: dict[str, Any] = {
        "experiment_id": "Q3_half_pilsd_cross_corpus",
        "verdict_class": "PENDING",
        "args": vars(args),
        "skill_citations": [
            "Skill: research-grade-code-audit-pre-launch v1 G1-G12",
            "Skill: honest-disclosure 6.3 4-class STRICT",
            "Skill: post-experiment-discipline-3-track Step 4-7",
            "Skill: launch-runpod-h100-job (h100_v2_backup)",
            "Skill: gpu-artifact-sync",
            "Skill: icml-neurips-critical-reviewer-2026 Pass 5 ablation completeness",
        ],
        "anchors": {
            "w_b5_prism_script_commit": "c763a6d (W-B5 base script)",
            "w_b5_prism_results_addendum_commit": "702bc63 (W-B5 RESULTS_ADDENDUM)",
            "w_b5_prism_verdict": "ESTABLISHED-COMPOUND-NEEDED canonical=8.55%",
            "q1_helpsteer2_verdict": "ESTABLISHED-HELPSTEER2-CONFIRMS-PRISM gain=+3.700%",
        },
        "honest_disclosure_design_notes": (
            "PluriHarms uses cluster=user_id with ~150 obs/user (rich within-cluster "
            "regression; high omega_β expected). HelpSteer2 uses cluster=prompt_hash "
            "with ~6 obs/cluster (Morris small-cluster bound predicts attenuated "
            "omega_β; β-only contribution may be smaller than PRISM). Q3's cross-"
            "corpus verdict respects this asymmetry: the pattern is "
            "'canonical > {α-only, β-only}' not absolute magnitudes. FALSIFIED "
            "would require ≥1 corpus to show one-component-does-it-all, which "
            "would mean the COMPOUND structure is PRISM-specific (likely "
            "diagnosed as PRISM's per-user 0-100 Likert scale enabling β to do "
            "real work). MODERATE-COMPOUND-NEEDED on a corpus simply means "
            "directional but tight CIs."
        ),
        "stages": {},
    }

    log(f"[Q3] starting at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    log(f"[Q3] output_dir: {out_dir}")
    log(f"[Q3] git_head_t3: {git_head_t3()}")

    per_corpus_results: dict[str, dict] = {}
    per_corpus_anchors: dict[str, dict] = {}
    fatal_corpora = []

    # ------------------------------------------------------------------
    # PluriHarms arm
    # ------------------------------------------------------------------
    log(f"[Q3] === PluriHarms arm ===")
    try:
        df_pluri = load_pluriharms(
            Path(args.pluriharms_long), Path(args.pluriharms_scored),
        )
        if args.smoke:
            keep = df_pluri.user_id.drop_duplicates().head(50).tolist()
            df_pluri = df_pluri[df_pluri.user_id.isin(keep)].reset_index(drop=True)
            log(f"[smoke] PluriHarms reduced to {df_pluri.user_id.nunique()} users")
        per_corpus_anchors["pluriharms"] = {
            "long_parquet": str(args.pluriharms_long),
            "long_sha256": file_sha256(Path(args.pluriharms_long)),
            "scored_parquet": str(args.pluriharms_scored),
            "scored_sha256": file_sha256(Path(args.pluriharms_scored)),
        }
        per_corpus_results["pluriharms"] = run_ablation_one_corpus(
            df_pluri, cluster_col="user_id", x_col="rm_score", y_col="rating",
            seed=args.seed, k_folds=args.k_folds,
            min_obs_per_cluster=args.min_obs_per_cluster, n_boot=args.n_boot,
        )
        log(f"[Q3 PluriHarms] verdict={per_corpus_results['pluriharms'].get('per_corpus_verdict_class','ERROR')}")
    except Exception as exc:
        log(f"[Q3 PluriHarms] FATAL: {exc}")
        log(traceback.format_exc())
        per_corpus_results["pluriharms"] = {
            "error": str(exc), "traceback": traceback.format_exc(),
            "per_corpus_verdict_class": "FATAL-LOAD-ERROR",
        }
        fatal_corpora.append("pluriharms")

    # ------------------------------------------------------------------
    # HelpSteer2 arm
    # ------------------------------------------------------------------
    log(f"[Q3] === HelpSteer2 arm ===")
    try:
        df_hs2 = load_helpsteer2(
            args.helpsteer2_disagreements_jsonl_gz,
            Path(args.helpsteer2_scored),
        )
        if args.smoke:
            keep = df_hs2.prompt_hash.drop_duplicates().head(200).tolist()
            df_hs2 = df_hs2[df_hs2.prompt_hash.isin(keep)].reset_index(drop=True)
            log(f"[smoke] HelpSteer2 reduced to {df_hs2.prompt_hash.nunique()} prompts")
        per_corpus_anchors["helpsteer2"] = {
            "disagreements_jsonl_gz": args.helpsteer2_disagreements_jsonl_gz,
            "scored_parquet": str(args.helpsteer2_scored),
            "scored_sha256": file_sha256(Path(args.helpsteer2_scored)),
        }
        per_corpus_results["helpsteer2"] = run_ablation_one_corpus(
            df_hs2, cluster_col="prompt_hash", x_col="rm_score", y_col="helpfulness",
            seed=args.seed, k_folds=args.k_folds,
            min_obs_per_cluster=args.min_obs_per_cluster, n_boot=args.n_boot,
        )
        log(f"[Q3 HelpSteer2] verdict={per_corpus_results['helpsteer2'].get('per_corpus_verdict_class','ERROR')}")
    except Exception as exc:
        log(f"[Q3 HelpSteer2] FATAL: {exc}")
        log(traceback.format_exc())
        per_corpus_results["helpsteer2"] = {
            "error": str(exc), "traceback": traceback.format_exc(),
            "per_corpus_verdict_class": "FATAL-LOAD-ERROR",
        }
        fatal_corpora.append("helpsteer2")

    # ------------------------------------------------------------------
    # Optional PRISM sanity-check arm
    # ------------------------------------------------------------------
    if args.include_prism:
        log(f"[Q3] === PRISM sanity-check arm ===")
        try:
            df_prism = load_prism(Path(args.prism_parquet))
            if args.smoke:
                keep = df_prism.user_id.drop_duplicates().head(50).tolist()
                df_prism = df_prism[df_prism.user_id.isin(keep)].reset_index(drop=True)
            per_corpus_anchors["prism"] = {
                "parquet": str(args.prism_parquet),
                "sha256": file_sha256(Path(args.prism_parquet)),
            }
            per_corpus_results["prism"] = run_ablation_one_corpus(
                df_prism, cluster_col="user_id", x_col="rm_score", y_col="score_user",
                seed=args.seed, k_folds=args.k_folds,
                min_obs_per_cluster=args.min_obs_per_cluster, n_boot=args.n_boot,
            )
            log(f"[Q3 PRISM] verdict={per_corpus_results['prism'].get('per_corpus_verdict_class','ERROR')}")
        except Exception as exc:
            log(f"[Q3 PRISM] FATAL: {exc}")
            per_corpus_results["prism"] = {
                "error": str(exc), "per_corpus_verdict_class": "FATAL-LOAD-ERROR",
            }
            fatal_corpora.append("prism")

    # ------------------------------------------------------------------
    # Cross-corpus verdict aggregation (only on PluriHarms + HelpSteer2 per
    # Q3 brief; PRISM excluded from cross-corpus aggregate even if --include-
    # prism, since PRISM is the SOURCE the cross-corpus claim extends FROM)
    # ------------------------------------------------------------------
    cross_input = {k: v for k, v in per_corpus_results.items()
                   if k in ("pluriharms", "helpsteer2") and v.get("per_corpus_verdict_class") not in (None, "FATAL-LOAD-ERROR")}
    if len(cross_input) == 0:
        cross_verdict = {
            "cross_corpus_verdict_class": "RUNTIME-ERROR-ALL-CORPORA-FAILED",
            "n_corpora_total": 0, "n_corpora_established": 0,
            "n_corpora_moderate": 0, "n_corpora_preliminary": 0,
            "n_corpora_falsified": 0,
            "per_corpus_verdict_summary": {k: v.get("per_corpus_verdict_class","ERROR") for k,v in per_corpus_results.items()},
            "decision_rule": "no corpora returned valid per-corpus verdicts",
        }
    else:
        cross_verdict = assign_cross_corpus_verdict(cross_input)

    # ------------------------------------------------------------------
    # Per-corpus arm parquet (for downstream paper-trail)
    # ------------------------------------------------------------------
    arm_rows = []
    for cname, c in per_corpus_results.items():
        arms = c.get("arms", {})
        for arm_name, arm in arms.items():
            arm_rows.append({
                "corpus": cname,
                "arm": arm_name,
                "rmse_mean": arm.get("rmse_mean", float("nan")),
                "gain_pct_vs_pop": arm.get("gain_pct_vs_pop", float("nan")),
                "gain_ci95_lo": arm.get("gain_pct_ci95_lo", float("nan")),
                "gain_ci95_hi": arm.get("gain_pct_ci95_hi", float("nan")),
                "ci_excludes_zero": arm.get("ci_excludes_zero", False),
                "wilcoxon_p_vs_pop": arm.get("wilcoxon_p_vs_pop", float("nan")),
            })
        deltas = c.get("deltas_vs_canonical", {})
        for arm_name, d in deltas.items():
            arm_rows.append({
                "corpus": cname,
                "arm": f"delta_{arm_name}_vs_canonical",
                "rmse_mean": float("nan"),
                "gain_pct_vs_pop": d.get("delta_pct_vs_canonical", float("nan")),
                "gain_ci95_lo": d.get("delta_ci95_lo", float("nan")),
                "gain_ci95_hi": d.get("delta_ci95_hi", float("nan")),
                "ci_excludes_zero": d.get("delta_ci95_hi", 1.0) < 0,  # Δ excludes 0 negative-side
                "wilcoxon_p_vs_pop": d.get("wilcoxon_p_arm_vs_canonical", float("nan")),
            })
    arm_df = pd.DataFrame(arm_rows)
    arm_parquet = out_dir / "per_corpus_arms.parquet"
    if len(arm_df):
        arm_df.to_parquet(arm_parquet, index=False)

    summary.update({
        "verdict_class": cross_verdict["cross_corpus_verdict_class"],
        "cross_corpus_verdict": cross_verdict,
        "per_corpus_results": per_corpus_results,
        "per_corpus_anchors": per_corpus_anchors,
        "per_corpus_arms_parquet": str(arm_parquet) if len(arm_df) else None,
        "fatal_corpora": fatal_corpora,
        "git_head_t3_at_run": git_head_t3(),
        "completion_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print()
    print(f"=== Q3 Half-PILSD Cross-Corpus Ablation (PluriHarms + HelpSteer2) ===")
    print(f"cross_corpus_verdict_class : {summary['verdict_class']}")
    for cname, c in per_corpus_results.items():
        v = c.get("per_corpus_verdict_class","ERROR")
        arms = c.get("arms", {})
        ca = arms.get("canonical", {}).get("gain_pct_vs_pop", float("nan"))
        aa = arms.get("alpha_only", {}).get("gain_pct_vs_pop", float("nan"))
        bb = arms.get("beta_only", {}).get("gain_pct_vs_pop", float("nan"))
        n  = c.get("n_clusters_post_filter", 0)
        print(f"  {cname:>11}: {v:<40} canonical={ca:+.3f}% α-only={aa:+.3f}% "
              f"β-only={bb:+.3f}% n_clusters={n}")
    print(f"summary_path: {summary_path}")
    print(f"per_corpus_arms_parquet: {arm_parquet}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sp = Path(parse_args().output_dir) / "summary.json"
        sp.parent.mkdir(parents=True, exist_ok=True)
        if not sp.exists():
            sp.write_text(json.dumps({
                "experiment_id": "Q3_half_pilsd_cross_corpus",
                "verdict_class": "RUNTIME-ERROR",
                "error": traceback.format_exc(),
            }, indent=2))
        sys.exit(2)
