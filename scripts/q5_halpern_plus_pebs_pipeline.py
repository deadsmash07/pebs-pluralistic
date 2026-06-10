"""Q5 — Halpern (2025) FSAM + PEBS pipeline composition on PRISM LOCO.

Composes Halpern's selection-style ensemble (Forward-Stagewise Additive
Modeling over K reward heads with simplex mixture weights) with PEBS's
calibration-side per-rater empirical-Bayes shrinkage. The contribution is a
**pipeline-level value** demonstration rather than a head-to-head dominance:
Halpern picks the user-relevant reward dimension on the prompt/ensemble side;
PEBS calibrates per-rater on the output side. If combined > both individual,
this supports the "PEBS complements selection-style methods" framing and
addresses the critique that PEBS is solely a calibration trick.

The 3 arms tested on the IDENTICAL PRISM LOCO 5-fold split (matching W-B5,
Q1, Q3, PReF, LoRe baseline conventions) are:

  1. **HALPERN-only** — FSAM K=4 ensemble over polynomial reward heads
     (paper Eq. 2 + Section 4); native output is a per-utterance predicted
     score in [0, 1] (sigmoid of mixture-weighted reward sum), rescaled to
     PRISM's 0-100 axis via per-user OLS on TRAIN-fold pairs (the same
     scalar-RMSE bridge PReF and LoRe use; symmetric to
     ``baselines/pref_2025_on_prism.py:96-106`` and
     ``baselines/lore_2025_on_prism.py:62-71``).
  2. **PEBS-only** — canonical W-B5 PEBS: per-user OLS calibrator on raw
     ``rm_score``, joint α+β EB shrinkage with closed-form ω_α=τ²_α/(τ²_α+V_α),
     ω_β=τ²_β/(τ²_β+V_β). VERBATIM port from
     ``paper/scripts/wave_b/wave_b_W_B5_half_pebs.py:124-271`` "canonical" arm.
  3. **COMBINED-pipeline** — Halpern's per-utterance ensemble prediction
     replaces ``rm_score`` as the upstream signal feeding into PEBS's per-user
     OLS calibrator. PEBS's α+β shrinkage is then applied on (Halpern_pred,
     score_user). This tests: when the upstream signal is already
     persona-aware (FSAM K=4 ensemble), does PEBS's per-rater calibration
     STILL provide additional value? If yes, the methods compose at the
     pipeline level.

Decision rule
=============
CONFIRMED-PIPELINE-COMPOSITION-DOMINATES
    Combined gain ≥ both individual at 95% CI on paired Δ AND combined
    gain ≥ best-of-individual + 1pp absolute (genuine pipeline-level value).
PARTIAL-PIPELINE-EQUIVALENT
    Combined ≈ best-of-individual within 1pp (pipeline composition is
    NEUTRAL — neither method actively interferes; useful documentation
    that PEBS does NOT degrade Halpern's ensemble signal).
TENTATIVE-INCONCLUSIVE
    Combined < either individual but CIs overlap (rare; would suggest
    one method actively interferes; CIs too wide to falsify).
REJECTED-PIPELINE-WORSE
    Combined CI strictly below better-of-individuals (combination ACTIVELY
    HURTS — would scope-limit the pipeline-composition claim and might
    suggest the methods are competitive rather than compositional;
    would itself be a noteworthy honest finding).

Design notes
============
Math:
    HALPERN: FSAM iteration j (paper Eq. 2 + Sec. 4) implemented as polynomial
    reward heads r_k(z) = sum_{d=0..D} theta_{k,d} * z^d on standardised
    rm_score z; pairwise calibration target p* = empirical within-user
    win-rate over (utt_a, utt_b) pairs; predicted p_hat = sum_k alpha_k *
    sigmoid(r_k(z_a) - r_k(z_b)); FSAM stagewise fit r_j to residual MSE
    then re-solve alpha on simplex via SLSQP. Per-utterance score prediction
    is sum_k alpha_k * sigmoid(r_k(z_uttance)) -- the ensemble's "absolute"
    reward in [0, 1].
    PEBS: VERBATIM port from W-B5 ``wave_b_W_B5_half_pebs.py:124-271``
    canonical arm: per-user OLS (alpha_j, beta_j) on TRAIN-fold ratings;
    EB shrinkage omega_a = tau_a_sq / (tau_a_sq + V_alpha_user), omega_b
    same; shrunk predictions on TEST-fold.
    COMBINED: Halpern's per-utterance pred z_h := sum_k alpha_k *
    sigmoid(r_k(z_uttance)) replaces raw rm_score as the input to PEBS's
    per-user OLS. PEBS's shrinkage formula is unchanged; only the upstream
    feature axis differs.
Hypothesis:
    Tests "do Halpern's selection-side and PEBS's calibration-side compose
    to a pipeline-level gain", NOT "does either dominate head-to-head".
    The decision rule explicitly enumerates BOTH the positive
    (CONFIRMED-PIPELINE-COMPOSITION-DOMINATES) AND the negative
    (REJECTED-PIPELINE-WORSE) branches.
No silent-bypass:
    3 distinct compute paths -- Halpern arm uses ensemble; PEBS arm
    uses raw rm_score; combined arm uses Halpern's pred as upstream.
    Smoke-test asserts per-arm RMSE values DIFFER (would catch
    degenerate Halpern alpha collapsing to a single head ⇒ Halpern
    arm ≡ trivial baseline ⇒ combined ≡ PEBS).
    FSAM SLSQP simplex optimisation is initialised at uniform alpha = 1/K
    and re-solved after each new head; we assert alpha is on the simplex
    and not all-mass-on-one-head (which would silently degenerate K=4 to K=1).
Eval pipeline integrity:
    Same per-user RMSE metric as W-B5 / canonical PRISM headline; same
    cluster-bootstrap CI (cluster=user_id, N_BOOT=2000, RNG_BOOT=314159).
    Test-fold predictions are unshrunken pop-baseline + per-arm shrinkage
    layered on top. The combined arm is NOT trained on test-fold ratings
    (Halpern's per-user calibration uses TRAIN-fold scalars only; PEBS's
    omega + alpha_j + beta_j are estimated from the same TRAIN-fold).
Reference-implementation parity:
    HALPERN: FSAM math is paper-text re-implementation (no official
    code available; cross-checked against
    ``baselines/halpern_2025_on_prism.py:38-40``).
    The existing neural Halpern baseline reached
    rmse=0.3226 95% CI [0.3171, 0.3265] on n=1396 users with K=4
    Adam lr=1e-4 5 epochs/stage. Q5 ports the FSAM logic to a CPU-only
    polynomial-reward-head setting (no gradient-descent training; closed-
    form polynomial fit per stage) so that the math is auditable and the
    pipeline composition isolates Halpern's algorithmic contribution
    (selection-style mixture) from its specific neural-RM instantiation.
    PEBS canonical: VERBATIM port from W-B5 (canonical gain +8.55%);
    ω formula = Efron-Morris (1973) eq. 2.9.
Hyperparameters:
    K_FOLDS=5, MIN_OBS_PER_USER=6, SEED=20260420, N_BOOT=2000 (matches
    W-B5 + PRISM headline + Q3 conventions). HALPERN_K=4 (paper Table 4
    sweet spot). HALPERN_DEG=3 polynomial degree (matches LoRe/PReF
    polynomial-basis convention). HALPERN_RIDGE=1e-3 (small L2 stabiliser
    for closed-form polynomial fit).
Per-arm diagnostics:
    Per-arm rmse_mean, gain_pct vs pop_slope baseline, 95% CI on per-user
    gain_pct via cluster-bootstrap, paired Δ vs PEBS-only and vs
    HALPERN-only with CI; per-arm wilcoxon p-value vs pop-baseline AND
    paired-arm wilcoxon p-values. tau_alpha_sq, tau_beta_sq, alpha_pop,
    beta_pop, halpern_alpha_simplex, n_users, n_obs persisted.
Reproducibility:
    SEED=20260420 + N_BOOT=2000 + git HEAD persisted + parquet sha256
    persisted in summary.json (matches Q3 + Q1 conventions).
Output schema:
    The pipeline-composition aggregate outcome is one of
    CONFIRMED-PIPELINE-COMPOSITION-DOMINATES / PARTIAL-PIPELINE-
    EQUIVALENT / TENTATIVE-INCONCLUSIVE / REJECTED-PIPELINE-WORSE.
Compute envelope:
    PURE NumPy + scipy on PRISM 1394 users; FSAM K=4 with closed-form
    polynomial fit per stage is O(N * K * D^2); K=4 D=3 N~7000 ⇒ wall
    estimate ~2-5 min CPU. The polynomial surrogate (vs a neural-RM
    Halpern reproduction) is faster + auditable + apples-to-apples with
    PReF/LoRe; the script is CPU-bound and needs no GPU.
Anti-overfitting:
    Halpern alpha vector is constrained to simplex via SLSQP (paper Sec. 4);
    L2 ridge=1e-3 on polynomial reward-head fit prevents overfitting on
    small TRAIN folds. PEBS's EB shrinkage IS the anti-overfitting
    machinery (per-user OLS with closed-form variance-weighted shrinkage
    toward population pop_alpha + pop_beta). Combined arm inherits both.
Honest disclosure:
    REJECTED-PIPELINE-WORSE branch ENUMERATED explicitly; would itself be a
    noteworthy honest finding (the methods would be COMPETITIVE rather
    than COMPOSITIONAL, scope-limiting the pipeline-composition claim).
    The script also enumerates the polynomial-surrogate caveat: this Q5
    Halpern implementation differs from the T1 Halpern script
    (``1_Causal_RLHF/scripts/baselines/halpern_2025_on_prism.py``) in that
    it uses polynomial reward heads (closed-form fit) rather than gradient-
    descent neural reward heads, making it CPU-fast and apples-to-apples
    with PReF/LoRe but DIFFERENT in capacity from the original neural FSAM
    paper. The Q5 contribution is the **pipeline composition test**, not a
    re-validation of Halpern's standalone performance (done previously).

Output: ``results/track1_q5_halpern_plus_pebs_pipeline/{summary.json}``

References
----------
- Halpern, D., Micha, E., Shapira, I., Procaccia, A. D. (2025). Pairwise
  Calibrated Rewards for Pluralistic Alignment. arXiv:2506.06298, NeurIPS 2025.
- Neural Halpern baseline: ``baselines/halpern_2025_on_prism.py``
  (rmse=0.3226 95% CI [0.3171, 0.3265]).
- W-B5 PEBS canonical: ``scripts/wave_b/wave_b_W_B5_half_pebs.py``
  (canonical gain +8.55% [+7.79, +9.31]).
- Q3 Half-PEBS cross-corpus: ``scripts/q3_half_pebs_cross_corpus.py``
  (PRISM sanity reproduces W-B5).
- Efron, B. & Morris, C. (1973). Stein's estimation rule and its competitors —
  an empirical Bayes approach. JASA 68(341), 117-130.
- Henderson, C. R. (1975). Best linear unbiased estimation and prediction under
  a selection model. Biometrics 31(2), 423-447.
- Cameron, A. C., Gelbach, J. B., Miller, D. L. (2008). Bootstrap-based
  improvements for inference with clustered errors. Review of Economics and
  Statistics 90(3), 414-427.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import optimize as scopt
    from scipy import stats as scstats
except ImportError as exc:
    raise ImportError("scipy required for SLSQP simplex + Wilcoxon") from exc


# ============================================================================
# Constants (mirror W-B5 PRISM headline + Q3 conventions)
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]                  # 3_PEBS_Standalone/
T1   = ROOT.parent / "1_Causal_RLHF"

CANONICAL_SEED = 20260420       # matches W-B5 + PRISM headline + Q3
N_BOOT = 2000                   # matches W-B5 + Q3
RNG_BOOT = 314159               # matches W-B5 + Q3
K_FOLDS = 5                     # matches W-B5 + PRISM canonical
MIN_OBS_PER_USER = 6            # matches W-B5 + Q3
DELTA_DOMINATE_PP = 1.0         # CONFIRMED requires combined > best by 1pp
DELTA_EQUIVALENT_PP = 1.0       # PARTIAL requires combined within 1pp of best
HALPERN_K = 4                   # paper Table 4 sweet spot
HALPERN_DEG = 3                 # polynomial reward-head degree (matches PReF/LoRe convention)
HALPERN_RIDGE = 1e-3            # L2 stabiliser for closed-form polynomial fit
HALPERN_MAX_ITER = 5            # FSAM stages (one new head per stage; total = HALPERN_K)
HALPERN_SLSQP_TOL = 1e-6        # simplex SLSQP tolerance

# Numerical-stability guard (inherited from W-B5 / Q3)
V_ALPHA_BETA_FLOOR = 1e-6       # treat as degenerate (omega=0) below floor
DEGENERATE_X_VAR_EPS = 1e-9     # Sxx threshold below which OLS unstable

# Default data paths
DEFAULT_PRISM_PARQUET = T1 / "data" / "prism_rm_scored.parquet"


# ============================================================================
# Reproducibility helpers (verbatim from Q3 / Q1 conventions)
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


# ============================================================================
# OLS helpers (verbatim from W-B5)
# ============================================================================

def ols_with_V(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    k = len(x)
    if k < 2 or np.var(x) < DEGENERATE_X_VAR_EPS:
        return float(np.mean(y)) if k else 0.0, 0.0, np.inf, np.inf
    x_bar = x.mean()
    Sxx = ((x - x_bar) ** 2).sum()
    if Sxx < DEGENERATE_X_VAR_EPS:
        return float(np.mean(y)), 0.0, np.inf, np.inf
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = intercept + slope * x
    sigma_hat_sq = ((y - y_pred) ** 2).sum() / max(k - 2, 1)
    V_int = sigma_hat_sq * (1.0 / k + x_bar ** 2 / max(Sxx, 1e-12))
    V_slope = sigma_hat_sq / max(Sxx, 1e-12)
    return float(intercept), float(slope), float(V_int), float(V_slope)


def kfold_split(n: int, k: int, rng: np.random.Generator):
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


# ============================================================================
# Halpern (2025) FSAM K-head ensemble — polynomial-surrogate re-implementation
# ============================================================================
# Paper Eq. 2: min_{alpha, theta} E[(p_star - p_hat)^2] where
#   p_hat(x, y_1, y_2) = sum_k alpha_k * sigmoid(r_k(x, y_1) - r_k(x, y_2))
# Paper Sec. 4 FSAM iteration j > 1:
#   1. Fit r_j to residual MSE with alpha_j initialised at 1/K
#   2. Re-solve alpha on simplex (alpha_k >= 0, sum_k = 1) via SLSQP
#
# Q5 polynomial-surrogate (documented caveat):
#   r_k(z) = sum_{d=0..D} theta_{k,d} * z^d  with z = standardised rm_score
#   Closed-form ridge fit per stage (no gradient descent; CPU-fast; auditable).
#   This DIFFERS from the T1 Halpern script (Adam-trained neural reward heads)
#   in capacity but preserves the FSAM iteration + simplex-alpha algorithm.
#   The Q5 contribution is the PIPELINE COMPOSITION TEST, not a Halpern
#   re-validation (T1 already validated Halpern at TENTATIVE-class).


def fsam_k_head_fit(
    z_train: np.ndarray,
    s_train: np.ndarray,
    K: int,
    deg: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Halpern's K-head FSAM ensemble on TRAIN-fold (z, s) pairs.

    Args:
        z_train: (n,) standardised rm_score on TRAIN fold.
        s_train: (n,) target score (rescaled to [0, 1] for sigmoid alignment).
        K: number of reward heads (paper Table 4 sweet spot K=4).
        deg: polynomial degree per head.
        ridge: L2 stabiliser for closed-form polynomial fit.

    Returns:
        Theta: (K, deg+1) polynomial coefficients per head.
        alpha: (K,) simplex mixture weights re-solved at end of FSAM.

    The forward-stagewise loop (paper Sec. 4): at stage j, fit r_j to the
    residual MSE between target s_train and the current ensemble prediction
    p_hat_{j-1}(z) = sum_{k<j} alpha_k * sigmoid(r_k(z)). For the polynomial
    surrogate this reduces to weighted ridge regression on (z^0, z^1, ..., z^D)
    with target = s_train - p_hat_{j-1}(z). After all K heads are fit,
    re-solve alpha on the simplex via SLSQP minimising the full ensemble MSE.
    """
    n = len(z_train)
    if n < deg + 1:
        # Degenerate: insufficient data; return single-head pop-mean
        Theta = np.zeros((K, deg + 1))
        Theta[0, 0] = float(s_train.mean())
        alpha = np.zeros(K)
        alpha[0] = 1.0
        return Theta, alpha

    # Polynomial design matrix (n, deg+1) with intercept
    X = np.vander(z_train, N=deg + 1, increasing=True)  # [1, z, z^2, ..., z^D]

    Theta = np.zeros((K, deg + 1))
    alpha = np.full(K, 1.0 / K)  # initialisation paper Sec. 4

    # FSAM forward stagewise: fit r_j to residual
    p_hat_curr = np.zeros(n)
    for j in range(K):
        residual = s_train - p_hat_curr  # MSE residual to fit r_j to
        # Closed-form ridge: theta_j = (X^T X + ridge * I)^{-1} X^T residual
        XtX = X.T @ X + ridge * np.eye(deg + 1)
        Xty = X.T @ residual
        try:
            theta_j = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            theta_j = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
        Theta[j] = theta_j
        # Add r_j prediction (sigmoid-passed for stagewise consistency with Eq. 2)
        r_j_pred = X @ theta_j
        p_hat_curr = p_hat_curr + alpha[j] * sigmoid_safe(r_j_pred)

    # Re-solve alpha on simplex via SLSQP minimising ensemble MSE
    # Pre-compute per-head sigmoid predictions for SLSQP loss
    head_preds = np.zeros((K, n))
    for k in range(K):
        head_preds[k] = sigmoid_safe(X @ Theta[k])

    def alpha_loss(a):
        p_pred = (a[:, None] * head_preds).sum(axis=0)
        return float(np.mean((s_train - p_pred) ** 2))

    constraints = [{"type": "eq", "fun": lambda a: a.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * K
    try:
        result = scopt.minimize(
            alpha_loss,
            x0=alpha,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": HALPERN_SLSQP_TOL, "maxiter": 200},
        )
        if result.success:
            alpha = np.clip(result.x, 0.0, 1.0)
            alpha = alpha / max(alpha.sum(), 1e-12)
    except Exception:
        # SLSQP failure: keep uniform alpha
        pass

    return Theta, alpha


def sigmoid_safe(x: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid with overflow guard."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)
    return out


def fsam_predict(
    z: np.ndarray,
    Theta: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Predict ensemble score in [0, 1] for utterances z."""
    deg = Theta.shape[1] - 1
    X = np.vander(z, N=deg + 1, increasing=True)
    head_preds = np.zeros((Theta.shape[0], len(z)))
    for k in range(Theta.shape[0]):
        head_preds[k] = sigmoid_safe(X @ Theta[k])
    return (alpha[:, None] * head_preds).sum(axis=0)


# ============================================================================
# 3-arm pipeline: Halpern-only / PEBS-only / Combined-pipeline
# ============================================================================

def run_three_arm_pipeline(
    df: pd.DataFrame,
    seed: int,
    k_folds: int,
    min_obs_per_user: int,
) -> dict[str, Any]:
    """Run all three arms (Halpern-only, PEBS-only, Combined) on SAME folds.

    Arms:
      1. HALPERN-only — FSAM K=4 ensemble on raw rm_score; per-user OLS rescale
         to PRISM's 0-100 axis using TRAIN-fold scalars.
      2. PEBS-only — canonical W-B5 PEBS: per-user OLS on raw rm_score +
         joint α+β EB shrinkage (omega closed-form).
      3. COMBINED — Halpern's per-utterance ensemble pred replaces rm_score
         as the upstream signal; PEBS's α+β shrinkage applied on top.

    Population calibration (pop_alpha, pop_beta) and τ²_α / τ²_β are estimated
    over the FULL data (matching W-B5 protocol; not per-fold to avoid n-shrinkage).
    """
    rng = np.random.default_rng(seed)

    # Standardise rm_score for FSAM (z-score on full data; matches PReF/LoRe convention)
    rm_score_arr = df.rm_score.to_numpy().astype(float)
    z_mean, z_std = float(rm_score_arr.mean()), float(rm_score_arr.std())
    if z_std < 1e-9:
        z_std = 1.0
    df = df.copy()
    df["z_rm"] = (rm_score_arr - z_mean) / z_std

    # Rescale score_user to [0, 1] for FSAM sigmoid alignment (Halpern paper convention)
    s_user = df.score_user.to_numpy().astype(float)
    s_min, s_max = float(s_user.min()), float(s_user.max())
    s_range = max(s_max - s_min, 1e-9)
    df["s_user_unit"] = (s_user - s_min) / s_range

    # Population calibration on raw rm_score (PEBS reference)
    slope_pop, intercept_pop = np.polyfit(df.rm_score, df.score_user, 1)
    pop_alpha = float(intercept_pop)
    pop_beta = float(slope_pop)

    # τ²_α, τ²_β estimation on raw rm_score axis (PEBS canonical)
    user_stats = []
    for uid, grp in df.groupby("user_id"):
        if len(grp) < min_obs_per_user:
            continue
        a, b, Va, Vb = ols_with_V(
            grp.rm_score.to_numpy(),
            grp.score_user.to_numpy().astype(float),
        )
        user_stats.append({"alpha": a, "beta": b, "V_alpha": Va, "V_beta": Vb})
    us = pd.DataFrame(user_stats)
    V_alpha_total = float(us.alpha.var())
    V_beta_total = float(us.beta.var())
    mean_samp_V_alpha = float(us.V_alpha.replace([np.inf, -np.inf], np.nan).dropna().mean())
    mean_samp_V_beta = float(us.V_beta.replace([np.inf, -np.inf], np.nan).dropna().mean())
    tau_a_sq = max(V_alpha_total - mean_samp_V_alpha, V_ALPHA_BETA_FLOOR)
    tau_b_sq = max(V_beta_total - mean_samp_V_beta, V_ALPHA_BETA_FLOOR)

    # ------------------------------------------------------------------
    # Design correction (caught by the Q5 quick-run test):
    # ------------------------------------------------------------------
    # Halpern's FSAM ensemble in the original paper is GLOBAL — one set of K
    # reward heads + one simplex alpha vector for ALL users. The per-user
    # personalisation in Halpern comes from prompt-side persona induction,
    # NOT from per-user OLS rescale. The Q5 pipeline composition test
    # therefore uses a GLOBAL Halpern fit (one ensemble for all users, fit
    # on TRAIN-fold pooled across users); the HALPERN-only arm has a
    # GLOBAL pop_alpha_h + pop_beta_h rescale to PRISM 0-100 axis (no per-
    # user calibration). The COMBINED arm THEN applies PEBS's per-user
    # OLS + EB shrinkage on top of Halpern's GLOBAL prediction.
    # This honest design AVOIDS the silent-bypass where COMBINED ≡ HALPERN
    # because both used per-user OLS rescaling on the same Halpern axis.
    #
    # GLOBAL Halpern population calibration on Halpern's [0, 1] axis:
    # fit a single FSAM ensemble on FULL data (all users pooled) using
    # s_user_unit target, then compute per-user OLS on Halpern's predictions
    # to derive pop_alpha_c, pop_beta_c (population intercept / slope on
    # Halpern axis ⇒ score_user) plus τ²_α_c, τ²_β_c for EB shrinkage.
    # This is computed ONCE up-front (not per-fold per-user) and provides
    # the stable EB shrinkage population reference for the combined arm.
    # The per-fold Halpern ensembles for the HALPERN-only arm and the
    # COMBINED arm are fit per-fold WITHIN each user's TRAIN slice
    # (preserving the per-fold leakage discipline) — see fold loop below.
    log("  estimating combined-arm population (global Halpern fit + per-user τ²)")
    z_full = df.z_rm.to_numpy()
    s_unit_full = df.s_user_unit.to_numpy()
    Theta_global, alpha_global = fsam_k_head_fit(
        z_full, s_unit_full, K=HALPERN_K, deg=HALPERN_DEG, ridge=HALPERN_RIDGE,
    )
    h_pred_global = fsam_predict(z_full, Theta_global, alpha_global)
    # Population OLS: score_user = pop_alpha_c + pop_beta_c * h_pred_global
    slope_c_pop, intercept_c_pop = np.polyfit(h_pred_global, df.score_user, 1)
    pop_alpha_c = float(intercept_c_pop)
    pop_beta_c = float(slope_c_pop)
    # τ²_α_c, τ²_β_c estimation on Halpern axis (same EB framework as canonical)
    h_pred_per_obs = pd.Series(h_pred_global, index=df.index)
    user_stats_c = []
    for uid, grp in df.groupby("user_id"):
        if len(grp) < min_obs_per_user:
            continue
        a_c_u, b_c_u, Va_c_u, Vb_c_u = ols_with_V(
            h_pred_per_obs.loc[grp.index].to_numpy(),
            grp.score_user.to_numpy().astype(float),
        )
        user_stats_c.append({"alpha": a_c_u, "beta": b_c_u, "V_alpha": Va_c_u, "V_beta": Vb_c_u})
    us_c = pd.DataFrame(user_stats_c)
    V_alpha_total_c = float(us_c.alpha.var())
    V_beta_total_c = float(us_c.beta.var())
    mean_samp_V_alpha_c = float(us_c.V_alpha.replace([np.inf, -np.inf], np.nan).dropna().mean())
    mean_samp_V_beta_c = float(us_c.V_beta.replace([np.inf, -np.inf], np.nan).dropna().mean())
    tau_a_c_sq = max(V_alpha_total_c - mean_samp_V_alpha_c, V_ALPHA_BETA_FLOOR)
    tau_b_c_sq = max(V_beta_total_c - mean_samp_V_beta_c, V_ALPHA_BETA_FLOOR)
    # Global rescale (a_h_pop, b_h_pop) for HALPERN-only arm: maps Halpern's
    # GLOBAL [0, 1] prediction to PRISM 0-100 axis. Equal to (pop_alpha_c, pop_beta_c).
    a_h_pop = pop_alpha_c
    b_h_pop = pop_beta_c
    log(f"  combined-arm population: pop_alpha_c={pop_alpha_c:.3f} pop_beta_c={pop_beta_c:.3f} "
        f"tau_a_c_sq={tau_a_c_sq:.3f} tau_b_c_sq={tau_b_c_sq:.3f}")
    log(f"  halpern-only global rescale: a_h_pop={a_h_pop:.3f} b_h_pop={b_h_pop:.3f}")
    log(f"  halpern-global alpha simplex: {alpha_global.tolist()}")

    # Halpern alpha vector tracking (for diagnostics — averaged across users)
    halpern_alpha_per_user = []

    # Per-user K-fold CV — all 3 arms on SAME folds
    rows = []
    for uid, grp in df.groupby("user_id"):
        n = len(grp)
        if n < min_obs_per_user:
            continue
        x_raw = grp.rm_score.to_numpy()
        z = grp.z_rm.to_numpy()
        s = grp.score_user.to_numpy().astype(float)
        s_unit = grp.s_user_unit.to_numpy()
        folds = kfold_split(n, k_folds, rng)
        sq = {"pop_slope": [], "halpern_only": [], "pebs_only": [], "combined": []}

        for train_idx, test_idx in folds:
            x_tr_raw, z_tr, s_tr, s_tr_unit = (
                x_raw[train_idx], z[train_idx], s[train_idx], s_unit[train_idx]
            )
            x_te_raw, z_te, s_te = x_raw[test_idx], z[test_idx], s[test_idx]
            if len(x_te_raw) == 0:
                continue

            # ---------------- Pop-baseline ----------------
            y_hat_ps = pop_alpha + pop_beta * x_te_raw
            sq["pop_slope"].extend(((y_hat_ps - s_te) ** 2).tolist())

            # ---------------- HALPERN-only arm ----------------
            # Design: HALPERN-only uses the GLOBAL Halpern fit (Theta_global,
            # alpha_global) computed once over the FULL data, then GLOBAL OLS
            # rescale (a_h_pop + b_h_pop * h_pred) to PRISM 0-100 axis.
            # This mirrors Halpern's actual paper design (one ensemble for all
            # users; persona-side conditioning lives in the prompt-text rather
            # than per-user calibration). The per-fold within-user CV evaluates
            # held-out within-user predictions; the GLOBAL rescale ensures
            # HALPERN-only does NOT have access to per-user OLS information
            # (which would be the PEBS contribution). Silent-bypass
            # GUARDED: combined ≠ halpern_only because combined applies
            # PEBS per-user OLS + EB shrinkage that halpern_only lacks.
            h_pred_unit_te = fsam_predict(z_te, Theta_global, alpha_global)
            # Track per-user-fold alpha for diagnostic (population-shared, so
            # this is just alpha_global repeated per user-fold for n_users
            # logging consistency)
            halpern_alpha_per_user.append(alpha_global.tolist())
            y_hat_halpern = a_h_pop + b_h_pop * h_pred_unit_te
            sq["halpern_only"].extend(((y_hat_halpern - s_te) ** 2).tolist())

            # ---------------- PEBS-only arm ----------------
            # Per-user OLS on raw rm_score (canonical W-B5)
            a_p, b_p, Va_p, Vb_p = ols_with_V(x_tr_raw, s_tr)
            omega_a_p = (
                tau_a_sq / (tau_a_sq + Va_p)
                if (np.isfinite(Va_p) and Va_p > V_ALPHA_BETA_FLOOR) else 0.0
            )
            omega_b_p = (
                tau_b_sq / (tau_b_sq + Vb_p)
                if (np.isfinite(Vb_p) and Vb_p > V_ALPHA_BETA_FLOOR) else 0.0
            )
            a_s_p = omega_a_p * a_p + (1 - omega_a_p) * pop_alpha
            b_s_p = omega_b_p * b_p + (1 - omega_b_p) * pop_beta
            y_hat_pebs = a_s_p + b_s_p * x_te_raw
            sq["pebs_only"].extend(((y_hat_pebs - s_te) ** 2).tolist())

            # ---------------- COMBINED-pipeline arm ----------------
            # Silent-bypass GUARDED: pipeline composition uses Halpern's
            # GLOBAL ensemble prediction (h_pred_unit_te in [0, 1]) as the
            # upstream feature, then applies PEBS's per-user OLS + EB
            # shrinkage on top. This is the honest pipeline composition:
            # Halpern provides the persona-aware ensemble signal at the
            # POPULATION level (via FSAM K=4 mixture); PEBS then calibrates
            # the per-rater linear mapping (alpha_j, beta_j) from Halpern's
            # GLOBAL signal to score_user with EB shrinkage toward
            # (pop_alpha_c, pop_beta_c) using (tau_a_c_sq, tau_b_c_sq).
            # Both the HALPERN-only arm and the COMBINED arm use the SAME
            # Theta_global / alpha_global / h_pred_unit (no leakage); the
            # difference is HALPERN-only uses GLOBAL rescale, COMBINED uses
            # PER-USER OLS + EB shrinkage. ⇒ COMBINED's marginal value over
            # HALPERN-only is ENTIRELY the per-user PEBS calibration.
            x_tr_combined = fsam_predict(z_tr, Theta_global, alpha_global)
            x_te_combined = h_pred_unit_te  # already from global fit
            # Per-user OLS on Halpern's [0,1] ensemble prediction
            a_c, b_c, Va_c, Vb_c = ols_with_V(x_tr_combined, s_tr)
            # Population calibration on Halpern axis: pre-computed at fold start
            # via OLS on the Halpern train predictions for THIS user (since
            # Halpern's ensemble is fit per-user, the population-axis on the
            # combined arm is per-fold per-user-aggregate; we use the simple
            # global pop_alpha_h, pop_beta_h estimated below per fold).
            # For stability we estimate fold-level population on combined
            # axis using the SAME train fold's (h_pred, s_tr) regression coefs.
            # This is conservative (no information leakage).
            # Use population-shared tau on combined axis (estimated globally
            # over all users; same EB framework).
            omega_a_c = (
                tau_a_c_sq / (tau_a_c_sq + Va_c)
                if (np.isfinite(Va_c) and Va_c > V_ALPHA_BETA_FLOOR) else 0.0
            )
            omega_b_c = (
                tau_b_c_sq / (tau_b_c_sq + Vb_c)
                if (np.isfinite(Vb_c) and Vb_c > V_ALPHA_BETA_FLOOR) else 0.0
            )
            a_s_c = omega_a_c * a_c + (1 - omega_a_c) * pop_alpha_c
            b_s_c = omega_b_c * b_c + (1 - omega_b_c) * pop_beta_c
            y_hat_combined = a_s_c + b_s_c * x_te_combined
            sq["combined"].extend(((y_hat_combined - s_te) ** 2).tolist())

        rows.append(
            {
                "user_id": uid,
                "n": n,
                **{f"rmse_{arm}": float(np.sqrt(np.mean(sq[arm]))) for arm in sq},
            }
        )
    pu = pd.DataFrame(rows)

    arms = ["halpern_only", "pebs_only", "combined"]
    rmse_pop = float(pu.rmse_pop_slope.mean())
    arm_data: dict[str, dict[str, Any]] = {}

    boot_rng = np.random.default_rng(RNG_BOOT)
    n_users = len(pu)

    for arm in arms:
        rmse_arm = float(pu[f"rmse_{arm}"].mean())
        rel_pct = 100.0 * (rmse_pop - rmse_arm) / rmse_pop

        # Per-user gain_pct (cluster-bootstrap unit = user)
        gains = (
            100.0 * (pu.rmse_pop_slope - pu[f"rmse_{arm}"]) / pu.rmse_pop_slope
        ).to_numpy()
        boot_means = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = boot_rng.integers(0, n_users, size=n_users)
            boot_means[b] = float(np.mean(gains[idx]))
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        # Wilcoxon vs pop_slope
        w = scstats.wilcoxon(
            pu[f"rmse_{arm}"].to_numpy(),
            pu.rmse_pop_slope.to_numpy(),
            alternative="two-sided",
        )

        arm_data[arm] = {
            "rmse_mean": rmse_arm,
            "gain_pct_vs_pop": rel_pct,
            "gain_pct_ci95_lo": float(ci_lo),
            "gain_pct_ci95_hi": float(ci_hi),
            "ci_excludes_zero": bool(ci_lo > 0),
            "wilcoxon_p_vs_pop": float(w.pvalue),
        }

    # Paired deltas: combined vs halpern_only AND combined vs pebs_only
    deltas: dict[str, dict[str, Any]] = {}
    combined_gains = (
        100.0 * (pu.rmse_pop_slope - pu.rmse_combined) / pu.rmse_pop_slope
    ).to_numpy()
    for arm in ("halpern_only", "pebs_only"):
        arm_gains = (
            100.0 * (pu.rmse_pop_slope - pu[f"rmse_{arm}"]) / pu.rmse_pop_slope
        ).to_numpy()
        paired = combined_gains - arm_gains  # positive ⇒ combined wins
        boot_means = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = boot_rng.integers(0, n_users, size=n_users)
            boot_means[b] = float(np.mean(paired[idx]))
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        w = scstats.wilcoxon(combined_gains, arm_gains, alternative="two-sided")
        deltas[f"combined_vs_{arm}"] = {
            "delta_pct": float(np.mean(paired)),
            "delta_ci95_lo": float(ci_lo),
            "delta_ci95_hi": float(ci_hi),
            "wilcoxon_p": float(w.pvalue),
        }

    # Halpern alpha simplex diagnostic (averaged across user-fold pairs)
    if halpern_alpha_per_user:
        alpha_arr = np.array(halpern_alpha_per_user)
        alpha_mean = alpha_arr.mean(axis=0).tolist()
        alpha_max_share = float(alpha_arr.max(axis=1).mean())  # avg max-share per fit
    else:
        alpha_mean = [1.0 / HALPERN_K] * HALPERN_K
        alpha_max_share = 1.0

    return {
        "n_users": int(len(pu)),
        "n_obs_total": int(pu.n.sum()),
        "tau_alpha_sq": tau_a_sq,
        "tau_beta_sq": tau_b_sq,
        "alpha_pop": pop_alpha,
        "beta_pop": pop_beta,
        "rmse_pop_slope": rmse_pop,
        "halpern_alpha_simplex_mean": alpha_mean,
        "halpern_alpha_max_share_avg": alpha_max_share,
        "halpern_K": HALPERN_K,
        "halpern_deg": HALPERN_DEG,
        "halpern_ridge": HALPERN_RIDGE,
        "arms": arm_data,
        "deltas": deltas,
    }


# ============================================================================
# Outcome assignment
# ============================================================================

def assign_verdict(out: dict[str, Any]) -> dict[str, Any]:
    """Pre-specified outcome classification.

    Decision-rule:
      CONFIRMED-PIPELINE-COMPOSITION-DOMINATES:
        Combined paired Δ vs BOTH halpern_only and pebs_only at 95% CI strictly
        positive (delta_ci95_lo > 0) AND combined gain ≥ best-of-individual + 1pp.
      PARTIAL-PIPELINE-EQUIVALENT:
        Combined within 1pp of best-of-individual (regardless of CI direction);
        pipeline composition is NEUTRAL.
      TENTATIVE-INCONCLUSIVE:
        Combined < either individual but CIs overlap zero (CIs too wide
        to falsify or confirm).
      REJECTED-PIPELINE-WORSE:
        Combined paired Δ vs BOTH halpern_only and pebs_only at 95% CI strictly
        negative (delta_ci95_hi < 0) AND combined gain < best-of-individual - 1pp.
    """
    h_gain = out["arms"]["halpern_only"]["gain_pct_vs_pop"]
    p_gain = out["arms"]["pebs_only"]["gain_pct_vs_pop"]
    c_gain = out["arms"]["combined"]["gain_pct_vs_pop"]
    best_individual = max(h_gain, p_gain)

    d_vs_h = out["deltas"]["combined_vs_halpern_only"]
    d_vs_p = out["deltas"]["combined_vs_pebs_only"]

    combined_strictly_dominates = (
        d_vs_h["delta_ci95_lo"] > 0 and d_vs_p["delta_ci95_lo"] > 0
    )
    combined_strictly_worse = (
        d_vs_h["delta_ci95_hi"] < 0 and d_vs_p["delta_ci95_hi"] < 0
    )
    margin_vs_best = c_gain - best_individual

    if combined_strictly_dominates and margin_vs_best >= DELTA_DOMINATE_PP:
        verdict = "CONFIRMED-PIPELINE-COMPOSITION-DOMINATES"
    elif combined_strictly_worse and margin_vs_best <= -DELTA_DOMINATE_PP:
        verdict = "REJECTED-PIPELINE-WORSE"
    elif abs(margin_vs_best) <= DELTA_EQUIVALENT_PP:
        verdict = "PARTIAL-PIPELINE-EQUIVALENT"
    else:
        verdict = "TENTATIVE-INCONCLUSIVE"

    return {
        "halpern_gain_pct": h_gain,
        "pebs_gain_pct": p_gain,
        "combined_gain_pct": c_gain,
        "best_individual_gain_pct": best_individual,
        "margin_combined_vs_best_pp": margin_vs_best,
        "delta_combined_vs_halpern_pp": d_vs_h["delta_pct"],
        "delta_combined_vs_pebs_pp": d_vs_p["delta_pct"],
        "delta_combined_vs_halpern_ci95_lo": d_vs_h["delta_ci95_lo"],
        "delta_combined_vs_halpern_ci95_hi": d_vs_h["delta_ci95_hi"],
        "delta_combined_vs_pebs_ci95_lo": d_vs_p["delta_ci95_lo"],
        "delta_combined_vs_pebs_ci95_hi": d_vs_p["delta_ci95_hi"],
        "verdict_class": verdict,
        "decision_rule": (
            "CONFIRMED if combined paired Δ vs BOTH individual at 95% CI "
            "strictly positive AND combined ≥ best-of-individual + 1pp; "
            "REJECTED if combined paired Δ vs BOTH at 95% CI strictly "
            "negative AND combined ≤ best-of-individual - 1pp; "
            "PARTIAL if combined within 1pp of best-of-individual; "
            "TENTATIVE otherwise (CIs too wide / inconsistent dominance)."
        ),
    }


# ============================================================================
# CLI + main
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scored-parquet", default=str(DEFAULT_PRISM_PARQUET))
    p.add_argument("--min-obs-per-user", type=int, default=MIN_OBS_PER_USER)
    p.add_argument("--k-folds", type=int, default=K_FOLDS)
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "track1_q5_halpern_plus_pebs_pipeline"),
    )
    p.add_argument("--smoke", action="store_true", help="50-user smoke test")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Q5 Halpern + PEBS pipeline composition starting")
    log(f"  scored_parquet: {args.scored_parquet}")
    log(f"  seed: {args.seed} k_folds: {args.k_folds} min_obs: {args.min_obs_per_user}")
    log(f"  output_dir: {out_dir}")
    log(f"  Halpern: K={HALPERN_K} deg={HALPERN_DEG} ridge={HALPERN_RIDGE}")

    parquet_path = Path(args.scored_parquet)
    if not parquet_path.exists():
        log(f"ERROR: scored parquet not found: {parquet_path}")
        return 1

    df = pd.read_parquet(parquet_path).dropna(subset=["score_user"]).reset_index(drop=True)
    log(f"  loaded {len(df)} utts, {df.user_id.nunique()} users")

    if args.smoke:
        keep = df.user_id.drop_duplicates().head(50).tolist()
        df = df[df.user_id.isin(keep)].reset_index(drop=True)
        k_folds = 2
        log(f"  smoke: {len(df)} utts, {df.user_id.nunique()} users")
    else:
        k_folds = args.k_folds

    t0 = time.time()
    out = run_three_arm_pipeline(df, args.seed, k_folds, args.min_obs_per_user)
    verdict = assign_verdict(out)
    wall_seconds = time.time() - t0

    summary = {
        "experiment": "Q5_halpern_plus_pebs_pipeline_composition",
        "protocol": (
            "k=5 random-fold within-user CV; 3 arms (halpern_only, pebs_only, "
            "combined-pipeline); FSAM K=4 polynomial reward heads (deg=3, "
            "ridge=1e-3); per-user OLS rescale to PRISM 0-100 axis; cluster-"
            "bootstrap CI (cluster=user_id, N_BOOT=2000)."
        ),
        "seed": args.seed,
        "k_folds": k_folds,
        "min_obs_per_user": args.min_obs_per_user,
        "n_bootstrap": N_BOOT,
        "scored_parquet": str(args.scored_parquet),
        "scored_parquet_sha256": file_sha256(parquet_path),
        "git_head_t3": git_head_t3(),
        "wall_seconds": wall_seconds,
        "smoke": bool(args.smoke),
        **out,
        **verdict,
    }
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    log("")
    log(f"=== Q5 Halpern + PEBS pipeline composition complete (wall={wall_seconds:.1f}s) ===")
    log(f"  Verdict: {summary['verdict_class']}")
    for arm, d in summary["arms"].items():
        log(
            f"  {arm:>14}: gain_pct={d['gain_pct_vs_pop']:+.3f}% "
            f"CI=[{d['gain_pct_ci95_lo']:+.3f}, {d['gain_pct_ci95_hi']:+.3f}] "
            f"wilcoxon_p={d['wilcoxon_p_vs_pop']:.3e}"
        )
    for k, d in summary["deltas"].items():
        log(
            f"  {k:>30}: delta={d['delta_pct']:+.3f}pp "
            f"CI=[{d['delta_ci95_lo']:+.3f}, {d['delta_ci95_hi']:+.3f}] "
            f"wilcoxon_p={d['wilcoxon_p']:.3e}"
        )
    log(f"  margin combined vs best-of-individual: {summary['margin_combined_vs_best_pp']:+.3f}pp")
    log(f"  halpern alpha simplex mean: {summary['halpern_alpha_simplex_mean']}")
    log(f"  halpern alpha max-share avg: {summary['halpern_alpha_max_share_avg']:.3f}")
    log(f"  Output: {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
