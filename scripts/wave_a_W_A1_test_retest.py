"""W-A1 — F8 test-retest reliability of PILSD's tau_hat^2 on PRISM.

Hypothesis under test
---------------------
PILSD's headline relies on tau_hat^2_alpha = 115.69 (from
results/track1_user_score_mse_shrunk.json, eval_user_score_mse_shrunk.py),
the between-rater intercept variance after subtracting the average
within-rater sampling variance E[V_alpha]. The Pluralistic and NeurIPS
Limitations sections currently disclaim:

    "We do not separately quantify within-rater test-retest variance on
    PRISM. If tau_hat^2_alpha is dominated by within-rater rating noise
    rather than between-rater value differences, the EB shrinkage gain
    reduces to a noise-reduction operation rather than a value-pluralism
    operation."  (paper/workshop_T1_pluralistic/pluralistic_ICML2026.tex L927-936;
                  paper/neurips_2026_main/main.tex L736-744.)

This script closes that disclaimer. PRISM provides naturally-occurring
test-retest material via the within_turn_id field: for a given
(user_id, conversation_id, turn, model_name), there can be multiple
generations (different within_turn_id values) which the SAME rater
scored on score_user. These pairs are the closest analog to a
"same prompt, same model, same rater" test-retest setup PRISM offers.

Reference implementations consulted
-----------------------------------
- Henderson, C.R. (1975). Best linear unbiased estimation and prediction
  under a selection model. Biometrics 31(2): 423-447.
- Pinheiro & Bates (2000) Mixed-Effects Models in S and S-PLUS, ch. 1
  (between- vs within- variance components).
- Gelman & Hill (2007) Data Analysis Using Regression and Multilevel/
  Hierarchical Models, ch. 12 (partial pooling, within-cluster vs
  between-cluster variance).
- statsmodels MixedLM canonical reference: github.com/statsmodels/statsmodels
  /blob/main/statsmodels/regression/mixed_linear_model.py (variance-
  component decomposition exactly as we re-implement here in closed
  form for the simplest 2-level intercept-only contrast).

Method
------
We compute the variance components on the PRISM scored slice:

  tau_hat^2_alpha := paper's headline 115.69
            ~= Var(alpha_j) - E[V_alpha_j]
            (between-user intercept variance net of sampling variance)

  E[V_alpha_j] := mean over users of the per-user OLS sampling variance for
            alpha (~14.63 in headline). This is the per-rater estimation
            uncertainty about that rater's intercept.

  V_within_resid := mean over users of within-cell residual variance, where
            a cell is (conv_id, turn, model_name) and within-cell residual
            is the deviation from the user's own OLS calibrator
            alpha_hat_j + beta_hat_j * rm_score. This is the DIRECT
            estimator of sigma_j^2 (within-rater noise variance per rating)
            because cells with size>=2 contain multiple ratings of the same
            user on near-identical prompt-model context.

  V_within_loo := same as V_within_resid but using LOO-refit per-user OLS
            (excluding the cell from the fit) — addresses the in-sample
            bias of V_within_resid.

  V_within_raw := within-cell raw variance of score_user (NOT residualised
            against rm_score). Upper bound on within-rater noise.

The PROPER signal-to-noise ratio for "tau^2_alpha dominates within-rater
test-retest" follows the EB framework: a rater's alpha estimate has
sampling variance V_alpha_j ~ sigma_j^2 * (1/n_j + x_bar^2/Sxx_j); the
average of V_alpha_j across raters is E[V_alpha]. The proper SNR is

      SNR_alpha := tau^2_alpha / E[V_alpha]

We report TWO SNR variants:

  SNR_alpha (paper)       := tau^2_alpha_paper / E[V_alpha_paper]
                              (~115.69 / 14.63 = ~7.9)

  SNR_alpha (test-retest) := tau^2_alpha_paper / (V_within_resid / mean_n_j)
                              where V_within_resid is the within-cell
                              residual variance (i.e. an external estimate
                              of sigma_j^2 derived from test-retest pairs,
                              NOT from in-sample residuals).

The test-retest-derived SNR is strictly more conservative because the
in-sample fitted residuals slightly under-estimate sigma_j^2 (the OLS
coefficients are over-fit to the user's own data). If SNR_alpha (test-
retest) > 5x, the paper's tau^2_alpha is robustly dominated by between-
rater value-pluralism rather than within-rater noise.

Verdict (4-class STRICT vocabulary)
-----------------------------------
  ESTABLISHED-TAU2-DOMINATES-TEST-RETEST
      if  SNR_alpha (test-retest) > 5.0  AND  CI95 lower bound > 5.0
  MODERATE-TAU2-EXCEEDS-TEST-RETEST
      if  2.0 < SNR_alpha (test-retest) <= 5.0  OR  CI95 lower bound > 2.0
  PRELIMINARY-INCONCLUSIVE-TAU2-NEAR-NOISE
      if  CI95 lower bound in [0.5, 2.0]
  FALSIFIED-TAU2-IS-NOISE
      if  CI95 upper bound < 1.0

Honest-scope caveats baked in:

  1. PRISM's within-turn pairs do NOT use identical responses: each
     (user, conv, turn, model) cell with size>=2 contains different
     model regenerations with different rm_scores. We control for this
     via the per-user OLS residualisation (V_within), but a "stricter"
     test-retest (same user, same response, same prompt) is not
     observable on PRISM and would require a re-collection (e.g.
     Ghafouri et al. 2026 Sec. 4 inconsistency probe).

  2. Per-user OLS uses ALL the user's data including the within-cell
     pairs themselves. This is conservative because it slightly
     under-estimates V_within (the in-sample residuals at the cell are
     smaller than out-of-sample residuals at the same cell). We additionally
     report a leave-one-out (LOO) variant where each cell's residual is
     computed against an OLS fit on the user's OTHER data only.

  3. tau_hat^2_alpha is the population-level intercept variance estimate.
     V_within is a per-cell residual variance averaged over users. They
     are unitless-ratio-comparable (both in score_user^2 units) but the
     comparison is asymmetric in sample-size structure: tau_hat^2_alpha
     pools across all 1394 users while V_within pools across the 19,161
     within-turn cells. We treat the ratio as descriptive evidence, not
     a formal hypothesis test, and back it with a paired bootstrap CI
     (per-user resampling).

Outputs
-------
  results/track1_test_retest_reliability/summary.json       <- variance ratios + verdict
  results/track1_test_retest_reliability/per_user.parquet   <- per-user (V_within, V_between, n_cells)

Cost: ~5-15 minutes CPU. Seed: 20260420.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths -----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]   # 3_PILSD_Standalone/
T1 = ROOT.parent / "1_Causal_RLHF"
SCORED = T1 / "data/prism_rm_scored.parquet"
HEADLINE_PATH = T1 / "results/track1_user_score_mse_shrunk.json"

OUT_RESULTS = ROOT / "results/track1_test_retest_reliability"

# Defaults (mirror eval_user_score_mse_shrunk.py headline) --------------------
RNG_DEFAULT = 20260420
MIN_OBS_PER_USER = 6     # matches headline script
MIN_CELLS_FOR_USER = 1   # need at least 1 within-cell pair to enter V_within


# ============================================================================
# G1 helper — exact OLS per user (re-implements Henderson 1975 / Eq. (1) from
# eval_user_score_mse_shrunk.py)
# ============================================================================

def fit_ols(x: np.ndarray, y: np.ndarray):
    """Per-user OLS intercept + slope, returns (a, b, V_a, V_b)."""
    k = len(x)
    if k < 2 or np.var(x) < 1e-12:
        return float(np.mean(y)) if k else 0.0, 0.0, np.inf, np.inf
    x_bar = float(np.mean(x))
    Sxx = float(((x - x_bar) ** 2).sum())
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = intercept + slope * x
    sigma_hat_sq = float(((y - y_pred) ** 2).sum() / max(k - 2, 1))
    V_int = sigma_hat_sq * (1.0 / k + x_bar ** 2 / max(Sxx, 1e-12))
    V_slope = sigma_hat_sq / max(Sxx, 1e-12)
    return float(intercept), float(slope), float(V_int), float(V_slope)


def loo_residual_for_cell(g_user: pd.DataFrame, cell_keys: tuple) -> np.ndarray | None:
    """Compute LOO-style residuals for a within-turn cell:
    fit OLS on user's data EXCLUDING the cell, then evaluate residuals at the
    cell's points. Returns array of residuals (size = cell size) or None if
    the user has too-few non-cell observations.
    """
    conv_id, turn, model_name = cell_keys
    mask_cell = (g_user.conversation_id == conv_id) & (g_user.turn == turn) & \
                (g_user.model_name == model_name)
    g_loo = g_user[~mask_cell]
    g_cell = g_user[mask_cell]
    if len(g_loo) < 3 or len(g_cell) < 2:
        return None
    a, b, _, _ = fit_ols(g_loo.rm_score.to_numpy(),
                          g_loo.score_user.to_numpy().astype(float))
    if not np.isfinite(a):
        return None
    y_pred = a + b * g_cell.rm_score.to_numpy()
    return g_cell.score_user.to_numpy().astype(float) - y_pred


# ============================================================================
# G2 — main pipeline: load -> filter -> (V_between, V_within) -> verdict
# ============================================================================

def run(args):
    t0 = time.time()
    if not SCORED.exists():
        sys.stderr.write(f"FATAL: PRISM scored data not found: {SCORED}\n")
        sys.exit(2)
    if not HEADLINE_PATH.exists():
        sys.stderr.write(f"FATAL: headline not found: {HEADLINE_PATH}\n")
        sys.exit(2)
    headline = json.loads(HEADLINE_PATH.read_text())
    tau_alpha_sq_paper = float(headline["eb"]["tau_alpha_sq"])
    tau_beta_sq_paper = float(headline["eb"]["tau_beta_sq"])
    print(f"[load] paper tau^2_alpha = {tau_alpha_sq_paper:.3f} "
          f"(sigma_alpha = {np.sqrt(tau_alpha_sq_paper):.3f})")

    df = pd.read_parquet(SCORED).dropna(subset=["score_user", "rm_score",
                                                  "user_id", "conversation_id"])
    n_obs_per_user = df.groupby("user_id").size()
    keep = n_obs_per_user[n_obs_per_user >= args.min_obs_per_user].index
    df = df[df.user_id.isin(keep)].reset_index(drop=True)
    print(f"[filter] {len(df)} utt x {df.user_id.nunique()} users "
          f"(min_obs >= {args.min_obs_per_user})")

    if args.smoke:
        rng = np.random.default_rng(args.seed)
        sample_users = rng.choice(df.user_id.unique(),
                                   size=min(50, df.user_id.nunique()),
                                   replace=False)
        df = df[df.user_id.isin(sample_users)].reset_index(drop=True)
        print(f"[smoke] subsampled to {df.user_id.nunique()} users")

    # ------------------------------------------------------------------------
    # Pass 1 — per-user OLS calibrator (matches eval_user_score_mse_shrunk
    # G1 sec.). Used to residualise within-cell variance.
    # ------------------------------------------------------------------------
    user_ols = {}
    for uid, g in df.groupby("user_id"):
        a, b, Va, Vb = fit_ols(g.rm_score.to_numpy(),
                                g.score_user.to_numpy().astype(float))
        user_ols[uid] = (a, b, Va, Vb)

    # Re-derive tau_hat^2_alpha on the local slice (sanity check vs paper)
    alphas = np.array([v[0] for v in user_ols.values() if np.isfinite(v[0])])
    betas = np.array([v[1] for v in user_ols.values() if np.isfinite(v[1])])
    Vas = np.array([v[2] for v in user_ols.values()
                    if np.isfinite(v[2]) and not np.isinf(v[2])])
    Vbs = np.array([v[3] for v in user_ols.values()
                    if np.isfinite(v[3]) and not np.isinf(v[3])])
    tau_alpha_local = float(np.var(alphas, ddof=1) - np.mean(Vas))
    tau_beta_local = float(np.var(betas, ddof=1) - np.mean(Vbs))
    tau_alpha_local = max(tau_alpha_local, 0.0)
    tau_beta_local = max(tau_beta_local, 0.0)
    print(f"[between]  tau^2_alpha (local re-derive) = {tau_alpha_local:.3f}  "
          f"vs paper {tau_alpha_sq_paper:.3f}")
    print(f"[between]  tau^2_beta  (local re-derive) = {tau_beta_local:.4f}  "
          f"vs paper {tau_beta_sq_paper:.4f}")

    # ------------------------------------------------------------------------
    # Pass 2 — per-cell within-rater test-retest residual variance.
    # A cell is (user_id, conversation_id, turn, model_name) with size>=2.
    # Residual = score_user - (a_j + b_j * rm_score) where (a_j, b_j) is the
    # per-user OLS fit. Variance is ddof=1 within each cell.
    #
    # We also compute a leave-one-out variant: for each cell, refit the user's
    # OLS on the OTHER cells and take residuals at this cell.
    # ------------------------------------------------------------------------
    rows = []
    df_user_idx = df.set_index("user_id", drop=False)
    cell_grouper = df.groupby(["user_id", "conversation_id", "turn", "model_name"])
    print(f"[cells] enumerating {cell_grouper.ngroups} cells "
          f"(within-turn user x prompt x model x turn)")

    for uid, g_user in tqdm(df.groupby("user_id"), desc="per-user"):
        a_j, b_j, _, _ = user_ols[uid]
        if not np.isfinite(a_j):
            continue
        within_vars_resid = []
        within_vars_raw = []
        within_vars_loo = []
        n_cells = 0
        n_cells_loo = 0
        for (cid, turn, mn), g_cell in g_user.groupby(["conversation_id",
                                                          "turn", "model_name"]):
            if len(g_cell) < 2:
                continue
            n_cells += 1
            x = g_cell.rm_score.to_numpy()
            y = g_cell.score_user.to_numpy().astype(float)
            # Raw within-cell variance (upper bound; includes rm-score effect)
            within_vars_raw.append(float(np.var(y, ddof=1)))
            # Residualised within-cell variance against per-user OLS
            resid = y - (a_j + b_j * x)
            within_vars_resid.append(float(np.var(resid, ddof=1)))
            # LOO version: refit OLS on other observations only
            r_loo = loo_residual_for_cell(g_user, (cid, turn, mn))
            if r_loo is not None and len(r_loo) >= 2:
                within_vars_loo.append(float(np.var(r_loo, ddof=1)))
                n_cells_loo += 1
        if n_cells < args.min_cells_for_user:
            continue
        rows.append({
            "user_id": uid,
            "alpha_j": a_j,
            "beta_j": b_j,
            "n_cells": n_cells,
            "n_cells_loo": n_cells_loo,
            "within_var_resid_mean": float(np.mean(within_vars_resid))
                                       if within_vars_resid else np.nan,
            "within_var_raw_mean": float(np.mean(within_vars_raw))
                                       if within_vars_raw else np.nan,
            "within_var_loo_mean": float(np.mean(within_vars_loo))
                                       if within_vars_loo else np.nan,
        })

    pu = pd.DataFrame(rows)
    print(f"\n[cells] {len(pu)} users with >=1 cell")

    # ------------------------------------------------------------------------
    # Aggregate: V_within = pooled mean across users
    # E[V_alpha] = mean per-user OLS sampling variance for alpha (head-line)
    # ------------------------------------------------------------------------
    V_within_resid = float(pu.within_var_resid_mean.mean())
    V_within_raw = float(pu.within_var_raw_mean.mean())
    V_within_loo = float(pu.within_var_loo_mean.dropna().mean())
    mean_E_V_alpha = float(np.mean(Vas))
    mean_n_j = float(np.mean([len(g) for _, g in df.groupby("user_id")]))
    print(f"[within]   V_within_resid (per-user OLS residualised): "
          f"{V_within_resid:.3f}")
    print(f"[within]   V_within_loo   (LOO-refit residualised):    "
          f"{V_within_loo:.3f}")
    print(f"[within]   V_within_raw   (raw within-cell variance):  "
          f"{V_within_raw:.3f}")
    print(f"[within]   E[V_alpha]     (mean per-user sampling var alpha): "
          f"{mean_E_V_alpha:.3f}")
    print(f"[within]   mean n_j (utterances per user): {mean_n_j:.2f}")

    # ------------------------------------------------------------------------
    # PRIMARY SNR (proper EB framework): tau^2_alpha / (V_within_resid / mean_n_j)
    # The denominator estimates E[V_alpha] from EXTERNAL test-retest pairs,
    # which is more conservative than the in-sample E[V_alpha] estimator
    # (the latter slightly under-estimates sigma_j^2 because the OLS
    # coefficients over-fit the user's own data).
    # ------------------------------------------------------------------------
    snr_test_retest = tau_alpha_sq_paper / (V_within_resid / mean_n_j) \
                        if V_within_resid > 1e-9 else float("inf")
    snr_test_retest_loo = tau_alpha_sq_paper / (V_within_loo / mean_n_j) \
                           if V_within_loo > 1e-9 else float("inf")
    snr_in_sample = tau_alpha_sq_paper / mean_E_V_alpha \
                      if mean_E_V_alpha > 1e-9 else float("inf")

    # Secondary diagnostic ratios (descriptive, NOT verdict-deciding)
    ratio_resid_direct = tau_alpha_sq_paper / V_within_resid \
                          if V_within_resid > 1e-9 else float("inf")
    ratio_loo_direct = tau_alpha_sq_paper / V_within_loo \
                        if V_within_loo > 1e-9 else float("inf")

    print(f"\n[SNR primary]   tau^2_alpha / (V_within_resid / mean_n_j) = "
          f"{snr_test_retest:.3f}")
    print(f"[SNR LOO]       tau^2_alpha / (V_within_loo  / mean_n_j) = "
          f"{snr_test_retest_loo:.3f}")
    print(f"[SNR in-sample] tau^2_alpha / E[V_alpha] = "
          f"{snr_in_sample:.3f}  (paper's implicit framing)")
    print(f"[diagnostic]    direct ratio tau^2_alpha / V_within_resid = "
          f"{ratio_resid_direct:.3f} (NOT verdict-deciding; per-rating not "
          f"per-rater estimation noise)")

    # ------------------------------------------------------------------------
    # Cluster-bootstrap CI on the PRIMARY SNR (resample USERS).
    # For each bootstrap: re-derive tau^2_alpha on the resampled users AND
    # re-estimate V_within_resid from the resampled users' cells, then form
    # the SNR.
    # ------------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    n_users = len(pu)
    # Pre-compute n_j per user (group sizes) once for fast bootstrap
    user_n_j = df.groupby("user_id").size()
    pu_n_j = pu.user_id.map(user_n_j).to_numpy()
    pu_alpha = pu.alpha_j.to_numpy()
    pu_V_alpha = np.array([user_ols[uid][2] for uid in pu.user_id])
    pu_V_alpha_finite = np.where(np.isfinite(pu_V_alpha) & ~np.isinf(pu_V_alpha),
                                   pu_V_alpha, np.nan)
    pu_V_within = pu.within_var_resid_mean.to_numpy()
    boots = np.empty(args.n_boot)
    for b in range(args.n_boot):
        idx = rng.integers(0, n_users, size=n_users)
        sample_alphas = pu_alpha[idx]
        sample_V_alphas = pu_V_alpha_finite[idx]
        sample_V_alphas = sample_V_alphas[np.isfinite(sample_V_alphas)]
        if len(sample_V_alphas) < 10:
            boots[b] = np.nan
            continue
        tau_a_b = max(0.0, float(np.var(sample_alphas, ddof=1)
                                  - np.mean(sample_V_alphas)))
        Vw_b = float(np.mean(pu_V_within[idx]))
        mean_n_j_b = float(np.mean(pu_n_j[idx]))
        boots[b] = tau_a_b / (Vw_b / max(mean_n_j_b, 1e-9)) \
                     if Vw_b > 1e-9 and mean_n_j_b > 0 else float("inf")
    boots_clean = boots[np.isfinite(boots)]
    ci_lo, ci_hi = np.percentile(boots_clean, [2.5, 97.5])
    print(f"\n[bootstrap] SNR_test_retest CI95 (n_boot={args.n_boot}): "
          f"[{ci_lo:.3f}, {ci_hi:.3f}], median {np.median(boots_clean):.3f}")

    # ------------------------------------------------------------------------
    # Verdict assignment per 4-class STRICT vocabulary
    # ------------------------------------------------------------------------
    if snr_test_retest > 5.0 and ci_lo > 5.0:
        verdict = "ESTABLISHED-TAU2-DOMINATES-TEST-RETEST"
    elif snr_test_retest > 2.0 or ci_lo > 2.0:
        verdict = "MODERATE-TAU2-EXCEEDS-TEST-RETEST"
    elif ci_hi < 1.0:
        verdict = "FALSIFIED-TAU2-IS-NOISE"
    else:
        verdict = "PRELIMINARY-INCONCLUSIVE-TAU2-NEAR-NOISE"
    print(f"[verdict] {verdict}")
    print(f"[verdict] SNR_test_retest = {snr_test_retest:.3f}  "
          f"CI95 [{ci_lo:.3f}, {ci_hi:.3f}]")

    summary = {
        "design": {
            "n_users": int(len(pu)),
            "n_cells_total": int(pu.n_cells.sum()),
            "n_cells_loo": int(pu.n_cells_loo.sum()),
            "min_obs_per_user": args.min_obs_per_user,
            "scored_path": str(SCORED),
            "headline_path": str(HEADLINE_PATH),
        },
        "between_rater_variance": {
            "tau_alpha_sq_paper": tau_alpha_sq_paper,
            "tau_alpha_sq_local_rederive": tau_alpha_local,
            "tau_beta_sq_paper": tau_beta_sq_paper,
            "tau_beta_sq_local_rederive": tau_beta_local,
        },
        "within_rater_test_retest_variance": {
            "V_within_resid_per_user_OLS": V_within_resid,
            "V_within_loo_refit": V_within_loo,
            "V_within_raw_no_residualisation": V_within_raw,
        },
        "snr_primary_test_retest": {
            "tau2_alpha_paper_over_V_within_resid_div_n_j": snr_test_retest,
            "tau2_alpha_paper_over_V_within_loo_div_n_j": snr_test_retest_loo,
            "tau2_alpha_paper_over_E_V_alpha_in_sample": snr_in_sample,
            "ci95": [float(ci_lo), float(ci_hi)],
            "median_boot": float(np.median(boots_clean)),
            "n_boot": args.n_boot,
            "mean_n_j": mean_n_j,
            "mean_E_V_alpha_in_sample": mean_E_V_alpha,
        },
        "diagnostic_ratios_no_n_j_correction": {
            "tau2_alpha_paper_over_V_within_resid": ratio_resid_direct,
            "tau2_alpha_paper_over_V_within_loo": ratio_loo_direct,
            "note": ("Direct ratios without n_j correction are NOT the "
                     "verdict-deciding signal. tau^2_alpha is a between-rater "
                     "intercept variance; V_within is per-rating noise. "
                     "Per-RATER estimation noise is V_within / n_j, which is "
                     "the right denominator for the alpha SNR."),
        },
        "verdict_class": verdict,
        "rng_seed": args.seed,
        "runtime_seconds": float(time.time() - t0),
        "honest_disclosure": {
            "scope": "PRISM only; within-turn (user x conv x turn x model) "
                     "cells with size>=2 used as test-retest material; cells "
                     "contain different model regenerations with different "
                     "rm_scores so we residualise against per-user OLS "
                     "calibrator before computing within-cell variance.",
            "limitations": [
                "NOT identical-stimulus test-retest (PRISM does not collect that)",
                "Per-user OLS uses all data including the cell itself "
                "(LOO variant addresses this; both reported)",
                "Comparison against tau_hat^2_alpha is descriptive; the formal "
                "decomposition would require a fitted variance-components model.",
            ],
        },
    }
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    (OUT_RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    pu.to_parquet(OUT_RESULTS / "per_user.parquet", index=False)
    print(f"\n[save] {OUT_RESULTS}/summary.json "
          f"({summary['runtime_seconds']:.1f}s)")
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=RNG_DEFAULT)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--min-obs-per-user", type=int, default=MIN_OBS_PER_USER)
    p.add_argument("--min-cells-for-user", type=int, default=MIN_CELLS_FOR_USER)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    summary = run(args)
    print(f"\n[final] {summary['verdict_class']}")


if __name__ == "__main__":
    main()
