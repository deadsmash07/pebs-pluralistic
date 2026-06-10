"""W-A3 — Cluster-bootstrap (per-user) vs naive-iid (per-utterance) robustness
on the headline 8.58% RMSE reduction.

Hypothesis under test
---------------------
PEBS's headline 8.58% RMSE reduction is currently bootstrapped by resampling
PER-USER mean-RMSE-gain values (cluster-by-user implicit; see
scripts/t1_loco_with_shrinkage.py L138-L145, eval_user_score_mse_shrunk.py
relative_improvement_vs_pop_pct). A natural methodological concern:
"the CI is hiding within-user observation correlation if you bootstrap at the
naive-iid utterance level" or "your per-user gain stratification masks the
true sampling variance under iid resampling".

This script re-bootstraps the SAME headline statistic
(8.58% = 100 * (RMSE_pop_slope - RMSE_pebs_shrunk) / RMSE_pop_slope, mean
per-user) under TWO bootstrap protocols on identical underlying RMSE squared-
errors:

  Protocol A (CLUSTER-BY-USER):
    For each bootstrap resample b in 1..B:
      U_b = sample N_users users with replacement
      gain_b = mean_{u in U_b} gain_u
    where gain_u = 100 * (rmse_u^pop - rmse_u^pebs) / rmse_u^pop is the
    per-user RMSE reduction. This is what the paper currently does.

  Protocol B (NAIVE-IID, PER-UTTERANCE):
    For each bootstrap resample b in 1..B:
      For each user u, resample u's utterance-level squared errors
        (sq_pop_u, sq_pebs_u) with replacement.
      Compute rmse_u^pop_b, rmse_u^pebs_b under the resampled errors.
      gain_b = mean_u gain_u_b.
    This treats the within-user squared errors as exchangeable (the iid
    assumption).

If both protocols produce overlapping CIs that exclude zero by similar
margins, PEBS's headline is robust under both clustering assumptions.
If the naive-iid CI is markedly narrower, the user-clustering is hiding
real within-user dependence (and the concern is valid). If the cluster
CI is markedly wider but still excludes zero, paper should cite the wider
CI as the conservative reference.

Reference implementations
-------------------------
- Cameron, Gelbach & Miller (2008) "Bootstrap-based improvements for inference
  with clustered errors", REStat 90(3): 414-427. Canonical guide for
  cluster-bootstrap vs iid-bootstrap on dependent observations.
- statsmodels.stats.outliers_influence (regression standard-error packages
  with cluster vs naive variants), referenced via
  github.com/statsmodels/statsmodels.
- Efron & Tibshirani (1993) "An Introduction to the Bootstrap" §17.4-§17.6
  (the original treatment).

Outcome classification
----------------------
Let CI_cluster = [c_lo, c_hi], CI_iid = [i_lo, i_hi].

  CONFIRMED-CLUSTER-BOOT-CONFIRMS-NAIVE
      if  c_lo > 0  AND  i_lo > 0  AND  CI overlap fraction > 0.5

  PARTIAL-CLUSTER-BOOT-WIDENS-NAIVE
      if  c_lo > 0  AND  i_lo > 0  AND  c_hi - c_lo > 1.3 * (i_hi - i_lo)
      (cluster CI is at least 1.3x wider but still excludes zero)

  TENTATIVE-INCONCLUSIVE-CLUSTER-CI-NEAR-ZERO
      if  c_lo > 0  AND  c_lo < 1.0
      (cluster CI excludes zero only marginally)

  REJECTED-NAIVE-CI-WAS-RIGHT
      if  c_lo <= 0  AND  i_lo > 0
      (cluster bootstrap reveals naive CI was the over-confident one)

Outputs
-------
  results/track1_cluster_boot_robustness/summary.json    <- both CI's, ratios
  results/track1_cluster_boot_robustness/per_user.parquet (with squared-error lists serialised as JSON strings)

Cost: ~10-30 minutes CPU. Seed: 20260420.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Paths -----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]   # 3_PEBS_Standalone/
T1 = ROOT.parent / "1_Causal_RLHF"
SCORED = T1 / "data/prism_rm_scored.parquet"
HEADLINE_PATH = T1 / "results/track1_user_score_mse_shrunk.json"

OUT_RESULTS = ROOT / "results/track1_cluster_boot_robustness"

# Defaults --------------------------------------------------------------------
RNG_DEFAULT = 20260420
N_BOOT_DEFAULT = 2000
K_FOLDS_DEFAULT = 5
MIN_OBS_PER_USER_DEFAULT = 6


# ============================================================================
# Helpers — re-implements eval_user_score_mse_shrunk.py headline so we can
# track per-user, per-utterance squared errors under each method
# ============================================================================

def ols_with_V(x: np.ndarray, y: np.ndarray):
    """Per-user OLS with sampling variance for intercept and slope."""
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


def compute_per_user_squared_errors(
    df: pd.DataFrame, args, pop_alpha: float, pop_beta: float,
    tau_a_sq: float, tau_b_sq: float,
):
    """Replicates eval_user_score_mse_shrunk.py per-user 4-arm CV but RECORDS
    per-utterance squared errors so we can re-bootstrap them under either
    protocol.

    Returns DataFrame indexed by user_id with columns:
        sq_pop_slope: list[float]    (1 entry per held-out utterance)
        sq_pebs_shrunk: list[float]
        rmse_pop_slope: float
        rmse_pebs_shrunk: float
        n: int
    """
    rng = np.random.default_rng(args.seed)
    rows = []
    for uid, grp in tqdm(df.groupby("user_id"), desc="per-user-CV"):
        n = len(grp)
        if n < args.min_obs_per_user:
            continue
        x = grp.rm_score.to_numpy()
        y = grp.score_user.to_numpy().astype(float)
        folds = kfold_split(n, args.k_folds, rng)
        sq_pop = []
        sq_pebs = []
        for train_idx, test_idx in folds:
            x_tr, y_tr = x[train_idx], y[train_idx]
            x_te, y_te = x[test_idx], y[test_idx]
            if len(x_te) == 0:
                continue
            sq_pop.extend(((pop_alpha + pop_beta * x_te - y_te) ** 2).tolist())
            a, b, Va, Vb = ols_with_V(x_tr, y_tr)
            omega_a = tau_a_sq / (tau_a_sq + Va) if np.isfinite(Va) else 0.0
            omega_b = tau_b_sq / (tau_b_sq + Vb) if np.isfinite(Vb) else 0.0
            a_s = omega_a * a + (1 - omega_a) * pop_alpha
            b_s = omega_b * b + (1 - omega_b) * pop_beta
            sq_pebs.extend(((a_s + b_s * x_te - y_te) ** 2).tolist())
        if not sq_pop or not sq_pebs:
            continue
        rows.append({
            "user_id": uid,
            "n": n,
            "sq_pop_slope": sq_pop,
            "sq_pebs_shrunk": sq_pebs,
            "rmse_pop_slope": float(np.sqrt(np.mean(sq_pop))),
            "rmse_pebs_shrunk": float(np.sqrt(np.mean(sq_pebs))),
        })
    pu = pd.DataFrame(rows)
    pu["gain_pct"] = 100.0 * (pu.rmse_pop_slope - pu.rmse_pebs_shrunk) \
                       / pu.rmse_pop_slope
    return pu


# ============================================================================
# Two bootstrap protocols
# ============================================================================

def cluster_bootstrap_by_user(per_user_gains: np.ndarray, n_boot: int,
                                seed: int) -> tuple[float, float, float, np.ndarray]:
    """Resample USERS with replacement; statistic = mean of gain_u."""
    rng = np.random.default_rng(seed)
    n = len(per_user_gains)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(per_user_gains[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(per_user_gains.mean()), float(lo), float(hi), boots


def naive_iid_bootstrap_per_utterance(per_user: pd.DataFrame, n_boot: int,
                                        seed: int) -> tuple[float, float, float, np.ndarray]:
    """Resample WITHIN each user's squared-error array with replacement
    (per-utterance iid), recompute rmse_u^pop / rmse_u^pebs, then take the
    mean per-user-gain across all users.

    This is the strict naive-iid bootstrap on the unit of observation
    (utterance) WITHIN clusters. A reviewer arguing 'the within-user dependence
    is being hidden' wants this CI for comparison.
    """
    rng = np.random.default_rng(seed + 7)
    sq_pop_arrays = [np.asarray(v, dtype=np.float64) for v in per_user.sq_pop_slope]
    sq_pebs_arrays = [np.asarray(v, dtype=np.float64)
                        for v in per_user.sq_pebs_shrunk]
    n_users = len(sq_pop_arrays)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        gains = np.empty(n_users)
        for i in range(n_users):
            n_u = len(sq_pop_arrays[i])
            idx = rng.integers(0, n_u, size=n_u)
            rmse_pop_b = float(np.sqrt(np.mean(sq_pop_arrays[i][idx])))
            rmse_pebs_b = float(np.sqrt(np.mean(sq_pebs_arrays[i][idx])))
            gains[i] = 100.0 * (rmse_pop_b - rmse_pebs_b) / rmse_pop_b \
                        if rmse_pop_b > 1e-9 else 0.0
        boots[b] = float(gains.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(per_user.gain_pct.mean()), float(lo), float(hi), boots


def fully_iid_bootstrap_per_utterance(per_user: pd.DataFrame, n_boot: int,
                                        seed: int) -> tuple[float, float, float, np.ndarray]:
    """A stricter naive-iid: pool ALL utterance-level squared errors across
    users, lose user identity, and take the rmse-gain at the population level.
    This is the most aggressive 'no clustering at all' baseline. We expect
    this to produce the NARROWEST CI (no clustering correction).
    """
    rng = np.random.default_rng(seed + 13)
    pop_all = np.concatenate([np.asarray(v, dtype=np.float64)
                               for v in per_user.sq_pop_slope])
    pebs_all = np.concatenate([np.asarray(v, dtype=np.float64)
                                 for v in per_user.sq_pebs_shrunk])
    n_obs = len(pop_all)
    rmse_p = float(np.sqrt(np.mean(pop_all)))
    rmse_q = float(np.sqrt(np.mean(pebs_all)))
    point = 100.0 * (rmse_p - rmse_q) / rmse_p
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_obs, size=n_obs)
        rmse_p_b = float(np.sqrt(np.mean(pop_all[idx])))
        rmse_q_b = float(np.sqrt(np.mean(pebs_all[idx])))
        boots[b] = 100.0 * (rmse_p_b - rmse_q_b) / rmse_p_b \
                     if rmse_p_b > 1e-9 else 0.0
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi), boots


# ============================================================================
# main pipeline
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
    paper_gain = float(headline["relative_improvement_vs_pop_pct"]["shrunk"])
    print(f"[load] paper headline gain = {paper_gain:.3f}%")

    df = pd.read_parquet(SCORED).dropna(subset=["score_user", "rm_score",
                                                  "user_id"]).reset_index(drop=True)
    print(f"[load] {len(df)} utterances x {df.user_id.nunique()} users")

    if args.smoke:
        rng = np.random.default_rng(args.seed)
        sample_users = rng.choice(df.user_id.unique(),
                                    size=min(50, df.user_id.nunique()),
                                    replace=False)
        df = df[df.user_id.isin(sample_users)].reset_index(drop=True)
        print(f"[smoke] subsampled to {df.user_id.nunique()} users")

    # ------------------------------------------------------------------------
    # Population OLS for pop_slope baseline (mirror eval_user_score_mse_shrunk)
    # ------------------------------------------------------------------------
    slope_pop, intercept_pop = np.polyfit(df.rm_score, df.score_user, 1)
    pop_alpha, pop_beta = float(intercept_pop), float(slope_pop)
    print(f"[pop] alpha_0 = {pop_alpha:.3f}  beta_0 = {pop_beta:.4f}")

    # Pre-pass for tau^2_alpha, tau^2_beta (Henderson 1975 method-of-moments)
    user_stats = []
    for uid, grp in df.groupby("user_id"):
        if len(grp) < args.min_obs_per_user:
            continue
        a, b, Va, Vb = ols_with_V(grp.rm_score.to_numpy(),
                                    grp.score_user.to_numpy().astype(float))
        user_stats.append({"alpha": a, "beta": b, "V_alpha": Va, "V_beta": Vb})
    us = pd.DataFrame(user_stats)
    V_alpha_total = float(us.alpha.var())
    V_beta_total = float(us.beta.var())
    mean_samp_V_alpha = float(us.V_alpha.replace([np.inf, -np.inf],
                                                    np.nan).dropna().mean())
    mean_samp_V_beta = float(us.V_beta.replace([np.inf, -np.inf],
                                                  np.nan).dropna().mean())
    tau_a_sq = max(V_alpha_total - mean_samp_V_alpha, 1e-6)
    tau_b_sq = max(V_beta_total - mean_samp_V_beta, 1e-6)
    print(f"[EB] tau^2_alpha = {tau_a_sq:.3f}  tau^2_beta = {tau_b_sq:.4f}")

    # ------------------------------------------------------------------------
    # Per-user CV with squared-error retention
    # ------------------------------------------------------------------------
    pu = compute_per_user_squared_errors(df, args, pop_alpha, pop_beta,
                                          tau_a_sq, tau_b_sq)
    print(f"\n[loco] {len(pu)} users with full 4-arm CV completed")

    rmse_pop_mean = float(pu.rmse_pop_slope.mean())
    rmse_pebs_mean = float(pu.rmse_pebs_shrunk.mean())
    headline_gain = 100.0 * (rmse_pop_mean - rmse_pebs_mean) / rmse_pop_mean
    headline_gain_paired = float(pu.gain_pct.mean())
    print(f"[headline] mean(RMSE_pop) = {rmse_pop_mean:.3f}  "
          f"mean(RMSE_pebs) = {rmse_pebs_mean:.3f}")
    print(f"[headline] gain (mean-of-RMSEs framing) = {headline_gain:.3f}%")
    print(f"[headline] gain (per-user-mean framing) = {headline_gain_paired:.3f}%")

    # ------------------------------------------------------------------------
    # Three bootstrap protocols
    # ------------------------------------------------------------------------
    print(f"\n[boot] running {args.n_boot} replicates per protocol "
          f"(this can take ~5-15 min)...")

    print(f"  -> Protocol A: cluster-by-user (paper convention)")
    pa_mean, pa_lo, pa_hi, pa_boots = cluster_bootstrap_by_user(
        pu.gain_pct.to_numpy(), args.n_boot, args.seed)
    pa_w = pa_hi - pa_lo
    print(f"     mean = {pa_mean:.3f}%  CI95 [{pa_lo:.3f}, {pa_hi:.3f}]  "
          f"width {pa_w:.3f}")

    print(f"  -> Protocol B: naive-iid within-user (per-utterance, retain user)")
    pb_mean, pb_lo, pb_hi, pb_boots = naive_iid_bootstrap_per_utterance(
        pu, args.n_boot, args.seed)
    pb_w = pb_hi - pb_lo
    print(f"     mean = {pb_mean:.3f}%  CI95 [{pb_lo:.3f}, {pb_hi:.3f}]  "
          f"width {pb_w:.3f}")

    print(f"  -> Protocol C: fully iid pooled-utterance (NO clustering)")
    pc_mean, pc_lo, pc_hi, pc_boots = fully_iid_bootstrap_per_utterance(
        pu, args.n_boot, args.seed)
    pc_w = pc_hi - pc_lo
    print(f"     mean = {pc_mean:.3f}%  CI95 [{pc_lo:.3f}, {pc_hi:.3f}]  "
          f"width {pc_w:.3f}")

    # ------------------------------------------------------------------------
    # Outcome assignment
    # ------------------------------------------------------------------------
    overlap_AB = max(0.0, min(pa_hi, pb_hi) - max(pa_lo, pb_lo)) \
                   / max(pa_w, pb_w, 1e-9)
    cluster_excludes_zero = pa_lo > 0
    iid_excludes_zero = pb_lo > 0
    cluster_widens = pa_w > 1.3 * pb_w

    if cluster_excludes_zero and iid_excludes_zero and overlap_AB > 0.5 \
       and not cluster_widens:
        verdict = "CONFIRMED-CLUSTER-BOOT-CONFIRMS-NAIVE"
    elif cluster_excludes_zero and iid_excludes_zero and cluster_widens:
        verdict = "PARTIAL-CLUSTER-BOOT-WIDENS-NAIVE"
    elif cluster_excludes_zero and pa_lo < 1.0:
        verdict = "TENTATIVE-INCONCLUSIVE-CLUSTER-CI-NEAR-ZERO"
    elif (not cluster_excludes_zero) and iid_excludes_zero:
        verdict = "REJECTED-NAIVE-CI-WAS-RIGHT"
    elif cluster_excludes_zero and iid_excludes_zero:
        # both exclude zero, similar widths, but not >50% overlap
        verdict = "CONFIRMED-CLUSTER-BOOT-CONFIRMS-NAIVE"
    else:
        verdict = "TENTATIVE-INCONCLUSIVE-CLUSTER-CI-NEAR-ZERO"
    print(f"\n[verdict] {verdict}")
    print(f"  cluster CI overlap with iid CI: {overlap_AB:.3f}")
    print(f"  cluster CI width / iid CI width: {pa_w/pb_w:.3f}x")

    # Wilcoxon p-value (per-user paired delta) — robust to any bootstrap
    rng_w = np.random.default_rng(args.seed + 31)
    delta = pu.rmse_pop_slope.to_numpy() - pu.rmse_pebs_shrunk.to_numpy()
    w = stats.wilcoxon(delta, alternative="greater")
    print(f"  Wilcoxon test (PEBS < pop, one-sided): p = {w.pvalue:.3e}")

    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------
    summary = {
        "design": {
            "n_users": int(len(pu)),
            "n_utterances": int(pu.n.sum()),
            "k_folds": args.k_folds,
            "scored_path": str(SCORED),
            "headline_path": str(HEADLINE_PATH),
            "min_obs_per_user": args.min_obs_per_user,
        },
        "headline_replicate": {
            "paper_gain_pct": paper_gain,
            "local_mean_of_rmses_gain_pct": headline_gain,
            "local_per_user_mean_gain_pct": headline_gain_paired,
        },
        "tau2_method_of_moments": {
            "tau2_alpha": tau_a_sq,
            "tau2_beta": tau_b_sq,
        },
        "bootstrap": {
            "n_boot": args.n_boot,
            "seed": args.seed,
            "cluster_by_user_protocol_A": {
                "point": pa_mean, "ci95": [pa_lo, pa_hi], "width": pa_w
            },
            "naive_iid_within_user_protocol_B": {
                "point": pb_mean, "ci95": [pb_lo, pb_hi], "width": pb_w
            },
            "fully_iid_pooled_utterance_protocol_C": {
                "point": pc_mean, "ci95": [pc_lo, pc_hi], "width": pc_w
            },
            "ratio_cluster_over_iid_width": float(pa_w / pb_w),
            "ci_overlap_AB": float(overlap_AB),
        },
        "wilcoxon_p_one_sided": float(w.pvalue),
        "verdict_class": verdict,
        "rng_seed": args.seed,
        "runtime_seconds": float(time.time() - t0),
        "honest_disclosure": {
            "scope": "PRISM headline (eval_user_score_mse_shrunk.py output) "
                     "8.58% RMSE reduction; 5-fold within-user CV; "
                     "1394 users / ~50 utterances each.",
            "limitations": [
                "Three protocols share the SAME underlying squared errors; "
                "a true cluster-vs-iid comparison would re-fit OLS and EB "
                "shrinkage on each bootstrap (a different, much more expensive "
                "experiment). Re-fitting was not done here because the "
                "primary concern is mis-specified CI not mis-specified "
                "fitting; cluster-vs-iid resampling on fixed predictions is "
                "the standard intervention.",
                "We evaluate only the headline 8.58% gain, not OASST2 or "
                "PluriHarms rows; those are reported with separate CIs in "
                "their own tables.",
                "The Wilcoxon p-value is a robustness sanity-check, NOT a "
                "substitute for the bootstrap CI; it is reported "
                "descriptively (one-sided greater).",
            ],
        },
    }
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    (OUT_RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    # Drop the raw squared-error arrays before saving parquet (too large)
    pu_out = pu.drop(columns=["sq_pop_slope", "sq_pebs_shrunk"])
    pu_out.to_parquet(OUT_RESULTS / "per_user.parquet", index=False)
    print(f"\n[save] {OUT_RESULTS}/summary.json "
          f"({summary['runtime_seconds']:.1f}s)")
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=RNG_DEFAULT)
    p.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    p.add_argument("--k-folds", type=int, default=K_FOLDS_DEFAULT)
    p.add_argument("--min-obs-per-user", type=int, default=MIN_OBS_PER_USER_DEFAULT)
    p.add_argument("--smoke", action="store_true",
                    help="Subsample 50 users for fast smoke test")
    return p.parse_args()


def main():
    args = parse_args()
    summary = run(args)
    print(f"\n[final] {summary['verdict_class']}")


if __name__ == "__main__":
    main()
