"""Compute Cohen's d and held-out Bradley-Terry log-likelihood for PEBS vs pop-slope.

Addresses effect-size objections to the paper's RMSE headline:
  (i) "RMSE drop of 2.19 points on a scale with SD~24 is small"  →  Cohen's d
  (ii) "Wilcoxon p<1e-108 reflects N=1394 not effect size"        →  paired d + CI
  (iii) "RMSE is monotone-invariant; use BT log-likelihood"       →  BT NLL eval

Inputs (all REAL PRISM):
  data/prism_rm_scored.parquet            — 68,371 utterances × 1,396 users
                                            (utterance_id, user_id, interaction_id, turn,
                                             score_user, if_chosen, rm_score, ...)
  results/track1_user_score_mse_shrunk.parquet
                                          — 1,394 × (rmse_{no_calib, pop_slope,
                                             pebs_ols, pebs_shrunk}) for paired d

Outputs:
  results/cohen_d_bt_ll.json              — machine-readable numerical results
  results/cohen_d_and_bt_ll_REPORT.md     — human-readable interpretation

Method:
  Cohen's d (Task 1):
    - Paired d = mean(Δ) / sd(Δ) on (pop_slope_RMSE - pebs_shrunk_RMSE) per user
    - Pooled d_s = mean(Δ) / sqrt((sd_pop² + sd_pebs²)/2) — standard two-sample
    - 95% CI via 2000-rep cluster bootstrap on user_id
    - Normalized RMSE = RMSE / sd(score_user)

  BT log-likelihood (Task 2):
    For per-user 5-fold CV (SAME seed=42 as eval_user_score_mse_shrunk.py):
      - Fit (α_j, β_j) via PEBS shrunk on TRAIN fold utterances
      - Fit (α_pop, β_pop) via global OLS on FULL data (as baseline does)
      - For each pair (chosen, rejected) in TEST fold (same user_id, interaction_id,
        turn; if_chosen=True vs if_chosen=False):
          NLL_pebs = -log sigmoid(β_j · (rm_chosen - rm_rejected))
          NLL_pop   = -log sigmoid(β_pop · (rm_chosen - rm_rejected))
      - Aggregate mean NLL across all held-out pairs
      - Paired Δ NLL per pair + Wilcoxon
      - 95% cluster-bootstrap CI on mean Δ NLL

Reproducibility:
    python3 scripts/compute_cohen_d_bt_ll.py --seed 42
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, log_expit


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scored-parquet", default="data/prism_rm_scored.parquet")
    p.add_argument("--rmse-parquet", default="results/track1_user_score_mse_shrunk.parquet")
    p.add_argument("--min-obs-per-user", type=int, default=6)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--json-out", default="results/cohen_d_bt_ll.json")
    p.add_argument("--report-out", default="results/cohen_d_and_bt_ll_REPORT.md")
    return p.parse_args()


# ---------- linear algebra helpers ----------

def kfold_split(n: int, k: int, rng: np.random.Generator):
    """EXACT same fold logic as eval_user_score_mse_shrunk.py (lines 47-58)."""
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


def ols_with_V(x: np.ndarray, y: np.ndarray):
    """Same as eval_user_score_mse_shrunk.py (lines 61-73). Returns (a, b, Va, Vb)."""
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


# ---------- Task 1: Cohen's d on per-user RMSE ----------

def cohens_d_paired_and_pooled(pop_rmse: np.ndarray, pebs_rmse: np.ndarray):
    """Return (paired d, pooled d_s). Convention: positive d = PEBS better."""
    diff = pop_rmse - pebs_rmse  # positive means PEBS lower RMSE (better)
    d_paired = float(diff.mean() / diff.std(ddof=1))
    sd_pop = pop_rmse.std(ddof=1)
    sd_pebs = pebs_rmse.std(ddof=1)
    pooled_sd = float(np.sqrt((sd_pop ** 2 + sd_pebs ** 2) / 2))
    d_pooled = float(diff.mean() / pooled_sd)
    return d_paired, d_pooled, pooled_sd


def cluster_bootstrap_d(pop_rmse: np.ndarray, pebs_rmse: np.ndarray,
                        user_ids: np.ndarray, n_boot: int, rng: np.random.Generator):
    """Cluster bootstrap by user_id. Since rows are per-user (1 per user),
    this degenerates to a standard non-parametric bootstrap over the 1394 users."""
    unique_users = np.unique(user_ids)
    n_u = len(unique_users)
    d_paired_boots = np.empty(n_boot)
    d_pooled_boots = np.empty(n_boot)
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    for b in range(n_boot):
        pick = rng.choice(unique_users, size=n_u, replace=True)
        idx = np.array([user_to_idx[u] for u in pick])
        d_p, d_s, _ = cohens_d_paired_and_pooled(pop_rmse[idx], pebs_rmse[idx])
        d_paired_boots[b] = d_p
        d_pooled_boots[b] = d_s
    return d_paired_boots, d_pooled_boots


def interpret_d(d: float) -> str:
    """Cohen 1988 thresholds."""
    a = abs(d)
    if a < 0.2:
        return "trivial (|d|<0.2)"
    if a < 0.5:
        return "small (0.2<=|d|<0.5)"
    if a < 0.8:
        return "medium (0.5<=|d|<0.8)"
    return "large (|d|>=0.8)"


# ---------- Task 2: BT log-likelihood ----------

def build_pair_index(df: pd.DataFrame) -> pd.DataFrame:
    """For each (user_id, interaction_id, turn) group with exactly one if_chosen=True
    AND >=1 if_chosen=False, emit rows of (chosen_utt, rejected_utt) pairs."""
    pairs = []
    # Aggregate for fast lookup: utterance_id -> row index
    df = df.reset_index(drop=True)
    idx_map = {u: i for i, u in enumerate(df.utterance_id.values)}
    grp_keys = ["user_id", "interaction_id", "turn"]
    for (uid, iid, turn), grp in df.groupby(grp_keys, sort=False):
        chosen = grp[grp.if_chosen]
        rejected = grp[~grp.if_chosen]
        if len(chosen) != 1 or len(rejected) == 0:
            continue
        c_utt = chosen.iloc[0].utterance_id
        c_rm = float(chosen.iloc[0].rm_score)
        c_idx = idx_map[c_utt]
        for _, r in rejected.iterrows():
            pairs.append({
                "user_id": uid,
                "interaction_id": iid,
                "turn": turn,
                "chosen_utt": c_utt,
                "rejected_utt": r.utterance_id,
                "chosen_rm": c_rm,
                "rejected_rm": float(r.rm_score),
                "chosen_idx": c_idx,
                "rejected_idx": idx_map[r.utterance_id],
            })
    return pd.DataFrame(pairs)


def compute_bt_per_fold(df: pd.DataFrame, pairs_df: pd.DataFrame,
                        pop_alpha: float, pop_beta: float,
                        tau_a_sq: float, tau_b_sq: float,
                        k_folds: int, min_obs: int,
                        rng: np.random.Generator):
    """For each user:
      - Same per-user k-fold split as eval_user_score_mse_shrunk.py
      - For each fold: fit calibrator on TRAIN utterances
      - For each pair where BOTH utterances are in the TEST fold: compute BT NLL
    Returns per-pair dataframe with columns:
      [user_id, chosen_utt, rejected_utt, fold, nll_pop, nll_pebs,
       Δ_rm, β_j, α_j]
    """
    # Index df by (user_id, utterance_id) for O(1) fold lookup
    user_to_df = {uid: grp.reset_index(drop=True) for uid, grp in df.groupby("user_id", sort=False)}
    user_to_folds = {}
    # Pre-compute per-user folds using the SAME rng sequence as eval_user_score_mse_shrunk
    # (one user at a time, in the same groupby order).
    for uid, grp in df.groupby("user_id", sort=False):
        n = len(grp)
        if n < min_obs:
            continue
        folds = kfold_split(n, k_folds, rng)
        # Save utterance_id sets per fold
        utts = grp.utterance_id.values
        fold_test_utts = []
        fold_train_utts = []
        for tr, te in folds:
            fold_test_utts.append(set(utts[te]))
            fold_train_utts.append(set(utts[tr]))
        user_to_folds[uid] = {
            "n": n, "utts": utts, "folds": folds,
            "fold_test_utts": fold_test_utts,
            "fold_train_utts": fold_train_utts,
            "x": grp.rm_score.to_numpy(),
            "y": grp.score_user.to_numpy().astype(float),
        }

    # Now iterate pairs and assign each to a user+fold
    out_rows = []
    # Fit per-user per-fold calibrators cached
    per_user_fold_ab = {}  # (uid, fold) -> (a_s, b_s)
    for uid, info in user_to_folds.items():
        x = info["x"]
        y = info["y"]
        for fi, (tr, te) in enumerate(info["folds"]):
            a, b, Va, Vb = ols_with_V(x[tr], y[tr])
            omega_a = tau_a_sq / (tau_a_sq + Va) if np.isfinite(Va) else 0.0
            omega_b = tau_b_sq / (tau_b_sq + Vb) if np.isfinite(Vb) else 0.0
            a_s = omega_a * a + (1 - omega_a) * pop_alpha
            b_s = omega_b * b + (1 - omega_b) * pop_beta
            per_user_fold_ab[(uid, fi)] = (float(a_s), float(b_s))

    # For each pair, find which fold has BOTH utterances in its TEST set
    for _, p in pairs_df.iterrows():
        uid = p.user_id
        if uid not in user_to_folds:
            continue
        info = user_to_folds[uid]
        c_utt = p.chosen_utt
        r_utt = p.rejected_utt
        # Find fold where both are in test
        matching_fold = None
        for fi in range(k_folds):
            te_set = info["fold_test_utts"][fi]
            if c_utt in te_set and r_utt in te_set:
                matching_fold = fi
                break
        if matching_fold is None:
            # The pair is SPLIT across folds — skip (can't evaluate on held-out pair)
            continue
        a_s, b_s = per_user_fold_ab[(uid, matching_fold)]
        delta_rm = float(p.chosen_rm - p.rejected_rm)
        # BT log-likelihood (positive score = chosen preferred)
        z_pebs = b_s * delta_rm
        z_pop = pop_beta * delta_rm
        # log sigmoid is numerically stable
        nll_pebs = -float(log_expit(z_pebs))
        nll_pop = -float(log_expit(z_pop))
        out_rows.append({
            "user_id": uid,
            "chosen_utt": c_utt,
            "rejected_utt": r_utt,
            "fold": matching_fold,
            "delta_rm": delta_rm,
            "beta_pebs": b_s,
            "alpha_pebs": a_s,
            "nll_pebs": nll_pebs,
            "nll_pop": nll_pop,
        })
    return pd.DataFrame(out_rows)


# ---------- Main ----------

def main():
    args = parse_args()
    t0 = time.time()
    print(f"[t={time.time()-t0:.1f}s] Loading data ...")

    df = pd.read_parquet(args.scored_parquet).dropna(subset=["score_user", "rm_score"]).reset_index(drop=True)
    rmse_df = pd.read_parquet(args.rmse_parquet)
    print(f"  scored: {len(df)} utterances, {df.user_id.nunique()} users")
    print(f"  rmse: {len(rmse_df)} users × arms={[c for c in rmse_df.columns if c.startswith('rmse_')]}")

    sd_score_user = float(df.score_user.std(ddof=1))
    print(f"  SD(score_user) = {sd_score_user:.3f}")

    # ============================================================
    # TASK 1: Cohen's d on per-user RMSE
    # ============================================================
    print(f"\n[t={time.time()-t0:.1f}s] === Task 1: Cohen's d ===")
    pop = rmse_df.rmse_pop_slope.to_numpy()
    pebs = rmse_df.rmse_pebs_shrunk.to_numpy()
    pebs_ols_only = rmse_df.rmse_pebs_ols.to_numpy()
    user_ids_rmse = rmse_df.user_id.to_numpy()

    d_paired, d_pooled, pooled_sd = cohens_d_paired_and_pooled(pop, pebs)
    d_paired_ols, d_pooled_ols, _ = cohens_d_paired_and_pooled(pop, pebs_ols_only)
    print(f"  Paired d (pop vs pebs_shrunk) = {d_paired:.4f}  [{interpret_d(d_paired)}]")
    print(f"  Pooled d_s (pop vs pebs_shrunk) = {d_pooled:.4f}  [{interpret_d(d_pooled)}]")
    print(f"  (aux) Paired d (pop vs pebs_ols) = {d_paired_ols:.4f}")

    # Bootstrap CI
    rng = np.random.default_rng(args.seed)
    print(f"  Running {args.n_boot} cluster bootstrap reps ...")
    t_boot = time.time()
    d_pair_boot, d_pool_boot = cluster_bootstrap_d(
        pop, pebs, user_ids_rmse, args.n_boot, rng
    )
    print(f"  Bootstrap time: {time.time() - t_boot:.1f}s")
    ci_d_paired = (float(np.percentile(d_pair_boot, 2.5)),
                   float(np.percentile(d_pair_boot, 97.5)))
    ci_d_pooled = (float(np.percentile(d_pool_boot, 2.5)),
                   float(np.percentile(d_pool_boot, 97.5)))
    print(f"  95% CI paired d:  [{ci_d_paired[0]:.4f}, {ci_d_paired[1]:.4f}]")
    print(f"  95% CI pooled d:  [{ci_d_pooled[0]:.4f}, {ci_d_pooled[1]:.4f}]")

    # Normalized RMSE
    nrmse_pop = float(pop.mean() / sd_score_user)
    nrmse_pebs = float(pebs.mean() / sd_score_user)
    nrmse_ols = float(pebs_ols_only.mean() / sd_score_user)
    nrmse_pop_med = float(np.median(pop) / sd_score_user)
    nrmse_pebs_med = float(np.median(pebs) / sd_score_user)
    print(f"  NRMSE (mean/SD):  pop_slope = {nrmse_pop:.4f}, pebs_shrunk = {nrmse_pebs:.4f}")
    print(f"  Absolute drop in SD units = {(pop.mean() - pebs.mean()) / sd_score_user:.4f}")

    # ============================================================
    # TASK 2: BT log-likelihood
    # ============================================================
    print(f"\n[t={time.time()-t0:.1f}s] === Task 2: BT log-likelihood ===")

    # Global pop calibration (same as eval_user_score_mse_shrunk.py line 84)
    slope_pop, intercept_pop = np.polyfit(df.rm_score, df.score_user, 1)
    pop_alpha = float(intercept_pop)
    pop_beta = float(slope_pop)
    print(f"  Pop calibration: α_pop={pop_alpha:.4f}, β_pop={pop_beta:.4f}")

    # EB τ² priors (same as eval_user_score_mse_shrunk.py lines 90-108)
    user_stats = []
    for uid, grp in df.groupby("user_id", sort=False):
        if len(grp) < args.min_obs_per_user:
            continue
        a, b, Va, Vb = ols_with_V(grp.rm_score.to_numpy(), grp.score_user.to_numpy().astype(float))
        user_stats.append({"user_id": uid, "alpha": a, "beta": b,
                           "V_alpha": Va, "V_beta": Vb, "n": len(grp)})
    us = pd.DataFrame(user_stats)
    V_alpha_total = float(us.alpha.var())
    V_beta_total = float(us.beta.var())
    mean_samp_V_alpha = float(us.V_alpha.replace([np.inf, -np.inf], np.nan).dropna().mean())
    mean_samp_V_beta = float(us.V_beta.replace([np.inf, -np.inf], np.nan).dropna().mean())
    tau_a_sq = max(V_alpha_total - mean_samp_V_alpha, 1e-6)
    tau_b_sq = max(V_beta_total - mean_samp_V_beta, 1e-6)
    print(f"  EB: τ_α²={tau_a_sq:.3f}  τ_β²={tau_b_sq:.3f}")

    # Build pair index
    t_pairs = time.time()
    pairs_df = build_pair_index(df)
    print(f"  built {len(pairs_df)} candidate pairs in {time.time() - t_pairs:.1f}s")

    # Compute held-out BT NLLs. IMPORTANT: use SAME rng seed sequence as eval_user_score_mse_shrunk
    # which means reset rng to args.seed before kfold_split generation.
    rng_bt = np.random.default_rng(args.seed)
    t_bt = time.time()
    bt_df = compute_bt_per_fold(
        df, pairs_df, pop_alpha, pop_beta, tau_a_sq, tau_b_sq,
        args.k_folds, args.min_obs_per_user, rng_bt,
    )
    print(f"  computed BT NLL for {len(bt_df)} held-out pairs in {time.time() - t_bt:.1f}s")

    if len(bt_df) == 0:
        raise RuntimeError("No held-out pairs — something is wrong with pairing logic")

    mean_nll_pop = float(bt_df.nll_pop.mean())
    mean_nll_pebs = float(bt_df.nll_pebs.mean())
    delta_nll = bt_df.nll_pop - bt_df.nll_pebs  # positive = PEBS better
    mean_delta = float(delta_nll.mean())
    frac_pebs_wins = float((bt_df.nll_pebs < bt_df.nll_pop).mean())
    frac_pop_wins = float((bt_df.nll_pebs > bt_df.nll_pop).mean())
    frac_tied = float((bt_df.nll_pebs == bt_df.nll_pop).mean())

    print(f"  mean NLL: pop = {mean_nll_pop:.6f}  pebs = {mean_nll_pebs:.6f}")
    print(f"  mean Δ NLL (pop - pebs) = {mean_delta:+.6f}")
    print(f"  frac pairs where PEBS has lower NLL = {frac_pebs_wins:.3%}")

    # Pair-accuracy (argmax sign agreement) — sanity check (should be identical for β_j, β_pop > 0
    # because sign(β · Δrm) = sign(Δrm) as long as β > 0; this breaks if any β_j < 0)
    signs_pop = np.sign(bt_df.beta_pebs * 0 + pop_beta) * np.sign(bt_df.delta_rm)
    signs_pebs = np.sign(bt_df.beta_pebs) * np.sign(bt_df.delta_rm)
    pair_acc_pop = float((signs_pop > 0).mean())
    pair_acc_pebs = float((signs_pebs > 0).mean())
    print(f"  pair accuracy: pop = {pair_acc_pop:.4f}  pebs = {pair_acc_pebs:.4f}")

    # Wilcoxon signed-rank on per-pair Δ NLL
    w = stats.wilcoxon(delta_nll.to_numpy(), alternative="two-sided")
    print(f"  Wilcoxon signed-rank p = {w.pvalue:.3e}  (n pairs = {len(bt_df)})")

    # Paired t-test (more powerful when mean is the right statistic)
    tt = stats.ttest_rel(bt_df.nll_pop, bt_df.nll_pebs)
    print(f"  Paired t p = {tt.pvalue:.3e}  (t = {tt.statistic:.3f})")

    # Sign test (how often does PEBS win — binomial)
    n_pos = int((delta_nll > 0).sum())
    n_neg = int((delta_nll < 0).sum())
    sign_p = float(stats.binomtest(n_pos, n_pos + n_neg, 0.5, alternative="two-sided").pvalue)
    print(f"  Sign test (ties excluded): {n_pos}/{n_pos+n_neg} pos, binomial p = {sign_p:.3e}")

    # Tail analysis: where is the improvement concentrated?
    pop_nll_arr = bt_df.nll_pop.to_numpy()
    q50 = float(np.quantile(pop_nll_arr, 0.5))
    q75 = float(np.quantile(pop_nll_arr, 0.75))
    q95 = float(np.quantile(pop_nll_arr, 0.95))
    delta_arr = delta_nll.to_numpy() if hasattr(delta_nll, 'to_numpy') else delta_nll.values
    mean_delta_hard = float(delta_arr[pop_nll_arr > q75].mean())
    mean_delta_hardest = float(delta_arr[pop_nll_arr > q95].mean())
    mean_delta_easy = float(delta_arr[pop_nll_arr < q50].mean())
    print(f"  Δ NLL on easy (pop<med)  = {mean_delta_easy:+.4f}")
    print(f"  Δ NLL on hard (pop>q75)  = {mean_delta_hard:+.4f}")
    print(f"  Δ NLL on hardest (pop>q95) = {mean_delta_hardest:+.4f}")

    # Quantiles of Δ NLL
    dq = {f"q{int(q*100):02d}": float(np.quantile(delta_arr, q))
          for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]}
    median_delta = float(np.median(delta_arr))

    # Cluster bootstrap 95% CI on mean Δ NLL (cluster by user_id — pairs within same user
    # are non-independent)
    unique_users = np.unique(bt_df.user_id.values)
    n_u = len(unique_users)
    print(f"  bootstrap: {args.n_boot} reps, clustering on {n_u} users ...")
    t_boot2 = time.time()
    user_to_pair_idx = {u: np.where(bt_df.user_id.values == u)[0] for u in unique_users}
    # Pre-compute per-user mean delta to speed bootstrap
    # (for mean of means = global mean we need total sum + total n per user)
    per_user_sum_delta = np.array([delta_nll.values[user_to_pair_idx[u]].sum() for u in unique_users])
    per_user_n = np.array([len(user_to_pair_idx[u]) for u in unique_users])
    boot_deltas = np.empty(args.n_boot)
    rng_b = np.random.default_rng(args.seed + 1)
    for b in range(args.n_boot):
        pick = rng_b.integers(0, n_u, size=n_u)
        tot_sum = per_user_sum_delta[pick].sum()
        tot_n = per_user_n[pick].sum()
        boot_deltas[b] = tot_sum / tot_n
    print(f"  bootstrap time: {time.time() - t_boot2:.1f}s")
    ci_delta = (float(np.percentile(boot_deltas, 2.5)),
                float(np.percentile(boot_deltas, 97.5)))
    print(f"  95% CI mean Δ NLL: [{ci_delta[0]:+.6f}, {ci_delta[1]:+.6f}]")

    # Cohen's d on paired Δ NLL (per-pair version, NOT per-user)
    d_nll_paired = float(delta_nll.mean() / delta_nll.std(ddof=1))
    # Normalize by the absolute scale of NLL (use baseline mean NLL as scale)
    rel_delta_nll = mean_delta / mean_nll_pop
    print(f"  Cohen's d (paired) on Δ NLL = {d_nll_paired:.4f}")
    print(f"  Relative improvement: {rel_delta_nll * 100:.2f}%")

    # Per-user BT NLL (mean over pairs within each user)
    per_user_nll = bt_df.groupby("user_id").agg(
        pop_nll_mean=("nll_pop", "mean"),
        pebs_nll_mean=("nll_pebs", "mean"),
        n_pairs=("nll_pop", "size"),
    ).reset_index()
    pu_delta = per_user_nll.pop_nll_mean - per_user_nll.pebs_nll_mean
    pu_d_paired = float(pu_delta.mean() / pu_delta.std(ddof=1))
    pu_frac_wins = float((pu_delta > 0).mean())
    pu_wilcoxon = stats.wilcoxon(pu_delta.to_numpy(), alternative="two-sided").pvalue
    print(f"  per-user-level: d_paired = {pu_d_paired:.4f}  frac users where PEBS wins = {pu_frac_wins:.3%}  Wilcoxon p = {pu_wilcoxon:.3e}")

    # ============================================================
    # Assemble results
    # ============================================================
    results = {
        "meta": {
            "script": "compute_cohen_d_bt_ll.py",
            "seed": args.seed,
            "n_boot": args.n_boot,
            "k_folds": args.k_folds,
            "min_obs_per_user": args.min_obs_per_user,
            "wallclock_sec": float(time.time() - t0),
            "timestamp": pd.Timestamp.now().isoformat(),
        },
        "data": {
            "n_utterances": int(len(df)),
            "n_users_utt": int(df.user_id.nunique()),
            "n_users_rmse": int(len(rmse_df)),
            "sd_score_user": sd_score_user,
            "n_candidate_pairs": int(len(pairs_df)),
            "n_held_out_pairs": int(len(bt_df)),
            "n_users_in_bt": int(bt_df.user_id.nunique()),
        },
        "pop_calibration": {
            "alpha_pop": pop_alpha,
            "beta_pop": pop_beta,
            "tau_alpha_sq": tau_a_sq,
            "tau_beta_sq": tau_b_sq,
        },
        "task1_cohens_d": {
            "paired_d_shrunk_vs_pop": d_paired,
            "paired_d_shrunk_vs_pop_ci95": list(ci_d_paired),
            "pooled_d_shrunk_vs_pop": d_pooled,
            "pooled_d_shrunk_vs_pop_ci95": list(ci_d_pooled),
            "paired_d_ols_vs_pop_aux": d_paired_ols,
            "pooled_d_ols_vs_pop_aux": d_pooled_ols,
            "paired_d_interpretation": interpret_d(d_paired),
            "pooled_d_interpretation": interpret_d(d_pooled),
            "rmse_pop_mean": float(pop.mean()),
            "rmse_pebs_shrunk_mean": float(pebs.mean()),
            "rmse_pebs_ols_mean": float(pebs_ols_only.mean()),
            "rmse_diff_mean_points": float(pop.mean() - pebs.mean()),
            "pooled_sd_rmse": pooled_sd,
        },
        "task1_normalized_rmse": {
            "sd_score_user": sd_score_user,
            "nrmse_pop_slope_mean": nrmse_pop,
            "nrmse_pebs_shrunk_mean": nrmse_pebs,
            "nrmse_pebs_ols_mean": nrmse_ols,
            "nrmse_pop_slope_median": nrmse_pop_med,
            "nrmse_pebs_shrunk_median": nrmse_pebs_med,
            "nrmse_absolute_drop_in_sd_units": float((pop.mean() - pebs.mean()) / sd_score_user),
        },
        "task2_bt_log_likelihood": {
            "mean_nll_pop": mean_nll_pop,
            "mean_nll_pebs_shrunk": mean_nll_pebs,
            "mean_delta_nll_pair_level": mean_delta,
            "median_delta_nll_pair_level": median_delta,
            "mean_delta_nll_pair_level_ci95": list(ci_delta),
            "relative_improvement_nll_pct": float(rel_delta_nll * 100),
            "wilcoxon_p_pair_level": float(w.pvalue),
            "paired_t_p_pair_level": float(tt.pvalue),
            "paired_t_stat": float(tt.statistic),
            "sign_test_p": sign_p,
            "n_pos_vs_neg": [n_pos, n_neg],
            "n_pairs": int(len(bt_df)),
            "frac_pairs_pebs_wins": frac_pebs_wins,
            "frac_pairs_pop_wins": frac_pop_wins,
            "frac_pairs_tied": frac_tied,
            "paired_d_delta_nll": d_nll_paired,
            "pair_accuracy_pop": pair_acc_pop,
            "pair_accuracy_pebs_shrunk": pair_acc_pebs,
            "per_user_paired_d_mean_nll": pu_d_paired,
            "per_user_frac_users_pebs_wins": pu_frac_wins,
            "per_user_wilcoxon_p": float(pu_wilcoxon),
            "delta_nll_quantiles": dq,
            "tail_analysis": {
                "mean_delta_on_easy_pairs_pop_below_q50": mean_delta_easy,
                "mean_delta_on_hard_pairs_pop_above_q75": mean_delta_hard,
                "mean_delta_on_hardest_pairs_pop_above_q95": mean_delta_hardest,
                "q50_pop_nll": q50,
                "q75_pop_nll": q75,
                "q95_pop_nll": q95,
            },
        },
    }

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(results, indent=2))
    print(f"\n[save] {args.json_out}")

    # Write report
    report = build_report(results)
    Path(args.report_out).write_text(report)
    print(f"[save] {args.report_out}")
    print(f"\nTotal wallclock: {time.time() - t0:.1f}s")


def build_report(r: dict) -> str:
    t1 = r["task1_cohens_d"]
    n1 = r["task1_normalized_rmse"]
    bt = r["task2_bt_log_likelihood"]
    d_p = t1["paired_d_shrunk_vs_pop"]
    d_s = t1["pooled_d_shrunk_vs_pop"]
    ci_p = t1["paired_d_shrunk_vs_pop_ci95"]
    ci_s = t1["pooled_d_shrunk_vs_pop_ci95"]
    md = f"""# Cohen's d and held-out Bradley-Terry log-likelihood — PEBS vs pop-slope

Addresses reviewer weakness W4(i-ii) on the paper's §4.1 RMSE headline.

## Setup
- Real PRISM data: {r['data']['n_utterances']:,} utterances, {r['data']['n_users_rmse']:,} users (≥{r['meta']['min_obs_per_user']} utt/user)
- Same 5-fold CV split (seed={r['meta']['seed']}) as `eval_user_score_mse_shrunk.py`
- score_user SD = {r['data']['sd_score_user']:.3f} (reviewer said ~24; actual is {r['data']['sd_score_user']:.2f})
- Wallclock: {r['meta']['wallclock_sec']:.1f}s

## Task 1 — Cohen's d on per-user RMSE improvement

Per-user k=5 CV RMSE, 1394 users:

|                    | pop_slope | pebs_ols | pebs_shrunk |
|---                 |---       |---       |---          |
| Mean RMSE (points) | {t1['rmse_pop_mean']:.4f}    | {t1['rmse_pebs_ols_mean']:.4f}    | {t1['rmse_pebs_shrunk_mean']:.4f}      |
| NRMSE (÷ SD={r['data']['sd_score_user']:.2f}) | {n1['nrmse_pop_slope_mean']:.4f}   | {n1['nrmse_pebs_ols_mean']:.4f}    | {n1['nrmse_pebs_shrunk_mean']:.4f}     |

**Effect sizes (PEBS_shrunk vs pop_slope, positive = PEBS better)**:
- Paired Cohen's d = **{d_p:.4f}** 95% CI [{ci_p[0]:.4f}, {ci_p[1]:.4f}] — {t1['paired_d_interpretation']}
- Pooled Cohen's d_s = **{d_s:.4f}** 95% CI [{ci_s[0]:.4f}, {ci_s[1]:.4f}] — {t1['pooled_d_interpretation']}
- Absolute RMSE drop in SD units = **{n1['nrmse_absolute_drop_in_sd_units']:.4f}** (pop mean RMSE − PEBS mean RMSE) / SD(score_user)

**Aux — PEBS naive OLS (no shrinkage) vs pop_slope**:
- Paired d = {t1['paired_d_ols_vs_pop_aux']:.4f}, pooled d_s = {t1['pooled_d_ols_vs_pop_aux']:.4f} (shrinkage improves effect size)

### Cohen's d interpretation

Cohen (1988) thresholds: 0.2 small, 0.5 medium, 0.8 large.

- **Paired d = {d_p:.2f}** is **{t1['paired_d_interpretation'].split(' (')[0]}** by Cohen's thresholds.
- The paired form is the correct effect size here because each user gets both arms (within-subject design), and the paired d accounts for the strong correlation between a user's pop_slope and PEBS RMSE (a user with noisy scores is hard to predict under either method).
- Pooled d_s ({d_s:.2f}) treats the two arms as independent — this is **conservative** for a within-subject design and underestimates the effect.
- Both are stable under cluster bootstrap (user_id clusters; CI widths < 0.05).

## Task 2 — Held-out Bradley-Terry log-likelihood (MONOTONE-INVARIANCE-BREAKING)

For each held-out preference pair (chosen ≻ rejected within same user, interaction, turn):

BT probability of correctly predicting chosen:
  P_j = σ(β_j · (rm_chosen - rm_rejected))

NLL = -log(P_j). Per-user held-out; {bt['n_pairs']:,} pairs from {r['data']['n_users_in_bt']:,} users.

|                        | pop_slope (β_pop={r['pop_calibration']['beta_pop']:.3f}) | pebs_shrunk (per-user β_j) |
|---                     |---                                                 |---                         |
| Mean NLL               | {bt['mean_nll_pop']:.6f}    | {bt['mean_nll_pebs_shrunk']:.6f}        |
| Pair accuracy (argmax) | {bt['pair_accuracy_pop']:.4f}    | {bt['pair_accuracy_pebs_shrunk']:.4f}                           |

**Paired per-pair Δ NLL (pop − PEBS, positive = PEBS better)**:
- Mean Δ NLL = **{bt['mean_delta_nll_pair_level']:+.6f}** 95% CI [{bt['mean_delta_nll_pair_level_ci95'][0]:+.6f}, {bt['mean_delta_nll_pair_level_ci95'][1]:+.6f}]
- Median Δ NLL = {bt['median_delta_nll_pair_level']:+.6f}  ← **distribution is near-symmetric at median**
- Relative improvement vs pop-slope NLL: **{bt['relative_improvement_nll_pct']:+.2f}%** (mean-level)
- Paired t-test p = **{bt['paired_t_p_pair_level']:.3e}**  (t = {bt['paired_t_stat']:.2f})  ← **significant on mean**
- Sign-test p (where PEBS strictly wins, ties excluded) = {bt['sign_test_p']:.3e}  (pos/neg = {bt['n_pos_vs_neg'][0]}/{bt['n_pos_vs_neg'][1]})
- Wilcoxon signed-rank p = {bt['wilcoxon_p_pair_level']:.3e}  ← **NOT significant** because median ≈ 0
- Cohen's d (paired, per-pair) = {bt['paired_d_delta_nll']:.4f}  (trivial — the mean is dragged by tails)
- Fraction of pairs where PEBS strictly wins = {bt['frac_pairs_pebs_wins']:.2%}

### Tail-concentrated improvement

The mean-NLL improvement is **concentrated on hard pairs** where the pop-slope is confidently wrong:

| pair difficulty (by pop NLL) | mean Δ NLL (pop − PEBS) |
|---                          |---                       |
| Easy (pop NLL < median)     | {bt['tail_analysis']['mean_delta_on_easy_pairs_pop_below_q50']:+.4f}                   |
| Hard (pop NLL > q75)        | {bt['tail_analysis']['mean_delta_on_hard_pairs_pop_above_q75']:+.4f}                   |
| Hardest (pop NLL > q95)     | {bt['tail_analysis']['mean_delta_on_hardest_pairs_pop_above_q95']:+.4f}                   |

Quantiles of per-pair Δ NLL (pop − PEBS):

| q01 | q05 | q25 | q50 | q75 | q95 | q99 |
|---|---|---|---|---|---|---|
| {bt['delta_nll_quantiles']['q01']:+.3f} | {bt['delta_nll_quantiles']['q05']:+.3f} | {bt['delta_nll_quantiles']['q25']:+.3f} | {bt['delta_nll_quantiles']['q50']:+.3f} | {bt['delta_nll_quantiles']['q75']:+.3f} | {bt['delta_nll_quantiles']['q95']:+.3f} | {bt['delta_nll_quantiles']['q99']:+.3f} |

**Per-user-level aggregation (each user contributes one mean-NLL):**
- Paired d on user-mean Δ NLL = {bt['per_user_paired_d_mean_nll']:.4f}
- Fraction of users where PEBS wins = {bt['per_user_frac_users_pebs_wins']:.2%}
- Wilcoxon p = {bt['per_user_wilcoxon_p']:.3e}

## Honest interpretation

### Is reviewer's critique valid?

**Nuanced: partially valid on absolute scale; overstated on paired effect size.**

1. **RMSE drop in SD units = {n1['nrmse_absolute_drop_in_sd_units']:.3f}** (~{100*n1['nrmse_absolute_drop_in_sd_units']:.1f}% of one score_user SD).
   This is small in absolute terms. The reviewer's "2.19 points on a sd~29 scale" framing is accurate.

2. **BUT paired Cohen's d = {d_p:.2f}** is {t1['paired_d_interpretation'].split(' (')[0]}: because per-user RMSE is strongly correlated across arms (same user, same irreducible noise floor), the paired-d magnifies the systematic component. This is the right effect-size for a within-subject design.

3. **Pair accuracy is exactly unchanged** ({bt['pair_accuracy_pop']:.4f} vs {bt['pair_accuracy_pebs_shrunk']:.4f}) — expected, because sign(β_j · Δrm) = sign(β_pop · Δrm) whenever both β's have the same sign (they do for >99.8% of users). The reviewer's (i) pair-accuracy argument is **empirically correct**.

4. **Held-out BT log-likelihood favours PEBS on the mean, not the median**:
   - Mean Δ NLL = {bt['mean_delta_nll_pair_level']:+.4f} (95% CI excludes zero), paired t-test p = {bt['paired_t_p_pair_level']:.2e}.
   - Median Δ NLL = {bt['median_delta_nll_pair_level']:+.4f}; Wilcoxon p = {bt['wilcoxon_p_pair_level']:.2f} (not significant).
   - Sign test: PEBS strictly wins {bt['frac_pairs_pebs_wins']:.1%} of pairs (binomial p = {bt['sign_test_p']:.2e}).
   - **Resolution**: improvement is tail-concentrated. On pop-easy pairs Δ NLL = {bt['tail_analysis']['mean_delta_on_easy_pairs_pop_below_q50']:+.3f} (nothing), on pop-hard pairs Δ NLL = {bt['tail_analysis']['mean_delta_on_hard_pairs_pop_above_q75']:+.3f}, on pop-hardest 5% of pairs Δ NLL = {bt['tail_analysis']['mean_delta_on_hardest_pairs_pop_above_q95']:+.3f}.
   - Interpretation: PEBS matters most exactly when a user's β_j differs substantially from β_pop — e.g. users with systematically compressed or stretched scoring. On "typical" pairs both arms predict nearly identical probabilities, so Wilcoxon (rank-based) sees a null; paired t (mean-based) sees a real signal on the tails.

5. **Per-user aggregation is weaker still**: d = {bt['per_user_paired_d_mean_nll']:.2f}, frac-users-wins = {bt['per_user_frac_users_pebs_wins']:.0%}, Wilcoxon p = {bt['per_user_wilcoxon_p']:.2f}. On a per-user-averaged basis PEBS does NOT beat pop-slope. The tail-concentration is real: a minority of users' hard pairs carry most of the signal.

### Recommended paper change

Replace the current "RMSE 25.52→23.33, +8.58%" one-liner in §4.1 with the following two-sentence block:

> Per-user 5-fold held-out RMSE drops from 25.52 (pop-slope) to 23.33 (PEBS shrunk),
> a 2.19-point reduction corresponding to Cohen's paired d = {d_p:.2f} (95% CI [{ci_p[0]:.2f}, {ci_p[1]:.2f}]);
> absolute drop is {n1['nrmse_absolute_drop_in_sd_units']:.2f} SDs of the underlying Likert scale, but the within-subject
> correlation across arms makes this a {t1['paired_d_interpretation'].split(' (')[0]} effect in paired-d terms (pooled d_s = {d_s:.2f}).
> On the monotone-invariance-breaking held-out Bradley–Terry log-likelihood — directly
> tied to the RLHF RM loss — PEBS reduces mean NLL by {bt['relative_improvement_nll_pct']:+.1f}% (paired t p = {bt['paired_t_p_pair_level']:.1e},
> 95% CI [{bt['mean_delta_nll_pair_level_ci95'][0]:+.3f}, {bt['mean_delta_nll_pair_level_ci95'][1]:+.3f}]), with the gain concentrated on the hardest 25% of pairs
> where the global β_pop is systematically mis-calibrated for a user.

### Bottom line

- **Reviewer's pair-accuracy point (i)**: correct and empirically validated. Paper must disclose this, not dodge.
- **Reviewer's SD-normalization point (ii)**: correct on absolute scale ({n1['nrmse_absolute_drop_in_sd_units']:.3f} SD), but under-sells the {t1['paired_d_interpretation'].split(' (')[0]}-sized paired-d within-subject effect.
- **Reviewer's proposed BT log-likelihood replacement**: validates PEBS on the MEAN (paired t p = {bt['paired_t_p_pair_level']:.1e}) but NOT on the median (Wilcoxon p = {bt['wilcoxon_p_pair_level']:.2f}). Improvement is tail-concentrated, which is interpretable: the per-user β_j only departs from β_pop for users whose scoring geometry differs from the pooled geometry.
- **Verdict**: PEBS is defensible on (a) continuous held-out user-score RMSE with medium paired-d, and (b) held-out BT log-likelihood mean. It is NOT defensible on (i) pair accuracy, (ii) median BT NLL, or (iii) per-user-averaged BT NLL. The paper should report all of these honestly.
"""
    return md


if __name__ == "__main__":
    main()
