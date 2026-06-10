"""EBPO x 3-backbone x 2-corpus head-to-head matrix.

Scoop #1 defense: EBPO (arxiv:2602.05165) is the algebraically-closest neighbor
to PEBS — both use the same omega = tau^2/(tau^2+V) shrinkage identity on
hierarchical group means. The PRISM Qwen-7B head-to-head showed PEBS > EBPO by
+0.48pp RMSE; the PRISM LoRe-slice pair-acc analysis showed EBPO > PEBS by
+0.66pp because chosen-vs-rejected margins cancel the per-user intercept.

This script expands to the full 3 backbone x 2 corpus matrix with BOTH metrics:

  Backbones: {qwen7b, skywork27b, llama32_3b}
  Corpora:   {PRISM, PluriHarms}
  Methods:   {pop, EBPO, PEBS}
  Metrics:   {RMSE gain %, pair-accuracy}

Metric details
--------------
- RMSE gain %: held-out CV per user
    - PRISM: leave-one-conversation-out (matches the PRISM LOCO protocol)
    - PluriHarms: 5-fold random split on Question_Index (matches eval script)
  Relative gain vs pop_only: (RMSE_pop - RMSE_method) / RMSE_pop * 100
  Cluster-bootstrap over user_id, n_boot=2000, seed=20260420.

- Pair-accuracy: only defined where preference pairs exist.
    - PRISM: within (user, conversation) expand chosen x rejected pairs from
      the LoRe-style slice (users with >=6 conversations); 50/50 train/test
      conversation split. Pair-acc = mean I[margin(chosen)-margin(rejected)>0]
      under each method's per-user calibrator.
    - PluriHarms: NO PAIRS (ratings are absolute harm scores, not preference
      pairs). We document and skip pair-acc for PluriHarms cells.

Method definitions (consistent across cells)
--------------------------------------------
- pop_only: alpha_pop + beta_pop * x, fit once on train union.
- EBPO (per neighbor_head_to_head.py): per-user INTERCEPT shrinkage
    S = (sigma2/G)/(sigma2/G + tau2), combined with population slope.
- PEBS: per-user (alpha_j, beta_j) with EB shrinkage
    omega_a = tau2_a / (tau2_a + SE(alpha_j)^2)
    omega_b = tau2_b / (tau2_b + SE(beta_j)^2)
  fit via MoM.

Outputs
-------
  results/track1_ebpo_3x2/summary.json
  results/track1_ebpo_3x2/per_cell.parquet

Wall-clock budget: 60 min on CPU.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
T1 = ROOT.parent / "1_Causal_RLHF"

OUT = ROOT / "results/track1_ebpo_3x2"

N_BOOT_DEFAULT = 2000
RNG_DEFAULT = 20260420
MIN_CONV_PER_USER = 2
MIN_UTT_PER_USER = 10
MIN_CONV_PAIR_SLICE = 6  # LoRe-style slice for pair-acc

# Corpus + backbone configuration
PRISM_QWEN = T1 / "data/prism_rm_scored.parquet"
PRISM_SKYWORK = T1 / "data/prism_skywork_scored.parquet"
PRISM_LLAMA = T1 / "data/prism_llama32_3b_scored.parquet"
PLURI_LONG = T1 / "data/pluriharms_long.parquet"
PLURI_SKYWORK = T1 / "data/pluriharms_skywork_scored.parquet"
PLURI_LLAMA = T1 / "data/pluriharms_llama32_3b_scored.parquet"

CELLS = [
    ("qwen7b", "prism"),
    ("skywork27b", "prism"),
    ("llama32_3b", "prism"),
    ("qwen7b", "pluriharms"),
    ("skywork27b", "pluriharms"),
    ("llama32_3b", "pluriharms"),
]


# ======================================================================
# Shared fits
# ======================================================================

def fit_ols_with_se(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.inf, np.inf
    xm = float(np.mean(x))
    ssx = float(np.sum((x - xm) ** 2))
    if ssx < 1e-10:
        return float(np.mean(y)), 0.0, np.inf, np.inf
    beta = float(np.sum((x - xm) * (y - np.mean(y))) / ssx)
    alpha = float(np.mean(y) - beta * xm)
    resid = y - (alpha + beta * x)
    mse = float(np.sum(resid ** 2) / (n - 2))
    se_alpha = float(np.sqrt(mse * (1.0 / n + xm ** 2 / ssx)))
    se_beta = float(np.sqrt(mse / ssx))
    return alpha, beta, se_alpha, se_beta


def fit_population_and_taus(df: pd.DataFrame, x_col: str, y_col: str):
    """Global pop OLS + per-user MoM tau^2_a, tau^2_b + EBPO sigma2, tau2."""
    x = df[x_col].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    alpha_pop, beta_pop = float(intercept), float(slope)

    # Per-user OLS
    alphas, betas, sas, sbs = [], [], [], []
    user_fits = {}
    for uid, g in df.groupby("user_id"):
        a, b, sa, sb = fit_ols_with_se(g[x_col].to_numpy(dtype=np.float64),
                                       g[y_col].to_numpy(dtype=np.float64))
        user_fits[uid] = (a, b, sa, sb)
        if np.isfinite(a) and np.isfinite(sa):
            alphas.append(a); sas.append(sa)
        if np.isfinite(b) and np.isfinite(sb):
            betas.append(b); sbs.append(sb)
    alphas = np.array(alphas); betas = np.array(betas)
    sas = np.array(sas); sbs = np.array(sbs)
    tau2_a = max(0.0, float(np.var(alphas, ddof=1) - np.mean(sas ** 2))) if len(alphas) > 1 else 0.0
    tau2_b = max(0.0, float(np.var(betas, ddof=1) - np.mean(sbs ** 2))) if len(betas) > 1 else 0.0

    # EBPO sigma2 / tau2 on residual-mean scale (residual-mean convention)
    resid = y - (alpha_pop + beta_pop * x)
    tmp = df.assign(_r=resid)
    user_mean_r = tmp.groupby("user_id")["_r"].mean().to_numpy()
    user_var_r = tmp.groupby("user_id")["_r"].var(ddof=1).to_numpy()
    # sigma2 = mean within-user residual variance (excluding singletons with NaN var)
    sigma2 = float(np.nanmean(user_var_r)) if np.any(np.isfinite(user_var_r)) else float(np.var(resid, ddof=1))
    tau2_eb = max(0.0, float(np.var(user_mean_r, ddof=1)))

    return dict(alpha_pop=alpha_pop, beta_pop=beta_pop,
                tau2_a=tau2_a, tau2_b=tau2_b,
                sigma2_eb=sigma2, tau2_eb=tau2_eb,
                user_fits=user_fits)


# ======================================================================
# Predictors
# ======================================================================

def predict_pop(x_te, fit):
    return fit["alpha_pop"] + fit["beta_pop"] * x_te


def predict_pebs(x_tr, y_tr, x_te, fit):
    a, b, sa, sb = fit_ols_with_se(x_tr, y_tr)
    if not np.isfinite(a):
        return predict_pop(x_te, fit)
    w_a = fit["tau2_a"] / (fit["tau2_a"] + sa ** 2 + 1e-12) if np.isfinite(sa) else 0.0
    w_b = fit["tau2_b"] / (fit["tau2_b"] + sb ** 2 + 1e-12) if np.isfinite(sb) else 0.0
    a_s = w_a * a + (1 - w_a) * fit["alpha_pop"]
    b_s = w_b * b + (1 - w_b) * fit["beta_pop"]
    return a_s + b_s * x_te


def predict_ebpo(x_tr, y_tr, x_te, fit):
    G = len(y_tr)
    if G == 0:
        return predict_pop(x_te, fit)
    resid_tr = y_tr - (fit["alpha_pop"] + fit["beta_pop"] * x_tr)
    mu_group = float(np.mean(resid_tr))
    S = (fit["sigma2_eb"] / G) / (fit["sigma2_eb"] / G + fit["tau2_eb"] + 1e-12)
    mu_shrunk = (1 - S) * mu_group + S * 0.0
    return fit["alpha_pop"] + fit["beta_pop"] * x_te + mu_shrunk


# ======================================================================
# Corpus loaders
# ======================================================================

def load_prism_cell(backbone: str) -> pd.DataFrame:
    qwen = pd.read_parquet(PRISM_QWEN)[["utterance_id", "user_id",
                                         "conversation_id", "score_user",
                                         "if_chosen", "rm_score"]]
    if backbone == "qwen7b":
        return qwen.rename(columns={"rm_score": "x"}).dropna(
            subset=["x", "score_user", "conversation_id"]).assign(
            y=lambda d: d["score_user"].astype(float))[
            ["user_id", "conversation_id", "x", "y", "if_chosen"]]
    elif backbone == "skywork27b":
        sky = pd.read_parquet(PRISM_SKYWORK)[["utterance_id", "skywork_score"]]
        m = qwen.merge(sky, on="utterance_id", how="inner")
        return m.rename(columns={"skywork_score": "x"}).dropna(
            subset=["x", "score_user", "conversation_id"]).assign(
            y=lambda d: d["score_user"].astype(float))[
            ["user_id", "conversation_id", "x", "y", "if_chosen"]]
    elif backbone == "llama32_3b":
        lla = pd.read_parquet(PRISM_LLAMA)[["utterance_id", "llama32_3b_score"]]
        m = qwen.merge(lla, on="utterance_id", how="inner")
        return m.rename(columns={"llama32_3b_score": "x"}).dropna(
            subset=["x", "score_user", "conversation_id"]).assign(
            y=lambda d: d["score_user"].astype(float))[
            ["user_id", "conversation_id", "x", "y", "if_chosen"]]
    raise ValueError(backbone)


def load_pluri_cell(backbone: str) -> pd.DataFrame:
    long = pd.read_parquet(PLURI_LONG)  # user_id, Question_Index, rating, Harm_Level
    if backbone == "qwen7b":
        out = long.rename(columns={"Harm_Level": "x", "rating": "y"}).copy()
    elif backbone == "skywork27b":
        sky = pd.read_parquet(PLURI_SKYWORK)[["Question_Index", "skywork_score"]]
        out = long.merge(sky, on="Question_Index")
        out = out.rename(columns={"skywork_score": "x", "rating": "y"})
    elif backbone == "llama32_3b":
        lla = pd.read_parquet(PLURI_LLAMA)[["Question_Index", "llama32_3b_score"]]
        out = long.merge(lla, on="Question_Index")
        out = out.rename(columns={"llama32_3b_score": "x", "rating": "y"})
    else:
        raise ValueError(backbone)
    out = out.dropna(subset=["x", "y"]).copy()
    out["user_id"] = out["user_id"].astype(str)
    out["x"] = out["x"].astype(float)
    out["y"] = out["y"].astype(float)
    # Rename Question_Index -> fold_key for uniform downstream handling
    out["fold_key"] = out["Question_Index"]
    return out


# ======================================================================
# RMSE gain CV (per cell)
# ======================================================================

def run_rmse_cv_prism(df: pd.DataFrame, fit: dict):
    """Leave-one-conversation-out per user; return per-user RMSE table."""
    conv_count = df.groupby("user_id")["conversation_id"].nunique()
    utt_count = df.groupby("user_id").size()
    keep = conv_count[(conv_count >= MIN_CONV_PER_USER) &
                      (utt_count >= MIN_UTT_PER_USER)].index
    df = df[df["user_id"].isin(keep)].reset_index(drop=True)

    per_user = []
    for uid, g in df.groupby("user_id"):
        conv_ids = g["conversation_id"].unique().tolist()
        sq_pop, sq_ebpo, sq_pebs = [], [], []
        n_used = 0
        for hc in conv_ids:
            tr = g[g["conversation_id"] != hc]
            te = g[g["conversation_id"] == hc]
            if len(tr) < 3 or len(te) < 1:
                continue
            x_tr = tr["x"].to_numpy(dtype=np.float64)
            y_tr = tr["y"].to_numpy(dtype=np.float64)
            x_te = te["x"].to_numpy(dtype=np.float64)
            y_te = te["y"].to_numpy(dtype=np.float64)
            sq_pop.extend(((predict_pop(x_te, fit) - y_te) ** 2).tolist())
            sq_ebpo.extend(((predict_ebpo(x_tr, y_tr, x_te, fit) - y_te) ** 2).tolist())
            sq_pebs.extend(((predict_pebs(x_tr, y_tr, x_te, fit) - y_te) ** 2).tolist())
            n_used += len(y_te)
        if n_used < MIN_UTT_PER_USER:
            continue
        per_user.append({
            "user_id": uid,
            "n_utt": n_used,
            "rmse_pop": float(np.sqrt(np.mean(sq_pop))),
            "rmse_ebpo": float(np.sqrt(np.mean(sq_ebpo))),
            "rmse_pebs": float(np.sqrt(np.mean(sq_pebs))),
        })
    return pd.DataFrame(per_user)


def run_rmse_cv_pluri(df: pd.DataFrame, fit: dict, k_folds: int = 5,
                      seed: int = RNG_DEFAULT):
    """5-fold random split on Question_Index within each user."""
    rng = np.random.default_rng(seed)
    per_user = []
    user_count = df.groupby("user_id").size()
    keep = user_count[user_count >= MIN_UTT_PER_USER].index
    df = df[df["user_id"].isin(keep)].reset_index(drop=True)
    for uid, g in df.groupby("user_id"):
        n = len(g)
        if n < MIN_UTT_PER_USER:
            continue
        idx = rng.permutation(n)
        folds = np.array_split(idx, k_folds)
        sq_pop, sq_ebpo, sq_pebs = [], [], []
        for f in range(k_folds):
            test_pos = folds[f]
            train_pos = np.concatenate([folds[j] for j in range(k_folds) if j != f])
            if len(train_pos) < 3 or len(test_pos) < 1:
                continue
            gx = g["x"].to_numpy(dtype=np.float64)
            gy = g["y"].to_numpy(dtype=np.float64)
            x_tr, y_tr = gx[train_pos], gy[train_pos]
            x_te, y_te = gx[test_pos], gy[test_pos]
            sq_pop.extend(((predict_pop(x_te, fit) - y_te) ** 2).tolist())
            sq_ebpo.extend(((predict_ebpo(x_tr, y_tr, x_te, fit) - y_te) ** 2).tolist())
            sq_pebs.extend(((predict_pebs(x_tr, y_tr, x_te, fit) - y_te) ** 2).tolist())
        per_user.append({
            "user_id": uid,
            "n_utt": int(n),
            "rmse_pop": float(np.sqrt(np.mean(sq_pop))) if sq_pop else np.nan,
            "rmse_ebpo": float(np.sqrt(np.mean(sq_ebpo))) if sq_ebpo else np.nan,
            "rmse_pebs": float(np.sqrt(np.mean(sq_pebs))) if sq_pebs else np.nan,
        })
    return pd.DataFrame(per_user)


def cluster_boot_ci(vals: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(vals)
    if n == 0:
        return dict(mean=np.nan, ci95=[np.nan, np.nan], se=np.nan)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(vals[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(mean=float(vals.mean()), ci95=[float(lo), float(hi)],
                se=float(np.std(boots)))


def paired_wilcoxon(a: np.ndarray, b: np.ndarray):
    """Paired Wilcoxon: does a > b per user? Return two-sided p + mean delta."""
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]; b = b[mask]
    if len(a) < 2:
        return dict(p=np.nan, delta=np.nan, n=int(len(a)))
    diff = a - b
    nz = diff != 0
    if nz.sum() < 1:
        return dict(p=1.0, delta=float(diff.mean()), n=int(len(a)))
    try:
        stat, p = stats.wilcoxon(a[nz], b[nz], zero_method="wilcox",
                                  alternative="two-sided", method="approx")
    except Exception:
        p = np.nan
    return dict(p=float(p), delta=float(diff.mean()), n=int(len(a)))


# ======================================================================
# Pair-acc slice (PRISM only, LoRe-style)
# ======================================================================

def build_prism_pair_slice(df: pd.DataFrame):
    """Filter PRISM to users with >= MIN_CONV_PAIR_SLICE conversations,
    expand chosen x rejected pairs within each (user, conv)."""
    # df has columns: user_id, conversation_id, x, y, if_chosen
    n_conv = df.groupby("user_id")["conversation_id"].nunique()
    keep = n_conv[n_conv >= MIN_CONV_PAIR_SLICE].index
    df = df[df["user_id"].isin(keep)].reset_index(drop=True)
    pairs = []
    for (u, c), g in df.groupby(["user_id", "conversation_id"], sort=False):
        chosen = g[g["if_chosen"] == True]
        rejected = g[g["if_chosen"] == False]
        if len(chosen) == 0 or len(rejected) == 0:
            continue
        rc_x = chosen["x"].to_numpy(dtype=np.float64)
        rc_y = chosen["y"].to_numpy(dtype=np.float64)
        rr_x = rejected["x"].to_numpy(dtype=np.float64)
        rr_y = rejected["y"].to_numpy(dtype=np.float64)
        for i in range(len(rc_x)):
            for j in range(len(rr_x)):
                pairs.append({
                    "user_id": u, "conversation_id": c,
                    "x_chosen": rc_x[i], "x_rejected": rr_x[j],
                    "y_chosen": rc_y[i], "y_rejected": rr_y[j],
                })
    return pd.DataFrame(pairs)


def split_pair_train_test(pairs: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []
    for u, g in pairs.groupby("user_id", sort=False):
        convs = g["conversation_id"].unique()
        perm = rng.permutation(convs)
        half = len(perm) // 2
        test_convs = set(perm[half:].tolist())
        is_test = g["conversation_id"].isin(test_convs)
        test_rows.append(g[is_test])
        train_rows.append(g[~is_test])
    return pd.concat(train_rows, ignore_index=True), pd.concat(test_rows, ignore_index=True)


def pair_acc_prism(pairs_df: pd.DataFrame, seed: int):
    """Fit PEBS/EBPO/pop on TRAIN half of pairs (stacked chosen+rejected as
    utterance-level obs per user), evaluate pair-acc on TEST half."""
    train_df, test_df = split_pair_train_test(pairs_df, seed)
    # Build utterance-level train table
    train_utt = pd.DataFrame({
        "user_id": np.concatenate([train_df["user_id"].to_numpy(),
                                    train_df["user_id"].to_numpy()]),
        "x": np.concatenate([train_df["x_chosen"].to_numpy(),
                              train_df["x_rejected"].to_numpy()]),
        "y": np.concatenate([train_df["y_chosen"].to_numpy(),
                              train_df["y_rejected"].to_numpy()]),
    })
    fit = fit_population_and_taus(train_utt, "x", "y")

    # Per-user pair-acc on test half
    acc_pebs, acc_ebpo, acc_pop = {}, {}, {}
    for uid, g in test_df.groupby("user_id"):
        xc = g["x_chosen"].to_numpy(dtype=np.float64)
        xr = g["x_rejected"].to_numpy(dtype=np.float64)
        # pop: margin = beta_pop*(xc - xr)
        acc_pop[uid] = float(np.mean(fit["beta_pop"] * (xc - xr) > 0)) if len(xc) else np.nan
        # EBPO: intercept cancels in margin => same ordering as pop-slope
        acc_ebpo[uid] = acc_pop[uid]
        # PEBS: use per-user train-fit shrunk (alpha, beta)
        user_tr = train_utt[train_utt["user_id"] == uid]
        if len(user_tr) < 3:
            acc_pebs[uid] = acc_pop[uid]
            continue
        a, b, sa, sb = fit_ols_with_se(user_tr["x"].to_numpy(dtype=np.float64),
                                       user_tr["y"].to_numpy(dtype=np.float64))
        if not np.isfinite(a):
            acc_pebs[uid] = acc_pop[uid]; continue
        w_a = fit["tau2_a"] / (fit["tau2_a"] + sa ** 2 + 1e-12) if np.isfinite(sa) else 0.0
        w_b = fit["tau2_b"] / (fit["tau2_b"] + sb ** 2 + 1e-12) if np.isfinite(sb) else 0.0
        b_s = w_b * b + (1 - w_b) * fit["beta_pop"]
        acc_pebs[uid] = float(np.mean(b_s * (xc - xr) > 0)) if len(xc) else np.nan
    return acc_pebs, acc_ebpo, acc_pop, len(pairs_df), len(train_df), len(test_df)


# ======================================================================
# Main 6-cell driver
# ======================================================================

def run_cell(backbone: str, corpus: str, args, global_seed: int):
    print(f"\n=== CELL backbone={backbone}  corpus={corpus} ===")
    t0 = time.time()
    if corpus == "prism":
        df = load_prism_cell(backbone)
        print(f"  loaded {len(df)} utterances, {df['user_id'].nunique()} users")
        # Global fit
        conv_count = df.groupby("user_id")["conversation_id"].nunique()
        utt_count = df.groupby("user_id").size()
        keep = conv_count[(conv_count >= MIN_CONV_PER_USER) &
                          (utt_count >= MIN_UTT_PER_USER)].index
        df_fit = df[df["user_id"].isin(keep)].reset_index(drop=True)
        fit = fit_population_and_taus(df_fit, "x", "y")
        print(f"  fit: alpha_pop={fit['alpha_pop']:.3f} beta_pop={fit['beta_pop']:.4f} "
              f"tau2_a={fit['tau2_a']:.2f} tau2_b={fit['tau2_b']:.4f}")
        per_user = run_rmse_cv_prism(df, fit)
    else:
        df = load_pluri_cell(backbone)
        print(f"  loaded {len(df)} rows, {df['user_id'].nunique()} users")
        fit = fit_population_and_taus(df, "x", "y")
        print(f"  fit: alpha_pop={fit['alpha_pop']:.3f} beta_pop={fit['beta_pop']:.4f} "
              f"tau2_a={fit['tau2_a']:.3f} tau2_b={fit['tau2_b']:.4f}")
        per_user = run_rmse_cv_pluri(df, fit, k_folds=5, seed=global_seed)

    if per_user.empty:
        return None

    # Gain % vs pop
    per_user["gain_ebpo_pct"] = 100.0 * (per_user["rmse_pop"] -
                                          per_user["rmse_ebpo"]) / per_user["rmse_pop"]
    per_user["gain_pebs_pct"] = 100.0 * (per_user["rmse_pop"] -
                                           per_user["rmse_pebs"]) / per_user["rmse_pop"]
    gain_ebpo = cluster_boot_ci(per_user["gain_ebpo_pct"].to_numpy(),
                                 args.n_boot, global_seed)
    gain_pebs = cluster_boot_ci(per_user["gain_pebs_pct"].to_numpy(),
                                  args.n_boot, global_seed + 1)
    wilcox_pebs_ebpo_rmse = paired_wilcoxon(
        per_user["gain_pebs_pct"].to_numpy(),
        per_user["gain_ebpo_pct"].to_numpy())

    # Pair-acc: only PRISM
    pair_res = None
    if corpus == "prism":
        pairs_df = build_prism_pair_slice(df)
        if len(pairs_df) > 0:
            acc_pebs, acc_ebpo, acc_pop, n_pairs, n_tr, n_te = \
                pair_acc_prism(pairs_df, global_seed)
            users_common = sorted(set(acc_pebs) & set(acc_ebpo))
            if users_common:
                pil_arr = np.array([acc_pebs[u] for u in users_common])
                ebpo_arr = np.array([acc_ebpo[u] for u in users_common])
                pop_arr = np.array([acc_pop[u] for u in users_common])
                pil_ci = cluster_boot_ci(pil_arr, args.n_boot, global_seed + 2)
                ebpo_ci = cluster_boot_ci(ebpo_arr, args.n_boot, global_seed + 3)
                pop_ci = cluster_boot_ci(pop_arr, args.n_boot, global_seed + 4)
                wilcox_pebs_ebpo_pa = paired_wilcoxon(pil_arr, ebpo_arr)
                pair_res = dict(
                    n_users=len(users_common), n_pairs=int(n_pairs),
                    n_train=int(n_tr), n_test=int(n_te),
                    pebs=pil_ci, ebpo=ebpo_ci, pop=pop_ci,
                    wilcox_pebs_vs_ebpo=wilcox_pebs_ebpo_pa,
                )

    out = dict(
        backbone=backbone, corpus=corpus,
        n_users=int(len(per_user)),
        n_utt=int(per_user["n_utt"].sum()),
        rmse_abs=dict(
            pop=float(per_user["rmse_pop"].mean()),
            ebpo=float(per_user["rmse_ebpo"].mean()),
            pebs=float(per_user["rmse_pebs"].mean()),
        ),
        rmse_gain_pct=dict(
            ebpo=gain_ebpo, pebs=gain_pebs,
        ),
        wilcox_pebs_vs_ebpo_rmse=wilcox_pebs_ebpo_rmse,
        pair_acc=pair_res,
        runtime_s=float(time.time() - t0),
    )
    print(f"  RMSE gain%: PEBS={gain_pebs['mean']:+.2f} "
          f"[{gain_pebs['ci95'][0]:+.2f},{gain_pebs['ci95'][1]:+.2f}] "
          f"EBPO={gain_ebpo['mean']:+.2f} "
          f"[{gain_ebpo['ci95'][0]:+.2f},{gain_ebpo['ci95'][1]:+.2f}] "
          f"wilcox p={wilcox_pebs_ebpo_rmse['p']:.2e}")
    if pair_res:
        print(f"  pair-acc: PEBS={pair_res['pebs']['mean']*100:.2f}% "
              f"EBPO={pair_res['ebpo']['mean']*100:.2f}% "
              f"pop={pair_res['pop']['mean']*100:.2f}%  "
              f"wilcox p={pair_res['wilcox_pebs_vs_ebpo']['p']:.2e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=RNG_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    args = ap.parse_args()

    t0 = time.time()
    results = []
    for backbone, corpus in CELLS:
        try:
            r = run_cell(backbone, corpus, args, args.seed)
            if r is not None:
                results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append(dict(backbone=backbone, corpus=corpus,
                                error=str(e)))

    # Aggregate mean deltas across cells
    rmse_deltas = []
    pair_deltas = []
    for r in results:
        if "error" in r: continue
        d_rmse = r["rmse_gain_pct"]["pebs"]["mean"] - r["rmse_gain_pct"]["ebpo"]["mean"]
        rmse_deltas.append(d_rmse)
        if r.get("pair_acc"):
            d_pa = (r["pair_acc"]["pebs"]["mean"] - r["pair_acc"]["ebpo"]["mean"]) * 100
            pair_deltas.append(d_pa)

    summary = dict(
        iter="N+262",
        seed=args.seed, n_boot=args.n_boot,
        cells=results,
        mean_rmse_gain_delta_pebs_minus_ebpo_pct=float(np.mean(rmse_deltas)) if rmse_deltas else np.nan,
        mean_pair_acc_delta_pebs_minus_ebpo_pp=float(np.mean(pair_deltas)) if pair_deltas else np.nan,
        n_cells_pebs_wins_rmse=int(sum(d > 0 for d in rmse_deltas)),
        n_cells_pebs_wins_pair_acc=int(sum(d > 0 for d in pair_deltas)),
        n_cells=len(rmse_deltas),
        runtime_s=float(time.time() - t0),
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Flat per-cell parquet
    rows = []
    for r in results:
        if "error" in r: continue
        rows.append({
            "backbone": r["backbone"], "corpus": r["corpus"],
            "n_users": r["n_users"], "n_utt": r["n_utt"],
            "rmse_pop": r["rmse_abs"]["pop"],
            "rmse_ebpo": r["rmse_abs"]["ebpo"],
            "rmse_pebs": r["rmse_abs"]["pebs"],
            "gain_ebpo_pct_mean": r["rmse_gain_pct"]["ebpo"]["mean"],
            "gain_ebpo_pct_lo": r["rmse_gain_pct"]["ebpo"]["ci95"][0],
            "gain_ebpo_pct_hi": r["rmse_gain_pct"]["ebpo"]["ci95"][1],
            "gain_pebs_pct_mean": r["rmse_gain_pct"]["pebs"]["mean"],
            "gain_pebs_pct_lo": r["rmse_gain_pct"]["pebs"]["ci95"][0],
            "gain_pebs_pct_hi": r["rmse_gain_pct"]["pebs"]["ci95"][1],
            "wilcox_rmse_p": r["wilcox_pebs_vs_ebpo_rmse"]["p"],
            "pa_pebs": r["pair_acc"]["pebs"]["mean"] if r.get("pair_acc") else np.nan,
            "pa_ebpo": r["pair_acc"]["ebpo"]["mean"] if r.get("pair_acc") else np.nan,
            "pa_pop": r["pair_acc"]["pop"]["mean"] if r.get("pair_acc") else np.nan,
            "wilcox_pa_p": r["pair_acc"]["wilcox_pebs_vs_ebpo"]["p"] if r.get("pair_acc") else np.nan,
        })
    pd.DataFrame(rows).to_parquet(OUT / "per_cell.parquet", index=False)

    # Pretty table
    print("\n" + "=" * 80)
    print("6-cell EBPO vs PEBS head-to-head matrix")
    print("=" * 80)
    print(f"{'backbone':>12s} {'corpus':>10s} {'n_u':>5s} "
          f"{'g_PIL':>8s} {'g_EBPO':>8s} {'Δ_g':>7s} {'w_p':>9s}   "
          f"{'PA_PIL':>7s} {'PA_EBPO':>8s} {'Δ_PA':>7s} {'w_p':>9s}")
    for r in results:
        if "error" in r:
            print(f"{r['backbone']:>12s} {r['corpus']:>10s}  ERROR {r['error']}")
            continue
        gp = r["rmse_gain_pct"]["pebs"]["mean"]
        ge = r["rmse_gain_pct"]["ebpo"]["mean"]
        dg = gp - ge
        wp_rmse = r["wilcox_pebs_vs_ebpo_rmse"]["p"]
        if r.get("pair_acc"):
            pap = r["pair_acc"]["pebs"]["mean"] * 100
            pae = r["pair_acc"]["ebpo"]["mean"] * 100
            dpa = pap - pae
            wp_pa = r["pair_acc"]["wilcox_pebs_vs_ebpo"]["p"]
            pa_str = f"{pap:7.2f} {pae:8.2f} {dpa:+7.2f} {wp_pa:9.2e}"
        else:
            pa_str = f"{'n/a':>7s} {'n/a':>8s} {'n/a':>7s} {'n/a':>9s}"
        print(f"{r['backbone']:>12s} {r['corpus']:>10s} {r['n_users']:>5d} "
              f"{gp:+7.2f} {ge:+7.2f} {dg:+7.2f} {wp_rmse:9.2e}   {pa_str}")
    print(f"\nmean Δ RMSE gain % (PEBS - EBPO) over {summary['n_cells']} cells: "
          f"{summary['mean_rmse_gain_delta_pebs_minus_ebpo_pct']:+.3f} pp")
    print(f"mean Δ pair-acc pp (PEBS - EBPO) over {len(pair_deltas)} pair-acc cells: "
          f"{summary['mean_pair_acc_delta_pebs_minus_ebpo_pp']:+.3f} pp")
    print(f"[save] {OUT / 'summary.json'}")
    print(f"[wall-clock] {summary['runtime_s']:.1f}s")


if __name__ == "__main__":
    main()
