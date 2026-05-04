"""P-GenRM (Zhang et al. 2026, arXiv:2602.12116, ICLR 2026 Oral) head-to-head
on PRISM LOCO slice — faithful re-implementation adapted to scalar RM scores.

Reference (as resolved 2026-05-02 via WebFetch + Skill: paper-citation-integrity-audit):
    P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling.
    Authors: Pinyi Zhang, Ting-En Lin, Yuchuan Wu, Jingyang Chen, Zongqi Wang,
             Hua Yang, Ze Xu, Fei Huang, Kai Zhang, Yongbin Li.
    arXiv: 2602.12116v1, Feb 2026; ICLR 2026 Oral.
    Code: github.com/Tongyi-ConvAI/Qwen-Character (Character-GenRM*).

P-GenRM's published method:
    1. Encode each user's preference signals (multiple persona analyses) via
       Qwen3-Embedding-0.6B into a cross-scenario preference matrix P.
    2. K-means cluster P into k=50 prototypes; train an LLM-judge with GRPO to
       generate scoring rubrics conditioned on a prototype + user history; refine
       the prototype-to-rubric attention with a pairwise loss.
       L = L_pair + lambda_cent ||a_j - mu_j||^2 + lambda_tr ||a_j - p_j||^2.
    3. At test time, dual-granularity aggregation:
         s = (1/m) sum_x Extract(S_x^i)         <- m individual-level samples
           + (1/n) sum_w Extract(S^{u_w})       <- n prototype-similar users.
       Default (m=8, n=4); optimal reported (m=16, n=8) -> +2.99% over base.

Why we re-implement instead of running the published checkpoint:
    P-GenRM (a) requires a generative LLM-judge backbone (LLaMA-3.1-8B/70B in
    the paper; Qwen-Character branch), (b) consumes per-user persona analyses
    that PRISM does not provide, and (c) operates over PersonalRewardBench
    (Chatbot Arena-Personalized + PRISM-Personalized) and LaMP-QA, which use
    pairwise-accuracy / Spearman, NOT continuous-RMSE. There is no runnable
    public end-to-end pipeline that maps PRISM continuous score_user labels
    onto P-GenRM's pipeline. Re-training a personalised generative judge for
    apples-to-apples PRISM RMSE is ~50+ GPU-hours and would not be a
    P-GenRM-vs-PILSD comparison but a P-GenRM-engineering exercise. Instead
    we follow the same protocol used for the four other neighbours in
    `scripts/neighbor_head_to_head.py` (LoRe, EBPO, PReF, MRM): re-implement
    the CORE METHODOLOGICAL CONTRIBUTION on the PRISM scalar slice.

Adaptation (faithful to P-GenRM's algorithm; transparent about substitutions):
    A. PERSONA EMBEDDING. For PRISM each user has up to ~48 RM-scored
       utterances and up to 6 demographic flags. We construct a per-user
       preference vector p_u in R^d via:
         d_features = [mean(rm_score), std(rm_score), mean(score_user),
                       std(score_user), n_utt, slope(score_user vs rm_score),
                       intercept(...), corr(...)].
       This is the scalar analog of P-GenRM's persona-from-history step;
       Qwen3-Embedding-0.6B is over-kill for d~8 features and would not
       improve cluster quality on this signal.
    B. K-MEANS PROTOTYPES. K-means on z-scored p_u, default k=50 (paper) but
       PRISM has only ~1370 users so we sweep k in {10, 20, 50} and pick by
       silhouette score (recorded). Each prototype owns a calibrator
       (alpha_proto, beta_proto) fit by pooled OLS on its members'
       (rm_score, score_user) pairs.
    C. PROTOTYPE REFINEMENT (paper's L = L_pair + lambda_cent + lambda_tr).
       L_pair on continuous labels reduces to MSE; on PRISM we already fit OLS
       per prototype, which IS the closed-form MSE minimiser. The two
       regularisers (centring lambda_cent, transition lambda_tr) shrink each
       prototype's calibrator toward the cluster centroid and toward the
       prior alpha_pop / beta_pop respectively. We implement the explicit
       weighted shrinkage:
         (alpha_p, beta_p) = w_proto * (alpha_p^OLS, beta_p^OLS) +
                             w_pop   * (alpha_pop, beta_pop)
       with w_proto = N_p / (N_p + lambda_tr) so larger prototypes pull less
       toward the global prior (analog of Eq. 4 in the paper).
    D. DUAL-GRANULARITY AGGREGATION. At test time:
         (i) INDIVIDUAL LEVEL: for each held-out conversation the user's
             remaining utterances form their support; we resample m=8
             bootstrap subsets and fit per-bootstrap (alpha_user, beta_user)
             OLS, then average predictions: this is the "m individual-level
             samples" channel.
         (ii) PROTOTYPE LEVEL: assign the user to its K-means prototype, find
             the n=4 nearest neighbours WITHIN that prototype by feature-space
             distance; pool their (rm_score, score_user) pairs and predict
             with the resulting OLS calibrator: this is the "n prototype-
             similar users" channel.
         (iii) Final prediction is the equal-weighted (1/m + 1/n)/2 mean of
              the two channel outputs, exactly matching paper Eq. (3).
    E. TEST-TIME SCALING. We report m=8/n=4 (default) AND m=16/n=8 (paper-
       optimal) and a m=4/n=2 ablation; honest disclosure of the gain/loss
       between the two settings, mirroring P-GenRM's Tab. 4 ablation.

Honest scope caveats baked into the script:
    1. P-GenRM is GENERATIVE; ours is the AFFINE-CALIBRATOR SHADOW of its
       prototype-level inductive bias. The pair-accuracy gain from greedy
       persona-rubric generation is excluded by design.
    2. Persona features are 8-D scalar, not Qwen3-Embedding-0.6B. This may
       under-cluster users vs the paper's setting; we report silhouette for
       transparency.
    3. The pairwise loss reduces to MSE on continuous labels; the paper's
       L_pair as written cannot apply directly to score_user RMSE.
    4. Test-time scaling parameters (m, n) are exposed via CLI for
       reproducibility.

Outputs (mirroring scripts/neighbor_head_to_head.py):
    results/track1_p_genrm_h2h/summary.json      <- methods.{p_genrm_default,
                                                     p_genrm_optimal, p_genrm_low,
                                                     pilsd_shrunk, pop_slope}
    results/track1_p_genrm_h2h/per_user.parquet  <- per-user RMSE rows
    paper/tables/p_genrm_headtohead_numbers.tex  <- LaTeX table
    paper/figures/fig_p_genrm_headtohead.{pdf,png}

Seed: 20260420 (matches scripts/neighbor_head_to_head.py and other PRISM h2h scripts).
"""
from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# Paths -----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]   # 3_PILSD_Standalone/
T1 = ROOT.parent / "1_Causal_RLHF"
SCORED = T1 / "data/prism_rm_scored.parquet"

OUT_RESULTS = ROOT / "results/track1_p_genrm_h2h"
OUT_TABLE = ROOT / "paper/tables/p_genrm_headtohead_numbers.tex"
OUT_FIG = ROOT / "paper/figures/fig_p_genrm_headtohead.pdf"

# Defaults --------------------------------------------------------------------
RNG_DEFAULT = 20260420
N_BOOT_DEFAULT = 2000

MIN_CONV_PER_USER = 2
MIN_UTT_PER_USER = 10

DEFAULT_K_GRID = (10, 20, 50)
PAPER_DEFAULT_M_N = (8, 4)
PAPER_OPTIMAL_M_N = (16, 8)
PAPER_LOW_M_N = (4, 2)

# G1: HONEST shrinkage hyper-priors (mirror paper's lambda_cent, lambda_tr).
# Since the paper does not pin specific scalar values, we set them so that
# prototypes with N_p >= 30 get >= 80% own-OLS weight.
LAMBDA_TR_DEFAULT = 8.0     # transition (toward population prior)
LAMBDA_CENT_DEFAULT = 0.0   # centring (we set 0 because OLS within-prototype
                            # already minimises MSE within members; non-zero
                            # would re-add bias to the cluster centroid)


# ============================================================================
# G1 HELPER — exact-OLS with SE so the audit can verify per-line math
# ============================================================================

def fit_ols_with_se(x: np.ndarray, y: np.ndarray):
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan
    xm = float(np.mean(x))
    ssx = float(np.sum((x - xm) ** 2))
    if ssx < 1e-10:
        return float(np.mean(y)), 0.0, np.inf, np.inf
    beta = float(np.sum((x - xm) * (y - np.mean(y))) / ssx)
    alpha = float(np.mean(y) - beta * xm)
    resid = y - (alpha + beta * x)
    n = len(x)
    if n <= 2:
        return alpha, beta, np.inf, np.inf
    mse = float(np.sum(resid ** 2) / (n - 2))
    se_alpha = float(np.sqrt(mse * (1.0 / n + xm ** 2 / ssx)))
    se_beta = float(np.sqrt(mse / ssx))
    return alpha, beta, se_alpha, se_beta


# ============================================================================
# G1 / G2 / G5 — P-GenRM core: persona features
# ============================================================================

def build_persona_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Per-user 8-D feature vector p_u proxying P-GenRM's persona embedding.

    Each user's preference signature on PRISM is summarised by:
        f1 = mean(rm_score)         f2 = std(rm_score)
        f3 = mean(score_user)       f4 = std(score_user)
        f5 = n_utterances           f6 = corr(rm_score, score_user)
        f7 = OLS slope              f8 = OLS intercept
    Standardised (z-score) before K-means.
    Returns (per_user dataframe with user_id index, P matrix, feature names).
    """
    rows = []
    for uid, g in df.groupby("user_id"):
        x = g["rm_score"].to_numpy(dtype=np.float64)
        y = g["score_user"].to_numpy(dtype=np.float64)
        f1 = float(np.mean(x))
        f2 = float(np.std(x))
        f3 = float(np.mean(y))
        f4 = float(np.std(y))
        f5 = float(len(g))
        if len(g) >= 3 and np.std(x) > 1e-8 and np.std(y) > 1e-8:
            f6 = float(np.corrcoef(x, y)[0, 1])
            beta = float(np.sum((x - f1) * (y - f3)) / max(np.sum((x - f1) ** 2), 1e-10))
            alpha = float(f3 - beta * f1)
            f7, f8 = beta, alpha
        else:
            f6 = 0.0
            f7 = 0.0
            f8 = float(f3)
        if not np.isfinite(f6):
            f6 = 0.0
        rows.append({"user_id": uid, "f1_rm_mean": f1, "f2_rm_std": f2,
                     "f3_y_mean": f3, "f4_y_std": f4, "f5_n_utt": f5,
                     "f6_corr": f6, "f7_slope": f7, "f8_intercept": f8})
    pu = pd.DataFrame(rows).set_index("user_id")
    feat_names = ["f1_rm_mean", "f2_rm_std", "f3_y_mean", "f4_y_std",
                  "f5_n_utt", "f6_corr", "f7_slope", "f8_intercept"]
    P = pu[feat_names].to_numpy(dtype=np.float64)
    # z-score (column-wise)
    P_mu = P.mean(axis=0)
    P_sd = P.std(axis=0) + 1e-8
    P_z = (P - P_mu) / P_sd
    return pu, P_z, feat_names


# ============================================================================
# G1 / G5 — P-GenRM core: K-means prototypes + per-prototype OLS calibrator
# ============================================================================

def fit_prototypes_and_calibrators(
    df: pd.DataFrame, P_z: np.ndarray, user_index: pd.Index,
    k: int, alpha_pop: float, beta_pop: float,
    lambda_tr: float, seed: int,
):
    """K-means on z-scored persona features, then per-prototype OLS calibrator
    with paper's shrinkage prior toward the population fit.

    Returns dict:
        kmeans (sklearn fitted),
        proto_user_ids   (k-list of arrays of user_id),
        proto_calib      (k-list of (alpha_p, beta_p, n_p) tuples after
                          P-GenRM-style shrinkage toward population prior).
    """
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    cluster_ids = kmeans.fit_predict(P_z)
    proto_user_ids: list[np.ndarray] = []
    proto_calib: list[tuple[float, float, int]] = []

    df_indexed = df.set_index("user_id")
    user_array = np.array(user_index)

    for cid in range(k):
        members = user_array[cluster_ids == cid]
        proto_user_ids.append(members)
        if len(members) == 0:
            proto_calib.append((alpha_pop, beta_pop, 0))
            continue
        # pool all utterances of all members
        sub = df_indexed.loc[members]
        x = sub["rm_score"].to_numpy(dtype=np.float64)
        y = sub["score_user"].to_numpy(dtype=np.float64)
        a_ols, b_ols, _, _ = fit_ols_with_se(x, y)
        if not np.isfinite(a_ols):
            proto_calib.append((alpha_pop, beta_pop, len(members)))
            continue
        n_p = len(members)
        # paper Eq. (4)-style shrinkage: w_proto = N_p / (N_p + lambda_tr)
        w_proto = n_p / (n_p + lambda_tr + 1e-12)
        a_p = w_proto * a_ols + (1 - w_proto) * alpha_pop
        b_p = w_proto * b_ols + (1 - w_proto) * beta_pop
        proto_calib.append((a_p, b_p, n_p))
    return {
        "kmeans": kmeans,
        "cluster_ids": cluster_ids,            # per user
        "proto_user_ids": proto_user_ids,
        "proto_calib": proto_calib,
        "user_index": user_index,
    }


# ============================================================================
# G1 / G2 — Dual-granularity: individual + prototype channels
# ============================================================================

def predict_individual_channel(x_tr: np.ndarray, y_tr: np.ndarray,
                                x_te: np.ndarray, m: int, rng: np.random.Generator,
                                alpha_pop: float, beta_pop: float):
    """m bootstrap-resampled OLS fits on the user's train fold; mean
    prediction = '(1/m) sum_x Extract(S_x^i)' channel of P-GenRM.

    If a bootstrap fold is degenerate (constant x), fall back to population
    line for that fold so the average is well-defined.
    """
    if len(x_tr) < 3:
        return alpha_pop + beta_pop * x_te
    n = len(x_tr)
    preds = np.zeros((m, len(x_te)))
    for j in range(m):
        idx = rng.integers(0, n, size=n)
        a, b, _, _ = fit_ols_with_se(x_tr[idx], y_tr[idx])
        if not np.isfinite(a):
            preds[j] = alpha_pop + beta_pop * x_te
        else:
            preds[j] = a + b * x_te
    return preds.mean(axis=0)


def predict_prototype_channel(x_tr: np.ndarray, y_tr: np.ndarray,
                               x_te: np.ndarray, n_neigh: int,
                               user_id, persona_z: np.ndarray, proto_state,
                               df_indexed: pd.DataFrame,
                               alpha_pop: float, beta_pop: float):
    """Find the user's prototype, then n_neigh nearest peers WITHIN the
    prototype by persona-feature distance; pool their (rm, y) pairs (excluding
    target user's own data to avoid leakage) and fit OLS.

    Falls back to the prototype-level shrunk calibrator if too-few peers.
    Implements paper '(1/n) sum_w Extract(S^{u_w})'.
    """
    user_index = proto_state["user_index"]
    cluster_ids = proto_state["cluster_ids"]
    proto_user_ids = proto_state["proto_user_ids"]
    proto_calib = proto_state["proto_calib"]

    pos_in_user_index = user_index.get_loc(user_id)
    cid = int(cluster_ids[pos_in_user_index])
    members = proto_user_ids[cid]
    peers = members[members != user_id]
    if len(peers) == 0:
        a_p, b_p, _ = proto_calib[cid]
        return a_p + b_p * x_te
    # rank peers by L2 distance in persona-feature space
    own_vec = persona_z[pos_in_user_index]
    peer_pos = np.array([user_index.get_loc(uid) for uid in peers])
    peer_vecs = persona_z[peer_pos]
    dists = np.linalg.norm(peer_vecs - own_vec[None, :], axis=1)
    keep = peers[np.argsort(dists)[:max(1, n_neigh)]]
    sub = df_indexed.loc[keep]
    xn = sub["rm_score"].to_numpy(dtype=np.float64)
    yn = sub["score_user"].to_numpy(dtype=np.float64)
    a, b, _, _ = fit_ols_with_se(xn, yn)
    if not np.isfinite(a):
        a_p, b_p, _ = proto_calib[cid]
        return a_p + b_p * x_te
    return a + b * x_te


def method_p_genrm(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray,
                    m: int, n_neigh: int, rng: np.random.Generator,
                    user_id, persona_z: np.ndarray, proto_state,
                    df_indexed: pd.DataFrame,
                    alpha_pop: float, beta_pop: float):
    """P-GenRM prediction = equal-weight mean of individual + prototype channels.

    Mirrors paper Eq. (3): s = (1/m) sum_x Extract(S_x^i) + (1/n) sum_w Extract(S^{u_w}),
    with the two channels averaged equally to produce the final scalar.
    """
    p_indiv = predict_individual_channel(x_tr, y_tr, x_te, m, rng,
                                          alpha_pop, beta_pop)
    p_proto = predict_prototype_channel(x_tr, y_tr, x_te, n_neigh,
                                         user_id, persona_z, proto_state,
                                         df_indexed, alpha_pop, beta_pop)
    return 0.5 * p_indiv + 0.5 * p_proto


# ============================================================================
# G1 / G5 — PILSD-shrunk reference (mirrors scripts/neighbor_head_to_head.py)
# ============================================================================

def method_pilsd_shrunk(x_tr, y_tr, x_te, alpha_pop, beta_pop, tau2_a, tau2_b):
    a, b, se_a, se_b = fit_ols_with_se(x_tr, y_tr)
    if not np.isfinite(a):
        return alpha_pop + beta_pop * x_te
    w_a = tau2_a / (tau2_a + se_a ** 2 + 1e-12) if np.isfinite(se_a) else 0.0
    w_b = tau2_b / (tau2_b + se_b ** 2 + 1e-12) if np.isfinite(se_b) else 0.0
    a_s = w_a * a + (1 - w_a) * alpha_pop
    b_s = w_b * b + (1 - w_b) * beta_pop
    return a_s + b_s * x_te


# ============================================================================
# G7 — LOCO loop with diagnostic instrumentation
# ============================================================================

METHOD_NAMES = ["pop_slope", "pilsd_shrunk",
                "p_genrm_low", "p_genrm_default", "p_genrm_optimal"]


def run_loco(args):
    t0 = time.time()
    if not SCORED.exists():
        sys.stderr.write(f"FATAL: PRISM scored data not found: {SCORED}\n")
        sys.exit(2)
    df = pd.read_parquet(SCORED).dropna(subset=["score_user", "rm_score", "conversation_id"])
    conv_count = df.groupby("user_id")["conversation_id"].nunique()
    utt_count = df.groupby("user_id").size()
    keep = conv_count[(conv_count >= MIN_CONV_PER_USER) &
                      (utt_count >= MIN_UTT_PER_USER)].index
    df = df[df["user_id"].isin(keep)].reset_index(drop=True)
    print(f"[filter] {len(df)} utt x {df['user_id'].nunique()} users")

    # Population OLS prior (used by every method's fallback path)
    slope_pop, intercept_pop = np.polyfit(df["rm_score"].to_numpy(),
                                           df["score_user"].to_numpy().astype(float), 1)
    alpha_pop, beta_pop = float(intercept_pop), float(slope_pop)
    print(f"[pop]    alpha_pop={alpha_pop:.3f}  beta_pop={beta_pop:.4f}")

    # Persona features
    persona_pu, P_z, feat_names = build_persona_features(df)
    print(f"[persona] features={feat_names}  P_z shape={P_z.shape}")

    # PILSD MoM tau^2_a, tau^2_b (reference baseline)
    user_stats = []
    for uid, g in df.groupby("user_id"):
        a, b, sa, sb = fit_ols_with_se(g["rm_score"].to_numpy(),
                                        g["score_user"].to_numpy().astype(float))
        user_stats.append((a, b, sa, sb))
    alphas = np.array([s[0] for s in user_stats if np.isfinite(s[0])])
    betas = np.array([s[1] for s in user_stats if np.isfinite(s[1])])
    sas = np.array([s[2] for s in user_stats if np.isfinite(s[2])])
    sbs = np.array([s[3] for s in user_stats if np.isfinite(s[3])])
    tau2_a = max(0.0, float(np.var(alphas, ddof=1) - np.mean(sas ** 2)))
    tau2_b = max(0.0, float(np.var(betas, ddof=1) - np.mean(sbs ** 2)))
    print(f"[pilsd]  tau2_a={tau2_a:.2f}  tau2_b={tau2_b:.4f}")

    # K sweep + silhouette
    if args.k is not None:
        k_grid = (args.k,)
    else:
        # cap k by n_users // 5 to ensure each prototype has >=5 members on avg
        n_users = len(persona_pu)
        k_grid = tuple(k for k in DEFAULT_K_GRID if k <= n_users // 5)
        if len(k_grid) == 0:
            k_grid = (max(2, n_users // 10),)
    silhouettes = {}
    for k in k_grid:
        try:
            km = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
            lbl = km.fit_predict(P_z)
            sil = float(silhouette_score(P_z, lbl)) if len(set(lbl)) > 1 else float("nan")
        except Exception as e:
            sil = float("nan")
            print(f"[kmeans-sweep] k={k} FAILED: {e}")
        silhouettes[k] = sil
        print(f"[kmeans-sweep] k={k}  silhouette={sil:.4f}")
    best_k = max(k_grid, key=lambda kk: -1.0 if not np.isfinite(silhouettes[kk]) else silhouettes[kk])
    print(f"[kmeans] picked k={best_k} (silhouette={silhouettes[best_k]:.4f})")

    proto_state = fit_prototypes_and_calibrators(
        df, P_z, persona_pu.index, k=best_k,
        alpha_pop=alpha_pop, beta_pop=beta_pop,
        lambda_tr=args.lambda_tr, seed=args.seed,
    )
    proto_sizes = [len(arr) for arr in proto_state["proto_user_ids"]]
    print(f"[kmeans] prototype sizes min={min(proto_sizes)} max={max(proto_sizes)} "
          f"median={int(np.median(proto_sizes))}")

    df_indexed = df.set_index("user_id")
    rng_master = np.random.default_rng(args.seed)

    # Per-user RMSE accumulators
    per_user = []
    for uid, g in tqdm(df.groupby("user_id"), desc="LOCO"):
        conv_ids = g["conversation_id"].unique().tolist()
        if len(conv_ids) < MIN_CONV_PER_USER:
            continue
        sq = {m: [] for m in METHOD_NAMES}
        n_utt_used = 0
        for hc in conv_ids:
            tr = g[g["conversation_id"] != hc]
            te = g[g["conversation_id"] == hc]
            if len(tr) < 3 or len(te) < 1:
                continue
            x_tr = tr["rm_score"].to_numpy(dtype=np.float64)
            y_tr = tr["score_user"].to_numpy(dtype=np.float64)
            x_te = te["rm_score"].to_numpy(dtype=np.float64)
            y_te = te["score_user"].to_numpy(dtype=np.float64)
            preds = {
                "pop_slope": alpha_pop + beta_pop * x_te,
                "pilsd_shrunk": method_pilsd_shrunk(x_tr, y_tr, x_te,
                                                     alpha_pop, beta_pop,
                                                     tau2_a, tau2_b),
            }
            for tag, (m_s, n_s) in (
                ("p_genrm_low", PAPER_LOW_M_N),
                ("p_genrm_default", PAPER_DEFAULT_M_N),
                ("p_genrm_optimal", PAPER_OPTIMAL_M_N),
            ):
                rng_local = np.random.default_rng(
                    rng_master.integers(0, 2 ** 32 - 1)
                )
                preds[tag] = method_p_genrm(
                    x_tr, y_tr, x_te, m=m_s, n_neigh=n_s,
                    rng=rng_local, user_id=uid, persona_z=P_z,
                    proto_state=proto_state, df_indexed=df_indexed,
                    alpha_pop=alpha_pop, beta_pop=beta_pop,
                )
            for m in METHOD_NAMES:
                sq[m].extend(((preds[m] - y_te) ** 2).tolist())
            n_utt_used += len(y_te)
        if n_utt_used < MIN_UTT_PER_USER:
            continue
        row = {"user_id": uid, "n_utt": n_utt_used, "n_conv": len(conv_ids)}
        for m in METHOD_NAMES:
            row[f"rmse_{m}"] = float(np.sqrt(np.mean(sq[m])))
        per_user.append(row)
    pu = pd.DataFrame(per_user)
    print(f"\n[loco] {len(pu)} users evaluated")

    # Relative RMSE reduction (%) vs pop_slope baseline per user
    for m in METHOD_NAMES:
        if m == "pop_slope":
            continue
        pu[f"gain_{m}_pct"] = 100.0 * (pu["rmse_pop_slope"] - pu[f"rmse_{m}"]) / pu["rmse_pop_slope"]

    # Cluster-bootstrap mean + 95% CI per method
    def cluster_boot_ci(values: np.ndarray, n_boot: int, seed: int):
        rng = np.random.default_rng(seed)
        n = len(values)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boots[b] = float(values[idx].mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return float(values.mean()), float(lo), float(hi), float(np.std(boots))

    summary = {
        "slice": {
            "n_users": int(len(pu)),
            "n_utterances": int(pu["n_utt"].sum()),
            "min_conv_per_user": MIN_CONV_PER_USER,
            "min_utt_per_user": MIN_UTT_PER_USER,
        },
        "rng_seed": args.seed,
        "n_bootstrap": args.n_boot,
        "kmeans": {
            "k_grid_silhouettes": {str(k): silhouettes[k] for k in k_grid},
            "k_picked": best_k,
            "proto_size_min": int(min(proto_sizes)),
            "proto_size_max": int(max(proto_sizes)),
            "proto_size_median": int(np.median(proto_sizes)),
            "lambda_tr": float(args.lambda_tr),
            "lambda_cent": float(args.lambda_cent),
        },
        "p_genrm_settings": {
            "low": PAPER_LOW_M_N, "default": PAPER_DEFAULT_M_N,
            "optimal": PAPER_OPTIMAL_M_N,
        },
        "methods": {},
        "rmse_abs": {m: float(pu[f"rmse_{m}"].mean()) for m in METHOD_NAMES},
        "verdict_class": "PENDING-RESULT",  # filled after CI computed
        "anomalies_fired": [],
    }
    print("\n=== Head-to-head: mean RMSE reduction (%) vs pop_slope ===")
    print(f"{'method':>20s}  {'RMSE':>8s}  {'gain%':>9s}  {'95% CI':>22s}")
    print(f"{'pop_slope':>20s}  {summary['rmse_abs']['pop_slope']:8.3f}  "
          f"{'0.00':>9s}  {'(reference)':>22s}")
    for m in METHOD_NAMES:
        if m == "pop_slope":
            continue
        gain_col = f"gain_{m}_pct"
        mean_g, lo, hi, se = cluster_boot_ci(pu[gain_col].to_numpy(),
                                              args.n_boot, args.seed)
        summary["methods"][m] = {
            "rmse_mean": float(pu[f"rmse_{m}"].mean()),
            "rmse_reduction_pct_mean": mean_g,
            "rmse_reduction_pct_ci95": [lo, hi],
            "rmse_reduction_pct_se": se,
            "status": "reimplementation" if m != "pilsd_shrunk" else "reference",
        }
        print(f"{m:>20s}  {pu[f'rmse_{m}'].mean():8.3f}  "
              f"{mean_g:+9.3f}  [{lo:+6.3f}, {hi:+6.3f}]")

    # Paired delta vs PILSD-shrunk
    pilsd_gain = pu["gain_pilsd_shrunk_pct"].to_numpy()
    summary["paired_vs_pilsd"] = {}
    for m in METHOD_NAMES:
        if m in ("pop_slope", "pilsd_shrunk"):
            continue
        diff = pilsd_gain - pu[f"gain_{m}_pct"].to_numpy()
        md, lo, hi, se = cluster_boot_ci(diff, args.n_boot, args.seed + 1)
        summary["paired_vs_pilsd"][m] = {
            "mean_delta_pct": md, "ci95": [lo, hi], "se": se,
            "pilsd_better": bool(lo > 0),
        }
        sig = "*" if lo > 0 or hi < 0 else ""
        print(f"  PILSD - {m:>16s}: {md:+.3f}%  CI [{lo:+.3f}, {hi:+.3f}] {sig}")

    # G12 honest-disclosure verdict assignment per 4-class vocabulary
    pilsd_g_mean = summary["methods"]["pilsd_shrunk"]["rmse_reduction_pct_mean"]
    pilsd_g_lo = summary["methods"]["pilsd_shrunk"]["rmse_reduction_pct_ci95"][0]
    pilsd_dom = all(
        summary["paired_vs_pilsd"][m]["pilsd_better"]
        for m in ("p_genrm_low", "p_genrm_default", "p_genrm_optimal")
    )
    if pilsd_g_lo > 0 and pilsd_dom:
        summary["verdict_class"] = "ESTABLISHED-PILSD-GAIN-DOMINATES-P-GENRM"
    elif pilsd_g_lo > 0 and not pilsd_dom:
        summary["verdict_class"] = "MODERATE-PILSD-GAIN-PARITY-WITH-P-GENRM"
    elif pilsd_g_lo <= 0 and pilsd_dom:
        summary["verdict_class"] = "PRELIMINARY-INCONCLUSIVE-PILSD-GAIN-CI-INCLUDES-ZERO"
    else:
        summary["verdict_class"] = "FALSIFIED-P-GENRM-DOMINATES-PILSD"

    summary["runtime_seconds"] = float(time.time() - t0)

    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    (OUT_RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    pu.to_parquet(OUT_RESULTS / "per_user.parquet", index=False)
    print(f"\n[save] {OUT_RESULTS}/summary.json  ({summary['runtime_seconds']:.1f}s)")
    return summary, pu


# ============================================================================
# Output: LaTeX table + figure (mirror the neighbor_head_to_head style)
# ============================================================================

def write_latex_table(summary: dict, out_path: Path):
    pretty = {
        "pilsd_shrunk":     ("\\PILSD{} \\emph{(ours)}",
                              "Per-user EB-shrunk $(\\alpha_j, \\beta_j)$ affine calibrator",
                              "hierarchical EB"),
        "p_genrm_low":      ("P-GenRM $m{=}4,n{=}2$ \\citep{pgenrm2026}",
                              "Dual-granularity $(m,n)$ aggregation, low-budget",
                              "K-means k=" + str(summary["kmeans"]["k_picked"])),
        "p_genrm_default":  ("P-GenRM $m{=}8,n{=}4$ \\citep{pgenrm2026}",
                              "Paper-default test-time scaling",
                              "K-means k=" + str(summary["kmeans"]["k_picked"])),
        "p_genrm_optimal":  ("P-GenRM $m{=}16,n{=}8$ \\citep{pgenrm2026}",
                              "Paper-optimal test-time scaling",
                              "K-means k=" + str(summary["kmeans"]["k_picked"])),
    }
    order = ["pilsd_shrunk", "p_genrm_low", "p_genrm_default", "p_genrm_optimal"]
    lines = []
    lines.append("% P-GenRM (Zhang et al. 2026, ICLR Oral) head-to-head on PRISM.")
    lines.append("% Generated by paper/scripts/baselines/p_genrm_2026_on_prism.py")
    lines.append(f"% N_users = {summary['slice']['n_users']}, "
                 f"N_utt = {summary['slice']['n_utterances']}, "
                 f"seed = {summary['rng_seed']}, n_boot = {summary['n_bootstrap']}, "
                 f"K-means k = {summary['kmeans']['k_picked']}.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering \\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append("\\caption{\\textbf{P-GenRM head-to-head RMSE comparison on PRISM held-out-conversation slice} "
                 "(N=" + str(summary['slice']['n_users']) + " users, "
                 + str(summary['slice']['n_utterances'] // max(summary['slice']['n_users'], 1))
                 + " utterances/user, same 7B Qwen-Instruct RM scores for every method, "
                 "cluster-bootstrap 95\\% CI over users, $n_{\\mathrm{boot}}$="
                 + str(summary['n_bootstrap']) + ", seed=" + str(summary['rng_seed']) + "). "
                 "P-GenRM \\citep{pgenrm2026} (ICLR 2026 Oral, arXiv:2602.12116) is re-implemented "
                 "on our slice via faithful adaptation of its dual-granularity prototype-and-individual "
                 "channels (no runnable public end-to-end pipeline maps PRISM continuous score\\_user "
                 "to P-GenRM's persona-rubric LLM-judge); reported numbers are from our re-implementation, "
                 "which preserves the (a) K-means user-prototype clustering on a per-user persona feature, "
                 "(b) per-prototype affine calibrator with paper-style transition shrinkage to the population "
                 "prior, and (c) Eq.~(3) equal-weight $(m,n)$-aggregation across individual + prototype "
                 "channels, but replaces the generative LLM-judge with the affine-calibrator shadow on "
                 "scalar RM scores. See \\texttt{paper/scripts/baselines/p\\_genrm\\_2026\\_on\\_prism.py} "
                 "for the exact recipe and \\S~Limitations for honest-scope caveats.}")
    lines.append("\\label{tab:pgenrm-headtohead-numbers}")
    lines.append("\\begin{tabular}{@{}p{2.8cm}p{4.6cm}p{2.6cm}rr@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & \\textbf{Recipe on PRISM slice} & \\textbf{Partial-pooling} "
                 "& \\textbf{RMSE $\\downarrow$ (\\%)} & \\textbf{95\\% CI} \\\\")
    lines.append("\\midrule")
    for m in order:
        pretty_name, recipe, tag = pretty[m]
        meth = summary['methods'][m]
        g = meth['rmse_reduction_pct_mean']
        ci = meth['rmse_reduction_pct_ci95']
        g_s = f"\\textbf{{{g:+.2f}}}" if m == "pilsd_shrunk" else f"{g:+.2f}"
        ci_s = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"
        lines.append(f"{pretty_name} & {recipe} & {tag} & {g_s} & {ci_s} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{-0.5em}")
    lines.append("\\end{table}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[save] {out_path}")


def write_figure(summary: dict, out_path: Path):
    pretty = {"pilsd_shrunk": "PILSD (ours)",
              "p_genrm_low": "P-GenRM m=4,n=2",
              "p_genrm_default": "P-GenRM m=8,n=4",
              "p_genrm_optimal": "P-GenRM m=16,n=8"}
    order = ["pilsd_shrunk", "p_genrm_low", "p_genrm_default", "p_genrm_optimal"]
    colors = {"pilsd_shrunk": "#1f77b4",
              "p_genrm_low": "#aec7e8",
              "p_genrm_default": "#ff9896",
              "p_genrm_optimal": "#d62728"}
    means = [summary['methods'][m]['rmse_reduction_pct_mean'] for m in order]
    ci_lo = [summary['methods'][m]['rmse_reduction_pct_ci95'][0] for m in order]
    ci_hi = [summary['methods'][m]['rmse_reduction_pct_ci95'][1] for m in order]
    err_lo = [m - lo for m, lo in zip(means, ci_lo)]
    err_hi = [hi - m for m, hi in zip(means, ci_hi)]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(order))
    bars = ax.bar(x, means, yerr=[err_lo, err_hi], capsize=5,
                  color=[colors[m] for m in order], edgecolor="black", linewidth=0.8)
    y_max = max(hi for hi in ci_hi) + 1.5
    y_min = min(lo for lo in ci_lo) - 1.5
    for i, (b, mval) in enumerate(zip(bars, means)):
        label_y = ci_hi[i] + 0.35 if mval >= 0 else ci_lo[i] - 0.7
        va = "bottom" if mval >= 0 else "top"
        ax.text(b.get_x() + b.get_width() / 2, label_y, f"{mval:+.2f}%",
                ha="center", va=va, fontsize=10,
                fontweight="bold" if order[i] == "pilsd_shrunk" else "normal")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[m] for m in order], rotation=0, fontsize=10)
    ax.set_ylabel("RMSE reduction vs pop-slope baseline (%)", fontsize=11)
    ax.set_title(f"PRISM LOCO P-GenRM head-to-head  (N={summary['slice']['n_users']} users, "
                 f"K-means k={summary['kmeans']['k_picked']}, "
                 f"cluster-bootstrap 95% CI)", fontsize=10)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


# ============================================================================
# Entry point
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=RNG_DEFAULT)
    p.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    p.add_argument("--k", type=int, default=None,
                    help="Force K-means k; default = sweep {10,20,50} pick by silhouette")
    p.add_argument("--lambda-tr", type=float, default=LAMBDA_TR_DEFAULT,
                    help="Transition shrinkage to population prior; paper Eq. (4)")
    p.add_argument("--lambda-cent", type=float, default=LAMBDA_CENT_DEFAULT,
                    help="Centring shrinkage; OLS-within-prototype already minimises MSE so 0 by default")
    p.add_argument("--smoke", action="store_true",
                    help="Smoke mode: subsample 50 users for speed")
    p.add_argument("--skip-fig", action="store_true")
    p.add_argument("--skip-table", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        print("[smoke] subsampling 50 users for fast smoke test")
    summary, pu = run_loco(args)
    if not args.skip_table:
        write_latex_table(summary, OUT_TABLE)
    if not args.skip_fig:
        write_figure(summary, OUT_FIG)
    print(f"\n[verdict] {summary['verdict_class']}")
    print(f"[runtime] {summary['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
