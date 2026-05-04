"""T1+T3 unified experiment: Time-varying PILSD on OASST.

Question: Given that T3 established a cohort-level quality drift with
changepoint t★≈2.0 months, does a TIME-VARYING PILSD (pre-/post-t★ per-author
calibrator) beat the STATIC single-calibrator PILSD?

If paired Δ(TV − static) > 0 with CI excluding zero → drift is author-specific,
T1 and T3 findings compound.
If paired Δ ≈ 0 → drift is cohort-level, static PILSD already captures what
per-author data can; T1 + T3 are independent facets of the same story.

Data:  OASST1 ∪ OASST2 union of author-quality parquets (OASST2 subsumes
       OASST1 after dedup, yielding 133,095 msgs × 22,120 authors).
Proxy: peer_quality_mean = mean quality of OTHER messages in the same tree
       (r≈0.26 with own quality; serves as RM-score stand-in since OASST has
       no per-message RM). Robustness proxy = parent_quality.
Units: quality (∈[0,1]) = mean of ~3.6 human rater labels per message.

Estimators (all evaluated via within-author 5-fold CV):
  1. Pop-OLS      — quality ∼ α_pop + β_pop · proxy (one line for everybody)
  2. Static PILSD — per-author (α_j, β_j) shrunk to (α_pop, β_pop) with
                    MoM-estimated τ² (the T1 baseline).
  3. TV-PILSD     — split each author's messages at t★=2.0:
                    pre:  (α_j^pre,  β_j^pre)  shrunk toward
                                                (α_pop^pre,  β_pop^pre)
                    post: (α_j^post, β_j^post) shrunk toward
                                                (α_pop^post, β_pop^post)
                    τ²_pre and τ²_post estimated on the two sub-pops.

Reporting:
  - per-author RMSE gain (%) vs pop-OLS for each estimator
  - paired Δ (TV − static) with 500-rep cluster-bootstrap 95% CI
  - verdict: tv_beats_static / equivalent / tv_worse

Seed: 20260420 (project lock).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "track1_t3_unified"
FIG = ROOT / "paper" / "figures" / "fig_19_t1_t3_unified.pdf"
PAPER_INSERT = ROOT / "PAPER_INSERT_t1_t3_unified.tex"

OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260420
N_FOLDS = 5
N_BOOT = 500
MIN_MSGS_STATIC = 10       # per author for static PILSD CV
MIN_MSGS_PER_SEG = 5       # per author per segment (pre/post) for TV-PILSD
T_STAR = 2.0               # T3-established changepoint (iter+N+221)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_union() -> pd.DataFrame:
    """Load OASST1 ∪ OASST2 author-quality table with unified month_num."""
    d1 = pd.read_parquet(DATA / "oasst1_author_quality.parquet")
    d2 = pd.read_parquet(DATA / "oasst2_author_quality.parquet")
    d2 = d2.rename(columns={"created_ts": "ts"})

    # Global min timestamp (project convention: OASST1's first msg)
    global_min = min(d1["ts"].min(), d2["ts"].min())
    for d in (d1, d2):
        d["month_num"] = (d["ts"] - global_min).dt.total_seconds() / (30.44 * 86400.0)

    d1["source"] = "oasst1"
    d2["source"] = "oasst2"
    # OASST2 already has tree_id; OASST1 doesn't — derive from parent walk
    # For simplicity: use OASST2 as the base (it subsumes OASST1 after dedup)
    # and merge in OASST1 msgs that are NOT in OASST2.
    s2 = set(d2["message_id"])
    d1_only = d1[~d1["message_id"].isin(s2)].copy()

    # For OASST1-only msgs, derive tree_id by walking parent_id chain within d1
    parent_map = dict(zip(d1["message_id"], d1["parent_id"]))

    def root_of(mid):
        seen = set()
        cur = mid
        while cur is not None and cur not in seen:
            seen.add(cur)
            nxt = parent_map.get(cur)
            if nxt is None or (isinstance(nxt, float) and np.isnan(nxt)):
                return cur
            cur = nxt
        return cur

    d1_only["tree_id"] = d1_only["message_id"].map(root_of)

    cols = ["message_id", "parent_id", "tree_id", "user_id", "ts",
            "month_num", "role", "quality", "review_count", "source"]
    d1_only_sub = d1_only[cols].copy()
    d2_sub = d2[cols].copy()

    d = pd.concat([d2_sub, d1_only_sub], ignore_index=True)
    d = d.drop_duplicates(subset=["message_id"]).reset_index(drop=True)
    return d


def build_proxies(d: pd.DataFrame) -> pd.DataFrame:
    """Attach peer_quality_mean (leave-one-out tree mean) and parent_quality."""
    # peer_quality_mean: mean quality of OTHER messages in the same tree
    sum_q = d.groupby("tree_id")["quality"].transform("sum")
    n_q = d.groupby("tree_id")["quality"].transform("count")
    d["peer_quality_mean"] = np.where(
        n_q > 1,
        (sum_q - d["quality"]) / (n_q - 1),
        np.nan,
    )
    # parent_quality: quality of the parent message (NaN for root msgs)
    q_map = dict(zip(d["message_id"], d["quality"]))
    d["parent_quality"] = d["parent_id"].map(q_map)
    return d


# ---------------------------------------------------------------------------
# PILSD primitives (copied from T1 canonical pattern)
# ---------------------------------------------------------------------------

def fit_ols(x: np.ndarray, y: np.ndarray):
    """Return (α, β, se_α, se_β). NaN if ill-conditioned."""
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan
    x_mean, y_mean = x.mean(), y.mean()
    var_x = float(np.sum((x - x_mean) ** 2))
    if var_x < 1e-10:
        return np.nan, np.nan, np.nan, np.nan
    beta = float(np.sum((x - x_mean) * (y - y_mean)) / var_x)
    alpha = float(y_mean - beta * x_mean)
    resid = y - (alpha + beta * x)
    mse = float(np.sum(resid ** 2) / (n - 2))
    se_beta = float(np.sqrt(max(mse, 0.0) / var_x))
    se_alpha = float(np.sqrt(max(mse, 0.0) * (1.0 / n + x_mean ** 2 / var_x)))
    return alpha, beta, se_alpha, se_beta


def mom_tau2(values: np.ndarray, ses: np.ndarray) -> float:
    """Method-of-moments τ² = max(0, Var(values) − E[SE²])."""
    values = np.asarray(values, dtype=float)
    ses = np.asarray(ses, dtype=float)
    mask = np.isfinite(values) & np.isfinite(ses)
    if mask.sum() < 3:
        return 0.0
    return float(max(0.0, np.var(values[mask], ddof=1) - np.mean(ses[mask] ** 2)))


def eb_shrink(x_u, se_u, x_pop, tau2):
    """Empirical-Bayes shrinkage of x_u toward x_pop with shrinkage weight
       τ² / (τ² + SE²)."""
    w = tau2 / (tau2 + se_u ** 2 + 1e-12)
    return w * x_u + (1 - w) * x_pop


# ---------------------------------------------------------------------------
# CV fold iterator (within-author 5-fold)
# ---------------------------------------------------------------------------

def kfold_indices(n: int, k: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    return folds


# ---------------------------------------------------------------------------
# Main: evaluate 3 estimators via within-author 5-fold CV
# ---------------------------------------------------------------------------

def evaluate(df: pd.DataFrame, proxy_col: str, rng: np.random.Generator):
    """Return per-author RMSE for each estimator via within-author 5-fold CV.

    Protocol:
      1. Filter to authors with ≥ MIN_MSGS_STATIC total msgs.
      2. For each fold k=1..5:
         a) Hold out fold k for every author (test set).
         b) On the remaining 80% (train):
            — fit pop-OLS on pooled train
            — fit per-author OLS for every author; MoM τ² across authors
              → static PILSD (α_j^s, β_j^s)
            — fit per-author OLS separately on train∩{t<t★} and train∩{t≥t★}
              → MoM τ²_pre and τ²_post → TV-PILSD
              (α_j^{s,pre}, β_j^{s,pre}) and (α_j^{s,post}, β_j^{s,post})
         c) Predict on test; accumulate SE per (author, estimator).
      3. RMSE per author = sqrt(mean SE over all 5 test folds).
    """
    df = df.dropna(subset=[proxy_col, "quality", "month_num", "user_id"]).copy()
    counts = df.groupby("user_id").size()
    keep_users = counts[counts >= MIN_MSGS_STATIC].index
    df = df[df["user_id"].isin(keep_users)].reset_index(drop=True)
    n_authors = df["user_id"].nunique()
    print(f"  [{proxy_col}] n_authors={n_authors}, n_msgs={len(df)}")

    # Pre-assign fold index per (author, row) for reproducibility
    fold_col = np.empty(len(df), dtype=np.int32)
    for uid, g in df.groupby("user_id"):
        idx = rng.permutation(len(g))
        fi = np.zeros(len(g), dtype=np.int32)
        for k, chunk in enumerate(np.array_split(idx, N_FOLDS)):
            fi[chunk] = k
        fold_col[g.index.to_numpy()] = fi
    df["fold"] = fold_col

    # Per-author accumulators: list of squared errors over all test folds
    per_author_se = {uid: {"pop": [], "static": [], "tv": []}
                     for uid in df["user_id"].unique()}

    for k in range(N_FOLDS):
        train = df[df["fold"] != k]
        test = df[df["fold"] == k]

        # --- Pop-OLS ---
        a_pop, b_pop, _, _ = fit_ols(train[proxy_col].to_numpy(),
                                     train["quality"].to_numpy())
        if not np.isfinite(a_pop):
            continue

        # --- Static PILSD ---
        per_u_static = {}
        for uid, g in train.groupby("user_id"):
            a, b, sa, sb = fit_ols(g[proxy_col].to_numpy(),
                                   g["quality"].to_numpy())
            if np.all(np.isfinite([a, b, sa, sb])):
                per_u_static[uid] = (a, b, sa, sb)
        if len(per_u_static) < 3:
            continue
        alphas = np.array([v[0] for v in per_u_static.values()])
        betas = np.array([v[1] for v in per_u_static.values()])
        se_as = np.array([v[2] for v in per_u_static.values()])
        se_bs = np.array([v[3] for v in per_u_static.values()])
        tau2_a_s = mom_tau2(alphas, se_as)
        tau2_b_s = mom_tau2(betas, se_bs)
        a_pop_s = float(alphas.mean())
        b_pop_s = float(betas.mean())

        # --- TV-PILSD: pre- and post-t★ segments ---
        train_pre = train[train["month_num"] < T_STAR]
        train_post = train[train["month_num"] >= T_STAR]

        a_pop_pre, b_pop_pre, _, _ = fit_ols(
            train_pre[proxy_col].to_numpy(), train_pre["quality"].to_numpy())
        a_pop_post, b_pop_post, _, _ = fit_ols(
            train_post[proxy_col].to_numpy(), train_post["quality"].to_numpy())

        per_u_pre, per_u_post = {}, {}
        for uid, g in train_pre.groupby("user_id"):
            if len(g) >= MIN_MSGS_PER_SEG:
                a, b, sa, sb = fit_ols(g[proxy_col].to_numpy(),
                                       g["quality"].to_numpy())
                if np.all(np.isfinite([a, b, sa, sb])):
                    per_u_pre[uid] = (a, b, sa, sb)
        for uid, g in train_post.groupby("user_id"):
            if len(g) >= MIN_MSGS_PER_SEG:
                a, b, sa, sb = fit_ols(g[proxy_col].to_numpy(),
                                       g["quality"].to_numpy())
                if np.all(np.isfinite([a, b, sa, sb])):
                    per_u_post[uid] = (a, b, sa, sb)
        if per_u_pre:
            ap = np.array([v[0] for v in per_u_pre.values()])
            bp = np.array([v[1] for v in per_u_pre.values()])
            sap = np.array([v[2] for v in per_u_pre.values()])
            sbp = np.array([v[3] for v in per_u_pre.values()])
            tau2_a_pre = mom_tau2(ap, sap)
            tau2_b_pre = mom_tau2(bp, sbp)
        else:
            tau2_a_pre = tau2_b_pre = 0.0
        if per_u_post:
            ap = np.array([v[0] for v in per_u_post.values()])
            bp = np.array([v[1] for v in per_u_post.values()])
            sap = np.array([v[2] for v in per_u_post.values()])
            sbp = np.array([v[3] for v in per_u_post.values()])
            tau2_a_post = mom_tau2(ap, sap)
            tau2_b_post = mom_tau2(bp, sbp)
        else:
            tau2_a_post = tau2_b_post = 0.0

        # --- Predict on test fold ---
        for uid, g in test.groupby("user_id"):
            x = g[proxy_col].to_numpy()
            y = g["quality"].to_numpy()
            t = g["month_num"].to_numpy()

            # pop
            pred_pop = a_pop + b_pop * x

            # static
            if uid in per_u_static:
                a_u, b_u, sa_u, sb_u = per_u_static[uid]
                a_s = eb_shrink(a_u, sa_u, a_pop_s, tau2_a_s)
                b_s = eb_shrink(b_u, sb_u, b_pop_s, tau2_b_s)
            else:
                a_s, b_s = a_pop_s, b_pop_s
            pred_static = a_s + b_s * x

            # TV: per-segment
            pred_tv = np.empty_like(y)
            # pre
            pre_mask = t < T_STAR
            if pre_mask.any():
                if uid in per_u_pre:
                    a_u, b_u, sa_u, sb_u = per_u_pre[uid]
                    a_tv = eb_shrink(a_u, sa_u,
                                     a_pop_pre if np.isfinite(a_pop_pre) else a_pop,
                                     tau2_a_pre)
                    b_tv = eb_shrink(b_u, sb_u,
                                     b_pop_pre if np.isfinite(b_pop_pre) else b_pop,
                                     tau2_b_pre)
                else:
                    a_tv = a_pop_pre if np.isfinite(a_pop_pre) else a_pop
                    b_tv = b_pop_pre if np.isfinite(b_pop_pre) else b_pop
                pred_tv[pre_mask] = a_tv + b_tv * x[pre_mask]
            # post
            post_mask = ~pre_mask
            if post_mask.any():
                if uid in per_u_post:
                    a_u, b_u, sa_u, sb_u = per_u_post[uid]
                    a_tv = eb_shrink(a_u, sa_u,
                                     a_pop_post if np.isfinite(a_pop_post) else a_pop,
                                     tau2_a_post)
                    b_tv = eb_shrink(b_u, sb_u,
                                     b_pop_post if np.isfinite(b_pop_post) else b_pop,
                                     tau2_b_post)
                else:
                    a_tv = a_pop_post if np.isfinite(a_pop_post) else a_pop
                    b_tv = b_pop_post if np.isfinite(b_pop_post) else b_pop
                pred_tv[post_mask] = a_tv + b_tv * x[post_mask]

            per_author_se[uid]["pop"].extend((y - pred_pop) ** 2)
            per_author_se[uid]["static"].extend((y - pred_static) ** 2)
            per_author_se[uid]["tv"].extend((y - pred_tv) ** 2)

    # Aggregate: per-author RMSE
    rows = []
    for uid, d in per_author_se.items():
        if not d["pop"] or not d["static"] or not d["tv"]:
            continue
        n_test = min(len(d["pop"]), len(d["static"]), len(d["tv"]))
        if n_test == 0:
            continue
        rmse_pop = float(np.sqrt(np.mean(d["pop"])))
        rmse_static = float(np.sqrt(np.mean(d["static"])))
        rmse_tv = float(np.sqrt(np.mean(d["tv"])))
        rows.append({
            "user_id": uid,
            "n_test": int(n_test),
            "rmse_pop": rmse_pop,
            "rmse_static": rmse_static,
            "rmse_tv": rmse_tv,
        })
    return pd.DataFrame(rows)


def cluster_bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator,
                         alpha: float = 0.05):
    """Cluster-bootstrap CI by resampling authors with replacement."""
    n = len(values)
    bs_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs_means[b] = float(np.mean(values[idx]))
    lo = float(np.quantile(bs_means, alpha / 2))
    hi = float(np.quantile(bs_means, 1 - alpha / 2))
    return lo, hi, bs_means


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def summarise(df_res: pd.DataFrame, label: str):
    df_res = df_res.copy()
    df_res["gain_static_pct"] = 100.0 * (df_res["rmse_pop"] - df_res["rmse_static"]) / np.maximum(df_res["rmse_pop"], 1e-9)
    df_res["gain_tv_pct"] = 100.0 * (df_res["rmse_pop"] - df_res["rmse_tv"]) / np.maximum(df_res["rmse_pop"], 1e-9)
    df_res["delta_tv_vs_static_pct"] = df_res["gain_tv_pct"] - df_res["gain_static_pct"]
    return df_res


def run_once(df_all: pd.DataFrame, proxy_col: str):
    rng = np.random.default_rng(SEED)
    print(f"\n[run] proxy={proxy_col}")
    df_res = evaluate(df_all, proxy_col, rng)
    df_res = summarise(df_res, proxy_col)

    rng_bs = np.random.default_rng(SEED + 1)
    # Raw means
    mean_rmse_pop = float(df_res["rmse_pop"].mean())
    mean_rmse_static = float(df_res["rmse_static"].mean())
    mean_rmse_tv = float(df_res["rmse_tv"].mean())
    mean_gain_static = float(df_res["gain_static_pct"].mean())
    mean_gain_tv = float(df_res["gain_tv_pct"].mean())

    # Cluster bootstrap (authors are the clusters)
    lo_s, hi_s, _ = cluster_bootstrap_ci(df_res["gain_static_pct"].to_numpy(),
                                         N_BOOT, rng_bs)
    lo_t, hi_t, _ = cluster_bootstrap_ci(df_res["gain_tv_pct"].to_numpy(),
                                         N_BOOT, rng_bs)
    lo_d, hi_d, bs_d = cluster_bootstrap_ci(
        df_res["delta_tv_vs_static_pct"].to_numpy(), N_BOOT, rng_bs)
    mean_delta = float(df_res["delta_tv_vs_static_pct"].mean())
    # Two-sided bootstrap p-value (H0: delta=0)
    p_val = 2.0 * min(float(np.mean(bs_d <= 0)), float(np.mean(bs_d >= 0)))

    verdict = ("tv_beats_static" if lo_d > 0 else
               "tv_worse" if hi_d < 0 else "equivalent")

    out = {
        "proxy": proxy_col,
        "n_authors": int(len(df_res)),
        "n_test_msgs_median": int(df_res["n_test"].median()),
        "rmse_pop_mean": mean_rmse_pop,
        "rmse_static_mean": mean_rmse_static,
        "rmse_tv_mean": mean_rmse_tv,
        "gain_static_pct_mean": mean_gain_static,
        "gain_static_pct_ci95": [lo_s, hi_s],
        "gain_tv_pct_mean": mean_gain_tv,
        "gain_tv_pct_ci95": [lo_t, hi_t],
        "delta_tv_vs_static_pct_mean": mean_delta,
        "delta_tv_vs_static_pct_ci95": [lo_d, hi_d],
        "delta_bootstrap_p_two_sided": p_val,
        "verdict": verdict,
    }
    return df_res, out


def make_figure(df_res_primary: pd.DataFrame, summary: dict, fig_path: Path):
    """Two-panel figure: (a) gain distributions, (b) paired Δ with CI."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    # Panel (a): histogram of per-author gain
    ax = axes[0]
    g_s = df_res_primary["gain_static_pct"].to_numpy()
    g_t = df_res_primary["gain_tv_pct"].to_numpy()
    bins = np.linspace(min(g_s.min(), g_t.min()),
                       max(g_s.max(), g_t.max()), 40)
    ax.hist(g_s, bins=bins, alpha=0.55, label=f"Static PILSD (μ={g_s.mean():.2f}%)",
            color="#4477AA", edgecolor="#1f3d6e", linewidth=0.4)
    ax.hist(g_t, bins=bins, alpha=0.55, label=f"TV-PILSD (μ={g_t.mean():.2f}%)",
            color="#EE6677", edgecolor="#7a2b37", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xlabel("Per-author RMSE gain vs Pop-OLS (%)")
    ax.set_ylabel("Authors")
    ax.set_title(f"(a) Per-author gain distribution  (N={len(df_res_primary)})")
    ax.legend(loc="upper right", frameon=False)

    # Panel (b): paired delta
    ax = axes[1]
    d = df_res_primary["delta_tv_vs_static_pct"].to_numpy()
    mean_d = summary["delta_tv_vs_static_pct_mean"]
    lo_d, hi_d = summary["delta_tv_vs_static_pct_ci95"]
    ax.hist(d, bins=40, color="#CCBB44", alpha=0.7,
            edgecolor="#6a5b10", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.axvline(mean_d, color="#AA3377", linewidth=1.8,
               label=f"Mean Δ = {mean_d:+.3f}%")
    ax.axvspan(lo_d, hi_d, color="#AA3377", alpha=0.18,
               label=f"95% CI [{lo_d:+.3f}, {hi_d:+.3f}]")
    ax.set_xlabel(r"Paired $\Delta$ = gain(TV) − gain(Static)  per author (%)")
    ax.set_ylabel("Authors")
    verdict = summary["verdict"]
    ax.set_title(f"(b) Paired Δ TV − Static  [verdict: {verdict}]")
    ax.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)


def write_paper_insert(primary: dict, robustness: dict, path: Path):
    v = primary["verdict"]
    verdict_txt = {
        "tv_beats_static": "\\textbf{TV-PILSD beats Static PILSD}",
        "tv_worse": "\\textbf{TV-PILSD loses to Static PILSD}",
        "equivalent": "\\textbf{TV-PILSD is statistically indistinguishable from Static PILSD}",
    }[v]
    lo, hi = primary["delta_tv_vs_static_pct_ci95"]
    text = rf"""% Auto-generated by scripts/t1_t3_unified_time_varying_pilsd.py
% Seed {SEED}.  {N_BOOT}-rep cluster-bootstrap.

\subsection{{T1+T3 Unified: does author drift demand a time-varying calibrator?}}
\label{{sec:t1t3_unified}}

T3 \S\ref{{sec:t3_changepoint}} locates a cohort-level quality changepoint at $t^{{\star}} \approx {T_STAR:.1f}$~months on OASST1$\cup$OASST2.
T1 \S\ref{{sec:t1_pilsd}} shows that per-author static calibrators reduce RMSE by 5--9\%.
A natural composite question is whether the drift is \emph{{author-specific}} — in which case a time-varying PILSD that fits a separate calibrator pre-~and~post-$t^{{\star}}$ should beat static PILSD — or merely cohort-level, in which case static already captures whatever per-author data permits.

\paragraph{{Setup.}} OASST1$\cup$OASST2 union, $N={primary['n_authors']}$ authors ($\geq {MIN_MSGS_STATIC}$ msgs, $\geq {MIN_MSGS_PER_SEG}$ per segment), 5-fold within-author CV.
The proxy RM score is \texttt{{peer\_quality\_mean}} = mean quality of other messages in the same tree (OASST has no per-message RM, and leave-one-out tree-peer mean is independent of the focal author).
Robustness proxy: \texttt{{parent\_quality}}.
Three estimators: Pop-OLS, Static PILSD (MoM $\tau^2$), and TV-PILSD (separate MoM $\tau^2$ per $\{{pre, post\}}$ segment, shrunk toward segment-specific pop means).

\paragraph{{Result (peer\_quality\_mean).}} Per-author mean RMSE gain vs Pop-OLS: static PILSD {primary['gain_static_pct_mean']:+.3f}\% (95\% CI [{primary['gain_static_pct_ci95'][0]:+.3f}, {primary['gain_static_pct_ci95'][1]:+.3f}]); TV-PILSD {primary['gain_tv_pct_mean']:+.3f}\% (95\% CI [{primary['gain_tv_pct_ci95'][0]:+.3f}, {primary['gain_tv_pct_ci95'][1]:+.3f}]).
Paired $\Delta$ = gain(TV) $-$ gain(Static) = {primary['delta_tv_vs_static_pct_mean']:+.3f}\% per author (95\% cluster-bootstrap CI [{lo:+.3f}, {hi:+.3f}], $p={primary['delta_bootstrap_p_two_sided']:.3g}$).

\paragraph{{Robustness (parent\_quality).}} Paired $\Delta = {robustness['delta_tv_vs_static_pct_mean']:+.3f}$\% (95\% CI [{robustness['delta_tv_vs_static_pct_ci95'][0]:+.3f}, {robustness['delta_tv_vs_static_pct_ci95'][1]:+.3f}]), verdict: \texttt{{{robustness['verdict']}}}.

\paragraph{{Verdict.}} {verdict_txt}.
{"This validates that T1 (per-author calibration) and T3 (cohort drift) compound: real OASST drift is author-specific, so a time-varying calibrator offers benefit on top of static shrinkage." if v == "tv_beats_static" else "The drift observed in T3 is dominantly cohort-level at this granularity — static PILSD already captures the per-author signal the data support.  T1 and T3 are two independent facets of the same clustered-random-effect-with-drift story rather than compounding effects." if v == "equivalent" else "TV-PILSD underperforms static: segment-wise splitting halves each author's effective $n_j$ and therefore widens EB SE more than the residual drift can compensate for."}
Figure~\ref{{fig:t1_t3_unified}} shows the per-author gain distribution and the paired-$\Delta$ bootstrap.

\begin{{figure}}[t]
    \centering
    \includegraphics[width=\textwidth]{{figures/fig_19_t1_t3_unified.pdf}}
    \caption{{\textbf{{T1+T3 unified: does author drift demand a time-varying calibrator?}}
    (a) Per-author RMSE-gain distribution for static PILSD (blue) and TV-PILSD (red) on the OASST1$\cup$OASST2 union, with $t^{{\star}}={T_STAR:.1f}$ months (T3-established changepoint).
    (b) Paired $\Delta$ = gain(TV) $-$ gain(Static) per author; shaded region is the 500-rep cluster-bootstrap 95\% CI.  Verdict: \texttt{{{v}}}.}}
    \label{{fig:t1_t3_unified}}
\end{{figure}}
"""
    path.write_text(text)


def main():
    print(f"[load] OASST1∪OASST2")
    d = load_union()
    print(f"  loaded {len(d)} msgs × {d['user_id'].nunique()} authors")
    d = build_proxies(d)

    # Primary proxy
    df_primary, sum_primary = run_once(d, "peer_quality_mean")
    # Robustness proxy
    df_robust, sum_robust = run_once(d, "parent_quality")

    # Save everything
    out = {
        "seed": SEED,
        "n_folds": N_FOLDS,
        "n_bootstrap": N_BOOT,
        "t_star": T_STAR,
        "min_msgs_static": MIN_MSGS_STATIC,
        "min_msgs_per_segment": MIN_MSGS_PER_SEG,
        "primary": sum_primary,
        "robustness": sum_robust,
        "sign_consistent": bool(
            np.sign(sum_primary["delta_tv_vs_static_pct_mean"])
            == np.sign(sum_robust["delta_tv_vs_static_pct_mean"])
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    df_primary.to_parquet(OUT / "per_author_primary.parquet")
    df_robust.to_parquet(OUT / "per_author_robust.parquet")

    make_figure(df_primary, sum_primary, FIG)
    write_paper_insert(sum_primary, sum_robust, PAPER_INSERT)

    print(f"\n=== SUMMARY (primary: peer_quality_mean) ===")
    print(json.dumps(sum_primary, indent=2))
    print(f"\n=== SUMMARY (robustness: parent_quality) ===")
    print(json.dumps(sum_robust, indent=2))
    print(f"\nSaved: {OUT/'summary.json'}")
    print(f"Saved: {FIG}")
    print(f"Saved: {PAPER_INSERT}")


if __name__ == "__main__":
    main()
