"""OASST1+OASST2 14-month drift stratified by author activity quintile.

Extension of iter+N+199 robust re-analysis: does the +8.75e-3/mo drift
concentrate in high-activity authors (who might be "anchoring" the
FWL-OLS slope by sheer weight), or is it distributed uniformly across
the activity distribution?

Method: power-author filter n_msgs ≥ 10 (3,486 authors), stratify into
5 quintiles by per-author message count, fit FWL-OLS within-author-
demeaned β̂ per quintile with 500-rep cluster-bootstrap.

Output: results/track3_oasst_union_by_activity/summary.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = (Path(__file__).resolve().parents[2] / "3_PILSD_Standalone/data")
OUT = (Path(__file__).resolve().parents[2] / "3_PILSD_Standalone/results/track3_oasst_union_by_activity")

N_BOOT = 500
RNG = 20260420


def unify() -> pd.DataFrame:
    d1 = pd.read_parquet(DATA / "oasst1_author_quality.parquet")
    d2 = pd.read_parquet(DATA / "oasst2_author_quality.parquet")
    t0 = min(d1["ts"].min(), d2["created_ts"].min())
    d1 = d1.assign(month_num_abs=(d1["ts"] - t0).dt.total_seconds() / (30.44 * 86400))
    d2 = d2.assign(month_num_abs=(d2["created_ts"] - t0).dt.total_seconds() / (30.44 * 86400))
    keep = ["user_id", "month_num_abs", "quality"]
    u = pd.concat([d1[keep], d2[keep]], ignore_index=True)
    u = u.dropna(subset=["quality", "month_num_abs"])
    counts = u.groupby("user_id").size()
    keep_users = counts[counts >= 10].index
    return u[u["user_id"].isin(keep_users)].copy().reset_index(drop=True), counts[keep_users]


def fwl_ols(y, x):
    return float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))


def cluster_bootstrap_fwl(df_sub, n_boot, rng):
    """Cluster-bootstrap within-user FWL-OLS β̂ on a subset df."""
    y = df_sub["quality"].to_numpy(dtype=np.float64)
    x = df_sub["month_num_abs"].to_numpy(dtype=np.float64)
    uidx, _ = pd.factorize(df_sub["user_id"], sort=False)
    counts = np.bincount(uidx)
    y_mean = np.bincount(uidx, weights=y) / np.maximum(counts, 1)
    x_mean = np.bincount(uidx, weights=x) / np.maximum(counts, 1)
    y_dem = y - y_mean[uidx]
    x_dem = x - x_mean[uidx]

    n_u = uidx.max() + 1
    groups = {u: np.where(uidx == u)[0] for u in range(n_u)}

    out = np.empty(n_boot, dtype=np.float64)
    beta_obs = fwl_ols(y_dem, x_dem)
    for b in range(n_boot):
        s = rng.integers(0, n_u, size=n_u)
        rows = np.concatenate([groups[u] for u in s])
        yd = y_dem[rows]; xd = x_dem[rows]
        # Re-demean within the bootstrap sample (cluster-bootstrap best practice)
        try:
            out[b] = fwl_ols(yd, xd)
        except Exception:
            out[b] = np.nan
    return beta_obs, out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG)

    df, counts = unify()
    print(f"[load] {len(df)} msgs, {counts.shape[0]} power-authors")

    # Quintile bins by per-author message count
    q_cuts = counts.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).to_numpy()
    print(f"[quintile cutoffs (msg count)] {q_cuts}")

    user_count_df = counts.reset_index()
    user_count_df.columns = ["user_id", "n_msg"]
    user_count_df["q_msg"] = pd.qcut(user_count_df["n_msg"], q=5, labels=False, duplicates="drop")
    df = df.merge(user_count_df, on="user_id", how="inner")

    overall_beta, overall_boot = cluster_bootstrap_fwl(df, N_BOOT, np.random.default_rng(RNG))
    overall_lo, overall_hi = np.percentile(overall_boot, [2.5, 97.5])
    print(f"\n[overall] β̂ = {overall_beta:+.4e}  CI95 = [{overall_lo:+.4e}, {overall_hi:+.4e}]")

    results = []
    for q, g in df.groupby("q_msg"):
        n_auth = g["user_id"].nunique()
        n_msg = len(g)
        auth_msg = g.groupby("user_id").size()
        beta, boot = cluster_bootstrap_fwl(g, N_BOOT, np.random.default_rng(RNG + int(q)))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        entry = {
            "quintile": int(q) + 1,
            "n_authors": int(n_auth),
            "n_messages": int(n_msg),
            "msg_per_author_range": [int(auth_msg.min()), int(auth_msg.max())],
            "msg_per_author_median": float(auth_msg.median()),
            "beta": float(beta),
            "ci95_lo": float(lo),
            "ci95_hi": float(hi),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
        }
        results.append(entry)
        print(f"  Q{int(q)+1}  n_auth={n_auth:>4d}  msgs={n_msg:>6d}  "
              f"range=[{auth_msg.min():>4d}, {auth_msg.max():>5d}]  "
              f"β̂={beta:+.4e}  CI [{lo:+.4e}, {hi:+.4e}]  "
              f"{'*' if (lo > 0 or hi < 0) else ''}")

    # Does beta grow / shrink / stay flat across activity quintiles?
    from scipy import stats
    betas = np.array([r["beta"] for r in results])
    q_ids = np.array([r["quintile"] for r in results])
    rho, p = stats.spearmanr(q_ids, betas)
    print(f"\n[Spearman activity-quintile vs β̂] ρ = {rho:+.3f}  p = {p:.3f}")

    summary = {
        "corpus": "OASST1+OASST2 union, 14mo",
        "overall": {"beta": overall_beta, "ci95": [overall_lo, overall_hi]},
        "quintiles": results,
        "spearman_activity_vs_beta": {"rho": float(rho), "p": float(p)},
        "n_ci_excluding_zero": int(sum(r["ci_excludes_zero"] for r in results)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] {summary['n_ci_excluding_zero']}/5 quintiles have CI excluding zero")
    print(f"Wrote {OUT}/summary.json")


if __name__ == "__main__":
    main()
