"""PEBS RMSE gain stratified by PRISM conversation type.

Sibling to the pair-acc-by-conv-type experiment, but tests the
calibration-sensitive RMSE metric (where PEBS's mechanism has leverage
per Prop T1.MI) instead of rank-invariant pair-acc.

Workflow:
  1. Load prism_rm_scored.parquet (68k utterances) — (user_id, conversation_id, rm_score, score_user)
  2. Load prism_user_calibrators_shrunk.parquet — per-user (α_j, β_j) shrunk
  3. For each utterance compute:
       pebs_score = α_j + β_j * rm_score
       naive_score = α_pop + β_pop * rm_score     (from naive OLS pop means)
  4. RMSE vs score_user, stratify by (user_id, conv_type)
  5. Per-user gain = RMSE_naive - RMSE_pebs ; average by conv_type
  6. Cluster-bootstrap over users (2000 reps) for CIs
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

T1 = (Path(__file__).resolve().parents[2] / "1_Causal_RLHF")
OUT = (Path(__file__).resolve().parents[2] / "3_PEBS_Standalone/results/track1_rmse_by_convtype")
N_BOOT = 2000
RNG = 20260420


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scored = pd.read_parquet(T1 / "data/prism_rm_scored.parquet")
    calib = pd.read_parquet(T1 / "data/prism_user_calibrators_shrunk.parquet")
    meta = pd.read_parquet(T1 / "data/_prism_conversation_meta.parquet")

    # score_user is 0..100; rm_score is real-valued. Need to find population OLS.
    df = scored.merge(calib, on="user_id", how="inner")
    df = df.dropna(subset=["score_user", "rm_score", "alpha_j", "beta_j"])
    conv_to_type = meta.set_index("conversation_id")["conversation_type"]
    df["conv_type"] = df["conversation_id"].map(conv_to_type)
    df = df.dropna(subset=["conv_type"])
    print(f"[merge] {len(df)} scored utterances × {df['user_id'].nunique()} users × 3 conv_types")

    # Population OLS fit on whole cohort (shared baseline)
    x = df["rm_score"].to_numpy(dtype=np.float64)
    y = df["score_user"].to_numpy(dtype=np.float64)
    x_mean, y_mean = x.mean(), y.mean()
    beta_pop = float(np.sum((x - x_mean) * (y - y_mean)) / max(np.sum((x - x_mean) ** 2), 1e-12))
    alpha_pop = float(y_mean - beta_pop * x_mean)
    print(f"[pop-OLS] α={alpha_pop:.2f}, β={beta_pop:.4f}")

    df["score_naive"] = alpha_pop + beta_pop * df["rm_score"]
    df["score_pebs"] = df["alpha_j"] + df["beta_j"] * df["rm_score"]
    df["sq_err_naive"] = (df["score_user"] - df["score_naive"]) ** 2
    df["sq_err_pebs"] = (df["score_user"] - df["score_pebs"]) ** 2

    def per_user_rmse_gain(sub: pd.DataFrame) -> pd.DataFrame:
        g = sub.groupby("user_id", observed=True).agg(
            rmse_naive=("sq_err_naive", lambda v: float(np.sqrt(v.mean()))),
            rmse_pebs=("sq_err_pebs", lambda v: float(np.sqrt(v.mean()))),
            n_obs=("sq_err_naive", "size"),
        )
        g["gain_abs"] = g["rmse_naive"] - g["rmse_pebs"]
        g["gain_pct"] = 100.0 * g["gain_abs"] / g["rmse_naive"]
        return g.reset_index()

    results = []
    rng = np.random.default_rng(RNG)
    for ctype, sub in df.groupby("conv_type", observed=True):
        per_u = per_user_rmse_gain(sub)
        users = per_u["user_id"].to_numpy()
        gains_abs = per_u["gain_abs"].to_numpy(dtype=np.float64)
        gains_pct = per_u["gain_pct"].to_numpy(dtype=np.float64)

        mean_abs = float(np.mean(gains_abs))
        mean_pct = float(np.mean(gains_pct))

        # Cluster-boot over users
        n = len(users)
        boots_abs = np.empty(N_BOOT, dtype=np.float64)
        boots_pct = np.empty(N_BOOT, dtype=np.float64)
        rng_c = np.random.default_rng(RNG + hash(ctype) % 10000)
        for b in range(N_BOOT):
            idx = rng_c.integers(0, n, size=n)
            boots_abs[b] = float(gains_abs[idx].mean())
            boots_pct[b] = float(gains_pct[idx].mean())

        lo_abs, hi_abs = np.percentile(boots_abs, [2.5, 97.5])
        lo_pct, hi_pct = np.percentile(boots_pct, [2.5, 97.5])

        results.append({
            "conv_type": ctype,
            "n_utterances": int(len(sub)),
            "n_users": int(n),
            "mean_rmse_naive": float(np.sqrt(sub["sq_err_naive"].mean())),
            "mean_rmse_pebs": float(np.sqrt(sub["sq_err_pebs"].mean())),
            "per_user_mean_gain_abs": mean_abs,
            "per_user_mean_gain_pct": mean_pct,
            "ci95_abs": [float(lo_abs), float(hi_abs)],
            "ci95_pct": [float(lo_pct), float(hi_pct)],
        })

    print("\n=== PEBS RMSE gain by conversation type ===")
    print(f"{'type':24s} {'n_utt':>7s} {'n_user':>7s} {'RMSE_naive':>11s} {'RMSE_pebs':>11s} "
          f"{'gain%':>7s} {'CI95_gain%':>20s}")
    for r in results:
        print(f"{r['conv_type']:24s} {r['n_utterances']:>7d} {r['n_users']:>7d}  "
              f"{r['mean_rmse_naive']:>10.3f}  {r['mean_rmse_pebs']:>10.3f}  "
              f"{r['per_user_mean_gain_pct']:>6.2f}%  "
              f"[{r['ci95_pct'][0]:+.2f}%, {r['ci95_pct'][1]:+.2f}%]")

    all_positive = all(r["ci95_pct"][0] > 0 for r in results)
    print(f"\n[robustness] all 3 types have CI entirely above zero: {all_positive}")

    summary = {
        "per_conv_type": results,
        "alpha_pop": alpha_pop,
        "beta_pop": beta_pop,
        "all_positive": bool(all_positive),
        "n_bootstrap": N_BOOT,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT}/summary.json")


if __name__ == "__main__":
    main()
