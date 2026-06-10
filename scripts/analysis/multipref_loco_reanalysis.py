"""MultiPref LOCO-Q (leave-one-comparison-out) re-analysis.

Like PluriHarms, MultiPref has single-rating-per-(evaluator,
comparison) structure. If our hypothesis is right ("LOCO correction is
dataset-specific, driven by conversational prompt recurrence"), MultiPref
should ALSO show LOCO-Q ≈ random-fold gain, with no leakage factor.

Target: rating = 0-100 `quality` proxy = mean_conf (evaluator confidence).
Regressor: overall_conf (evaluator's overall confidence on the comparison).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = (Path(__file__).resolve().parents[2] / "3_PEBS_Standalone/data/multipref_evaluator_quality.parquet")
OUT = (Path(__file__).resolve().parents[2] / "3_PEBS_Standalone/results/track1_multipref_loco_q")
N_BOOT = 500
RNG = 20260420


def fit_ols(x, y):
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan
    x_mean, y_mean = x.mean(), y.mean()
    var_x = np.sum((x - x_mean) ** 2)
    if var_x < 1e-10:
        return np.nan, np.nan, np.nan, np.nan
    beta = float(np.sum((x - x_mean) * (y - y_mean)) / var_x)
    alpha = float(y_mean - beta * x_mean)
    n = len(x)
    if n <= 2:
        return alpha, beta, np.nan, np.nan
    resid = y - (alpha + beta * x)
    mse = float(np.sum(resid ** 2)) / (n - 2)
    se_beta = float(np.sqrt(mse / var_x))
    se_alpha = float(np.sqrt(mse * (1.0 / n + x_mean ** 2 / var_x)))
    return alpha, beta, se_alpha, se_beta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG)

    df = pd.read_parquet(DATA).dropna(subset=["overall_conf", "quality", "evaluator"]).copy()
    df["evaluator"] = df["evaluator"].astype(str)
    print(f"[load] {len(df)} ratings × {df['evaluator'].nunique()} evaluators")

    # Per-evaluator OLS: quality ~ overall_conf
    per_eval = {}
    for eid, g in df.groupby("evaluator"):
        if len(g) < 10:
            continue
        a, b, sa, sb = fit_ols(g["overall_conf"].to_numpy(dtype=np.float64),
                                g["quality"].to_numpy(dtype=np.float64))
        if np.all(np.isfinite([a, b, sa, sb])):
            per_eval[eid] = (a, b, sa, sb)
    alphas = np.array([v[0] for v in per_eval.values()])
    betas = np.array([v[1] for v in per_eval.values()])
    se_as = np.array([v[2] for v in per_eval.values()])
    se_bs = np.array([v[3] for v in per_eval.values()])
    a_pop = float(alphas.mean())
    b_pop = float(betas.mean())
    tau2_a = max(0.0, float(np.var(alphas, ddof=1) - np.mean(se_as ** 2)))
    tau2_b = max(0.0, float(np.var(betas, ddof=1) - np.mean(se_bs ** 2)))
    print(f"[MoM] α_pop={a_pop:.3f}  β_pop={b_pop:.3f}  τ²_α={tau2_a:.3f}  τ²_β={tau2_b:.3f}")

    # LOCO-Q per evaluator (leave one comparison out)
    per_eval_gains = []
    for eid, ev_df in df.groupby("evaluator"):
        if len(ev_df) < 20:
            continue
        comparisons = ev_df["comparison_id"].unique()
        sq_pop, sq_pebs = [], []
        for cid in comparisons:
            train = ev_df[ev_df["comparison_id"] != cid]
            test = ev_df[ev_df["comparison_id"] == cid]
            if len(train) < 10 or len(test) < 1:
                continue
            a_e, b_e, sa_e, sb_e = fit_ols(train["overall_conf"].to_numpy(),
                                            train["quality"].to_numpy())
            if not np.all(np.isfinite([a_e, b_e, sa_e, sb_e])):
                continue
            w_a = tau2_a / (tau2_a + sa_e ** 2 + 1e-12)
            w_b = tau2_b / (tau2_b + sb_e ** 2 + 1e-12)
            a_s = w_a * a_e + (1 - w_a) * a_pop
            b_s = w_b * b_e + (1 - w_b) * b_pop
            x_te = test["overall_conf"].to_numpy(dtype=np.float64)
            y_te = test["quality"].to_numpy(dtype=np.float64)
            pred_pop = a_pop + b_pop * x_te
            pred_pebs = a_s + b_s * x_te
            sq_pop.append(float(np.sum((y_te - pred_pop) ** 2)))
            sq_pebs.append(float(np.sum((y_te - pred_pebs) ** 2)))
        if not sq_pop:
            continue
        rmse_pop = float(np.sqrt(sum(sq_pop) / len(ev_df)))
        rmse_pebs = float(np.sqrt(sum(sq_pebs) / len(ev_df)))
        if rmse_pop > 1e-6:
            per_eval_gains.append(100.0 * (rmse_pop - rmse_pebs) / rmse_pop)

    gains = np.array(per_eval_gains)
    mean = float(gains.mean())
    boots = np.array([rng.choice(gains, len(gains), replace=True).mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\n[MultiPref LOCO-Q PEBS-shrunk vs pop-OLS] gain = {mean:+.3f}%  CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"[n_evaluators] {len(gains)}")

    summary = {
        "protocol": "LOCO-Q per evaluator (leave one comparison out)",
        "n_evaluators": int(len(gains)),
        "alpha_pop": a_pop, "beta_pop": b_pop,
        "tau2_alpha": tau2_a, "tau2_beta": tau2_b,
        "mean_gain_pct": mean,
        "ci95": [float(lo), float(hi)],
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "n_bootstrap": N_BOOT,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT}/summary.json")


if __name__ == "__main__":
    main()
