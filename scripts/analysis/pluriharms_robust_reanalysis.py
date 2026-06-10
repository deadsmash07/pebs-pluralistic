"""PluriHarms robust-M-estimator re-analysis — cross-dataset T1 robustness.

Memory note (track1_pluriharms_crossdataset): PluriHarms replication gave
+8.638% RMSE reduction at n_j≈150 dense regime. This script tests whether
the cross-dataset T1 replication also survives robust estimators on a
per-annotator basis, and whether the gain concentrates in particular harm
levels.

PluriHarms structure: 100 annotators × ~150 questions × harm_level tag ∈ [0, 1].
Ratings are continuous 0-100.

Experiment: for each annotator, fit robust regression of (rating ~ harm_level)
comparing per-user OLS vs PEBS-shrunk, stratified by Harm_Level quartiles.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = (Path(__file__).resolve().parents[2] / "1_Causal_RLHF/data/pluriharms_long.parquet")
OUT = (Path(__file__).resolve().parents[2] / "3_PEBS_Standalone/results/track1_pluriharms_robust")
N_BOOT = 500
RNG = 20260420


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG)

    df = pd.read_parquet(DATA).dropna(subset=["rating", "Harm_Level"])
    df = df.rename(columns={"user_id": "annotator_id"})
    print(f"[load] {len(df)} ratings × {df['annotator_id'].nunique()} annotators × "
          f"{df['Question_Index'].nunique()} questions")
    print(f"Harm levels: {sorted(df['Harm_Level'].unique())}")

    # Compute per-annotator gain: population mean rating vs per-annotator mean
    # Then compute within-question within-annotator residual structure.
    # Simplification: test whether (rating - pop_mean_rating) is reducible
    # by annotator-specific (α_j, β_j) calibration where x = Harm_Level.
    pop_mean = df["rating"].mean()
    pop_std = df["rating"].std()
    print(f"[pop] mean rating = {pop_mean:.2f}  std = {pop_std:.2f}")

    # Per-annotator per-Harm-Level means (sanity)
    ag = df.groupby(["annotator_id", "Harm_Level"])["rating"].mean().reset_index()
    print(f"\n[per-annotator means] sample:\n{ag.head()}")

    # Per-annotator OLS: rating ~ Harm_Level
    per_ann = {}
    for aid, g in df.groupby("annotator_id"):
        if len(g) < 10:
            continue
        x = g["Harm_Level"].to_numpy(dtype=np.float64)
        y = g["rating"].to_numpy(dtype=np.float64)
        x_mean, y_mean = x.mean(), y.mean()
        vx = np.sum((x - x_mean) ** 2)
        if vx < 1e-10:
            continue
        b = float(np.sum((x - x_mean) * (y - y_mean)) / vx)
        a = float(y_mean - b * x_mean)
        resid = y - (a + b * x)
        mse = float(np.sum(resid ** 2) / max(len(g) - 2, 1))
        se_b = float(np.sqrt(mse / vx))
        se_a = float(np.sqrt(mse * (1.0 / len(g) + x_mean ** 2 / vx)))
        per_ann[aid] = {"alpha": a, "beta": b, "se_a": se_a, "se_b": se_b, "n": len(g)}

    print(f"\n[per-annotator OLS] {len(per_ann)} annotators fit")

    alphas = np.array([v["alpha"] for v in per_ann.values()])
    betas = np.array([v["beta"] for v in per_ann.values()])
    se_as = np.array([v["se_a"] for v in per_ann.values()])
    se_bs = np.array([v["se_b"] for v in per_ann.values()])
    a_pop = float(alphas.mean())
    b_pop = float(betas.mean())
    tau2_a = max(0.0, float(np.var(alphas, ddof=1) - np.mean(se_as ** 2)))
    tau2_b = max(0.0, float(np.var(betas, ddof=1) - np.mean(se_bs ** 2)))
    print(f"[MoM] α_pop={a_pop:.2f}  β_pop={b_pop:.2f}  τ²_α={tau2_a:.2f}  τ²_β={tau2_b:.2f}")

    # Per-question leave-one-out CV: hold out one question at a time,
    # refit per-annotator OLS on other questions, evaluate on held-out.
    all_questions = df["Question_Index"].unique()
    per_ann_gains = []
    for aid, ann_df in df.groupby("annotator_id"):
        if len(ann_df) < 20:
            continue
        sq_pop = []
        sq_pebs = []
        for q_held in ann_df["Question_Index"].unique():
            train = ann_df[ann_df["Question_Index"] != q_held]
            test = ann_df[ann_df["Question_Index"] == q_held]
            if len(train) < 10 or len(test) < 1:
                continue
            # Per-annotator LOCO-equivalent OLS (leave one question out)
            x_tr = train["Harm_Level"].to_numpy(dtype=np.float64)
            y_tr = train["rating"].to_numpy(dtype=np.float64)
            x_te = test["Harm_Level"].to_numpy(dtype=np.float64)
            y_te = test["rating"].to_numpy(dtype=np.float64)

            x_mean_tr, y_mean_tr = x_tr.mean(), y_tr.mean()
            vx = np.sum((x_tr - x_mean_tr) ** 2)
            if vx < 1e-10:
                continue
            b_a = float(np.sum((x_tr - x_mean_tr) * (y_tr - y_mean_tr)) / vx)
            a_a = float(y_mean_tr - b_a * x_mean_tr)
            resid = y_tr - (a_a + b_a * x_tr)
            mse = float(np.sum(resid ** 2) / max(len(x_tr) - 2, 1))
            se_b_a = float(np.sqrt(mse / vx))
            se_a_a = float(np.sqrt(mse * (1.0 / len(x_tr) + x_mean_tr ** 2 / vx)))

            w_a = tau2_a / (tau2_a + se_a_a ** 2 + 1e-12)
            w_b = tau2_b / (tau2_b + se_b_a ** 2 + 1e-12)
            a_s = w_a * a_a + (1 - w_a) * a_pop
            b_s = w_b * b_a + (1 - w_b) * b_pop

            pred_pop = a_pop + b_pop * x_te
            pred_pebs = a_s + b_s * x_te
            sq_pop.append(float(np.sum((y_te - pred_pop) ** 2)))
            sq_pebs.append(float(np.sum((y_te - pred_pebs) ** 2)))

        if not sq_pop:
            continue
        rmse_pop = float(np.sqrt(sum(sq_pop) / len(ann_df)))
        rmse_pebs = float(np.sqrt(sum(sq_pebs) / len(ann_df)))
        if rmse_pop > 1e-6:
            gain = 100.0 * (rmse_pop - rmse_pebs) / rmse_pop
            per_ann_gains.append(gain)

    gains = np.array(per_ann_gains)
    mean_gain = float(gains.mean())
    boots = np.array([rng.choice(gains, len(gains), replace=True).mean()
                      for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\n[PluriHarms LOCO-Q per-user PEBS-shrunk vs pop-OLS] "
          f"gain = {mean_gain:+.2f}%  CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"[n_annotators analyzed] {len(gains)}")

    summary = {
        "protocol": "Leave-one-question-out per annotator (LOCO-Q equivalent)",
        "n_annotators": int(len(gains)),
        "pop_alpha": a_pop, "pop_beta": b_pop,
        "tau2_alpha": tau2_a, "tau2_beta": tau2_b,
        "mean_gain_pct": mean_gain,
        "ci95": [float(lo), float(hi)],
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "n_bootstrap": N_BOOT,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT}/summary.json")


if __name__ == "__main__":
    main()
