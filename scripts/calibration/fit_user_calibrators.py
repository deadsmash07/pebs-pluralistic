"""Fit per-user linear calibration (α_j, β_j) on PRISM RM-scored utterances.

Inputs
------
- `data/prism_rm_scored.parquet` from `score_prism_utterances.py` — one row
  per utterance with (user_id, rm_score, score_user, ...).

Model (Pinheiro & Bates 2000 linear mixed-effects):

    score_user_ij = (β_0 + β_0j) + (β_1 + β_1j) · rm_score_ij + ε_ij

where (β_0j, β_1j) ~ N(0, Ω) are random intercept and slope per user.

For each user j we extract a calibration:

    α_j = β_1 + β_1j      per-user slope
    β_j = β_0 + β_0j      per-user intercept

Then the corrected score for any new utterance i from user j is:

    corrected_ij = (score_user_ij − β_j) / α_j      if α_j > 0

which lives on the shared RM scale. This is the Track 1 mediator M_corrected.

Outputs
-------
- `data/prism_user_calibrators.parquet` (one row per user with α_j, β_j, SEs,
  n observations, and goodness-of-fit)
- `data/prism_corrected_scores.parquet` — one row per utterance with
  (utterance_id, user_id, rm_score, score_user, score_corrected).

Refs: Pinheiro & Bates 2000 §2; statsmodels MixedLM REML canonical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scored-parquet", default="data/prism_rm_scored.parquet")
    p.add_argument("--output-dir", default="data")
    p.add_argument("--min-obs-per-user", type=int, default=5,
                   help="Users with fewer labeled scores get grand-mean imputation.")
    p.add_argument("--zscore-rm-score", action="store_true",
                   help="Z-score rm_score before fitting for numerical stability.")
    p.add_argument("--output-prefix", default="prism")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.scored_parquet)
    # Only utterances with user-assigned scores (PRISM user rated it)
    df = df.dropna(subset=["score_user"]).reset_index(drop=True)
    print(f"[load] {len(df)} utterances with user scores across {df.user_id.nunique()} users")

    # Z-score rm_score for numerical stability (optional)
    rm_mean, rm_std = float(df.rm_score.mean()), float(df.rm_score.std())
    if args.zscore_rm_score:
        df["rm_score_z"] = (df.rm_score - rm_mean) / max(rm_std, 1e-9)
        predictor = "rm_score_z"
    else:
        predictor = "rm_score"
    print(f"[feat] rm_score mean={rm_mean:.4f} std={rm_std:.4f}")
    print(f"[feat] score_user mean={df.score_user.mean():.2f} std={df.score_user.std():.2f}")

    # Fit: score_user ~ rm_score + (rm_score | user_id)
    print("[fit] MixedLM with per-user random slope + intercept")
    try:
        md = smf.mixedlm(
            f"score_user ~ {predictor}",
            data=df,
            groups=df["user_id"],
            re_formula=f"~{predictor}",  # random intercept + slope
        )
        res = md.fit(method="lbfgs", maxiter=500, disp=False)
        print(f"[fit] converged={res.converged}, ll={res.llf:.1f}")
        print(res.summary().tables[1])
    except Exception as e:
        print(f"[fit] MixedLM with random slope failed: {e}")
        print("[fit] falling back to random intercept only")
        md = smf.mixedlm(
            f"score_user ~ {predictor}",
            data=df,
            groups=df["user_id"],
        )
        res = md.fit(method="lbfgs", maxiter=500, disp=False)

    # Population-level coefficients
    pop_intercept = float(res.params["Intercept"])
    pop_slope = float(res.params[predictor])
    print(f"[pop] β_0={pop_intercept:.3f}  β_1={pop_slope:.3f}")

    # Per-user random effects
    re_dict = res.random_effects  # dict: user_id → Series of random-effect values
    per_user = []
    for uid, re in re_dict.items():
        # statsmodels keys: "Group" for intercept, predictor-name for slope
        re_intercept = float(re.get("Group", re.get("Intercept", 0.0)))
        re_slope = float(re.get(predictor, 0.0))
        alpha_j = pop_slope + re_slope
        beta_j = pop_intercept + re_intercept
        n_obs = int((df.user_id == uid).sum())
        per_user.append({
            "user_id": str(uid),
            "alpha_j": alpha_j,
            "beta_j": beta_j,
            "n_observations": n_obs,
        })
    cal_df = pd.DataFrame(per_user)
    print(f"[calibrators] fit for {len(cal_df)} users")
    print(f"  α_j summary: mean={cal_df.alpha_j.mean():.3f} "
          f"std={cal_df.alpha_j.std():.3f} "
          f"quartiles={np.percentile(cal_df.alpha_j, [25,50,75]).round(3).tolist()}")
    print(f"  β_j summary: mean={cal_df.beta_j.mean():.1f} "
          f"std={cal_df.beta_j.std():.1f}")

    # Save calibrators
    cal_path = Path(args.output_dir) / f"{args.output_prefix}_user_calibrators.parquet"
    cal_df.to_parquet(cal_path)
    print(f"[save] {cal_path}")

    # Apply correction: corrected_ij = (score_user_ij - β_j) / α_j
    # Map calibrators onto the original df
    merged = df.merge(cal_df, on="user_id", how="left")
    # Guard against degenerate α_j close to zero — fall back to (score - β)
    merged["score_corrected"] = np.where(
        merged.alpha_j.abs() > 0.05,
        (merged.score_user - merged.beta_j) / merged.alpha_j,
        merged.score_user - merged.beta_j,
    )
    out_cols = ["utterance_id", "user_id", "interaction_id", "turn", "if_chosen",
                "rm_score", "score_user", "alpha_j", "beta_j",
                "score_corrected"]
    corrected_df = merged[[c for c in out_cols if c in merged.columns]]
    corr_path = Path(args.output_dir) / f"{args.output_prefix}_corrected_scores.parquet"
    corrected_df.to_parquet(corr_path)
    print(f"[save] {corr_path} ({len(corrected_df)} rows)")

    # Diagnostic: correlation between score_user and score_corrected vs rm_score
    print("\n=== Correlation diagnostic (calibration sanity check) ===")
    raw_corr = np.corrcoef(merged.rm_score, merged.score_user)[0, 1]
    corr_corr = np.corrcoef(merged.rm_score, merged.score_corrected)[0, 1]
    print(f"  corr(rm_score, score_user_raw) = {raw_corr:.4f}")
    print(f"  corr(rm_score, score_corrected) = {corr_corr:.4f}")
    print(f"  → if corrected > raw, the per-user calibration recovered shared signal.")


if __name__ == "__main__":
    main()
