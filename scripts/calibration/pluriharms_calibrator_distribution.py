"""PluriHarms vs PRISM per-user calibrator distribution comparison (iter+N+125).

Content-first new finding: do the per-user (alpha_j, beta_j) distributions
fitted on PRISM and PluriHarms look similar? Similar distributions would
strengthen the cross-dataset PILSD generalization claim: the
"annotator idiosyncrasy" structure is consistent across completely
different crowdsourcing protocols (open-ended dialogue preference vs
harm rating).

Specifically:
1. Per-user OLS (alpha_j, beta_j) for each dataset
2. Population-slope alpha_pop, beta_pop for each
3. tau_alpha, tau_beta between-user SD for each
4. Pearson correlation corr(alpha_j, beta_j) for each
5. Demographic correlates of (alpha_j, beta_j) in each

If the (tau, correlation) structure looks similar, it's a strong signal
that PILSD is capturing a GENUINE psychometric property of human
annotators rather than a dataset-specific artifact.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]


def per_user_ols(df: pd.DataFrame, anchor_col: str, rating_col: str) -> pd.DataFrame:
    """Fit per-user (alpha_j, beta_j) via OLS on anchor_score."""
    rows = []
    for uid, g in df.groupby("user_id"):
        x = g[anchor_col].values.reshape(-1, 1)
        y = g[rating_col].values
        if len(g) < 5:
            continue
        lr = LinearRegression().fit(x, y)
        alpha_j = float(lr.coef_[0])
        beta_j = float(lr.intercept_)
        # standard errors
        yhat = lr.predict(x)
        resid = y - yhat
        s2 = float(np.var(resid, ddof=2)) if len(resid) > 2 else float(np.var(resid))
        xx = float(np.sum((x - x.mean()) ** 2))
        se_alpha = float(np.sqrt(s2 / (xx + 1e-9)))
        se_beta = float(np.sqrt(s2 * (1.0 / len(g) + (x.mean() ** 2) / (xx + 1e-9))))
        rows.append(
            dict(
                user_id=uid,
                alpha=alpha_j,
                beta=beta_j,
                n=len(g),
                se_alpha=se_alpha,
                se_beta=se_beta,
            )
        )
    return pd.DataFrame(rows)


def prism_long() -> pd.DataFrame:
    s = pd.read_parquet(ROOT / "data" / "prism_rm_scored.parquet")
    s = s.dropna(subset=["rm_score", "score_user"]).reset_index(drop=True)
    s = s.rename(columns={"rm_score": "anchor", "score_user": "rating"})
    return s[["user_id", "anchor", "rating"]]


def pluriharms_long() -> pd.DataFrame:
    ann = pd.read_csv(ROOT / "data" / "pluriharms" / "annotations.csv")
    prm = pd.read_csv(ROOT / "data" / "pluriharms" / "prompts.csv")
    rating_cols = [c for c in ann.columns if c.startswith("Rating_")]
    long = ann.melt(
        id_vars=["Participant_ID"],
        value_vars=rating_cols,
        var_name="_rating_col",
        value_name="rating",
    )
    long["Question_Index"] = long["_rating_col"].str.replace("Rating_", "").astype(int)
    long = long.merge(prm[["Question_Index", "Harm_Level"]], on="Question_Index")
    long = long.dropna(subset=["rating", "Harm_Level"]).reset_index(drop=True)
    long = long.rename(
        columns={"Participant_ID": "user_id", "Harm_Level": "anchor"}
    )
    return long[["user_id", "anchor", "rating"]]


def summarize(df_per_user: pd.DataFrame, df_long: pd.DataFrame, label: str) -> dict:
    # Population-slope reference
    lr = LinearRegression().fit(
        df_long[["anchor"]].values, df_long["rating"].values
    )
    alpha_pop = float(lr.coef_[0])
    beta_pop = float(lr.intercept_)

    # Between-user SD
    alpha_j = df_per_user["alpha"].values
    beta_j = df_per_user["beta"].values
    # Method-of-moments tau estimate (between-user variance net of sampling variance)
    tau_alpha_sq_mom = float(np.var(alpha_j, ddof=1) - np.mean(df_per_user["se_alpha"].values ** 2))
    tau_beta_sq_mom = float(np.var(beta_j, ddof=1) - np.mean(df_per_user["se_beta"].values ** 2))
    tau_alpha = float(np.sqrt(max(tau_alpha_sq_mom, 0)))
    tau_beta = float(np.sqrt(max(tau_beta_sq_mom, 0)))

    corr_alpha_beta = float(np.corrcoef(alpha_j, beta_j)[0, 1])

    frac_alpha_neg = float((alpha_j < 0).mean())

    return {
        "dataset": label,
        "n_users": int(len(df_per_user)),
        "alpha_pop": alpha_pop,
        "beta_pop": beta_pop,
        "tau_alpha": tau_alpha,
        "tau_beta": tau_beta,
        "tau_alpha_sq_mom": tau_alpha_sq_mom,
        "tau_beta_sq_mom": tau_beta_sq_mom,
        "corr_alpha_beta_raw": corr_alpha_beta,
        "frac_alpha_negative": frac_alpha_neg,
        "alpha_iqr": [float(np.quantile(alpha_j, 0.25)), float(np.quantile(alpha_j, 0.75))],
        "beta_iqr": [float(np.quantile(beta_j, 0.25)), float(np.quantile(beta_j, 0.75))],
        "alpha_median": float(np.median(alpha_j)),
        "beta_median": float(np.median(beta_j)),
    }


def main():
    prism_df = prism_long()
    pluri_df = pluriharms_long()
    print(f"[data] PRISM N={len(prism_df)} ({prism_df['user_id'].nunique()} users)")
    print(f"[data] PluriHarms N={len(pluri_df)} ({pluri_df['user_id'].nunique()} users)")

    prism_cal = per_user_ols(prism_df, "anchor", "rating")
    pluri_cal = per_user_ols(pluri_df, "anchor", "rating")

    prism_summary = summarize(prism_cal, prism_df, "PRISM")
    pluri_summary = summarize(pluri_cal, pluri_df, "PluriHarms")

    # Side-by-side comparison
    print("\n=== Per-user calibrator distribution comparison ===")
    print(f"{'metric':<30} {'PRISM':<20} {'PluriHarms':<20}")
    for k in ["n_users", "alpha_pop", "beta_pop", "tau_alpha", "tau_beta",
              "corr_alpha_beta_raw", "frac_alpha_negative",
              "alpha_median", "beta_median"]:
        p = prism_summary.get(k, "n/a")
        ph = pluri_summary.get(k, "n/a")
        if isinstance(p, float):
            p = f"{p:+.4f}"
        if isinstance(ph, float):
            ph = f"{ph:+.4f}"
        print(f"{k:<30} {str(p):<20} {str(ph):<20}")

    # Structural comparison: do the two datasets show SIMILAR calibrator structure?
    # Compare tau_alpha / alpha_pop ratio (coefficient of variation between users)
    cv_alpha_prism = prism_summary["tau_alpha"] / max(abs(prism_summary["alpha_pop"]), 1e-9)
    cv_alpha_pluri = pluri_summary["tau_alpha"] / max(abs(pluri_summary["alpha_pop"]), 1e-9)
    cv_beta_prism = prism_summary["tau_beta"] / max(abs(prism_summary["beta_pop"]), 1e-9)
    cv_beta_pluri = pluri_summary["tau_beta"] / max(abs(pluri_summary["beta_pop"]), 1e-9)
    print("\n=== Structural similarity: coefficient of variation ===")
    print(f"CV(alpha) = tau_alpha / |alpha_pop|:")
    print(f"  PRISM       : {cv_alpha_prism:.3f}")
    print(f"  PluriHarms  : {cv_alpha_pluri:.3f}")
    print(f"CV(beta) = tau_beta / |beta_pop|:")
    print(f"  PRISM       : {cv_beta_prism:.3f}")
    print(f"  PluriHarms  : {cv_beta_pluri:.3f}")

    out = {
        "PRISM": prism_summary,
        "PluriHarms": pluri_summary,
        "structural_comparison": {
            "cv_alpha_prism": cv_alpha_prism,
            "cv_alpha_pluriharms": cv_alpha_pluri,
            "cv_beta_prism": cv_beta_prism,
            "cv_beta_pluriharms": cv_beta_pluri,
            "cv_alpha_ratio_pluri_over_prism": cv_alpha_pluri / max(cv_alpha_prism, 1e-9),
            "cv_beta_ratio_pluri_over_prism": cv_beta_pluri / max(cv_beta_prism, 1e-9),
        },
    }
    out_path = ROOT / "results" / "pluriharms_vs_prism_calibrator_distribution.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
