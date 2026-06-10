#!/usr/bin/env python3
"""Morris g 2-parameter forecast validated on 2 additional corpora.

CONCERN:
  The 2-param fit (morris_g_2param_validate.py) reports g(r_alpha, r_beta) = 17.96% vs MultiPref
  observed 0.47% — miss unchanged from 1-param's 17.7pp gap. Paper
  concedes "predicted risk-gap > total POP residual" — inconsistency.
  Claim "Morris g is a validated forecasting tool" rests on only 2
  corpora (PRISM, PluriHarms) — under-powered for a "validated" label.

RESOLUTION:
  Add 2 new real corpora to the forecasting test:
    (i)  OASST2 author-rank (from 3_PEBS_Standalone/data/oasst2_author_quality.parquet)
    (ii) SHP domain-subreddit (from 3_PEBS_Standalone/data/shp_domain_quality.parquet)
  Fit MoM (tau^2_alpha, tau^2_beta, sigma^2_eps) and compute 2-param forecast.
  Compare predicted vs observed PEBS gain from the earlier OASST and SHP
  runs where available.

  If 2/2 new corpora are within 5 pp of observed: forecasting-tool framing
  survives with scope clause "works on rating corpora with continuous y
  and bounded range; fails on ordinal MultiPref-style preference data".
  If 1/2 or 0/2 within 5 pp: demote to "sanity check for Gaussian-RE
  regime"; strike "validated" from abstract.

CPU-only. Runtime ~10-30s.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Reuse the fit + predict functions from morris_g_2param_validate.py
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from morris_g_2param_validate import fit_two_param, predict_2param  # noqa: E402


def load_oasst2(path: str):
    """OASST2 — user_id is the rater (22k users, many with >=5 obs).
    y = quality (continuous 0-1 bounded), x = rank (discrete 0-15).
    """
    df = pd.read_parquet(path)
    # drop rows with missing rank or quality
    df = df.dropna(subset=["quality", "rank", "user_id"])
    df = df[["user_id", "quality", "rank"]]
    return df, "quality", "rank"


def load_shp(path: str):
    """SHP — user_id is the SUBREDDIT (18 subreddits, each with thousands
    of obs). Stratifying by subreddit mirrors the MULTI-USER RE structure
    that PEBS models (within-community heterogeneity as proxy for
    within-user).
    y = quality = log(winner_score), x = log_score_ratio.
    """
    df = pd.read_parquet(path)
    df = df.dropna(subset=["quality", "log_score_ratio", "user_id"])
    df = df[["user_id", "quality", "log_score_ratio"]]
    return df, "quality", "log_score_ratio"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--oasst2-parquet",
                   default="../3_PEBS_Standalone/data/oasst2_author_quality.parquet")
    p.add_argument("--shp-parquet",
                   default="../3_PEBS_Standalone/data/shp_domain_quality.parquet")
    p.add_argument("--output-dir",
                   default="results/track1_morris_g_extended")
    # Observed PEBS gains: from the OASST robust M-estimator runs
    # OASST union drift β̂ measures a DIFFERENT quantity (drift coef).
    # For this closure, we compute the PEBS relative-RMSE gain on
    # oasst2_author_quality using the SAME protocol as morris_g_2param_validate:
    # per-user CV rel-RMSE(PEBS) vs rel-RMSE(POP). If the PEBS gain hasn't
    # been run on OASST2/SHP before, we compute it here (leave-one-out
    # per-user CV), so the 'observed' column is a deterministic function
    # of the data + the same EB formula that morris_g uses.
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def compute_pebs_gain_loo_cv(df: pd.DataFrame,
                               y_col: str, x_col: str,
                               user_col: str = "user_id",
                               min_obs: int = 5) -> float:
    """Empirical PEBS relative-RMSE gain via leave-one-row-out CV.

    For each held-out row (i, j):
      y_hat_POP    = alpha_pop + beta_pop * x_ij        (full-data POP)
      y_hat_PEBS  = alpha_j^EB + beta_j^EB * x_ij      (per-user EB fit on
                                                         the REMAINING n_j-1
                                                         rows, shrunk toward
                                                         (alpha_pop,beta_pop)
                                                         via omega_j)
    Then rel_gain = 1 - sqrt(SSE_PEBS / SSE_POP).

    Uses the SAME MoM tau^2_alpha, tau^2_beta, sigma^2_eps from fit_two_param
    on the FULL dataset (hold-out-one shrinkage parameters are not
    re-estimated per fold — this mirrors morris_g_validate.py and the paper's standard
    PEBS evaluation protocol where (tau, sigma) are global MoM estimates).

    Runtime: vectorized per user; O(sum_j n_j) = O(n_total).
    """
    # Global MoM via same helper
    fit = fit_two_param(df, y_col, x_col, user_col=user_col, min_obs=min_obs)
    tau_a = fit["tau_a_within_sq"]
    tau_b = fit["tau_beta_sq"]
    sigma2 = fit["sigma_eps_sq"]
    alpha_pop = fit["alpha_pop"]
    beta_pop = fit["beta_pop"]

    if sigma2 <= 0 or np.isnan(sigma2):
        return float("nan")

    # For each user, compute leave-one-out residual under POP and under EB
    # Use vectorized per-user leave-one-out: for user j with n_j obs, the
    # leave-i-out mean is (n_j*mean - x_i) / (n_j-1), etc.
    # Vectorised per-user leave-one-out using closed-form sufficient-statistics.
    # Per user j: maintain sums S_x = sum x_k, S_y, S_xx, S_xy, S_yy, n.
    # Leave-i-out sufficient statistics:
    #   ni' = n-1
    #   Sx' = S_x - x_i;  Sy' = S_y - y_i;
    #   Sxx' = S_xx - x_i^2;  Sxy' = S_xy - x_i*y_i;  Syy' = S_yy - y_i^2
    #   xbar' = Sx'/ni';  ybar' = Sy'/ni'
    #   Sxx_c' = Sxx' - ni' * xbar'^2   (centred SSx)
    #   Sxy_c' = Sxy' - ni' * xbar' * ybar'
    #   b_j' = Sxy_c' / Sxx_c'
    #   varx_within' = Sxx_c' / ni'
    sse_pop = 0.0
    sse_pebs = 0.0
    n_total_folds = 0
    for uid, grp in df.groupby(user_col, sort=False):
        n = len(grp)
        if n < min_obs:
            continue
        x = grp[x_col].to_numpy(dtype=float)
        y = grp[y_col].to_numpy(dtype=float)
        # POP residuals (vectorised)
        resid_pop = y - (alpha_pop + beta_pop * x)
        # Full-user sufficient statistics:
        Sx = float(x.sum()); Sy = float(y.sum())
        Sxx = float((x * x).sum()); Sxy = float((x * y).sum())
        # Leave-one-out (vectorised across all i):
        ni = n - 1
        if ni < 3:
            continue
        Sx_i = Sx - x
        Sy_i = Sy - y
        Sxx_i = Sxx - x * x
        Sxy_i = Sxy - x * y
        xbar_i = Sx_i / ni
        ybar_i = Sy_i / ni
        Sxx_c_i = Sxx_i - ni * xbar_i * xbar_i
        Sxy_c_i = Sxy_i - ni * xbar_i * ybar_i
        # Guard against degenerate within-x variance
        mask = Sxx_c_i > 1e-12
        if not mask.any():
            continue
        b_ji = np.where(mask, Sxy_c_i / np.where(mask, Sxx_c_i, 1.0), 0.0)
        varx_within = Sxx_c_i / ni
        # EB weights using LEAVE-ONE-OUT ni
        r_alpha_i = ni * tau_a / sigma2
        r_beta_i = ni * tau_b * varx_within / sigma2
        omega_a = r_alpha_i / (1.0 + r_alpha_i)
        omega_b = r_beta_i / (1.0 + r_beta_i)
        # Shrinkage targets at the LEAVE-ONE-OUT xbar_i
        a_within_pop = alpha_pop + beta_pop * xbar_i
        a_within_eb = omega_a * ybar_i + (1.0 - omega_a) * a_within_pop
        b_eb = omega_b * b_ji + (1.0 - omega_b) * beta_pop
        # Prediction at held-out x[i], centred at LOO xbar_i
        y_hat_pebs = a_within_eb + b_eb * (x - xbar_i)
        resid_pebs = y - y_hat_pebs
        sse_pop += float((resid_pop[mask] ** 2).sum())
        sse_pebs += float((resid_pebs[mask] ** 2).sum())
        n_total_folds += int(mask.sum())
    if n_total_folds == 0 or sse_pop <= 0:
        return float("nan")
    rmse_pop = np.sqrt(sse_pop / n_total_folds)
    rmse_pebs = np.sqrt(sse_pebs / n_total_folds)
    return 100.0 * (1.0 - rmse_pebs / rmse_pop)


# Kept for back-compat; delegates to the real LOO CV.
def compute_pebs_gain_from_fit(fit: dict, df: pd.DataFrame,
                                 y_col: str, x_col: str,
                                 user_col: str = "user_id",
                                 min_obs: int = 5) -> float:
    return compute_pebs_gain_loo_cv(df, y_col, x_col, user_col, min_obs)


def main():
    args = parse_args()
    t0 = time.time()
    root = Path(__file__).resolve().parents[1]  # 1_Causal_RLHF/
    oasst2_path = str(root / args.oasst2_parquet) if not args.oasst2_parquet.startswith("/") else args.oasst2_parquet
    shp_path = str(root / args.shp_parquet) if not args.shp_parquet.startswith("/") else args.shp_parquet

    corpora_config = [
        ("OASST2-author", load_oasst2, oasst2_path, 5),
        ("SHP-subreddit", load_shp, shp_path, 20),
    ]

    table = []
    corpora_out = {}
    for name, loader, path, min_obs in corpora_config:
        print(f"\n[{name}] loading from {path} (min_obs={min_obs})...")
        try:
            df, y_col, x_col = loader(path)
            print(f"  shape={df.shape}  users={df['user_id'].nunique()}  "
                  f"y='{y_col}' x='{x_col}'")
            # Filter to users with >= min_obs observations
            n_per_user = df.groupby("user_id").size()
            users_ok = n_per_user[n_per_user >= min_obs].index
            df = df[df["user_id"].isin(users_ok)]
            print(f"  after filter >= {min_obs} obs: shape={df.shape}  "
                  f"users={df['user_id'].nunique()}")
            fit = fit_two_param(df, y_col, x_col,
                                user_col="user_id", min_obs=min_obs)
            pred = predict_2param(fit)
            # Compute "observed" PEBS gain via same-data ratio-form MSE
            # (matches morris_g_2param_validate's metric).
            observed_pct = compute_pebs_gain_from_fit(
                fit, df, y_col, x_col, user_col="user_id", min_obs=min_obs,
            )
            pred_1p = pred["rel_imp_pct_1param_ref"]
            pred_2p = pred["rel_imp_pct_2param"]  # CV (ratio form)
            pred_2p_global = pred["rel_imp_pct_2param_global_x"]

            corpora_out[name] = {
                "fit": {k: v for k, v in fit.items()
                        if k not in ("n_j", "varx_within_j", "xbar_j")},
                "predicted": pred,
                "observed_pct": observed_pct,
                "y_col": y_col,
                "x_col": x_col,
            }
            table.append({
                "corpus": name,
                "n_users": fit["n_users"],
                "tau_a_within_sq": round(fit["tau_a_within_sq"], 5),
                "tau_beta_sq": round(fit["tau_beta_sq"], 6),
                "sigma_eps_sq": round(fit["sigma_eps_sq"], 5),
                "mse_pop_baseline": round(fit["mse_pop_baseline"], 5),
                "rmbar2": round(fit["rmbar2_global"], 5),
                "mean_r_alpha": round(pred["mean_r_alpha"], 3),
                "mean_r_beta": round(pred["mean_r_beta"], 3),
                "pred_1param_pct": round(pred_1p, 3) if pred_1p is not None else None,
                "pred_2param_pct_CV": round(pred_2p, 3) if pred_2p is not None else None,
                "pred_2param_pct_global_x": round(pred_2p_global, 3) if pred_2p_global is not None else None,
                "observed_pct": round(observed_pct, 3) if not np.isnan(observed_pct) else None,
                "abs_delta_1p_pp": round(abs(pred_1p - observed_pct), 3)
                                   if pred_1p is not None and not np.isnan(observed_pct) else None,
                "abs_delta_2p_cv_pp": round(abs(pred_2p - observed_pct), 3)
                                      if pred_2p is not None and not np.isnan(observed_pct) else None,
                "re_model_violated": pred.get("re_model_violated", False),
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            table.append({"corpus": name, "error": str(e)})

    # ---- Closure analysis ----
    within_5pp = 0
    within_10pp = 0
    total_ok = 0
    for r in table:
        if "error" in r:
            continue
        d2 = r.get("abs_delta_2p_cv_pp")
        if d2 is None:
            continue
        total_ok += 1
        if d2 <= 5.0:
            within_5pp += 1
        if d2 <= 10.0:
            within_10pp += 1

    if total_ok == 0:
        verdict = "UNABLE_TO_EVALUATE"
        verdict_note = "Both new corpora errored — cannot resolve the concern."
    elif within_5pp == total_ok:
        verdict = "FORECASTING_TOOL_SURVIVES"
        verdict_note = (f"{within_5pp}/{total_ok} new corpora within 5pp; "
                        "retain 'forecasting tool' framing with honest scope "
                        "clause: works on continuous-y bounded-range rating "
                        "corpora, fails on ordinal MultiPref.")
    elif within_5pp >= total_ok // 2:
        verdict = "DOWNGRADE_TO_SANITY_CHECK"
        verdict_note = (f"Only {within_5pp}/{total_ok} new corpora within 5pp; "
                        "downgrade 'validated forecasting rule' → "
                        "'sanity check for Gaussian-RE regime'; strike "
                        "'validated' from abstract.")
    else:
        verdict = "DEMOTE_TO_SCOPED_SANITY_CHECK"
        verdict_note = (f"Only {within_5pp}/{total_ok} new corpora within 5pp. "
                        "Demote to qualitative diagnostic; abstract must not "
                        "claim forecasting.")

    out = {
        "concern": "Morris g 2-param misses MultiPref by 17.49pp",
        "n_corpora_tested": total_ok,
        "n_corpora_within_5pp": within_5pp,
        "n_corpora_within_10pp": within_10pp,
        "table": table,
        "corpora": corpora_out,
        "verdict": verdict,
        "verdict_note": verdict_note,
        "wall_seconds": time.time() - t0,
    }
    outdir = root / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(out, indent=2, default=str))

    print(f"\n=== SUMMARY ===")
    print(f"  Corpus       |  pred_1p%  pred_2p_CV%  observed%  |Δ_1p|pp  |Δ_2p|pp  violated")
    for r in table:
        if "error" in r:
            print(f"  {r['corpus']:12s} | ERROR: {r['error']}")
            continue
        p1 = r.get("pred_1param_pct"); p2 = r.get("pred_2param_pct_CV")
        obs = r.get("observed_pct")
        d1 = r.get("abs_delta_1p_pp"); d2 = r.get("abs_delta_2p_cv_pp")
        def _f(v, w):
            if v is None: return " " * (w-1) + "—"
            return f"{v:>{w}.3f}"
        print(f"  {r['corpus']:12s} | {_f(p1,9)}  {_f(p2,10)}  {_f(obs,8)}  {_f(d1,7)}  {_f(d2,7)}  "
              f"{str(r.get('re_model_violated', False)):>8s}")
    print(f"  VERDICT : {verdict}")
    print(f"  NOTE    : {verdict_note}")
    print(f"  wall    : {out['wall_seconds']:.1f}s")
    print(f"\n[save] {outdir/'summary.json'}")


if __name__ == "__main__":
    main()
