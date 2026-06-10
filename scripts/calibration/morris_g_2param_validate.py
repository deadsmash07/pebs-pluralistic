"""Two-parameter Morris g-function closed-form validation.

Extends the one-way g(r) = r/(1+r) risk-gap theorem (morris_g_validate.py) to the PEBS
two-parameter random-effects model
    s_ji = alpha_j + beta_j * x_ji + eps_ji,
    alpha_j ~ N(mu_alpha, tau_alpha^2),  beta_j ~ N(mu_beta, tau_beta^2),
    eps_ji ~ N(0, sigma_eps^2)  i.i.d.

Theorem T3 (two-parameter Morris g)
-----------------------------------
Let
    r_alpha_j = n_j * tau_alpha^2 / sigma_eps^2
    r_beta_j  = n_j * tau_beta^2  * Var_within(x_j) / sigma_eps^2
    g(r) = r / (1 + r)   (same concave form as the one-parameter validation).

Under i.i.d. Gaussian REs with *within-user-centered* design
(equivalently, per-user OLS uses the local x-mean \bar x_j as origin),
the cross-term vanishes and the expected prediction-risk gap of the EB
predictor vs the POP predictor (with true mu_alpha, mu_beta) is

    E[ POP-risk - EB-risk ]
        = tau_alpha^2 * g(r_alpha)  +  tau_beta^2 * rmbar2 * g(r_beta),

where rmbar2 = E[x^2] is the second (raw) moment of x across the corpus
(because the intercept in the per-user regression is parameterized at
x = 0 globally, so the predictor's contribution to MSE(y_new) at a new
x-point is x^2 * Var(beta_hat) + Var(alpha_hat)).

Proof sketch
------------
* Per-user OLS with local-mean centering gives independent (alpha_hat, beta_hat).
  The EB posterior mean is alpha_hat^EB = omega_alpha_j * alpha_hat_j
  + (1-omega_alpha_j) * mu_alpha (similarly for beta), with
  omega_alpha_j = r_alpha_j / (1 + r_alpha_j) (James-Stein).
* Risk(alpha_hat^EB | alpha_j) = tau_alpha^2 / (1 + r_alpha_j) (one-parameter Lemma).
* Risk(beta_hat^EB  | beta_j)  = tau_beta^2  / (1 + r_beta_j).
* POP risks are tau_alpha^2 and tau_beta^2.
* Summing with the x^2-weighting on the beta component and subtracting
  gives the theorem.  QED.

Empirical protocol
------------------
For each of PRISM / PluriHarms / MultiPref:
  1. Fit per-user OLS on within-user-centered x: alpha_hat_j, beta_hat_j,
     SE_alpha_j, SE_beta_j, Sxx_j.
  2. MoM on intercepts:  tau_alpha^2 = max(var(alpha_hat) - mean(SE_alpha^2), 0).
     MoM on slopes:      tau_beta^2  = max(var(beta_hat)  - mean(SE_beta^2),  0).
     sigma_eps^2 = pooled residual variance across all users.
  3. Per-user r_alpha_j, r_beta_j.
  4. x^2 second-moment (global): rmbar2 = mean(x**2) over all observations.
  5. Predicted corpus-level MSE-gap (absolute):
         dMSE = tau_alpha^2 * mean(g(r_alpha))  +  tau_beta^2 * rmbar2 * mean(g(r_beta))
     with n_j-weighted means.
  6. Predicted corpus-level RMSE vs POP: MSE_POP_empirical from the
     POP-predictor (already in the one-parameter fit), MSE_EB = MSE_POP_empirical - dMSE,
     rel_imp = 1 - sqrt(MSE_EB / MSE_POP).
  7. Compare to observed PEBS gain.

Also report the 1-parameter prediction for reference.

CPU-only; <5s wall.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def g(r: np.ndarray) -> np.ndarray:
    return r / (1.0 + r)


# --------------------------------------------------------------------------- #
# Fitter                                                                      #
# --------------------------------------------------------------------------- #
def _global_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_m, y_m = x.mean(), y.mean()
    Sxx = float(np.sum((x - x_m) ** 2))
    if Sxx < 1e-12:
        return float(y_m), 0.0
    beta = float(np.sum((x - x_m) * (y - y_m)) / Sxx)
    alpha = float(y_m - beta * x_m)
    return alpha, beta


def fit_two_param(df: pd.DataFrame, y_col: str, x_col: str,
                  user_col: str = "user_id", min_obs: int = 5) -> dict:
    """Fit per-user OLS with within-user centering; return MoM for tau_alpha^2,
    tau_beta^2, sigma_eps^2.

    Parameterization: for user j we fit y = a_j + b_j * (x - xbar_j) + eps.
    Then a_j is the predicted y at x = xbar_j (within-user centered intercept);
    but we REPARAMETERIZE back to the global zero: alpha_j = a_j - b_j * xbar_j
    so that the prediction at a new x = x0 is alpha_j + b_j * x0. We use the
    GLOBAL-zero parameterization throughout because the POP predictor is
    expressed at global zero.

    For risk decomposition we need, however, that alpha_hat_j and beta_hat_j are
    INDEPENDENT given (alpha_j, beta_j). Under Gaussian OLS, within-user
    centering achieves independence (Sxx design's off-diagonal = 0). That is
    preserved under the reparameterization a = alpha + b*xbar_j because xbar_j
    is a constant for user j (conditional on the design). So the
    within-centered fit gives independent (a_j, b_j); in the risk decomposition
    we use the GLOBAL MSE formula with x^2 second-moment which captures the
    reparameterization penalty automatically.
    """
    alpha_pop, beta_pop = _global_ols(df[x_col].to_numpy(dtype=float),
                                       df[y_col].to_numpy(dtype=float))

    groups = df.groupby(user_col)
    rows = []
    s_within_sum = 0.0
    df_within = 0
    s_popbaseline_sum = 0.0
    df_popbaseline = 0
    all_x_sq_sum = 0.0
    all_x_sum = 0.0
    n_total = 0

    for uid, grp in groups:
        n = len(grp)
        if n < min_obs:
            continue
        y = grp[y_col].to_numpy(dtype=float)
        x = grp[x_col].to_numpy(dtype=float)
        x_m = x.mean(); y_m = y.mean()
        Sxx = float(np.sum((x - x_m) ** 2))
        if Sxx < 1e-12 or n < 3:
            continue
        beta_j = float(np.sum((x - x_m) * (y - y_m)) / Sxx)
        # per-user global-zero intercept
        alpha_j_global = y_m - beta_j * x_m
        resid = y - (alpha_j_global + beta_j * x)
        sse = float(np.sum(resid ** 2))
        mse_j = sse / (n - 2)
        # within-centered SEs (standard OLS):
        #   SE(a_within)^2 = mse / n
        #   SE(b)^2        = mse / Sxx
        # But we need SE of GLOBAL-zero intercept:
        #   SE(alpha_global)^2 = mse * (1/n + x_m^2 / Sxx)
        # Use within-centered SE for (a_within, b) as the INDEPENDENT pair.
        se_a_within_sq = mse_j / n
        se_b_sq = mse_j / Sxx
        # Also within-user variance of x:
        varx_within = Sxx / n  # = E_within[(x - xbar)^2]
        rows.append({
            "uid": uid,
            "n_j": n,
            "alpha_j_global": alpha_j_global,
            "a_j_within": y_m,        # within-centered intercept (alpha at x=xbar_j)
            "beta_j": beta_j,
            "se_a_within_sq": se_a_within_sq,
            "se_b_sq": se_b_sq,
            "varx_within_j": varx_within,
            "xbar_j": x_m,
        })
        s_within_sum += sse
        df_within += n - 2
        pop_resid = y - (alpha_pop + beta_pop * x)
        s_popbaseline_sum += float(np.sum(pop_resid ** 2))
        df_popbaseline += n
        all_x_sq_sum += float(np.sum(x ** 2))
        all_x_sum += float(np.sum(x))
        n_total += n

    ru = pd.DataFrame(rows)
    sigma_eps_sq = s_within_sum / df_within if df_within else float("nan")
    mse_pop_baseline = (s_popbaseline_sum / df_popbaseline
                        if df_popbaseline else float(sigma_eps_sq))

    # MoM: use INDEPENDENT within-centered (a, b) for MoM estimation
    a_within = ru["a_j_within"].to_numpy()
    beta = ru["beta_j"].to_numpy()
    se_a_sq = ru["se_a_within_sq"].to_numpy()
    se_b_sq = ru["se_b_sq"].to_numpy()
    tau_a_within_sq = max(float(a_within.var(ddof=1)) - float(np.mean(se_a_sq)), 0.0)
    tau_beta_sq = max(float(beta.var(ddof=1)) - float(np.mean(se_b_sq)), 0.0)

    # tau_alpha^2 (global-zero intercept) = Var(a_within - b * xbar)
    # Under independence of true (a_within, beta) and constant xbar_j across
    # j distribution: Var(alpha_global_true) = tau_a_within^2 + E[xbar_j]^2 * tau_beta^2
    #                                          - 2 E[xbar_j] * Cov(a_within, beta).
    # For the risk-gap theorem we use the within-centered parameterization, in
    # which the intercept risk is tau_a_within^2 (what matters for prediction at
    # the user's own xbar_j). For PREDICTION at a new global x = x0 the total
    # risk is what we decompose below with the rmbar2 weighting.
    #
    # For consistency with the 4-estimator summary (which reports tau_alpha^2 at global zero
    # via MoM on alpha_hat_j with SE^2 = mse*(1/n + xbar^2/Sxx)), also compute:
    alpha_global = ru["alpha_j_global"].to_numpy()
    se_alpha_global_sq = se_a_sq + (ru["xbar_j"].to_numpy() ** 2) * se_b_sq
    tau_alpha_global_sq = max(float(alpha_global.var(ddof=1))
                              - float(np.mean(se_alpha_global_sq)), 0.0)

    x_bar_global = all_x_sum / n_total if n_total else 0.0
    rmbar2 = all_x_sq_sum / n_total if n_total else 0.0  # E[x^2]
    varx_global = rmbar2 - x_bar_global ** 2

    return {
        "tau_a_within_sq": tau_a_within_sq,
        "tau_alpha_global_sq": tau_alpha_global_sq,
        "tau_beta_sq": tau_beta_sq,
        "sigma_eps_sq": float(sigma_eps_sq),
        "mse_pop_baseline": float(mse_pop_baseline),
        "alpha_pop": float(alpha_pop),
        "beta_pop": float(beta_pop),
        "n_users": int(len(ru)),
        "n_j": ru["n_j"].astype(int).tolist(),
        "varx_within_j": ru["varx_within_j"].tolist(),
        "xbar_j": ru["xbar_j"].tolist(),
        "rmbar2_global": float(rmbar2),
        "x_bar_global": float(x_bar_global),
        "varx_global": float(varx_global),
        "n_total": int(n_total),
    }


def predict_2param(fit: dict) -> dict:
    """Apply the two-parameter Morris g theorem and a one-parameter baseline.

    We report TWO variants of the slope contribution:
      (main) rmbar2-weighted: dMSE_beta = tau_b * E[x^2] * g(r_beta).
             This corresponds to prediction-risk at the GLOBAL x-distribution
             (random test point drawn from population x).
      (within-centered) varx_within-weighted: dMSE_beta = tau_b * E_j[Var_within(x_j)] * g(r_beta).
             This corresponds to prediction-risk at the USER'S OWN x-distribution,
             which matches leave-one-row-out CV within-user.

    The CV RMSE metric in the published PEBS gains evaluates held-out rows at
    the user's own x-distribution, so the within-centered form is the right
    comparator for published %-gain numbers. We report both.
    """
    tau_a = fit["tau_a_within_sq"]           # within-centered intercept variance
    tau_a_glob = fit["tau_alpha_global_sq"]  # global-zero intercept variance (for 1p ref)
    tau_b = fit["tau_beta_sq"]
    sigma2 = fit["sigma_eps_sq"]
    mse_pop = fit["mse_pop_baseline"]
    rmbar2 = fit["rmbar2_global"]
    varx_global = fit["varx_global"]
    ns = np.array(fit["n_j"], dtype=float)
    varx_within_j = np.array(fit["varx_within_j"], dtype=float)

    r_alpha = ns * tau_a / max(sigma2, 1e-18)
    r_beta = ns * tau_b * varx_within_j / max(sigma2, 1e-18)

    weights = ns / ns.sum()
    g_alpha = g(r_alpha)
    g_beta = g(r_beta)

    dMSE_alpha = tau_a * float((weights * g_alpha).sum())
    dMSE_beta_global = tau_b * rmbar2 * float((weights * g_beta).sum())
    dMSE_beta_withinx = tau_b * float((weights * varx_within_j * g_beta).sum())

    # MAIN (within-user CV evaluation — matches PEBS reported RMSE):
    dMSE_2param_CV = dMSE_alpha + dMSE_beta_withinx
    # Alternative (global x-distribution):
    dMSE_2param_global = dMSE_alpha + dMSE_beta_global

    def _rel_imp(dmse):
        """Subtractive form: 1 - sqrt( (mse_pop - dmse) / mse_pop ). NaN if dmse > mse_pop."""
        mse_eb = mse_pop - dmse
        if mse_eb <= 0:
            return float("nan"), True
        return 1.0 - float(np.sqrt(mse_eb / mse_pop)), False

    def _rel_imp_ratio(dmse_alpha, dmse_beta_cv):
        """Ratio form (one-parameter style):
          MSE_EB_j  = tau_a/(1+r_a_j) + varx_within_j * tau_b/(1+r_b_j) + sigma^2
          rel_imp = 1 - sqrt( mean_j(n_j * MSE_EB_j / sum_n) / mse_pop ).
        More stable when MoM over-estimates tau (MultiPref case).
        """
        mse_eb_j = (
            tau_a / (1.0 + r_alpha)
            + varx_within_j * tau_b / (1.0 + r_beta)
            + sigma2
        )
        mse_eb_weighted = float((weights * mse_eb_j).sum())
        if mse_eb_weighted <= 0 or mse_pop <= 0:
            return float("nan")
        return 1.0 - float(np.sqrt(mse_eb_weighted / mse_pop))

    ri_2p_cv, clip_2p_cv = _rel_imp(dMSE_2param_CV)
    ri_2p_g, clip_2p_g = _rel_imp(dMSE_2param_global)
    ri_2p_ratio = _rel_imp_ratio(dMSE_alpha, dMSE_beta_withinx)

    # 1-parameter reference
    r_alpha_1p = ns * tau_a_glob / max(sigma2, 1e-18)
    g_alpha_1p = g(r_alpha_1p)
    dMSE_1param = tau_a_glob * float((weights * g_alpha_1p).sum())
    ri_1p, clip_1p = _rel_imp(dMSE_1param)
    # 1-parameter RATIO form (baseline reproduction):
    mse_eb_1p_j = tau_a_glob / (1.0 + r_alpha_1p) + sigma2
    mse_eb_1p_w = float((weights * mse_eb_1p_j).sum())
    ri_1p_ratio = (1.0 - float(np.sqrt(mse_eb_1p_w / mse_pop))
                   if mse_eb_1p_w > 0 else float("nan"))

    return {
        # MAIN prediction: within-user-CV evaluation (ratio form — stable to MoM inflation)
        "rel_imp_pct_2param": (100.0 * ri_2p_ratio) if not np.isnan(ri_2p_ratio) else float("nan"),
        "rel_imp_pct_2param_subtractive_cv": (100.0 * ri_2p_cv) if not np.isnan(ri_2p_cv) else float("nan"),
        "rel_imp_pct_2param_global_x": (100.0 * ri_2p_g) if not np.isnan(ri_2p_g) else float("nan"),
        "rel_imp_pct_1param_ref": (100.0 * ri_1p_ratio) if not np.isnan(ri_1p_ratio) else float("nan"),
        "rel_imp_pct_1param_ref_subtractive": (100.0 * ri_1p) if not np.isnan(ri_1p) else float("nan"),
        "pred_clipped_2param_cv": bool(clip_2p_cv),
        "pred_clipped_2param_global": bool(clip_2p_g),
        "pred_clipped_1param": bool(clip_1p),
        "dMSE_alpha": float(dMSE_alpha),
        "dMSE_beta_withinx": float(dMSE_beta_withinx),
        "dMSE_beta_global": float(dMSE_beta_global),
        "dMSE_2param_CV": float(dMSE_2param_CV),
        "dMSE_2param_global": float(dMSE_2param_global),
        "dMSE_1param_ref": float(dMSE_1param),
        "mean_r_alpha": float((weights * r_alpha).sum()),
        "mean_r_beta": float((weights * r_beta).sum()),
        "median_r_alpha": float(np.median(r_alpha)),
        "median_r_beta": float(np.median(r_beta)),
        "mean_omega_alpha": float((weights * g_alpha).sum()),
        "mean_omega_beta": float((weights * g_beta).sum()),
        "re_model_violated": bool(
            (dMSE_alpha + max(dMSE_beta_withinx, dMSE_beta_global)) >= mse_pop
        ),
    }


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #
def load_prism(path: str):
    df = pd.read_parquet(path)
    return df[["user_id", "score_user", "rm_score"]].dropna(), "score_user", "rm_score"


def load_pluriharms(path: str):
    df = pd.read_parquet(path)
    return df[["user_id", "rating", "Harm_Level"]].dropna(), "rating", "Harm_Level"


def load_multipref(path: str):
    df = pd.read_parquet(path)
    return df[["user_id", "quality", "overall_conf"]].dropna(), "quality", "overall_conf"


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prism-parquet", default="data/prism_rm_scored.parquet")
    p.add_argument("--pluriharms-parquet", default="data/pluriharms_long.parquet")
    p.add_argument("--multipref-parquet",
                   default="../3_PEBS_Standalone/data/multipref_evaluator_quality.parquet")
    p.add_argument("--output-dir", default="results/track1_morris_g_2param_validate")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    root = Path(__file__).resolve().parents[1]
    prism_path = str(root / args.prism_parquet) if not args.prism_parquet.startswith("/") else args.prism_parquet
    ph_path = str(root / args.pluriharms_parquet) if not args.pluriharms_parquet.startswith("/") else args.pluriharms_parquet
    mp_path = str(Path(root, args.multipref_parquet).resolve()) if not args.multipref_parquet.startswith("/") else args.multipref_parquet

    corpora = {}
    observed = {"PRISM": 8.58, "PluriHarms": 8.64, "MultiPref": 0.47}

    for name, (loader, path, min_obs) in {
        "PRISM": (load_prism, prism_path, 6),
        "PluriHarms": (load_pluriharms, ph_path, 10),
        "MultiPref": (load_multipref, mp_path, 10),
    }.items():
        try:
            df, y_col, x_col = loader(path)
            fit = fit_two_param(df, y_col, x_col, min_obs=min_obs)
            pred = predict_2param(fit)
            # Strip large arrays for JSON
            fit_summary = {k: v for k, v in fit.items()
                           if k not in ("n_j", "varx_within_j", "xbar_j")}
            corpora[name] = {
                "fit": fit_summary,
                "predicted": pred,
                "observed_pebs_gain_pct": observed[name],
            }
        except Exception as e:
            corpora[name] = {"error": str(e)}

    # Build the headline table
    table = []
    for name, rec in corpora.items():
        if "error" in rec:
            table.append({"corpus": name, "error": rec["error"]})
            continue
        p = rec["predicted"]
        f = rec["fit"]
        pred_2p_cv = p["rel_imp_pct_2param"]
        pred_2p_g = p["rel_imp_pct_2param_global_x"]
        pred_1p = p["rel_imp_pct_1param_ref"]
        obs = rec["observed_pebs_gain_pct"]
        def _round_or_nan(v, d):
            return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(v, d)
        def _delta(v, o):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return round(v - o, 3)
        table.append({
            "corpus": name,
            "n_users": f["n_users"],
            "tau_alpha_sq_within": round(f["tau_a_within_sq"], 5),
            "tau_alpha_sq_global": round(f["tau_alpha_global_sq"], 5),
            "tau_beta_sq": round(f["tau_beta_sq"], 6),
            "sigma_eps_sq": round(f["sigma_eps_sq"], 4),
            "rmbar2": round(f["rmbar2_global"], 5),
            "mean_r_alpha": round(p["mean_r_alpha"], 3),
            "mean_r_beta": round(p["mean_r_beta"], 3),
            "mean_omega_alpha": round(p["mean_omega_alpha"], 4),
            "mean_omega_beta": round(p["mean_omega_beta"], 4),
            "pred_1param_pct": _round_or_nan(pred_1p, 3),
            "pred_2param_pct_CV": _round_or_nan(pred_2p_cv, 3),
            "pred_2param_pct_global_x": _round_or_nan(pred_2p_g, 3),
            "observed_pct": obs,
            "delta_1p_pp": _delta(pred_1p, obs),
            "delta_2p_cv_pp": _delta(pred_2p_cv, obs),
            "delta_2p_global_pp": _delta(pred_2p_g, obs),
            "abs_delta_1p_pp": _round_or_nan(abs(pred_1p - obs) if pred_1p is not None and not (isinstance(pred_1p, float) and np.isnan(pred_1p)) else None, 3),
            "abs_delta_2p_cv_pp": _round_or_nan(abs(pred_2p_cv - obs) if pred_2p_cv is not None and not (isinstance(pred_2p_cv, float) and np.isnan(pred_2p_cv)) else None, 3),
            "re_model_violated": p.get("re_model_violated", False),
        })

    # Closure check: did 2-param CLOSE the MultiPref gap?
    mp_row = [r for r in table if r.get("corpus") == "MultiPref"]
    mp_closure = None
    if mp_row:
        mr = mp_row[0]
        d1 = mr.get("abs_delta_1p_pp")
        d2 = mr.get("abs_delta_2p_cv_pp")
        mp_closure = {
            "abs_delta_1p_pp": d1,
            "abs_delta_2p_cv_pp": d2,
            "re_model_violated": mr.get("re_model_violated", False),
            "within_2pp_after_2param": d2 is not None and d2 <= 2.0,
            "honest_disclosure": (
                "MultiPref MoM tau estimates inflated (tau_alpha^2 > mse_pop_baseline) "
                "- Gaussian RE assumption violated; 2-param g over-predicts."
            ) if mr.get("re_model_violated", False) else None,
        }

    def _abs_le(r, key, thresh):
        v = r.get(key)
        return v is not None and not (isinstance(v, float) and np.isnan(v)) and v <= thresh

    within_1pp_2p = sum(1 for r in table if _abs_le(r, "abs_delta_2p_cv_pp", 1.0))
    within_2pp_2p = sum(1 for r in table if _abs_le(r, "abs_delta_2p_cv_pp", 2.0))

    out = {
        "theorem": (
            "Two-parameter Morris g: E[POP-risk - EB-risk] = "
            "tau_alpha^2 * g(r_alpha) + tau_beta^2 * E[x^2] * g(r_beta), "
            "g(r)=r/(1+r), under Gaussian RE with within-user-centered design."
        ),
        "corpora": corpora,
        "table": table,
        "multipref_closure": mp_closure,
        "summary": {
            "n_corpora_validated": sum(1 for r in table if "error" not in r),
            "within_1pp_2param": within_1pp_2p,
            "within_2pp_2param": within_2pp_2p,
        },
        "wall_seconds": time.time() - t0,
    }

    outdir = root / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[morris_g_2param_validate] wrote {outdir/'summary.json'}")
    print(f"[morris_g_2param_validate] wall = {out['wall_seconds']:.1f}s\n")
    print(f"{'Corpus':12s} {'pred_1p%':>10s} {'pred_2p_CV%':>12s} {'pred_2p_g%':>12s} "
          f"{'obs%':>8s} {'Δ_1p':>8s} {'Δ_2p_cv':>10s} {'violated':>10s}")
    for r in table:
        if "error" in r:
            print(f"  {r['corpus']}: ERROR {r['error']}")
            continue
        p1 = r['pred_1param_pct']
        p2c = r['pred_2param_pct_CV']
        p2g = r['pred_2param_pct_global_x']
        d1 = r['delta_1p_pp']
        d2 = r['delta_2p_cv_pp']
        fmt = lambda v, w: (f"{v:>{w}.3f}" if isinstance(v, (int, float)) and v is not None and not (isinstance(v, float) and np.isnan(v)) else " " * (w - 3) + "NaN")
        print(f"{r['corpus']:12s} {fmt(p1,10)} {fmt(p2c,12)} {fmt(p2g,12)} "
              f"{r['observed_pct']:8.3f} {fmt(d1,8)} {fmt(d2,10)} {str(r['re_model_violated']):>10s}")

    if mp_closure:
        d1 = mp_closure['abs_delta_1p_pp']
        d2 = mp_closure['abs_delta_2p_cv_pp']
        d1s = f"{d1:.2f}" if d1 is not None else "NaN"
        d2s = f"{d2:.2f}" if d2 is not None else "NaN"
        print(f"\nMultiPref closure: |Δ_1p|={d1s}pp → |Δ_2p_cv|={d2s}pp "
              f"(within 2pp: {mp_closure['within_2pp_after_2param']}, "
              f"RE violated: {mp_closure['re_model_violated']})")
        if mp_closure.get("honest_disclosure"):
            print(f"  HONEST DISCLOSURE: {mp_closure['honest_disclosure']}")


if __name__ == "__main__":
    main()
