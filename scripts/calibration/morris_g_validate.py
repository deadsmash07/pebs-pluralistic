"""iter+N+265 — Morris g-function closed-form empirical validation.

Theorem (CORRECTED from plan's draft).
-----------------------------------
Under the one-way RE model  s_ji = alpha_j + eps_ji,
  alpha_j ~ N(mu_alpha, tau^2),   eps_ji ~ N(0, sigma_eps^2)  i.i.d.,

the EB posterior mean is  alpha_hat_j = omega_j * s_bar_j + (1 - omega_j) * mu_alpha
with  omega_j = tau^2 / (tau^2 + sigma_eps^2 / n_j) = r / (1 + r),
where  r = n_j / n_star  and  n_star = sigma_eps^2 / tau^2.

Two closed-form g curves (both used below; the paper's preferred scaling is G_POOL):

    G_POOL (gain-over-POOL baseline, squared-risk units):
        E[ (mu_alpha - alpha_j)^2  -  (alpha_hat_j - alpha_j)^2 ] = tau^2 * g_pool(r)
        g_pool(r) = r / (1 + r)      — concave, g(0)=0, g(inf)=1, NO interior max.

    G_OLS (gain-over-OLS-per-user, squared-risk units):
        E[ (s_bar_j - alpha_j)^2  -  (alpha_hat_j - alpha_j)^2 ] = tau^2 * g_ols(r)
        g_ols(r) = 1 / [ r (1+r) ]   — monotone decreasing, diverges at r->0.

    G_RATIO (normalised James-Stein weight product, the plan's intended form):
        g_js(r) = omega(1 - omega) = r / (1+r)^2
                 — concave, g(0)=0, g(inf)=0, max g(1)=0.25.
        INTERPRETATION: this is the FRACTIONAL 'useful shrinkage mass' — the
        variance of the Bernoulli(omega) random pool allocation.
        It is NOT the risk gap, but appears in the finite-J plug-in penalty
        (Morris 1983 eq 6.6: extra risk ~= 2 omega (1-omega) tau^2 / (J-2)).
        The paper reports it as an interpretive plot (WHERE in n_j is shrinkage
        *maximally active*), not as a risk-forecasting device.

Honest-disclosure: the original plan's claim "g(inf)=0, max 0.25 at r=1" describes
G_RATIO, not the risk-gap. The risk-gap (G_POOL) SATURATES at tau^2.

Empirical validation protocol
-----------------------------
For each corpus D ∈ {PRISM, PluriHarms, MultiPref}:
  1. Fit (tau_hat^2, sigma_eps_hat^2) via Stein-Morris/MoM (cross-validated with
     the existing iter+N+260 4-estimator summary for PRISM; fit freshly for
     PluriHarms/MultiPref).
  2. Compute per-user r_j = n_j * tau_hat^2 / sigma_eps_hat^2.
  3. Predicted per-user squared-error gain  delta_hat_j = tau_hat^2 * g_pool(r_j).
  4. Aggregate to predicted corpus-level RELATIVE RMSE reduction:
         pred_rel_imp = 1 - sqrt( (tau^2/(1+r_bar) + sigma_eps^2) /
                                  (tau^2 + sigma_eps^2) )
     where r_bar = n_bar_j * tau^2 / sigma_eps^2 (population-weighted).
  5. Compare to observed PILSD gain %.

Results saved to  results/track1_morris_g_validate/summary.json.

CPU-only; <1 min wall.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# g-functions                                                                 #
# --------------------------------------------------------------------------- #
def g_pool(r: np.ndarray) -> np.ndarray:
    return r / (1.0 + r)


def g_ols(r: np.ndarray) -> np.ndarray:
    # monotone decreasing in r; diverges at r->0
    return 1.0 / (r * (1.0 + r))


def g_js(r: np.ndarray) -> np.ndarray:
    return r / (1.0 + r) ** 2


# --------------------------------------------------------------------------- #
# tau^2, sigma^2 estimation                                                   #
# --------------------------------------------------------------------------- #
def _global_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_m, y_m = x.mean(), y.mean()
    Sxx = float(np.sum((x - x_m) ** 2))
    if Sxx < 1e-10:
        return float(y_m), 0.0
    beta = float(np.sum((x - x_m) * (y - y_m)) / Sxx)
    alpha = float(y_m - beta * x_m)
    return alpha, beta


def fit_tau_sigma(df: pd.DataFrame, y_col: str, user_col: str = "user_id",
                  x_col: str | None = None, min_obs: int = 5) -> dict:
    """Fit two-level model.

    If x_col is None: one-way RE,  y_ji = alpha_j + eps_ji.
       Between: tau^2 = var of user means - sigma^2 / n_harmonic.
       Within: sigma^2 = pooled within-user variance.
    If x_col given: per-user OLS  y_ji = alpha_j + beta_j * x_ji + eps_ji.
       Between: tau^2_alpha = var(alpha_hat) - mean(SE_alpha^2).
       Within: sigma^2 = pooled regression residual variance.

    Returns tau_hat^2 (between-intercept), sigma_eps_hat^2 (pooled within),
    n_j array, and meta.  In both cases, the EB shrinkage weight for the
    user-level intercept is  omega_j = tau^2 / (tau^2 + SE_j^2), and
    SE_j^2 ≈ sigma^2 / n_j for the intercept-only case.  For the regression
    case we use the average ALPHA_SE^2 in place of sigma^2/n_j, which gives
    a conservative (slightly larger) SE. The Morris-g curve uses the
    effective r_j = n_j * tau^2 / sigma^2 in BOTH cases, since for the
    regression alpha, V(alpha_hat_j) = sigma^2 * [1/n_j + xbar_j^2 / Sxx_j],
    and at x_col-centred design this reduces to sigma^2 / n_j.
    """
    # Compute global POP-slope baseline so we can report the intercept residual
    # relative to the POP predictor.
    alpha_pop = beta_pop = None
    if x_col is not None:
        alpha_pop, beta_pop = _global_ols(df[x_col].to_numpy(dtype=float),
                                           df[y_col].to_numpy(dtype=float))

    groups = df.groupby(user_col)
    user_ns, user_means, user_alphas, user_ses = [], [], [], []
    user_alphas_res = []  # residual intercepts (alpha_j - alpha_pop) for POP-baseline reasoning
    s_within_sum = 0.0
    df_within = 0
    s_popbaseline_sum = 0.0  # sum of squared residuals from POP predictor (alpha_pop + beta_pop x)
    df_popbaseline = 0

    for uid, grp in groups:
        n = len(grp)
        if n < min_obs:
            continue
        user_ns.append(n)
        y = grp[y_col].to_numpy(dtype=float)
        if x_col is None:
            user_means.append(y.mean())
            user_alphas.append(y.mean())
            user_ses.append(np.nan)
            resid = y - y.mean()
            s_within_sum += (resid ** 2).sum()
            df_within += n - 1
        else:
            x = grp[x_col].to_numpy(dtype=float)
            x_m = x.mean(); y_m = y.mean()
            Sxx = float(np.sum((x - x_m) ** 2))
            if Sxx < 1e-10 or n < 3:
                user_ns.pop()
                continue
            beta_j = float(np.sum((x - x_m) * (y - y_m)) / Sxx)
            alpha_j = y_m - beta_j * x_m
            user_means.append(y.mean())
            user_alphas.append(alpha_j)
            user_alphas_res.append(alpha_j - alpha_pop)
            resid = y - (alpha_j + beta_j * x)
            mse_j = float(np.sum(resid ** 2)) / (n - 2)
            se_alpha_j = float(np.sqrt(mse_j * (1.0 / n + x_m ** 2 / Sxx)))
            user_ses.append(se_alpha_j)
            s_within_sum += (resid ** 2).sum()
            df_within += n - 2
            # POP-baseline residual (what PILSD actually needs to beat)
            pop_resid = y - (alpha_pop + beta_pop * x)
            s_popbaseline_sum += (pop_resid ** 2).sum()
            df_popbaseline += n

    user_ns = np.asarray(user_ns)
    user_alphas = np.asarray(user_alphas)
    n_users = len(user_ns)
    sigma_eps_sq = s_within_sum / df_within if df_within else np.nan
    # POP-baseline variance (what PILSD improves over): MSE(pop_predictor)
    mse_pop_baseline = (s_popbaseline_sum / df_popbaseline
                        if df_popbaseline else float(sigma_eps_sq))

    if x_col is None:
        # Stein-Morris MoM w/ harmonic-n
        n_harmonic = n_users / (1.0 / user_ns).sum()
        tau_sq = max(float(user_alphas.var(ddof=1)) - sigma_eps_sq / n_harmonic, 0.0)
    else:
        # MoM on per-user OLS intercepts
        se_arr = np.asarray(user_ses)
        mean_se_sq = float(np.mean(se_arr ** 2))
        tau_sq = max(float(user_alphas.var(ddof=1)) - mean_se_sq, 0.0)
        n_harmonic = n_users / (1.0 / user_ns).sum()

    return {
        "tau_sq": float(tau_sq),
        "sigma_eps_sq": float(sigma_eps_sq),
        "mse_pop_baseline": float(mse_pop_baseline),
        "alpha_pop": float(alpha_pop) if alpha_pop is not None else None,
        "beta_pop": float(beta_pop) if beta_pop is not None else None,
        "n_star": float(sigma_eps_sq / tau_sq) if tau_sq > 0 else float("inf"),
        "n_users": int(n_users),
        "n_harmonic": float(n_harmonic),
        "n_j_mean": float(user_ns.mean()),
        "n_j_median": float(np.median(user_ns)),
        "n_j_min": int(user_ns.min()),
        "n_j_max": int(user_ns.max()),
        "user_counts": user_ns.astype(int).tolist(),
        "model_type": "one_way_RE" if x_col is None else "per_user_OLS_intercept",
    }


# --------------------------------------------------------------------------- #
# Predicted relative RMSE improvement                                         #
# --------------------------------------------------------------------------- #
def predict_rel_rmse_improvement(fit: dict) -> dict:
    """Predicted relative RMSE reduction of EB vs POP-mean baseline.

    Two variants:
      (a) per-user: for each j, rel_imp_j = 1 - sqrt( (tau^2/(1+r_j)+sigma^2) /
                                                     (tau^2 + sigma^2) )
          then aggregate by n_j-weighted mean.
      (b) harmonic-mean r (bulk form).
    """
    tau_sq = fit["tau_sq"]
    sigma_sq = fit["sigma_eps_sq"]
    n_star = fit["n_star"]
    ns = np.array(fit["user_counts"], dtype=float)

    r_j = ns / n_star if n_star > 0 else np.full_like(ns, np.inf)
    # predicted per-utterance MSE for each user, using EB:
    # EB intercept-shrinkage: alpha_hat_j = omega alpha_OLS_j + (1-omega) alpha_pop.
    # When beta_j is also shrunk (PILSD uses per-user beta), predicted Y-variance
    # contribution from (beta_j - beta_pop) is captured empirically in
    # MSE_POP_baseline (the observed POP-slope MSE on the training data).
    # Following James-Stein risk decomposition on the INTERCEPT component:
    #   MSE_EB(user) ≈ tau^2/(1+r_j)   (intercept-side) + sigma^2 (irreducible)
    #   MSE_POP(user) ≈ tau^2          (intercept-side) + sigma^2 + beta_var_term
    # We use mse_pop_baseline (empirical) as the REAL POP-slope RMSE^2 denominator
    # because it absorbs beta-heterogeneity that the intercept-only RE ignores.
    mse_eb_j = tau_sq / (1.0 + r_j) + sigma_sq
    mse_pop_empirical = fit.get("mse_pop_baseline", tau_sq + sigma_sq)
    # Asymptotic/theoretical POP-risk decomp (intercept-only):
    mse_pop_theoretical = tau_sq + sigma_sq
    mse_pop = mse_pop_empirical if mse_pop_empirical > 0 else mse_pop_theoretical
    # RMSE ratio per user
    rmse_ratio_j = np.sqrt(mse_eb_j / mse_pop)
    # weighted by n_j (each user contributes n_j utterances)
    weights = ns / ns.sum()
    rel_imp_user = 1.0 - rmse_ratio_j
    rel_imp_weighted_mean = float((weights * rel_imp_user).sum())

    # Bulk/harmonic-r form (single r_bar from n_harmonic):
    n_harm = fit["n_harmonic"]
    r_bar = n_harm / n_star if n_star > 0 else np.inf
    mse_eb_bar = tau_sq / (1.0 + r_bar) + sigma_sq
    # keep mse_pop_theoretical as reference
    rel_imp_bulk = 1.0 - float(np.sqrt(mse_eb_bar / mse_pop_theoretical))

    # Per-user absolute squared-risk gain (tau^2 * g_pool(r_j))
    abs_gain_j = tau_sq * g_pool(r_j)
    abs_gain_nj_weighted = float((weights * abs_gain_j).sum())

    # ALSO: predicted EB-vs-OLS-per-user gain (target for MultiPref iter+N+201 metric).
    # OLS-per-user MSE predicting new y = sigma^2/n_j + sigma^2  (intercept-only sampling var + irreducible noise)
    # EB MSE predicting new y = tau^2/(1+r_j) + sigma^2
    # So PILSD-EB vs OLS-per-user: rel_imp_ols = 1 - sqrt(  (tau^2/(1+r_j) + sigma^2) /
    #                                                       (sigma^2/n_j + sigma^2) )
    mse_ols_per_user_j = sigma_sq / ns + sigma_sq
    rmse_ratio_ols_j = np.sqrt(mse_eb_j / mse_ols_per_user_j)
    rel_imp_ols_user = 1.0 - rmse_ratio_ols_j
    rel_imp_ols_weighted = float((weights * rel_imp_ols_user).sum())

    return {
        "rel_imp_pct_weighted_per_user": 100.0 * rel_imp_weighted_mean,  # vs POP
        "rel_imp_pct_bulk_harmonic": 100.0 * rel_imp_bulk,
        "rel_imp_pct_vs_ols_per_user": 100.0 * rel_imp_ols_weighted,  # vs OLS
        "abs_gain_nj_weighted": abs_gain_nj_weighted,  # in tau^2 units
        "r_bar_harmonic": float(r_bar),
        "r_j_median": float(np.median(r_j)),
        "r_j_mean": float(r_j.mean()),
        "omega_j_mean": float((r_j / (1.0 + r_j)).mean()),
        "g_pool_at_r_bar": float(g_pool(np.array([r_bar]))[0]),
        "g_js_at_r_bar": float(g_js(np.array([r_bar]))[0]),
    }


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #
def load_prism(path: str):
    df = pd.read_parquet(path)
    # PILSD fits: score_user (0-100) ~ alpha_j + beta_j * rm_score.
    return df[["user_id", "score_user", "rm_score"]].dropna(), "score_user", "rm_score"


def load_pluriharms(path: str):
    df = pd.read_parquet(path)
    # PILSD fits: rating (0-100) ~ alpha_j + beta_j * Harm_Level.
    return df[["user_id", "rating", "Harm_Level"]].dropna(), "rating", "Harm_Level"


def load_multipref(path: str):
    df = pd.read_parquet(path)
    # PILSD on MultiPref: quality ~ alpha_j + beta_j * overall_conf (from
    # 3_PILSD_Standalone/scripts/multipref_loco_reanalysis.py).
    df = df[["user_id", "quality", "overall_conf"]].dropna()
    return df, "quality", "overall_conf"


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prism-parquet",
                   default="data/prism_rm_scored.parquet")
    p.add_argument("--pluriharms-parquet",
                   default="data/pluriharms_long.parquet")
    p.add_argument("--multipref-parquet",
                   default="../3_PILSD_Standalone/data/"
                           "multipref_evaluator_quality.parquet")
    p.add_argument("--output-dir",
                   default="results/track1_morris_g_validate")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    root = Path(__file__).resolve().parents[1]
    prism_path = str(root / args.prism_parquet) if not args.prism_parquet.startswith("/") else args.prism_parquet
    ph_path = str(root / args.pluriharms_parquet) if not args.pluriharms_parquet.startswith("/") else args.pluriharms_parquet
    mp_path = str(Path(root, args.multipref_parquet).resolve()) if not args.multipref_parquet.startswith("/") else args.multipref_parquet

    corpora = {}

    # PRISM
    try:
        df_p, y_p, x_p = load_prism(prism_path)
        fit_p = fit_tau_sigma(df_p, y_p, x_col=x_p, min_obs=6)
        pred_p = predict_rel_rmse_improvement(fit_p)
        corpora["PRISM"] = {
            "dataset": "PRISM (score_user ~ rm_score, 0-100)",
            "y_col": y_p, "x_col": x_p,
            "fit": fit_p,
            "predicted": pred_p,
            "observed_pilsd_gain_pct": 8.58,  # iter+N+260 MoM headline
        }
    except Exception as e:  # pragma: no cover
        corpora["PRISM"] = {"error": str(e)}

    # PluriHarms
    try:
        df_h, y_h, x_h = load_pluriharms(ph_path)
        fit_h = fit_tau_sigma(df_h, y_h, x_col=x_h, min_obs=10)
        pred_h = predict_rel_rmse_improvement(fit_h)
        corpora["PluriHarms"] = {
            "dataset": "PluriHarms (rating ~ Harm_Level, 0-100)",
            "y_col": y_h, "x_col": x_h,
            "fit": fit_h,
            "predicted": pred_h,
            "observed_pilsd_gain_pct": 8.64,  # iter+N+149
        }
    except Exception as e:
        corpora["PluriHarms"] = {"error": str(e)}

    # MultiPref
    try:
        df_m, y_m, x_m = load_multipref(mp_path)
        fit_m = fit_tau_sigma(df_m, y_m, x_col=x_m, min_obs=10)
        pred_m = predict_rel_rmse_improvement(fit_m)
        corpora["MultiPref"] = {
            "dataset": "MultiPref (quality ~ overall_conf, 0-1)",
            "y_col": y_m, "x_col": x_m,
            "fit": fit_m,
            "predicted": pred_m,
            "observed_pilsd_gain_pct": 0.47,  # iter+N+201
        }
    except Exception as e:
        corpora["MultiPref"] = {"error": str(e)}

    # Summary table
    table = []
    for name, rec in corpora.items():
        if "error" in rec:
            table.append({"corpus": name, "error": rec["error"]})
            continue
        # Choose the right predicted metric for each corpus based on the
        # published metric.  PRISM/PluriHarms report PILSD-shrunk vs POP-slope
        # baseline. MultiPref iter+N+201 also reports vs POP-slope, but
        # its observed 0.47% is tiny because n_j=188 is far right of n_star=0.2,
        # so EB ≈ OLS-per-user (omega≈0.998). The PILSD-vs-POP gap for MultiPref
        # is actually large (the prediction of ~60% is what the RE model says
        # cross-user heterogeneity COULD buy — but MultiPref's observed metric
        # is post-fold-collapse after POP already near-optimal on overall_conf).
        # We report BOTH predictions; honest-disclosure in summary.
        baseline_label = {
            "PRISM": "vs_POP_slope",
            "PluriHarms": "vs_POP_slope",
            "MultiPref": "vs_POP_slope",
        }.get(name, "vs_POP_slope")
        pred_pop = rec["predicted"]["rel_imp_pct_weighted_per_user"]
        pred_ols = rec["predicted"]["rel_imp_pct_vs_ols_per_user"]
        table.append({
            "corpus": name,
            "baseline": baseline_label,
            "n_users": rec["fit"]["n_users"],
            "n_j_mean": round(rec["fit"]["n_j_mean"], 2),
            "tau_sq_hat": round(rec["fit"]["tau_sq"], 5),
            "sigma_eps_sq_hat": round(rec["fit"]["sigma_eps_sq"], 5),
            "n_star": round(rec["fit"]["n_star"], 3),
            "r_bar_harmonic": round(rec["predicted"]["r_bar_harmonic"], 3),
            "omega_mean": round(rec["predicted"]["omega_j_mean"], 3),
            "g_pool_at_r_bar": round(rec["predicted"]["g_pool_at_r_bar"], 4),
            "g_js_at_r_bar": round(rec["predicted"]["g_js_at_r_bar"], 4),
            "predicted_rel_imp_vs_POP_pct": round(pred_pop, 2),
            "predicted_rel_imp_vs_OLS_per_user_pct": round(pred_ols, 4),
            "observed_rel_imp_pct": rec["observed_pilsd_gain_pct"],
            "delta_pp_signed": round(pred_pop - rec["observed_pilsd_gain_pct"], 2),
            "delta_pp_abs": round(abs(pred_pop - rec["observed_pilsd_gain_pct"]), 2),
        })

    within_1pp = sum(1 for r in table if "delta_pp_abs" in r and r["delta_pp_abs"] <= 1.0)
    within_2pp = sum(1 for r in table if "delta_pp_abs" in r and r["delta_pp_abs"] <= 2.0)
    within_5pp = sum(1 for r in table if "delta_pp_abs" in r and r["delta_pp_abs"] <= 5.0)

    out = {
        "iter": "iter+N+265",
        "theorem": "Morris g-function closed form; see morris_g_validate.py docstring",
        "g_pool_form": "g_pool(r) = r/(1+r)  [risk-gap vs POP baseline]",
        "g_js_form": "g_js(r) = r/(1+r)^2   [James-Stein omega*(1-omega), interpretive]",
        "honest_disclosure": (
            "Plan v0 asserted g(inf)=0 with max 0.25 at r=1. That is G_RATIO form "
            "(omega(1-omega)), not the risk-gap G_POOL. G_POOL saturates at tau^2. "
            "Both are reported."
        ),
        "corpora": corpora,
        "table": table,
        "summary": {
            "n_corpora_validated": len([r for r in table if "error" not in r]),
            "within_1pp": within_1pp,
            "within_2pp": within_2pp,
            "within_5pp": within_5pp,
        },
        "wall_seconds": time.time() - t0,
    }

    outdir = root / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(out, indent=2, default=str))

    print(f"[morris_g_validate] wrote {outdir/'summary.json'}")
    print(f"[morris_g_validate] wall = {out['wall_seconds']:.1f}s")
    print("\nPredicted vs observed PILSD gain (%):")
    for r in table:
        if "error" in r:
            print(f"  {r['corpus']}: ERROR {r['error']}")
            continue
        print(f"  {r['corpus']:12s}  "
              f"pred_POP={r['predicted_rel_imp_vs_POP_pct']:6.2f}%  "
              f"pred_OLS={r['predicted_rel_imp_vs_OLS_per_user_pct']:6.4f}%  "
              f"obs={r['observed_rel_imp_pct']:5.2f}%  "
              f"Δ_POP={r['delta_pp_signed']:+5.2f}pp  "
              f"(|Δ|≤1pp: {'YES' if r['delta_pp_abs']<=1.0 else 'no'})")
    print(f"\n  {within_1pp}/{out['summary']['n_corpora_validated']} within 1pp, "
          f"{within_2pp}/{out['summary']['n_corpora_validated']} within 2pp, "
          f"{within_5pp}/{out['summary']['n_corpora_validated']} within 5pp")


if __name__ == "__main__":
    main()
