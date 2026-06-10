"""REML verification of HelpSteer2 attribute-as-rater PEBS.

Purpose
-------
The HelpSteer2 MoM claim (+15.92% pooled RMSE reduction, CI
[+10.54, +22.42]) relies on a Method-of-Moments (MoM) estimator of
(tau_alpha^2, tau_beta^2) across n=5 HelpSteer2 attributes. MoM at n=5
has ~3 orders of magnitude sampling spread (classical result, Morris
1983 JASA 78:381; also Patterson & Thompson 1971 Biometrika 58:545,
Harville 1977 JASA 72:320). This raises a real numerical risk:
if canonical REML (restricted maximum likelihood) pulls
tau^2_alpha to zero, the headline gain collapses to the Efron-Morris
intercept-only floor (~+6%).

This script runs the canonical REML estimator via statsmodels MixedLM
(reml=True default; statsmodels 0.14.6 confirmed) on the same 1038-row
HelpSteer2 validation set, using the SAME 80/20 row split (seed=42) as
eval_helpsteer2_pebs_attribute.py, and reports:

    ratio_alpha = tau_alpha_reml / tau_alpha_mom
    ratio_beta  = tau_beta_reml  / tau_beta_mom

Pre-committed branches
----------------------
    HOLDS        : ratio_alpha in [0.5, 2.0]  -> +15.92% headline survives
    COLLAPSES    : ratio_alpha < 0.1          -> collapse to ~+6% floor
    INTERMEDIATE : ratio_alpha in [0.1, 0.5]  -> blended / conservative

Model
-----
Long-form: one row per (row_i, attribute_a), 1038*5 = 5190 rows.
Fixed effects:  score ~ 1 + rm_score
Random effects (by attribute): ~ 1 + rm_score   (unstructured 2x2 cov)

In lme4 notation:
    lmer(score ~ rm_score + (1 + rm_score | attribute), REML=TRUE)

In statsmodels:
    MixedLM.from_formula(
        "score ~ rm_score",
        groups="attribute_id",
        re_formula="~rm_score",
        data=long,
    ).fit(reml=True)

The random-intercept variance = cov_re[0, 0] = tau_alpha^2_REML.
The random-slope variance     = cov_re[1, 1] = tau_beta^2_REML.
Off-diagonal = cov(alpha_a, beta_a) across attributes (we report it
but do not use it in the single-axis EB weight).

Refit the PEBS predictions using REML-estimated tau^2 in place of MoM,
keeping the same per-attribute V_alpha, V_beta from OLS. Recompute pooled
RMSE and the rel-impr-vs-pop_slope headline.

Canonical refs
--------------
* statsmodels 0.14.6, MixedLM.fit() reml=True default
  (verified via help(MixedLM.fit) signature line `reml=True`).
* Bates et al. (2015) "Fitting Linear Mixed-Effects Models Using lme4",
  J. Stat. Software 67(1). lmer() REML=TRUE default.
* Morris (1983), "Parametric Empirical Bayes Inference: Theory and
  Applications", JASA 78(381):47-55. Canonical EB weight
  omega_a = tau^2 / (tau^2 + V_a).
* Patterson & Thompson (1971), Biometrika 58:545 - REML origin.
* Harville (1977) JASA 72:320 - REML as Bayesian-motivated marginal
  likelihood eliminating fixed-effects nuisance.

Outputs
-------
  results/track1_helpsteer2_attribute/reml_verification.json

"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQ = ROOT / "data" / "helpsteer2" / "helpsteer2_qwen15b_scored.parquet"
EVAL_JSON = ROOT / "results" / "track1_helpsteer2_attribute" / "eval.json"
OUT_JSON = ROOT / "results" / "track1_helpsteer2_attribute" / "reml_verification.json"

ATTRS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
SEED = 42
TEST_FRAC = 0.2


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape to long-form: one row per (row_i, attribute_a)."""
    rows = []
    for aidx, a in enumerate(ATTRS):
        tmp = pd.DataFrame({
            "row_id": df["row_id"].values if "row_id" in df else np.arange(len(df)),
            "rm_score": df["rm_score"].values,
            "score": df[a].values.astype(float),
            "attribute_id": aidx,
            "attribute": a,
        })
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def fit_reml_mixedlm(long: pd.DataFrame) -> dict:
    """Fit MixedLM with random intercept + random slope by attribute.

    Model: score ~ 1 + rm_score + (1 + rm_score | attribute_id)
    REML=True (canonical default, statsmodels 0.14.6).
    """
    md = MixedLM.from_formula(
        "score ~ rm_score",
        groups="attribute_id",
        re_formula="~rm_score",
        data=long,
    )
    # reml=True is the canonical default. We pass explicitly for clarity.
    res = md.fit(reml=True, method=["lbfgs"])

    # Extract variance components
    # cov_re is 2x2: [[var(alpha), cov], [cov, var(beta)]]
    cov_re = np.asarray(res.cov_re)
    tau_alpha_sq = float(cov_re[0, 0])
    tau_beta_sq = float(cov_re[1, 1])
    cov_alpha_beta = float(cov_re[0, 1])
    rho_ab = float(cov_alpha_beta / np.sqrt(max(tau_alpha_sq * tau_beta_sq, 1e-18)))

    # Residual variance
    sigma2_resid = float(res.scale)

    return {
        "tau_alpha_sq_reml": tau_alpha_sq,
        "tau_beta_sq_reml": tau_beta_sq,
        "cov_alpha_beta_reml": cov_alpha_beta,
        "rho_alpha_beta": rho_ab,
        "sigma2_resid": sigma2_resid,
        "alpha_pop_fixed": float(res.fe_params["Intercept"]),
        "beta_pop_fixed": float(res.fe_params["rm_score"]),
        "converged": bool(res.converged),
        "loglike_reml": float(res.llf),
        "n_obs": int(res.nobs),
        "n_groups": int(res.model.n_groups),
        "cov_re_raw": cov_re.tolist(),
    }


def fit_per_attribute_ols(train: pd.DataFrame) -> dict:
    """Replicate eval_helpsteer2_pebs_attribute.py per-attr OLS."""
    out = {}
    x = train["rm_score"].values
    n = len(train)
    x_mean = float(x.mean())
    xx_c = float(np.sum((x - x_mean) ** 2)) + 1e-12
    for a in ATTRS:
        y = train[a].values
        beta, alpha = np.polyfit(x, y, 1)
        y_hat = alpha + beta * x
        resid = y - y_hat
        s2 = float(np.sum(resid ** 2) / max(n - 2, 1))
        V_alpha = s2 * (1.0 / n + (x_mean ** 2) / xx_c)
        V_beta = s2 / xx_c
        out[a] = {
            "alpha": float(alpha), "beta": float(beta),
            "V_alpha": float(V_alpha), "V_beta": float(V_beta),
            "s2_resid": s2,
        }
    return out


def fit_pop(train: pd.DataFrame) -> tuple[float, float]:
    rows = []
    for a in ATTRS:
        rows.append(pd.DataFrame({
            "rm_score": train["rm_score"].values,
            "y": train[a].values,
        }))
    long = pd.concat(rows, ignore_index=True)
    beta, alpha = np.polyfit(long["rm_score"].values, long["y"].values, 1)
    return float(alpha), float(beta)


def shrink(theta: float, V: float, theta_pop: float, tau_sq: float) -> tuple[float, float]:
    if tau_sq <= 0 or V <= 0:
        return theta_pop, 0.0
    w = tau_sq / (tau_sq + V)
    return w * theta + (1 - w) * theta_pop, w


def evaluate_with_tau(train, test, tau_alpha_sq, tau_beta_sq, label):
    """Recompute pebs_shrunk predictions with supplied tau^2 values."""
    pop_alpha, pop_beta = fit_pop(train)
    per_attr = fit_per_attribute_ols(train)

    # population means across attributes (MoM-style centering)
    alphas = np.array([per_attr[a]["alpha"] for a in ATTRS])
    betas = np.array([per_attr[a]["beta"] for a in ATTRS])
    a_pop = float(alphas.mean())
    b_pop = float(betas.mean())

    x_test = test["rm_score"].values
    sq_err_pop = []
    sq_err_eb = []
    per_attr_rmse = {}
    for a in ATTRS:
        y = test[a].values.astype(float)
        pa = per_attr[a]
        a_eb, _ = shrink(pa["alpha"], pa["V_alpha"], a_pop, tau_alpha_sq)
        b_eb, _ = shrink(pa["beta"], pa["V_beta"], b_pop, tau_beta_sq)
        y_pop = pop_alpha + pop_beta * x_test
        y_eb = a_eb + b_eb * x_test
        sq_err_pop.append((y_pop - y) ** 2)
        sq_err_eb.append((y_eb - y) ** 2)
        per_attr_rmse[a] = {
            "pop_slope": float(np.sqrt(np.mean((y_pop - y) ** 2))),
            "pebs_shrunk": float(np.sqrt(np.mean((y_eb - y) ** 2))),
        }

    rmse_pop = float(np.sqrt(np.mean(np.concatenate(sq_err_pop))))
    rmse_eb = float(np.sqrt(np.mean(np.concatenate(sq_err_eb))))
    rel = 100.0 * (rmse_pop - rmse_eb) / rmse_pop if rmse_pop > 0 else float("nan")

    return {
        "label": label,
        "tau_alpha_sq": tau_alpha_sq,
        "tau_beta_sq": tau_beta_sq,
        "rmse_pop_slope": rmse_pop,
        "rmse_pebs_shrunk": rmse_eb,
        "rel_impr_pct": rel,
        "per_attr_rmse": per_attr_rmse,
    }


def main():
    # Load data
    df = pd.read_parquet(DEFAULT_PARQ)
    print(f"[reml] loaded {len(df)} rows from {DEFAULT_PARQ}")

    # Same 80/20 split (seed=42) as eval_helpsteer2_pebs_attribute.py
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    n_test = int(TEST_FRAC * len(df))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    print(f"[reml] split  train={len(train)}  test={len(test)}")

    # Load MoM baseline eval.json
    with EVAL_JSON.open() as f:
        mom = json.load(f)
    tau_alpha_mom = float(mom["eval"]["tau_alpha_sq"])
    tau_beta_mom = float(mom["eval"]["tau_beta_sq"])
    gain_pct_mom = float(mom["rel_impr_overall_pct_point"])
    print(f"[reml] MoM baseline: tau_alpha^2={tau_alpha_mom:.6f}  "
          f"tau_beta^2={tau_beta_mom:.6f}  gain={gain_pct_mom:+.3f}%")

    # Fit REML MixedLM on TRAIN set (long-form)
    train_long = wide_to_long(train)
    print(f"[reml] long-form train: {len(train_long)} rows, "
          f"{train_long['attribute_id'].nunique()} groups")
    reml = fit_reml_mixedlm(train_long)
    tau_alpha_reml = reml["tau_alpha_sq_reml"]
    tau_beta_reml = reml["tau_beta_sq_reml"]
    print(f"[reml] REML estimates:")
    print(f"       tau_alpha^2 = {tau_alpha_reml:.6f}  "
          f"(MoM {tau_alpha_mom:.6f}; "
          f"ratio {tau_alpha_reml/max(tau_alpha_mom,1e-12):.4f}x)")
    print(f"       tau_beta^2  = {tau_beta_reml:.6f}  "
          f"(MoM {tau_beta_mom:.6f}; "
          f"ratio {tau_beta_reml/max(tau_beta_mom,1e-12):.4f}x)")
    print(f"       rho(alpha,beta) = {reml['rho_alpha_beta']:+.3f}")
    print(f"       sigma2_resid    = {reml['sigma2_resid']:.4f}")
    print(f"       converged = {reml['converged']}, "
          f"loglike_REML = {reml['loglike_reml']:.2f}")

    # Recompute pooled RMSE gain under REML tau^2
    mom_eval = evaluate_with_tau(train, test, tau_alpha_mom, tau_beta_mom, "MoM_rerun")
    reml_eval = evaluate_with_tau(train, test, tau_alpha_reml, tau_beta_reml, "REML")
    print(f"\n[reml] MoM re-run gain  = {mom_eval['rel_impr_pct']:+.3f}%  "
          f"(matches MoM baseline {gain_pct_mom:+.3f}%?)")
    print(f"[reml] REML gain        = {reml_eval['rel_impr_pct']:+.3f}%")

    # Also compute intercept-only floor (tau_beta_sq forced to 0 under REML)
    # and slope-only ceiling
    intercept_only = evaluate_with_tau(train, test, tau_alpha_reml, 0.0,
                                       "REML_intercept_only")
    slope_only = evaluate_with_tau(train, test, 0.0, tau_beta_reml,
                                   "REML_slope_only")
    print(f"[reml] REML intercept-only floor = "
          f"{intercept_only['rel_impr_pct']:+.3f}%")
    print(f"[reml] REML slope-only            = "
          f"{slope_only['rel_impr_pct']:+.3f}%")

    # Pre-registered branch disposition
    ratio_alpha = tau_alpha_reml / max(tau_alpha_mom, 1e-12)
    ratio_beta = tau_beta_reml / max(tau_beta_mom, 1e-12)
    if ratio_alpha >= 0.5 and ratio_alpha <= 2.0:
        branch = "HOLDS"
    elif ratio_alpha < 0.1:
        branch = "COLLAPSES"
    else:
        branch = "INTERMEDIATE"

    print(f"\n[reml] BRANCH DISPOSITION: {branch}")
    print(f"       ratio_alpha = {ratio_alpha:.4f}  "
          f"(HOLDS [0.5,2.0], COLLAPSES <0.1, INTERMEDIATE [0.1,0.5])")

    summary = {
        "iter": "N+286",
        "purpose": "REML verification of the HelpSteer2 MoM headline",
                "estimator_canonical": "statsmodels 0.14.6 MixedLM.fit(reml=True) [default]",
        "formula": "score ~ rm_score + (1 + rm_score | attribute_id)",
        "canonical_refs": {
            "statsmodels": "MixedLM.fit() reml=True default; 0.14.6 verified",
            "lme4": "Bates et al. 2015 JSS 67(1); lmer() REML=TRUE default",
            "morris1983": "JASA 78(381):47-55 EB shrinkage weight",
            "patterson_thompson_1971": "Biometrika 58:545 REML origin",
            "harville1977": "JASA 72:320 REML marginal likelihood",
        },
        "n_rows": int(len(df)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_attributes": len(ATTRS),
        "n_long_form": int(len(train_long)),
        "seed": SEED,
        "mom_baseline": {
            "tau_alpha_sq": tau_alpha_mom,
            "tau_beta_sq": tau_beta_mom,
            "gain_pct_mom_baseline": gain_pct_mom,
        },
        "reml_fit": reml,
        "comparison": {
            "tau_alpha_mom": tau_alpha_mom,
            "tau_alpha_reml": tau_alpha_reml,
            "ratio_alpha": ratio_alpha,
            "tau_beta_mom": tau_beta_mom,
            "tau_beta_reml": tau_beta_reml,
            "ratio_beta": ratio_beta,
        },
        "recompute_gain": {
            "mom_rerun": mom_eval,
            "reml": reml_eval,
            "reml_intercept_only": intercept_only,
            "reml_slope_only": slope_only,
        },
        "gain_pct_mom": mom_eval["rel_impr_pct"],
        "gain_pct_reml": reml_eval["rel_impr_pct"],
        "gain_pct_reml_intercept_only": intercept_only["rel_impr_pct"],
        "branch": branch,
        "branch_criteria": {
            "HOLDS": "ratio_alpha in [0.5, 2.0]",
            "COLLAPSES": "ratio_alpha < 0.1",
            "INTERMEDIATE": "ratio_alpha in [0.1, 0.5)",
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[ok] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
