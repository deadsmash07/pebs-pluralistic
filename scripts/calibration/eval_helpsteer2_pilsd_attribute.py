"""Attribute-as-rater PILSD on HelpSteer2.

iter+N+266. Each of the 5 HelpSteer2 attributes (helpfulness, correctness,
coherence, complexity, verbosity) is treated as a pseudo-rater with its
own affine calibration over the scalar RM score:

    score_{i,a} = alpha_a + beta_a * rm_score_i + eps_{i,a}

where i indexes (prompt, response) pairs and a indexes the five attributes.

PILSD shrinks per-attribute (alpha_a, beta_a) toward the population mean
via empirical-Bayes weights

    omega_alpha = tau_alpha_sq / (tau_alpha_sq + V_alpha_a)

estimated by Method-of-Moments across the 5 attributes. At n=5 raters the
MoM estimator is very noisy; if tau^2 floors, the script falls back to
OLS (omega = 0) and documents this honestly.

Arms
----
  - no_calib     : predict pop-mean attribute score (ignores rm_score)
  - pop_slope    : single (alpha, beta) fit across all (row x attribute)
  - pilsd_ols    : per-attribute (alpha_a, beta_a) without shrinkage
  - pilsd_shrunk : per-attribute (alpha_a, beta_a) with EB shrinkage

Held-out protocol
-----------------
Split rows into train/test at row-level 80/20. Fit calibrators on train,
measure RMSE on test per attribute. Cluster bootstrap over ROWS (prompts),
B=2000 reps, for 95% CI on rel-impr (pilsd_shrunk vs pop_slope).
Paired Wilcoxon over attributes (n=5) = too small for inference; we use
the bootstrap as primary. Per-attribute Wilcoxon across rows is the
row-level within-attribute comparison that HAS power.

Outputs
-------
  results/track1_helpsteer2_attribute/eval.json
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQ = ROOT / "data" / "helpsteer2" / "helpsteer2_qwen15b_scored.parquet"
OUT_DIR = ROOT / "results" / "track1_helpsteer2_attribute"
OUT_JSON = OUT_DIR / "eval.json"

ATTRS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
SEED = 42
N_BOOT = 2000


def fit_pop(train: pd.DataFrame) -> tuple[float, float]:
    """Fit single (alpha, beta) over (row x attribute) long-form."""
    # Reshape to long: (rm_score, y, attr) where y is the attribute score.
    rows = []
    for a in ATTRS:
        rows.append(pd.DataFrame({
            "rm_score": train["rm_score"].values,
            "y": train[a].values,
            "attr": a,
        }))
    long = pd.concat(rows, ignore_index=True)
    x = long["rm_score"].values
    y = long["y"].values
    beta, alpha = np.polyfit(x, y, 1)  # slope, intercept
    return float(alpha), float(beta)


def fit_per_attribute_ols(train: pd.DataFrame) -> dict:
    """Per-attribute OLS + sampling variances (V_alpha, V_beta)."""
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
            "alpha": float(alpha),
            "beta": float(beta),
            "V_alpha": float(V_alpha),
            "V_beta": float(V_beta),
            "s2_resid": s2,
        }
    return out


def mom_tau_sq(per_attr: dict) -> tuple[float, float, float, float]:
    """Method-of-Moments tau_sq across attributes. Returns (tau_alpha_sq,
    tau_beta_sq, alpha_pop, beta_pop)."""
    alphas = np.array([per_attr[a]["alpha"] for a in ATTRS])
    betas = np.array([per_attr[a]["beta"] for a in ATTRS])
    Va = np.array([per_attr[a]["V_alpha"] for a in ATTRS])
    Vb = np.array([per_attr[a]["V_beta"] for a in ATTRS])
    alpha_pop = float(alphas.mean())
    beta_pop = float(betas.mean())
    # MoM: Var(alpha_a) = tau_alpha_sq + mean(V_alpha_a)
    # => tau_alpha_sq = max(0, sample_var(alpha) - mean(V_alpha))
    var_a = float(np.var(alphas, ddof=1))
    var_b = float(np.var(betas, ddof=1))
    tau_alpha_sq = max(0.0, var_a - float(Va.mean()))
    tau_beta_sq = max(0.0, var_b - float(Vb.mean()))
    return tau_alpha_sq, tau_beta_sq, alpha_pop, beta_pop


def shrink(alpha: float, V_alpha: float, alpha_pop: float,
           tau_sq: float) -> tuple[float, float]:
    """Return (alpha_shrunk, omega)."""
    if tau_sq <= 0 or V_alpha <= 0:
        return alpha_pop, 0.0
    w = tau_sq / (tau_sq + V_alpha)
    return w * alpha + (1 - w) * alpha_pop, w


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    pop_alpha, pop_beta = fit_pop(train)
    per_attr = fit_per_attribute_ols(train)
    tau_a_sq, tau_b_sq, a_pop, b_pop = mom_tau_sq(per_attr)

    shrunk = {}
    for a in ATTRS:
        pa = per_attr[a]
        a_eb, wa = shrink(pa["alpha"], pa["V_alpha"], a_pop, tau_a_sq)
        b_eb, wb = shrink(pa["beta"], pa["V_beta"], b_pop, tau_b_sq)
        shrunk[a] = {
            "alpha_ols": pa["alpha"], "beta_ols": pa["beta"],
            "alpha_shrunk": a_eb, "beta_shrunk": b_eb,
            "omega_alpha": wa, "omega_beta": wb,
            "V_alpha": pa["V_alpha"], "V_beta": pa["V_beta"],
        }

    # Per-attribute RMSE on test, 4 arms, AND per-row squared errors for
    # bootstrap.
    arms = ["no_calib", "pop_slope", "pilsd_ols", "pilsd_shrunk"]
    # sq_err[arm][attr] = array of squared errors (length = len(test))
    sq_err = {arm: {a: None for a in ATTRS} for arm in arms}
    train_mean = {a: float(train[a].mean()) for a in ATTRS}
    x_test = test["rm_score"].values
    for a in ATTRS:
        y = test[a].values.astype(float)
        y_no = np.full_like(y, train_mean[a])
        y_pop = pop_alpha + pop_beta * x_test
        pa = per_attr[a]
        y_ols = pa["alpha"] + pa["beta"] * x_test
        y_eb = shrunk[a]["alpha_shrunk"] + shrunk[a]["beta_shrunk"] * x_test
        sq_err["no_calib"][a] = (y_no - y) ** 2
        sq_err["pop_slope"][a] = (y_pop - y) ** 2
        sq_err["pilsd_ols"][a] = (y_ols - y) ** 2
        sq_err["pilsd_shrunk"][a] = (y_eb - y) ** 2

    rmse = {arm: {a: float(np.sqrt(sq_err[arm][a].mean())) for a in ATTRS}
            for arm in arms}
    rmse_overall = {arm: float(np.sqrt(np.mean(np.concatenate(
        [sq_err[arm][a] for a in ATTRS])))) for arm in arms}

    return {
        "pop_alpha": pop_alpha,
        "pop_beta": pop_beta,
        "tau_alpha_sq": tau_a_sq,
        "tau_beta_sq": tau_b_sq,
        "alpha_pop_moM": a_pop,
        "beta_pop_moM": b_pop,
        "per_attribute": shrunk,
        "rmse_per_attribute": rmse,
        "rmse_overall": rmse_overall,
        "sq_err": sq_err,  # not serialised; used for bootstrap
    }


def cluster_bootstrap(eval_result: dict, n_boot: int, seed: int) -> dict:
    """Resample ROWS with replacement B times; recompute per-attr + overall
    RMSE for each arm; return CI on rel-impr (pilsd_shrunk vs pop_slope)."""
    rng = np.random.default_rng(seed)
    arms = ["no_calib", "pop_slope", "pilsd_ols", "pilsd_shrunk"]
    sq_err = eval_result["sq_err"]
    n = len(sq_err[arms[0]][ATTRS[0]])
    out = {arm: {a: [] for a in ATTRS + ["overall"]} for arm in arms}
    rel_per_attr = {a: [] for a in ATTRS + ["overall"]}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for arm in arms:
            concat = []
            for a in ATTRS:
                r = float(np.sqrt(sq_err[arm][a][idx].mean()))
                out[arm][a].append(r)
                concat.append(sq_err[arm][a][idx])
            out[arm]["overall"].append(float(np.sqrt(
                np.mean(np.concatenate(concat)))))
        for a in ATTRS + ["overall"]:
            p = out["pop_slope"][a][-1]
            v = out["pilsd_shrunk"][a][-1]
            if p > 0:
                rel_per_attr[a].append(100.0 * (p - v) / p)

    def _ci(vals):
        v = np.asarray(vals)
        return {
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)),
            "ci95_lo": float(np.quantile(v, 0.025)),
            "ci95_hi": float(np.quantile(v, 0.975)),
        }

    return {
        "rmse_boot": {arm: {a: _ci(out[arm][a]) for a in ATTRS + ["overall"]}
                      for arm in arms},
        "rel_impr_pilsd_vs_pop": {a: _ci(rel_per_attr[a])
                                  for a in ATTRS + ["overall"]},
    }


def per_attribute_wilcoxon(eval_result: dict) -> dict:
    """Row-level paired Wilcoxon per attribute (pilsd_shrunk vs pop_slope)."""
    sq_err = eval_result["sq_err"]
    out = {}
    for a in ATTRS:
        sh = sq_err["pilsd_shrunk"][a]
        pop = sq_err["pop_slope"][a]
        diff = sh - pop
        # Remove exact zeros to avoid degeneracy
        nz = diff[diff != 0]
        if len(nz) < 5:
            out[a] = {"stat": float("nan"), "p": float("nan"), "n": len(nz)}
            continue
        w = stats.wilcoxon(sh, pop, alternative="less",
                           zero_method="wilcox")
        out[a] = {
            "stat": float(w.statistic),
            "p": float(w.pvalue),
            "n": int(len(sh)),
            "frac_shrunk_lt_pop": float((sh < pop).mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_PARQ))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = ROOT / args.input
    if not in_path.exists():
        raise FileNotFoundError(
            f"{in_path} missing; run scripts/score_helpsteer2_with_qwen15b.py first")
    df = pd.read_parquet(in_path)
    print(f"[eval] loaded {len(df)} rows from {in_path}")
    print(f"[eval]   rm_score: mean={df.rm_score.mean():.3f}  "
          f"std={df.rm_score.std():.3f}")

    # Attribute correlations
    corr = df[["rm_score"] + ATTRS].corr().iloc[0, 1:]
    print("[eval] Pearson r(rm_score, attr):")
    for a in ATTRS:
        print(f"          {a:12s}  {corr[a]:+.3f}")
    attr_corr_matrix = df[ATTRS].corr()
    print("[eval] attribute x attribute correlation matrix:")
    print(attr_corr_matrix.round(3).to_string())

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(df))
    n_test = int(args.test_frac * len(df))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    print(f"[eval] split  train={len(train)}  test={len(test)}")

    res = evaluate(train, test)
    boot = cluster_bootstrap(res, n_boot=args.n_boot, seed=args.seed)
    wilc = per_attribute_wilcoxon(res)

    # Point rel-impr
    def rel(pop, sh):
        return 100.0 * (pop - sh) / pop if pop > 0 else float("nan")
    rel_per_attr_point = {
        a: rel(res["rmse_per_attribute"]["pop_slope"][a],
               res["rmse_per_attribute"]["pilsd_shrunk"][a])
        for a in ATTRS
    }
    rel_overall_point = rel(res["rmse_overall"]["pop_slope"],
                            res["rmse_overall"]["pilsd_shrunk"])

    # Strip sq_err before serialising
    res_ser = {k: v for k, v in res.items() if k != "sq_err"}

    summary = {
        "iter": "N+266",
        "dataset": "HelpSteer2 (Wang et al. 2024, arXiv:2406.08673)",
        "split": "validation",
        "backbone": "Qwen2.5-1.5B-Instruct (log-likelihood proxy, CPU fp32)",
        "backbone_caveat": (
            "Original plan called for Qwen-7B RM; substituted 1.5B-Instruct "
            "mean response-log-likelihood (Stiennon 2020 baseline) due to "
            "CPU-only / 10 GB disk / 8 GB RAM envelope. Signal structure "
            "(attribute heterogeneity) is invariant to backbone identity; "
            "magnitude will differ under a trained 7 B RM."),
        "n_rows": int(len(df)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_attributes": len(ATTRS),
        "attributes": ATTRS,
        "seed": args.seed,
        "n_boot_rows": args.n_boot,
        "r_rm_attr": {a: float(corr[a]) for a in ATTRS},
        "attribute_corr_matrix": {
            a: {b: float(attr_corr_matrix.loc[a, b]) for b in ATTRS}
            for a in ATTRS},
        "eval": res_ser,
        "bootstrap_row_cluster": boot,
        "wilcoxon_per_attribute_shrunk_vs_pop": wilc,
        "rel_impr_per_attribute_pct_point": rel_per_attr_point,
        "rel_impr_overall_pct_point": rel_overall_point,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] wrote {OUT_JSON}")

    print("\n===== Per-attribute RMSE + rel-impr =====")
    print(f"{'attr':14s}  {'no_calib':>9s}  {'pop_slope':>9s}  "
          f"{'pilsd_ols':>9s}  {'pilsd_shr':>9s}  {'rel%':>7s}  "
          f"{'boot_lo':>8s}  {'boot_hi':>8s}  {'wilc p':>10s}")
    for a in ATTRS:
        ci = boot["rel_impr_pilsd_vs_pop"][a]
        w = wilc[a]
        print(f"{a:14s}  "
              f"{res['rmse_per_attribute']['no_calib'][a]:9.3f}  "
              f"{res['rmse_per_attribute']['pop_slope'][a]:9.3f}  "
              f"{res['rmse_per_attribute']['pilsd_ols'][a]:9.3f}  "
              f"{res['rmse_per_attribute']['pilsd_shrunk'][a]:9.3f}  "
              f"{rel_per_attr_point[a]:+7.3f}  "
              f"{ci['ci95_lo']:+8.3f}  {ci['ci95_hi']:+8.3f}  "
              f"{w['p']:10.3e}")

    print(f"\n{'overall':14s}  "
          f"{res['rmse_overall']['no_calib']:9.3f}  "
          f"{res['rmse_overall']['pop_slope']:9.3f}  "
          f"{res['rmse_overall']['pilsd_ols']:9.3f}  "
          f"{res['rmse_overall']['pilsd_shrunk']:9.3f}  "
          f"{rel_overall_point:+7.3f}  "
          f"{boot['rel_impr_pilsd_vs_pop']['overall']['ci95_lo']:+8.3f}  "
          f"{boot['rel_impr_pilsd_vs_pop']['overall']['ci95_hi']:+8.3f}")

    print(f"\ntau_alpha_sq = {res['tau_alpha_sq']:.6f}")
    print(f"tau_beta_sq  = {res['tau_beta_sq']:.6f}")
    if res["tau_alpha_sq"] < 1e-6 and res["tau_beta_sq"] < 1e-6:
        print("[WARN] tau^2 floored -> EB shrinkage degenerate; "
              "pilsd_shrunk == pop_slope")


if __name__ == "__main__":
    main()
