"""PRISM anchor noise robustness.

Question: how robust is PEBS's +8.58% RMSE reduction to noise in the
anchor score (RM)? At what noise level does PEBS lose its edge over
pop-slope? This is an ANCHOR-QUALITY ablation — tests how dependent
PEBS is on having a well-trained RM as the shared reference.

Protocol: take trained Qwen2.5-7B RM scores. Add Gaussian noise with
SD = c * sigma(rm_score) where c in {0.0, 0.5, 1.0, 2.0, 5.0}.
Fit pop-slope vs PEBS on the noised anchor, evaluate held-out-user
RMSE.

Hypotheses:
- At c=0: canonical result (PEBS +8.58%).
- At c=5: RM is essentially pure noise; PEBS should lose or tie.
- Crossover c where PEBS gain disappears: informative for how
  anchor-quality-dependent PEBS is.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def load():
    s = pd.read_parquet(ROOT / "data" / "prism_rm_scored.parquet")
    s = s.dropna(subset=["rm_score", "score_user"]).reset_index(drop=True)
    return s[["user_id", "rm_score", "score_user"]]


def fit_and_score(df_train, df_test):
    pop_lr = LinearRegression().fit(
        df_train[["rm_z"]].values, df_train["score_user"].values
    )
    pop_a = float(pop_lr.coef_[0])
    pop_b = float(pop_lr.intercept_)

    try:
        md = smf.mixedlm(
            "score_user ~ rm_z",
            data=df_train,
            groups=df_train["user_id"],
            re_formula="~ rm_z",
        )
        res = md.fit(method="lbfgs", reml=True)
        alpha_pop = float(res.fe_params["rm_z"])
        beta_pop = float(res.fe_params["Intercept"])
        cov = np.asarray(res.cov_re)
        tau_a = float(cov[1, 1]) if cov.shape == (2, 2) else 0.0
        tau_b = float(cov[0, 0]) if cov.shape == (2, 2) else 0.0
    except Exception:
        alpha_pop, beta_pop = pop_a, pop_b
        tau_a = tau_b = 0.0

    user_calib = {}
    for uid, g in df_train.groupby("user_id"):
        if len(g) < 3:
            continue
        x = g["rm_z"].values.reshape(-1, 1)
        y = g["score_user"].values
        if np.var(x) < 1e-12:
            user_calib[uid] = (alpha_pop, float(y.mean()))
            continue
        lr2 = LinearRegression().fit(x, y)
        a_ols = float(lr2.coef_[0])
        b_ols = float(lr2.intercept_)
        yhat = lr2.predict(x)
        resid = y - yhat
        s2 = float(np.var(resid, ddof=2)) if len(resid) > 2 else float(np.var(resid))
        xx = float(np.sum((x - x.mean()) ** 2))
        v_a = s2 / (xx + 1e-9)
        v_b = s2 * (1.0 / len(g) + (x.mean() ** 2) / (xx + 1e-9))
        w_a = tau_a / (tau_a + v_a + 1e-9) if tau_a > 0 else 0.0
        w_b = tau_b / (tau_b + v_b + 1e-9) if tau_b > 0 else 0.0
        user_calib[uid] = (
            w_a * a_ols + (1 - w_a) * alpha_pop,
            w_b * b_ols + (1 - w_b) * beta_pop,
        )

    per_user_rmse = {"pop": [], "pebs": []}
    for uid, g in df_test.groupby("user_id"):
        x = g["rm_z"].values
        y = g["score_user"].values
        pop_pred = pop_a * x + pop_b
        if uid in user_calib:
            a, b = user_calib[uid]
            pebs_pred = a * x + b
        else:
            pebs_pred = pop_pred
        per_user_rmse["pop"].append(float(np.sqrt(np.mean((y - pop_pred) ** 2))))
        per_user_rmse["pebs"].append(float(np.sqrt(np.mean((y - pebs_pred) ** 2))))
    return {
        "rmse_pop_mean": float(np.mean(per_user_rmse["pop"])),
        "rmse_pebs_mean": float(np.mean(per_user_rmse["pebs"])),
    }


def main():
    df = load()
    rng_outer = np.random.default_rng(SEED)

    # 5-fold within-user split
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df["fold"] = -1
    for uid, g in df.groupby("user_id"):
        idxs = g.index.values.copy()
        rng_outer.shuffle(idxs)
        for i, idx in enumerate(idxs):
            df.loc[idx, "fold"] = i % 5

    noise_grid = [0.0, 0.5, 1.0, 2.0, 5.0]
    print(f"[data] N={len(df)}, {df['user_id'].nunique()} users, "
          f"rm_score SD={df['rm_score'].std():.3f}")

    results = {}
    for c in noise_grid:
        rng = np.random.default_rng(SEED + int(100 * c))
        # Add Gaussian noise of SD = c * rm_score_sd
        rm_sd = df["rm_score"].std()
        df_noise = df.copy()
        df_noise["rm_score_noised"] = df["rm_score"].values + rng.normal(0, c * rm_sd, len(df))
        # Standardize noised anchor
        df_noise["rm_z"] = (
            df_noise["rm_score_noised"] - df_noise["rm_score_noised"].mean()
        ) / df_noise["rm_score_noised"].std()

        fold_rmse = {"pop": [], "pebs": []}
        for k in range(5):
            tr = df_noise[df_noise["fold"] != k]
            te = df_noise[df_noise["fold"] == k]
            r = fit_and_score(tr, te)
            fold_rmse["pop"].append(r["rmse_pop_mean"])
            fold_rmse["pebs"].append(r["rmse_pebs_mean"])

        pop_m = float(np.mean(fold_rmse["pop"]))
        pebs_m = float(np.mean(fold_rmse["pebs"]))
        rel = 100 * (pop_m - pebs_m) / pop_m
        print(f"[noise c={c:.1f}]  pop={pop_m:.3f}  pebs={pebs_m:.3f}  "
              f"Δ(pop-pebs)={pop_m-pebs_m:+.3f} ({rel:+.2f}% rel)")
        results[f"c={c}"] = {
            "noise_c": c,
            "rmse_pop_mean": pop_m,
            "rmse_pebs_mean": pebs_m,
            "delta_pop_minus_pebs": pop_m - pebs_m,
            "rel_improvement_pct": rel,
        }

    out = ROOT / "results" / "prism_anchor_noise_robustness.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
