"""PluriHarms cross-dataset PEBS eval with bootstrap RMSE CIs.

This is the 3rd-real-dataset replication of Track 1's PRISM headline
(+8.58% RMSE improvement over pop-slope). The goal is to close the
"PRISM-only" generalization concern by showing the PEBS EB-shrunk calibrator
transfers to an independent corpus with per-annotator harm ratings.

Dataset: PluriHarms (Li et al. 2026, arXiv:2601.08951).

Protocol
--------
- 4 arms (same as PRISM eval_user_score_mse_shrunk):
    1. no_calib        : predict train-mean rating
    2. pop_slope       : shared (alpha_pop, beta_pop) from OLS
    3. pebs_ols       : per-user (alpha_j, beta_j) via OLS (no shrinkage)
    4. pebs_shrunk    : per-user EB shrunk toward (alpha_pop, beta_pop)
- Within-user k=5 CV: for each user, ~20% of their prompt-ratings are
  held out per fold; the other 80% train the model for that fold.
- Per-user RMSE aggregated across 5 folds per arm.
- Paired Wilcoxon (pebs_shrunk vs each baseline, alt=less).
- Bootstrap RMSE CI: resample USERS with replacement B=N_BOOT times per arm,
  recompute mean RMSE across bootstrap sample -> 2.5% / 97.5% quantile.
  This is a user-cluster bootstrap (honours within-user dependence) and
  gives honest between-user uncertainty on the mean RMSE.

Outputs
-------
results/track1_pluriharms_pebs/eval.json
    full summary with bootstrap CIs + Wilcoxon + per-arm per-user RMSE.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
LONG_PARQ = ROOT / "data" / "pluriharms_long.parquet"
OUT_DIR = ROOT / "results" / "track1_pluriharms_pebs"
OUT_JSON = OUT_DIR / "eval.json"

SEED = 42
K_FOLDS = 5
N_BOOT = 200  # >> 30 seeds as task spec requires


def fit_pop_slope(train: pd.DataFrame) -> tuple[float, float]:
    lr = LinearRegression().fit(
        train[["Harm_Level"]].values, train["rating"].values
    )
    return float(lr.coef_[0]), float(lr.intercept_)


def fit_per_user_ols(train: pd.DataFrame) -> dict:
    out: dict[int, tuple[float, float, float, float]] = {}
    for uid, g in train.groupby("user_id"):
        x = g["Harm_Level"].values
        y = g["rating"].values
        n = len(g)
        if n < 3:
            out[uid] = (np.nan, np.nan, np.nan, np.nan)
            continue
        lr = LinearRegression().fit(x.reshape(-1, 1), y)
        a, b = float(lr.coef_[0]), float(lr.intercept_)
        y_hat = lr.predict(x.reshape(-1, 1))
        resid = y - y_hat
        s2 = float(np.sum(resid ** 2) / max(n - 2, 1))
        xx_c = float(np.sum((x - x.mean()) ** 2)) + 1e-12
        V_alpha = s2 / xx_c
        V_beta = s2 * (1.0 / n + (x.mean() ** 2) / xx_c)
        out[uid] = (a, b, V_alpha, V_beta)
    return out


def fit_mixedlm(train: pd.DataFrame) -> dict:
    md = smf.mixedlm(
        "rating ~ Harm_Level",
        data=train,
        groups=train["user_id"],
        re_formula="~ Harm_Level",
    )
    res = md.fit(method="lbfgs", reml=True)
    cov_re = np.asarray(res.cov_re)
    return {
        "alpha_pop": float(res.fe_params["Harm_Level"]),
        "beta_pop": float(res.fe_params["Intercept"]),
        "tau_alpha_sq": float(cov_re[1, 1]) if cov_re.shape == (2, 2) else 0.0,
        "tau_beta_sq": float(cov_re[0, 0]) if cov_re.shape == (2, 2) else 0.0,
    }


def _w(tau_sq: float, v: float) -> float:
    if tau_sq <= 0 or v <= 0 or np.isnan(v):
        return 0.0
    return float(tau_sq / (tau_sq + v))


def predict_arms(
    x: float,
    uid: int,
    train_mean: float,
    pop_ab: tuple[float, float],
    user_ols: dict,
    mlm: dict,
) -> dict:
    pop_a, pop_b = pop_ab
    a, b, Va, Vb = user_ols.get(uid, (np.nan, np.nan, np.nan, np.nan))
    y_no = train_mean
    y_pop = pop_a * x + pop_b
    if np.isnan(a):
        y_ols = y_pop
        y_pebs = y_pop
    else:
        y_ols = a * x + b
        wa = _w(mlm["tau_alpha_sq"], Va)
        wb = _w(mlm["tau_beta_sq"], Vb)
        a_eb = wa * a + (1 - wa) * mlm["alpha_pop"]
        b_eb = wb * b + (1 - wb) * mlm["beta_pop"]
        y_pebs = a_eb * x + b_eb
    return {"no_calib": y_no, "pop_slope": y_pop, "pebs_ols": y_ols, "pebs_shrunk": y_pebs}


def within_user_kfold_rmse(
    df: pd.DataFrame, k: int = K_FOLDS, seed: int = SEED
) -> dict:
    """Per-user held-out RMSE across 4 arms using k-fold within-user split."""
    users = df["user_id"].unique()
    arms = ["no_calib", "pop_slope", "pebs_ols", "pebs_shrunk"]
    sq: dict[str, dict] = {a: {u: [] for u in users} for a in arms}

    for fold in range(k):
        train_rows, test_rows = [], []
        for uid, g in df.groupby("user_id"):
            gg = g.sample(frac=1, random_state=seed + fold).reset_index(drop=True)
            m = len(gg)
            n_test = max(1, m // k)
            test_rows.append(gg.iloc[:n_test])
            train_rows.append(gg.iloc[n_test:])
        train = pd.concat(train_rows, ignore_index=True)
        test = pd.concat(test_rows, ignore_index=True)

        pop_ab = fit_pop_slope(train)
        ols = fit_per_user_ols(train)
        mlm = fit_mixedlm(train)
        train_mean = float(train["rating"].mean())

        for _, r in test.iterrows():
            x = float(r["Harm_Level"])
            y = float(r["rating"])
            uid = int(r["user_id"])
            preds = predict_arms(x, uid, train_mean, pop_ab, ols, mlm)
            for a in arms:
                sq[a][uid].append((preds[a] - y) ** 2)

    rmse_per_user: dict[str, pd.Series] = {}
    for a in arms:
        out = {u: float(np.sqrt(np.mean(v))) for u, v in sq[a].items() if len(v) > 0}
        rmse_per_user[a] = pd.Series(out).sort_index()
    return rmse_per_user


def user_cluster_bootstrap(
    rmse: dict[str, pd.Series], n_boot: int, seed: int
) -> dict:
    """Resample users with replacement; recompute mean RMSE per arm each draw."""
    rng = np.random.default_rng(seed)
    arms = list(rmse.keys())
    users = np.array(rmse[arms[0]].index)
    n = len(users)
    boot: dict[str, list] = {a: [] for a in arms}
    boot_rel_impr: list = []  # pebs_shrunk vs pop_slope
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sampled = users[idx]
        means = {a: float(rmse[a].loc[sampled].mean()) for a in arms}
        for a in arms:
            boot[a].append(means[a])
        # % reduction pebs vs pop
        p = means["pop_slope"]
        v = means["pebs_shrunk"]
        if p > 0:
            boot_rel_impr.append(100.0 * (p - v) / p)

    def _ci(vals):
        v = np.asarray(vals)
        return {
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)),
            "ci95_lo": float(np.quantile(v, 0.025)),
            "ci95_hi": float(np.quantile(v, 0.975)),
        }

    return {
        **{f"rmse_{a}_boot": _ci(boot[a]) for a in arms},
        "rel_impr_pebs_vs_pop_pct_boot": _ci(boot_rel_impr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--k-folds", type=int, default=K_FOLDS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if not LONG_PARQ.exists():
        raise FileNotFoundError(
            f"{LONG_PARQ} missing; run scripts/load_pluriharms.py first"
        )
    df = pd.read_parquet(LONG_PARQ)
    print(
        f"[eval] loaded {len(df)} ratings / "
        f"{df['user_id'].nunique()} users / "
        f"{df['Question_Index'].nunique()} prompts"
    )

    rmse = within_user_kfold_rmse(df, k=args.k_folds, seed=args.seed)
    arms = ["no_calib", "pop_slope", "pebs_ols", "pebs_shrunk"]

    point = {
        a: {
            "mean": float(rmse[a].mean()),
            "median": float(rmse[a].median()),
            "sd": float(rmse[a].std(ddof=1)),
        }
        for a in arms
    }
    pop_mean = point["pop_slope"]["mean"]
    shrunk_mean = point["pebs_shrunk"]["mean"]
    rel_impr = 100.0 * (pop_mean - shrunk_mean) / pop_mean

    # Wilcoxon paired (user-level) — shrunk vs each baseline, alt=less
    wilc = {}
    for base in ["no_calib", "pop_slope", "pebs_ols"]:
        w = stats.wilcoxon(
            rmse["pebs_shrunk"].values, rmse[base].values, alternative="less"
        )
        wilc[f"pebs_shrunk_vs_{base}"] = {
            "stat": float(w.statistic),
            "p": float(w.pvalue),
            "frac_shrunk_lt_base": float(
                (rmse["pebs_shrunk"] < rmse[base]).mean()
            ),
        }

    boot = user_cluster_bootstrap(rmse, n_boot=args.n_boot, seed=args.seed)

    summary = {
        "dataset": "PluriHarms (Li et al. 2026, arXiv:2601.08951)",
        "release": "github.com/jl3676/PluriHarms-release",
        "n_users": int(df["user_id"].nunique()),
        "n_prompts": int(df["Question_Index"].nunique()),
        "n_ratings": int(len(df)),
        "anchor_col": "Harm_Level (0-1 classifier prob, per-prompt)",
        "rating_dim": "aggregate 0-100 harm rating",
        "k_folds": args.k_folds,
        "seed": args.seed,
        "n_boot_users": args.n_boot,
        "rmse": point,
        "rel_impr_pebs_vs_pop_pct_point": float(rel_impr),
        "wilcoxon": wilc,
        "bootstrap_user_cluster": boot,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] wrote {OUT_JSON}")

    # Nice stdout summary
    print("\n===== 4-arm RMSE table =====")
    print(f"{'arm':20s}  {'mean':>8s}  {'median':>8s}  {'boot 95% CI':>24s}")
    for a in arms:
        ci = boot[f"rmse_{a}_boot"]
        print(
            f"{a:20s}  {point[a]['mean']:8.3f}  {point[a]['median']:8.3f}  "
            f"[{ci['ci95_lo']:7.3f}, {ci['ci95_hi']:7.3f}]"
        )
    print(
        f"\nrel_impr pebs_shrunk vs pop_slope (mean): {rel_impr:+.3f}% "
        f"boot 95% CI [{boot['rel_impr_pebs_vs_pop_pct_boot']['ci95_lo']:+.3f}, "
        f"{boot['rel_impr_pebs_vs_pop_pct_boot']['ci95_hi']:+.3f}]"
    )
    for k, v in wilc.items():
        print(f"  wilcoxon {k}: p={v['p']:.3e}  frac<base={v['frac_shrunk_lt_base']:.3f}")


if __name__ == "__main__":
    main()
