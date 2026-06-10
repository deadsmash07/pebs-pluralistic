"""PluriHarms practical baselines head-to-head.

Mirrors the PRISM practical-baselines battery on PluriHarms.
Tests whether naive practitioner-alternative calibrations (z-score,
min-max, quantile-match, residual-only, demographic-stratum) hurt
or help on PluriHarms. Expected: partial similarity to PRISM but
with larger effect-size at the best alternative due to PluriHarms'
higher between-user CV(beta).

Arms:
1. Pop-slope (baseline)
2. Per-user z-score normalization
3. Per-user min-max [0, 100]
4. Per-user quantile-match to pop CDF
5. Per-user residual-only (β_j intercept only; slope fixed at pop)
6. Demographic-stratum pop-slope (Gender)
7. PEBS EB-shrunk (our method)
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def load():
    ann = pd.read_csv(ROOT / "data" / "pluriharms" / "annotations.csv")
    prm = pd.read_csv(ROOT / "data" / "pluriharms" / "prompts.csv")
    rating_cols = [c for c in ann.columns if c.startswith("Rating_")]
    long = ann[["Participant_ID", "Gender"] + rating_cols].melt(
        id_vars=["Participant_ID", "Gender"],
        value_vars=rating_cols,
        var_name="_r",
        value_name="rating",
    )
    long["Question_Index"] = long["_r"].str.replace("Rating_", "").astype(int)
    long = long.merge(prm[["Question_Index", "Harm_Level"]], on="Question_Index")
    long = long.dropna(subset=["rating", "Harm_Level"]).reset_index(drop=True)
    long = long.rename(
        columns={"Participant_ID": "user_id", "Harm_Level": "anchor"}
    )
    return long


def score_per_user_rmse(df_train, df_test, arm):
    """Return per-user RMSE on test set for a given arm."""
    anchor_z_train = (df_train["anchor"] - df_train["anchor"].mean()) / df_train["anchor"].std()
    anchor_z_test = (df_test["anchor"] - df_train["anchor"].mean()) / df_train["anchor"].std()
    df_train = df_train.assign(anchor_z=anchor_z_train.values)
    df_test = df_test.assign(anchor_z=anchor_z_test.values)

    pop_lr = LinearRegression().fit(
        df_train[["anchor_z"]].values, df_train["rating"].values
    )
    pop_a = float(pop_lr.coef_[0])
    pop_b = float(pop_lr.intercept_)

    per_user_pred = {}

    if arm == "pop":
        for uid, g in df_test.groupby("user_id"):
            per_user_pred[uid] = pop_a * g["anchor_z"].values + pop_b

    elif arm == "zscore":
        # Per-user z-score of anchor
        for uid, g in df_test.groupby("user_id"):
            tr_u = df_train[df_train["user_id"] == uid]
            if len(tr_u) < 3:
                pred = pop_a * g["anchor_z"].values + pop_b
            else:
                u_mu = tr_u["anchor"].mean()
                u_sd = tr_u["anchor"].std() + 1e-9
                te_z = (g["anchor"].values - u_mu) / u_sd
                # Fit pop-slope on train where anchors are per-user z-scored
                tr_z = (df_train["anchor"].values - df_train.groupby("user_id")["anchor"].transform("mean").values) / (
                    df_train.groupby("user_id")["anchor"].transform("std").values + 1e-9
                )
                lr2 = LinearRegression().fit(tr_z.reshape(-1, 1), df_train["rating"].values)
                pred = float(lr2.coef_[0]) * te_z + float(lr2.intercept_)
            per_user_pred[uid] = pred

    elif arm == "minmax":
        for uid, g in df_test.groupby("user_id"):
            tr_u = df_train[df_train["user_id"] == uid]
            if len(tr_u) < 3:
                pred = pop_a * g["anchor_z"].values + pop_b
            else:
                lo, hi = tr_u["anchor"].min(), tr_u["anchor"].max()
                width = hi - lo + 1e-9
                te_mm = 100 * (g["anchor"].values - lo) / width
                tr_mm = np.zeros(len(df_train))
                for u2, g2 in df_train.groupby("user_id"):
                    g2_lo, g2_hi = g2["anchor"].min(), g2["anchor"].max()
                    g2_w = g2_hi - g2_lo + 1e-9
                    tr_mm[g2.index] = 100 * (g2["anchor"].values - g2_lo) / g2_w
                lr2 = LinearRegression().fit(tr_mm.reshape(-1, 1), df_train["rating"].values)
                pred = float(lr2.coef_[0]) * te_mm + float(lr2.intercept_)
            per_user_pred[uid] = pred

    elif arm == "quantile":
        for uid, g in df_test.groupby("user_id"):
            tr_u = df_train[df_train["user_id"] == uid]
            if len(tr_u) < 5:
                pred = pop_a * g["anchor_z"].values + pop_b
            else:
                # Per-user quantile of anchor within train; map to pop quantile
                u_sorted = np.sort(tr_u["anchor"].values)
                pop_sorted = np.sort(df_train["anchor"].values)
                te_q = np.searchsorted(u_sorted, g["anchor"].values) / max(len(u_sorted), 1)
                te_pop_anchor = np.quantile(pop_sorted, np.clip(te_q, 0, 1))
                tr_pop_z = (te_pop_anchor - df_train["anchor"].mean()) / df_train["anchor"].std()
                pred = pop_a * tr_pop_z + pop_b
            per_user_pred[uid] = pred

    elif arm == "residual":
        # Per-user β_j intercept correction only (slope fixed at pop)
        for uid, g in df_test.groupby("user_id"):
            tr_u = df_train[df_train["user_id"] == uid]
            if len(tr_u) < 3:
                pred = pop_a * g["anchor_z"].values + pop_b
            else:
                tr_u_pred = pop_a * tr_u["anchor_z"].values + pop_b
                beta_j = float((tr_u["rating"].values - tr_u_pred).mean())
                pred = pop_a * g["anchor_z"].values + pop_b + beta_j
            per_user_pred[uid] = pred

    elif arm == "demo_stratum":
        # Fit pop-slope within each demographic stratum (Gender)
        stratum_lr = {}
        for s, gs in df_train.groupby("Gender"):
            if len(gs) >= 10:
                lr_s = LinearRegression().fit(
                    gs[["anchor_z"]].values, gs["rating"].values
                )
                stratum_lr[s] = (float(lr_s.coef_[0]), float(lr_s.intercept_))
        for uid, g in df_test.groupby("user_id"):
            g_gender = g["Gender"].iloc[0]
            a, b = stratum_lr.get(g_gender, (pop_a, pop_b))
            pred = a * g["anchor_z"].values + b
            per_user_pred[uid] = pred

    elif arm == "pebs":
        md = smf.mixedlm(
            "rating ~ anchor_z",
            data=df_train,
            groups=df_train["user_id"],
            re_formula="~ anchor_z",
        )
        res = md.fit(method="lbfgs", reml=True)
        alpha_pop = float(res.fe_params["anchor_z"])
        beta_pop = float(res.fe_params["Intercept"])
        cov = np.asarray(res.cov_re)
        tau_a = float(cov[1, 1]) if cov.shape == (2, 2) else 0.0
        tau_b = float(cov[0, 0]) if cov.shape == (2, 2) else 0.0
        user_calib = {}
        for uid, g in df_train.groupby("user_id"):
            if len(g) < 3:
                continue
            x = g["anchor_z"].values.reshape(-1, 1)
            y = g["rating"].values
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
        for uid, g in df_test.groupby("user_id"):
            if uid in user_calib:
                a, b = user_calib[uid]
                pred = a * g["anchor_z"].values + b
            else:
                pred = alpha_pop * g["anchor_z"].values + beta_pop
            per_user_pred[uid] = pred

    # Compute per-user RMSE
    rows = []
    for uid, g in df_test.groupby("user_id"):
        pred = per_user_pred[uid]
        y = g["rating"].values
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        rows.append((uid, rmse))
    return rows


def main():
    df = load()
    print(f"[data] {len(df)} ratings, {df['user_id'].nunique()} users")

    # 5-fold within-user split
    rng = np.random.default_rng(SEED)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df["fold"] = -1
    for uid, g in df.groupby("user_id"):
        idxs = g.index.values.copy()
        rng.shuffle(idxs)
        for i, idx in enumerate(idxs):
            df.loc[idx, "fold"] = i % 5

    arms = ["pop", "zscore", "minmax", "quantile", "residual", "demo_stratum", "pebs"]
    per_user_rmse = {a: {} for a in arms}

    for k in range(5):
        tr = df[df["fold"] != k].copy()
        te = df[df["fold"] == k].copy()
        for arm in arms:
            try:
                rows = score_per_user_rmse(tr, te, arm)
                for uid, rmse in rows:
                    per_user_rmse[arm].setdefault(uid, []).append(rmse)
            except Exception as e:
                print(f"[{arm} fold {k}] ERROR: {e}")

    # Aggregate mean-across-folds per user, then mean across users
    summary = {}
    print("\n=== PluriHarms practical baselines battery ===")
    print(f"{'arm':<16} {'mean RMSE':<12} {'vs pop':<10}")
    pop_mean = float(np.mean([np.mean(v) for v in per_user_rmse["pop"].values() if v]))
    for arm in arms:
        user_means = np.array([np.mean(v) for v in per_user_rmse[arm].values() if v])
        mean_rmse = float(np.mean(user_means))
        rel = 100 * (pop_mean - mean_rmse) / pop_mean
        summary[arm] = {"mean_rmse": mean_rmse, "rel_vs_pop_pct": rel, "n_users": int(len(user_means))}
        print(f"{arm:<16} {mean_rmse:<12.3f} {rel:+.2f}%")

    # Wilcoxon PEBS vs each other arm
    pebs_user = np.array([np.mean(v) for v in per_user_rmse["pebs"].values() if v])
    pebs_uids = [u for u in per_user_rmse["pebs"] if per_user_rmse["pebs"][u]]
    for arm in arms:
        if arm == "pebs":
            continue
        a_user = np.array([
            np.mean(per_user_rmse[arm].get(u, [np.nan]))
            for u in pebs_uids
        ])
        mask = ~np.isnan(a_user)
        if mask.sum() < 2:
            continue
        w = stats.wilcoxon(pebs_user[mask], a_user[mask], alternative="less")
        summary[f"wilcoxon_pebs_vs_{arm}_p"] = float(w.pvalue)
        summary[f"frac_pebs_lt_{arm}"] = float((pebs_user[mask] < a_user[mask]).mean())
        print(f"  Wilcoxon PEBS < {arm}: p={w.pvalue:.3e}, frac_win={float((pebs_user[mask] < a_user[mask]).mean()):.3f}")

    out = ROOT / "results" / "pluriharms_practical_baselines.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
