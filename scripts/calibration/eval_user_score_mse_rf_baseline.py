"""Random Forest non-parametric baseline — is per-user EB shrinkage beaten by
a flexible non-parametric learner?

Hypothesis
----------
If PEBS (linear + 2-param per user, or quadratic + 3-param per user) still
beats a non-parametric learner (no functional-form assumption), then the
RANDOM-EFFECT structure matters MORE than model flexibility.

Design: 6-arm within-user k=5 CV on PRISM (same folds as
eval_user_score_mse_shrunk.py and eval_user_score_mse_quadratic.py, so seed=42
is locked):

    1. no_calib             predict train-fold mean of y
    2. pop_slope            global α₀ + β₀·x  (paper headline baseline)
    3. pebs_shrunk         per-user EB-shrunk linear
    4. pebs_quadratic      per-user EB-shrunk quadratic (N+179 headline)
    5. rf_per_user          RF(200, depth 10) on [rm_score, user_id_onehot]
    6. rf_global            RF(200, depth 10) on [rm_score, demographic_onehot]
                            — tests whether user_id is truly irreducible vs
                              predictable from demographics

For arms 5-6 the RF is a SINGLE GLOBAL MODEL per fold (not per-user) trained
on the union of within-user train-fold rows, then predicted on the union of
within-user test-fold rows, then per-user RMSE computed over that user's
test rows only. This matches the paper's per-user CV semantics for arms 1-4.

Per-user RMSE is the unit of analysis for Wilcoxon and bootstrap CIs, giving
1,394 paired observations.

Reports:
  - mean/median RMSE per arm
  - pairwise Wilcoxon (2-sided) vs pebs_shrunk
  - cluster-bootstrap 95% CI on mean(RMSE_RF) - mean(RMSE_PEBS_shrunk)
  - relative improvement vs pop_slope (paper-style)

Refs: Breiman 2001 RF; Gelman & Hill 2007 §12 partial pooling.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.ensemble import RandomForestRegressor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scored-parquet", default="data/prism_rm_scored.parquet")
    p.add_argument("--demographics-parquet", default="data/prism_demographics.parquet")
    p.add_argument("--min-obs-per-user", type=int, default=6)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42,
                   help="Match eval_user_score_mse_{shrunk,quadratic}.py to "
                        "align fold indices across arms.")
    p.add_argument("--rf-n-estimators", type=int, default=200)
    p.add_argument("--rf-max-depth", type=int, default=10)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--output-path",
                   default="results/track1_rf_baseline/summary.json")
    return p.parse_args()


def kfold_split(n: int, k: int, rng: np.random.Generator):
    """SAME deterministic k-fold routine as eval_user_score_mse_shrunk.py."""
    idx = np.arange(n)
    rng.shuffle(idx)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        stop = (i + 1) * fold_size if i < k - 1 else n
        test_idx = idx[start:stop]
        train_idx = np.concatenate([idx[:start], idx[stop:]])
        folds.append((train_idx, test_idx))
    return folds


def ols_linear_with_V(x, y):
    k = len(x)
    if k < 2 or np.var(x) < 1e-12:
        return float(np.mean(y)) if k else 0.0, 0.0, np.inf, np.inf
    x_bar = x.mean()
    Sxx = ((x - x_bar) ** 2).sum()
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = intercept + slope * x
    sigma_hat_sq = ((y - y_pred) ** 2).sum() / max(k - 2, 1)
    V_int = sigma_hat_sq * (1.0 / k + x_bar ** 2 / max(Sxx, 1e-12))
    V_slope = sigma_hat_sq / max(Sxx, 1e-12)
    return float(intercept), float(slope), float(V_int), float(V_slope)


def ols_quadratic_with_V(x, y):
    k = len(x)
    if k < 3 or np.var(x) < 1e-12:
        if k >= 2 and np.var(x) >= 1e-12:
            a, b, Va, Vb = ols_linear_with_V(x, y)
            return a, b, 0.0, Va, Vb, np.inf
        return float(np.mean(y)) if k else 0.0, 0.0, 0.0, np.inf, np.inf, np.inf
    X = np.column_stack([np.ones(k), x, x ** 2])
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        a, b, Va, Vb = ols_linear_with_V(x, y)
        return a, b, 0.0, Va, Vb, np.inf
    beta_hat = XtX_inv @ X.T @ y
    resid = y - X @ beta_hat
    sigma_hat_sq = (resid @ resid) / max(k - 3, 1)
    V = sigma_hat_sq * np.diag(XtX_inv)
    return (float(beta_hat[0]), float(beta_hat[1]), float(beta_hat[2]),
            float(V[0]), float(V[1]), float(V[2]))


def build_demographic_features(df: pd.DataFrame,
                               demographics: pd.DataFrame) -> tuple[sparse.csr_matrix, list[str]]:
    """Return sparse one-hot demographic features aligned to df.user_id order.

    Missing users (in df but not in demographics) get an all-zero row +
    a "missing" indicator per demographic column.
    """
    demo_cols = ["age", "gender", "education", "employment_status",
                 "english_proficiency", "marital_status",
                 "lm_familiarity", "lm_frequency_use", "study_locale"]
    # Keep only columns that exist
    demo_cols = [c for c in demo_cols if c in demographics.columns]

    # Left-join demographics onto df by user_id
    demo_small = demographics[["user_id"] + demo_cols].drop_duplicates("user_id")
    merged = df[["user_id"]].merge(demo_small, on="user_id", how="left")

    pieces = []
    names = []
    for col in demo_cols:
        # Fill NaN with explicit category
        s = merged[col].astype("object").fillna("__MISSING__")
        dummies = pd.get_dummies(s, prefix=col, drop_first=False, sparse=True)
        pieces.append(sparse.csr_matrix(dummies.sparse.to_coo()))
        names.extend(dummies.columns.tolist())
    X = sparse.hstack(pieces).tocsr() if pieces else sparse.csr_matrix((len(df), 0))
    return X, names


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.scored_parquet).dropna(subset=["score_user"]).reset_index(drop=True)
    print(f"[load] {len(df)} utterances, {df.user_id.nunique()} users")

    # Apply min-obs floor first (matches other scripts)
    counts = df.groupby("user_id").size()
    keep_uids = counts[counts >= args.min_obs_per_user].index
    df = df[df.user_id.isin(keep_uids)].reset_index(drop=True)
    print(f"[filter] min_obs={args.min_obs_per_user}: {len(df)} rows, "
          f"{df.user_id.nunique()} users")

    # Load demographics (may be missing)
    demographics = None
    demo_path = Path(args.demographics_parquet)
    if demo_path.exists():
        demographics = pd.read_parquet(demo_path)
        print(f"[demo] loaded {len(demographics)} rows, "
              f"overlap with kept users = {demographics.user_id.isin(keep_uids).sum()}")
    else:
        print(f"[demo] WARNING: {demo_path} not found — rf_global arm will be skipped")

    # ------------------ Global pop baseline ------------------
    slope_pop, intercept_pop = np.polyfit(df.rm_score, df.score_user, 1)
    pop_alpha = float(intercept_pop)
    pop_beta = float(slope_pop)
    print(f"[pop] α₀={pop_alpha:.3f}  β₀={pop_beta:.3f}")

    q, m, c = np.polyfit(df.rm_score, df.score_user, 2)
    pop_int_qd, pop_lin_qd, pop_quad_qd = float(c), float(m), float(q)

    # ------------------ EB τ² pre-pass ------------------
    user_stats_lin = []
    user_stats_qd = []
    for uid, grp in df.groupby("user_id"):
        if len(grp) < args.min_obs_per_user:
            continue
        x = grp.rm_score.to_numpy()
        y = grp.score_user.to_numpy().astype(float)
        a, b, Va, Vb = ols_linear_with_V(x, y)
        user_stats_lin.append({"user_id": uid, "a": a, "b": b, "Va": Va, "Vb": Vb})
        aq, bq, cq, Vaq, Vbq, Vcq = ols_quadratic_with_V(x, y)
        user_stats_qd.append({"user_id": uid, "a": aq, "b": bq, "c": cq,
                              "Va": Vaq, "Vb": Vbq, "Vc": Vcq})
    us_lin = pd.DataFrame(user_stats_lin)
    us_qd = pd.DataFrame(user_stats_qd)

    def _tau(col_var, col_V):
        return max(col_var - col_V, 1e-6)

    tau_a_lin = _tau(float(us_lin.a.var()),
                     float(us_lin.Va.replace([np.inf, -np.inf], np.nan).dropna().mean()))
    tau_b_lin = _tau(float(us_lin.b.var()),
                     float(us_lin.Vb.replace([np.inf, -np.inf], np.nan).dropna().mean()))
    tau_a_qd = _tau(float(us_qd.a.var()),
                    float(us_qd.Va.replace([np.inf, -np.inf], np.nan).dropna().mean()))
    tau_b_qd = _tau(float(us_qd.b.var()),
                    float(us_qd.Vb.replace([np.inf, -np.inf], np.nan).dropna().mean()))
    tau_c_qd = _tau(float(us_qd.c.var()),
                    float(us_qd.Vc.replace([np.inf, -np.inf], np.nan).dropna().mean()))
    print(f"[EB-lin] τ_α²={tau_a_lin:.3f}  τ_β²={tau_b_lin:.3f}")
    print(f"[EB-qd]  τ_α²={tau_a_qd:.3f}  τ_β²={tau_b_qd:.3f}  τ_γ²={tau_c_qd:.3f}")

    # ------------------ Assign global row indices for RF ------------------
    # For RF arms we need per-row train/test membership per fold across all users.
    # We stash per-row train_fold_id (which of k folds this row belongs to
    # as TEST) so we can slice sparse X quickly.
    df = df.reset_index(drop=True)
    df["_row"] = np.arange(len(df))
    per_user_folds = {}  # uid -> list of (train_local, test_local) LOCAL indices within user
    per_row_fold_of_test = np.full(len(df), -1, dtype=np.int32)

    for uid, grp in df.groupby("user_id"):
        n = len(grp)
        folds_local = kfold_split(n, args.k_folds, rng)
        global_rows = grp._row.to_numpy()
        for f_idx, (tr_loc, te_loc) in enumerate(folds_local):
            per_row_fold_of_test[global_rows[te_loc]] = f_idx
        per_user_folds[uid] = (folds_local, global_rows)

    assert (per_row_fold_of_test >= 0).all(), "some row did not get a test fold"
    print(f"[folds] rows-per-test-fold distribution: "
          f"{np.bincount(per_row_fold_of_test).tolist()}")

    # ------------------ Build RF feature matrices ------------------
    # Arm 5: [rm_score, user_id_onehot]
    uid_cat = df.user_id.astype("category")
    uid_code = uid_cat.cat.codes.to_numpy()
    n_users_total = uid_cat.cat.categories.size

    rm = df.rm_score.to_numpy().astype(np.float32)
    y = df.score_user.to_numpy().astype(np.float64)

    rm_col = sparse.csr_matrix(rm.reshape(-1, 1))
    rows_arr = np.arange(len(df))
    uid_onehot = sparse.csr_matrix(
        (np.ones(len(df), dtype=np.float32), (rows_arr, uid_code)),
        shape=(len(df), n_users_total),
    )
    X_per_user = sparse.hstack([rm_col, uid_onehot]).tocsr()
    print(f"[X] per_user shape={X_per_user.shape}")

    # Arm 6: [rm_score, demographic_onehot]
    if demographics is not None:
        X_demo, demo_names = build_demographic_features(df, demographics)
        X_global = sparse.hstack([rm_col, X_demo]).tocsr()
        print(f"[X] global (demo) shape={X_global.shape}  ({len(demo_names)} demo dims)")
    else:
        X_global, demo_names = None, []

    # ------------------ RF per-fold fits ------------------
    # For each of the k folds, train on UNION of train rows, predict on UNION of
    # test rows. Collect per-row predictions.
    rf_per_user_pred = np.full(len(df), np.nan, dtype=np.float64)
    rf_global_pred = np.full(len(df), np.nan, dtype=np.float64)

    rf_kwargs = dict(
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        n_jobs=-1,
        random_state=args.seed,
    )

    for f_idx in range(args.k_folds):
        te_mask = per_row_fold_of_test == f_idx
        tr_mask = ~te_mask
        n_tr, n_te = int(tr_mask.sum()), int(te_mask.sum())
        print(f"\n[fold {f_idx+1}/{args.k_folds}]  train={n_tr}  test={n_te}")

        # rf_per_user
        t0 = time.time()
        rf = RandomForestRegressor(**rf_kwargs)
        rf.fit(X_per_user[tr_mask], y[tr_mask])
        pred = rf.predict(X_per_user[te_mask])
        rf_per_user_pred[te_mask] = pred
        rmse_fold = float(np.sqrt(((pred - y[te_mask]) ** 2).mean()))
        print(f"  rf_per_user  fit+pred={time.time()-t0:.1f}s  fold_rmse={rmse_fold:.3f}")

        # rf_global
        if X_global is not None:
            t0 = time.time()
            rf = RandomForestRegressor(**rf_kwargs)
            rf.fit(X_global[tr_mask], y[tr_mask])
            pred = rf.predict(X_global[te_mask])
            rf_global_pred[te_mask] = pred
            rmse_fold = float(np.sqrt(((pred - y[te_mask]) ** 2).mean()))
            print(f"  rf_global    fit+pred={time.time()-t0:.1f}s  fold_rmse={rmse_fold:.3f}")

    # ------------------ Per-user k-fold CV for arms 1-4 ------------------
    # Re-derive the fold splits with a SEPARATE RNG to match the shrunk/quadratic
    # scripts exactly. They use a single rng drawn once, then advanced per
    # user. We mirror that pattern here for per-user arms so the exact same
    # fold indices get used that the paper headline used.
    rng_pebs = np.random.default_rng(args.seed)

    arms = ["no_calib", "pop_slope", "pebs_shrunk", "pebs_quadratic",
            "rf_per_user", "rf_global"]
    per_user_rows = []
    skipped_no_demo = (X_global is None)

    for uid, grp in df.groupby("user_id"):
        n = len(grp)
        if n < args.min_obs_per_user:
            continue
        x = grp.rm_score.to_numpy()
        yu = grp.score_user.to_numpy().astype(float)
        # Regenerate identical folds with fresh rng
        folds = kfold_split(n, args.k_folds, rng_pebs)
        squared = {a: [] for a in arms}
        global_rows = grp._row.to_numpy()

        for f_idx, (train_idx, test_idx) in enumerate(folds):
            x_tr, y_tr = x[train_idx], yu[train_idx]
            x_te, y_te = x[test_idx], yu[test_idx]
            if len(x_te) == 0:
                continue
            g_te = global_rows[test_idx]

            # 1. no_calib
            squared["no_calib"].extend(((float(np.mean(y_tr)) - y_te) ** 2).tolist())
            # 2. pop_slope
            yhat = pop_alpha + pop_beta * x_te
            squared["pop_slope"].extend(((yhat - y_te) ** 2).tolist())
            # 3. pebs_shrunk (linear)
            a, b, Va, Vb = ols_linear_with_V(x_tr, y_tr)
            wa = tau_a_lin / (tau_a_lin + Va) if np.isfinite(Va) else 0.0
            wb = tau_b_lin / (tau_b_lin + Vb) if np.isfinite(Vb) else 0.0
            a_s = wa * a + (1 - wa) * pop_alpha
            b_s = wb * b + (1 - wb) * pop_beta
            yhat = a_s + b_s * x_te
            squared["pebs_shrunk"].extend(((yhat - y_te) ** 2).tolist())
            # 4. pebs_quadratic
            aq, bq, cq, Vaq, Vbq, Vcq = ols_quadratic_with_V(x_tr, y_tr)
            waq = tau_a_qd / (tau_a_qd + Vaq) if np.isfinite(Vaq) else 0.0
            wbq = tau_b_qd / (tau_b_qd + Vbq) if np.isfinite(Vbq) else 0.0
            wcq = tau_c_qd / (tau_c_qd + Vcq) if np.isfinite(Vcq) else 0.0
            a_sq = waq * aq + (1 - waq) * pop_int_qd
            b_sq = wbq * bq + (1 - wbq) * pop_lin_qd
            c_sq = wcq * cq + (1 - wcq) * pop_quad_qd
            yhat = a_sq + b_sq * x_te + c_sq * x_te ** 2
            squared["pebs_quadratic"].extend(((yhat - y_te) ** 2).tolist())
            # 5. rf_per_user (look up per-row predictions)
            yhat = rf_per_user_pred[g_te]
            squared["rf_per_user"].extend(((yhat - y_te) ** 2).tolist())
            # 6. rf_global (if available)
            if not skipped_no_demo:
                yhat = rf_global_pred[g_te]
                squared["rf_global"].extend(((yhat - y_te) ** 2).tolist())

        row = {"user_id": uid, "n": n}
        for a in arms:
            if a == "rf_global" and skipped_no_demo:
                row[f"rmse_{a}"] = float("nan")
                continue
            row[f"rmse_{a}"] = float(np.sqrt(np.mean(squared[a])))
        per_user_rows.append(row)

    pu = pd.DataFrame(per_user_rows)

    # Sanity: RF per-user folds ≡ per-row predictions assembled from the global
    # fold partition. Any row without a prediction (NaN) means our fold
    # alignment drifted. Guard:
    if np.isnan(rf_per_user_pred).any():
        miss = int(np.isnan(rf_per_user_pred).sum())
        print(f"[warn] {miss} rows with no RF per-user prediction (unexpected)")

    # ------------------ Aggregate ------------------
    print(f"\n=== 6-arm within-user CV (n_users={len(pu)}, k={args.k_folds}) ===")
    agg_mean = {}
    agg_median = {}
    for a in arms:
        col = f"rmse_{a}"
        if col not in pu.columns:
            continue
        vals = pu[col].dropna()
        agg_mean[a] = float(vals.mean()) if len(vals) else float("nan")
        agg_median[a] = float(vals.median()) if len(vals) else float("nan")
        print(f"  {col:<22} mean={agg_mean[a]:.4f}  median={agg_median[a]:.4f}")

    # ------------------ Wilcoxon pairs (each arm vs pebs_shrunk) ------------------
    def paired(a_col, b_col):
        # Drop rows where either is NaN (rf_global may be nan if demo missing)
        aa, bb = pu[a_col].to_numpy(), pu[b_col].to_numpy()
        mask = np.isfinite(aa) & np.isfinite(bb)
        a, b = aa[mask], bb[mask]
        if len(a) < 10 or (a == b).all():
            return {"mean_delta_a_minus_b": float("nan"),
                    "frac_a_smaller": float("nan"),
                    "wilcoxon_p": float("nan"),
                    "n": int(len(a))}
        w = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        return {
            "mean_delta_a_minus_b": float((a - b).mean()),
            "median_delta": float(np.median(a - b)),
            "frac_a_smaller": float((a < b).mean()),
            "wilcoxon_stat": float(w.statistic),
            "wilcoxon_p": float(w.pvalue),
            "n": int(len(a)),
        }

    comparisons = {}
    for a in ["no_calib", "pop_slope", "pebs_quadratic", "rf_per_user", "rf_global"]:
        if f"rmse_{a}" not in pu.columns:
            continue
        comparisons[f"{a}_vs_pebs_shrunk"] = paired(f"rmse_{a}", "rmse_pebs_shrunk")
    print(f"\n=== Paired Wilcoxon (each arm vs pebs_shrunk; Δ<0 ⇒ arm better) ===")
    for name, d in comparisons.items():
        if not np.isfinite(d.get("mean_delta_a_minus_b", float("nan"))):
            print(f"  {name:<38} (no data)")
            continue
        sign = "↓" if d["mean_delta_a_minus_b"] < 0 else "↑"
        print(f"  {name:<38} Δ={d['mean_delta_a_minus_b']:+.4f} {sign}  "
              f"frac_first_smaller={d['frac_a_smaller']:.1%}  "
              f"p={d['wilcoxon_p']:.3e}  n={d['n']}")

    # ------------------ Cluster-bootstrap CIs ------------------
    rng_ci = np.random.default_rng(args.seed + 1)
    idx_all = np.arange(len(pu))

    def boot_ci(col_a, col_b):
        a_vals = pu[col_a].to_numpy()
        b_vals = pu[col_b].to_numpy()
        mask = np.isfinite(a_vals) & np.isfinite(b_vals)
        a_vals, b_vals = a_vals[mask], b_vals[mask]
        if len(a_vals) < 10:
            return {"ci95_lo": float("nan"), "ci95_hi": float("nan"),
                    "point_estimate": float("nan"), "includes_zero": None,
                    "n_effective": int(len(a_vals))}
        n_eff = len(a_vals)
        deltas = []
        for _ in range(args.n_boot):
            sub = rng_ci.choice(n_eff, size=n_eff, replace=True)
            deltas.append(float(a_vals[sub].mean() - b_vals[sub].mean()))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        pt = float(a_vals.mean() - b_vals.mean())
        return {
            "point_estimate": pt,
            "bootstrap_mean": float(np.mean(deltas)),
            "ci95_lo": float(lo), "ci95_hi": float(hi),
            "includes_zero": bool(lo <= 0 <= hi),
            "n_effective": int(n_eff),
            "n_boot": int(args.n_boot),
        }

    bootstraps = {}
    for a in ["rf_per_user", "rf_global", "pebs_quadratic"]:
        col = f"rmse_{a}"
        if col in pu.columns:
            bootstraps[f"{a}_minus_pebs_shrunk"] = boot_ci(col, "rmse_pebs_shrunk")
    print(f"\n=== Cluster-bootstrap 95% CI on mean(arm) - mean(pebs_shrunk) ===")
    for name, d in bootstraps.items():
        if not np.isfinite(d.get("point_estimate", float("nan"))):
            print(f"  {name:<38} (no data)")
            continue
        iz = "YES" if d["includes_zero"] else "NO"
        print(f"  {name:<38} pt={d['point_estimate']:+.4f}  "
              f"95%CI=[{d['ci95_lo']:+.4f},{d['ci95_hi']:+.4f}]  includes_zero={iz}")

    # ------------------ Relative improvements (paper-frame) ------------------
    rel = {}
    base = agg_mean.get("pop_slope", float("nan"))
    print(f"\n=== Relative RMSE improvement vs pop_slope ===")
    for a in arms:
        if a not in agg_mean or not np.isfinite(agg_mean[a]):
            continue
        rel[a] = 100 * (base - agg_mean[a]) / base
        print(f"  {a:<22} {rel[a]:+.3f}%")

    # ------------------ Verdict ------------------
    verdict_parts = []
    rf_pu = comparisons.get("rf_per_user_vs_pebs_shrunk", {})
    if rf_pu and np.isfinite(rf_pu.get("mean_delta_a_minus_b", float("nan"))):
        if rf_pu["mean_delta_a_minus_b"] > 0 and rf_pu["wilcoxon_p"] < 0.05:
            verdict_parts.append("PEBS_SHRUNK_BEATS_RF_PER_USER")
        elif rf_pu["mean_delta_a_minus_b"] < 0 and rf_pu["wilcoxon_p"] < 0.05:
            verdict_parts.append("RF_PER_USER_BEATS_PEBS_SHRUNK")
        else:
            verdict_parts.append("RF_PER_USER_VS_PEBS_NULL")
    rf_g = comparisons.get("rf_global_vs_pebs_shrunk", {})
    if rf_g and np.isfinite(rf_g.get("mean_delta_a_minus_b", float("nan"))):
        if rf_g["mean_delta_a_minus_b"] > 0 and rf_g["wilcoxon_p"] < 0.05:
            verdict_parts.append("PEBS_SHRUNK_BEATS_RF_GLOBAL")
        elif rf_g["mean_delta_a_minus_b"] < 0 and rf_g["wilcoxon_p"] < 0.05:
            verdict_parts.append("RF_GLOBAL_BEATS_PEBS_SHRUNK")
        else:
            verdict_parts.append("RF_GLOBAL_VS_PEBS_NULL")
    verdict = " | ".join(verdict_parts) if verdict_parts else "INCONCLUSIVE"
    print(f"\n=== VERDICT: {verdict} ===")

    # ------------------ Save ------------------
    out = {
        "n_users": int(len(pu)),
        "k_folds": int(args.k_folds),
        "seed": int(args.seed),
        "rf_config": {
            "n_estimators": int(args.rf_n_estimators),
            "max_depth": int(args.rf_max_depth),
            "n_jobs": -1,
        },
        "n_rows": int(len(df)),
        "demographics_available": demographics is not None,
        "n_demographic_dims": int(len(demo_names)),
        "eb_linear": {"tau_alpha_sq": tau_a_lin, "tau_beta_sq": tau_b_lin},
        "eb_quadratic": {"tau_intercept_sq": tau_a_qd,
                         "tau_linear_sq": tau_b_qd,
                         "tau_quadratic_sq": tau_c_qd},
        "pop_linear": {"intercept": pop_alpha, "slope": pop_beta},
        "aggregate_rmse_mean": agg_mean,
        "aggregate_rmse_median": agg_median,
        "relative_improvement_vs_pop_slope_pct": rel,
        "comparisons_vs_pebs_shrunk": comparisons,
        "bootstrap_delta_vs_pebs_shrunk": bootstraps,
        "verdict": verdict,
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    pu.to_parquet(out_path.with_suffix(".parquet"))
    print(f"\n[save] {out_path}  +  {out_path.with_suffix('.parquet')}")


if __name__ == "__main__":
    main()
