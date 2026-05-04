"""Cross-user calibrator transfer — does PILSD's population calibration generalize
to users NOT seen during MixedLM REML training?

Research question
-----------------
The H2e within-user CV result (mean RMSE 23.33, 8.58% better than pop-slope
via EB shrinkage) is a SAMPLE-LEVEL claim: per-user calibrators fit on each
user's own data improve that user's held-out utterances. But real deployment
has NEW users with ZERO prior signal. Does the POPULATION calibration
(α_pop, β_pop) — the BLUP limit at ω→0 — transfer to users outside the
MixedLM training pool? And does even a handful of anchor utterances (k=1..10)
from a new user, shrunk toward the population via τ² estimated on OTHER users,
help more than pop-slope alone?

Design
------
1. 80/20 USER split (seed=42 default) on all users with ≥ min-obs-per-user
   utterances → train-users get REML fit, holdout-users are ENTIRELY unseen.
2. Fit statsmodels MixedLM (REML) on train-user rows ONLY
   → (α_pop, β_pop, τ_α², τ_β², σ_resid).
3. For each holdout user, evaluate 3+|ks| prediction modes on all their
   utterances (using chronological order so "first k" simulates first k
   submitted scores at deployment):
     - no_calib:        ŷ = train_mean_y (a flat constant baseline)
     - pop_slope:       ŷ = α_pop + β_pop · rm_score  (zero-shot MixedLM)
     - pilsd_zero_shot: SAME as pop_slope  (shrinkage with 0 obs → ω=0 → pop)
     - pilsd_few_shot(k): fit OLS on the user's first k rows, apply EB
                         shrinkage ω = τ²/(τ² + V(·)) using pop τ² from
                         REML train fit, score the REMAINING n−k rows.
4. Report mean / median RMSE per mode across holdout users, Wilcoxon paired
   p-values, and the % reduction vs no-calib / vs pop-slope.

Important caveats
-----------------
- The existing 8.58% H2e improvement is WITHIN-USER (own-data CV). This
  cross-user transfer measures a DIFFERENT quantity: how well pop
  calibration + few-shot shrinkage fit NEW users. Expected smaller magnitude.
- Expected: pop_slope beats no_calib by ~3-8% (BLUP transfers to held-out
  users; MixedLM REML claim).
- Expected: few_shot k=5 adds ~1-4 pp over pop_slope (shrinkage regime
  dominated by pop prior at small k; signal from user's own anchors
  contributes modestly).

References
----------
- Henderson 1975 "Best Linear Unbiased Estimation and Prediction" (BLUP
  for new subjects = population mean if unseen).
- Morris 1983 "Parametric empirical Bayes inference" (τ² transfer).
- Pinheiro & Bates 2000 §1.4 (Laird-Ware random-effects model).
- Prior: memory/track1_shrinkage_H2e_upgrade.md (within-user result).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scored-parquet", default="data/prism_rm_scored.parquet")
    p.add_argument("--output-path",
                   default="results/track1_cross_user_transfer.json")
    p.add_argument("--ks", default="1,3,5,10",
                   help="Comma-separated k values for few-shot.")
    p.add_argument("--min-obs-per-user", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    return p.parse_args()


def ols_with_V(x: np.ndarray, y: np.ndarray):
    """OLS intercept/slope plus sampling variances.

    V(α̂) = σ̂² (1/k + x̄²/Sxx), V(β̂) = σ̂²/Sxx.
    Degenerate (k<2 or zero-x-variance) returns ∞ variances, which forces
    ω → 0 (full pop-slope) downstream.
    """
    k = len(x)
    if k < 2 or np.var(x) < 1e-12:
        return (float(np.mean(y)) if k else 0.0, 0.0, np.inf, np.inf)
    x_bar = x.mean()
    Sxx = ((x - x_bar) ** 2).sum()
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = intercept + slope * x
    if k >= 3:
        sigma_hat_sq = ((y - y_pred) ** 2).sum() / (k - 2)
    else:
        sigma_hat_sq = float(np.var(y)) if np.var(y) > 0 else 1.0
    V_alpha = sigma_hat_sq * (1.0 / k + x_bar ** 2 / max(Sxx, 1e-12))
    V_beta = sigma_hat_sq / max(Sxx, 1e-12)
    return float(intercept), float(slope), float(V_alpha), float(V_beta)


def shrink(alpha_hat, beta_hat, V_alpha, V_beta,
           alpha_pop, beta_pop, tau_alpha_sq, tau_beta_sq):
    """Parametric-EB shrinkage to population mean.

    ω_α = τ_α² / (τ_α² + V(α̂)). Infinite V (no evidence) → ω=0 → pop-slope.
    """
    omega_alpha = (tau_alpha_sq / (tau_alpha_sq + V_alpha)
                   if np.isfinite(V_alpha) else 0.0)
    omega_beta = (tau_beta_sq / (tau_beta_sq + V_beta)
                  if np.isfinite(V_beta) else 0.0)
    alpha_shrunk = omega_alpha * alpha_hat + (1 - omega_alpha) * alpha_pop
    beta_shrunk = omega_beta * beta_hat + (1 - omega_beta) * beta_pop
    return (float(alpha_shrunk), float(beta_shrunk),
            float(omega_alpha), float(omega_beta))


def fit_reml_on_train(train_df: pd.DataFrame) -> dict:
    """MixedLM REML fit on train-user utterances only.

    Returns (α_pop, β_pop, τ_α², τ_β², σ_resid, corr, converged, wall_s).
    """
    t0 = time.time()
    md = smf.mixedlm("score_user ~ rm_score",
                     data=train_df, groups=train_df["user_id"],
                     re_formula="~rm_score")
    res = md.fit(method="lbfgs", maxiter=300, disp=False, reml=True)
    cov_re = res.cov_re
    tau_a = max(float(cov_re.iloc[0, 0]), 1e-6)
    tau_b = max(float(cov_re.iloc[1, 1]), 1e-6)
    cov_ab = float(cov_re.iloc[0, 1])
    return {
        "alpha_pop": float(res.params["Intercept"]),
        "beta_pop": float(res.params["rm_score"]),
        "tau_alpha_sq": tau_a,
        "tau_beta_sq": tau_b,
        "corr_alpha_beta": cov_ab / np.sqrt(max(tau_a * tau_b, 1e-18)),
        "sigma_resid_sq": float(res.scale),
        "sigma_resid": float(np.sqrt(res.scale)),
        "converged": bool(res.converged),
        "wall_s": float(time.time() - t0),
        "n_train_rows": int(len(train_df)),
        "n_train_users": int(train_df.user_id.nunique()),
    }


def split_users(all_users: list, holdout_frac: float, seed: int):
    """80/20 USER split — reproducible under fixed seed. Returns (train, holdout) sets."""
    rng = np.random.default_rng(seed)
    shuffled = list(all_users)
    rng.shuffle(shuffled)
    n_hold = int(len(shuffled) * holdout_frac)
    return set(shuffled[n_hold:]), set(shuffled[:n_hold])


def _rmse(y_hat: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_hat - y) ** 2)))


def evaluate_holdout_user(
    x: np.ndarray, y: np.ndarray,
    ks: list,
    alpha_pop: float, beta_pop: float,
    tau_a_sq: float, tau_b_sq: float,
    train_mean_y: float,
) -> dict:
    """Compute RMSE under every mode for one held-out user.

    - no_calib/pop_slope/pilsd_zero_shot are evaluated on ALL n utterances.
    - pilsd_few_shot(k) fits OLS on x[:k] / y[:k] and evaluates on x[k:] / y[k:]
      (chronological first-k as the user's "submitted anchor scores").
    """
    n = len(x)
    rec = {"n": int(n)}

    # no_calib: flat constant baseline
    rec["rmse_no_calib"] = _rmse(np.full_like(y, train_mean_y), y)

    # pop_slope: α_pop + β_pop · x
    y_hat_pop = alpha_pop + beta_pop * x
    rec["rmse_pop_slope"] = _rmse(y_hat_pop, y)

    # pilsd_zero_shot: identical to pop_slope by ω=0 construction
    rec["rmse_pilsd_zero_shot"] = rec["rmse_pop_slope"]

    # pilsd_few_shot(k): fit on first k, score last (n-k)
    for k in ks:
        col_rmse = f"rmse_pilsd_few_shot_k{k}"
        col_oa = f"omega_alpha_k{k}"
        col_ob = f"omega_beta_k{k}"
        if k + 2 > n:
            # Need ≥2 eval points AND k anchors; skip otherwise
            rec[col_rmse] = np.nan
            rec[col_oa] = np.nan
            rec[col_ob] = np.nan
            continue
        x_tr, y_tr = x[:k], y[:k]
        x_te, y_te = x[k:], y[k:]
        a, b, Va, Vb = ols_with_V(x_tr, y_tr)
        a_s, b_s, w_a, w_b = shrink(a, b, Va, Vb,
                                     alpha_pop, beta_pop,
                                     tau_a_sq, tau_b_sq)
        y_hat = a_s + b_s * x_te
        rec[col_rmse] = _rmse(y_hat, y_te)
        rec[col_oa] = w_a
        rec[col_ob] = w_b
    return rec


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon signed-rank on (a − b), plus mean / win-frac."""
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3 or np.allclose(a, b):
        return {"mean_delta": float(np.mean(a - b)) if len(a) else float("nan"),
                "frac_a_smaller": float(np.mean(a < b)) if len(a) else float("nan"),
                "wilcoxon_p": float("nan"),
                "n": int(len(a))}
    w = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return {
        "mean_delta": float((a - b).mean()),
        "frac_a_smaller": float((a < b).mean()),
        "wilcoxon_p": float(w.pvalue),
        "n": int(len(a)),
    }


def main():
    args = parse_args()
    ks = [int(k) for k in args.ks.split(",") if k.strip()]

    df = (pd.read_parquet(args.scored_parquet)
            .dropna(subset=["score_user"])
            .reset_index(drop=True))
    user_n = df.groupby("user_id").size()
    kept = user_n[user_n >= args.min_obs_per_user].index.tolist()
    df = df[df.user_id.isin(kept)].reset_index(drop=True)
    print(f"[load] {len(df)} rows, {len(kept)} users "
          f"(min_obs={args.min_obs_per_user})")

    # 80/20 user split
    train_users, holdout_users = split_users(kept, args.holdout_frac, args.seed)
    train_df = df[df.user_id.isin(train_users)].reset_index(drop=True)
    holdout_df = df[df.user_id.isin(holdout_users)].reset_index(drop=True)
    print(f"[split] train_users={len(train_users)}  holdout_users={len(holdout_users)}")
    assert not (train_users & holdout_users), "user split must be disjoint"

    # REML fit on train-user rows only
    print("\n=== REML MixedLM on train-users ===")
    reml = fit_reml_on_train(train_df)
    for k, v in reml.items():
        print(f"  {k}: {v}")

    alpha_pop = reml["alpha_pop"]
    beta_pop = reml["beta_pop"]
    tau_a_sq = reml["tau_alpha_sq"]
    tau_b_sq = reml["tau_beta_sq"]

    # Baseline constant (train-only mean_y, to avoid any peek at holdout)
    train_mean_y = float(train_df.score_user.mean())
    print(f"\n[baseline] train_mean_y = {train_mean_y:.4f}")

    # Evaluate every holdout user
    per_user = []
    for uid, grp in holdout_df.groupby("user_id"):
        grp_sorted = grp.sort_values(["turn", "within_turn_id"]).reset_index(drop=True) \
            if {"turn", "within_turn_id"}.issubset(grp.columns) else grp.reset_index(drop=True)
        x = grp_sorted.rm_score.to_numpy(dtype=float)
        y = grp_sorted.score_user.to_numpy(dtype=float)
        rec = evaluate_holdout_user(
            x, y, ks, alpha_pop, beta_pop, tau_a_sq, tau_b_sq, train_mean_y,
        )
        rec["user_id"] = uid
        per_user.append(rec)

    pu = pd.DataFrame(per_user)
    print(f"\n=== Holdout RMSE summary ({len(pu)} users) ===")
    mode_cols = ["rmse_no_calib", "rmse_pop_slope", "rmse_pilsd_zero_shot"] + [
        f"rmse_pilsd_few_shot_k{k}" for k in ks
    ]
    for col in mode_cols:
        vals = pu[col].dropna()
        if len(vals) == 0:
            continue
        print(f"  {col:>32}: mean={vals.mean():8.3f}  "
              f"median={vals.median():8.3f}  n={len(vals)}")

    # Wilcoxon paired tests
    comparisons = {}
    a = pu["rmse_pop_slope"].to_numpy()
    b = pu["rmse_no_calib"].to_numpy()
    comparisons["pop_slope_vs_no_calib"] = paired_stats(a, b)
    for k in ks:
        col_k = f"rmse_pilsd_few_shot_k{k}"
        comparisons[f"few_shot_k{k}_vs_pop_slope"] = paired_stats(
            pu[col_k].to_numpy(), pu["rmse_pop_slope"].to_numpy(),
        )
        comparisons[f"few_shot_k{k}_vs_no_calib"] = paired_stats(
            pu[col_k].to_numpy(), pu["rmse_no_calib"].to_numpy(),
        )

    print(f"\n=== Paired comparisons (Wilcoxon signed-rank) ===")
    for name, d in comparisons.items():
        if np.isnan(d.get("wilcoxon_p", np.nan)):
            continue
        print(f"  {name:>38}: mean Δ={d['mean_delta']:+.4f}  "
              f"A wins {d['frac_a_smaller']:.1%}  "
              f"p={d['wilcoxon_p']:.3e}  n={d['n']}")

    # Relative improvements
    nc_mean = float(pu.rmse_no_calib.mean())
    ps_mean = float(pu.rmse_pop_slope.mean())
    ps_improve_pct = 100.0 * (nc_mean - ps_mean) / nc_mean if nc_mean > 0 else float("nan")
    fs_improve_pct = {}
    for k in ks:
        col_k = f"rmse_pilsd_few_shot_k{k}"
        mv = float(pu[col_k].dropna().mean()) if pu[col_k].notna().any() else float("nan")
        base_mv = float(pu.loc[pu[col_k].notna(), "rmse_pop_slope"].mean()) \
            if pu[col_k].notna().any() else float("nan")
        if np.isnan(mv) or base_mv == 0 or np.isnan(base_mv):
            fs_improve_pct[k] = float("nan")
        else:
            fs_improve_pct[k] = 100.0 * (base_mv - mv) / base_mv

    print(f"\n=== Relative improvements ===")
    print(f"  pop_slope vs no_calib: {ps_improve_pct:+.2f}%")
    for k in ks:
        pct = fs_improve_pct[k]
        print(f"  few_shot k={k} vs pop_slope: "
              f"{'nan' if np.isnan(pct) else f'{pct:+.2f}%'}")

    summary = {
        "config": {
            "ks": ks,
            "min_obs_per_user": args.min_obs_per_user,
            "seed": args.seed,
            "holdout_frac": args.holdout_frac,
            "scored_parquet": args.scored_parquet,
        },
        "reml": reml,
        "train_mean_y": train_mean_y,
        "n_holdout_users": int(len(pu)),
        "rmse_mean": {col: float(pu[col].dropna().mean())
                      for col in mode_cols
                      if pu[col].notna().any()},
        "rmse_median": {col: float(pu[col].dropna().median())
                        for col in mode_cols
                        if pu[col].notna().any()},
        "comparisons": comparisons,
        "relative_improvement_pct": {
            "pop_slope_vs_no_calib": float(ps_improve_pct),
            **{f"few_shot_k{k}_vs_pop_slope": float(fs_improve_pct[k])
               for k in ks},
        },
    }
    out_json = Path(args.output_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    parquet_out = out_json.with_suffix(".parquet")
    pu.to_parquet(parquet_out)
    print(f"\n[save] {out_json}")
    print(f"[save] {parquet_out}")

    # Verdict
    print("\n=== Verdict ===")
    best_k = None
    best_mean = float("inf")
    for k in ks:
        col_k = f"rmse_pilsd_few_shot_k{k}"
        m = summary["rmse_mean"].get(col_k, float("nan"))
        if not np.isnan(m) and m < best_mean:
            best_mean = m
            best_k = k
    if best_k is not None:
        print(f"  Best few-shot k = {best_k} (mean RMSE {best_mean:.3f})")
    if ps_improve_pct > 0:
        print(f"  Pop-slope TRANSFERS to held-out users: "
              f"{ps_improve_pct:.2f}% lower RMSE vs no-calib")
    else:
        print(f"  WARNING: pop-slope did NOT beat no-calib on holdout "
              f"({ps_improve_pct:+.2f}%)")


if __name__ == "__main__":
    main()
