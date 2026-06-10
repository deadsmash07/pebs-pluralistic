"""H2e + EB shrinkage arm — does partial pooling improve the main PEBS headline?

Last iteration's cold-start work showed EB shrinkage drops break-even k from
20 to 5. That's at the COLD-START regime (2-20 train utterances). The main
H2e result uses WITHIN-USER k=5 CV with ~40 train utterances per user — a
data-rich regime where ω should be close to 1 and shrinkage should barely
change anything.

BUT: even a small RMSE improvement matters for reviewer defense. "Does
shrinkage help at all scales?" is a cleaner narrative than "shrinkage
helps at cold start only".

Design: same as eval_user_score_mse.py but with a 4th arm:
  - no_calib: predict train-fold mean
  - pop_slope: global α₀ + β₀ · x
  - pebs_ols: per-user OLS on train fold (naive)
  - pebs_shrunk: shrinkage ω · OLS + (1-ω) · pop  (new)

Expected outcome: pebs_shrunk ≤ pebs_ols (lower or equal RMSE) at ≥95% of
users because shrinkage is always Pareto-dominant at finite k.

References
----------
- Gelman & Hill 2007 §12 partial pooling (same as cold-start script)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scored-parquet", default="data/prism_rm_scored.parquet")
    p.add_argument("--min-obs-per-user", type=int, default=6)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-path", default="results/track1_user_score_mse_shrunk.json")
    return p.parse_args()


def kfold_split(n: int, k: int, rng: np.random.Generator):
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


def ols_with_V(x: np.ndarray, y: np.ndarray):
    """Returns (intercept, slope, V_intercept, V_slope)."""
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


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.scored_parquet).dropna(subset=["score_user"]).reset_index(drop=True)
    print(f"[load] {len(df)} utterances, {df.user_id.nunique()} users")

    # Global calibration
    slope_pop, intercept_pop = np.polyfit(df.rm_score, df.score_user, 1)
    pop_alpha = float(intercept_pop)
    pop_beta = float(slope_pop)
    print(f"[pop] α₀={pop_alpha:.3f}  β₀={pop_beta:.3f}")

    # Pre-pass: per-user OLS on ALL data (used to estimate τ_α², τ_β² for shrinkage)
    user_stats = []
    for uid, grp in df.groupby("user_id"):
        if len(grp) < args.min_obs_per_user:
            continue
        a, b, Va, Vb = ols_with_V(
            grp.rm_score.to_numpy(),
            grp.score_user.to_numpy().astype(float),
        )
        user_stats.append({"user_id": uid, "alpha": a, "beta": b,
                           "V_alpha": Va, "V_beta": Vb, "n": len(grp)})
    us = pd.DataFrame(user_stats)
    V_alpha_total = float(us.alpha.var())
    V_beta_total = float(us.beta.var())
    mean_samp_V_alpha = float(us.V_alpha.replace([np.inf, -np.inf], np.nan).dropna().mean())
    mean_samp_V_beta = float(us.V_beta.replace([np.inf, -np.inf], np.nan).dropna().mean())
    tau_a_sq = max(V_alpha_total - mean_samp_V_alpha, 1e-6)
    tau_b_sq = max(V_beta_total - mean_samp_V_beta, 1e-6)
    print(f"[EB] τ_α²={tau_a_sq:.3f} (σ_α={np.sqrt(tau_a_sq):.3f})")
    print(f"[EB] τ_β²={tau_b_sq:.3f} (σ_β={np.sqrt(tau_b_sq):.3f})")

    # k-fold CV per user, 4 arms
    per_user_rows = []
    for uid, grp in df.groupby("user_id"):
        n = len(grp)
        if n < args.min_obs_per_user:
            continue
        x = grp.rm_score.to_numpy()
        y = grp.score_user.to_numpy().astype(float)
        folds = kfold_split(n, args.k_folds, rng)
        squared = {"no_calib": [], "pop_slope": [], "pebs_ols": [], "pebs_shrunk": []}

        for train_idx, test_idx in folds:
            x_tr, y_tr = x[train_idx], y[train_idx]
            x_te, y_te = x[test_idx], y[test_idx]
            if len(x_te) == 0:
                continue

            # No-calib
            y_hat_nc = np.full_like(y_te, np.mean(y_tr))
            squared["no_calib"].extend(((y_hat_nc - y_te) ** 2).tolist())

            # Pop-slope
            y_hat_ps = pop_alpha + pop_beta * x_te
            squared["pop_slope"].extend(((y_hat_ps - y_te) ** 2).tolist())

            # PEBS naive OLS
            a, b, Va, Vb = ols_with_V(x_tr, y_tr)
            y_hat_ols = a + b * x_te
            squared["pebs_ols"].extend(((y_hat_ols - y_te) ** 2).tolist())

            # PEBS shrunk
            omega_a = tau_a_sq / (tau_a_sq + Va) if np.isfinite(Va) else 0.0
            omega_b = tau_b_sq / (tau_b_sq + Vb) if np.isfinite(Vb) else 0.0
            a_s = omega_a * a + (1 - omega_a) * pop_alpha
            b_s = omega_b * b + (1 - omega_b) * pop_beta
            y_hat_sh = a_s + b_s * x_te
            squared["pebs_shrunk"].extend(((y_hat_sh - y_te) ** 2).tolist())

        per_user_rows.append({
            "user_id": uid,
            "n": n,
            **{f"rmse_{arm}": float(np.sqrt(np.mean(squared[arm])))
               for arm in squared},
        })

    pu = pd.DataFrame(per_user_rows)
    print(f"\n=== 4-arm within-user CV ({len(pu)} users, k={args.k_folds}) ===")
    for arm in ["rmse_no_calib", "rmse_pop_slope", "rmse_pebs_ols", "rmse_pebs_shrunk"]:
        print(f"  {arm}: mean={pu[arm].mean():.3f}  median={pu[arm].median():.3f}")

    # Paired comparisons
    def paired(a_col: str, b_col: str):
        a, b = pu[a_col].to_numpy(), pu[b_col].to_numpy()
        w = stats.wilcoxon(a, b, alternative="two-sided")
        return {
            "mean_delta": float((a - b).mean()),
            "frac_a_smaller": float((a < b).mean()),
            "wilcoxon_p": float(w.pvalue),
        }

    comparisons = {
        "shrunk_vs_ols":        paired("rmse_pebs_shrunk", "rmse_pebs_ols"),
        "shrunk_vs_pop":        paired("rmse_pebs_shrunk", "rmse_pop_slope"),
        "ols_vs_pop":           paired("rmse_pebs_ols", "rmse_pop_slope"),
    }
    print(f"\n=== Paired comparisons ===")
    for name, d in comparisons.items():
        print(f"  {name:>22}: mean Δ={d['mean_delta']:+.4f}  "
              f"{name.split('_vs_')[0]} wins {d['frac_a_smaller']:.1%}  "
              f"Wilcoxon p={d['wilcoxon_p']:.3e}")

    rel_improvement = 100 * (pu.rmse_pop_slope.mean() - pu.rmse_pebs_shrunk.mean()) / pu.rmse_pop_slope.mean()
    rel_ols = 100 * (pu.rmse_pop_slope.mean() - pu.rmse_pebs_ols.mean()) / pu.rmse_pop_slope.mean()
    print(f"\n=== Relative improvement vs pop-slope ===")
    print(f"  PEBS naive OLS: {rel_ols:+.2f}%")
    print(f"  PEBS shrunk:    {rel_improvement:+.2f}%")
    print(f"  Shrunk gain over naive: {rel_improvement - rel_ols:+.3f} pp")

    out = {
        "n_users": int(len(pu)),
        "k_folds": args.k_folds,
        "eb": {"tau_alpha_sq": tau_a_sq, "tau_beta_sq": tau_b_sq,
               "sigma_alpha": float(np.sqrt(tau_a_sq)),
               "sigma_beta": float(np.sqrt(tau_b_sq))},
        "rmse_mean": {arm: float(pu[f"rmse_{arm}"].mean())
                      for arm in ["no_calib", "pop_slope", "pebs_ols", "pebs_shrunk"]},
        "rmse_median": {arm: float(pu[f"rmse_{arm}"].median())
                        for arm in ["no_calib", "pop_slope", "pebs_ols", "pebs_shrunk"]},
        "comparisons": comparisons,
        "relative_improvement_vs_pop_pct": {
            "naive_ols": float(rel_ols),
            "shrunk": float(rel_improvement),
        },
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).write_text(json.dumps(out, indent=2))
    pu.to_parquet(Path(args.output_path).with_suffix(".parquet"))
    print(f"\n[save] {args.output_path}")


if __name__ == "__main__":
    main()
