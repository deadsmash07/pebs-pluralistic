"""5-arm H2e eval — add Skywork frontier-RM baseline.

Arms
----
1. no_calib       : predict train-fold mean of score_user (no RM at all)
2. pop_slope_qwen : α₀ + β₀ · rm_score (pop-OLS on Qwen-7B RM scores)
3. pilsd_shrunk_qwen : EB-shrunk (α_j, β_j) on Qwen RM  ← existing 8.58% headline
4. pop_slope_skywork : α₀ + β₀ · skywork_score (pop-OLS on Skywork 27B)
5. pilsd_shrunk_skywork : EB-shrunk (α_j, β_j) on Skywork  ← tests "stack PILSD on any RM"

The 4-arm table requested in the brief is really arms {1, 2, 3, 4}. We add
arm 5 as well because the paper's key rebuttal framing (risk #2: "PILSD is
orthogonal, stacks on any RM") can only be validated with both a Skywork
pop-slope AND a Skywork PILSD-shrunk comparison.

Identical k-fold structure, seed, min-obs-per-user, and EB τ estimation
method as `eval_user_score_mse_shrunk.py` so the Qwen 8.58% number is
reproduced exactly (sanity check).

Outputs
-------
- `results/track1_user_score_mse_skywork.json` : per-arm mean/median + all
  10 pairwise Wilcoxon comparisons (5 choose 2).
- `results/track1_user_score_mse_skywork.parquet` : per-user RMSE for every arm.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ARMS = [
    "no_calib",
    "pop_slope_qwen",
    "pilsd_shrunk_qwen",
    "pop_slope_skywork",
    "pilsd_shrunk_skywork",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--qwen-parquet",
                   default="data/prism_rm_scored.parquet",
                   help="Existing Qwen-7B RM scoring (column: rm_score).")
    p.add_argument("--skywork-parquet",
                   default="data/prism_skywork_scored.parquet",
                   help="Skywork 27B scoring (column: skywork_score).")
    p.add_argument("--min-obs-per-user", type=int, default=6)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-path",
                   default="results/track1_user_score_mse_skywork.json")
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


def pop_ols(df: pd.DataFrame, x_col: str) -> tuple[float, float]:
    slope, intercept = np.polyfit(df[x_col], df.score_user, 1)
    return float(intercept), float(slope)


def estimate_tau(df: pd.DataFrame, x_col: str, min_obs: int):
    """Moment-based τ²_α, τ²_β for EB shrinkage."""
    rows = []
    for uid, grp in df.groupby("user_id"):
        if len(grp) < min_obs:
            continue
        a, b, Va, Vb = ols_with_V(
            grp[x_col].to_numpy(),
            grp.score_user.to_numpy().astype(float),
        )
        rows.append({"user_id": uid, "alpha": a, "beta": b,
                     "V_alpha": Va, "V_beta": Vb})
    us = pd.DataFrame(rows)
    V_alpha_total = float(us.alpha.var())
    V_beta_total = float(us.beta.var())
    mean_Va = float(us.V_alpha.replace([np.inf, -np.inf], np.nan).dropna().mean())
    mean_Vb = float(us.V_beta.replace([np.inf, -np.inf], np.nan).dropna().mean())
    tau_a_sq = max(V_alpha_total - mean_Va, 1e-6)
    tau_b_sq = max(V_beta_total - mean_Vb, 1e-6)
    return tau_a_sq, tau_b_sq


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Load & join
    # ------------------------------------------------------------------
    qwen = pd.read_parquet(args.qwen_parquet)
    sky = pd.read_parquet(args.skywork_parquet)

    print(f"[load] qwen: {len(qwen)} rows, sky: {len(sky)} rows")
    df = qwen.merge(
        sky[["utterance_id", "skywork_score"]],
        on="utterance_id", how="inner",
    )
    df = df.dropna(subset=["score_user"]).reset_index(drop=True)
    print(f"[join] {len(df)} utterances with both scores + score_user, "
          f"{df.user_id.nunique()} users")

    # ------------------------------------------------------------------
    # Population calibrations
    # ------------------------------------------------------------------
    pop_a_q, pop_b_q = pop_ols(df, "rm_score")
    pop_a_s, pop_b_s = pop_ols(df, "skywork_score")
    print(f"[pop qwen]    α={pop_a_q:+.3f}  β={pop_b_q:+.3f}")
    print(f"[pop skywork] α={pop_a_s:+.3f}  β={pop_b_s:+.3f}")

    # Population correlations (diagnostic)
    corr_q = float(np.corrcoef(df.rm_score, df.score_user)[0, 1])
    corr_s = float(np.corrcoef(df.skywork_score, df.score_user)[0, 1])
    print(f"[pop corr] qwen={corr_q:.4f}  skywork={corr_s:.4f}")

    # ------------------------------------------------------------------
    # EB τ estimation per RM
    # ------------------------------------------------------------------
    tau_a_q, tau_b_q = estimate_tau(df, "rm_score", args.min_obs_per_user)
    tau_a_s, tau_b_s = estimate_tau(df, "skywork_score", args.min_obs_per_user)
    print(f"[EB qwen]    τ_α²={tau_a_q:.2f}  τ_β²={tau_b_q:.2f}")
    print(f"[EB skywork] τ_α²={tau_a_s:.2f}  τ_β²={tau_b_s:.2f}")

    # ------------------------------------------------------------------
    # k-fold CV per user, 5 arms
    # ------------------------------------------------------------------
    rows = []
    for uid, grp in df.groupby("user_id"):
        n = len(grp)
        if n < args.min_obs_per_user:
            continue
        x_q = grp.rm_score.to_numpy()
        x_s = grp.skywork_score.to_numpy()
        y = grp.score_user.to_numpy().astype(float)

        sq = {a: [] for a in ARMS}
        for train_idx, test_idx in kfold_split(n, args.k_folds, rng):
            if len(test_idx) == 0:
                continue
            y_tr, y_te = y[train_idx], y[test_idx]
            x_q_tr, x_q_te = x_q[train_idx], x_q[test_idx]
            x_s_tr, x_s_te = x_s[train_idx], x_s[test_idx]

            # 1. no_calib
            yh = np.full_like(y_te, np.mean(y_tr))
            sq["no_calib"].extend(((yh - y_te) ** 2).tolist())

            # 2. pop_slope_qwen
            yh = pop_a_q + pop_b_q * x_q_te
            sq["pop_slope_qwen"].extend(((yh - y_te) ** 2).tolist())

            # 3. pilsd_shrunk_qwen
            a, b, Va, Vb = ols_with_V(x_q_tr, y_tr)
            wa = tau_a_q / (tau_a_q + Va) if np.isfinite(Va) else 0.0
            wb = tau_b_q / (tau_b_q + Vb) if np.isfinite(Vb) else 0.0
            a_s_ = wa * a + (1 - wa) * pop_a_q
            b_s_ = wb * b + (1 - wb) * pop_b_q
            yh = a_s_ + b_s_ * x_q_te
            sq["pilsd_shrunk_qwen"].extend(((yh - y_te) ** 2).tolist())

            # 4. pop_slope_skywork
            yh = pop_a_s + pop_b_s * x_s_te
            sq["pop_slope_skywork"].extend(((yh - y_te) ** 2).tolist())

            # 5. pilsd_shrunk_skywork
            a, b, Va, Vb = ols_with_V(x_s_tr, y_tr)
            wa = tau_a_s / (tau_a_s + Va) if np.isfinite(Va) else 0.0
            wb = tau_b_s / (tau_b_s + Vb) if np.isfinite(Vb) else 0.0
            a_sh = wa * a + (1 - wa) * pop_a_s
            b_sh = wb * b + (1 - wb) * pop_b_s
            yh = a_sh + b_sh * x_s_te
            sq["pilsd_shrunk_skywork"].extend(((yh - y_te) ** 2).tolist())

        rows.append({
            "user_id": uid,
            "n": n,
            **{f"rmse_{a}": float(np.sqrt(np.mean(sq[a]))) for a in ARMS},
        })

    pu = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\n=== 5-arm within-user CV ({len(pu)} users, k={args.k_folds}) ===")
    print(f"{'Arm':<28}{'Mean RMSE':>12}{'Median':>12}{'vs pop_qwen':>14}")
    base = pu["rmse_pop_slope_qwen"].mean()
    for a in ARMS:
        m = pu[f"rmse_{a}"].mean()
        med = pu[f"rmse_{a}"].median()
        rel = 100 * (base - m) / base
        print(f"  {a:<26}{m:>12.3f}{med:>12.3f}{rel:>+13.2f}%")

    # All pairwise comparisons
    print(f"\n=== Pairwise Wilcoxon (two-sided) ===")
    pair_records = {}
    for a, b in combinations(ARMS, 2):
        ca = pu[f"rmse_{a}"].to_numpy()
        cb = pu[f"rmse_{b}"].to_numpy()
        delta = ca - cb
        w = stats.wilcoxon(ca, cb, alternative="two-sided")
        rec = {
            "mean_delta": float(delta.mean()),
            "frac_a_smaller": float((ca < cb).mean()),
            "wilcoxon_p": float(w.pvalue),
        }
        pair_records[f"{a}__vs__{b}"] = rec
        print(f"  {a} vs {b}:")
        print(f"    mean Δ = {rec['mean_delta']:+.4f}  "
              f"{a} wins {rec['frac_a_smaller']:.1%}  "
              f"p = {rec['wilcoxon_p']:.3e}")

    # Headline relative improvements vs pop_slope_qwen
    rel_improvements = {}
    for a in ARMS:
        rel = 100 * (base - pu[f"rmse_{a}"].mean()) / base
        rel_improvements[a] = float(rel)

    out = {
        "n_users": int(len(pu)),
        "k_folds": args.k_folds,
        "pop_correlations": {"qwen": corr_q, "skywork": corr_s},
        "pop_calibration": {
            "qwen": {"alpha": pop_a_q, "beta": pop_b_q},
            "skywork": {"alpha": pop_a_s, "beta": pop_b_s},
        },
        "eb": {
            "qwen": {"tau_alpha_sq": tau_a_q, "tau_beta_sq": tau_b_q},
            "skywork": {"tau_alpha_sq": tau_a_s, "tau_beta_sq": tau_b_s},
        },
        "rmse_mean": {a: float(pu[f"rmse_{a}"].mean()) for a in ARMS},
        "rmse_median": {a: float(pu[f"rmse_{a}"].median()) for a in ARMS},
        "relative_improvement_vs_pop_qwen_pct": rel_improvements,
        "pairwise_comparisons": pair_records,
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).write_text(json.dumps(out, indent=2))
    pu.to_parquet(Path(args.output_path).with_suffix(".parquet"))
    print(f"\n[save] {args.output_path}")


if __name__ == "__main__":
    main()
