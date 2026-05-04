"""Temporal-CV version of H2e — adversarial review attack #4.

Random k-fold CV treats (α_j, β_j) as time-invariant. Track 3's thesis is
that latent annotator effects drift. If calibrators drift within a user's
collection window, the random-fold 8.58% headline overstates generalization
to future rollouts.

This script re-runs the 4-arm H2e eval under a STRICT TEMPORAL split:
  - For each user, sort utterances by (generated_datetime, utterance_id)
  - First 80% → train, last 20% → test (no shuffle)
  - Compute per-user RMSE for all 4 arms
  - Report mean RMSE + Wilcoxon-paired test vs random-CV baseline
  - Bootstrap CI on temporal-CV RMSE across 30 seeds (each seed shuffles
    user-level ordering for the user-weighted mean; within-user order is
    fixed by timestamp)
  - Compute split-half calibrator drift: fit OLS on first-half vs second-half,
    correlate (α_first, α_second) and (β_first, β_second).

Timestamp source: PRISM's `conversations` config has `generated_datetime`
(conversation-level). Joined via `conversation_id` in the scored parquet.
Within-conversation ordering uses (turn, within_turn_id, utterance_id ordinal)
as sub-second tiebreak, since `generated_datetime` is conversation-level.

Caveat: PRISM's entire collection window is 30 days and 90% of users span
<1.5 hours. So "temporal drift" here mostly means drift within a session
(reviewer fatigue, anchoring effects, prompt-type-within-session). For real
long-term drift T3 addresses on OASST2 (15-month span), this is the best
public-RLHF-dataset proxy we have.

Expected outcomes:
  - If PILSD survives temporal split → headline robust to drift
  - If headline degrades → honest finding; PILSD serves IID holdout, not
    temporal extrapolation
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scored-parquet", default="data/prism_rm_scored.parquet")
    p.add_argument("--hf-dataset", default="HannahRoseKirk/prism-alignment")
    p.add_argument("--min-obs-per-user", type=int, default=6)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--n-bootstrap", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--timestamp-cache",
        default="data/prism_conversation_timestamps.parquet",
        help="Path to cached conversation_id -> generated_datetime map.",
    )
    p.add_argument(
        "--output-path",
        default="results/track1_temporal_cv/temporal_cv_results.json",
    )
    return p.parse_args()


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


def load_timestamps(hf_id: str, cache_path: Path) -> pd.DataFrame:
    """Load PRISM conversations table → (conversation_id, generated_datetime)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"[ts] cache hit: {cache_path}")
        return pd.read_parquet(cache_path)
    from datasets import load_dataset

    print(f"[ts] downloading {hf_id} conversations for timestamps …")
    ds = load_dataset(hf_id, "conversations", split="train")
    df = ds.to_pandas()[["conversation_id", "generated_datetime"]].copy()
    df["generated_datetime"] = pd.to_datetime(df["generated_datetime"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    print(f"[ts] cached {len(df)} conversation-timestamps to {cache_path}")
    return df


def ut_ordinal(uid: str) -> int:
    m = re.match(r"ut(\d+)", str(uid))
    return int(m.group(1)) if m else 0


def temporal_sort_key(df: pd.DataFrame) -> pd.DataFrame:
    """Within-user ordering: (generated_datetime, turn, within_turn_id, ut_ord)."""
    df = df.copy()
    df["ut_ord"] = df["utterance_id"].apply(ut_ordinal)
    return df.sort_values(
        ["generated_datetime", "turn", "within_turn_id", "ut_ord"],
        kind="mergesort",
    )


def compute_arm_rmses(
    df_user: pd.DataFrame,
    test_fraction: float,
    pop_alpha: float,
    pop_beta: float,
    tau_a_sq: float,
    tau_b_sq: float,
) -> dict | None:
    """Run all 4 arms for one user on a single train/test split."""
    n = len(df_user)
    n_test = max(1, int(round(n * test_fraction)))
    if n - n_test < 2:
        return None
    train = df_user.iloc[: n - n_test]
    test = df_user.iloc[n - n_test :]
    x_tr = train["rm_score"].to_numpy()
    y_tr = train["score_user"].to_numpy().astype(float)
    x_te = test["rm_score"].to_numpy()
    y_te = test["score_user"].to_numpy().astype(float)
    out = {}
    # no_calib: mean of train
    yh = np.full_like(y_te, np.mean(y_tr))
    out["no_calib"] = float(np.sqrt(np.mean((yh - y_te) ** 2)))
    # pop_slope
    yh = pop_alpha + pop_beta * x_te
    out["pop_slope"] = float(np.sqrt(np.mean((yh - y_te) ** 2)))
    # PILSD naive OLS
    a, b, Va, Vb = ols_with_V(x_tr, y_tr)
    yh = a + b * x_te
    out["pilsd_ols"] = float(np.sqrt(np.mean((yh - y_te) ** 2)))
    # PILSD shrunk
    omega_a = tau_a_sq / (tau_a_sq + Va) if np.isfinite(Va) else 0.0
    omega_b = tau_b_sq / (tau_b_sq + Vb) if np.isfinite(Vb) else 0.0
    a_s = omega_a * a + (1 - omega_a) * pop_alpha
    b_s = omega_b * b + (1 - omega_b) * pop_beta
    yh = a_s + b_s * x_te
    out["pilsd_shrunk"] = float(np.sqrt(np.mean((yh - y_te) ** 2)))
    # diagnostic: per-user alpha/beta under temporal train
    out["_alpha_train"] = a
    out["_beta_train"] = b
    out["_n_train"] = int(len(train))
    out["_n_test"] = int(len(test))
    return out


def split_half_calibrators(
    df_user: pd.DataFrame,
) -> dict | None:
    """Fit OLS on first half vs second half, return (α, β) for each."""
    n = len(df_user)
    if n < 8:
        return None
    mid = n // 2
    first = df_user.iloc[:mid]
    second = df_user.iloc[mid:]
    a1, b1, _, _ = ols_with_V(
        first["rm_score"].to_numpy(), first["score_user"].to_numpy().astype(float)
    )
    a2, b2, _, _ = ols_with_V(
        second["rm_score"].to_numpy(), second["score_user"].to_numpy().astype(float)
    )
    return {"alpha_first": a1, "beta_first": b1, "alpha_second": a2, "beta_second": b2}


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # 1. Load scored RM + PRISM timestamps
    df = (
        pd.read_parquet(args.scored_parquet)
        .dropna(subset=["score_user"])
        .reset_index(drop=True)
    )
    print(f"[load] {len(df)} utterances, {df.user_id.nunique()} users")
    ts_df = load_timestamps(args.hf_dataset, args.timestamp_cache)
    before_n = len(df)
    df = df.merge(ts_df, on="conversation_id", how="left")
    missing = df["generated_datetime"].isna().sum()
    print(f"[join] joined timestamps; {missing}/{before_n} rows missing → dropped")
    df = df.dropna(subset=["generated_datetime"]).reset_index(drop=True)
    print(
        f"[join] temporal span: {df.generated_datetime.min()} → "
        f"{df.generated_datetime.max()} "
        f"({(df.generated_datetime.max()-df.generated_datetime.min()).days} days)"
    )

    # 2. Global pop-slope calibration (same as random-CV headline)
    slope_pop, intercept_pop = np.polyfit(df.rm_score, df.score_user, 1)
    pop_alpha = float(intercept_pop)
    pop_beta = float(slope_pop)
    print(f"[pop] α₀={pop_alpha:.3f}  β₀={pop_beta:.3f}")

    # 3. EB hyperparams (same as random-CV — estimate τ² from per-user OLS
    # using ALL data; this is not the leakage concern since τ² is a
    # population-level shrinkage prior, not a prediction).
    user_stats = []
    for uid, grp in df.groupby("user_id"):
        if len(grp) < args.min_obs_per_user:
            continue
        a, b, Va, Vb = ols_with_V(
            grp.rm_score.to_numpy(), grp.score_user.to_numpy().astype(float)
        )
        user_stats.append(
            {
                "user_id": uid,
                "alpha": a,
                "beta": b,
                "V_alpha": Va,
                "V_beta": Vb,
                "n": len(grp),
            }
        )
    us = pd.DataFrame(user_stats)
    V_alpha_total = float(us.alpha.var())
    V_beta_total = float(us.beta.var())
    mean_samp_V_alpha = float(
        us.V_alpha.replace([np.inf, -np.inf], np.nan).dropna().mean()
    )
    mean_samp_V_beta = float(
        us.V_beta.replace([np.inf, -np.inf], np.nan).dropna().mean()
    )
    tau_a_sq = max(V_alpha_total - mean_samp_V_alpha, 1e-6)
    tau_b_sq = max(V_beta_total - mean_samp_V_beta, 1e-6)
    print(f"[EB] τ_α²={tau_a_sq:.3f}  τ_β²={tau_b_sq:.3f}")

    # 4. Temporal split per user
    per_user_rows = []
    split_half_rows = []
    user_span_days = []
    for uid, grp in df.groupby("user_id"):
        if len(grp) < args.min_obs_per_user:
            continue
        grp_sorted = temporal_sort_key(grp)
        span = (
            grp_sorted.generated_datetime.max()
            - grp_sorted.generated_datetime.min()
        ).total_seconds() / 86400.0
        user_span_days.append(span)
        res = compute_arm_rmses(
            grp_sorted,
            args.test_fraction,
            pop_alpha,
            pop_beta,
            tau_a_sq,
            tau_b_sq,
        )
        if res is None:
            continue
        row = {"user_id": uid, "n": len(grp_sorted), "span_days": span}
        for arm in ["no_calib", "pop_slope", "pilsd_ols", "pilsd_shrunk"]:
            row[f"rmse_{arm}"] = res[arm]
        row["alpha_train"] = res["_alpha_train"]
        row["beta_train"] = res["_beta_train"]
        per_user_rows.append(row)
        sh = split_half_calibrators(grp_sorted)
        if sh is not None:
            sh["user_id"] = uid
            sh["n"] = len(grp_sorted)
            split_half_rows.append(sh)

    pu = pd.DataFrame(per_user_rows)
    sh_df = pd.DataFrame(split_half_rows)

    # 5. Headline RMSE
    print(f"\n=== 4-arm TEMPORAL CV ({len(pu)} users, 80/20) ===")
    arms = ["no_calib", "pop_slope", "pilsd_ols", "pilsd_shrunk"]
    for arm in arms:
        col = f"rmse_{arm}"
        print(
            f"  {arm:>14}: mean={pu[col].mean():.3f}  "
            f"median={pu[col].median():.3f}"
        )

    # 6. Paired Wilcoxon
    def paired(a, b):
        w = stats.wilcoxon(a, b, alternative="two-sided")
        return {
            "mean_delta": float((a - b).mean()),
            "frac_a_smaller": float((a < b).mean()),
            "wilcoxon_p": float(w.pvalue),
        }

    comparisons = {
        "shrunk_vs_ols": paired(pu.rmse_pilsd_shrunk, pu.rmse_pilsd_ols),
        "shrunk_vs_pop": paired(pu.rmse_pilsd_shrunk, pu.rmse_pop_slope),
        "ols_vs_pop": paired(pu.rmse_pilsd_ols, pu.rmse_pop_slope),
    }
    print(f"\n=== Paired comparisons ===")
    for name, d in comparisons.items():
        print(
            f"  {name:>18}: Δ={d['mean_delta']:+.4f}  "
            f"wins {d['frac_a_smaller']:.1%}  "
            f"Wilcoxon p={d['wilcoxon_p']:.3e}"
        )

    rel_shrunk = (
        100
        * (pu.rmse_pop_slope.mean() - pu.rmse_pilsd_shrunk.mean())
        / pu.rmse_pop_slope.mean()
    )
    rel_ols = (
        100
        * (pu.rmse_pop_slope.mean() - pu.rmse_pilsd_ols.mean())
        / pu.rmse_pop_slope.mean()
    )
    print(f"\n=== Relative improvement vs pop-slope (TEMPORAL) ===")
    print(f"  PILSD naive OLS: {rel_ols:+.2f}%")
    print(f"  PILSD shrunk:    {rel_shrunk:+.2f}%")

    # 7. Bootstrap CI on relative improvement over users (user-clustered resample)
    boot_shrunk = []
    boot_ols = []
    for seed in range(args.n_bootstrap):
        local_rng = np.random.default_rng(args.seed + seed)
        idx = local_rng.integers(0, len(pu), size=len(pu))
        sample = pu.iloc[idx]
        rel_s = (
            100
            * (sample.rmse_pop_slope.mean() - sample.rmse_pilsd_shrunk.mean())
            / sample.rmse_pop_slope.mean()
        )
        rel_o = (
            100
            * (sample.rmse_pop_slope.mean() - sample.rmse_pilsd_ols.mean())
            / sample.rmse_pop_slope.mean()
        )
        boot_shrunk.append(rel_s)
        boot_ols.append(rel_o)
    ci_s = (float(np.percentile(boot_shrunk, 2.5)), float(np.percentile(boot_shrunk, 97.5)))
    ci_o = (float(np.percentile(boot_ols, 2.5)), float(np.percentile(boot_ols, 97.5)))
    print(
        f"\n=== Bootstrap 95% CI (N={args.n_bootstrap}, user-resampled) ==="
    )
    print(f"  shrunk: [{ci_s[0]:+.2f}%, {ci_s[1]:+.2f}%]")
    print(f"  ols:    [{ci_o[0]:+.2f}%, {ci_o[1]:+.2f}%]")

    # 8. Split-half calibrator drift (α_first vs α_second, β_first vs β_second)
    if len(sh_df) > 0:
        rho_a_p = stats.pearsonr(sh_df.alpha_first, sh_df.alpha_second)
        rho_b_p = stats.pearsonr(sh_df.beta_first, sh_df.beta_second)
        rho_a_s = stats.spearmanr(sh_df.alpha_first, sh_df.alpha_second)
        rho_b_s = stats.spearmanr(sh_df.beta_first, sh_df.beta_second)
        print(f"\n=== Split-half calibrator stability (N={len(sh_df)} users) ===")
        print(
            f"  Pearson ρ(α_first, α_second) = {rho_a_p.statistic:+.3f}  "
            f"p={rho_a_p.pvalue:.3e}"
        )
        print(
            f"  Pearson ρ(β_first, β_second) = {rho_b_p.statistic:+.3f}  "
            f"p={rho_b_p.pvalue:.3e}"
        )
        print(
            f"  Spearman ρ(α) = {rho_a_s.statistic:+.3f}  "
            f"Spearman ρ(β) = {rho_b_s.statistic:+.3f}"
        )
        stable = (
            abs(rho_a_p.statistic) > 0.8 and abs(rho_b_p.statistic) > 0.8
        )
        print(
            f"  Calibrator stability verdict: "
            f"{'STABLE (both |ρ|>0.8)' if stable else 'SOME DRIFT (|ρ|<=0.8)'}"
        )
        split_half_summary = {
            "n_users": int(len(sh_df)),
            "pearson_alpha_r": float(rho_a_p.statistic),
            "pearson_alpha_p": float(rho_a_p.pvalue),
            "pearson_beta_r": float(rho_b_p.statistic),
            "pearson_beta_p": float(rho_b_p.pvalue),
            "spearman_alpha_r": float(rho_a_s.statistic),
            "spearman_beta_r": float(rho_b_s.statistic),
            "mean_alpha_delta": float(
                (sh_df.alpha_second - sh_df.alpha_first).mean()
            ),
            "mean_beta_delta": float(
                (sh_df.beta_second - sh_df.beta_first).mean()
            ),
            "stable_08_threshold": bool(stable),
        }
    else:
        split_half_summary = {"n_users": 0}

    # 9. Compare vs random-CV headline (mean RMSE paired by user)
    random_cv_path = Path("results/track1_user_score_mse_shrunk.parquet")
    random_vs_temporal = None
    if random_cv_path.exists():
        rand = pd.read_parquet(random_cv_path)
        merged = pu.merge(rand, on="user_id", suffixes=("_temp", "_rand"))
        if len(merged) > 0:
            # Within-user Δ for shrunk arm: temporal - random
            delta = (
                merged.rmse_pilsd_shrunk_temp - merged.rmse_pilsd_shrunk_rand
            )
            w_shrunk = stats.wilcoxon(
                merged.rmse_pilsd_shrunk_temp,
                merged.rmse_pilsd_shrunk_rand,
                alternative="two-sided",
            )
            random_vs_temporal = {
                "n_users": int(len(merged)),
                "mean_rmse_random_shrunk": float(merged.rmse_pilsd_shrunk_rand.mean()),
                "mean_rmse_temporal_shrunk": float(merged.rmse_pilsd_shrunk_temp.mean()),
                "mean_delta_temp_minus_random": float(delta.mean()),
                "frac_temporal_worse": float((delta > 0).mean()),
                "wilcoxon_p": float(w_shrunk.pvalue),
            }
            print(f"\n=== Random-CV vs Temporal-CV (shrunk arm, N={len(merged)}) ===")
            print(
                f"  mean RMSE random : {merged.rmse_pilsd_shrunk_rand.mean():.3f}"
            )
            print(
                f"  mean RMSE temporal: {merged.rmse_pilsd_shrunk_temp.mean():.3f}"
            )
            print(f"  Δ (temp - rand)  : {delta.mean():+.3f}")
            print(
                f"  % users worse under temporal: {100*(delta>0).mean():.1f}%"
            )
            print(f"  Wilcoxon p       : {w_shrunk.pvalue:.3e}")

    # 10. Save everything
    out = {
        "n_users_temporal_cv": int(len(pu)),
        "min_obs_per_user": args.min_obs_per_user,
        "test_fraction": args.test_fraction,
        "temporal_span": {
            "dataset_days": float(
                (df.generated_datetime.max() - df.generated_datetime.min()).days
            ),
            "per_user_span_days_mean": float(np.mean(user_span_days)),
            "per_user_span_days_median": float(np.median(user_span_days)),
            "per_user_span_days_p90": float(np.percentile(user_span_days, 90)),
            "per_user_span_days_max": float(np.max(user_span_days)),
        },
        "rmse_mean": {arm: float(pu[f"rmse_{arm}"].mean()) for arm in arms},
        "rmse_median": {arm: float(pu[f"rmse_{arm}"].median()) for arm in arms},
        "comparisons": comparisons,
        "relative_improvement_vs_pop_pct": {
            "naive_ols": float(rel_ols),
            "shrunk": float(rel_shrunk),
        },
        "bootstrap_ci_95": {
            "shrunk_pct": list(ci_s),
            "ols_pct": list(ci_o),
            "n_bootstrap": args.n_bootstrap,
        },
        "split_half_calibrator_drift": split_half_summary,
        "random_cv_vs_temporal_cv": random_vs_temporal,
        "eb": {
            "tau_alpha_sq": tau_a_sq,
            "tau_beta_sq": tau_b_sq,
        },
        "pop": {"alpha": pop_alpha, "beta": pop_beta},
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).write_text(json.dumps(out, indent=2))
    pu.to_parquet(Path(args.output_path).with_suffix(".parquet"))
    if len(sh_df):
        sh_df.to_parquet(
            Path(args.output_path).parent / "split_half_calibrators.parquet"
        )
    print(f"\n[save] {args.output_path}")


if __name__ == "__main__":
    main()
