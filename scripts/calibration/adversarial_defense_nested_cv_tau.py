"""Leakage check: the 8.58% PEBS headline depends on τ² and α_pop,β_pop
estimated IN-SAMPLE (including the rows used at test time within k-fold CV).

CONCERN (target leakage in hyperparameter estimation)
-----------------------------------------------------
The headline eval_user_score_mse_shrunk.py computes:

    # Pre-pass: per-user OLS on ALL data
    user_stats = [ols(grp.rm_score, grp.score_user) for uid, grp in df.groupby(uid)]
    tau_alpha_sq = var(user_stats.alpha) - mean(V_alpha)   # ALL rows
    tau_beta_sq  = var(user_stats.beta)  - mean(V_beta)    # ALL rows
    slope_pop, intercept_pop = np.polyfit(df.rm_score, df.score_user, 1)  # ALL rows

    # k-fold CV
    for fold in folds:
        omega = tau_alpha_sq / (tau_alpha_sq + V_hat)     # <-- tau leaked
        a_shrunk = omega * a_tr + (1 - omega) * pop_alpha  # <-- pop leaked

Both τ² and (α_pop, β_pop) are estimated from rows that include the test-fold
labels. One can reasonably object: the shrinkage-intensity schedule
knows what answers are being held out. A deployment-realistic estimator can
only use in-training-fold rows.

DEFENSE DESIGN
--------------
Re-run the within-user k=5 CV four-arm comparison with an honest, nested-CV
protocol:

  (A) PAPER ARM:
        τ², pop-slope estimated from ALL rows (matches paper/eval script)
  (B) NESTED-CV ARM:
        τ², pop-slope estimated from TRAIN ROWS of the current fold only

Report the paper's 8.58% vs the nested-CV estimate. Compute:

  * Relative RMSE gain vs pop-slope (both arms)
  * Paired Wilcoxon signed-rank
  * User-win rate
  * τ²-estimate leakage magnitude (|τ²_all − mean(τ²_train_folds)| / τ²_all)
  * Fold-level τ² dispersion

If the two agree within noise (e.g., nested gain > 0.9× of paper gain,
overlapping CIs), the concern is resolved. If nested gain collapses (e.g.,
< half of paper gain, Wilcoxon n.s.), the leakage is material and the
paper claim must be softened.

Dependencies: numpy, pandas, scipy (CPU). No GPU needed.

Expected runtime: ~30-60s on 1394 users × 5 folds × 4 arms.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scored-parquet",
        default="<DATA_ROOT>/1_Causal_RLHF/data/prism_rm_scored.parquet",
    )
    p.add_argument("--min-obs-per-user", type=int, default=6)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-path",
        default="<DATA_ROOT>/1_Causal_RLHF/results/adversarial_defense_nested_cv_tau.json",
    )
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


def estimate_moment_tau(user_stats_list):
    """Moment estimator for τ²: var(α_j) - mean(V(α_j))."""
    us = pd.DataFrame(user_stats_list)
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
    return tau_a_sq, tau_b_sq


def fit_pop_slope(df_train: pd.DataFrame):
    """Global pop-slope from all rows (not grouped)."""
    slope, intercept = np.polyfit(df_train.rm_score, df_train.score_user, 1)
    return float(intercept), float(slope)


def run_full_leakage_arm(df: pd.DataFrame, k_folds: int, rng, min_obs: int):
    """Paper arm: τ² and pop-slope from ALL rows (leakage). Matches paper eval."""
    # Step 1: τ², pop-slope from ALL rows
    all_user_stats = []
    for uid, grp in df.groupby("user_id"):
        if len(grp) < min_obs:
            continue
        a, b, Va, Vb = ols_with_V(
            grp.rm_score.to_numpy(), grp.score_user.to_numpy().astype(float)
        )
        all_user_stats.append(
            {"user_id": uid, "alpha": a, "beta": b, "V_alpha": Va, "V_beta": Vb}
        )
    tau_a_sq_global, tau_b_sq_global = estimate_moment_tau(all_user_stats)
    pop_alpha_global, pop_beta_global = fit_pop_slope(df)

    # Step 2: CV evaluation using GLOBAL τ², pop-slope
    per_user_rows = []
    for uid, grp in df.groupby("user_id"):
        n = len(grp)
        if n < min_obs:
            continue
        x = grp.rm_score.to_numpy()
        y = grp.score_user.to_numpy().astype(float)
        folds = kfold_split(n, k_folds, rng)
        sq = {"pop_slope": [], "pebs_shrunk": []}
        for tr, te in folds:
            if len(te) == 0:
                continue
            x_tr, y_tr = x[tr], y[tr]
            x_te, y_te = x[te], y[te]
            a, b, Va, Vb = ols_with_V(x_tr, y_tr)
            # Pop-slope from GLOBAL
            y_hat_ps = pop_alpha_global + pop_beta_global * x_te
            sq["pop_slope"].extend(((y_hat_ps - y_te) ** 2).tolist())
            # Shrunk with GLOBAL τ², pop
            omega_a = tau_a_sq_global / (tau_a_sq_global + Va) if np.isfinite(Va) else 0.0
            omega_b = tau_b_sq_global / (tau_b_sq_global + Vb) if np.isfinite(Vb) else 0.0
            a_s = omega_a * a + (1 - omega_a) * pop_alpha_global
            b_s = omega_b * b + (1 - omega_b) * pop_beta_global
            y_hat_sh = a_s + b_s * x_te
            sq["pebs_shrunk"].extend(((y_hat_sh - y_te) ** 2).tolist())
        per_user_rows.append(
            {
                "user_id": uid,
                "rmse_pop_slope": float(np.sqrt(np.mean(sq["pop_slope"]))),
                "rmse_pebs_shrunk": float(np.sqrt(np.mean(sq["pebs_shrunk"]))),
            }
        )
    return (
        pd.DataFrame(per_user_rows),
        {"tau_alpha_sq": tau_a_sq_global, "tau_beta_sq": tau_b_sq_global,
         "pop_alpha": pop_alpha_global, "pop_beta": pop_beta_global},
    )


def run_nested_cv_arm(df: pd.DataFrame, k_folds: int, rng, min_obs: int):
    """Honest arm: τ² and pop-slope estimated per-fold from TRAIN rows only.

    For each of the k folds, we:
      1. Build a train subset = all rows across all users EXCEPT
         the test-fold rows of every user.
      2. Estimate τ², pop_alpha, pop_beta on this train-pool.
      3. Compute shrinkage for each user's test fold using those fold-specific hyperparams.
    """
    per_user_rows = []
    fold_tau_records = []

    # Pre-assign user-level fold membership so fold k's train set is reproducible
    user_fold_map = {}  # uid -> dict{fold_i -> (train_idx, test_idx)}
    for uid, grp in df.groupby("user_id"):
        n = len(grp)
        if n < min_obs:
            continue
        user_fold_map[uid] = {
            "folds": kfold_split(n, k_folds, rng),
            "x": grp.rm_score.to_numpy(),
            "y": grp.score_user.to_numpy().astype(float),
            "global_idx": grp.index.to_numpy(),
        }

    uids_valid = list(user_fold_map.keys())
    print(f"  [nested_cv] {len(uids_valid)} users × {k_folds} folds")

    # For each fold, compute τ² from training-fold rows only across all users
    for fold_i in range(k_folds):
        # Assemble the TRAIN POOL for this fold
        train_rows_mask = np.ones(len(df), dtype=bool)  # start with all
        for uid in uids_valid:
            entry = user_fold_map[uid]
            _, test_idx = entry["folds"][fold_i]
            global_test_idx = entry["global_idx"][test_idx]
            train_rows_mask[global_test_idx] = False
        df_train_fold = df[train_rows_mask]

        # Estimate τ² from train rows grouped by user
        fold_user_stats = []
        for uid in uids_valid:
            entry = user_fold_map[uid]
            train_idx, _ = entry["folds"][fold_i]
            x_tr, y_tr = entry["x"][train_idx], entry["y"][train_idx]
            if len(x_tr) < 2:
                continue
            a, b, Va, Vb = ols_with_V(x_tr, y_tr)
            fold_user_stats.append(
                {"user_id": uid, "alpha": a, "beta": b, "V_alpha": Va, "V_beta": Vb}
            )
        tau_a_fold, tau_b_fold = estimate_moment_tau(fold_user_stats)
        pop_alpha_fold, pop_beta_fold = fit_pop_slope(df_train_fold)
        fold_tau_records.append(
            {
                "fold": fold_i,
                "tau_alpha_sq": tau_a_fold,
                "tau_beta_sq": tau_b_fold,
                "pop_alpha": pop_alpha_fold,
                "pop_beta": pop_beta_fold,
            }
        )

        # Persist per-fold hyperparams on entry
        for rec in fold_user_stats:
            user_fold_map[rec["user_id"]].setdefault("fold_params", {})[fold_i] = {
                "Va": rec["V_alpha"],
                "Vb": rec["V_beta"],
                "a": rec["alpha"],
                "b": rec["beta"],
                "tau_a": tau_a_fold,
                "tau_b": tau_b_fold,
                "pop_a": pop_alpha_fold,
                "pop_b": pop_beta_fold,
            }

    # Now compute per-user RMSE using per-fold hyperparams
    for uid in uids_valid:
        entry = user_fold_map[uid]
        sq = {"pop_slope": [], "pebs_shrunk": []}
        for fold_i in range(k_folds):
            fp = entry.get("fold_params", {}).get(fold_i)
            if fp is None:
                continue
            train_idx, test_idx = entry["folds"][fold_i]
            if len(test_idx) == 0:
                continue
            x_te = entry["x"][test_idx]
            y_te = entry["y"][test_idx]
            # Pop-slope from this fold's train pool
            y_hat_ps = fp["pop_a"] + fp["pop_b"] * x_te
            sq["pop_slope"].extend(((y_hat_ps - y_te) ** 2).tolist())
            # Shrunk with this fold's τ², pop-slope
            omega_a = (
                fp["tau_a"] / (fp["tau_a"] + fp["Va"]) if np.isfinite(fp["Va"]) else 0.0
            )
            omega_b = (
                fp["tau_b"] / (fp["tau_b"] + fp["Vb"]) if np.isfinite(fp["Vb"]) else 0.0
            )
            a_s = omega_a * fp["a"] + (1 - omega_a) * fp["pop_a"]
            b_s = omega_b * fp["b"] + (1 - omega_b) * fp["pop_b"]
            y_hat_sh = a_s + b_s * x_te
            sq["pebs_shrunk"].extend(((y_hat_sh - y_te) ** 2).tolist())
        if len(sq["pop_slope"]) == 0:
            continue
        per_user_rows.append(
            {
                "user_id": uid,
                "rmse_pop_slope": float(np.sqrt(np.mean(sq["pop_slope"]))),
                "rmse_pebs_shrunk": float(np.sqrt(np.mean(sq["pebs_shrunk"]))),
            }
        )
    return pd.DataFrame(per_user_rows), fold_tau_records


def compute_arm_summary(pu: pd.DataFrame, arm_label: str):
    pop = pu.rmse_pop_slope.to_numpy()
    sh = pu.rmse_pebs_shrunk.to_numpy()
    rel_gain_pct = 100 * (pop.mean() - sh.mean()) / pop.mean()
    wilcox = stats.wilcoxon(sh, pop, alternative="less")  # test sh < pop
    # 64.2% figure from paper
    win_rate = float((sh < pop).mean())
    return {
        "arm": arm_label,
        "n_users": int(len(pu)),
        "mean_rmse_pop_slope": float(pop.mean()),
        "mean_rmse_pebs_shrunk": float(sh.mean()),
        "relative_gain_pct": float(rel_gain_pct),
        "wilcoxon_p": float(wilcox.pvalue),
        "user_win_rate": win_rate,
        "median_delta_rmse": float(np.median(pop - sh)),
    }


def bootstrap_ci(pu: pd.DataFrame, B: int, seed: int):
    """Cluster bootstrap by user: resample users with replacement, recompute rel gain.

    Returns 95% percentile CI on (mean_rmse_pop − mean_rmse_shrunk) / mean_rmse_pop.
    """
    pop = pu.rmse_pop_slope.to_numpy()
    sh = pu.rmse_pebs_shrunk.to_numpy()
    n = len(pu)
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        p = pop[idx]
        s = sh[idx]
        gains.append(100 * (p.mean() - s.mean()) / p.mean())
    gains = np.array(gains)
    return float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))


def main():
    args = parse_args()
    t0 = time.time()
    rng_full = np.random.default_rng(args.seed)
    rng_nested = np.random.default_rng(args.seed)  # same seed => identical fold assignments

    df = (
        pd.read_parquet(args.scored_parquet)
        .dropna(subset=["score_user"])
        .reset_index(drop=True)
    )
    print(f"[load] {len(df)} utterances, {df.user_id.nunique()} users")

    # ---- Arm A: paper/leaked ----
    print("\n=== ARM A: paper protocol (global τ², global pop-slope) ===")
    pu_A, global_params = run_full_leakage_arm(df, args.k_folds, rng_full, args.min_obs_per_user)
    print(f"  global τ_α² = {global_params['tau_alpha_sq']:.3f}")
    print(f"  global τ_β² = {global_params['tau_beta_sq']:.3f}")
    print(f"  global α_pop = {global_params['pop_alpha']:.3f}, β_pop = {global_params['pop_beta']:.3f}")

    summary_A = compute_arm_summary(pu_A, "paper_leaked")
    print(f"  mean RMSE pop-slope = {summary_A['mean_rmse_pop_slope']:.3f}")
    print(f"  mean RMSE shrunk    = {summary_A['mean_rmse_pebs_shrunk']:.3f}")
    print(f"  RELATIVE GAIN       = {summary_A['relative_gain_pct']:+.3f}%")
    print(f"  Wilcoxon p          = {summary_A['wilcoxon_p']:.3e}")
    print(f"  user win-rate       = {summary_A['user_win_rate']:.3%}")

    # ---- Arm B: nested CV ----
    print("\n=== ARM B: nested CV (per-fold τ², per-fold pop-slope) ===")
    pu_B, fold_taus = run_nested_cv_arm(df, args.k_folds, rng_nested, args.min_obs_per_user)
    fold_tau_alpha = [r["tau_alpha_sq"] for r in fold_taus]
    fold_tau_beta = [r["tau_beta_sq"] for r in fold_taus]
    fold_pop_alpha = [r["pop_alpha"] for r in fold_taus]
    fold_pop_beta = [r["pop_beta"] for r in fold_taus]
    print(f"  fold τ_α²: mean={np.mean(fold_tau_alpha):.3f}, range=[{min(fold_tau_alpha):.3f}, {max(fold_tau_alpha):.3f}]")
    print(f"  fold τ_β²: mean={np.mean(fold_tau_beta):.3f}, range=[{min(fold_tau_beta):.3f}, {max(fold_tau_beta):.3f}]")
    print(f"  fold α_pop: mean={np.mean(fold_pop_alpha):.3f}, range=[{min(fold_pop_alpha):.3f}, {max(fold_pop_alpha):.3f}]")
    print(f"  fold β_pop: mean={np.mean(fold_pop_beta):.3f}, range=[{min(fold_pop_beta):.3f}, {max(fold_pop_beta):.3f}]")

    summary_B = compute_arm_summary(pu_B, "nested_cv_honest")
    print(f"  mean RMSE pop-slope = {summary_B['mean_rmse_pop_slope']:.3f}")
    print(f"  mean RMSE shrunk    = {summary_B['mean_rmse_pebs_shrunk']:.3f}")
    print(f"  RELATIVE GAIN       = {summary_B['relative_gain_pct']:+.3f}%")
    print(f"  Wilcoxon p          = {summary_B['wilcoxon_p']:.3e}")
    print(f"  user win-rate       = {summary_B['user_win_rate']:.3%}")

    # ---- Leakage magnitude ----
    tau_leak_alpha = (
        abs(global_params["tau_alpha_sq"] - np.mean(fold_tau_alpha))
        / global_params["tau_alpha_sq"]
    )
    tau_leak_beta = (
        abs(global_params["tau_beta_sq"] - np.mean(fold_tau_beta))
        / global_params["tau_beta_sq"]
    )
    pop_alpha_leak = (
        abs(global_params["pop_alpha"] - np.mean(fold_pop_alpha))
        / global_params["pop_alpha"]
    )
    pop_beta_leak = (
        abs(global_params["pop_beta"] - np.mean(fold_pop_beta))
        / global_params["pop_beta"]
    )
    print("\n=== LEAKAGE MAGNITUDE (global vs mean-of-folds) ===")
    print(f"  τ_α² leakage = {tau_leak_alpha:.2%}")
    print(f"  τ_β² leakage = {tau_leak_beta:.2%}")
    print(f"  α_pop leakage = {pop_alpha_leak:.2%}")
    print(f"  β_pop leakage = {pop_beta_leak:.2%}")

    # ---- Paired comparison: pu_A vs pu_B on SAME users ----
    # Both arms iterate the SAME users with SAME folds (same RNG seed), so
    # we can pair the per-user shrunk RMSEs.
    merged = pd.merge(
        pu_A[["user_id", "rmse_pebs_shrunk"]].rename(
            columns={"rmse_pebs_shrunk": "rmse_A"}
        ),
        pu_B[["user_id", "rmse_pebs_shrunk"]].rename(
            columns={"rmse_pebs_shrunk": "rmse_B"}
        ),
        on="user_id",
    )
    delta = merged.rmse_B - merged.rmse_A  # positive => nested is worse
    print(f"\n=== PAIRED A-vs-B DIFFERENCE (pebs_shrunk only) ===")
    print(f"  median(B - A) RMSE = {delta.median():+.4f}")
    print(f"  mean(B - A) RMSE   = {delta.mean():+.4f}")
    print(f"  fraction B ≥ A     = {(delta >= 0).mean():.3%}")
    wilcox_AB = stats.wilcoxon(merged.rmse_A, merged.rmse_B, alternative="two-sided")
    print(f"  Wilcoxon A vs B p  = {wilcox_AB.pvalue:.3e}")

    # ---- Bootstrap CIs on each arm's relative gain ----
    print("\n=== Cluster-bootstrap 95% CIs (B=2000) ===")
    B = 2000
    lo_A, hi_A = bootstrap_ci(pu_A, B, args.seed)
    lo_B, hi_B = bootstrap_ci(pu_B, B, args.seed + 1)
    print(f"  Arm A (leaked) gain: {summary_A['relative_gain_pct']:.3f}% [{lo_A:.3f}, {hi_A:.3f}]")
    print(f"  Arm B (nested) gain: {summary_B['relative_gain_pct']:.3f}% [{lo_B:.3f}, {hi_B:.3f}]")

    # ---- Verdict ----
    gain_ratio = summary_B["relative_gain_pct"] / summary_A["relative_gain_pct"]
    ci_overlap = not (hi_B < lo_A or hi_A < lo_B)
    verdict = {
        "nested_gain_pct": summary_B["relative_gain_pct"],
        "paper_gain_pct": summary_A["relative_gain_pct"],
        "ratio_nested_over_paper": float(gain_ratio),
        "cis_overlap": bool(ci_overlap),
        "nested_ci_low": lo_B,
        "nested_ci_high": hi_B,
        "paper_ci_low": lo_A,
        "paper_ci_high": hi_A,
    }
    if gain_ratio >= 0.9 and ci_overlap:
        verdict["status"] = "DEFENDED"
        verdict["rationale"] = (
            f"Nested-CV gain {summary_B['relative_gain_pct']:.2f}% is "
            f"{gain_ratio:.2%} of paper gain {summary_A['relative_gain_pct']:.2f}% "
            f"and 95% CIs overlap; target leakage is ≤10% and does not "
            f"invalidate the headline."
        )
    elif gain_ratio >= 0.7:
        verdict["status"] = "PARTIAL"
        verdict["rationale"] = (
            f"Nested-CV gain {summary_B['relative_gain_pct']:.2f}% is "
            f"{gain_ratio:.2%} of paper gain {summary_A['relative_gain_pct']:.2f}%; "
            f"some in-sample τ²-estimation effect exists but magnitude "
            f"is not catastrophic; paper claim stands with honest-protocol disclosure."
        )
    else:
        verdict["status"] = "REFUTED"
        verdict["rationale"] = (
            f"Nested-CV gain {summary_B['relative_gain_pct']:.2f}% collapses to "
            f"{gain_ratio:.2%} of paper gain {summary_A['relative_gain_pct']:.2f}%; "
            f"target leakage is material and the paper headline must be softened."
        )
    print(f"\n=== VERDICT: {verdict['status']} ===")
    print(f"  {verdict['rationale']}")
    print(f"  (wall: {time.time() - t0:.1f}s)")

    # ---- Save ----
    out = {
        "concern": (
            "τ² and pop-slope are estimated from ALL rows (including k-fold test rows) "
            "in the paper's evaluation script. This is target leakage in the "
            "hyperparameter-estimation pass. Does the headline 8.58% survive when τ² "
            "and pop-slope are estimated only from the training fold?"
        ),
        "config": {
            "n_users_evaluated": int(len(pu_B)),
            "k_folds": args.k_folds,
            "seed": args.seed,
            "min_obs_per_user": args.min_obs_per_user,
        },
        "arm_A_paper_leaked": summary_A,
        "arm_B_nested_cv_honest": summary_B,
        "global_hyperparams": global_params,
        "fold_hyperparams": fold_taus,
        "leakage_magnitude": {
            "tau_alpha_rel_diff": float(tau_leak_alpha),
            "tau_beta_rel_diff": float(tau_leak_beta),
            "pop_alpha_rel_diff": float(pop_alpha_leak),
            "pop_beta_rel_diff": float(pop_beta_leak),
        },
        "paired_A_vs_B": {
            "median_delta_rmse": float(delta.median()),
            "mean_delta_rmse": float(delta.mean()),
            "frac_B_ge_A": float((delta >= 0).mean()),
            "wilcoxon_two_sided_p": float(wilcox_AB.pvalue),
        },
        "bootstrap_ci_gain_pct": {
            "arm_A_paper": {"low": lo_A, "high": hi_A, "B": B},
            "arm_B_nested": {"low": lo_B, "high": hi_B, "B": B},
        },
        "verdict": verdict,
        "wall_seconds": float(time.time() - t0),
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).write_text(json.dumps(out, indent=2))
    print(f"\n[save] {args.output_path}")


if __name__ == "__main__":
    main()
