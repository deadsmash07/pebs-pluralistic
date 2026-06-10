"""W-B5: Half-PEBS α-only / β-only / both ablation (gain decomposition).

This is a gain-decomposition ablation that tests the paper's
narrative that the 8.58% RMSE-reduction headline = ~4.70pp Efron-Morris
intercept-shrinkage (α-only) + ~1.17pp PEBS-specific slope-shrinkage
(β-only). The ablation compares three arms on the canonical k=5
within-user CV protocol:

  1. α-only (β=β_pop): pure Efron-Morris floor — shrinks user-specific
     intercept α_j toward population α_pop with weight ω_α; uses
     population slope β_pop (NO per-user slope adjustment)
  2. β-only (α=α_pop): pure slope-shrinkage — shrinks user-specific
     slope β_j toward β_pop with weight ω_β; uses population intercept
     α_pop (NO per-user intercept adjustment)
  3. both (canonical PEBS): joint α+β shrinkage (the paper's method)

Outcome classification:
  * CONFIRMED-COMPOUND-NEEDED — both α-only AND β-only strictly worse
    than canonical (gain delta ≤ -0.5pp at 95% CI) AND each ablation
    contributes ≥10% of canonical gain
  * PARTIAL-COMPOUND-NEEDED — directional dominance but CIs overlap
    OR one component contributes <10% of canonical gain
  * TENTATIVE-INCONCLUSIVE — CIs broad / outcome ambiguous
  * REJECTED-ONE-COMPONENT-DOES-IT-ALL — α-only OR β-only ≥ canonical
    within 0.5pp 95% CI (would itself be a noteworthy honest finding;
    simpler method beats compound; strengthens paper)

Design notes:
  * Math: each arm computes a strict subset of the canonical
    PEBS shrinkage; α-only sets ω_β=0; β-only sets ω_α=0; canonical
    uses both. Verified line-aligned to eval_user_score_mse_shrunk.py
    lines 89-146.
  * Hypothesis: this script tests "which component does
    the work", NOT "does PEBS beat baseline"; canonical gain is the
    reference point. Decision-rule: gain decomposition + 95% CI on
    paired delta vs canonical.
  * No silent-bypass: each arm is a DISTINCT compute path —
    the quick-run test asserts arm rmse values are different.
  * Eval pipeline integrity: same RMSE eval as canonical
  * Reference implementation: canonical is eval_user_score_mse_shrunk.py;
    α-only / β-only are documented strict subsets per Efron-Morris 1973
  * Hyperparameters: k_folds=5, min_obs_per_user=6, seed=20260420
    (matches canonical headline seed)
  * Diagnostics: per-arm rmse, per-arm gain_pct, paired Δ vs
    canonical, 95% bootstrap CI
  * Reproducibility: single seed=20260420 (matches headline)
  * Output schema: summary.json with arms[α_only, β_only, canonical] +
    deltas vs canonical + verdict_class
  * Compute: 3 arms × ~3-5min = 9-15min wall on 64 vCPU CPU-only
  * REJECTED-ONE-COMPONENT-DOES-IT-ALL would itself be a noteworthy
    honest finding, NOT a paper-killer; the classification
    explicitly enumerates this.

Output: ``results/track1_half_pebs_ablation/{summary.json}``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

CANONICAL_SEED = 20260420  # matches canonical headline seed
N_BOOT = 2000
RNG_BOOT = 314159
DELTA_ONE_DOES_IT_PP = 0.5  # threshold where α-only or β-only ≥ canonical

# Numerical-stability guard (backported from
# `scripts/q3_half_pebs_cross_corpus.py`).
#
# Without this guard the omega computation can silently collapse onto a
# wildly extrapolated per-cluster OLS intercept when the within-cluster
# x-variance Sxx is tiny but non-zero (Sxx > 1e-12 so `ols_with_V`
# returns finite Va = sigma^2 (1/k + xbar^2/Sxx) but Va << tau^2_alpha
# means omega = tau^2_alpha / (tau^2_alpha + Va) ~ 1 collapsing
# `a_s = omega * a + (1 - omega) * pop_alpha` onto the wildly
# extrapolated per-cluster OLS intercept). On PRISM rich within-user
# x-variance keeps Va > 1e-6 always so the guard is INERT and the
# canonical W-B5 PRISM result is unchanged. On
# any future small-cluster corpus where annotators rate same response
# with replicated rm_score, Va can be < 1e-6 and the guard correctly
# clips omega = 0 (treats degenerate case as "no shrinkage information"
# per Q3's INCONCLUSIVE-DEGENERATE-X scope-limit).
V_ALPHA_BETA_FLOOR = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    default_data = (
        Path(__file__).resolve().parents[4]
        / "1_Causal_RLHF/data/prism_rm_scored.parquet"
    )
    p.add_argument("--scored-parquet", default=str(default_data))
    p.add_argument("--min-obs-per-user", type=int, default=6)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument(
        "--output-dir",
        default=str(
            Path(__file__).resolve().parents[2]
            / "../results/track1_half_pebs_ablation"
        ),
    )
    p.add_argument("--smoke", action="store_true", help="50-user smoke")
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


def ols_with_V(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
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


def run_ablation(
    df: pd.DataFrame, seed: int, k_folds: int, min_obs_per_user: int
) -> dict[str, Any]:
    """Run all three arms (α-only, β-only, canonical) on the same CV splits."""
    rng = np.random.default_rng(seed)

    # Pop calibration
    slope_pop, intercept_pop = np.polyfit(df.rm_score, df.score_user, 1)
    pop_alpha = float(intercept_pop)
    pop_beta = float(slope_pop)

    # τ²_α, τ²_β estimation (same as canonical)
    user_stats = []
    for uid, grp in df.groupby("user_id"):
        if len(grp) < min_obs_per_user:
            continue
        a, b, Va, Vb = ols_with_V(
            grp.rm_score.to_numpy(),
            grp.score_user.to_numpy().astype(float),
        )
        user_stats.append(
            {"alpha": a, "beta": b, "V_alpha": Va, "V_beta": Vb, "n": len(grp)}
        )
    us = pd.DataFrame(user_stats)
    V_alpha_total = float(us.alpha.var())
    V_beta_total = float(us.beta.var())
    mean_samp_V_alpha = float(us.V_alpha.replace([np.inf, -np.inf], np.nan).dropna().mean())
    mean_samp_V_beta = float(us.V_beta.replace([np.inf, -np.inf], np.nan).dropna().mean())
    tau_a_sq = max(V_alpha_total - mean_samp_V_alpha, 1e-6)
    tau_b_sq = max(V_beta_total - mean_samp_V_beta, 1e-6)

    # Per-user CV — all three arms on SAME folds
    rows = []
    for uid, grp in df.groupby("user_id"):
        n = len(grp)
        if n < min_obs_per_user:
            continue
        x = grp.rm_score.to_numpy()
        y = grp.score_user.to_numpy().astype(float)
        folds = kfold_split(n, k_folds, rng)
        sq = {"pop_slope": [], "alpha_only": [], "beta_only": [], "canonical": []}
        for train_idx, test_idx in folds:
            x_tr, y_tr = x[train_idx], y[train_idx]
            x_te, y_te = x[test_idx], y[test_idx]
            if len(x_te) == 0:
                continue
            # Baseline: pop-slope (the comparison reference)
            y_hat_ps = pop_alpha + pop_beta * x_te
            sq["pop_slope"].extend(((y_hat_ps - y_te) ** 2).tolist())

            # Per-user OLS
            a, b, Va, Vb = ols_with_V(x_tr, y_tr)
            # Numerical-stability guard: clip omega = 0 when sampling
            # variance V is degenerate (Sxx<=0 ⇒ OLS extrapolates wildly).
            # See V_ALPHA_BETA_FLOOR docstring at module top for full
            # rationale. Backported from the Q3 cross-corpus script.
            # INERT on PRISM (rich within-user x-variance keeps Va > 1e-6
            # always so existing W-B5 PRISM result is unchanged); ACTIVE
            # on any future small-cluster corpus replication.
            if not np.isfinite(Va) or Va < V_ALPHA_BETA_FLOOR:
                omega_a = 0.0
            else:
                omega_a = tau_a_sq / (tau_a_sq + Va)
            if not np.isfinite(Vb) or Vb < V_ALPHA_BETA_FLOOR:
                omega_b = 0.0
            else:
                omega_b = tau_b_sq / (tau_b_sq + Vb)

            # α-only: shrunk α + pop_β  (intercept-only EB shrinkage)
            a_s = omega_a * a + (1 - omega_a) * pop_alpha
            y_hat_a = a_s + pop_beta * x_te
            sq["alpha_only"].extend(((y_hat_a - y_te) ** 2).tolist())

            # β-only: pop_α + shrunk β  (slope-only EB shrinkage)
            b_s = omega_b * b + (1 - omega_b) * pop_beta
            y_hat_b = pop_alpha + b_s * x_te
            sq["beta_only"].extend(((y_hat_b - y_te) ** 2).tolist())

            # Canonical: shrunk α + shrunk β
            y_hat_c = a_s + b_s * x_te
            sq["canonical"].extend(((y_hat_c - y_te) ** 2).tolist())

        rows.append(
            {
                "user_id": uid,
                "n": n,
                **{f"rmse_{arm}": float(np.sqrt(np.mean(sq[arm]))) for arm in sq},
            }
        )
    pu = pd.DataFrame(rows)

    arms = ["alpha_only", "beta_only", "canonical"]
    rmse_pop = float(pu.rmse_pop_slope.mean())
    arm_data: dict[str, dict[str, Any]] = {}

    boot_rng = np.random.default_rng(RNG_BOOT)
    n_users = len(pu)

    for arm in arms:
        rmse_arm = float(pu[f"rmse_{arm}"].mean())
        rel_pct = 100.0 * (rmse_pop - rmse_arm) / rmse_pop

        # Per-user gain_pct (cluster-bootstrap unit = user)
        gains = (
            100.0 * (pu.rmse_pop_slope - pu[f"rmse_{arm}"]) / pu.rmse_pop_slope
        ).to_numpy()
        boot_means = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = boot_rng.integers(0, n_users, size=n_users)
            boot_means[b] = float(np.mean(gains[idx]))
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        # Wilcoxon vs pop_slope
        w = stats.wilcoxon(
            pu[f"rmse_{arm}"].to_numpy(),
            pu.rmse_pop_slope.to_numpy(),
            alternative="two-sided",
        )

        arm_data[arm] = {
            "rmse_mean": rmse_arm,
            "gain_pct_vs_pop": rel_pct,
            "gain_pct_ci95_lo": float(ci_lo),
            "gain_pct_ci95_hi": float(ci_hi),
            "ci_excludes_zero": bool(ci_lo > 0),
            "wilcoxon_p_vs_pop": float(w.pvalue),
        }

    # Paired deltas vs canonical (per-user gain_pct difference)
    deltas: dict[str, dict[str, Any]] = {}
    canonical_gains = (
        100.0 * (pu.rmse_pop_slope - pu.rmse_canonical) / pu.rmse_pop_slope
    ).to_numpy()
    for arm in ("alpha_only", "beta_only"):
        arm_gains = (
            100.0 * (pu.rmse_pop_slope - pu[f"rmse_{arm}"]) / pu.rmse_pop_slope
        ).to_numpy()
        paired = arm_gains - canonical_gains  # negative ⇒ canonical better
        boot_means = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = boot_rng.integers(0, n_users, size=n_users)
            boot_means[b] = float(np.mean(paired[idx]))
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        w = stats.wilcoxon(arm_gains, canonical_gains, alternative="two-sided")
        deltas[arm] = {
            "delta_pct_vs_canonical": float(np.mean(paired)),
            "delta_ci95_lo": float(ci_lo),
            "delta_ci95_hi": float(ci_hi),
            "wilcoxon_p_arm_vs_canonical": float(w.pvalue),
        }

    return {
        "n_users": int(len(pu)),
        "tau_alpha_sq": tau_a_sq,
        "tau_beta_sq": tau_b_sq,
        "alpha_pop": pop_alpha,
        "beta_pop": pop_beta,
        "rmse_pop_slope": rmse_pop,
        "arms": arm_data,
        "deltas_vs_canonical": deltas,
    }


def assign_verdict(out: dict[str, Any]) -> dict[str, Any]:
    """Pre-specified outcome classification.

    Decision-rule:
      * REJECTED-ONE-COMPONENT-DOES-IT-ALL — α-only OR β-only delta CI95
        upper bound ≥ -0.5pp (i.e., not strictly worse than canonical at 95%)
      * CONFIRMED-COMPOUND-NEEDED — both α-only AND β-only strictly worse
        (delta CI95 upper < -0.5pp) AND each component contributes ≥10% of
        canonical gain
      * PARTIAL-COMPOUND-NEEDED — both arms directionally worse but CIs
        overlap zero OR one component <10% of canonical
      * TENTATIVE-INCONCLUSIVE — fallback for ambiguous cases
    """
    canonical_gain = out["arms"]["canonical"]["gain_pct_vs_pop"]
    a_gain = out["arms"]["alpha_only"]["gain_pct_vs_pop"]
    b_gain = out["arms"]["beta_only"]["gain_pct_vs_pop"]
    a_delta_hi = out["deltas_vs_canonical"]["alpha_only"]["delta_ci95_hi"]
    b_delta_hi = out["deltas_vs_canonical"]["beta_only"]["delta_ci95_hi"]
    a_delta_mean = out["deltas_vs_canonical"]["alpha_only"]["delta_pct_vs_canonical"]
    b_delta_mean = out["deltas_vs_canonical"]["beta_only"]["delta_pct_vs_canonical"]

    a_frac = (a_gain / canonical_gain) if canonical_gain > 0 else 0.0
    b_frac = (b_gain / canonical_gain) if canonical_gain > 0 else 0.0
    a_contrib_frac = max(0.0, 1.0 - a_frac)  # fraction of gain from β-side
    b_contrib_frac = max(0.0, 1.0 - b_frac)  # fraction of gain from α-side

    # REJECTED: at least one ablation matches canonical within 0.5pp at 95% CI
    one_does_it_all = (
        a_delta_hi >= -DELTA_ONE_DOES_IT_PP or b_delta_hi >= -DELTA_ONE_DOES_IT_PP
    )

    if one_does_it_all:
        verdict = "REJECTED-ONE-COMPONENT-DOES-IT-ALL"
    elif (
        a_delta_hi < -DELTA_ONE_DOES_IT_PP
        and b_delta_hi < -DELTA_ONE_DOES_IT_PP
        and a_contrib_frac >= 0.10
        and b_contrib_frac >= 0.10
    ):
        verdict = "CONFIRMED-COMPOUND-NEEDED"
    elif a_delta_mean < 0 and b_delta_mean < 0:
        verdict = "PARTIAL-COMPOUND-NEEDED"
    else:
        verdict = "TENTATIVE-INCONCLUSIVE"

    return {
        "canonical_gain_pct": canonical_gain,
        "alpha_only_gain_pct": a_gain,
        "beta_only_gain_pct": b_gain,
        "alpha_fraction_of_canonical": float(a_frac),
        "beta_fraction_of_canonical": float(b_frac),
        "delta_alpha_only_vs_canonical_pp": a_delta_mean,
        "delta_beta_only_vs_canonical_pp": b_delta_mean,
        "verdict_class": verdict,
        "decision_rule": (
            "REJECTED if α-only OR β-only delta CI95 upper ≥ -0.5pp; "
            "CONFIRMED if both arms strictly worse (delta CI95 upper < -0.5pp) "
            "AND each component contributes ≥10% of canonical gain; "
            "PARTIAL if directionally consistent but CIs overlap zero OR one "
            "component <10%; TENTATIVE otherwise."
        ),
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.scored_parquet).dropna(subset=["score_user"]).reset_index(drop=True)
    print(f"[load] {len(df)} utts, {df.user_id.nunique()} users from {args.scored_parquet}")

    if args.smoke:
        keep = df.user_id.drop_duplicates().head(50).tolist()
        df = df[df.user_id.isin(keep)].reset_index(drop=True)
        k_folds = 2
        print(f"[smoke] {len(df)} utts, {df.user_id.nunique()} users")
    else:
        k_folds = args.k_folds

    print(f"[run] seed={args.seed} k_folds={k_folds}")
    out = run_ablation(df, args.seed, k_folds, args.min_obs_per_user)
    verdict = assign_verdict(out)

    summary = {
        "experiment": "W-B5_half_pebs_ablation",
        "protocol": "k=5 random-fold within-user CV; 3 arms (α-only, β-only, canonical)",
        "seed": args.seed,
        "k_folds": k_folds,
        "min_obs_per_user": args.min_obs_per_user,
        "n_bootstrap": N_BOOT,
        "scored_parquet": str(args.scored_parquet),
        "smoke": bool(args.smoke),
        **out,
        **verdict,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== W-B5 Half-PEBS ablation complete ===")
    print(f"  Verdict: {summary['verdict_class']}")
    for arm, d in summary["arms"].items():
        print(
            f"  {arm:>11}: gain_pct={d['gain_pct_vs_pop']:+.3f}% "
            f"CI=[{d['gain_pct_ci95_lo']:+.3f}, {d['gain_pct_ci95_hi']:+.3f}]"
        )
    for arm, d in summary["deltas_vs_canonical"].items():
        print(
            f"  Δ {arm:>11} vs canonical: {d['delta_pct_vs_canonical']:+.3f}pp "
            f"CI=[{d['delta_ci95_lo']:+.3f}, {d['delta_ci95_hi']:+.3f}]"
        )
    print(
        f"  α-fraction of canonical: {summary['alpha_fraction_of_canonical']:.3f} | "
        f"β-fraction: {summary['beta_fraction_of_canonical']:.3f}"
    )
    print(f"  Output: {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
