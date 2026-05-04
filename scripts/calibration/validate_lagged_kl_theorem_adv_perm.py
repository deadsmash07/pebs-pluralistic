#!/usr/bin/env python3
"""iter+N+279 — Adversarial permutation null for T2 lagged-KL empirical validation.

ADVERSARIAL ATTACK (iter+N+278 subagent a5f098d8):
  iter+N+269 reports Ĉ=2.05e-4, 95% paired-bootstrap CI [6.4e-5, 5.1e-4]
  excludes zero on n=4 seeds. But:
    - |Δ_BT-LL| is non-negative by construction (absolute value)
    - ε̄_T = λ₀|1-β| is non-negative (|1-β| is absolute)
    - Regressing ≥0 on ≥0 through origin always produces a positive slope
      and positive CI — "CI excludes zero" is a measure-zero trivial fact
      of the degenerate design, not a test of Theorem T2.

CLOSURE:
  (a) Permutation null: randomly permute (ε̄_T, |Δ_BT-LL|) pairings across
      the 4 seeds; for each permutation, re-run the paired-bootstrap 95%
      CI; count fraction where CI excludes zero. If this fraction is ≥50%,
      the "CI excludes zero" claim is a design artifact.
  (b) Signed test: compute the SIGNED Δ_BT-LL (not |·|); regress signed
      against SIGNED ε(t) = λ₀(1-β). Since β>1 in all 4 seeds, ε is strictly
      negative; if T2 is really about invariance-with-rate, signed Δ should
      track signed ε with a proper (non-trivial) slope.
  (c) R² diagnostic: confirm R²=0.38; ~62% unexplained variance may indicate
      T2's bound is loose but directionally correct.

CPU-only. Runtime <30s.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats


def fit_no_intercept(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return {"slope": float("nan"), "r2": float("nan"), "n": n}
    slope = float(np.dot(x, y) / denom)
    resid = y - slope * x
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y, y))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": slope, "r2": r2, "n": n}


def bootstrap_slope_ci(x: np.ndarray, y: np.ndarray,
                       n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(x)
    slopes = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        denom = float(np.dot(x[idx], x[idx]))
        slopes[b] = float(np.dot(x[idx], y[idx]) / denom) if denom > 0 else np.nan
    slopes = slopes[np.isfinite(slopes)]
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "n_boot_eff": int(len(slopes))}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per-seed-json", required=True,
                   help="Path to per_seed_integrals.json from iter+N+269.")
    p.add_argument("--output-json", required=True)
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rows = json.loads(Path(args.per_seed_json).read_text())
    seeds = [r["seed"] for r in rows]
    betas = np.array([r["beta"] for r in rows])
    deltas_signed = np.array([r["delta_bt_ll"] for r in rows])
    deltas_abs = np.array([r["delta_bt_ll_abs"] for r in rows])
    integrals_abs = np.array([r["integral_avg"] for r in rows])
    lam0 = float(rows[0]["lambda_0"])
    integrals_signed = lam0 * (1.0 - betas)  # signed (all negative since beta>1)

    n = len(rows)
    print(f"[data] n={n} seeds: {seeds}")
    print(f"  beta          = {betas}")
    print(f"  integral (abs)= {integrals_abs}")
    print(f"  integral (sig)= {integrals_signed}  (all negative since beta>1)")
    print(f"  delta (signed)= {deltas_signed}")
    print(f"  delta (abs)   = {deltas_abs}")

    # --- Observed fit (replicates iter+N+269) ---
    obs_fit = fit_no_intercept(integrals_abs, deltas_abs)
    obs_ci = bootstrap_slope_ci(integrals_abs, deltas_abs,
                                 n_boot=args.n_boot, seed=args.seed)
    obs_excludes_zero = bool(obs_ci["lo"] > 0.0 or obs_ci["hi"] < 0.0)
    print(f"\n[obs] C_hat={obs_fit['slope']:.4e}  "
          f"R2={obs_fit['r2']:.4f}  "
          f"CI=[{obs_ci['lo']:.4e}, {obs_ci['hi']:.4e}]  "
          f"excludes_0={obs_excludes_zero}")

    # --- TEST (a): permutation null ---
    # For each permutation of y (deltas_abs) against x (integrals_abs),
    # recompute bootstrap CI and check exclusion of zero.
    all_perms = list(itertools.permutations(range(n)))
    # Exclude the identity permutation only when checking "how many permutations
    # produce CI excluding 0" — we want the rate among arbitrary pairings.
    perm_results = []
    n_excl = 0
    slopes_perm = []
    for pidx, perm in enumerate(all_perms):
        y_perm = deltas_abs[list(perm)]
        fit = fit_no_intercept(integrals_abs, y_perm)
        ci = bootstrap_slope_ci(integrals_abs, y_perm,
                                 n_boot=2000,  # 2000 per perm; 24 perms = 48k
                                 seed=args.seed + pidx)
        excludes = bool(ci["lo"] > 0.0 or ci["hi"] < 0.0)
        perm_results.append({
            "perm": list(perm),
            "slope": fit["slope"],
            "ci_lo": ci["lo"],
            "ci_hi": ci["hi"],
            "excludes_zero": excludes,
            "is_identity": list(perm) == list(range(n)),
        })
        slopes_perm.append(fit["slope"])
        if excludes:
            n_excl += 1
    n_total_perm = len(all_perms)  # 24 for n=4
    frac_excl = n_excl / n_total_perm
    print(f"\n[(a) permutation null] n={n_total_perm} perms tested (enumerated)")
    print(f"  fraction with CI excluding zero: {frac_excl*100:.1f}% "
          f"({n_excl}/{n_total_perm})")
    print(f"  (if >= 50%, 'CI excludes zero' is a degenerate-design artifact)")

    # Permutation slope range
    slopes_perm = np.array(slopes_perm)
    obs_rank = int((slopes_perm >= obs_fit["slope"]).sum())
    p_perm = (obs_rank + 1) / (n_total_perm + 1)
    print(f"  permutation-test p (obs slope rank): {p_perm:.4f}")

    # --- TEST (b): signed correlation ---
    # Signed epsilon = lambda_0 * (1 - beta). Signed delta = delta_bt_ll.
    # Under T2 in its tight form, signed delta should track signed eps
    # (both negative, so positive correlation).
    pearson_r, pearson_p = stats.pearsonr(integrals_signed, deltas_signed)
    spearman_rho, spearman_p = stats.spearmanr(integrals_signed, deltas_signed)
    # Also fit linear signed-on-signed
    sfit_slope, sfit_intercept = np.polyfit(integrals_signed, deltas_signed, 1)
    print(f"\n[(b) signed test]")
    print(f"  Pearson r(signed_eps, signed_delta)  = {pearson_r:+.4f}  p={pearson_p:.4f}")
    print(f"  Spearman rho(signed_eps, signed_delta)= {spearman_rho:+.4f}  p={spearman_p:.4f}")
    print(f"  Signed-on-signed OLS: slope={sfit_slope:+.4e}  intercept={sfit_intercept:+.4e}")
    signed_consistent = bool(pearson_r > 0)  # directionally consistent
    print(f"  directionally consistent with T2 (r>0): {signed_consistent}")

    # --- TEST (c): R^2 ---
    r2 = obs_fit["r2"]
    r2_discussion = (
        f"R^2_through_origin = {r2:.3f}; {100*(1-r2):.0f}% of |Delta| "
        "variance is unexplained by the |1-beta| signal. Consistent with "
        "T2 being a loose upper bound rather than a tight identity."
    )
    print(f"\n[(c) R^2 diagnostic] {r2_discussion}")

    # --- Verdict ---
    if frac_excl >= 0.50:
        verdict = "STRIKE_CI_EXCLUDES_ZERO"
        verdict_note = (
            f"'CI excludes zero' holds in {frac_excl*100:.0f}% of arbitrary "
            f"pairings — it is a degenerate-design artifact of regressing "
            f"|delta|>=0 on |integral|>=0 through origin. Replace with "
            f"'directionally consistent with T2' (signed test) + R^2 discussion."
        )
    elif frac_excl < 0.25:
        verdict = "KEEP_CI_LANGUAGE_WITH_FOOTNOTE"
        verdict_note = (
            f"Only {frac_excl*100:.0f}% of permutations produce CI-exclusion; "
            f"the observed claim has discriminative power. Keep language but "
            f"add footnote re: non-negative design."
        )
    else:
        verdict = "DOWNGRADE_TO_DIRECTIONAL"
        verdict_note = (
            f"{frac_excl*100:.0f}% of permutations produce CI-exclusion; "
            f"in the ambiguous middle — downgrade to 'directionally consistent' "
            f"with CI as a secondary diagnostic."
        )

    out = {
        "iter": "iter+N+279_adv_lagged_kl_perm",
        "attack": "NEW-2 T2 lagged-KL CI-excludes-zero is measure-zero trivial",
        "n_seeds": n,
        "seeds": seeds,
        "inputs": {
            "betas": betas.tolist(),
            "integrals_abs": integrals_abs.tolist(),
            "integrals_signed": integrals_signed.tolist(),
            "deltas_abs": deltas_abs.tolist(),
            "deltas_signed": deltas_signed.tolist(),
        },
        "observed_fit": {
            "C_hat": obs_fit["slope"],
            "R2": obs_fit["r2"],
            "ci_lo": obs_ci["lo"],
            "ci_hi": obs_ci["hi"],
            "excludes_zero": obs_excludes_zero,
        },
        "test_a_permutation_null": {
            "n_perms": n_total_perm,
            "n_perms_with_ci_excluding_zero": n_excl,
            "frac_ci_excludes_zero": frac_excl,
            "permutation_test_p": float(p_perm),
            "per_perm": perm_results,
        },
        "test_b_signed": {
            "pearson_r_signed": float(pearson_r),
            "pearson_p_signed": float(pearson_p),
            "spearman_rho_signed": float(spearman_rho),
            "spearman_p_signed": float(spearman_p),
            "signed_ols_slope": float(sfit_slope),
            "signed_ols_intercept": float(sfit_intercept),
            "directionally_consistent_r_positive": signed_consistent,
        },
        "test_c_r2_diagnostic": {
            "R2_through_origin": r2,
            "discussion": r2_discussion,
        },
        "verdict": verdict,
        "verdict_note": verdict_note,
    }
    outpath = Path(args.output_json)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\n=== SUMMARY ===")
    print(f"  (a) CI-exclusion rate : {frac_excl*100:.1f}% ({n_excl}/{n_total_perm})")
    print(f"  (a) perm-test p       : {p_perm:.4f}")
    print(f"  (b) pearson r (signed): {pearson_r:+.4f}  p={pearson_p:.4f}")
    print(f"  (c) R^2               : {r2:.3f}")
    print(f"  VERDICT               : {verdict}")
    print(f"  NOTE                  : {verdict_note}")
    print(f"\n[save] {outpath}")


if __name__ == "__main__":
    main()
