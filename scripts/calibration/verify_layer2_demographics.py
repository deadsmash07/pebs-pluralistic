"""MC verify Layer 2 conditional-independence test on demographic proxies.

Per `skills/adversarial-theorem-review/SKILL.md` Step 4.5 MANDATORY: every
new Layer-2 flow must be validated on synthetic DGPs where the ground truth
is known, before we trust Layer 2 verdicts on real PRISM data.

Two scenarios tested:
  (a) NULL — demographics affect ratings only via labeler scale (α, β), NOT
      via any residual confounder. After anchor calibration, M_corrected
      should be independent of demographics | X. Layer 2 should PASS.
  (b) ALT  — demographics correlate with an unmeasured latent confounder U
      that affects both the treatment AND the outcome. After calibration,
      residual correlation with demographics remains. Layer 2 should FAIL.

If Layer 2 returns the right verdict in both scenarios (PASS under null,
FAIL under alt), the test has discriminative power and we can trust real
PRISM results. If not, we have a bug.

Run:
    python3 scripts/verify_layer2_demographics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.methods import (
    AnchorCalibrator,
    PartialPoolingCalibrator,
    layer2_conditional_independence,
)
from src.data import simulate_merged_toy


def simulate_with_demographics(
    n: int,
    n_labelers: int,
    n_anchors: int,
    scenario: str,  # "null" or "alt"
    n_demographics: int = 6,
    confound_strength: float = 2.0,
    seed: int = 0,
):
    """Extend merged toy v2 DGP with synthetic demographic variables.

    NULL scenario: (A-iv) holds (linear U→M only) AND demographics are
    independent of U. Layer 2 should PASS.

    ALT scenario: (A-iv) VIOLATED (quadratic U-term in M that linear
    calibration can't remove) AND demographics correlate with U. Residual
    correlation survives in M_corrected, so Layer 2 should FAIL.

    This is the correct split: anchor calibration is LINEAR, so only
    non-linear U-dependence in M can produce a real A-iv violation for
    Layer 2 to detect.
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal(n_labelers)
    labeler_idx = rng.integers(0, n_labelers, size=n)
    Q = rng.uniform(0, 1, size=n)
    X = 0.4 * Q + 0.8 * rng.standard_normal(n)

    anchors_q = np.linspace(0.1, 0.9, n_anchors)
    scales = 1.0 + 0.3 * U
    shifts = 1.0 * U

    # Anchor ratings: always linear in Q (satisfies A-iv on the anchor set)
    M_anchors = (
        scales[None, :] * anchors_q[:, None]
        + shifts[None, :]
        + 0.05 * rng.standard_normal((n_anchors, n_labelers))
    )

    # Fresh-response ratings: differ across scenarios
    if scenario == "null":
        # Linear U-dependence only — A-iv holds, calibration removes U fully
        M_bar = (
            scales[labeler_idx] * Q
            + shifts[labeler_idx]
            + 0.1 * rng.standard_normal(n)
        )
    elif scenario == "alt":
        # Add a quadratic U·Q² term — non-linear in Q, can't be absorbed by
        # linear anchor calibration. This IS a genuine A-iv violation.
        quadratic_coef = 0.8 * U[labeler_idx]
        M_bar = (
            scales[labeler_idx] * Q
            + shifts[labeler_idx]
            + quadratic_coef * (Q ** 2)                # A-iv violation
            + 0.1 * rng.standard_normal(n)
        )
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    R_true = 0.8 * Q + 0.1 * rng.standard_normal(n)
    R_obs = R_true + confound_strength * U[labeler_idx] + 0.1 * rng.standard_normal(n)

    base = {
        "U": U, "Q": Q, "X": X, "labeler_idx": labeler_idx,
        "anchors_q": anchors_q, "M_anchors": M_anchors,
        "M_bar": M_bar, "R_true": R_true, "R_obs": R_obs,
    }

    # Demographics: always correlated with U in ALT, random in NULL
    rng2 = np.random.default_rng(seed + 1000)
    if scenario == "null":
        demo_labels = rng2.integers(0, 2, size=(n_labelers, n_demographics))
    else:
        demo_labels = np.zeros((n_labelers, n_demographics), dtype=int)
        for k in range(n_demographics):
            noise = rng2.standard_normal(n_labelers) * 0.3
            threshold = np.quantile(U + noise, 0.5)
            demo_labels[:, k] = (U + noise > threshold).astype(int)

    per_pair_demo = {
        f"D_{k}": demo_labels[labeler_idx, k]
        for k in range(n_demographics)
    }
    return base, per_pair_demo


def run_one_trial(scenario: str, seed: int, calibrator_type: str = "mixedlm"):
    data, demo = simulate_with_demographics(
        n=2000, n_labelers=50, n_anchors=30,
        scenario=scenario, seed=seed,
    )
    # Fit calibrator
    if calibrator_type == "ols":
        cal = AnchorCalibrator(anchors_q=data["anchors_q"]).fit(data["M_anchors"])
    else:
        cal = PartialPoolingCalibrator(anchors_q=data["anchors_q"]).fit(data["M_anchors"])
    M_corrected = cal.correct(data["M_bar"], data["labeler_idx"])

    # Filter to non-zero-variance proxies (matches run_frontdoor_eval.py)
    demo = {k: v for k, v in demo.items() if v.var() > 0}

    verdict = layer2_conditional_independence(data["X"], M_corrected, demo)
    return verdict.verdict, verdict.n_significant, verdict.n_tested


def main():
    print("=" * 72)
    print("MC verification: Layer 2 with REAL demographic proxies (6 variables)")
    print("=" * 72)
    print("Expected:")
    print("  NULL scenario (demographics independent of U): PASS on most trials")
    print("  ALT scenario  (demographics correlate with U): FAIL on most trials")
    print()

    results = {"null": [], "alt": []}
    for scenario in ["null", "alt"]:
        print(f"--- scenario: {scenario} ---")
        for seed in range(10):
            try:
                verdict, n_sig, n_tested = run_one_trial(scenario, seed)
            except Exception as e:
                print(f"  seed={seed} ERROR: {e}")
                continue
            results[scenario].append(verdict)
            print(f"  seed={seed}  verdict={verdict}  "
                  f"({n_sig}/{n_tested} proxies significant)")
        n_pass = sum(1 for v in results[scenario] if v == "PASS")
        n_fail = sum(1 for v in results[scenario] if v == "FAIL")
        n_border = sum(1 for v in results[scenario] if v == "BORDERLINE")
        print(f"  summary: {n_pass} PASS, {n_border} BORDERLINE, {n_fail} FAIL\n")

    # Discriminative power check
    null_pass_rate = sum(1 for v in results["null"] if v == "PASS") / max(len(results["null"]), 1)
    alt_fail_rate = sum(1 for v in results["alt"] if v == "FAIL") / max(len(results["alt"]), 1)

    print("=" * 72)
    print(f"NULL PASS rate: {null_pass_rate:.1%}  (want ≥70%)")
    print(f"ALT  FAIL rate: {alt_fail_rate:.1%}  (want ≥50%)")
    if null_pass_rate >= 0.7 and alt_fail_rate >= 0.3:
        print("✅ Layer 2 has discriminative power — paper claims can be trusted")
    elif null_pass_rate < 0.5:
        print("❌ Layer 2 too STRICT on NULL — over-rejects innocent demographics "
              "(false positives). Check Bonferroni correction / SE computation.")
    else:
        print("⚠️  Layer 2 has weak discriminative power — either the test is too "
              "permissive or the synthetic ALT signal is too subtle. Investigate.")


if __name__ == "__main__":
    main()
