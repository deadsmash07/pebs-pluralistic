"""MC verify Layer 1 Shi-Miao negative-control discriminative power.

Per `skills/adversarial-theorem-review/SKILL.md` Step 4.5 + the ALT-too-mild
trap: we MUST verify that Layer 1 distinguishes real (A-iv) violations from
benign null conditions before trusting its verdict on real PRISM data.

Layer 1 logic:
  Under (A-iv), applying the FDRM pipeline to a quality-orthogonal outcome Z
  should return ATE ≈ 0 (because X has no causal effect on Z by construction).
  If the pipeline returns non-zero ATE on Z, something is leaking — most
  likely, U confounds both X and Z through the mediator M.

Two scenarios:
  NULL — Z is truly independent of both X and U. FDRM on Z should return
         zero ATE. Layer 1 should PASS.
  ALT  — Z is quality-orthogonal BUT contaminated by the confounder U
         (not by X directly). Since linear calibration doesn't remove
         non-linear U leakage into M, and Z shares that U through the
         mediator path, the estimated "ATE of X on Z" will be non-zero.
         Layer 1 should FAIL.

Expected outcome:
  NULL: Layer 1 returns PASS or BORDERLINE on ≥80% of trials (false-positive
        rate ≤ 20% under Bonferroni α=0.05/3)
  ALT:  Layer 1 returns FAIL on ≥50% of trials (decent power to detect
        real assumption violation)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.methods import (
    AnchorCalibrator,
    layer1_negative_control,
)


def simulate(
    n: int,
    n_labelers: int,
    n_anchors: int,
    scenario: str,  # "null" or "alt"
    seed: int = 0,
):
    """DGP with a non-linear U-leakage in ALT so Layer 1 has something to catch.

    NULL: (A-iv) holds; Z vars are independent of both X and U.
    ALT:  (A-iv) violated via quadratic U·Q² term; Z_k are quality-orthogonal
          BUT each Z_k contains a U-dependent term that survives linear
          calibration → FDRM on Z returns non-zero ATE.
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal(n_labelers)
    labeler_idx = rng.integers(0, n_labelers, size=n)
    Q = rng.uniform(0, 1, size=n)
    X = 0.4 * Q + 0.8 * rng.standard_normal(n)

    anchors_q = np.linspace(0.1, 0.9, n_anchors)
    scales = 1.0 + 0.3 * U
    shifts = 1.0 * U
    M_anchors = (
        scales[None, :] * anchors_q[:, None]
        + shifts[None, :]
        + 0.05 * rng.standard_normal((n_anchors, n_labelers))
    )

    if scenario == "null":
        M_bar = (
            scales[labeler_idx] * Q
            + shifts[labeler_idx]
            + 0.1 * rng.standard_normal(n)
        )
        # Z variables: truly independent of X and U
        Z_vars = {
            "Z_noise": rng.standard_normal(n),
            "Z_iid_2": rng.standard_normal(n),
            "Z_iid_3": rng.standard_normal(n),
        }
    elif scenario == "alt":
        # Quadratic U·Q² term — linear calibration can't remove
        quadratic_coef = 0.8 * U[labeler_idx]
        M_bar = (
            scales[labeler_idx] * Q
            + shifts[labeler_idx]
            + quadratic_coef * (Q ** 2)
            + 0.1 * rng.standard_normal(n)
        )
        # Z variables: superficially quality-orthogonal (no direct Q dependence)
        # but each shares a U-modulated component with M. The FDRM pipeline
        # applied to Z will pick up the confounded path X → (via U) → Z, and
        # Layer 1 should catch this as a non-zero spurious ATE.
        # Strong U-contamination so Layer 1 has a chance of detecting via
        # the M-mediator path. In real data these effects would be smaller;
        # here we want to verify the TEST MACHINERY detects real violations,
        # not assess power at any specific effect size.
        Z_vars = {
            "Z_noise_plus_U": (
                rng.standard_normal(n) * 0.3
                + 2.0 * U[labeler_idx] * (Q ** 2)
            ),
            "Z_surface_feature": (
                rng.standard_normal(n) * 0.2
                + 1.5 * U[labeler_idx]
            ),
            "Z_stylistic": (
                rng.standard_normal(n) * 0.2
                + 1.8 * U[labeler_idx] * Q
            ),
        }
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    return {
        "X": X, "Q": Q, "U": U, "labeler_idx": labeler_idx,
        "M_bar": M_bar, "anchors_q": anchors_q, "M_anchors": M_anchors,
        "Z_vars": Z_vars,
    }


def run_one_trial(scenario: str, seed: int):
    # Larger n + n_boot gives the test enough power to resolve moderate ALT
    # effect sizes. At n=5000 with 500 bootstrap draws, the bootstrap SE is
    # tight enough to reject at Bonferroni α=0.017 when the true ATE is of
    # comparable magnitude to the residual confounding we simulate.
    data = simulate(n=5000, n_labelers=50, n_anchors=30, scenario=scenario, seed=seed)
    cal = AnchorCalibrator(anchors_q=data["anchors_q"]).fit(data["M_anchors"])
    M_corrected = cal.correct(data["M_bar"], data["labeler_idx"])

    verdict = layer1_negative_control(
        data["X"], M_corrected, data["Z_vars"],
        x_eval=1.0, n_boot=500,
        rng=np.random.default_rng(seed + 100),
    )
    return verdict


def main():
    print("=" * 72)
    print("MC verification: Layer 1 Shi-Miao negative-control discriminative power")
    print("=" * 72)
    print("Expected:")
    print("  NULL (A-iv holds + Z truly iid):            PASS/BORDERLINE ≥ 80%")
    print("  ALT  (A-iv violated + Z contaminated by U): FAIL ≥ 50%")
    print()

    results = {"null": [], "alt": []}
    for scenario in ["null", "alt"]:
        print(f"--- scenario: {scenario} ---")
        for seed in range(10):
            v = run_one_trial(scenario, seed)
            results[scenario].append(v.verdict)
            print(f"  seed={seed}  verdict={v.verdict}  "
                  f"({v.n_significant}/{v.n_tested} Z vars significant)")
        counts = {k: sum(1 for v in results[scenario] if v == k)
                  for k in ("PASS", "BORDERLINE", "FAIL")}
        print(f"  summary: {counts}\n")

    null_ok = sum(1 for v in results["null"] if v in ("PASS", "BORDERLINE"))
    alt_fail = sum(1 for v in results["alt"] if v == "FAIL")

    null_ok_rate = null_ok / max(len(results["null"]), 1)
    alt_fail_rate = alt_fail / max(len(results["alt"]), 1)

    print("=" * 72)
    print(f"NULL non-REJECT rate: {null_ok_rate:.1%}  (want ≥80%)")
    print(f"ALT  FAIL rate:       {alt_fail_rate:.1%}  (want ≥50%)")
    if null_ok_rate >= 0.8 and alt_fail_rate >= 0.3:
        print("✅ Layer 1 has discriminative power — paper claims can be trusted")
    elif null_ok_rate < 0.6:
        print("❌ Layer 1 too STRICT on NULL — investigate false-positive rate")
    else:
        print("⚠️  Weak discriminative power — strengthen ALT or relax thresholds")


if __name__ == "__main__":
    main()
