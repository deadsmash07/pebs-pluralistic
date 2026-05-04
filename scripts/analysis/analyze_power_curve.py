"""Fit a logistic TPR curve to the E1b power pilot + report MDE at 80% power.

Inputs
------
- `results/track3_e1b_power/power_curve.json` from `synthetic_drift_power_analysis.py`

Method
------
For TPR(β) monotone increasing in drift, fit a two-parameter logistic:
    TPR(β) = 1 / (1 + exp(-(a + b · log(β + ε))))
via scipy.optimize on the TPR values (treating each TPR as
Binomial(n_seeds, TPR) for the log-likelihood).

MDE at power π₀ solves TPR(β) = π₀ → β_{MDE}.

References
----------
- Cohen 1988 §11 — inverse-power calculation for MDE
- Wilson 1927 score interval for per-cell CI around TPR
- Gelman & Hill 2007 §21.4 — hierarchical power curves
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import optimize, stats


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval — more accurate than Wald for small n / extreme p."""
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    halfwidth = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - halfwidth), min(1.0, center + halfwidth)


def neg_binom_loglik(params: np.ndarray, xs: np.ndarray, ks: np.ndarray, ns: np.ndarray) -> float:
    """Binomial NLL for TPR(β) = sigmoid(a + b · log(β + eps))."""
    a, b = params
    eps = 1e-4
    logits = a + b * np.log(xs + eps)
    p = 1 / (1 + np.exp(-logits))
    # Clip for numerical safety
    p = np.clip(p, 1e-9, 1 - 1e-9)
    nll = -np.sum(ks * np.log(p) + (ns - ks) * np.log(1 - p))
    return float(nll)


def fit_logistic(drifts: np.ndarray, tprs: np.ndarray, n_seeds: np.ndarray):
    """Fit TPR(β) = sigmoid(a + b · log(β + eps)); return (a, b)."""
    ks = np.round(tprs * n_seeds).astype(int)
    # Drop β=0 row for log-space fit (undefined); it's the null anyway
    mask = drifts > 0
    x = drifts[mask].astype(float)
    k = ks[mask]
    n = n_seeds[mask]
    res = optimize.minimize(
        neg_binom_loglik,
        x0=np.array([0.0, 1.0]),
        args=(x, k, n),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000},
    )
    return float(res.x[0]), float(res.x[1]), float(res.fun)


def mde_from_logistic(a: float, b: float, target_power: float = 0.80) -> float:
    """Solve TPR(β) = π for β."""
    if b == 0:
        return float("nan")
    logit = math.log(target_power / (1 - target_power))
    eps = 1e-4
    return float(math.exp((logit - a) / b) - eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power-json",
                    default="results/track3_e1b_power/power_curve.json")
    ap.add_argument("--output",
                    default="results/track3_e1b_power/mde_analysis.json")
    ap.add_argument("--target-powers", default="0.50,0.80,0.95")
    args = ap.parse_args()

    data = json.loads(Path(args.power_json).read_text())
    curve = data["power_curve"]
    drifts = np.array([c["drift_per_month"] for c in curve])
    rejections = np.array([c["rejections"] for c in curve])
    n_seeds = np.array([c["n_seeds"] for c in curve])
    tprs = rejections / np.maximum(n_seeds, 1)

    # Per-cell Wilson 95% CI
    cell_cis = []
    for r, n, b in zip(rejections, n_seeds, drifts):
        lo, hi = wilson_ci(int(r), int(n))
        cell_cis.append({"drift_per_month": float(b),
                        "TPR": float(r / n),
                        "ci95": [float(lo), float(hi)],
                        "n_seeds": int(n)})

    # Type-I rate from β=0 row
    null_row = next((c for c in curve if c["drift_per_month"] == 0.0), None)
    alpha_empirical = (null_row["rejections"] / null_row["n_seeds"]) if null_row else None

    # Fit logistic
    a, b, nll = fit_logistic(drifts, tprs, n_seeds)

    target_powers = [float(p) for p in args.target_powers.split(",")]
    mdes = {f"power_{p:.2f}": mde_from_logistic(a, b, p) for p in target_powers}
    mdes_10mo = {k: 10 * v if not math.isnan(v) else v for k, v in mdes.items()}

    out = {
        "cohort_shape": data.get("cohort_shape", {}),
        "empirical_type_I_rate": alpha_empirical,
        "wilson_ci_per_cell": cell_cis,
        "logistic_fit": {
            "a": a,
            "b": b,
            "nll": nll,
            "functional_form": "TPR(beta) = sigmoid(a + b * log(beta + 1e-4))",
        },
        "mde_per_month": mdes,
        "mde_10_month_cumulative": mdes_10mo,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"[save] {args.output}")

    print("\n=== Pilot power analysis ===")
    print(f"Cohort: {data['cohort_shape']}")
    print(f"Empirical type-I rate (β=0): {alpha_empirical:.3f}  "
          f"[target: {data['test_config']['alpha']}]")
    print("\nPer-cell TPR with Wilson 95% CI:")
    for c in cell_cis:
        print(f"  β={c['drift_per_month']:.4f}/mo  "
              f"TPR={c['TPR']:.2%}  [{c['ci95'][0]:.2%}, {c['ci95'][1]:.2%}]")
    print(f"\nLogistic fit: TPR(β) = σ({a:+.3f} + {b:+.3f} · log(β+ε))")
    print(f"\n=== MDE at target powers ===")
    for p in target_powers:
        mde = mde_from_logistic(a, b, p)
        if not math.isnan(mde) and mde > 0:
            print(f"  power={p:.2f}:  β_MDE = {mde:.5f} pts/mo  "
                  f"(10-mo cumulative: {10*mde:.4f})")
        else:
            print(f"  power={p:.2f}:  MDE undefined (logistic didn't converge)")


if __name__ == "__main__":
    main()
