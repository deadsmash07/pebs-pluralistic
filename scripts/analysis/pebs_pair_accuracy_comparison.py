"""PEBS pair-accuracy on PRISM held-out slice — raw vs affine vs quadratic.

Reviewer gap: concurrent work (LoRe 71%, PReF, etc.) reports pair-accuracy on
PRISM, while PEBS papers lead with RMSE reduction. This script closes the gap
by computing pair-accuracy under the SAME three calibrators already fit in
results/track1_quadratic_calibrator/per_pair.parquet:

  1. raw 7B Qwen-Instruct RM scores (no calibration)
  2. PEBS-affine per-user (α_j, β_j) calibration
  3. PEBS-quadratic per-user (α_j, β_j, γ_j) calibration

With cluster-bootstrap CIs (resample users w/ replacement, 2000 reps).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

IN = (Path(__file__).resolve().parents[2] / "1_Causal_RLHF/results/track1_quadratic_calibrator/per_pair.parquet")
OUT_DIR = (Path(__file__).resolve().parents[2] / "3_PEBS_Standalone/results/track1_pair_accuracy_comparison")

N_BOOT = 2000
RNG = 20260420


def pair_acc(margin: np.ndarray) -> float:
    valid = np.isfinite(margin)
    return float(np.mean(margin[valid] > 0))


def cluster_bootstrap_pair_acc(df: pd.DataFrame, col: str, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    users = df["user_id"].to_numpy()
    uniq = np.unique(users)
    groups = {u: np.where(users == u)[0] for u in uniq}
    margin = df[col].to_numpy(dtype=np.float64)
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([groups[u] for u in sampled])
        m = margin[rows]
        valid = np.isfinite(m)
        out[b] = float(np.mean(m[valid] > 0))
    return out


def paired_delta_ci(df: pd.DataFrame, col_a: str, col_b: str, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Bootstrap paired Δ (pair_acc[col_a] − pair_acc[col_b]) on user clusters."""
    users = df["user_id"].to_numpy()
    uniq = np.unique(users)
    groups = {u: np.where(users == u)[0] for u in uniq}
    m_a = df[col_a].to_numpy(dtype=np.float64)
    m_b = df[col_b].to_numpy(dtype=np.float64)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([groups[u] for u in sampled])
        ma, mb = m_a[rows], m_b[rows]
        va, vb = np.isfinite(ma), np.isfinite(mb)
        deltas[b] = float(np.mean(ma[va] > 0) - np.mean(mb[vb] > 0))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    point = float(np.mean(deltas))
    return point, float(lo), float(hi)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG)
    df = pd.read_parquet(IN)
    print(f"[load] {len(df)} pairs, {df['user_id'].nunique()} users")

    estimators = [
        ("raw_7B",         "margin_raw"),
        ("pebs_affine",   "margin_affine"),
        ("pebs_quadratic","margin_quadratic"),
    ]

    results = {}
    for name, col in estimators:
        point = pair_acc(df[col].to_numpy(dtype=np.float64))
        boots = cluster_bootstrap_pair_acc(df, col, N_BOOT, np.random.default_rng(RNG))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        results[name] = {
            "pair_accuracy": point,
            "ci95": [float(lo), float(hi)],
            "n_pairs": int(df[col].notna().sum()),
        }
        print(f"  {name:18s}  acc = {point:.4f}  95% CI = [{lo:.4f}, {hi:.4f}]")

    print("\nPaired Δ (bootstrap resamples users):")
    delta_aff_raw  = paired_delta_ci(df, "margin_affine",    "margin_raw", N_BOOT, np.random.default_rng(RNG))
    delta_quad_raw = paired_delta_ci(df, "margin_quadratic", "margin_raw", N_BOOT, np.random.default_rng(RNG))
    delta_quad_aff = paired_delta_ci(df, "margin_quadratic", "margin_affine", N_BOOT, np.random.default_rng(RNG))
    for label, t in [
        ("affine − raw",       delta_aff_raw),
        ("quadratic − raw",    delta_quad_raw),
        ("quadratic − affine", delta_quad_aff),
    ]:
        d, lo, hi = t
        sig = "*" if lo > 0 or hi < 0 else ""
        print(f"  Δ {label:20s} = {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {sig}")

    summary = {
        "n_pairs": int(len(df)),
        "n_users": int(df["user_id"].nunique()),
        "n_bootstrap": N_BOOT,
        "rng_seed": RNG,
        "estimators": results,
        "paired_deltas": {
            "affine_minus_raw":       {"delta": delta_aff_raw[0],  "ci95": [delta_aff_raw[1],  delta_aff_raw[2]]},
            "quadratic_minus_raw":    {"delta": delta_quad_raw[0], "ci95": [delta_quad_raw[1], delta_quad_raw[2]]},
            "quadratic_minus_affine": {"delta": delta_quad_aff[0], "ci95": [delta_quad_aff[1], delta_quad_aff[2]]},
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
