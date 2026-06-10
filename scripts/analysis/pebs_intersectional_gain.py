"""Intersectional PEBS gain: thin-user × non-English-native × non-male.

Combines the demographic-fairness analysis with the gain-vs-n_j-quintile
analysis. Addresses the concern that a marginal fairness analysis can hide
harm to intersectional subgroups.

Strategy: form the hardest subgroup for PEBS = small n_j AND minority
demographics, then verify gain CI still above zero.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RMSE_IN = (Path(__file__).resolve().parents[2] / "1_Causal_RLHF/results/track1_user_score_mse_shrunk.parquet")
DEMO_IN = (Path(__file__).resolve().parents[2] / "1_Causal_RLHF/data/prism_demographics.parquet")
OUT_DIR = (Path(__file__).resolve().parents[2] / "3_PEBS_Standalone/results/track1_intersectional_gain")

N_BOOT = 2000
RNG = 20260420


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG)

    rmse = pd.read_parquet(RMSE_IN)
    demo = pd.read_parquet(DEMO_IN)
    df = rmse.merge(demo, on="user_id", how="inner")
    df["gain"] = df["rmse_pop_slope"] - df["rmse_pebs_shrunk"]
    print(f"[merged] {len(df)} users overall  mean gain = {df['gain'].mean():+.3f}")

    # Build 4 intersectional cells in order of "difficulty" for PEBS
    cells = []

    # Cell 1: overall baseline
    cells.append(("ALL (baseline)", df))

    # Cell 2: thinnest quintile (n_j < 40)
    thin = df[df["n"] < 40]
    cells.append((f"thin n_j<40 (n={len(thin)})", thin))

    # Cell 3: thin ∩ non-English native
    thin_minority_eng = thin[thin["english_proficiency"] != "Native speaker"]
    cells.append((f"thin ∩ non-Native-English (n={len(thin_minority_eng)})", thin_minority_eng))

    # Cell 4: thin ∩ non-English-native ∩ non-male
    thin_minority_eng_nonmale = thin_minority_eng[thin_minority_eng["gender"] != "Male"]
    cells.append((f"thin ∩ non-Native-English ∩ non-male (n={len(thin_minority_eng_nonmale)})",
                  thin_minority_eng_nonmale))

    # Cell 5: thin ∩ non-English-native ∩ non-male ∩ <=secondary education
    thin_quad = thin_minority_eng_nonmale[
        thin_minority_eng_nonmale["education"].str.contains("Secondary|Vocational", na=False)
    ]
    cells.append((f"thin ∩ non-Native ∩ non-male ∩ non-university (n={len(thin_quad)})",
                  thin_quad))

    print("\n=== Intersectional gain cells (hardest PEBS subgroups) ===")
    results = []
    for label, sub in cells:
        if len(sub) < 10:
            print(f"{label}: SKIP (n<10)")
            continue
        g = sub["gain"].to_numpy(dtype=np.float64)
        n = len(g)
        point = float(g.mean())
        boots = np.array([rng.choice(g, size=n, replace=True).mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sig = "*" if lo > 0 or hi < 0 else ""
        print(f"  {label:60s} n={n:>4d}  gain = {point:+.3f}  CI [{lo:+.3f}, {hi:+.3f}] {sig}")
        results.append({
            "cell": label, "n_users": int(n), "mean_gain": point,
            "ci95_lo": float(lo), "ci95_hi": float(hi),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
        })

    n_excluding_zero = sum(r["ci_excludes_zero"] for r in results)
    all_positive = all(r["ci95_lo"] > 0 for r in results)
    print(f"\n[intersectional fairness] {n_excluding_zero}/{len(results)} cells have CI entirely above zero")
    print(f"[all_positive] {all_positive}")

    summary = {
        "cells": results,
        "n_cells": len(results),
        "n_ci_entirely_positive": int(n_excluding_zero),
        "all_ci_lo_above_zero": bool(all_positive),
        "n_bootstrap": N_BOOT,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
