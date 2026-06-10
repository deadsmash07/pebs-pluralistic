"""PEBS gain stratified by PRISM demographics — fairness/robustness check.

Reviewer question: does the 8.58% PRISM gain concentrate in particular
demographic subgroups (age, gender, education, English proficiency,
study locale)? Answer: compute per-demographic-bin mean gain with
cluster-bootstrap CI.

This is BOTH a robustness check (PEBS should help users roughly
uniformly across demographics if it's capturing heterogeneity faithfully)
AND a fairness audit (it must not disproportionately harm any group).

Outputs: results/track1_gain_by_demographic/summary.json + plots
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RMSE_IN = (Path(__file__).resolve().parents[2] / "1_Causal_RLHF/results/track1_user_score_mse_shrunk.parquet")
DEMO_IN = (Path(__file__).resolve().parents[2] / "1_Causal_RLHF/data/prism_demographics.parquet")
OUT_DIR = (Path(__file__).resolve().parents[2] / "3_PEBS_Standalone/results/track1_gain_by_demographic")

N_BOOT = 2000
RNG = 20260420

DEMOGRAPHICS = [
    ("age", None),
    ("gender", None),
    ("education", None),
    ("english_proficiency", None),
    ("study_locale", None),
    ("employment_status", None),
]


def bin_age(s: pd.Series) -> pd.Series:
    return s.fillna("Unspecified").astype(str)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG)

    rmse = pd.read_parquet(RMSE_IN)
    demo = pd.read_parquet(DEMO_IN)

    df = rmse.merge(demo, on="user_id", how="inner")
    df["gain"] = df["rmse_pop_slope"] - df["rmse_pebs_shrunk"]
    print(f"[merged] {len(df)} users with both RMSE + demographics")
    print(f"[overall] mean gain = {df['gain'].mean():+.4f}")

    # Pre-process age -> bins
    df["age_bin"] = bin_age(df["age"])
    df["gender"] = df["gender"].fillna("Unspecified")
    df["education"] = df["education"].fillna("Unspecified")
    df["english_proficiency"] = df["english_proficiency"].fillna("Unspecified")
    df["study_locale"] = df["study_locale"].fillna("Unspecified")
    df["employment_status"] = df["employment_status"].fillna("Unspecified")

    # Compute per-bin mean + cluster bootstrap CI
    def bootstrap_mean_ci(values: np.ndarray, n_boot: int, rng_) -> tuple[float, float]:
        n = len(values)
        if n < 3:
            return float("nan"), float("nan")
        ms = np.array([rng_.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
        lo, hi = np.percentile(ms, [2.5, 97.5])
        return float(lo), float(hi)

    results = {}
    summary_rows = []

    columns_to_scan = [
        ("age_bin", "Age"),
        ("gender", "Gender"),
        ("education", "Education"),
        ("english_proficiency", "English proficiency"),
        ("study_locale", "Study locale"),
        ("employment_status", "Employment status"),
    ]

    for col, label in columns_to_scan:
        bin_results = []
        rng_col = np.random.default_rng(RNG + hash(col) % 10000)
        for val, g in df.groupby(col, dropna=False, observed=True):
            if len(g) < 10:
                continue
            vals = g["gain"].to_numpy(dtype=np.float64)
            mean = float(vals.mean())
            ci_lo, ci_hi = bootstrap_mean_ci(vals, N_BOOT, rng_col)
            bin_results.append({
                "value": str(val),
                "n_users": int(len(g)),
                "mean_gain": mean,
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
            })
        results[col] = bin_results
        print(f"\n[{label}]")
        for r in bin_results:
            print(f"  {r['value']:20s}  n={r['n_users']:4d}  gain = {r['mean_gain']:+.3f}  "
                  f"CI95 = [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]")
            summary_rows.append({"attribute": label, **r})

    # Figure: forest plot of mean gain ± CI for every subgroup
    fig, axes = plt.subplots(len(columns_to_scan), 1, figsize=(7, 2 + 0.45 * sum(len(results[c]) for c, _ in columns_to_scan)), sharex=True)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax, (col, label) in zip(axes, columns_to_scan):
        rs = results[col]
        if not rs:
            ax.set_title(f"{label}: no bins with n≥10")
            continue
        ypos = np.arange(len(rs))[::-1]
        means = np.array([r["mean_gain"] for r in rs])
        los = np.array([r["ci95_lo"] for r in rs])
        his = np.array([r["ci95_hi"] for r in rs])
        labels = [f"{r['value']}  (n={r['n_users']})" for r in rs]
        ax.errorbar(means, ypos, xerr=[means - los, his - means], fmt="o", capsize=3, color="C0")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.axvline(df["gain"].mean(), color="C3", lw=0.7, ls=":", label="overall mean")
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xlabel("PEBS gain (RMSE units)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "forest_plot.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "forest_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Overall summary stats
    overall_ci = bootstrap_mean_ci(df["gain"].to_numpy(dtype=np.float64), N_BOOT, np.random.default_rng(RNG))
    summary = {
        "n_users_analysed": int(len(df)),
        "overall_mean_gain": float(df["gain"].mean()),
        "overall_ci95": list(overall_ci),
        "n_bootstrap": N_BOOT,
        "by_attribute": results,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "forest_table.csv", index=False)

    # Check if ALL bins have CIs that exclude zero (fairness check)
    all_positive = True
    harmful_bins = []
    for col, _ in columns_to_scan:
        for r in results[col]:
            if r["ci95_hi"] < 0:
                harmful_bins.append(r)
            if r["ci95_lo"] <= 0:
                all_positive = False
    print(f"\n[fairness] all subgroup CIs entirely above zero: {all_positive}")
    print(f"[fairness] subgroups with CI entirely BELOW zero (PEBS actively harms): "
          f"{len(harmful_bins)}  {harmful_bins if harmful_bins else '(none)'}")
    print(f"\nWrote {OUT_DIR}/summary.json + forest_plot.{{pdf,png}} + forest_table.csv")


if __name__ == "__main__":
    main()
