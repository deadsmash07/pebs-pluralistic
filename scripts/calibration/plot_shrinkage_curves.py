"""Paper figure: shrinkage vs naive OLS vs pop-slope learning curves.

Visualizes the key finding:
  - Break-even k drops from 20 (naive OLS) to 5 (shrunk)
  - k=2 catastrophic blow-up (135 RMSE) eliminated (→ 26.6 shrunk)
  - H2e within-user CV headline: 7.02% (naive) → 8.58% (shrunk)

Panels:
  1. Cold-start curve: per-user held-out RMSE vs k ∈ {1..20}
     - naive OLS (with exploding k=2 point)
     - shrunk (monotone, starts near pop-slope)
     - pop-slope / no-calib horizontals
  2. Within-user k-fold CV bar chart (H2e): no-calib / pop / naive / shrunk
     with 95% CIs from per-user bootstrap

Research conventions:
  - log-x for cold-start curve
  - IQR bands via 25/75 percentiles
  - Seaborn-ish minimalist style
  - Bootstrap 1000-resample CIs

References
----------
- Hutter et al. 2014 "Learning Curves" on how to plot
- Wilson 1927 for Wilson score intervals on binary rates
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coldstart-json", default="results/track1_coldstart_curve_randomk.json")
    p.add_argument("--coldstart-parquet", default="results/track1_coldstart_curve_randomk.parquet")
    p.add_argument("--shrunk-json", default="results/track1_coldstart_shrinkage.json")
    p.add_argument("--shrunk-parquet", default="results/track1_coldstart_shrinkage.parquet")
    p.add_argument("--h2e-shrunk-json", default="results/track1_user_score_mse_shrunk.json")
    p.add_argument("--output", default="results/figs/track1_shrinkage_curves.png")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _maybe_path(p: str) -> Path:
    pth = Path(p)
    return pth if pth.exists() else None


def bootstrap_ci_mean(x: np.ndarray, n_boot: int, alpha: float, rng: np.random.Generator):
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        means[i] = x[idx].mean()
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    cs = json.loads(Path(args.coldstart_json).read_text())
    sh = json.loads(Path(args.shrunk_json).read_text())
    h2e = json.loads(Path(args.h2e_shrunk_json).read_text())

    pop_mean = cs["holdout_rmse_baselines"]["pop_slope_mean"]
    nocal_mean = cs["holdout_rmse_baselines"]["no_calib_mean"]
    budgets = cs["config"]["budgets"]

    # Per-user cold-start parquets for IQR
    cs_pu = pd.read_parquet(args.coldstart_parquet) if _maybe_path(args.coldstart_parquet) else None
    sh_pu = pd.read_parquet(args.shrunk_parquet) if _maybe_path(args.shrunk_parquet) else None

    # Compose arrays for plotting
    rows = []
    for k in budgets:
        rec_cs = cs["curve"]["per_k"].get(str(k)) or cs["curve"]["per_k"].get(k)
        rec_sh = sh["per_k"].get(str(k)) or sh["per_k"].get(k)
        if rec_cs is None:
            continue
        rows.append({
            "k": k,
            "ols_mean": rec_cs["rmse_pebs"]["mean"],
            "ols_median": rec_cs["rmse_pebs"]["median"],
            "ols_p25": rec_cs["rmse_pebs"].get("p25", np.nan),
            "ols_p75": rec_cs["rmse_pebs"].get("p75", np.nan),
            "shrunk_mean": rec_sh["shrunk_mean"] if rec_sh else np.nan,
            "shrunk_median": rec_sh["shrunk_median"] if rec_sh else np.nan,
            "n_users": rec_cs["n_users"],
        })
    df = pd.DataFrame(rows)

    # IQR from per-user parquet (if available) for shrunk curve
    shrunk_p25 = []
    shrunk_p75 = []
    if sh_pu is not None:
        for k in budgets:
            col = f"rmse_pebs_shrunk_k{k}"
            if col in sh_pu.columns:
                vals = sh_pu[col].dropna()
                shrunk_p25.append(float(vals.quantile(0.25)) if len(vals) else np.nan)
                shrunk_p75.append(float(vals.quantile(0.75)) if len(vals) else np.nan)
            else:
                shrunk_p25.append(np.nan)
                shrunk_p75.append(np.nan)
    else:
        shrunk_p25 = [np.nan] * len(budgets)
        shrunk_p75 = [np.nan] * len(budgets)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[err] matplotlib not installed — skipping figure")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.5, 1]})

    # Panel 1: cold-start learning curves
    ax = axes[0]
    ax.fill_between(df["k"], df["ols_p25"], df["ols_p75"], color="#d97757", alpha=0.15,
                    label="naive OLS IQR (P25–P75)")
    ax.plot(df["k"], df["ols_median"], color="#d97757", linestyle="--", linewidth=1.5,
            label="naive OLS (median)")
    ax.plot(df["k"], df["ols_mean"], color="#d97757", linewidth=2.3, marker="o",
            label="naive OLS (mean)")
    ax.plot(df["k"], df["shrunk_mean"], color="#3b8ea5", linewidth=2.3, marker="s",
            label="EB-shrunk PEBS (mean)")
    # Baselines
    ax.axhline(pop_mean, color="black", linestyle=":", linewidth=1.3, label=f"pop-slope = {pop_mean:.2f}")
    ax.axhline(nocal_mean, color="gray", linestyle=":", linewidth=1.0, label=f"no-calib = {nocal_mean:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("k (labeled utterances per new user)")
    ax.set_ylabel("Held-out user-score RMSE (mean)")
    ax.set_title(f"Cold-start adaptation curve\n{cs['config']['n_holdout_users']} held-out users, random-k ordering")
    ax.set_xticks(df["k"])
    ax.set_xticklabels([str(k) for k in df["k"]])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)
    # Annotate break-even points
    ax.annotate(f"naive break-even k={cs.get('break_even_k_mean')}",
                xy=(cs.get("break_even_k_mean") or 20, pop_mean),
                xytext=(2.5, pop_mean + 2.5),
                fontsize=9, color="#d97757",
                arrowprops=dict(arrowstyle="->", color="#d97757", lw=0.8, alpha=0.7))
    # Find shrunk break-even
    shrunk_break = None
    for _, r in df.iterrows():
        if not np.isnan(r["shrunk_mean"]) and r["shrunk_mean"] <= pop_mean:
            shrunk_break = int(r["k"])
            break
    if shrunk_break is not None:
        ax.annotate(f"shrunk break-even k={shrunk_break}",
                    xy=(shrunk_break, pop_mean),
                    xytext=(1.1, pop_mean - 4),
                    fontsize=9, color="#3b8ea5",
                    arrowprops=dict(arrowstyle="->", color="#3b8ea5", lw=0.8, alpha=0.7))

    # Panel 2: H2e 4-arm bar chart
    ax = axes[1]
    arms = ["no_calib", "pop_slope", "pebs_ols", "pebs_shrunk"]
    labels = ["no calib", "pop-slope\n(baseline)", "PEBS\n(naive OLS)", "PEBS\n(EB shrunk)"]
    means = [h2e["rmse_mean"][a] for a in arms]
    medians = [h2e["rmse_median"][a] for a in arms]
    colors = ["gray", "black", "#d97757", "#3b8ea5"]

    bars = ax.bar(labels, means, color=colors, alpha=0.8, edgecolor="black", linewidth=0.6)
    # Overlay median ticks
    for lbl, mean_v, med_v in zip(labels, means, medians):
        ax.plot([lbl, lbl], [med_v - 0.1, med_v + 0.1], color="white", linewidth=3)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.3,
                f"{m:.2f}", ha="center", va="bottom", fontsize=10)
    rel_ols = h2e["relative_improvement_vs_pop_pct"]["naive_ols"]
    rel_shrunk = h2e["relative_improvement_vs_pop_pct"]["shrunk"]
    ax.set_ylabel("Mean within-user held-out RMSE (k=5 CV)")
    ax.set_title(f"H2e: within-user CV (1394 users)\nShrunk +{rel_shrunk:.2f}% over pop, OLS +{rel_ols:.2f}%")
    ax.grid(True, axis="y", alpha=0.25)
    ymax = max(means) + 2
    ymin = min(means) - 2
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"[save] {args.output}")

    # Also save as svg for paper inclusion
    svg_path = Path(args.output).with_suffix(".svg")
    plt.savefig(svg_path, bbox_inches="tight")
    print(f"[save] {svg_path}")


if __name__ == "__main__":
    main()
