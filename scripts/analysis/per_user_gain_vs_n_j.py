"""Per-user PEBS gain as a function of per-user observations n_j.

Morris 1983 EB theory predicts the shrinkage weight ω_j = τ²/(τ² + σ²/n_j).
As n_j grows, ω_j → 1 (PEBS approaches OLS) and the per-user GAIN over
OLS-pop-slope decays. Testing this *explicitly* on PRISM (1394 users,
n_j varying 10..~500+) is a second causal check on top of the c-sweep
already reported in §4.1.

Inputs:
  results/track1_user_score_mse_shrunk.parquet — per-user RMSEs + n_j

Outputs:
  results/track1_gain_vs_n_j/summary.json
  results/track1_gain_vs_n_j/per_bin.parquet
  results/track1_gain_vs_n_j/figure_gain_vs_n.pdf

Analyses:
  1. Gain_j = rmse_pop_slope - rmse_pebs_shrunk (positive = PEBS wins)
  2. Spearman rank correlation of Gain_j against -n_j (theory: positive rho)
  3. Bin by n_j quintile, report mean Gain + cluster-bootstrap CI
  4. Overlay Morris 1983 theoretical decay curve
     ω(n) = τ² / (τ² + σ_ε²/n) using REML-fit τ², σ_ε²
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN = (Path(__file__).resolve().parents[2] / "1_Causal_RLHF/results/track1_user_score_mse_shrunk.parquet")
OUT_DIR = (Path(__file__).resolve().parents[2] / "3_PEBS_Standalone/results/track1_gain_vs_n_j")

TAU2_ALPHA = 111.7  # from the REML fit of tau^2 on PRISM
SIGMA2_EPS = 23.47 ** 2
N_BOOT = 2000
RNG = 20260420


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG)

    df = pd.read_parquet(IN).copy()
    df["gain"] = df["rmse_pop_slope"] - df["rmse_pebs_shrunk"]
    df = df.dropna(subset=["n", "gain"]).reset_index(drop=True)
    print(f"[load] {len(df)} users, n_j range [{df['n'].min()}, {df['n'].max()}]")
    print(f"[overall] mean gain = {df['gain'].mean():+.4f}, median = {df['gain'].median():+.4f}")

    # Test 1: Spearman rho (gain vs n_j). Morris predicts NEGATIVE (gain decays in n_j)
    rho, p = stats.spearmanr(df["n"], df["gain"])
    print(f"[Spearman] rho(gain, n_j) = {rho:+.4f}  p = {p:.3e}  (theory: < 0)")

    # Test 2: quintile bins
    df["quintile"] = pd.qcut(df["n"], q=5, labels=False, duplicates="drop")
    bins = []
    for q, g in df.groupby("quintile"):
        # Bootstrap CI for mean gain in this bin
        boots = np.array([rng.choice(g["gain"].to_numpy(), size=len(g), replace=True).mean()
                          for _ in range(N_BOOT)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        bins.append({
            "quintile": int(q),
            "n_j_lo": int(g["n"].min()),
            "n_j_hi": int(g["n"].max()),
            "n_j_median": float(g["n"].median()),
            "n_users": int(len(g)),
            "mean_gain": float(g["gain"].mean()),
            "ci95_lo": float(lo),
            "ci95_hi": float(hi),
            # Theoretical ω at this n
            "omega_theory": float(TAU2_ALPHA / (TAU2_ALPHA + SIGMA2_EPS / g["n"].median())),
        })
    bin_df = pd.DataFrame(bins)
    print("\n  quintile  n_j range        median  users  mean_gain  CI95                        ω_theory")
    for _, r in bin_df.iterrows():
        print(f"  Q{int(r['quintile'])+1:<8d} [{int(r['n_j_lo']):>3d}, {int(r['n_j_hi']):>4d}]      "
              f"{r['n_j_median']:>5.0f}   {int(r['n_users']):>3d}  "
              f"{r['mean_gain']:+.3f}     [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]   "
              f"{r['omega_theory']:.4f}")

    # Test 3: Spearman rank between quintile mean gain and ω_theory
    rho_bin, p_bin = stats.spearmanr(bin_df["mean_gain"], bin_df["omega_theory"])
    print(f"\n[Spearman bin-level] rho(mean_gain, ω_theory) = {rho_bin:+.4f}  p = {p_bin:.3e}")

    # Figure: gain vs n_j scatter + quintile means + theory curve
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.scatter(df["n"], df["gain"], alpha=0.20, s=12, color="gray", label="per-user gain")
    ax1.errorbar(bin_df["n_j_median"], bin_df["mean_gain"],
                 yerr=[bin_df["mean_gain"] - bin_df["ci95_lo"],
                       bin_df["ci95_hi"] - bin_df["mean_gain"]],
                 fmt="o-", color="C0", capsize=4, label="quintile mean ± 95% CI", zorder=5)
    ax1.axhline(0, color="k", lw=0.5, ls="--")
    ax1.set_xlabel(r"per-user observations $n_j$")
    ax1.set_ylabel(r"Gain = RMSE$_\mathrm{pop}$ $-$ RMSE$_\mathrm{PEBS}$ (higher = better)")
    ax1.legend(loc="upper right")
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    n_grid = np.linspace(df["n"].min(), df["n"].max(), 500)
    omega_grid = TAU2_ALPHA / (TAU2_ALPHA + SIGMA2_EPS / n_grid)
    ax2.plot(n_grid, omega_grid, color="C3", lw=1.5, ls=":", label=r"Morris $\omega(n)$")
    ax2.set_ylabel(r"Theoretical shrinkage weight $\omega$", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.legend(loc="lower right")

    fig.suptitle(r"PEBS gain vs per-user $n_j$ — Morris 1983 falsifiability check")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_gain_vs_n.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure_gain_vs_n.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_users": int(len(df)),
        "n_j_min": int(df["n"].min()),
        "n_j_max": int(df["n"].max()),
        "overall_mean_gain": float(df["gain"].mean()),
        "spearman_gain_vs_n_j": {"rho": float(rho), "p": float(p)},
        "spearman_bin_level": {"rho": float(rho_bin), "p": float(p_bin)},
        "quintiles": bins,
        "tau2_alpha_used": TAU2_ALPHA,
        "sigma2_eps_used": SIGMA2_EPS,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    bin_df.to_parquet(OUT_DIR / "per_bin.parquet", index=False)
    print(f"\nWrote {OUT_DIR}/summary.json + figure_gain_vs_n.{{pdf,png}}")


if __name__ == "__main__":
    main()
