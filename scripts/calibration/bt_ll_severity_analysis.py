"""Severity / MDE analysis for T1 7B PPO BT-LL audit (iter+N+173).

Motivation: adversarial-review attack #4 — "2-seed Δ=-3.0e-4, CI ±0.003
cannot distinguish theorem-predicted null from a 2nd-order signal ~0.001."

This script computes the minimum detectable effect (MDE) at 80% power for
n ∈ {2, 3, 5} seeds using the per-seed observed SD on PRISM 500 held-out
pairs, so the paper can disclose the exact effect size below which our
test has low power — i.e. convert attack #4 from a rhetorical concern
into a quantified scope bound.

MDE formula (paired design, normal approx):
    MDE(n, α, β, σ) = (z_{1-α/2} + z_{1-β}) * σ * sqrt(2 / n)

where σ is the per-seed SD of Δ BT-LL and n is the number of seeds.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
BT_LL_DIR = ROOT / "results" / "track1_ppo_7b"


def main():
    seeds = [17, 101]
    deltas = []
    per_pair_se = []  # within-seed paired-t SE of mean
    n_pairs = None
    for s in seeds:
        p = BT_LL_DIR / f"seed{s}" / "bt_ll.json"
        if not p.exists():
            print(f"[warn] missing {p}")
            continue
        d = json.loads(p.read_text())
        deltas.append(d["mean_bt_ll_delta_pilsd_minus_vanilla"])
        per_pair_se.append(d["paired_t_bt_ll"]["se"])
        n_pairs = d["n_eval_pairs"]
    deltas = np.array(deltas)
    mean = float(deltas.mean())
    # between-seed SD (2 seeds → 1 dof, very noisy)
    sigma_between = float(deltas.std(ddof=1)) if len(deltas) > 1 else float("nan")
    # within-seed per-pair SD from the paired-t SE: SE = sd/sqrt(n_pairs) so sd = SE*sqrt(n_pairs).
    # Pooled across seeds: sd_pool = sqrt(mean(SE^2) * n_pairs)
    se_arr = np.array(per_pair_se)
    sigma_per_pair = float(np.sqrt(np.mean(se_arr**2) * n_pairs))
    # MDE for within-seed paired design (use per-pair SD, n = n_pairs × n_seeds):
    sigma = sigma_per_pair
    n_pairs_total_used = lambda n_seeds: n_pairs * n_seeds

    alpha = 0.05
    beta = 0.20  # 80% power
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(1 - beta)

    # MDE using within-seed per-pair sd and pooled n = n_pairs × n_seeds
    mde = {}
    for n_seeds in [2, 3, 5, 10]:
        n_eff = n_pairs * n_seeds  # total pairs (paired design)
        se = sigma / np.sqrt(n_eff)
        m = (z_a + z_b) * se * np.sqrt(2.0)  # paired z-test MDE
        mde[f"n_seeds={n_seeds}_n_pairs={n_eff}"] = float(m)

    # Power at current n=2 seeds for a range of hypothetical Δ_true:
    pow_at = {}
    n_eff_now = n_pairs * 2
    se_now = sigma / np.sqrt(n_eff_now)
    for delta_true in [0.0001, 0.0005, 0.001, 0.002, 0.005]:
        z_non = delta_true / se_now
        p_pow = 1 - norm.cdf(z_a - z_non)  # one-sided upper tail approx for |δ|
        pow_at[f"delta_true={delta_true}"] = float(p_pow)

    out = {
        "seeds_observed": seeds,
        "delta_values": deltas.tolist(),
        "delta_mean": mean,
        "sigma_between_seeds_1dof": sigma_between,
        "sigma_per_pair_pooled": sigma_per_pair,
        "n_pairs_per_seed": n_pairs,
        "alpha": alpha,
        "beta_target": beta,
        "MDE_at_80pct_power": mde,
        "power_at_n2_seeds_given_true_delta": pow_at,
        "note": "Within-seed paired design uses per-pair sd (n=500 pairs per seed); between-seed sd is 1-dof and unreliable. The within-seed MDE is the correct scope bound.",
    }
    out_path = BT_LL_DIR / "mde_severity.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\n[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
