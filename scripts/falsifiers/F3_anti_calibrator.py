"""F3 — Anti-calibrator (inverted-slope).

Pre-registered falsifier for the claim that PEBS's
directionality (slope sign) is load-bearing, as opposed to the shrinkage
STRUCTURE alone driving the gain regardless of direction.

Design
------
Apply standard PEBS LOCO pipeline. For each user, compute the standard
EB-shrunk (alpha_shrunk, beta_shrunk), then deliberately construct the
ANTI-calibrator prediction on the held-out fold by flipping the slope
sign:
    pred_anti  = alpha_shrunk - beta_shrunk * x_te      (sign-flipped slope)
    pred_anti2 = alpha_shrunk + (1 - beta_shrunk) * x_te (1-beta spec)

Compare RMSE of anti-calibrator vs pop-slope baseline and vs standard
PEBS on the same held-out slice, with cluster-bootstrap 95% CI.

Pre-registered criterion:
  CONFIRMING:  RMSE(anti) - RMSE(pop_slope) (as a relative %) is >= +5 pp
               WORSE than pop-slope (lower bound of CI > 0 increase),
               demonstrating directionality is load-bearing.
  FALSIFYING:  Anti-calibrator does NOT hurt by >= +5 pp relative to pop
               (CI straddles zero or suggests shrinkage structure alone drives gain).

Outputs
-------
results/falsifiers/F3_anti_calibrator.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T1 = ROOT.parent / "1_Causal_RLHF"
SCORED = T1 / "data/prism_rm_scored.parquet"
OUT = ROOT / "results/falsifiers"

MIN_CONV_PER_USER = 2
MIN_UTT_PER_USER = 10
N_BOOT = 2000
RNG_BASE = 20260420


def fit_ols_with_se(x, y):
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan
    xm = float(np.mean(x))
    ssx = float(np.sum((x - xm) ** 2))
    if ssx < 1e-10:
        return float(np.mean(y)), 0.0, np.inf, np.inf
    beta = float(np.sum((x - xm) * (y - np.mean(y))) / ssx)
    alpha = float(np.mean(y) - beta * xm)
    resid = y - (alpha + beta * x)
    n = len(x)
    if n <= 2:
        return alpha, beta, np.inf, np.inf
    mse = float(np.sum(resid ** 2) / (n - 2))
    se_alpha = float(np.sqrt(mse * (1.0 / n + xm ** 2 / ssx)))
    se_beta = float(np.sqrt(mse / ssx))
    return alpha, beta, se_alpha, se_beta


def shrunk_params(x_tr, y_tr, alpha_pop, beta_pop, tau2_a, tau2_b):
    a, b, se_a, se_b = fit_ols_with_se(x_tr, y_tr)
    if not np.isfinite(a):
        return alpha_pop, beta_pop, 0.0, 0.0
    w_a = tau2_a / (tau2_a + se_a ** 2 + 1e-12) if np.isfinite(se_a) else 0.0
    w_b = tau2_b / (tau2_b + se_b ** 2 + 1e-12) if np.isfinite(se_b) else 0.0
    a_s = w_a * a + (1 - w_a) * alpha_pop
    b_s = w_b * b + (1 - w_b) * beta_pop
    return a_s, b_s, w_a, w_b


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet(SCORED).dropna(subset=["score_user", "rm_score", "conversation_id"])
    conv_count = df.groupby("user_id")["conversation_id"].nunique()
    utt_count = df.groupby("user_id").size()
    keep = conv_count[(conv_count >= MIN_CONV_PER_USER) & (utt_count >= MIN_UTT_PER_USER)].index
    df = df[df["user_id"].isin(keep)].reset_index(drop=True)
    print(f"[filter] {len(df)} utt x {df['user_id'].nunique()} users")

    # Global fits
    slope_pop, intercept_pop = np.polyfit(df["rm_score"].to_numpy(),
                                          df["score_user"].to_numpy().astype(float), 1)
    alpha_pop, beta_pop = float(intercept_pop), float(slope_pop)

    user_stats = []
    for uid, g in df.groupby("user_id"):
        a, b, sa, sb = fit_ols_with_se(g["rm_score"].to_numpy(),
                                        g["score_user"].to_numpy().astype(float))
        user_stats.append((a, b, sa, sb))
    alphas = np.array([s[0] for s in user_stats if np.isfinite(s[0])])
    betas = np.array([s[1] for s in user_stats if np.isfinite(s[1])])
    sas = np.array([s[2] for s in user_stats if np.isfinite(s[2])])
    sbs = np.array([s[3] for s in user_stats if np.isfinite(s[3])])
    tau2_a = max(0.0, float(np.var(alphas, ddof=1) - np.mean(sas ** 2)))
    tau2_b = max(0.0, float(np.var(betas, ddof=1) - np.mean(sbs ** 2)))
    print(f"[global] alpha_pop={alpha_pop:.3f}  beta_pop={beta_pop:.5f}  "
          f"tau2_a={tau2_a:.2f}  tau2_b={tau2_b:.4f}")

    per_user = []
    for uid, g in df.groupby("user_id"):
        conv_ids = g["conversation_id"].unique().tolist()
        if len(conv_ids) < MIN_CONV_PER_USER:
            continue
        sq_pop, sq_pebs, sq_anti_flip, sq_anti_1minus = [], [], [], []
        n_utt_used = 0
        for hc in conv_ids:
            tr = g[g["conversation_id"] != hc]
            te = g[g["conversation_id"] == hc]
            if len(tr) < 3 or len(te) < 1:
                continue
            x_tr = tr["rm_score"].to_numpy(dtype=np.float64)
            y_tr = tr["score_user"].to_numpy(dtype=np.float64)
            x_te = te["rm_score"].to_numpy(dtype=np.float64)
            y_te = te["score_user"].to_numpy(dtype=np.float64)
            a_s, b_s, _, _ = shrunk_params(x_tr, y_tr, alpha_pop, beta_pop, tau2_a, tau2_b)

            pred_pop = alpha_pop + beta_pop * x_te
            pred_pebs = a_s + b_s * x_te
            pred_anti_flip = a_s - b_s * x_te            # slope-sign flipped
            pred_anti_1minus = a_s + (1.0 - b_s) * x_te  # 1-beta inversion

            sq_pop.extend(((pred_pop - y_te) ** 2).tolist())
            sq_pebs.extend(((pred_pebs - y_te) ** 2).tolist())
            sq_anti_flip.extend(((pred_anti_flip - y_te) ** 2).tolist())
            sq_anti_1minus.extend(((pred_anti_1minus - y_te) ** 2).tolist())
            n_utt_used += len(y_te)
        if n_utt_used < MIN_UTT_PER_USER:
            continue
        per_user.append({
            "user_id": uid, "n_utt": n_utt_used,
            "rmse_pop": float(np.sqrt(np.mean(sq_pop))),
            "rmse_pebs": float(np.sqrt(np.mean(sq_pebs))),
            "rmse_anti_flip": float(np.sqrt(np.mean(sq_anti_flip))),
            "rmse_anti_1minus": float(np.sqrt(np.mean(sq_anti_1minus))),
        })
    pu = pd.DataFrame(per_user)
    pu["gain_pebs_pct"] = 100.0 * (pu["rmse_pop"] - pu["rmse_pebs"]) / pu["rmse_pop"]
    pu["hurt_anti_flip_pct"] = 100.0 * (pu["rmse_anti_flip"] - pu["rmse_pop"]) / pu["rmse_pop"]
    pu["hurt_anti_1minus_pct"] = 100.0 * (pu["rmse_anti_1minus"] - pu["rmse_pop"]) / pu["rmse_pop"]

    def boot_ci(values):
        rng = np.random.default_rng(RNG_BASE)
        n = len(values)
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            boots[b] = float(values[idx].mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return float(values.mean()), float(lo), float(hi)

    gain_pebs = boot_ci(pu["gain_pebs_pct"].to_numpy())
    hurt_flip = boot_ci(pu["hurt_anti_flip_pct"].to_numpy())
    hurt_1m = boot_ci(pu["hurt_anti_1minus_pct"].to_numpy())

    print("\n=== F3 anti-calibrator results ===")
    print(f"N users evaluated: {len(pu)}")
    print(f"PEBS gain vs pop: {gain_pebs[0]:+.3f}pp  CI [{gain_pebs[1]:+.2f}, {gain_pebs[2]:+.2f}]")
    print(f"Anti-flip HURT vs pop: {hurt_flip[0]:+.3f}pp  CI [{hurt_flip[1]:+.2f}, {hurt_flip[2]:+.2f}]")
    print(f"Anti-(1-b) HURT vs pop: {hurt_1m[0]:+.3f}pp  CI [{hurt_1m[1]:+.2f}, {hurt_1m[2]:+.2f}]")

    # Pre-registered criterion: anti must hurt by >= 5 pp (positive magnitude) with CI lower bound > 0
    threshold_pp = 5.0
    flip_passes = (hurt_flip[0] >= threshold_pp and hurt_flip[1] > 0)
    # Confirmed if EITHER anti variant hurts by >= 5 pp with CI > 0
    branch = "confirming" if flip_passes else "falsifying"

    summary = {
        "config": {
            "threshold_pp": threshold_pp,
            "criterion": "anti_flip must increase RMSE by >= 5 pp vs pop_slope with CI lower > 0",
            "n_bootstrap": N_BOOT,
            "rng_base": RNG_BASE,
        },
        "n_users": int(len(pu)),
        "alpha_pop": alpha_pop,
        "beta_pop": beta_pop,
        "tau2_a": tau2_a,
        "tau2_b": tau2_b,
        "pebs_gain_pp": {"mean": gain_pebs[0], "ci95": [gain_pebs[1], gain_pebs[2]]},
        "anti_flip_hurt_pp": {"mean": hurt_flip[0], "ci95": [hurt_flip[1], hurt_flip[2]]},
        "anti_1minus_hurt_pp": {"mean": hurt_1m[0], "ci95": [hurt_1m[1], hurt_1m[2]]},
        "branch_disposition": {
            "branch": branch,
            "flip_passes_threshold": bool(flip_passes),
            "note": "Primary criterion: slope-sign-flipped anti-calibrator. Secondary 1-beta inversion reported for completeness.",
        },
        "runtime_seconds": float(time.time() - t0),
    }
    (OUT / "F3_anti_calibrator.json").write_text(json.dumps(summary, indent=2))
    pu.to_parquet(OUT / "F3_per_user.parquet", index=False)

    print(f"\n=== BRANCH: {branch.upper()} ===")
    print(f"Anti-flip hurt {hurt_flip[0]:+.2f}pp, threshold {threshold_pp}pp")


if __name__ == "__main__":
    main()
