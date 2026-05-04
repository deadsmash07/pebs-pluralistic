"""F4 — Adversarial-user injection.

Pre-registered (iter+N+290) falsifier for PILSD's deployment safety story:
does the partial-pooling structure degrade gracefully under adversarial
users, or can a minority of adversaries pull the population prior (and
shrunken estimates of all other users) toward zero / wrong sign?

Design
------
Take PRISM (1394 users, ~49 utt/user, 7B Qwen RM). For each corruption
fraction p in {0.10, 0.20, 0.30}:
  1. Randomly select floor(p * N_users) users to corrupt.
  2. Corruption scheme: FLIP the per-user score_user via
         score_user_corrupted = 100 - score_user
     (PRISM scores live in [0, 100]; flipping inverts the user's preference
     monotonically, matching the "label-flipped" setting in the spec.)
  3. Fit tau^2 + per-user (alpha, beta) on the CORRUPTED data (this is what
     a production system would do if it didn't know who is adversarial).
  4. Evaluate held-out RMSE ONLY on the NON-CORRUPTED users' LOCO folds,
     with the SAME conversation-hold-out splits as the clean-reference run.
  5. Compare PILSD gain vs pop-slope and vs naive no-pooling OLS.

Pre-registered criterion (iter+N+290):
  CONFIRMING:  PILSD gain on clean users remains > 0 and > naive-OLS gain
               through p = 0.20 (at p = 0.30, may drop but no catastrophic collapse).
  FALSIFYING:  PILSD gain on clean users falls BELOW pop-slope (i.e., gain
               becomes negative or below 0) at p <= 0.20, meaning adversaries
               pulled tau^2/alpha_pop/beta_pop into wrong regime.

Outputs
-------
results/falsifiers_iter290/F4_adversarial_user_injection.json
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
OUT = ROOT / "results/falsifiers_iter290"

MIN_CONV_PER_USER = 2
MIN_UTT_PER_USER = 10
N_BOOT = 2000
RNG_BASE = 20260420
CORRUPTION_LEVELS = [0.0, 0.10, 0.20, 0.30]


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


def shrunk_predict(x_tr, y_tr, x_te, alpha_pop, beta_pop, tau2_a, tau2_b):
    a, b, se_a, se_b = fit_ols_with_se(x_tr, y_tr)
    if not np.isfinite(a):
        return alpha_pop + beta_pop * x_te
    w_a = tau2_a / (tau2_a + se_a ** 2 + 1e-12) if np.isfinite(se_a) else 0.0
    w_b = tau2_b / (tau2_b + se_b ** 2 + 1e-12) if np.isfinite(se_b) else 0.0
    a_s = w_a * a + (1 - w_a) * alpha_pop
    b_s = w_b * b + (1 - w_b) * beta_pop
    return a_s + b_s * x_te


def nopool_predict(x_tr, y_tr, x_te, alpha_pop, beta_pop):
    """Naive per-user OLS with fallback to pop on degenerate folds."""
    a, b, _, _ = fit_ols_with_se(x_tr, y_tr)
    if not np.isfinite(a):
        return alpha_pop + beta_pop * x_te
    return a + b * x_te


def evaluate_one(df_corrupted: pd.DataFrame, clean_users: set, label: str):
    """Fit PILSD on df_corrupted (global tau2, alpha_pop, beta_pop use corrupted data);
    evaluate LOCO RMSE only on clean_users (using their clean data stored in
    df_corrupted since only corrupted users had their score_user flipped)."""
    slope_pop, intercept_pop = np.polyfit(df_corrupted["rm_score"].to_numpy(),
                                          df_corrupted["score_user"].to_numpy().astype(float), 1)
    alpha_pop, beta_pop = float(intercept_pop), float(slope_pop)

    user_stats = []
    for uid, g in df_corrupted.groupby("user_id"):
        a, b, sa, sb = fit_ols_with_se(g["rm_score"].to_numpy(),
                                        g["score_user"].to_numpy().astype(float))
        user_stats.append((a, b, sa, sb))
    alphas = np.array([s[0] for s in user_stats if np.isfinite(s[0])])
    betas = np.array([s[1] for s in user_stats if np.isfinite(s[1])])
    sas = np.array([s[2] for s in user_stats if np.isfinite(s[2])])
    sbs = np.array([s[3] for s in user_stats if np.isfinite(s[3])])
    tau2_a = max(0.0, float(np.var(alphas, ddof=1) - np.mean(sas ** 2)))
    tau2_b = max(0.0, float(np.var(betas, ddof=1) - np.mean(sbs ** 2)))
    print(f"[{label}] alpha_pop={alpha_pop:.3f}  beta_pop={beta_pop:.5f}  "
          f"tau2_a={tau2_a:.2f}  tau2_b={tau2_b:.4f}")

    per_user = []
    # Evaluate only on clean users; use their clean (non-flipped) data for evaluation
    # BUT training-fold uses what's in df_corrupted for that user (which IS their clean data
    # because we only flipped the corrupted subset).
    for uid in clean_users:
        g = df_corrupted[df_corrupted["user_id"] == uid]
        if len(g) == 0:
            continue
        conv_ids = g["conversation_id"].unique().tolist()
        if len(conv_ids) < MIN_CONV_PER_USER:
            continue
        sq_pop, sq_pilsd, sq_nopool = [], [], []
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
            pred_pop = alpha_pop + beta_pop * x_te
            pred_pilsd = shrunk_predict(x_tr, y_tr, x_te, alpha_pop, beta_pop, tau2_a, tau2_b)
            pred_nopool = nopool_predict(x_tr, y_tr, x_te, alpha_pop, beta_pop)
            sq_pop.extend(((pred_pop - y_te) ** 2).tolist())
            sq_pilsd.extend(((pred_pilsd - y_te) ** 2).tolist())
            sq_nopool.extend(((pred_nopool - y_te) ** 2).tolist())
            n_utt_used += len(y_te)
        if n_utt_used < MIN_UTT_PER_USER:
            continue
        per_user.append({
            "user_id": uid, "n_utt": n_utt_used,
            "rmse_pop": float(np.sqrt(np.mean(sq_pop))),
            "rmse_pilsd": float(np.sqrt(np.mean(sq_pilsd))),
            "rmse_nopool": float(np.sqrt(np.mean(sq_nopool))),
        })
    pu = pd.DataFrame(per_user)
    pu["gain_pilsd_pct"] = 100.0 * (pu["rmse_pop"] - pu["rmse_pilsd"]) / pu["rmse_pop"]
    pu["gain_nopool_pct"] = 100.0 * (pu["rmse_pop"] - pu["rmse_nopool"]) / pu["rmse_pop"]

    def boot_ci(values):
        rng = np.random.default_rng(RNG_BASE)
        n = len(values)
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            boots[b] = float(values[idx].mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return float(values.mean()), float(lo), float(hi)

    g_pilsd = boot_ci(pu["gain_pilsd_pct"].to_numpy())
    g_nopool = boot_ci(pu["gain_nopool_pct"].to_numpy())

    return {
        "n_clean_users_evaluated": int(len(pu)),
        "alpha_pop": alpha_pop,
        "beta_pop": beta_pop,
        "tau2_a": tau2_a,
        "tau2_b": tau2_b,
        "rmse_pop_mean": float(pu["rmse_pop"].mean()),
        "rmse_pilsd_mean": float(pu["rmse_pilsd"].mean()),
        "rmse_nopool_mean": float(pu["rmse_nopool"].mean()),
        "gain_pilsd_pct": {"mean": g_pilsd[0], "ci95": [g_pilsd[1], g_pilsd[2]]},
        "gain_nopool_pct": {"mean": g_nopool[0], "ci95": [g_nopool[1], g_nopool[2]]},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet(SCORED).dropna(subset=["score_user", "rm_score", "conversation_id"])
    conv_count = df.groupby("user_id")["conversation_id"].nunique()
    utt_count = df.groupby("user_id").size()
    keep = conv_count[(conv_count >= MIN_CONV_PER_USER) & (utt_count >= MIN_UTT_PER_USER)].index
    df = df[df["user_id"].isin(keep)].reset_index(drop=True)
    print(f"[filter] {len(df)} utt x {df['user_id'].nunique()} users")

    all_users = sorted(df["user_id"].unique())
    rng = np.random.default_rng(RNG_BASE)

    results = {}
    for p in CORRUPTION_LEVELS:
        print(f"\n[corrupt p={p:.2f}]")
        n_corrupt = int(np.floor(p * len(all_users)))
        if n_corrupt == 0:
            corrupt_users = set()
        else:
            # Seeded fixed draw per level so different p's can be compared
            rng_level = np.random.default_rng(RNG_BASE + int(p * 1000))
            corrupt_users = set(rng_level.choice(all_users, size=n_corrupt, replace=False))
        clean_users = set(all_users) - corrupt_users

        # Build corrupted df: flip score_user for corrupt_users
        df_c = df.copy()
        mask = df_c["user_id"].isin(corrupt_users)
        # PRISM score_user is 0..100; flip monotonically: y -> 100 - y
        df_c.loc[mask, "score_user"] = 100.0 - df_c.loc[mask, "score_user"].astype(float)
        print(f"  n_corrupt_users={len(corrupt_users)}  n_clean_users={len(clean_users)}  "
              f"n_rows_flipped={int(mask.sum())}")

        res = evaluate_one(df_c, clean_users, label=f"p={p:.2f}")
        res["corruption_fraction"] = float(p)
        res["n_corrupt_users"] = int(n_corrupt)
        results[f"p_{int(round(p*100))}"] = res

        print(f"  PILSD gain on clean users: {res['gain_pilsd_pct']['mean']:+.3f}pp  "
              f"CI [{res['gain_pilsd_pct']['ci95'][0]:+.2f}, {res['gain_pilsd_pct']['ci95'][1]:+.2f}]")
        print(f"  naive-OLS gain on clean users: {res['gain_nopool_pct']['mean']:+.3f}pp  "
              f"CI [{res['gain_nopool_pct']['ci95'][0]:+.2f}, {res['gain_nopool_pct']['ci95'][1]:+.2f}]")

    # Pre-registered criterion:
    # confirming if PILSD gain at p=0.20 is still >0 AND >= naive-OLS gain
    g20 = results["p_20"]["gain_pilsd_pct"]
    n20 = results["p_20"]["gain_nopool_pct"]
    pilsd_positive_at_20 = (g20["mean"] > 0 and g20["ci95"][0] > 0)
    pilsd_beats_nopool_at_20 = (g20["mean"] > n20["mean"])
    branch = "confirming" if (pilsd_positive_at_20 and pilsd_beats_nopool_at_20) else "falsifying"

    summary = {
        "config": {
            "corruption_levels": CORRUPTION_LEVELS,
            "corruption_scheme": "score_user -> 100 - score_user (PRISM 0..100 scale)",
            "criterion": "At p=0.20: PILSD gain > 0 (CI > 0) AND > naive-OLS gain",
            "n_bootstrap": N_BOOT,
            "rng_base": RNG_BASE,
        },
        "results_by_corruption": results,
        "branch_disposition": {
            "branch": branch,
            "pilsd_positive_at_p20": bool(pilsd_positive_at_20),
            "pilsd_beats_nopool_at_p20": bool(pilsd_beats_nopool_at_20),
        },
        "runtime_seconds": float(time.time() - t0),
    }
    (OUT / "F4_adversarial_user_injection.json").write_text(json.dumps(summary, indent=2))

    print("\n=== F4 SUMMARY ===")
    for p in CORRUPTION_LEVELS:
        key = f"p_{int(round(p*100))}"
        r = results[key]
        print(f"p={p:.2f}: PILSD={r['gain_pilsd_pct']['mean']:+6.3f}pp  "
              f"naive-OLS={r['gain_nopool_pct']['mean']:+6.3f}pp")
    print(f"\nBranch: {branch.upper()}")


if __name__ == "__main__":
    main()
