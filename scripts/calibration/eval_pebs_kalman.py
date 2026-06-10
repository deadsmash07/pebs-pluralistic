"""Time-varying (alpha_j(t), beta_j(t)) PEBS via Kalman filter on PRISM.

This is the *falsifiable* extension predicted by Prop T1.MI (paper §3.2 /
track1_monotone_invariance_theorem.md): static affine calibration is
monotone-invariant and therefore cannot move rank-only downstream metrics at
PPO convergence. Prop T1.MI predicts that a **time-varying** calibrator breaks
the monotone-invariance premise. If the time-varying generalisation also
fails to separate from static on scale-aware metrics (RMSE), that constrains
the theorem's scope: drift in (alpha_j, beta_j) must be *predictively*
large, not merely statistically detectable, to change the within-user
calibration story.

Three arms, same temporal 80/20 split per user, same 1394 users used for
POS-T1.1:

  1. pebs_static       — EB-shrunk (alpha_j, beta_j) fit ONCE on train-half
  2. pebs_random_walk  — Kalman F=I (pure random walk), (q_alpha, q_beta, R)
                          estimated by POOLED MLE across all users on train
  3. pebs_kalman       — Kalman F=diag(rho_alpha, rho_beta) AR(1),
                          (rho_alpha, rho_beta, q_alpha, q_beta, R) all MLE

All three use the SAME per-user temporal ordering, SAME prior
(alpha_pop, beta_pop, diag(tau_alpha^2, tau_beta^2)), SAME train observations.

MLE: we fit the 3 process parameters (or 5 for AR(1)) by maximising the
pooled likelihood
    L(theta) = sum_j log p(y_j^train | theta, alpha_pop, beta_pop, tau^2)
where p(y_j | theta) is the Kalman filter innovation likelihood integrated
over the user's train trajectory. Runtime budget: a full pooled MLE with
L-BFGS-B converges in ~200 likelihood calls; each call is O(n_total) for
the filter pass, so total pooled-MLE cost is O(k * n_total) ~ a few minutes
on CPU for 68k utterances.

Bootstrap: 30-seed user-level resample (n=1394 with replacement) to build
RMSE CI and relative-improvement CI.

Diagnostics emitted:
  * per-user beta_j(t) filtered trajectory: Var(beta_j(t)) over train obs
  * Across-user distribution of these within-user variances
  * Correlation (alpha_j(t_end), beta_j(t_end)) from Kalman vs static OLS
  * Top-K users ranked by |beta_j(t_end) - beta_j_static| to show where
    time-varying matters

Output:
  results/track1_pebs_kalman/eval.json
  results/track1_pebs_kalman/per_user.parquet
  results/track1_pebs_kalman/beta_trajectories.parquet  (long-format)

Run:
  cd IMPLEMENTATION/1_Causal_RLHF
  python3 scripts/eval_pebs_kalman.py

  (fast smoke: --max-users 100; default is all 1394)

NO GPU required. Pure CPU (numpy + scipy.optimize). ~2-4 min total.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scored-parquet",
                   default="data/prism_rm_scored.parquet")
    p.add_argument("--timestamp-cache",
                   default="data/prism_conversation_timestamps.parquet")
    p.add_argument("--min-obs-per-user", type=int, default=6)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--n-bootstrap", type=int, default=30,
                   help="Bootstrap seeds for RMSE CI")
    p.add_argument("--max-users", type=int, default=0,
                   help="If >0, subsample users (for smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir",
                   default="results/track1_pebs_kalman")
    p.add_argument("--mle-max-iter", type=int, default=200)
    p.add_argument("--rho-fixed", type=float, default=None,
                   help="If set, fix rho_alpha=rho_beta=value instead of MLE"
                        " (e.g. 1.0 for random walk, 0.2 per split-half rho).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers (parity with temporal-CV script)
# ---------------------------------------------------------------------------

def ut_ordinal(uid: str) -> int:
    m = re.match(r"ut(\d+)", str(uid))
    return int(m.group(1)) if m else 0


def temporal_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ut_ord"] = df["utterance_id"].apply(ut_ordinal)
    return df.sort_values(
        ["generated_datetime", "turn", "within_turn_id", "ut_ord"],
        kind="mergesort",
    )


def ols_with_V(x: np.ndarray, y: np.ndarray):
    """Per-user OLS + sampling variance on (intercept, slope)."""
    k = len(x)
    if k < 2 or np.var(x) < 1e-12:
        return float(np.mean(y)) if k else 0.0, 0.0, np.inf, np.inf, np.inf
    x_bar = x.mean()
    Sxx = ((x - x_bar) ** 2).sum()
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = intercept + slope * x
    sigma_hat_sq = ((y - y_pred) ** 2).sum() / max(k - 2, 1)
    V_int = sigma_hat_sq * (1.0 / k + x_bar ** 2 / max(Sxx, 1e-12))
    V_slope = sigma_hat_sq / max(Sxx, 1e-12)
    return (float(intercept), float(slope),
            float(V_int), float(V_slope), float(sigma_hat_sq))


# ---------------------------------------------------------------------------
# Kalman filter (AR(1) state, bivariate)
# ---------------------------------------------------------------------------

def kalman_filter(
    x: np.ndarray,
    y: np.ndarray,
    mu0: np.ndarray,       # (2,) prior mean
    Sigma0: np.ndarray,    # (2,2) prior cov
    F: np.ndarray,         # (2,2) state transition (diag for AR(1))
    Q: np.ndarray,         # (2,2) process noise cov
    R: float,              # scalar obs noise
    mu_stationary: np.ndarray | None = None,  # AR(1) mean-reversion target
    return_trajectory: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray | None]:
    """Kalman filter for observation y_t = [1, x_t] * [alpha_t, beta_t]^T + v,
    state transition (x_t+1 - mu_*) = F (x_t - mu_*) + w, w ~ N(0, Q).

    Returns (mu_T, Sigma_T, neg_log_lik, trajectory_or_None).
    trajectory has shape (T, 2) with filtered state means if requested.
    """
    mu = mu0.copy()
    Sigma = Sigma0.copy()
    T = len(x)
    nll = 0.0
    traj = np.empty((T, 2)) if return_trajectory else None
    if mu_stationary is None:
        mu_stationary = np.zeros(2)

    for t in range(T):
        # Predict
        mu_pred = mu_stationary + F @ (mu - mu_stationary)
        Sigma_pred = F @ Sigma @ F.T + Q
        h = np.array([1.0, x[t]])
        y_hat = h @ mu_pred
        S = float(h @ Sigma_pred @ h + R)
        if S <= 0 or not np.isfinite(S):
            return mu_pred, Sigma_pred, np.inf, traj
        innov = y[t] - y_hat
        nll += 0.5 * (np.log(2 * np.pi * S) + innov * innov / S)
        K = Sigma_pred @ h / S
        mu = mu_pred + K * innov
        I_KH = np.eye(2) - np.outer(K, h)
        Sigma = I_KH @ Sigma_pred @ I_KH.T + R * np.outer(K, K)
        if return_trajectory:
            traj[t] = mu
    return mu, Sigma, nll, traj


# ---------------------------------------------------------------------------
# Pooled MLE for process parameters
# ---------------------------------------------------------------------------

def _unpack_theta_rw(theta):
    """Random-walk: F=I. theta = log([q_alpha, q_beta, R]). Positive via exp."""
    q_a, q_b, R = np.exp(theta)
    return q_a, q_b, R


def _unpack_theta_ar1(theta):
    """AR(1): F = diag(rho_alpha, rho_beta). theta=
    [logit(rho_a), logit(rho_b), log(q_a), log(q_b), log(R)]. rho in (0,1)."""
    logit_ra, logit_rb, log_qa, log_qb, log_R = theta
    rho_a = 1.0 / (1.0 + np.exp(-logit_ra))
    rho_b = 1.0 / (1.0 + np.exp(-logit_rb))
    q_a = np.exp(log_qa)
    q_b = np.exp(log_qb)
    R = np.exp(log_R)
    return rho_a, rho_b, q_a, q_b, R


def pooled_nll_rw(theta, user_trains, mu0, Sigma0):
    q_a, q_b, R = _unpack_theta_rw(theta)
    F = np.eye(2)
    Q = np.diag([q_a, q_b])
    total = 0.0
    for x_tr, y_tr in user_trains:
        _, _, nll, _ = kalman_filter(
            x_tr, y_tr, mu0, Sigma0, F, Q, R,
            mu_stationary=mu0, return_trajectory=False,
        )
        if not np.isfinite(nll):
            return 1e12
        total += nll
    return total


def pooled_nll_ar1(theta, user_trains, mu0, Sigma0):
    rho_a, rho_b, q_a, q_b, R = _unpack_theta_ar1(theta)
    F = np.diag([rho_a, rho_b])
    Q = np.diag([q_a, q_b])
    total = 0.0
    for x_tr, y_tr in user_trains:
        _, _, nll, _ = kalman_filter(
            x_tr, y_tr, mu0, Sigma0, F, Q, R,
            mu_stationary=mu0, return_trajectory=False,
        )
        if not np.isfinite(nll):
            return 1e12
        total += nll
    return total


def fit_rw_mle(user_trains, mu0, Sigma0, sigma2_init, tau2_init, max_iter=200):
    """Fit (q_alpha, q_beta, R) via pooled MLE for random-walk state."""
    # Initialise: small drift + R near residual variance
    theta0 = np.log([tau2_init[0] * 0.01, tau2_init[1] * 0.01, sigma2_init])
    t0 = time.time()
    res = minimize(
        pooled_nll_rw, theta0,
        args=(user_trains, mu0, Sigma0),
        method="L-BFGS-B",
        options={"maxiter": max_iter, "disp": False, "ftol": 1e-6},
    )
    q_a, q_b, R = _unpack_theta_rw(res.x)
    print(f"[MLE-RW] converged={res.success} nll={res.fun:.2f} "
          f"q_alpha={q_a:.4f} q_beta={q_b:.4f} R={R:.2f} "
          f"in {time.time()-t0:.1f}s ({res.nit} iter, {res.nfev} nfev)")
    return q_a, q_b, R, float(res.fun), bool(res.success)


def fit_ar1_mle(user_trains, mu0, Sigma0, sigma2_init, tau2_init, max_iter=200):
    """Fit (rho_alpha, rho_beta, q_alpha, q_beta, R) via pooled MLE for
    AR(1) state."""
    # Initialise: rho~0.99 (near-RW) + small drift + R near residual
    # logit(0.99) = 4.595
    theta0 = np.array([
        4.595,  # rho_alpha ~ 0.99
        4.595,  # rho_beta  ~ 0.99
        np.log(tau2_init[0] * 0.01),
        np.log(tau2_init[1] * 0.01),
        np.log(sigma2_init),
    ])
    t0 = time.time()
    res = minimize(
        pooled_nll_ar1, theta0,
        args=(user_trains, mu0, Sigma0),
        method="L-BFGS-B",
        options={"maxiter": max_iter, "disp": False, "ftol": 1e-6},
    )
    rho_a, rho_b, q_a, q_b, R = _unpack_theta_ar1(res.x)
    print(f"[MLE-AR1] converged={res.success} nll={res.fun:.2f} "
          f"rho_alpha={rho_a:.4f} rho_beta={rho_b:.4f} "
          f"q_alpha={q_a:.4f} q_beta={q_b:.4f} R={R:.2f} "
          f"in {time.time()-t0:.1f}s ({res.nit} iter, {res.nfev} nfev)")
    return rho_a, rho_b, q_a, q_b, R, float(res.fun), bool(res.success)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load ----
    df = pd.read_parquet(args.scored_parquet).dropna(
        subset=["score_user"]).reset_index(drop=True)
    print(f"[load] {len(df)} utterances, {df.user_id.nunique()} users")
    ts = pd.read_parquet(args.timestamp_cache)
    df = df.merge(ts, on="conversation_id", how="left").dropna(
        subset=["generated_datetime"]).reset_index(drop=True)
    print(f"[join] {len(df)} utterances with timestamps, "
          f"span={(df.generated_datetime.max()-df.generated_datetime.min()).days} days")

    if args.max_users > 0:
        uids = sorted(df.user_id.unique().tolist())
        rng = np.random.default_rng(args.seed)
        chosen = set(rng.choice(uids, size=min(args.max_users, len(uids)),
                                replace=False).tolist())
        df = df[df.user_id.isin(chosen)].reset_index(drop=True)
        print(f"[subsample] {df.user_id.nunique()} users, {len(df)} utterances")

    # ---- population pop-slope ----
    slope_pop, intercept_pop = np.polyfit(df.rm_score, df.score_user, 1)
    pop_alpha = float(intercept_pop)
    pop_beta = float(slope_pop)
    print(f"[pop] alpha0={pop_alpha:.3f}  beta0={pop_beta:.3f}")

    # ---- EB: per-user OLS → tau^2 moment estimate (informs prior) ----
    stats_rows = []
    sigma_user = []
    for uid, grp in df.groupby("user_id"):
        if len(grp) < args.min_obs_per_user:
            continue
        a, b, Va, Vb, s2 = ols_with_V(grp.rm_score.to_numpy(),
                                      grp.score_user.to_numpy().astype(float))
        stats_rows.append({"user_id": uid, "alpha": a, "beta": b,
                           "V_alpha": Va, "V_beta": Vb, "sigma2": s2,
                           "n": len(grp)})
        if np.isfinite(s2):
            sigma_user.append(s2)
    us = pd.DataFrame(stats_rows)
    V_a_tot = float(us.alpha.var())
    V_b_tot = float(us.beta.var())
    mean_Va = float(us.V_alpha.replace([np.inf, -np.inf], np.nan)
                    .dropna().mean())
    mean_Vb = float(us.V_beta.replace([np.inf, -np.inf], np.nan)
                    .dropna().mean())
    tau_a2 = max(V_a_tot - mean_Va, 1e-6)
    tau_b2 = max(V_b_tot - mean_Vb, 1e-6)
    sigma2_pop = float(np.median(sigma_user))
    print(f"[EB] tau_alpha^2={tau_a2:.3f}  tau_beta^2={tau_b2:.3f}  "
          f"sigma^2={sigma2_pop:.3f}")

    # ---- temporal 80/20 split + prepare per-user train/test arrays ----
    user_splits = {}   # uid -> (x_tr, y_tr, x_te, y_te, a_ols, b_ols, Va, Vb)
    for uid, grp in df.groupby("user_id"):
        if len(grp) < args.min_obs_per_user:
            continue
        g = temporal_sort(grp)
        n = len(g)
        n_test = max(1, int(round(n * args.test_fraction)))
        if n - n_test < 2:
            continue
        tr = g.iloc[:n - n_test]
        te = g.iloc[n - n_test:]
        x_tr = tr["rm_score"].to_numpy()
        y_tr = tr["score_user"].to_numpy().astype(float)
        x_te = te["rm_score"].to_numpy()
        y_te = te["score_user"].to_numpy().astype(float)
        a, b, Va, Vb, _ = ols_with_V(x_tr, y_tr)
        user_splits[uid] = (x_tr, y_tr, x_te, y_te, a, b, Va, Vb)
    n_users = len(user_splits)
    print(f"[split] {n_users} users with train/test both non-trivial")

    # ---- pooled MLE for RW and AR(1) ----
    mu0 = np.array([pop_alpha, pop_beta])
    Sigma0 = np.diag([tau_a2, tau_b2])

    # Use a subsample for MLE fitting to keep runtime bounded (pooled lik is
    # concentrated enough that 400 users is plenty); filter+predict run on ALL.
    rng = np.random.default_rng(args.seed + 11)
    mle_uids = list(user_splits.keys())
    if len(mle_uids) > 400:
        mle_uids = rng.choice(mle_uids, size=400, replace=False).tolist()
    user_trains_mle = [(user_splits[u][0], user_splits[u][1]) for u in mle_uids]
    print(f"[MLE] fitting pooled likelihood on {len(user_trains_mle)} users")

    # RW (F=I, pure random walk)
    q_a_rw, q_b_rw, R_rw, nll_rw, ok_rw = fit_rw_mle(
        user_trains_mle, mu0, Sigma0, sigma2_pop, (tau_a2, tau_b2),
        max_iter=args.mle_max_iter,
    )

    # AR(1) (F=diag(rho_a, rho_b))
    if args.rho_fixed is not None:
        rho_a = rho_b = float(args.rho_fixed)
        # Fit only (q_a, q_b, R) at fixed rho
        def _fixed_rho_nll(theta):
            return pooled_nll_ar1(
                np.concatenate([
                    np.array([
                        np.log(rho_a / (1 - rho_a + 1e-12)),
                        np.log(rho_b / (1 - rho_b + 1e-12)),
                    ]),
                    theta,
                ]),
                user_trains_mle, mu0, Sigma0,
            )
        theta0 = np.log([tau_a2 * 0.01, tau_b2 * 0.01, sigma2_pop])
        res = minimize(_fixed_rho_nll, theta0, method="L-BFGS-B",
                       options={"maxiter": args.mle_max_iter, "ftol": 1e-6})
        q_a, q_b, R = np.exp(res.x)
        nll_ar1 = float(res.fun)
        ok_ar1 = bool(res.success)
        print(f"[MLE-AR1-fixed] rho={rho_a} q_a={q_a:.4f} q_b={q_b:.4f} "
              f"R={R:.2f} nll={nll_ar1:.2f} ok={ok_ar1}")
    else:
        rho_a, rho_b, q_a, q_b, R, nll_ar1, ok_ar1 = fit_ar1_mle(
            user_trains_mle, mu0, Sigma0, sigma2_pop, (tau_a2, tau_b2),
            max_iter=args.mle_max_iter,
        )

    # ---- per-user arm scoring ----
    Q_rw = np.diag([q_a_rw, q_b_rw])
    F_rw = np.eye(2)
    Q_ar = np.diag([q_a, q_b])
    F_ar = np.diag([rho_a, rho_b])

    per_user = []
    beta_traj_rows = []   # long-format: (user_id, t, alpha_t, beta_t) for Kalman AR1
    t0 = time.time()
    for uid, (x_tr, y_tr, x_te, y_te, a_ols, b_ols, Va, Vb) in user_splits.items():
        row = {"user_id": uid, "n_train": len(x_tr), "n_test": len(x_te)}

        # (A) no_calib = train-fold mean
        row["rmse_no_calib"] = float(np.sqrt(np.mean(
            (np.full_like(y_te, y_tr.mean()) - y_te) ** 2)))

        # (B) pop_slope
        yh = pop_alpha + pop_beta * x_te
        row["rmse_pop_slope"] = float(np.sqrt(np.mean((yh - y_te) ** 2)))

        # (C) pebs_static: EB-shrunk
        omega_a = tau_a2 / (tau_a2 + Va) if np.isfinite(Va) else 0.0
        omega_b = tau_b2 / (tau_b2 + Vb) if np.isfinite(Vb) else 0.0
        a_s = omega_a * a_ols + (1 - omega_a) * pop_alpha
        b_s = omega_b * b_ols + (1 - omega_b) * pop_beta
        yh = a_s + b_s * x_te
        row["rmse_pebs_static"] = float(np.sqrt(np.mean((yh - y_te) ** 2)))
        row["alpha_static"] = a_s
        row["beta_static"] = b_s

        # (D) pebs_random_walk Kalman with MLE-fit (q_a, q_b, R_rw)
        mu_T, _, _, traj_rw = kalman_filter(
            x_tr, y_tr, mu0, Sigma0, F_rw, Q_rw, R_rw,
            mu_stationary=mu0, return_trajectory=True,
        )
        yh = mu_T[0] + mu_T[1] * x_te
        row["rmse_pebs_random_walk"] = float(np.sqrt(np.mean((yh - y_te) ** 2)))
        row["alpha_rw_end"] = float(mu_T[0])
        row["beta_rw_end"] = float(mu_T[1])
        row["beta_rw_var"] = float(np.var(traj_rw[:, 1]))
        row["alpha_rw_var"] = float(np.var(traj_rw[:, 0]))

        # (E) pebs_kalman AR(1) with MLE-fit (rho_a, rho_b, q_a, q_b, R)
        mu_T, _, _, traj_ar = kalman_filter(
            x_tr, y_tr, mu0, Sigma0, F_ar, Q_ar, R,
            mu_stationary=mu0, return_trajectory=True,
        )
        yh = mu_T[0] + mu_T[1] * x_te
        row["rmse_pebs_kalman"] = float(np.sqrt(np.mean((yh - y_te) ** 2)))
        row["alpha_kalman_end"] = float(mu_T[0])
        row["beta_kalman_end"] = float(mu_T[1])
        row["beta_kalman_var"] = float(np.var(traj_ar[:, 1]))
        row["alpha_kalman_var"] = float(np.var(traj_ar[:, 0]))
        # range diagnostics
        row["beta_kalman_range"] = float(traj_ar[:, 1].max() - traj_ar[:, 1].min())
        row["alpha_kalman_range"] = float(traj_ar[:, 0].max() - traj_ar[:, 0].min())

        per_user.append(row)

        # For memory sanity: only emit trajectories for a random 100 users
        # (otherwise ~68k rows, fine; but we cap at 200 for parquet artifact)
        if len(beta_traj_rows) < 20000:
            for t_idx in range(len(x_tr)):
                beta_traj_rows.append({
                    "user_id": uid, "t": t_idx,
                    "alpha_kalman_t": float(traj_ar[t_idx, 0]),
                    "beta_kalman_t": float(traj_ar[t_idx, 1]),
                    "alpha_rw_t": float(traj_rw[t_idx, 0]),
                    "beta_rw_t": float(traj_rw[t_idx, 1]),
                })
    pu = pd.DataFrame(per_user)
    traj_df = pd.DataFrame(beta_traj_rows)
    print(f"[score] {len(pu)} users scored in {time.time()-t0:.1f}s")

    # ---- headline RMSE means ----
    arms = ["no_calib", "pop_slope", "pebs_static",
            "pebs_random_walk", "pebs_kalman"]
    print(f"\n=== RMSE TEMPORAL 80/20 (n={len(pu)} users) ===")
    for arm in arms:
        col = f"rmse_{arm}"
        print(f"  {arm:>20}: mean={pu[col].mean():.4f}  "
              f"median={pu[col].median():.4f}")

    def pct_imp(col: str) -> float:
        return 100 * (pu["rmse_pop_slope"].mean() - pu[col].mean()) \
            / pu["rmse_pop_slope"].mean()

    rel = {a: pct_imp(f"rmse_{a}") for a in arms if a != "pop_slope"}
    print(f"\n=== Relative improvement vs pop_slope ===")
    for k, v in rel.items():
        print(f"  {k:>20}: {v:+.2f}%")

    # ---- paired Wilcoxon ----
    def paired(a_col, b_col):
        a = pu[a_col].to_numpy(); b = pu[b_col].to_numpy()
        w = stats.wilcoxon(a, b, alternative="two-sided")
        return {
            "mean_delta": float((a - b).mean()),
            "median_delta": float(np.median(a - b)),
            "frac_a_smaller": float((a < b).mean()),
            "wilcoxon_p": float(w.pvalue),
        }
    comparisons = {
        "kalman_vs_static":      paired("rmse_pebs_kalman", "rmse_pebs_static"),
        "kalman_vs_random_walk": paired("rmse_pebs_kalman", "rmse_pebs_random_walk"),
        "random_walk_vs_static": paired("rmse_pebs_random_walk", "rmse_pebs_static"),
        "static_vs_pop":         paired("rmse_pebs_static", "rmse_pop_slope"),
    }
    print(f"\n=== Paired Wilcoxon (n={len(pu)}) ===")
    for k, d in comparisons.items():
        print(f"  {k:>22}: Δ={d['mean_delta']:+.4f}  "
              f"a<b={d['frac_a_smaller']:.1%}  p={d['wilcoxon_p']:.3e}")

    # ---- Bootstrap 95% CI (user-level resample, ≥30 seeds) ----
    n_boot = max(args.n_bootstrap, 30)
    rng_boot = np.random.default_rng(args.seed + 31)
    boot_rmse = {a: [] for a in arms}
    boot_rel = {a: [] for a in arms if a != "pop_slope"}
    boot_delta_ks = []   # kalman − static
    boot_delta_rws = []  # random_walk − static
    boot_delta_kr = []   # kalman − random_walk
    a_static = pu["rmse_pebs_static"].to_numpy()
    a_kalman = pu["rmse_pebs_kalman"].to_numpy()
    a_rw = pu["rmse_pebs_random_walk"].to_numpy()
    rms_pop_arr = pu["rmse_pop_slope"].to_numpy()
    for _ in range(n_boot):
        idx = rng_boot.integers(0, len(pu), size=len(pu))
        for arm in arms:
            boot_rmse[arm].append(float(pu[f"rmse_{arm}"].to_numpy()[idx].mean()))
        pop_mean = rms_pop_arr[idx].mean()
        for arm in arms:
            if arm == "pop_slope":
                continue
            m = pu[f"rmse_{arm}"].to_numpy()[idx].mean()
            boot_rel[arm].append(100 * (pop_mean - m) / pop_mean)
        boot_delta_ks.append(float((a_kalman[idx] - a_static[idx]).mean()))
        boot_delta_rws.append(float((a_rw[idx] - a_static[idx]).mean()))
        boot_delta_kr.append(float((a_kalman[idx] - a_rw[idx]).mean()))

    def ci(arr):
        return [float(np.percentile(arr, 2.5)),
                float(np.percentile(arr, 97.5))]

    print(f"\n=== Bootstrap 95% CI (n_boot={n_boot}, user-resampled) ===")
    print(f"  RMSE means:")
    for arm in arms:
        ci_a = ci(boot_rmse[arm])
        print(f"    {arm:>20}: [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
    print(f"  Rel-improvement vs pop-slope:")
    for arm in arms:
        if arm == "pop_slope":
            continue
        ci_a = ci(boot_rel[arm])
        print(f"    {arm:>20}: [{ci_a[0]:+.2f}%, {ci_a[1]:+.2f}%]")
    print(f"  Δ RMSE (kalman − static):      {ci(boot_delta_ks)}")
    print(f"  Δ RMSE (random_walk − static): {ci(boot_delta_rws)}")
    print(f"  Δ RMSE (kalman − random_walk): {ci(boot_delta_kr)}")

    # ---- β drift diagnostics ----
    print(f"\n=== β_j(t) drift diagnostics ===")
    bv = pu["beta_kalman_var"].to_numpy()
    br = pu["beta_kalman_range"].to_numpy()
    av = pu["alpha_kalman_var"].to_numpy()
    ar_ = pu["alpha_kalman_range"].to_numpy()
    print(f"  Per-user Var[β_j(t)]  (AR1):   mean={bv.mean():.4f}  median={np.median(bv):.4f}  p95={np.percentile(bv,95):.4f}")
    print(f"  Per-user Range[β_j(t)] (AR1):   mean={br.mean():.3f}  median={np.median(br):.3f}  p95={np.percentile(br,95):.3f}")
    print(f"  Per-user Var[α_j(t)]  (AR1):   mean={av.mean():.3f}  median={np.median(av):.3f}  p95={np.percentile(av,95):.3f}")
    print(f"  Per-user Range[α_j(t)] (AR1):   mean={ar_.mean():.2f}  median={np.median(ar_):.2f}  p95={np.percentile(ar_,95):.2f}")
    # Compare beta_kalman_end vs beta_static — how different are endpoints?
    diff = pu["beta_kalman_end"] - pu["beta_static"]
    print(f"  |β_kalman_end − β_static|: mean={diff.abs().mean():.4f}  "
          f"median={diff.abs().median():.4f}  p95={np.percentile(diff.abs(),95):.4f}")
    corr_ab = float(pu[["alpha_kalman_end", "alpha_static"]].corr().iloc[0, 1])
    corr_bb = float(pu[["beta_kalman_end", "beta_static"]].corr().iloc[0, 1])
    print(f"  corr(α_kalman_end, α_static) = {corr_ab:.3f}")
    print(f"  corr(β_kalman_end, β_static) = {corr_bb:.3f}")

    # ---- Verdict ----
    delta_mean_ks = float((a_kalman - a_static).mean())
    ci_ks = ci(boot_delta_ks)
    if delta_mean_ks < 0 and comparisons["kalman_vs_static"]["wilcoxon_p"] < 0.05 and ci_ks[1] < 0:
        verdict = "HELPS: Kalman AR(1) beats static PEBS (significant)"
    elif delta_mean_ks > 0 and comparisons["kalman_vs_static"]["wilcoxon_p"] < 0.05 and ci_ks[0] > 0:
        verdict = "HURTS: Kalman AR(1) worse than static PEBS (significant)"
    else:
        verdict = "NULL: Kalman AR(1) not significantly different from static PEBS"
    print(f"\n=== VERDICT ===\n  {verdict}")

    # ---- save ----
    out = {
        "n_users": int(len(pu)),
        "test_fraction": args.test_fraction,
        "mle_params": {
            "mu0": [pop_alpha, pop_beta],
            "Sigma0_diag": [tau_a2, tau_b2],
            "sigma2_median_ols": sigma2_pop,
            "random_walk": {
                "q_alpha": float(q_a_rw), "q_beta": float(q_b_rw),
                "R": float(R_rw), "nll": nll_rw, "converged": ok_rw,
            },
            "ar1": {
                "rho_alpha": float(rho_a), "rho_beta": float(rho_b),
                "q_alpha": float(q_a), "q_beta": float(q_b), "R": float(R),
                "nll": nll_ar1, "converged": ok_ar1,
            },
            "lr_test_rw_vs_ar1": {
                "delta_nll": float(nll_rw - nll_ar1),
                "df": 2,
                "p_value_approx": float(
                    stats.chi2.sf(2 * max(nll_rw - nll_ar1, 0.0), df=2)
                ),
            },
        },
        "rmse_mean": {a: float(pu[f"rmse_{a}"].mean()) for a in arms},
        "rmse_median": {a: float(pu[f"rmse_{a}"].median()) for a in arms},
        "relative_improvement_vs_pop_pct": rel,
        "comparisons": comparisons,
        "bootstrap_ci_95": {
            "n_bootstrap": n_boot,
            "rmse_mean": {a: ci(boot_rmse[a]) for a in arms},
            "rel_improvement_pct": {a: ci(boot_rel[a]) for a in arms
                                    if a != "pop_slope"},
            "delta_kalman_minus_static": ci(boot_delta_ks),
            "delta_random_walk_minus_static": ci(boot_delta_rws),
            "delta_kalman_minus_random_walk": ci(boot_delta_kr),
        },
        "drift_diagnostics": {
            "beta_var_mean": float(bv.mean()),
            "beta_var_median": float(np.median(bv)),
            "beta_var_p95": float(np.percentile(bv, 95)),
            "beta_range_mean": float(br.mean()),
            "beta_range_median": float(np.median(br)),
            "beta_range_p95": float(np.percentile(br, 95)),
            "alpha_var_mean": float(av.mean()),
            "alpha_var_median": float(np.median(av)),
            "alpha_var_p95": float(np.percentile(av, 95)),
            "alpha_range_mean": float(ar_.mean()),
            "alpha_range_median": float(np.median(ar_)),
            "alpha_range_p95": float(np.percentile(ar_, 95)),
            "mean_abs_beta_end_diff_vs_static": float(diff.abs().mean()),
            "corr_alpha_end_vs_static": corr_ab,
            "corr_beta_end_vs_static": corr_bb,
        },
        "verdict": verdict,
    }
    (out_dir / "eval.json").write_text(json.dumps(out, indent=2))
    pu.to_parquet(out_dir / "per_user.parquet")
    traj_df.to_parquet(out_dir / "beta_trajectories.parquet")
    print(f"\n[save] {out_dir}/eval.json")
    print(f"[save] {out_dir}/per_user.parquet  ({len(pu)} rows)")
    print(f"[save] {out_dir}/beta_trajectories.parquet  ({len(traj_df)} rows)")


if __name__ == "__main__":
    main()
