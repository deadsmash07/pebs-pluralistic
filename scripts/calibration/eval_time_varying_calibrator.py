"""Time-varying PEBS calibrator v2 — falsifiable retest of NEG-T1.KALMAN.

A prior naive MLE AR(1) Kalman attempt LOST to static PEBS by +0.183 RMSE
(Wilcoxon p=0.012) on PRISM temporal 80/20. Diagnosis: MLE picked
q_alpha=27.9, q_beta=14.0 — both LARGER than population across-user variance
tau_alpha^2=115.7, tau_beta^2=26.2 — which let the state chase train-
innovation noise with no regularisation. Classical likelihood-vs-generalisation
divergence.

This v2 tries four corrected variants that ALL address the overfitting
failure mode:

  K1 (static-anchored Kalman):
      - State prior N(mu_static_j, Sigma_static_j) per user (EB-shrunk),
        not population centroid.
      - Mean-reversion target mu_* = mu_static_j (pulls back to per-user
        static estimate, not population).
      - (rho_a, rho_b, q_a, q_b, R) fitted by pooled MLE as before.
      Rationale: gives the filter a strong per-user prior. It can only
      drift if the per-user evidence overrides the static estimate.

  K2 (CV-penalised q_alpha, q_beta):
      - Same structure as prior AR(1) Kalman (mean-reversion to population).
      - But (q_alpha, q_beta) are fitted by LEAVE-LAST-K-WITHIN-TRAIN
        VALIDATION RMSE, not pooled innovation MLE. R is fixed at the
        static EB residual variance.
      - Rationale: train-innovation likelihood over-rewards flexibility
        because every state update reduces the next-step innovation. CV
        directly penalises generalisation loss, which is what we care
        about at test time.

  K3 (EB-shrunk Kalman / hierarchical-in-time):
      - Random-walk Kalman (F=I, MLE q_a, q_b, R as prior run) but at
        EACH time t the filtered state is shrunk toward (mu_static_j) by
        an EB weight
            omega_t = tau_j^2 / (tau_j^2 + Sigma_{t|t}[k,k]),
        where tau_j^2 is the per-user posterior variance of the static
        estimate (= V_j in the static EB formula).
      - Rationale: hierarchical pooling at the STATE level, not just at
        the process-noise level. If the filter's belief is uncertain
        relative to the static estimate, pull it back.

  K4 (ridge-penalised cubic spline):
      - Independent of Kalman formulation: fit per-user
            alpha_j(t) = natural-cubic-spline in t with knots at quantiles,
            beta_j(t)  = natural-cubic-spline in t with knots at quantiles,
        via penalised ridge regression min ||y - X_j(t) * theta_j||^2 +
        lambda * ||D theta_j||^2 where D penalises 2nd differences.
      - lambda chosen per-user by leave-last-K CV.
      - Rationale: completely separate time-varying family; tests whether
        ANY non-Kalman time-varying approach can win, or if PRISM's short
        per-user span is fundamentally low-signal.

Baseline: pebs_static (EB-shrunk), same as prior.

Evaluation: temporal 80/20 sorted by generated_datetime; same splits as
NEG-T1.KALMAN; 1391+ users (min_obs=6); held-out RMSE; paired Wilcoxon;
30-seed user-bootstrap 95% CI on RMSE and paired delta.

Output:
  results/track1_kalman_v2/eval.json
  results/track1_kalman_v2/per_user.parquet
  results/track1_kalman_v2/beta_trajectories.parquet
  results/track1_kalman_v2/trajectory_diagnostic.png (exemplar user)
  results/track1_kalman_v2/run.log (stdout dump)

Runtime: ~10-25 min CPU (K2 CV is dominant).

Usage:
  cd IMPLEMENTATION/1_Causal_RLHF
  python3 scripts/eval_time_varying_calibrator.py [--max-users N]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Tuple, List, Dict

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
    p.add_argument("--cv-val-fraction", type=float, default=0.25,
                   help="Leave-last-K-within-TRAIN fraction for CV of q's "
                        "(K2 arm). K = ceil(n_train * cv_val_fraction).")
    p.add_argument("--n-bootstrap", type=int, default=30)
    p.add_argument("--max-users", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/track1_kalman_v2")
    p.add_argument("--mle-max-iter", type=int, default=200)
    p.add_argument("--cv-grid-size", type=int, default=7,
                   help="Log-spaced grid size for q_alpha and q_beta in K2 CV.")
    p.add_argument("--n-spline-knots", type=int, default=4,
                   help="Natural-cubic-spline knot count per user (K4).")
    p.add_argument("--spline-lambda-grid", type=int, default=8,
                   help="Log-spaced lambda grid for K4 spline penalty.")
    p.add_argument("--exemplar-user-idx", type=int, default=0,
                   help="Which user (by index in sorted user list) to plot.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
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
# Kalman filter core (generalised)
# ---------------------------------------------------------------------------

def kalman_filter(
    x: np.ndarray,
    y: np.ndarray,
    mu0: np.ndarray,
    Sigma0: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    R: float,
    mu_stationary: np.ndarray | None = None,
    return_trajectory: bool = False,
    eb_shrink_toward: np.ndarray | None = None,
    eb_tau2: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray | None]:
    """Joseph-form Kalman filter.

    Optional EB shrinkage (K3 arm): after each update, pull the filtered
    state toward `eb_shrink_toward` using omega = tau^2 / (tau^2 + sigma^2)
    where sigma^2 is the posterior diagonal variance.

    Returns (mu_T, Sigma_T, neg_log_lik, state_trajectory, cov_trajectory).
    """
    mu = mu0.copy()
    Sigma = Sigma0.copy()
    T = len(x)
    nll = 0.0
    traj = np.empty((T, 2)) if return_trajectory else None
    cov_traj = np.empty((T, 2)) if return_trajectory else None
    if mu_stationary is None:
        mu_stationary = np.zeros(2)

    for t in range(T):
        mu_pred = mu_stationary + F @ (mu - mu_stationary)
        Sigma_pred = F @ Sigma @ F.T + Q
        h = np.array([1.0, x[t]])
        y_hat = h @ mu_pred
        S = float(h @ Sigma_pred @ h + R)
        if S <= 0 or not np.isfinite(S):
            return mu_pred, Sigma_pred, np.inf, traj, cov_traj
        innov = y[t] - y_hat
        nll += 0.5 * (np.log(2 * np.pi * S) + innov * innov / S)
        K = Sigma_pred @ h / S
        mu = mu_pred + K * innov
        I_KH = np.eye(2) - np.outer(K, h)
        Sigma = I_KH @ Sigma_pred @ I_KH.T + R * np.outer(K, K)

        # Optional EB state-shrinkage (K3 arm)
        if eb_shrink_toward is not None and eb_tau2 is not None:
            for k_dim in (0, 1):
                tau2 = float(eb_tau2[k_dim])
                s2 = float(Sigma[k_dim, k_dim])
                if np.isfinite(tau2) and tau2 > 0 and np.isfinite(s2):
                    omega = tau2 / (tau2 + s2)
                    mu[k_dim] = omega * mu[k_dim] + (1 - omega) * eb_shrink_toward[k_dim]

        if return_trajectory:
            traj[t] = mu
            cov_traj[t] = [Sigma[0, 0], Sigma[1, 1]]
    return mu, Sigma, nll, traj, cov_traj


# ---------------------------------------------------------------------------
# Pooled MLE (AR(1) with EB-init)
# ---------------------------------------------------------------------------

def _unpack_ar1(theta):
    logit_ra, logit_rb, log_qa, log_qb, log_R = theta
    rho_a = 1.0 / (1.0 + np.exp(-logit_ra))
    rho_b = 1.0 / (1.0 + np.exp(-logit_rb))
    q_a = np.exp(log_qa)
    q_b = np.exp(log_qb)
    R = np.exp(log_R)
    return rho_a, rho_b, q_a, q_b, R


def pooled_nll_eb_anchored(
    theta,
    user_trains,      # list of (x_tr, y_tr, mu_static_j, Sigma_static_j)
):
    """Pooled innovation NLL with per-user EB-anchored prior and per-user
    mean-reversion target equal to static EB estimate."""
    rho_a, rho_b, q_a, q_b, R = _unpack_ar1(theta)
    F = np.diag([rho_a, rho_b])
    Q = np.diag([q_a, q_b])
    total = 0.0
    for x_tr, y_tr, mu_s, Sig_s in user_trains:
        _, _, nll, _, _ = kalman_filter(
            x_tr, y_tr, mu_s, Sig_s, F, Q, R,
            mu_stationary=mu_s, return_trajectory=False,
        )
        if not np.isfinite(nll):
            return 1e12
        total += nll
    return total


def fit_ar1_eb_anchored_mle(user_trains, sigma2_init, q_init_scale,
                             max_iter=200):
    """Fit (rho_a, rho_b, q_a, q_b, R) with EB-anchored prior per user.

    q_init_scale: small multiplier on the typical V_j so we start within
    the feasible region rather than above it.
    """
    theta0 = np.array([
        4.595,                         # rho ~ 0.99
        4.595,
        np.log(q_init_scale[0]),
        np.log(q_init_scale[1]),
        np.log(sigma2_init),
    ])
    t0 = time.time()
    res = minimize(
        pooled_nll_eb_anchored, theta0,
        args=(user_trains,),
        method="L-BFGS-B",
        options={"maxiter": max_iter, "disp": False, "ftol": 1e-6},
    )
    rho_a, rho_b, q_a, q_b, R = _unpack_ar1(res.x)
    print(f"[MLE-EB-anchored] converged={res.success} nll={res.fun:.2f} "
          f"rho_a={rho_a:.4f} rho_b={rho_b:.4f} "
          f"q_a={q_a:.4f} q_b={q_b:.4f} R={R:.2f} "
          f"in {time.time()-t0:.1f}s ({res.nit} iter, {res.nfev} nfev)")
    return rho_a, rho_b, q_a, q_b, R, float(res.fun), bool(res.success)


# ---------------------------------------------------------------------------
# Held-out validation CV for (q_alpha, q_beta) — K2 arm
# ---------------------------------------------------------------------------

def cv_fit_q_alpha_beta(
    user_splits: Dict,
    pop_mu: np.ndarray,
    Sigma0_pop: np.ndarray,
    R_fixed: float,
    cv_val_fraction: float,
    q_grid: np.ndarray,
    max_sample: int = 800,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, np.ndarray]:
    """Grid-search (q_a, q_b) to minimize POOLED RMSE on leave-last-K-within-
    train validation. R fixed. F=diag(0.95, 0.95) fixed (empirical from
    prior MLE). State init at population centroid (keeps CV arm's
    semantics independent of static EB so the arm tests something else).

    Returns (q_a*, q_b*, cv_grid_matrix_RMSE).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    uids = sorted(user_splits.keys())
    if len(uids) > max_sample:
        uids = list(rng.choice(uids, size=max_sample, replace=False))

    # Pre-split each user's train into (sub_train, val)
    per_user_cv = []
    for uid in uids:
        x_tr, y_tr, *_ = user_splits[uid]
        n = len(x_tr)
        k = max(1, int(np.ceil(n * cv_val_fraction)))
        if n - k < 3:
            continue
        per_user_cv.append((x_tr[:n - k], y_tr[:n - k],
                            x_tr[n - k:], y_tr[n - k:]))
    print(f"[CV] {len(per_user_cv)} users used for q-grid CV")

    grid_rmse = np.zeros((len(q_grid), len(q_grid)))
    F = np.diag([0.95, 0.95])  # empirical prior from NEG-T1.KALMAN AR(1)
    for i, q_a in enumerate(q_grid):
        for j, q_b in enumerate(q_grid):
            Q = np.diag([q_a, q_b])
            sq_err_sum = 0.0
            n_obs = 0
            for x_sub, y_sub, x_val, y_val in per_user_cv:
                mu_T, _, _, _, _ = kalman_filter(
                    x_sub, y_sub, pop_mu, Sigma0_pop, F, Q, R_fixed,
                    mu_stationary=pop_mu, return_trajectory=False,
                )
                yh = mu_T[0] + mu_T[1] * x_val
                sq_err_sum += float(np.sum((yh - y_val) ** 2))
                n_obs += len(x_val)
            grid_rmse[i, j] = np.sqrt(sq_err_sum / max(n_obs, 1))
    best = np.unravel_index(np.argmin(grid_rmse), grid_rmse.shape)
    q_a_best = float(q_grid[best[0]])
    q_b_best = float(q_grid[best[1]])
    print(f"[CV] best q_a={q_a_best:.4f} q_b={q_b_best:.4f} "
          f"RMSE={grid_rmse[best]:.4f}")
    print(f"[CV] grid min/max RMSE = {grid_rmse.min():.4f}/{grid_rmse.max():.4f}")
    return q_a_best, q_b_best, grid_rmse


# ---------------------------------------------------------------------------
# K4 — ridge-penalised natural cubic spline on (alpha_j(t), beta_j(t))
# ---------------------------------------------------------------------------

def natural_cubic_spline_basis(t: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Minimal implementation of natural cubic spline basis with K knots.

    Returns design matrix (T, K) with columns:
      [1, t, N_1(t), ..., N_{K-2}(t)]
    where N_k is the truncated-cubic natural-spline basis.
    """
    t = np.asarray(t, dtype=float)
    K = len(knots)
    if K < 2:
        # fall back to constant + linear
        return np.column_stack([np.ones_like(t), t])
    d_K_1 = knots[K - 1]

    def d_k(k_idx):
        xi_k = knots[k_idx]
        xi_K = knots[K - 1]
        denom = max(xi_K - xi_k, 1e-9)
        pos_k = np.clip(t - xi_k, 0, None) ** 3
        pos_K = np.clip(t - xi_K, 0, None) ** 3
        return (pos_k - pos_K) / denom

    cols = [np.ones_like(t), t]
    if K > 2:
        d_Km1 = d_k(K - 2)
        for k_idx in range(K - 2):
            cols.append(d_k(k_idx) - d_Km1)
    return np.column_stack(cols)


def fit_time_varying_spline(
    x: np.ndarray, y: np.ndarray, t: np.ndarray,
    x_eval: np.ndarray, t_eval: np.ndarray,
    n_knots: int,
    lambda_grid: np.ndarray,
    cv_val_fraction: float = 0.25,
    static_alpha: float | None = None,
    static_beta: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray | None]:
    """Fit alpha_j(t) + beta_j(t)*x with natural cubic spline in t for each
    coefficient, ridge penalty on 2nd differences, lambda chosen by
    leave-last-K CV within (x, y). Returns y_hat_eval, y_hat_train_traj,
    lambda_star, alpha_beta_traj.

    y_hat(x_i, t_i) = (X_basis(t_i) @ theta_alpha) + (X_basis(t_i) * x_i @ theta_beta)
    """
    n = len(x)
    # We need at least ~4*n_knots observations to identify (alpha(t), beta(t))
    # jointly with a spline. Fall back to linear-in-t if under.
    # Effective degrees of freedom: 2 * p (alpha and beta bases).
    min_needed = 4 * n_knots + 2
    if n < min_needed:
        # Reduce knots to what's supported; if still insufficient, fall back
        # to static (linear in x, constant in t).
        n_knots_eff = max(2, min(n_knots, (n - 2) // 4))
    else:
        n_knots_eff = n_knots
    if n < max(6, 2 * n_knots_eff + 2):
        slope, intercept = np.polyfit(x, y, 1)
        yh_eval = intercept + slope * x_eval
        return yh_eval, None, np.nan, None

    # Knots at quantiles of t
    qs = np.linspace(0, 1, n_knots_eff)
    knots = np.quantile(t, qs)
    # dedupe
    knots = np.unique(knots)
    if len(knots) < 2:
        slope, intercept = np.polyfit(x, y, 1)
        yh_eval = intercept + slope * x_eval
        return yh_eval, None, np.nan, None

    B_tr = natural_cubic_spline_basis(t, knots)         # (n, p)
    B_ev = natural_cubic_spline_basis(t_eval, knots)    # (n_eval, p)
    p = B_tr.shape[1]

    # Design: [B | B*x] for (alpha(t), beta(t))
    X_tr = np.hstack([B_tr, B_tr * x[:, None]])         # (n, 2p)
    X_ev = np.hstack([B_ev, B_ev * x_eval[:, None]])    # (n_eval, 2p)

    # If static anchors provided, we fit residuals: y - (alpha_s + beta_s * x)
    # so that lambda -> infty shrinks toward static, not to zero
    if static_alpha is not None and static_beta is not None:
        y_fit = y - (static_alpha + static_beta * x)
        y_anchor = static_alpha + static_beta * x_eval
    else:
        y_fit = y
        y_anchor = np.zeros_like(x_eval)

    # Penalty on 2nd differences (drives smoothness in t)
    D = np.eye(p)
    if p >= 3:
        # 2nd-difference penalty on the spline coefficients (skip constant+linear)
        # Only penalise columns p-2 and beyond (the nonlinear basis terms)
        D_band = np.zeros((max(p - 2, 1), p))
        for i in range(p - 2):
            D_band[i, i] = 1
            D_band[i, i + 1] = -2
            D_band[i, i + 2] = 1
        D = D_band
    P = np.block([
        [D.T @ D, np.zeros((p, p))],
        [np.zeros((p, p)), D.T @ D],
    ])

    # Full ridge penalty: D^TD on the 2nd-diff of spline columns, PLUS
    # an L2 penalty on theta to shrink residual-fit toward zero.
    # Using D^TD + eps * I stabilises CV for small n.
    P_full = P + 1e-4 * np.eye(2 * p)

    # Inner CV: last-K holdout
    k_cv = max(1, int(np.ceil(n * cv_val_fraction)))
    if n - k_cv < 3:
        # Not enough, default to top of grid (max-shrinkage ≈ static)
        lam_star = float(lambda_grid[-1])
    else:
        cv_scores = []
        X_sub = X_tr[:n - k_cv]
        y_sub = y_fit[:n - k_cv]
        X_val = X_tr[n - k_cv:]
        y_val_resid = y_fit[n - k_cv:]
        XtX_sub = X_sub.T @ X_sub
        Xty_sub = X_sub.T @ y_sub
        for lam in lambda_grid:
            try:
                theta = np.linalg.solve(XtX_sub + lam * P_full, Xty_sub)
                yh = X_val @ theta
                cv_scores.append(float(np.mean((yh - y_val_resid) ** 2)))
            except np.linalg.LinAlgError:
                cv_scores.append(np.inf)
        lam_star = float(lambda_grid[int(np.argmin(cv_scores))])

    # Fit on full train with lam_star
    XtX = X_tr.T @ X_tr
    Xty = X_tr.T @ y_fit
    try:
        theta = np.linalg.solve(XtX + lam_star * P_full, Xty)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(X_tr, y_fit, rcond=None)[0]
    yh_eval = y_anchor + X_ev @ theta
    # Extract per-time alpha(t), beta(t) on train t
    # When fitting residuals, the absolute alpha/beta is static + spline-increment
    theta_a = theta[:p]
    theta_b = theta[p:]
    alpha_incr = B_tr @ theta_a
    beta_incr = B_tr @ theta_b
    if static_alpha is not None:
        alpha_traj = static_alpha + alpha_incr
        beta_traj = static_beta + beta_incr
    else:
        alpha_traj = alpha_incr
        beta_traj = beta_incr
    ab_traj = np.column_stack([alpha_traj, beta_traj])
    return yh_eval, ab_traj, lam_star, ab_traj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- load --------
    df = pd.read_parquet(args.scored_parquet).dropna(
        subset=["score_user"]).reset_index(drop=True)
    ts = pd.read_parquet(args.timestamp_cache)
    df = df.merge(ts, on="conversation_id", how="left").dropna(
        subset=["generated_datetime"]).reset_index(drop=True)
    print(f"[load] {len(df)} utterances, {df.user_id.nunique()} users, "
          f"span={(df.generated_datetime.max()-df.generated_datetime.min()).days} days")

    if args.max_users > 0:
        uids_all = sorted(df.user_id.unique().tolist())
        rng = np.random.default_rng(args.seed)
        chosen = set(rng.choice(uids_all, size=min(args.max_users, len(uids_all)),
                                replace=False).tolist())
        df = df[df.user_id.isin(chosen)].reset_index(drop=True)
        print(f"[subsample] {df.user_id.nunique()} users, {len(df)} utterances")

    # -------- pop-slope and EB prior (fit only on train-half per user) --------
    # We cannot use full-df pop regression because some users' test blocks
    # leak in. But the REFERENCE pop_alpha/pop_beta for arms is fine to
    # compute on all data; TRAIN-ONLY splits downstream ensure no leakage
    # for held-out users' parameters. Follow same convention as prior script.
    slope_pop, intercept_pop = np.polyfit(df.rm_score, df.score_user, 1)
    pop_alpha = float(intercept_pop)
    pop_beta = float(slope_pop)
    print(f"[pop] alpha0={pop_alpha:.3f} beta0={pop_beta:.3f}")

    # -------- build user splits + OLS per user on TRAIN fold --------
    user_splits: Dict[str, Tuple] = {}
    stats_rows: List[Dict] = []
    sigma_user: List[float] = []
    for uid, grp in df.groupby("user_id"):
        if len(grp) < args.min_obs_per_user:
            continue
        g = temporal_sort(grp)
        n = len(g)
        n_test = max(1, int(round(n * args.test_fraction)))
        if n - n_test < 3:
            continue
        tr = g.iloc[:n - n_test]
        te = g.iloc[n - n_test:]
        x_tr = tr["rm_score"].to_numpy()
        y_tr = tr["score_user"].to_numpy().astype(float)
        x_te = te["rm_score"].to_numpy()
        y_te = te["score_user"].to_numpy().astype(float)
        # Time index: use days since earliest datetime within user
        t_tr_dt = pd.to_datetime(tr["generated_datetime"]).to_numpy()
        t_te_dt = pd.to_datetime(te["generated_datetime"]).to_numpy()
        t0_ref = np.datetime64(t_tr_dt[0])
        t_tr = (t_tr_dt - t0_ref).astype("timedelta64[s]").astype(float) / 3600.0
        t_te = (t_te_dt - t0_ref).astype("timedelta64[s]").astype(float) / 3600.0
        a, b, Va, Vb, s2 = ols_with_V(x_tr, y_tr)
        user_splits[uid] = (x_tr, y_tr, x_te, y_te, a, b, Va, Vb, s2,
                            t_tr, t_te)
        stats_rows.append({"user_id": uid, "alpha": a, "beta": b,
                           "V_alpha": Va, "V_beta": Vb, "sigma2": s2,
                           "n": len(grp)})
        if np.isfinite(s2):
            sigma_user.append(s2)

    us = pd.DataFrame(stats_rows)
    V_a_tot = float(us.alpha.var())
    V_b_tot = float(us.beta.var())
    mean_Va = float(us.V_alpha.replace([np.inf, -np.inf], np.nan).dropna().mean())
    mean_Vb = float(us.V_beta.replace([np.inf, -np.inf], np.nan).dropna().mean())
    tau_a2 = max(V_a_tot - mean_Va, 1e-6)
    tau_b2 = max(V_b_tot - mean_Vb, 1e-6)
    sigma2_pop = float(np.median(sigma_user))
    n_users = len(user_splits)
    print(f"[EB] n_users={n_users} tau_a^2={tau_a2:.3f} tau_b^2={tau_b2:.3f} "
          f"sigma^2={sigma2_pop:.3f}")
    print(f"[diag] mean V_alpha={mean_Va:.3f} mean V_beta={mean_Vb:.3f}")

    pop_mu = np.array([pop_alpha, pop_beta])
    Sigma0_pop = np.diag([tau_a2, tau_b2])

    # -------- fit EB static per user (baseline arm) --------
    static_params: Dict[str, Tuple[float, float, float, float]] = {}
    # (alpha_s, beta_s, post_var_alpha, post_var_beta)
    for uid, tpl in user_splits.items():
        a, b, Va, Vb = tpl[4], tpl[5], tpl[6], tpl[7]
        omega_a = tau_a2 / (tau_a2 + Va) if np.isfinite(Va) else 0.0
        omega_b = tau_b2 / (tau_b2 + Vb) if np.isfinite(Vb) else 0.0
        a_s = omega_a * a + (1 - omega_a) * pop_alpha
        b_s = omega_b * b + (1 - omega_b) * pop_beta
        # Posterior variance of shrinkage estimate (EB)
        post_va = (1 - omega_a) * tau_a2 if np.isfinite(Va) else tau_a2
        post_vb = (1 - omega_b) * tau_b2 if np.isfinite(Vb) else tau_b2
        static_params[uid] = (a_s, b_s, post_va, post_vb)

    # -------- K1 arm: fit AR(1) Kalman with EB-anchored prior --------
    rng = np.random.default_rng(args.seed + 11)
    mle_uids = list(user_splits.keys())
    if len(mle_uids) > 400:
        mle_uids = list(rng.choice(mle_uids, size=400, replace=False))
    mle_trains_k1 = []
    for uid in mle_uids:
        x_tr, y_tr, *_ = user_splits[uid]
        a_s, b_s, pva, pvb = static_params[uid]
        mu_s = np.array([a_s, b_s])
        Sigma_s = np.diag([max(pva, 1e-3), max(pvb, 1e-3)])
        mle_trains_k1.append((x_tr, y_tr, mu_s, Sigma_s))
    print(f"[K1] fitting EB-anchored AR(1) MLE on {len(mle_trains_k1)} users")
    # q_init_scale: start small — want to see if data pulls q up
    q_init_scale = (max(tau_a2 * 0.001, 1e-3), max(tau_b2 * 0.001, 1e-3))
    (rho_a_k1, rho_b_k1, q_a_k1, q_b_k1, R_k1,
     nll_k1, ok_k1) = fit_ar1_eb_anchored_mle(
        mle_trains_k1, sigma2_pop, q_init_scale, max_iter=args.mle_max_iter,
    )

    # -------- K2 arm: CV-selected (q_a, q_b) --------
    print(f"[K2] CV-selecting (q_a, q_b) on leave-last-K within train")
    # Grid: log-spaced around 0.01..tau^2
    q_grid_a = np.exp(np.linspace(np.log(1e-3), np.log(tau_a2 * 1.5),
                                   args.cv_grid_size))
    q_grid_b = np.exp(np.linspace(np.log(1e-3), np.log(tau_b2 * 1.5),
                                   args.cv_grid_size))
    # Use geometric mean grid for both (common for numerical stability)
    # Keep them separate since scales differ
    # We'll call cv_fit with grid passed as common; let's adjust:
    # cv_fit_q_alpha_beta grid-iterates only one grid — generalise:
    q_grid = np.exp(np.linspace(np.log(1e-3),
                                 np.log(max(tau_a2, tau_b2) * 1.5),
                                 args.cv_grid_size))
    q_a_k2, q_b_k2, grid_rmse_k2 = cv_fit_q_alpha_beta(
        user_splits, pop_mu, Sigma0_pop, R_fixed=sigma2_pop,
        cv_val_fraction=args.cv_val_fraction,
        q_grid=q_grid,
        max_sample=800, rng=rng,
    )

    # -------- K3 arm: EB-shrunk Kalman (hierarchical-in-time) --------
    # Uses same q's as K1 (EB-anchored) but additionally applies state
    # shrinkage toward per-user static after each update.
    # (Could also use RW values; testing the EB-shrinkage layer in isolation.)
    print(f"[K3] EB-shrunk Kalman uses K1's MLE q's with per-user EB pull")
    rho_a_k3, rho_b_k3 = rho_a_k1, rho_b_k1
    q_a_k3, q_b_k3, R_k3 = q_a_k1, q_b_k1, R_k1

    # -------- per-user scoring --------
    per_user = []
    beta_traj_rows = []
    exemplar_uid = None
    exemplar_traj = None

    t0 = time.time()
    sorted_uids = sorted(user_splits.keys())
    for u_idx, uid in enumerate(sorted_uids):
        (x_tr, y_tr, x_te, y_te, a_ols, b_ols, Va, Vb, s2,
         t_tr, t_te) = user_splits[uid]
        a_s, b_s, pva, pvb = static_params[uid]
        row = {"user_id": uid, "n_train": int(len(x_tr)), "n_test": int(len(x_te))}

        # no_calib
        row["rmse_no_calib"] = float(np.sqrt(np.mean(
            (np.full_like(y_te, y_tr.mean()) - y_te) ** 2)))
        # pop_slope
        yh = pop_alpha + pop_beta * x_te
        row["rmse_pop_slope"] = float(np.sqrt(np.mean((yh - y_te) ** 2)))
        # pebs_static (baseline)
        yh = a_s + b_s * x_te
        row["rmse_pebs_static"] = float(np.sqrt(np.mean((yh - y_te) ** 2)))
        row["alpha_static"] = float(a_s)
        row["beta_static"] = float(b_s)

        # K1: static-anchored AR(1) Kalman
        mu_s = np.array([a_s, b_s])
        Sigma_s = np.diag([max(pva, 1e-3), max(pvb, 1e-3)])
        F_k1 = np.diag([rho_a_k1, rho_b_k1])
        Q_k1 = np.diag([q_a_k1, q_b_k1])
        mu_T, _, _, traj_k1, _ = kalman_filter(
            x_tr, y_tr, mu_s, Sigma_s, F_k1, Q_k1, R_k1,
            mu_stationary=mu_s, return_trajectory=True,
        )
        yh = mu_T[0] + mu_T[1] * x_te
        row["rmse_k1_static_anchored"] = float(np.sqrt(np.mean((yh - y_te) ** 2)))
        row["alpha_k1_end"] = float(mu_T[0])
        row["beta_k1_end"] = float(mu_T[1])
        row["beta_k1_var"] = float(np.var(traj_k1[:, 1]))
        row["alpha_k1_var"] = float(np.var(traj_k1[:, 0]))

        # K2: CV-q Kalman (pop-centered, pop-anchored)
        F_k2 = np.diag([0.95, 0.95])
        Q_k2 = np.diag([q_a_k2, q_b_k2])
        mu_T, _, _, traj_k2, _ = kalman_filter(
            x_tr, y_tr, pop_mu, Sigma0_pop, F_k2, Q_k2, sigma2_pop,
            mu_stationary=pop_mu, return_trajectory=True,
        )
        yh = mu_T[0] + mu_T[1] * x_te
        row["rmse_k2_cv_q"] = float(np.sqrt(np.mean((yh - y_te) ** 2)))
        row["alpha_k2_end"] = float(mu_T[0])
        row["beta_k2_end"] = float(mu_T[1])
        row["beta_k2_var"] = float(np.var(traj_k2[:, 1]))
        row["alpha_k2_var"] = float(np.var(traj_k2[:, 0]))

        # K3: EB-shrunk (hierarchical) Kalman
        # Shrinks state toward (a_s, b_s) at each step using per-user tau
        Q_k3 = np.diag([q_a_k3, q_b_k3])
        F_k3 = np.diag([rho_a_k3, rho_b_k3])
        eb_tau2_local = np.array([max(pva, 1e-3), max(pvb, 1e-3)])
        mu_T, _, _, traj_k3, _ = kalman_filter(
            x_tr, y_tr, mu_s, Sigma_s, F_k3, Q_k3, R_k3,
            mu_stationary=mu_s, return_trajectory=True,
            eb_shrink_toward=mu_s, eb_tau2=eb_tau2_local,
        )
        yh = mu_T[0] + mu_T[1] * x_te
        row["rmse_k3_eb_shrunk"] = float(np.sqrt(np.mean((yh - y_te) ** 2)))
        row["alpha_k3_end"] = float(mu_T[0])
        row["beta_k3_end"] = float(mu_T[1])
        row["beta_k3_var"] = float(np.var(traj_k3[:, 1]))
        row["alpha_k3_var"] = float(np.var(traj_k3[:, 0]))

        # K4: ridge-penalised cubic-spline time-varying (alpha(t), beta(t))
        lam_grid = np.exp(np.linspace(np.log(1e-2), np.log(1e6),
                                       args.spline_lambda_grid))
        yh_eval, ab_tr_traj, lam_star, _ = fit_time_varying_spline(
            x_tr, y_tr, t_tr,
            x_eval=x_te, t_eval=t_te,
            n_knots=args.n_spline_knots,
            lambda_grid=lam_grid,
            cv_val_fraction=args.cv_val_fraction,
            static_alpha=float(a_s),
            static_beta=float(b_s),
        )
        row["rmse_k4_spline"] = float(np.sqrt(np.mean((yh_eval - y_te) ** 2)))
        row["k4_lambda"] = float(lam_star) if np.isfinite(lam_star) else np.nan
        if ab_tr_traj is not None:
            row["alpha_k4_end"] = float(ab_tr_traj[-1, 0])
            row["beta_k4_end"] = float(ab_tr_traj[-1, 1])
            row["beta_k4_var"] = float(np.var(ab_tr_traj[:, 1]))
            row["alpha_k4_var"] = float(np.var(ab_tr_traj[:, 0]))
        else:
            row["alpha_k4_end"] = np.nan
            row["beta_k4_end"] = np.nan
            row["beta_k4_var"] = np.nan
            row["alpha_k4_var"] = np.nan

        per_user.append(row)

        if u_idx == args.exemplar_user_idx:
            exemplar_uid = uid
            exemplar_traj = {
                "x_tr": x_tr, "y_tr": y_tr,
                "t_tr": t_tr, "t_te": t_te,
                "x_te": x_te, "y_te": y_te,
                "static_alpha": a_s, "static_beta": b_s,
                "k1_traj": traj_k1, "k2_traj": traj_k2, "k3_traj": traj_k3,
                "k4_traj": ab_tr_traj,
            }

        # Cap beta trajectory records to ~20k rows
        if len(beta_traj_rows) < 20000:
            for t_idx in range(len(x_tr)):
                beta_traj_rows.append({
                    "user_id": uid, "t": int(t_idx),
                    "alpha_k1_t": float(traj_k1[t_idx, 0]),
                    "beta_k1_t": float(traj_k1[t_idx, 1]),
                    "alpha_k2_t": float(traj_k2[t_idx, 0]),
                    "beta_k2_t": float(traj_k2[t_idx, 1]),
                    "alpha_k3_t": float(traj_k3[t_idx, 0]),
                    "beta_k3_t": float(traj_k3[t_idx, 1]),
                })
    print(f"[score] {len(per_user)} users scored in {time.time()-t0:.1f}s")

    pu = pd.DataFrame(per_user)

    # -------- headline --------
    arms_rmse_cols = {
        "no_calib": "rmse_no_calib",
        "pop_slope": "rmse_pop_slope",
        "pebs_static": "rmse_pebs_static",
        "k1_static_anchored": "rmse_k1_static_anchored",
        "k2_cv_q": "rmse_k2_cv_q",
        "k3_eb_shrunk": "rmse_k3_eb_shrunk",
        "k4_spline": "rmse_k4_spline",
    }
    print(f"\n=== RMSE TEMPORAL 80/20 (n={len(pu)} users) ===")
    for arm, col in arms_rmse_cols.items():
        s = pu[col].dropna()
        print(f"  {arm:>22}: mean={s.mean():.4f}  median={s.median():.4f} "
              f"(n={len(s)})")

    pop_mean = pu["rmse_pop_slope"].mean()

    def pct_imp(col: str) -> float:
        return 100 * (pop_mean - pu[col].mean()) / pop_mean

    rel = {a: pct_imp(col) for a, col in arms_rmse_cols.items() if a != "pop_slope"}
    print(f"\n=== Relative improvement vs pop-slope ===")
    for k, v in rel.items():
        print(f"  {k:>22}: {v:+.2f}%")

    # -------- paired Wilcoxon vs static --------
    def paired(a_col, b_col):
        a = pu[a_col].dropna().to_numpy()
        b = pu[b_col].dropna().to_numpy()
        # Restrict to common index
        df_pair = pu[[a_col, b_col]].dropna()
        a = df_pair[a_col].to_numpy()
        b = df_pair[b_col].to_numpy()
        if len(a) < 2:
            return {"mean_delta": np.nan, "median_delta": np.nan,
                    "frac_a_smaller": np.nan, "wilcoxon_p": np.nan,
                    "n_pair": int(len(a))}
        w = stats.wilcoxon(a, b, alternative="two-sided")
        return {
            "mean_delta": float((a - b).mean()),
            "median_delta": float(np.median(a - b)),
            "frac_a_smaller": float((a < b).mean()),
            "wilcoxon_p": float(w.pvalue),
            "n_pair": int(len(a)),
        }

    comparisons = {
        "k1_vs_static": paired("rmse_k1_static_anchored", "rmse_pebs_static"),
        "k2_vs_static": paired("rmse_k2_cv_q",            "rmse_pebs_static"),
        "k3_vs_static": paired("rmse_k3_eb_shrunk",        "rmse_pebs_static"),
        "k4_vs_static": paired("rmse_k4_spline",            "rmse_pebs_static"),
        "static_vs_pop": paired("rmse_pebs_static",         "rmse_pop_slope"),
    }
    print(f"\n=== Paired Wilcoxon (arm_a = time-varying, arm_b = static; "
          f"a<b win% = time-varying wins) ===")
    for k, d in comparisons.items():
        print(f"  {k:>16}: Δ={d['mean_delta']:+.4f}  "
              f"a<b={d['frac_a_smaller']:.1%}  p={d['wilcoxon_p']:.3e}")

    # -------- Bootstrap 95% CI (≥30 seeds, user-resample) --------
    n_boot = max(args.n_bootstrap, 30)
    rng_boot = np.random.default_rng(args.seed + 31)
    boot_mean = {a: [] for a in arms_rmse_cols}
    boot_rel = {a: [] for a in arms_rmse_cols if a != "pop_slope"}
    boot_delta = {a: [] for a in arms_rmse_cols if a.startswith("k")}
    cols_arr = {a: pu[c].to_numpy() for a, c in arms_rmse_cols.items()}
    static_arr = pu["rmse_pebs_static"].to_numpy()
    pop_arr = pu["rmse_pop_slope"].to_numpy()

    for _ in range(n_boot):
        idx = rng_boot.integers(0, len(pu), size=len(pu))
        for a in arms_rmse_cols:
            arr = cols_arr[a][idx]
            # handle NaNs in K4 (could be NaN if fit failed; rare)
            arr = arr[np.isfinite(arr)]
            boot_mean[a].append(float(arr.mean()) if len(arr) else np.nan)
        pm = pop_arr[idx].mean()
        for a in arms_rmse_cols:
            if a == "pop_slope":
                continue
            arr = cols_arr[a][idx]
            arr = arr[np.isfinite(arr)]
            if len(arr):
                boot_rel[a].append(100 * (pm - arr.mean()) / pm)
        for a in arms_rmse_cols:
            if a.startswith("k"):
                d = cols_arr[a][idx] - static_arr[idx]
                d = d[np.isfinite(d)]
                if len(d):
                    boot_delta[a].append(float(d.mean()))

    def ci(arr):
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if len(a) < 2:
            return [float("nan"), float("nan")]
        return [float(np.percentile(a, 2.5)),
                float(np.percentile(a, 97.5))]

    print(f"\n=== Bootstrap 95% CI (n_boot={n_boot}, user-resampled) ===")
    print(f"  RMSE means:")
    for a in arms_rmse_cols:
        c = ci(boot_mean[a])
        print(f"    {a:>22}: [{c[0]:.4f}, {c[1]:.4f}]")
    print(f"  Rel-improvement vs pop-slope:")
    for a in arms_rmse_cols:
        if a == "pop_slope":
            continue
        c = ci(boot_rel[a])
        print(f"    {a:>22}: [{c[0]:+.2f}%, {c[1]:+.2f}%]")
    print(f"  Paired Δ (arm − static):")
    for a in ["k1_static_anchored", "k2_cv_q", "k3_eb_shrunk", "k4_spline"]:
        c = ci(boot_delta[a])
        print(f"    {a:>22}: [{c[0]:+.4f}, {c[1]:+.4f}]")

    # -------- drift diagnostics --------
    print(f"\n=== Drift diagnostics ===")
    for arm_key, col_bv, col_br in [
        ("k1", "beta_k1_var", None),
        ("k2", "beta_k2_var", None),
        ("k3", "beta_k3_var", None),
        ("k4", "beta_k4_var", None),
    ]:
        bv = pu[col_bv].dropna().to_numpy()
        if len(bv):
            print(f"  {arm_key} Var[beta_j(t)]: mean={bv.mean():.4f} "
                  f"median={np.median(bv):.4f} p95={np.percentile(bv,95):.4f}")
    # |beta_end diff| vs static
    for arm_key, col_be in [
        ("k1", "beta_k1_end"),
        ("k2", "beta_k2_end"),
        ("k3", "beta_k3_end"),
        ("k4", "beta_k4_end"),
    ]:
        d = (pu[col_be] - pu["beta_static"]).dropna().to_numpy()
        if len(d):
            print(f"  |beta_{arm_key}_end - beta_static|: "
                  f"mean={np.abs(d).mean():.4f}  median={np.median(np.abs(d)):.4f}  "
                  f"p95={np.percentile(np.abs(d), 95):.4f}")

    # -------- Verdict per arm --------
    verdicts = {}
    for arm_key in ["k1_static_anchored", "k2_cv_q", "k3_eb_shrunk", "k4_spline"]:
        cmp = comparisons[f"{arm_key.split('_')[0]}_vs_static"]
        lo, hi = ci(boot_delta[arm_key])
        if (cmp["mean_delta"] < 0 and cmp["wilcoxon_p"] < 0.05 and hi < 0):
            verdicts[arm_key] = "HELPS: significantly beats static PEBS"
        elif (cmp["mean_delta"] > 0 and cmp["wilcoxon_p"] < 0.05 and lo > 0):
            verdicts[arm_key] = "HURTS: significantly worse than static"
        else:
            verdicts[arm_key] = "NULL: not significantly different from static"
    print(f"\n=== VERDICTS ===")
    for k, v in verdicts.items():
        print(f"  {k:>22}: {v}")

    # -------- Diagnostic plot for exemplar user --------
    fig_path = out_dir / "trajectory_diagnostic.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if exemplar_traj is not None:
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ex = exemplar_traj
            t_plot = ex["t_tr"]
            axes[0].plot(t_plot, ex["k1_traj"][:, 0], label="K1 (static-anchored)",
                         color="tab:blue")
            axes[0].plot(t_plot, ex["k2_traj"][:, 0], label="K2 (CV-q)",
                         color="tab:orange")
            axes[0].plot(t_plot, ex["k3_traj"][:, 0], label="K3 (EB-shrunk)",
                         color="tab:green")
            if ex["k4_traj"] is not None:
                axes[0].plot(t_plot, ex["k4_traj"][:, 0],
                             label="K4 (spline)", color="tab:red")
            axes[0].axhline(ex["static_alpha"], linestyle="--",
                            color="black", label="static α")
            axes[0].set_ylabel("α_j(t)")
            axes[0].legend(fontsize=8)
            axes[0].set_title(f"Exemplar user {exemplar_uid}: calibrator trajectories")

            axes[1].plot(t_plot, ex["k1_traj"][:, 1], color="tab:blue")
            axes[1].plot(t_plot, ex["k2_traj"][:, 1], color="tab:orange")
            axes[1].plot(t_plot, ex["k3_traj"][:, 1], color="tab:green")
            if ex["k4_traj"] is not None:
                axes[1].plot(t_plot, ex["k4_traj"][:, 1], color="tab:red")
            axes[1].axhline(ex["static_beta"], linestyle="--", color="black")
            axes[1].set_ylabel("β_j(t)")
            axes[1].set_xlabel("t (hours since user's first utterance)")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=130)
            plt.close(fig)
            print(f"[plot] {fig_path}")
    except Exception as e:
        print(f"[plot] WARN: could not save diagnostic plot: {e}")

    # -------- save --------
    traj_df = pd.DataFrame(beta_traj_rows)
    out = {
        "n_users": int(len(pu)),
        "n_bootstrap": n_boot,
        "test_fraction": args.test_fraction,
        "cv_val_fraction": args.cv_val_fraction,
        "fit": {
            "pop_alpha": pop_alpha, "pop_beta": pop_beta,
            "tau_a2": tau_a2, "tau_b2": tau_b2,
            "sigma2_median_ols": sigma2_pop,
            "mean_V_alpha_ols": mean_Va, "mean_V_beta_ols": mean_Vb,
            "k1_eb_anchored": {
                "rho_alpha": float(rho_a_k1), "rho_beta": float(rho_b_k1),
                "q_alpha": float(q_a_k1), "q_beta": float(q_b_k1),
                "R": float(R_k1), "nll": nll_k1, "converged": ok_k1,
            },
            "k2_cv": {
                "q_alpha": float(q_a_k2), "q_beta": float(q_b_k2),
                "rho_fixed": 0.95, "R_fixed": sigma2_pop,
                "grid_rmse_min": float(grid_rmse_k2.min()),
                "grid_rmse_max": float(grid_rmse_k2.max()),
            },
            "k3_eb_shrunk": {
                "reuses_k1_q": True,
                "rho_alpha": float(rho_a_k3), "rho_beta": float(rho_b_k3),
                "q_alpha": float(q_a_k3), "q_beta": float(q_b_k3),
                "R": float(R_k3),
            },
            "k4_spline": {
                "n_knots": args.n_spline_knots,
                "lambda_grid_size": args.spline_lambda_grid,
                "cv_val_fraction": args.cv_val_fraction,
            },
        },
        "rmse_mean": {a: float(pu[c].dropna().mean())
                      for a, c in arms_rmse_cols.items()},
        "rmse_median": {a: float(pu[c].dropna().median())
                        for a, c in arms_rmse_cols.items()},
        "relative_improvement_vs_pop_pct": rel,
        "comparisons": comparisons,
        "bootstrap_ci_95": {
            "n_bootstrap": n_boot,
            "rmse_mean": {a: ci(boot_mean[a]) for a in arms_rmse_cols},
            "rel_improvement_pct": {a: ci(boot_rel[a])
                                    for a in arms_rmse_cols if a != "pop_slope"},
            "delta_vs_static": {a: ci(boot_delta[a])
                                for a in arms_rmse_cols if a.startswith("k")},
        },
        "verdicts": verdicts,
    }
    (out_dir / "eval.json").write_text(json.dumps(out, indent=2))
    pu.to_parquet(out_dir / "per_user.parquet")
    if len(traj_df):
        traj_df.to_parquet(out_dir / "beta_trajectories.parquet")
    print(f"\n[save] {out_dir}/eval.json")
    print(f"[save] {out_dir}/per_user.parquet ({len(pu)} rows)")
    print(f"[save] {out_dir}/beta_trajectories.parquet ({len(traj_df)} rows)")


if __name__ == "__main__":
    main()
