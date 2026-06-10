"""2PL IRT per-user NON-AFFINE calibrator — paper §5.3 future-work test.

Paper's Limitations section flags 2PL IRT (Birnbaum 1968) as a future-work
alternative to PEBS's affine per-user calibrator. Proposition T1.MI (paper
§3.2) shows that any monotone-invariant (in particular: affine with
alpha_j > 0) per-user calibrator CANNOT move rank-only downstream metrics
at PPO convergence. A NON-AFFINE calibrator breaks T1.MI's monotone-
invariance premise: predictions like "sigmoid compresses the ends" are
the exact kinds of transformations where scale-aware gains (RMSE) can
differ from pair-accuracy.

This script tests empirically whether 2PL IRT improves within-user
held-out RMSE over PEBS-static (paper headline 22.343 @ temporal-CV).

Model (continuous-2PL, per-user parameterisation)
-------------------------------------------------
    z_ij = standardize(rm_score_ij)     (global mean/std from train fold)
    score_user_ji = 100 * sigmoid(alpha_j * (z_ij - beta_j)) + eps_ji
    alpha_j  ~ LogNormal(mu_log_alpha, tau_log_alpha^2)  # discrimination
    beta_j   ~ Normal(mu_beta, tau_beta^2)               # midpoint
    eps_ji   ~ Normal(0, sigma^2)
    mu_log_alpha, mu_beta ~ Normal(0, 1) (loose hyperpriors)
    tau_log_alpha, tau_beta, sigma ~ HalfCauchy(1)

This is non-affine in rm_score because the sigmoid squashes extreme values
toward {0, 100}. Two parameters per user — same degrees of freedom as
PEBS's (alpha_j, beta_j). The comparison is therefore apples-to-apples
on DOF, and any RMSE difference is attributable to the functional form,
not to extra flexibility.

Why NOT py-irt
--------------
py-irt (Lalor et al. EMNLP 2019) exposes a CLI + JSONLINES API for the
classical 2PL BINARY model P(y=1) = sigma(alpha_item * (theta_user -
beta_item)). Two obstacles for our use-case:
  1. Responses are continuous scores in [0,100], not binary. Adapting to
     a graded-response model requires custom Pyro code anyway.
  2. py-irt's per-ITEM parameterisation assumes MANY users answer the SAME
     item. In PRISM each (user, prompt) pair is mostly unique, so per-item
     alpha_i / beta_i are under-identified. A per-USER parameterisation
     is the only identifiable continuous-2PL for this data.
We therefore implement the model directly in Pyro (GPU-accelerated SVI)
and cite py-irt as the canonical binary 2PL reference.

Inference
---------
Pyro SVI with AutoNormal guide + Adam (LR=0.01) for 2000 max steps, ELBO
plateau early-stop (patience 200, rel-tol 1e-4). All tensors on CUDA.

Data + evaluation
-----------------
PRISM 68k utterances x 1394 users (same cohort as POS-T1.1 headline).
Per-user temporal 80/20 split (sorted by generated_datetime, same as
PEBS-static temporal-CV harness). Train fold fits the hierarchical
2PL; test fold evaluates per-user RMSE. 30-seed user-level bootstrap CI.

Arms reported (3-arm head-to-head)
----------------------------------
  A. PEBS-static          — EB-shrunk (alpha_j, beta_j), affine  (22.343)
  B. PEBS-random-walk     — Kalman F=I pooled MLE                (22.403)
  C. PEBS-2PL             — this script                          (???)

If C < A: NON-AFFINE calibration delivers additional RMSE reduction
beyond affine — supports the prediction that T1.MI's affine premise
is a real restriction.

If C >=A: 2PL's sigmoid nonlinearity does not help beyond affine on
PRISM's score distribution — another instance of the Kalman pattern
where a strictly-more-general model fails to separate empirically,
documenting a precondition-vs-sufficiency refinement.

References
----------
- Birnbaum, A. (1968) "Some latent trait models and their use in
  inferring an examinee's ability" in Lord & Novick, Statistical theories
  of mental test scores.
- Rasch, G. (1960) "Probabilistic models for some intelligence and
  attainment tests."
- Lalor, Wu, Yu (2019) "Learning Latent Parameters without Human Response
  Patterns: Item Response Theory with Artificial Crowds" EMNLP — py-irt.
- Amidei, Piwek, Willis (2020) "Identifying Annotator Bias" COLING —
  IRT for annotator modelling (paper cites as amidei2020identifying).
- Natesan et al. (2016) "Bayesian Prior Choice in IRT Estimation Using
  MCMC and Variational Bayes" Frontiers in Psychology.
- Bingham et al. (2019) "Pyro: Deep Universal Probabilistic Programming"
  JMLR.


Run
---
    python scripts/eval_pebs_2pl_irt.py \
        --scored-parquet data/prism_rm_scored.parquet \
        --timestamp-cache data/prism_conversation_timestamps.parquet \
        --output-dir results/track1_pebs_2pl_irt \
        --seed 42

Requires: pyro-ppl >= 1.9, torch >= 2.0 CUDA, pandas, numpy, scipy, matplotlib.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scored-parquet", default="data/prism_rm_scored.parquet")
    p.add_argument(
        "--timestamp-cache",
        default="data/prism_conversation_timestamps.parquet",
    )
    p.add_argument("--min-obs-per-user", type=int, default=6)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--n-bootstrap", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--rel-tol", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--output-dir", default="results/track1_pebs_2pl_irt"
    )
    p.add_argument(
        "--gpu-log",
        default="results/track1_pebs_2pl_irt/gpu_util.log",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers (parity with temporal-CV / Kalman scripts)
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


def load_and_prepare(args) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = (
        pd.read_parquet(args.scored_parquet)
        .dropna(subset=["score_user"])
        .reset_index(drop=True)
    )
    print(f"[load] {len(df)} utterances, {df.user_id.nunique()} users")
    ts = pd.read_parquet(args.timestamp_cache)
    ts["generated_datetime"] = pd.to_datetime(ts["generated_datetime"])
    df = df.merge(ts, on="conversation_id", how="left").dropna(
        subset=["generated_datetime"]
    )
    print(f"[join] {len(df)} utterances after timestamp join")

    # Build train/test per user using temporal sort (mirrors PEBS-static)
    train_parts, test_parts = [], []
    for uid, grp in df.groupby("user_id"):
        if len(grp) < args.min_obs_per_user:
            continue
        s = temporal_sort(grp)
        n = len(s)
        n_test = max(1, int(round(n * args.test_fraction)))
        if n - n_test < 2:
            continue
        train_parts.append(s.iloc[: n - n_test])
        test_parts.append(s.iloc[n - n_test :])
    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    uids_train = set(train_df.user_id.unique())
    uids_test = set(test_df.user_id.unique())
    users = sorted(uids_train & uids_test)
    print(
        f"[split] {len(users)} users pass min-obs+train/test constraints; "
        f"train n={len(train_df)} test n={len(test_df)}"
    )

    user_to_idx = {uid: i for i, uid in enumerate(users)}
    train_df = train_df[train_df.user_id.isin(users)].copy()
    test_df = test_df[test_df.user_id.isin(users)].copy()
    train_df["user_idx"] = train_df.user_id.map(user_to_idx)
    test_df["user_idx"] = test_df.user_id.map(user_to_idx)

    # Standardize rm_score using TRAIN-fold statistics only (no leakage)
    rm_mean = float(train_df.rm_score.mean())
    rm_std = float(train_df.rm_score.std())
    if rm_std < 1e-6:
        rm_std = 1.0
    train_df["z_rm"] = (train_df.rm_score - rm_mean) / rm_std
    test_df["z_rm"] = (test_df.rm_score - rm_mean) / rm_std

    meta = {
        "users": users,
        "user_to_idx": user_to_idx,
        "rm_mean": rm_mean,
        "rm_std": rm_std,
    }
    return (train_df, test_df, meta)


# ---------------------------------------------------------------------------
# 2PL Pyro model
# ---------------------------------------------------------------------------


def irt_2pl_model(
    user_idx: torch.Tensor,  # (N,) long
    z_rm: torch.Tensor,       # (N,) float
    score_user: torch.Tensor, # (N,) float in [0,100]
    n_users: int,
):
    """Continuous 2PL per-user calibrator with hierarchical priors.

    score_user_ji = 100 * sigmoid(alpha_j * (z_ij - beta_j)) + N(0, sigma^2)

    alpha_j  ~ LogNormal(mu_log_alpha, tau_log_alpha^2)
    beta_j   ~ Normal(mu_beta, tau_beta^2)
    sigma    ~ HalfCauchy(1)
    mu_log_alpha, mu_beta ~ Normal(0, 1)
    tau_log_alpha, tau_beta ~ HalfCauchy(1)
    """
    mu_log_alpha = pyro.sample("mu_log_alpha", dist.Normal(0.0, 1.0))
    mu_beta = pyro.sample("mu_beta", dist.Normal(0.0, 1.0))
    tau_log_alpha = pyro.sample("tau_log_alpha", dist.HalfCauchy(1.0))
    tau_beta = pyro.sample("tau_beta", dist.HalfCauchy(1.0))
    sigma = pyro.sample("sigma", dist.HalfCauchy(10.0))

    with pyro.plate("users", n_users):
        log_alpha = pyro.sample(
            "log_alpha", dist.Normal(mu_log_alpha, tau_log_alpha)
        )
        beta = pyro.sample("beta", dist.Normal(mu_beta, tau_beta))

    alpha = log_alpha.exp()
    alpha_i = alpha[user_idx]
    beta_i = beta[user_idx]
    logits = alpha_i * (z_rm - beta_i)
    mu = 100.0 * torch.sigmoid(logits)
    with pyro.plate("data", z_rm.shape[0]):
        pyro.sample("obs", dist.Normal(mu, sigma), obs=score_user)


def fit_2pl(
    train_df: pd.DataFrame,
    n_users: int,
    device: str,
    lr: float,
    max_steps: int,
    patience: int,
    rel_tol: float,
    seed: int,
    output_dir: Path,
    gpu_log_path: Path,
) -> Tuple[Dict[str, torch.Tensor], list, float, float]:
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    # Ensure Pyro samples + AutoNormal init tensors live on the same device
    # as the indexed data (avoids "indices cpu vs indexed cuda" error).
    torch.set_default_device(device)

    u_t = torch.tensor(train_df.user_idx.to_numpy(), dtype=torch.long, device=device)
    z_t = torch.tensor(train_df.z_rm.to_numpy(), dtype=torch.float32, device=device)
    y_t = torch.tensor(
        train_df.score_user.to_numpy(), dtype=torch.float32, device=device
    )

    guide = AutoNormal(irt_2pl_model)
    optim = Adam({"lr": lr})
    svi = SVI(irt_2pl_model, guide, optim, loss=Trace_ELBO())

    # Kick the GPU monitor as a background process (simple polling loop)
    gpu_log_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_log_path.write_text("")  # truncate
    mon = subprocess.Popen(
        [
            "bash",
            "-c",
            f"for i in $(seq 1 400); do "
            f"  echo -n \"step_approx=$i \"; "
            f"  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total "
            f"--format=csv,noheader,nounits | tr '\\n' '|'; echo ''; "
            f"  sleep 2; "
            f"done >> {gpu_log_path}",
        ],
    )
    try:
        losses = []
        best = None  # seeded on step 0
        best_step = 0
        t0 = time.time()
        for step in range(max_steps):
            loss = svi.step(u_t, z_t, y_t, n_users)
            losses.append(loss)
            if best is None:
                best = loss
                best_step = step
            elif loss < best - max(abs(best) * rel_tol, 1.0):
                best = loss
                best_step = step
            if step - best_step > patience and step > 500:
                print(
                    f"[svi] early-stop @ step {step}  "
                    f"(best={best:.1f} @ {best_step}; patience={patience})"
                )
                break
            if step % 100 == 0:
                print(
                    f"[svi] step {step:4d}  loss={loss:.1f}  best={best:.1f}"
                )
        elapsed = time.time() - t0
        print(
            f"[svi] {step+1} steps in {elapsed:.1f}s "
            f"({(step+1)/elapsed:.1f} steps/s)"
        )
    finally:
        mon.terminate()
        mon.wait(timeout=5)

    # Posterior means (for prediction we use the guide's MAP point estimate)
    params = {}
    with torch.no_grad():
        for name in [
            "mu_log_alpha",
            "mu_beta",
            "tau_log_alpha",
            "tau_beta",
            "sigma",
            "log_alpha",
            "beta",
        ]:
            params[name] = pyro.param(f"AutoNormal.locs.{name}").detach()

    return params, losses, float(best), elapsed


def predict_2pl(
    test_df: pd.DataFrame,
    params: Dict[str, torch.Tensor],
    device: str,
) -> np.ndarray:
    u_t = torch.tensor(test_df.user_idx.to_numpy(), dtype=torch.long, device=device)
    z_t = torch.tensor(test_df.z_rm.to_numpy(), dtype=torch.float32, device=device)
    log_alpha = params["log_alpha"]
    beta = params["beta"]
    alpha_i = log_alpha[u_t].exp()
    beta_i = beta[u_t]
    mu = 100.0 * torch.sigmoid(alpha_i * (z_t - beta_i))
    return mu.cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print(
            f"[fatal] CUDA unavailable; this script requires GPU. "
            f"Set --device=cpu at your own risk.",
            file=sys.stderr,
        )
        if args.device.startswith("cuda"):
            sys.exit(2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pyro.set_rng_seed(args.seed)

    print(
        f"[env] torch={torch.__version__} "
        f"pyro={pyro.__version__} "
        f"cuda_available={torch.cuda.is_available()} "
        f"device={args.device}"
    )

    train_df, test_df, meta = load_and_prepare(args)
    users = meta["users"]
    n_users = len(users)

    # Fit 2PL
    params, losses, best_elbo, elapsed = fit_2pl(
        train_df=train_df,
        n_users=n_users,
        device=args.device,
        lr=args.lr,
        max_steps=args.max_steps,
        patience=args.patience,
        rel_tol=args.rel_tol,
        seed=args.seed,
        output_dir=out_dir,
        gpu_log_path=Path(args.gpu_log),
    )

    # ELBO curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses, lw=1)
    ax.set_xlabel("SVI step")
    ax.set_ylabel("ELBO loss")
    ax.set_title(
        f"2PL IRT SVI ELBO (n_users={n_users}, n_obs_train={len(train_df)})"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "train_elbo.png", dpi=130)
    plt.close(fig)

    # Predict on test
    yhat = predict_2pl(test_df, params, args.device)
    test_df = test_df.copy()
    test_df["yhat_2pl"] = yhat

    # Per-user held-out RMSE
    per_user_rows = []
    for uid, grp in test_df.groupby("user_id"):
        resid = grp.score_user.to_numpy().astype(float) - grp.yhat_2pl.to_numpy()
        per_user_rows.append(
            {
                "user_id": uid,
                "n_test": int(len(grp)),
                "rmse_2pl": float(np.sqrt(np.mean(resid ** 2))),
            }
        )
    pu_2pl = pd.DataFrame(per_user_rows)

    # Attach PEBS-static / random-walk / kalman baselines via per_user merge
    base_pu_path = Path("results/track1_pebs_kalman/per_user.parquet")
    if base_pu_path.exists():
        base = pd.read_parquet(base_pu_path)
        merged = base.merge(pu_2pl, on="user_id", how="inner")
        print(
            f"[merge] {len(merged)} users intersect 2PL and baseline "
            f"per_user.parquet"
        )
    else:
        print(
            f"[warn] baseline per_user.parquet not found at {base_pu_path}; "
            f"reporting 2PL alone."
        )
        merged = pu_2pl.copy()

    # Rank arms
    arm_means = {"pebs_2pl": float(merged.rmse_2pl.mean())}
    for col in ["rmse_pebs_static", "rmse_pebs_random_walk", "rmse_pebs_kalman"]:
        if col in merged:
            arm_means[col.replace("rmse_", "")] = float(merged[col].mean())

    print("\n=== 3-arm RMSE (mean, per-user held-out, 80/20 temporal) ===")
    for arm, m in sorted(arm_means.items(), key=lambda kv: kv[1]):
        print(f"  {arm:>22}: {m:.3f}")

    # Paired Wilcoxon: 2PL vs PEBS-static (if available)
    from scipy import stats as sstats

    wilcoxon_info = {}
    if "rmse_pebs_static" in merged:
        w = sstats.wilcoxon(
            merged.rmse_2pl, merged.rmse_pebs_static, alternative="two-sided"
        )
        wins = float((merged.rmse_2pl < merged.rmse_pebs_static).mean())
        delta = float((merged.rmse_2pl - merged.rmse_pebs_static).mean())
        wilcoxon_info["2pl_vs_static"] = {
            "mean_delta": delta,
            "frac_2pl_better": wins,
            "wilcoxon_p": float(w.pvalue),
        }
        print(
            f"\n  2PL vs static   : Δ={delta:+.4f}  "
            f"2PL_better={wins:.1%}  wilcoxon p={w.pvalue:.3e}"
        )

    # Bootstrap CI on arm means (user-level resample, 30 seeds)
    boot_rows = []
    for b in range(args.n_bootstrap):
        local_rng = np.random.default_rng(args.seed + b)
        idx = local_rng.integers(0, len(merged), size=len(merged))
        smp = merged.iloc[idx]
        row = {"seed": args.seed + b, "rmse_2pl": float(smp.rmse_2pl.mean())}
        for col in [
            "rmse_pebs_static",
            "rmse_pebs_random_walk",
            "rmse_pebs_kalman",
        ]:
            if col in smp:
                row[col] = float(smp[col].mean())
        boot_rows.append(row)
    boot = pd.DataFrame(boot_rows)

    ci = {}
    for col in boot.columns:
        if col == "seed":
            continue
        ci[col] = [
            float(np.percentile(boot[col], 2.5)),
            float(np.percentile(boot[col], 97.5)),
        ]
    print(f"\n=== Bootstrap 95% CI (N={args.n_bootstrap}) ===")
    for col, (lo, hi) in ci.items():
        print(f"  {col:>24}: [{lo:.3f}, {hi:.3f}]")

    # Save artifacts
    out = {
        "n_users_merged": int(len(merged)),
        "n_users_2pl": int(len(pu_2pl)),
        "arm_rmse_mean": arm_means,
        "bootstrap_ci_95": ci,
        "wilcoxon": wilcoxon_info,
        "svi": {
            "max_steps": args.max_steps,
            "patience": args.patience,
            "rel_tol": args.rel_tol,
            "lr": args.lr,
            "final_elbo": best_elbo,
            "elapsed_s": elapsed,
            "n_steps_ran": len(losses),
            "n_users": n_users,
            "n_train_obs": int(len(train_df)),
            "n_test_obs": int(len(test_df)),
        },
        "seed": args.seed,
        "device": args.device,
        "paths": {
            "elbo_png": str(out_dir / "train_elbo.png"),
            "gpu_util_log": args.gpu_log,
        },
    }
    (out_dir / "eval.json").write_text(json.dumps(out, indent=2))
    pu_2pl.to_parquet(out_dir / "per_user_2pl.parquet")
    merged.to_parquet(out_dir / "per_user_merged.parquet")
    boot.to_parquet(out_dir / "bootstrap.parquet")
    np.save(out_dir / "elbo_trace.npy", np.array(losses))
    print(f"\n[save] {out_dir}")


if __name__ == "__main__":
    main()
