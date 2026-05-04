"""Fit per-user QUADRATIC calibration on PRISM RM-scored utterances.

Future-work item from paper §5.3: extend linear per-user calibrator to a
quadratic form that can absorb response saturation near the score scale
endpoints (0 and 100).

Model (Pinheiro & Bates 2000 linear mixed-effects, random-slope on two
fixed-effects regressors):

    score_user_ij = (γ_0 + γ_0j) + (β_0 + β_0j) · rm_z_ij
                    + (α_0 + α_0j) · rm_z_ij^2 + ε_ij

where (γ_0j, β_0j, α_0j) ~ N(0, Ω) are per-user deviations. For each user
j we extract:

    α_j = α_0 + α_0j      per-user quadratic slope
    β_j = β_0 + β_0j      per-user linear slope
    γ_j = γ_0 + γ_0j      per-user intercept

To avoid multicollinearity between rm_z and rm_z^2, rm_z is z-scored and
the squared regressor is recentered to have zero grand mean (common
polynomial-regression trick).

Outputs
-------
- `data/prism_user_calibrators_quadratic.parquet` — one row per user with
  α_j (quadratic), β_j (linear), γ_j (intercept), and sample-size metadata.

Refs
----
- Pinheiro & Bates 2000 §2 (MixedLM random-slope framework)
- Gelman & Hill 2007 §4, §12 (polynomial fixed effects + partial pooling)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scored-parquet", default="data/prism_rm_scored.parquet")
    p.add_argument("--output-dir", default="data")
    p.add_argument("--min-obs-per-user", type=int, default=5)
    p.add_argument("--output-prefix", default="prism")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.scored_parquet)
    df = df.dropna(subset=["score_user"]).reset_index(drop=True)
    print(f"[load] {len(df)} utterances, {df.user_id.nunique()} users")

    # Z-score rm_score for numerical stability (matches linear fit convention).
    rm_mean, rm_std = float(df.rm_score.mean()), float(df.rm_score.std())
    df["rm_z"] = (df.rm_score - rm_mean) / max(rm_std, 1e-9)
    # Center the squared regressor to reduce collinearity with rm_z.
    sq_raw = df["rm_z"] ** 2
    sq_mean = float(sq_raw.mean())
    df["rm_z_sq_c"] = sq_raw - sq_mean
    corr_lin_sq = float(np.corrcoef(df["rm_z"], df["rm_z_sq_c"])[0, 1])
    print(f"[feat] rm_z  mean={df.rm_z.mean():.3e} std={df.rm_z.std():.3f}")
    print(f"[feat] rm_z_sq_c mean={df.rm_z_sq_c.mean():.3e} std={df.rm_z_sq_c.std():.3f}")
    print(f"[feat] corr(rm_z, rm_z_sq_c) = {corr_lin_sq:+.4f}  "
          f"(near-zero -> low collinearity)")

    # Fit: score_user ~ rm_z + rm_z_sq_c + (rm_z + rm_z_sq_c | user_id)
    print("[fit] MixedLM with per-user random intercept + slope on rm_z + slope on rm_z_sq_c")
    t0 = time.time()
    converged = False
    res = None
    try:
        md = smf.mixedlm(
            "score_user ~ rm_z + rm_z_sq_c",
            data=df,
            groups=df["user_id"],
            re_formula="~rm_z + rm_z_sq_c",
        )
        res = md.fit(method="lbfgs", maxiter=1000, disp=False)
        converged = bool(res.converged)
        print(f"[fit] converged={converged}  ll={res.llf:.2f}  "
              f"t={time.time() - t0:.1f}s")
        print(res.summary().tables[1])
    except Exception as e:
        print(f"[fit] full random-slope MixedLM failed: {e}")

    # Fallback 1: random intercept + slope on rm_z only (drop rm_z_sq_c random
    # effect — keep it as fixed effect).
    if res is None or not converged:
        print("[fit] falling back to random intercept + random slope on rm_z only")
        t0 = time.time()
        try:
            md = smf.mixedlm(
                "score_user ~ rm_z + rm_z_sq_c",
                data=df,
                groups=df["user_id"],
                re_formula="~rm_z",
            )
            res = md.fit(method="lbfgs", maxiter=1000, disp=False)
            converged = bool(res.converged)
            print(f"[fit] fallback converged={converged}  ll={res.llf:.2f}  "
                  f"t={time.time() - t0:.1f}s")
        except Exception as e:
            print(f"[fit] fallback-1 also failed: {e}")

    # Fallback 2: random intercept only.
    if res is None or not converged:
        print("[fit] final fallback: random intercept only")
        t0 = time.time()
        md = smf.mixedlm(
            "score_user ~ rm_z + rm_z_sq_c",
            data=df,
            groups=df["user_id"],
        )
        res = md.fit(method="lbfgs", maxiter=1000, disp=False)
        converged = bool(res.converged)
        print(f"[fit] fallback-2 converged={converged}  ll={res.llf:.2f}  "
              f"t={time.time() - t0:.1f}s")

    # Population coefficients
    pop_intercept = float(res.params["Intercept"])
    pop_linear = float(res.params["rm_z"])
    pop_quad = float(res.params["rm_z_sq_c"])
    print(f"[pop] γ_0={pop_intercept:.3f}  β_0={pop_linear:.3f}  α_0={pop_quad:.3f}")

    # Per-user random effects -> absolute per-user coefficients
    re_dict = res.random_effects
    per_user = []
    for uid, re in re_dict.items():
        re_int = float(re.get("Group", re.get("Intercept", 0.0)))
        re_lin = float(re.get("rm_z", 0.0))
        re_quad = float(re.get("rm_z_sq_c", 0.0))
        gamma_j = pop_intercept + re_int
        beta_j = pop_linear + re_lin
        alpha_j = pop_quad + re_quad
        n_obs = int((df.user_id == uid).sum())
        per_user.append({
            "user_id": str(uid),
            "alpha_j": alpha_j,   # quadratic coef
            "beta_j": beta_j,     # linear coef
            "gamma_j": gamma_j,   # intercept
            "n_observations": n_obs,
        })
    cal_df = pd.DataFrame(per_user)
    print(f"[calibrators] fit for {len(cal_df)} users")
    for col in ["alpha_j", "beta_j", "gamma_j"]:
        v = cal_df[col]
        print(f"  {col}: mean={v.mean():.3f}  std={v.std():.3f}  "
              f"q=[{v.quantile(0.25):.3f}, {v.median():.3f}, {v.quantile(0.75):.3f}]")

    out_path = Path(args.output_dir) / f"{args.output_prefix}_user_calibrators_quadratic.parquet"
    cal_df.to_parquet(out_path)
    print(f"[save] {out_path}")

    # Save feature metadata (rm_mean, rm_std, sq_mean) for downstream use
    meta = {
        "rm_mean": rm_mean, "rm_std": rm_std, "sq_mean": sq_mean,
        "pop_intercept": pop_intercept, "pop_linear": pop_linear,
        "pop_quad": pop_quad, "converged": converged,
        "corr_lin_sq_after_centering": corr_lin_sq,
    }
    meta_path = Path(args.output_dir) / f"{args.output_prefix}_user_calibrators_quadratic_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[save] {meta_path}")


if __name__ == "__main__":
    main()
