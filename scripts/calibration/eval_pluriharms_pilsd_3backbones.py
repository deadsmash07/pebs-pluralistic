"""3-backbone PILSD eval on PluriHarms.

Parallel to `llama32_3b_backbone/eval_pilsd_3backbones.py` which handles
PRISM. For each of {qwen7b-Harm_Level, skywork27b, llama32_3b} we fit
per-user (alpha_j, beta_j) via within-user 5-fold CV with PRISM-matched
EB shrinkage and report:

    rating_ij = alpha_j * anchor_i + beta_j + epsilon_ij

Outputs
-------
- `results/pluriharms_pilsd_3backbones.json`
- `results/pluriharms_pilsd_3backbones.parquet` (per-user RMSE table)

Anchor sources
--------------
- qwen7b: `Harm_Level` column from `data/pluriharms/prompts.csv` (existing
  iter+N+224 protocol; serves as the Qwen-equivalent anchor since iter+N+224
  used a Qwen-based SafetyAnalyst classifier for `Harm_Level`).
- skywork27b: `data/pluriharms_skywork_scored.parquet` produced by
  `scripts/score_pluriharms_with_skywork_27b.py`.
- llama32_3b: `data/pluriharms_llama32_3b_scored.parquet` produced by
  `scripts/score_pluriharms_with_llama32_3b.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from src.methods.cluster_bootstrap import cluster_bootstrap  # type: ignore
except Exception:
    cluster_bootstrap = None  # Fall back to iid if unavailable


BACKBONES = [
    # (name, parquet_or_csv, score_column)
    ("qwen7b",     None,                                        "Harm_Level"),
    ("skywork27b", "data/pluriharms_skywork_scored.parquet",    "skywork_score"),
    ("llama32_3b", "data/pluriharms_llama32_3b_scored.parquet", "llama32_3b_score"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations-csv",
                   default="data/pluriharms/annotations.csv")
    p.add_argument("--prompts-csv", default="data/pluriharms/prompts.csv")
    p.add_argument("--min-obs-per-user", type=int, default=6)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--output-path",
                   default="results/pluriharms_pilsd_3backbones.json")
    return p.parse_args()


def load_long(annotations_csv: Path, prompts_csv: Path) -> pd.DataFrame:
    ann = pd.read_csv(annotations_csv)
    prm = pd.read_csv(prompts_csv)
    rating_cols = [c for c in ann.columns if c.startswith("Rating_")]
    long = ann.melt(
        id_vars=["Participant_ID"],
        value_vars=rating_cols,
        var_name="_rating_col",
        value_name="rating",
    )
    long["Question_Index"] = long["_rating_col"].str.replace(
        "Rating_", "").astype(int)
    long = long.merge(prm[["Question_Index", "Harm_Level"]],
                      on="Question_Index")
    long = long.dropna(subset=["rating", "Harm_Level"]).reset_index(drop=True)
    long = long.rename(columns={"Participant_ID": "user_id"})
    return long[["user_id", "Question_Index", "rating", "Harm_Level"]]


def kfold_split(n: int, k: int, rng):
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = []
    fold_size = n // k
    for i in range(k):
        start = i * fold_size
        stop = (i + 1) * fold_size if i < k - 1 else n
        test_idx = idx[start:stop]
        train_idx = np.concatenate([idx[:start], idx[stop:]])
        folds.append((train_idx, test_idx))
    return folds


def ols_with_V(x, y):
    k = len(x)
    if k < 2 or np.var(x) < 1e-12:
        return (float(np.mean(y)) if k else 0.0), 0.0, np.inf, np.inf
    x_bar = x.mean()
    Sxx = ((x - x_bar) ** 2).sum()
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = intercept + slope * x
    sigma_hat_sq = ((y - y_pred) ** 2).sum() / max(k - 2, 1)
    V_int = sigma_hat_sq * (1.0 / k + x_bar ** 2 / max(Sxx, 1e-12))
    V_slope = sigma_hat_sq / max(Sxx, 1e-12)
    return float(intercept), float(slope), float(V_int), float(V_slope)


def pop_ols(df: pd.DataFrame, x_col: str, y_col: str = "rating"):
    slope, intercept = np.polyfit(df[x_col], df[y_col], 1)
    return float(intercept), float(slope)


def estimate_tau(df: pd.DataFrame, x_col: str, min_obs: int,
                 y_col: str = "rating"):
    rows = []
    for uid, grp in df.groupby("user_id"):
        if len(grp) < min_obs:
            continue
        a, b, Va, Vb = ols_with_V(
            grp[x_col].to_numpy(),
            grp[y_col].to_numpy().astype(float),
        )
        rows.append({"alpha": a, "beta": b, "V_alpha": Va, "V_beta": Vb})
    us = pd.DataFrame(rows)
    V_alpha_total = float(us.alpha.var())
    V_beta_total = float(us.beta.var())
    mean_Va = float(us.V_alpha.replace([np.inf, -np.inf], np.nan).dropna().mean())
    mean_Vb = float(us.V_beta.replace([np.inf, -np.inf], np.nan).dropna().mean())
    tau_a_sq = max(V_alpha_total - mean_Va, 1e-6)
    tau_b_sq = max(V_beta_total - mean_Vb, 1e-6)
    return tau_a_sq, tau_b_sq


def bootstrap_gain_ci(per_user_df: pd.DataFrame, name: str,
                      n_boot: int, rng) -> tuple[float, float, float, float]:
    """Cluster bootstrap (resample users) for the gain percentage.

    gain% = 100 * (mean_pop_slope - mean_pilsd_shrunk) / mean_pop_slope

    Returns (gain_mean, gain_lo95, gain_hi95, se).
    """
    col_pop = f"rmse_pop_slope_{name}"
    col_shr = f"rmse_pilsd_shrunk_{name}"
    ids = per_user_df["user_id"].to_numpy()
    pop = per_user_df[col_pop].to_numpy()
    shr = per_user_df[col_shr].to_numpy()

    def stat_fn(p, s):
        mp = float(np.mean(p))
        return 100.0 * (mp - float(np.mean(s))) / mp if mp > 0 else 0.0

    if cluster_bootstrap is not None:
        mean, se, boots = cluster_bootstrap(
            ids, stat_fn, pop, shr, n_boot=n_boot, rng=rng,
        )
    else:
        # Fallback: resample rows (less strict; users already unique per-row here)
        n = len(pop)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            i = rng.integers(0, n, size=n)
            boots[b] = stat_fn(pop[i], shr[i])
        mean = float(np.mean(boots))
        se = float(np.std(boots, ddof=1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(mean), float(lo), float(hi), float(se)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ann_path = Path(args.annotations_csv)
    if not ann_path.is_absolute():
        ann_path = ROOT / args.annotations_csv
    prm_path = Path(args.prompts_csv)
    if not prm_path.is_absolute():
        prm_path = ROOT / args.prompts_csv

    long = load_long(ann_path, prm_path)
    print(f"[load] PluriHarms long-format: "
          f"{len(long)} ratings, "
          f"{long.user_id.nunique()} users, "
          f"{long.Question_Index.nunique()} prompts")

    # --- Attach Skywork + Llama anchors via Question_Index merge ---
    sky_path = ROOT / "data/pluriharms_skywork_scored.parquet"
    llm_path = ROOT / "data/pluriharms_llama32_3b_scored.parquet"
    print(f"[load] Skywork parquet: {sky_path.exists()}")
    print(f"[load] Llama   parquet: {llm_path.exists()}")

    backbone_specs = {"qwen7b": "Harm_Level"}
    df = long.copy()
    if sky_path.exists():
        sky = pd.read_parquet(sky_path)
        df = df.merge(sky[["Question_Index", "skywork_score"]],
                      on="Question_Index", how="inner")
        backbone_specs["skywork27b"] = "skywork_score"
    else:
        print("[warn] Skywork anchor missing — cell will be absent from output.")
    if llm_path.exists():
        llm = pd.read_parquet(llm_path)
        df = df.merge(llm[["Question_Index", "llama32_3b_score"]],
                      on="Question_Index", how="inner")
        backbone_specs["llama32_3b"] = "llama32_3b_score"
    else:
        print("[warn] Llama anchor missing — cell will be absent from output.")
    print(f"[join] {len(df)} ratings with {len(backbone_specs)} anchors, "
          f"{df.user_id.nunique()} users, {df.Question_Index.nunique()} prompts")

    # --- Population calibrations & tau estimates per backbone ---
    pop = {}
    taus = {}
    corrs = {}
    for name, col in backbone_specs.items():
        a, b = pop_ols(df, col)
        ta, tb = estimate_tau(df, col, args.min_obs_per_user)
        corr = float(np.corrcoef(df[col], df.rating)[0, 1])
        pop[name] = {"alpha": a, "beta": b}
        taus[name] = {"tau_alpha_sq": ta, "tau_beta_sq": tb}
        corrs[name] = corr
        print(f"[pop] {name:<12} alpha={a:+.3f}  beta={b:+.3f}  "
              f"corr={corr:+.4f}  tau_a2={ta:.2f}  tau_b2={tb:.2f}")

    arms = ["no_calib"]
    for name in backbone_specs:
        arms.append(f"pop_slope_{name}")
        arms.append(f"pilsd_shrunk_{name}")

    per_user_rows = []
    for uid, grp in df.groupby("user_id"):
        n = len(grp)
        if n < args.min_obs_per_user:
            continue
        y = grp.rating.to_numpy().astype(float)
        x = {name: grp[col].to_numpy().astype(float)
             for name, col in backbone_specs.items()}
        sq = {a: [] for a in arms}
        for train_idx, test_idx in kfold_split(n, args.k_folds, rng):
            if len(test_idx) == 0:
                continue
            y_tr, y_te = y[train_idx], y[test_idx]

            yh = np.full_like(y_te, np.mean(y_tr))
            sq["no_calib"].extend(((yh - y_te) ** 2).tolist())

            for name in backbone_specs:
                xt = x[name]
                x_tr, x_te = xt[train_idx], xt[test_idx]

                yh = pop[name]["alpha"] + pop[name]["beta"] * x_te
                sq[f"pop_slope_{name}"].extend(((yh - y_te) ** 2).tolist())

                a, b, Va, Vb = ols_with_V(x_tr, y_tr)
                ta = taus[name]["tau_alpha_sq"]
                tb = taus[name]["tau_beta_sq"]
                wa = ta / (ta + Va) if np.isfinite(Va) else 0.0
                wb = tb / (tb + Vb) if np.isfinite(Vb) else 0.0
                a_s = wa * a + (1 - wa) * pop[name]["alpha"]
                b_s = wb * b + (1 - wb) * pop[name]["beta"]
                yh = a_s + b_s * x_te
                sq[f"pilsd_shrunk_{name}"].extend(((yh - y_te) ** 2).tolist())

        per_user_rows.append({
            "user_id": int(uid),
            "n": int(n),
            **{f"rmse_{a}": float(np.sqrt(np.mean(sq[a]))) for a in arms},
        })

    pu = pd.DataFrame(per_user_rows)
    print(f"\n=== 3-backbone within-user CV ({len(pu)} users, k={args.k_folds}) ===")
    print(f"{'Arm':<32}{'Mean RMSE':>12}{'Median':>12}")
    for a in arms:
        m = pu[f"rmse_{a}"].mean()
        med = pu[f"rmse_{a}"].median()
        print(f"  {a:<30}{m:>12.3f}{med:>12.3f}")

    per_backbone = {}
    for name in backbone_specs:
        base = float(pu[f"rmse_pop_slope_{name}"].mean())
        shrunk = float(pu[f"rmse_pilsd_shrunk_{name}"].mean())
        rel = 100.0 * (base - shrunk) / base
        ca = pu[f"rmse_pilsd_shrunk_{name}"].to_numpy()
        cb = pu[f"rmse_pop_slope_{name}"].to_numpy()
        w = stats.wilcoxon(ca, cb, alternative="less")
        gain_mean, gain_lo, gain_hi, gain_se = bootstrap_gain_ci(
            pu, name, args.n_boot, rng,
        )
        per_backbone[name] = {
            "pop_slope_rmse_mean": base,
            "pilsd_shrunk_rmse_mean": shrunk,
            "relative_improvement_pct": float(rel),
            "gain_bootstrap_mean_pct": gain_mean,
            "gain_bootstrap_lo95_pct": gain_lo,
            "gain_bootstrap_hi95_pct": gain_hi,
            "gain_bootstrap_se_pct": gain_se,
            "ci_strictly_positive": bool(gain_lo > 0),
            "frac_shrunk_smaller": float((ca < cb).mean()),
            "wilcoxon_less_p": float(w.pvalue),
            "pop_correlation": corrs[name],
            "tau_alpha_sq": taus[name]["tau_alpha_sq"],
            "tau_beta_sq": taus[name]["tau_beta_sq"],
        }
        print(f"[backbone] {name:<12}  pop={base:.3f}  shrunk={shrunk:.3f}  "
              f"gain%={rel:+.2f}  CI=[{gain_lo:+.2f}, {gain_hi:+.2f}]  "
              f"CI>0={gain_lo > 0}  p={w.pvalue:.2e}")

    pair_records = {}
    for a, b in combinations(arms, 2):
        ca = pu[f"rmse_{a}"].to_numpy()
        cb = pu[f"rmse_{b}"].to_numpy()
        w = stats.wilcoxon(ca, cb, alternative="two-sided")
        pair_records[f"{a}__vs__{b}"] = {
            "mean_delta": float((ca - cb).mean()),
            "frac_a_smaller": float((ca < cb).mean()),
            "wilcoxon_p": float(w.pvalue),
        }

    out = {
        "dataset": "PluriHarms (Li et al. 2026, arXiv:2601.08951)",
        "n_users": int(len(pu)),
        "n_ratings": int(len(df)),
        "k_folds": args.k_folds,
        "min_obs_per_user": args.min_obs_per_user,
        "seed": args.seed,
        "n_boot": args.n_boot,
        "pop_correlations": corrs,
        "pop_calibration": pop,
        "eb_taus": taus,
        "rmse_mean": {a: float(pu[f"rmse_{a}"].mean()) for a in arms},
        "rmse_median": {a: float(pu[f"rmse_{a}"].median()) for a in arms},
        "per_backbone": per_backbone,
        "pairwise_comparisons": pair_records,
    }
    out_path = Path(args.output_path)
    if not out_path.is_absolute():
        out_path = ROOT / args.output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    pu.to_parquet(out_path.with_suffix(".parquet"))
    print(f"\n[save] {out_path}")
    print(f"[save] {out_path.with_suffix('.parquet')}")


if __name__ == "__main__":
    main()
