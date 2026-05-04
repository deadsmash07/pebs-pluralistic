"""Per-category breakdown of the Track 1 PILSD 8.58% RMSE headline.

Reviewer question: "does the 8.58% hold uniformly across conversation types,
or does PILSD help only on specific categories?"

This script re-runs the `eval_user_score_mse_shrunk.py` k=5 within-user CV
headline on every category slice (≥50 utterances) of PRISM, using the same
EB-shrinkage formula. τ², pop-slope, and per-user (α_j, β_j) are all
re-estimated ON THE SLICE (no leakage across slices — each cohort is its
own self-contained headline re-run).

Stratifiers:
  1. conversation_type (controversy guided / values guided / unguided)
     joined from HF `HannahRoseKirk/prism-alignment[conversations]`
  2. turn_bucket: (turn 0 | 1-2 | 3+)
  3. model_family: coarse grouping from `model_name`
     (openai / anthropic / meta-llama / mistral / cohere / google / other)
  4. if_chosen: True vs False (chosen utterance vs rejected alternatives)
  5. user_mean_score_quartile: Q1/Q2/Q3/Q4 by per-user mean score_user
     (captures "lenient" vs "harsh" users)
  6. user_n_utter_quartile: by per-user utterance count

For each slice we report:
  - n_obs, n_users surviving CV
  - pop-slope RMSE (mean across users)
  - PILSD-shrunk RMSE (mean across users)
  - relative improvement (%)
  - Wilcoxon signed-rank p (PILSD < pop one-sided)
  - user win rate (fraction of users where PILSD beats pop-slope)

A "category" with <50 test observations OR <10 CV-surviving users is dropped.

Protocol matches paper headline: k=5 CV, min_obs_per_user=6, seed=42.

Outputs:
  results/t1_per_category_breakdown.json
  T1_PER_CATEGORY_BREAKDOWN.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
SCORED_PARQUET = REPO_ROOT / "data" / "prism_rm_scored.parquet"
CONV_META_PARQUET = REPO_ROOT / "data" / "_prism_conversation_meta.parquet"
OUT_JSON = REPO_ROOT / "results" / "t1_per_category_breakdown.json"
OUT_MD = REPO_ROOT / "T1_PER_CATEGORY_BREAKDOWN.md"

K_FOLDS = 5
MIN_OBS_PER_USER = 6
SEED = 42
MIN_OBS_PER_CATEGORY = 50
MIN_USERS_SURVIVING = 10


def kfold_split(n: int, k: int, rng: np.random.Generator):
    idx = np.arange(n)
    rng.shuffle(idx)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        stop = (i + 1) * fold_size if i < k - 1 else n
        test_idx = idx[start:stop]
        train_idx = np.concatenate([idx[:start], idx[stop:]])
        folds.append((train_idx, test_idx))
    return folds


def ols_with_V(x: np.ndarray, y: np.ndarray):
    k = len(x)
    if k < 2 or np.var(x) < 1e-12:
        return float(np.mean(y)) if k else 0.0, 0.0, np.inf, np.inf
    x_bar = x.mean()
    Sxx = ((x - x_bar) ** 2).sum()
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = intercept + slope * x
    sigma_hat_sq = ((y - y_pred) ** 2).sum() / max(k - 2, 1)
    V_int = sigma_hat_sq * (1.0 / k + x_bar**2 / max(Sxx, 1e-12))
    V_slope = sigma_hat_sq / max(Sxx, 1e-12)
    return float(intercept), float(slope), float(V_int), float(V_slope)


def run_slice_headline(
    df: pd.DataFrame,
    k_folds: int = K_FOLDS,
    min_obs: int = MIN_OBS_PER_USER,
    seed: int = SEED,
) -> Dict:
    """Reproduce paper's eval_user_score_mse_shrunk.py on a data slice."""
    rng = np.random.default_rng(seed)
    df = df.dropna(subset=["score_user"]).reset_index(drop=True)

    if len(df) < MIN_OBS_PER_CATEGORY or np.var(df.rm_score) < 1e-12:
        return {"error": "insufficient_data", "n_users": 0, "n_rows": int(len(df))}

    slope_pop, intercept_pop = np.polyfit(df.rm_score, df.score_user, 1)
    pop_alpha = float(intercept_pop)
    pop_beta = float(slope_pop)

    # pre-pass: estimate τ²
    user_stats = []
    for uid, grp in df.groupby("user_id"):
        if len(grp) < min_obs:
            continue
        a, b, Va, Vb = ols_with_V(
            grp.rm_score.to_numpy(), grp.score_user.to_numpy().astype(float)
        )
        user_stats.append(
            {"user_id": uid, "alpha": a, "beta": b, "V_alpha": Va, "V_beta": Vb}
        )
    if len(user_stats) < MIN_USERS_SURVIVING:
        return {"error": "too_few_users_for_tau", "n_users": len(user_stats)}
    us = pd.DataFrame(user_stats)
    V_alpha_total = float(us.alpha.var())
    V_beta_total = float(us.beta.var())
    mean_samp_V_alpha = float(
        us.V_alpha.replace([np.inf, -np.inf], np.nan).dropna().mean()
    )
    mean_samp_V_beta = float(
        us.V_beta.replace([np.inf, -np.inf], np.nan).dropna().mean()
    )
    tau_a_sq = max(V_alpha_total - mean_samp_V_alpha, 1e-6)
    tau_b_sq = max(V_beta_total - mean_samp_V_beta, 1e-6)

    # within-user k-fold CV
    per_user_rows = []
    for uid, grp in df.groupby("user_id"):
        n = len(grp)
        if n < min_obs:
            continue
        x = grp.rm_score.to_numpy()
        y = grp.score_user.to_numpy().astype(float)
        folds = kfold_split(n, k_folds, rng)
        sq = {"pop_slope": [], "pilsd_shrunk": []}
        for tr, te in folds:
            if len(te) == 0:
                continue
            x_tr, y_tr = x[tr], y[tr]
            x_te, y_te = x[te], y[te]
            a, b, Va, Vb = ols_with_V(x_tr, y_tr)
            sq["pop_slope"].extend(((pop_alpha + pop_beta * x_te - y_te) ** 2).tolist())
            omega_a = tau_a_sq / (tau_a_sq + Va) if np.isfinite(Va) else 0.0
            omega_b = tau_b_sq / (tau_b_sq + Vb) if np.isfinite(Vb) else 0.0
            a_s = omega_a * a + (1 - omega_a) * pop_alpha
            b_s = omega_b * b + (1 - omega_b) * pop_beta
            sq["pilsd_shrunk"].extend(((a_s + b_s * x_te - y_te) ** 2).tolist())
        if len(sq["pop_slope"]) == 0:
            continue
        per_user_rows.append(
            {
                "user_id": uid,
                "n": n,
                "rmse_pop_slope": float(np.sqrt(np.mean(sq["pop_slope"]))),
                "rmse_pilsd_shrunk": float(np.sqrt(np.mean(sq["pilsd_shrunk"]))),
            }
        )

    pu = pd.DataFrame(per_user_rows)
    if len(pu) < MIN_USERS_SURVIVING:
        return {"error": "insufficient_surviving_users", "n_users": int(len(pu))}

    pop_mean = float(pu.rmse_pop_slope.mean())
    sh_mean = float(pu.rmse_pilsd_shrunk.mean())
    rel_gain = 100.0 * (pop_mean - sh_mean) / pop_mean
    try:
        wilcox = stats.wilcoxon(
            pu.rmse_pilsd_shrunk.to_numpy(),
            pu.rmse_pop_slope.to_numpy(),
            alternative="less",
        )
        wilcox_p = float(wilcox.pvalue)
    except Exception:
        wilcox_p = float("nan")
    win_rate = float((pu.rmse_pilsd_shrunk < pu.rmse_pop_slope).mean())

    return {
        "n_obs": int(len(df)),
        "n_users": int(len(pu)),
        "mean_rmse_pop_slope": pop_mean,
        "mean_rmse_pilsd_shrunk": sh_mean,
        "relative_gain_pct": float(rel_gain),
        "wilcoxon_p_shrunk_lt_pop": wilcox_p,
        "user_win_rate": win_rate,
        "tau_alpha_sq": tau_a_sq,
        "tau_beta_sq": tau_b_sq,
    }


# -------------------- feature engineering --------------------


def add_conversation_type(df: pd.DataFrame) -> pd.DataFrame:
    if not CONV_META_PARQUET.exists():
        print(f"[warn] {CONV_META_PARQUET} missing — skipping conversation_type")
        df["conversation_type"] = "_unknown"
        return df
    conv_meta = pd.read_parquet(CONV_META_PARQUET)
    df = df.merge(
        conv_meta[["conversation_id", "conversation_type", "conversation_turns"]],
        on="conversation_id",
        how="left",
    )
    return df


def add_turn_bucket(df: pd.DataFrame) -> pd.DataFrame:
    def _bucket(t: int) -> str:
        if t == 0:
            return "0_first_turn"
        if t <= 2:
            return "1-2_early"
        return "3+_late"

    df["turn_bucket"] = df["turn"].apply(_bucket)
    return df


def add_model_family(df: pd.DataFrame) -> pd.DataFrame:
    def _family(name: str) -> str:
        n = str(name).lower()
        if "gpt-" in n or n.startswith("gpt"):
            return "openai"
        if "claude" in n:
            return "anthropic"
        if "llama" in n:
            return "meta-llama"
        if "mistral" in n:
            return "mistral"
        if "command" in n or "cohere" in n:
            return "cohere"
        if "bison" in n or "flan-t5" in n or "palm" in n or "google" in n:
            return "google"
        if "zephyr" in n or "huggingface" in n:
            return "hf-zephyr"
        if "guanaco" in n:
            return "guanaco"
        if "falcon" in n:
            return "falcon"
        if "luminous" in n:
            return "aleph-alpha"
        if "oasst" in n or "openassistant" in n:
            return "oasst"
        return "other"

    df["model_family"] = df["model_name"].apply(_family)
    return df


def add_user_level_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    per_user = df.groupby("user_id").agg(
        user_mean_score=("score_user", "mean"),
        user_n_utter=("user_id", "size"),
    )
    # Score leniency quartiles
    per_user["score_q"] = pd.qcut(
        per_user.user_mean_score,
        q=4,
        labels=["Q1_lowest_mean", "Q2", "Q3", "Q4_highest_mean"],
    )
    # Utterance count quartiles
    per_user["n_q"] = pd.qcut(
        per_user.user_n_utter,
        q=4,
        labels=["Q1_fewest_utter", "Q2", "Q3", "Q4_most_utter"],
        duplicates="drop",
    )
    df = df.merge(
        per_user[["score_q", "n_q"]].reset_index(),
        on="user_id",
        how="left",
    )
    df = df.rename(
        columns={
            "score_q": "user_mean_score_quartile",
            "n_q": "user_n_utter_quartile",
        }
    )
    return df


# -------------------- orchestration --------------------


def run_stratifier(
    df_all: pd.DataFrame, col: str, baseline_gain: float
) -> List[Dict]:
    """Run per-category headline for each unique value in df[col]."""
    rows = []
    vc = df_all[col].value_counts(dropna=False)
    print(f"\n=== {col}  (values: {len(vc)}) ===")
    for val, count in vc.items():
        if count < MIN_OBS_PER_CATEGORY:
            print(f"  skip {val!r}: only {count} obs (<{MIN_OBS_PER_CATEGORY})")
            continue
        df_sub = df_all[df_all[col] == val].reset_index(drop=True)
        res = run_slice_headline(df_sub)
        if res.get("error"):
            print(f"  skip {val!r}: {res['error']} (n_users={res.get('n_users', 0)})")
            continue
        delta = res["relative_gain_pct"] - baseline_gain
        rows.append(
            {
                "stratifier": col,
                "category": str(val),
                **res,
                "delta_vs_baseline_pp": float(delta),
            }
        )
        print(
            f"  {str(val):>25s}: n_obs={res['n_obs']:>5d}  "
            f"n_users={res['n_users']:>4d}  "
            f"gain={res['relative_gain_pct']:+.3f}%  "
            f"(Δ={delta:+.2f} pp, p={res['wilcoxon_p_shrunk_lt_pop']:.2e}, "
            f"win={100 * res['user_win_rate']:.1f}%)"
        )
    return rows


def build_markdown(results: Dict, baseline_gain: float, wall: float) -> str:
    lines = []
    lines.append("# Track 1 PILSD Per-Category Breakdown")
    lines.append("")
    lines.append(
        f"_Runtime: {wall:.1f}s_   "
        f"_Protocol: within-user 5-fold CV, min_obs_per_user=6, seed=42_"
    )
    lines.append("")
    lines.append(
        f"**Headline under test**: paper §4.1 reports +8.58% relative RMSE "
        f"improvement of PILSD EB-shrinkage over population-slope baseline "
        f"on PRISM. Baseline re-run on full cohort: "
        f"**{baseline_gain:+.3f}%**. This document breaks the headline into "
        f"interpretable category slices."
    )
    lines.append("")
    lines.append(
        f"_Inclusion rule_: category must have ≥{MIN_OBS_PER_CATEGORY} obs AND "
        f"≥{MIN_USERS_SURVIVING} users surviving CV."
    )
    lines.append("")

    # All-row table sorted by gain
    all_rows = [r for strat_rows in results["per_stratifier"].values() for r in strat_rows]
    all_rows_sorted = sorted(all_rows, key=lambda r: r["relative_gain_pct"], reverse=True)

    lines.append("## All categories (sorted by improvement)")
    lines.append("")
    lines.append(
        "| stratifier | category | n_obs | n_users | pop RMSE | "
        "PILSD RMSE | gain | Wilcoxon p | win % |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in all_rows_sorted:
        lines.append(
            f"| {r['stratifier']} | {r['category']} | "
            f"{r['n_obs']} | {r['n_users']} | "
            f"{r['mean_rmse_pop_slope']:.3f} | "
            f"{r['mean_rmse_pilsd_shrunk']:.3f} | "
            f"**{r['relative_gain_pct']:+.2f}%** | "
            f"{r['wilcoxon_p_shrunk_lt_pop']:.2e} | "
            f"{100*r['user_win_rate']:.1f}% |"
        )
    lines.append("")

    # Per-stratifier sections
    for strat, rows in results["per_stratifier"].items():
        if not rows:
            continue
        lines.append(f"## Stratifier: `{strat}`")
        lines.append("")
        lines.append(
            "| category | n_obs | n_users | pop RMSE | PILSD RMSE | "
            "gain | Δ vs baseline | win % |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|"
        )
        rows_sorted = sorted(rows, key=lambda r: r["relative_gain_pct"], reverse=True)
        for r in rows_sorted:
            lines.append(
                f"| {r['category']} | {r['n_obs']} | {r['n_users']} | "
                f"{r['mean_rmse_pop_slope']:.3f} | "
                f"{r['mean_rmse_pilsd_shrunk']:.3f} | "
                f"**{r['relative_gain_pct']:+.2f}%** | "
                f"{r['delta_vs_baseline_pp']:+.2f} pp | "
                f"{100*r['user_win_rate']:.1f}% |"
            )
        lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    if len(all_rows_sorted) > 0:
        top = all_rows_sorted[0]
        bot = all_rows_sorted[-1]
        gains = [r["relative_gain_pct"] for r in all_rows_sorted]
        neg_rows = [r for r in all_rows_sorted if r["relative_gain_pct"] < 0]
        small_rows = [
            r
            for r in all_rows_sorted
            if r["relative_gain_pct"] < baseline_gain / 2.0
        ]
        lines.append(
            f"- **Baseline gain**: {baseline_gain:+.3f}% (full cohort, 1394 users)"
        )
        lines.append(
            f"- **Range across {len(all_rows_sorted)} categories**: "
            f"[{min(gains):+.2f}%, {max(gains):+.2f}%]"
        )
        lines.append(
            f"- **Strongest**: `{top['stratifier']}={top['category']}` "
            f"→ {top['relative_gain_pct']:+.2f}% "
            f"(n_users={top['n_users']})"
        )
        lines.append(
            f"- **Weakest**: `{bot['stratifier']}={bot['category']}` "
            f"→ {bot['relative_gain_pct']:+.2f}% "
            f"(n_users={bot['n_users']})"
        )
        if neg_rows:
            lines.append(
                f"- **Categories where PILSD HURTS (gain < 0)**: "
                + ", ".join(
                    f"`{r['stratifier']}={r['category']}` ({r['relative_gain_pct']:+.2f}%)"
                    for r in neg_rows
                )
            )
        else:
            lines.append("- **Categories where PILSD HURTS**: none (all gains ≥ 0).")
        if small_rows:
            lines.append(
                f"- **Fragile (gain < half of baseline = {baseline_gain/2:.2f}%)**: "
                + ", ".join(
                    f"`{r['stratifier']}={r['category']}` ({r['relative_gain_pct']:+.2f}%)"
                    for r in small_rows
                )
            )
        else:
            lines.append(
                f"- **Fragile cells (gain < half baseline)**: none."
            )
        # Uniformity heuristic
        ratio = max(gains) / max(min(gains), 1e-3) if min(gains) > 0 else float("inf")
        lines.append("")
        if min(gains) > 0 and max(gains) - min(gains) < 0.5 * baseline_gain:
            lines.append(
                "**Conclusion**: PILSD benefit is approximately uniform "
                "across conversation types, turn positions, model families, "
                "and user score regimes. The 8.58% headline is not carried "
                "by a single subgroup."
            )
        elif min(gains) > 0:
            lines.append(
                f"**Conclusion**: PILSD is uniformly beneficial (no negative "
                f"cells) but heterogeneous in magnitude — "
                f"strongest/weakest ratio = {ratio:.2f}×."
            )
        else:
            lines.append(
                f"**Conclusion**: PILSD has {len(neg_rows)} category(ies) "
                f"where it hurts, indicating non-uniform benefit. "
                f"Disclosure in paper recommended."
            )
    lines.append("")
    return "\n".join(lines)


def main():
    t0 = time.time()
    print(f"[load] {SCORED_PARQUET}")
    df = (
        pd.read_parquet(SCORED_PARQUET)
        .dropna(subset=["score_user"])
        .reset_index(drop=True)
    )
    print(f"[load] {len(df)} utterances, {df.user_id.nunique()} users")

    # Feature engineering
    df = add_conversation_type(df)
    df = add_turn_bucket(df)
    df = add_model_family(df)
    df = add_user_level_quartiles(df)

    # Baseline (full cohort)
    print("\n=== BASELINE (full cohort, target 8.58%) ===")
    baseline = run_slice_headline(df)
    print(
        f"  gain={baseline['relative_gain_pct']:+.3f}% "
        f"(n_users={baseline['n_users']}, n_obs={baseline['n_obs']})"
    )
    baseline_gain = baseline["relative_gain_pct"]

    # Run per-stratifier
    stratifiers = [
        "conversation_type",
        "turn_bucket",
        "model_family",
        "if_chosen",
        "user_mean_score_quartile",
        "user_n_utter_quartile",
    ]
    per_stratifier = {}
    for col in stratifiers:
        if col not in df.columns:
            print(f"[warn] {col} missing — skipping")
            continue
        # Cast categorical to str for hashing
        df[col] = df[col].astype(str)
        per_stratifier[col] = run_stratifier(df, col, baseline_gain)

    results = {
        "config": {
            "k_folds": K_FOLDS,
            "min_obs_per_user": MIN_OBS_PER_USER,
            "min_obs_per_category": MIN_OBS_PER_CATEGORY,
            "min_users_surviving": MIN_USERS_SURVIVING,
            "seed": SEED,
            "paper_headline_pct": 8.58,
        },
        "baseline_full_cohort": baseline,
        "per_stratifier": per_stratifier,
        "wall_seconds": float(time.time() - t0),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[save] {OUT_JSON}")

    md = build_markdown(results, baseline_gain, time.time() - t0)
    OUT_MD.write_text(md)
    print(f"[save] {OUT_MD}")

    print(f"\n[done] wall={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
