"""W-A4 — PRISM demographic-balance post-filter audit.

Hypothesis under test
---------------------
PEBS's headline analysis applies a min_obs_per_user >= 6 (or n_j >= 3
conversations) filter to retain users with enough data for the per-user
EB shrinkage to be well-defined. A natural fairness concern:
"the filter selectively excludes demographic minorities (e.g. older users
with low engagement, raters from under-represented regions) and the 8.58%
gain is therefore not generalisable to the full PRISM population."

This script audits the n_j>=3 (and the closely-related min-obs-per-user>=6)
filter on each PRISM demographic axis. For each axis, we compute pre-filter
vs post-filter shares per category and report the absolute and relative
shifts.

The PRISM demographics file (data/prism_demographics.parquet) has the
following axes we audit:

  - age (6 bands: 18-24, 25-34, 35-44, 45-54, 55-64, 65+)
  - gender (4 cats: Male, Female, Non-binary, Prefer not to say)
  - study_locale (10+ countries: US, UK, South Africa, etc.)
  - english_proficiency (5 levels: Native, Fluent, Advanced, Intermediate, Basic)
  - education (8 levels: Bachelors, Graduate, ..., Some Primary)
  - employment_status (8 cats: Working full-time, Part-time, Student, etc.)
  - religion_simplified (Christian / No Affiliation / Jewish / Muslim / etc.)
  - ethnicity_simplified (White / Asian / Black / Hispanic / Mixed / etc.)
  - birth_region (Americas / Europe / Africa / Asia / Oceania)

The dict-typed PRISM demographic fields (religion, ethnicity, location) are
NATIVE Python dicts in the parquet file (verified during pre-flight); we
extract sub-fields by direct dict-key access and DO NOT need ast.literal_eval
or json.loads.

Per axis we report:

  pre_count[c]   := raw count of users in category c BEFORE filter
  post_count[c]  := count of users in category c AFTER filter
  pre_share[c]   := pre_count[c] / sum(pre_count)
  post_share[c]  := post_count[c] / sum(post_count)
  delta_share[c] := post_share[c] - pre_share[c]      (pp shift, can be -ve)
  retention[c]   := post_count[c] / pre_count[c]      (fraction retained)

Aggregate axis-level metric:
  total_variation_distance := 0.5 * sum_c |post_share[c] - pre_share[c]|
                               (in [0, 1]; 0 = identical distributions)

Outcome classification
----------------------
Let TVD_max = max over axes of total_variation_distance, R_min = min over
axes/categories of retention[c] (only categories with pre_count >= 10).

  CONFIRMED-DEMOGRAPHIC-BALANCE-PRESERVED
      if  TVD_max < 0.03  AND  R_min > 0.85

  PARTIAL-DEMOGRAPHIC-BALANCE-PRESERVED
      if  0.03 <= TVD_max < 0.06  OR  R_min in [0.75, 0.85]

  TENTATIVE-INCONCLUSIVE-DEMOGRAPHIC-IMBALANCE
      if  TVD_max in [0.06, 0.10]  OR  R_min in [0.50, 0.75]

  REJECTED-DEMOGRAPHIC-IMBALANCE-INTRODUCED
      if  TVD_max >= 0.10  OR  R_min < 0.50

Outputs
-------
  results/track1_demographic_balance/summary.json       <- per-axis TVD, retention
  results/track1_demographic_balance/per_axis.parquet   <- detailed per-axis-per-category counts/shares
  results/track1_demographic_balance/demographic_table.tex <- LaTeX table for paper §3 Setup

Cost: <1 min CPU. Seed: 20260420.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Paths -----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]   # 3_PEBS_Standalone/
T1 = ROOT.parent / "1_Causal_RLHF"
SCORED = T1 / "data/prism_rm_scored.parquet"
DEMO = T1 / "data/prism_demographics.parquet"

OUT_RESULTS = ROOT / "results/track1_demographic_balance"

# Defaults --------------------------------------------------------------------
RNG_DEFAULT = 20260420
MIN_OBS_PER_USER_DEFAULT = 6   # matches headline filter
MIN_CONV_PER_USER_DEFAULT = 3  # matches the alternative n_j>=3 filter

AXES_TO_AUDIT = [
    "age",
    "gender",
    "study_locale",
    "english_proficiency",
    "education",
    "employment_status",
    "religion_simplified",
    "ethnicity_simplified",
    "birth_region",
]

MIN_PRE_COUNT_FOR_R_MIN = 10  # category needs >=10 users pre-filter to count


# ============================================================================
# Helper — extract dict-keyed sub-fields from PRISM JSON-typed columns
# ============================================================================

def safe_dict_get(v, key: str):
    """PRISM stores religion/ethnicity/location as native Python dicts in
    parquet (verified via type-check on load). Returns the key value, or
    None if v is not a dict or the key is absent."""
    if isinstance(v, dict):
        return v.get(key)
    return None


def expand_demographic_fields(demo: pd.DataFrame) -> pd.DataFrame:
    """Add parsed columns for the dict-stored fields:
    - religion_simplified
    - ethnicity_simplified
    - birth_region
    - reside_region
    """
    out = demo.copy()
    out["religion_simplified"] = out["religion"].apply(
        lambda v: safe_dict_get(v, "simplified")
    )
    out["ethnicity_simplified"] = out["ethnicity"].apply(
        lambda v: safe_dict_get(v, "simplified")
    )
    out["birth_region"] = out["location"].apply(
        lambda v: safe_dict_get(v, "birth_region")
    )
    out["reside_region"] = out["location"].apply(
        lambda v: safe_dict_get(v, "reside_region")
    )
    return out


# ============================================================================
# main pipeline
# ============================================================================

def compute_filter_membership(scored: pd.DataFrame,
                                min_obs: int, min_conv: int) -> pd.DataFrame:
    """Per-user table of (n_obs, n_conv, passes_filter).
    passes_filter requires BOTH min_obs and min_conv (matches the
    eval_user_score_mse_shrunk.py + t1_loco_with_shrinkage.py joint filter).
    """
    obs = scored.groupby("user_id").size().rename("n_obs")
    conv = scored.groupby("user_id")["conversation_id"].nunique().rename("n_conv")
    out = pd.concat([obs, conv], axis=1).reset_index()
    out["passes_obs_filter"] = out["n_obs"] >= min_obs
    out["passes_conv_filter"] = out["n_conv"] >= min_conv
    out["passes_joint_filter"] = out["passes_obs_filter"] & out["passes_conv_filter"]
    return out


def audit_axis(merged: pd.DataFrame, axis: str) -> pd.DataFrame:
    """Per-axis pre/post share table.

    merged columns required: user_id, passes_joint_filter, <axis>.
    Returns DataFrame with one row per unique non-NaN value of <axis>.
    """
    if axis not in merged.columns:
        return pd.DataFrame()
    pre = merged[axis].value_counts(dropna=False)
    post = merged[merged.passes_joint_filter][axis].value_counts(dropna=False)

    pre_total = int(pre.sum())
    post_total = int(post.sum())
    rows = []
    for cat in pre.index:
        pre_count = int(pre.get(cat, 0))
        post_count = int(post.get(cat, 0))
        pre_share = pre_count / pre_total if pre_total > 0 else np.nan
        post_share = post_count / post_total if post_total > 0 else np.nan
        delta_share = post_share - pre_share
        retention = post_count / pre_count if pre_count > 0 else np.nan
        rows.append({
            "axis": axis,
            "category": str(cat),
            "pre_count": pre_count,
            "post_count": post_count,
            "pre_share": pre_share,
            "post_share": post_share,
            "delta_share_pp": 100.0 * delta_share,
            "retention": retention,
        })
    return pd.DataFrame(rows)


def total_variation_distance(axis_df: pd.DataFrame) -> float:
    """0.5 * sum_c |post_share - pre_share|. NaN-safe."""
    deltas = axis_df.delta_share_pp.dropna().abs() / 100.0
    return float(0.5 * deltas.sum())


def write_latex_table(per_axis: pd.DataFrame, summary: dict, out_path: Path):
    """LaTeX table summarising per-axis TVD and per-category retention."""
    lines = []
    lines.append("% W-A4 PRISM demographic-balance post-filter audit.")
    lines.append("% Generated by paper/scripts/wave_a_W_A4_demographic_balance.py")
    lines.append(f"% Filter: n_obs >= {summary['design']['min_obs_per_user']}, "
                 f"n_conv >= {summary['design']['min_conv_per_user']}")
    lines.append(f"% Pre-filter: {summary['design']['n_users_pre']} users; "
                 f"Post-filter: {summary['design']['n_users_post']} users.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering \\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\caption{\\textbf{PRISM demographic-balance audit of "
                 "the $n_j {\\ge} 3$ conversations / $\\ge 6$ utterances "
                 "per-user filter.} Per-axis total-variation distance "
                 "(TVD; $0$ = pre and post identical) between the pre-filter "
                 "and post-filter rater-share distributions, and the worst-case "
                 "retention rate among categories with $\\ge 10$ pre-filter "
                 "raters. Retention = post-filter count / pre-filter count.}")
    lines.append("\\label{tab:wa4-demographic-balance}")
    lines.append("\\begin{tabular}{@{}lrrrr@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Axis} & \\textbf{Categories} "
                 "& \\textbf{Pre / Post } & \\textbf{TVD} "
                 "& \\textbf{Min retention} \\\\")
    lines.append("\\midrule")
    for axis in AXES_TO_AUDIT:
        axis_summ = summary["per_axis_summary"].get(axis)
        if axis_summ is None:
            continue
        lines.append(f"{axis.replace('_', ' ').title()} "
                     f"& {axis_summ['n_categories']} "
                     f"& {axis_summ['n_users_pre']} / {axis_summ['n_users_post']} "
                     f"& {axis_summ['total_variation_distance']:.4f} "
                     f"& {axis_summ['min_retention_eligible']:.3f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{-0.5em}")
    lines.append("\\end{table}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[save] {out_path}")


def run(args):
    t0 = time.time()
    if not SCORED.exists():
        sys.stderr.write(f"FATAL: PRISM scored data not found: {SCORED}\n")
        sys.exit(2)
    if not DEMO.exists():
        sys.stderr.write(f"FATAL: PRISM demographics not found: {DEMO}\n")
        sys.exit(2)

    df = pd.read_parquet(SCORED).dropna(subset=["score_user", "rm_score",
                                                  "user_id", "conversation_id"])
    demo = pd.read_parquet(DEMO)
    demo = expand_demographic_fields(demo)
    print(f"[load] PRISM scored {len(df)} utterances x {df.user_id.nunique()} users")
    print(f"[load] PRISM demographics {len(demo)} users")

    # Compute filter membership
    fm = compute_filter_membership(df, args.min_obs_per_user,
                                     args.min_conv_per_user)
    print(f"[filter] joint filter (>= {args.min_obs_per_user} obs AND "
          f">= {args.min_conv_per_user} conv): "
          f"{fm.passes_joint_filter.sum()} / {len(fm)} users pass")

    # Merge with demographics
    merged = fm.merge(demo, on="user_id", how="left")
    n_users_pre = len(merged)
    n_users_post = int(merged.passes_joint_filter.sum())
    n_users_with_demo = int(merged["age"].notna().sum())
    print(f"[merge] {n_users_with_demo} users have demographic info "
          f"(others = scored-only)")

    # Audit each axis
    all_axis_rows = []
    per_axis_summary = {}
    for axis in AXES_TO_AUDIT:
        axis_df = audit_axis(merged, axis)
        if len(axis_df) == 0:
            print(f"  [{axis}] axis missing from demographics file")
            continue
        all_axis_rows.append(axis_df)
        eligible = axis_df[axis_df.pre_count >= MIN_PRE_COUNT_FOR_R_MIN]
        if len(eligible) > 0:
            min_retention = float(eligible.retention.min())
            min_retention_cat = str(eligible.loc[eligible.retention.idxmin(), "category"])
        else:
            min_retention = np.nan
            min_retention_cat = "<no eligible category>"
        tvd = total_variation_distance(axis_df)
        per_axis_summary[axis] = {
            "n_categories": int(len(axis_df)),
            "n_users_pre": int(axis_df.pre_count.sum()),
            "n_users_post": int(axis_df.post_count.sum()),
            "total_variation_distance": tvd,
            "min_retention_eligible": min_retention,
            "min_retention_category": min_retention_cat,
            "max_delta_share_pp": float(axis_df.delta_share_pp.abs().max()),
        }
        print(f"  [{axis}] TVD = {tvd:.4f}  min_retention = {min_retention:.3f} "
              f"(category: {min_retention_cat})")

    per_axis_df = pd.concat(all_axis_rows, ignore_index=True) \
                    if all_axis_rows else pd.DataFrame()

    # Aggregate verdict
    tvd_values = [v["total_variation_distance"] for v in per_axis_summary.values()
                   if np.isfinite(v["total_variation_distance"])]
    retention_values = [v["min_retention_eligible"] for v in per_axis_summary.values()
                         if np.isfinite(v["min_retention_eligible"])]
    tvd_max = float(max(tvd_values)) if tvd_values else float("nan")
    r_min = float(min(retention_values)) if retention_values else float("nan")
    print(f"\n[aggregate] TVD_max across axes = {tvd_max:.4f}")
    print(f"[aggregate] R_min across categories (pre>=10) = {r_min:.3f}")

    # Outcome assignment
    if not (np.isfinite(tvd_max) and np.isfinite(r_min)):
        verdict = "TENTATIVE-INCONCLUSIVE-DEMOGRAPHIC-IMBALANCE"
    elif tvd_max < 0.03 and r_min > 0.85:
        verdict = "CONFIRMED-DEMOGRAPHIC-BALANCE-PRESERVED"
    elif tvd_max >= 0.10 or r_min < 0.50:
        verdict = "REJECTED-DEMOGRAPHIC-IMBALANCE-INTRODUCED"
    elif tvd_max >= 0.06 or r_min < 0.75:
        verdict = "TENTATIVE-INCONCLUSIVE-DEMOGRAPHIC-IMBALANCE"
    else:
        verdict = "PARTIAL-DEMOGRAPHIC-BALANCE-PRESERVED"

    print(f"[verdict] {verdict}")

    summary = {
        "design": {
            "n_users_pre": n_users_pre,
            "n_users_post": n_users_post,
            "n_users_with_demo": n_users_with_demo,
            "min_obs_per_user": args.min_obs_per_user,
            "min_conv_per_user": args.min_conv_per_user,
            "axes_audited": AXES_TO_AUDIT,
            "scored_path": str(SCORED),
            "demographics_path": str(DEMO),
            "min_pre_count_for_r_min": MIN_PRE_COUNT_FOR_R_MIN,
        },
        "per_axis_summary": per_axis_summary,
        "aggregate": {
            "tvd_max": tvd_max,
            "r_min": r_min,
            "tvd_max_axis": max(per_axis_summary.items(),
                                  key=lambda kv: kv[1]["total_variation_distance"]
                                  if np.isfinite(kv[1]["total_variation_distance"])
                                  else -1.0)[0] if per_axis_summary else None,
        },
        "verdict_class": verdict,
        "rng_seed": args.seed,
        "runtime_seconds": float(time.time() - t0),
        "honest_disclosure": {
            "scope": "Audit of the n_obs >= 6 / n_conv >= 3 PEBS filter on "
                     "PRISM demographic axes; pre-filter users include those "
                     "with no demographic data (study_only and similar).",
            "limitations": [
                "Some users in PRISM have no demographic row "
                "(survey_only=True with no engagement); these contribute "
                "to the n_users_pre but have NaN values across all demographic "
                "axes, so they neither boost nor subtract from any category's "
                "share.",
                "A category-level chi-square test is NOT reported because the "
                "filter's null hypothesis is 'no shift'; under that null the "
                "post-filter sample is a non-random subset of the pre-filter "
                "sample (selected by engagement) and the chi-square assumption "
                "of independent sampling is violated. We use TVD + per-category "
                "retention as descriptive evidence instead.",
                "Dict-typed PRISM fields (religion, ethnicity, location) are "
                "extracted via direct .get() calls on native Python dicts; "
                "NaN values fall through to the 'NaN' bucket in value_counts.",
            ],
        },
    }
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    (OUT_RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    if len(per_axis_df) > 0:
        per_axis_df.to_parquet(OUT_RESULTS / "per_axis.parquet", index=False)
    write_latex_table(per_axis_df, summary, OUT_RESULTS / "demographic_table.tex")
    print(f"\n[save] {OUT_RESULTS}/summary.json "
          f"({summary['runtime_seconds']:.1f}s)")
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=RNG_DEFAULT)
    p.add_argument("--min-obs-per-user", type=int,
                    default=MIN_OBS_PER_USER_DEFAULT)
    p.add_argument("--min-conv-per-user", type=int,
                    default=MIN_CONV_PER_USER_DEFAULT)
    p.add_argument("--smoke", action="store_true",
                    help="Audit only the first 3 axes (age, gender, locale)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        global AXES_TO_AUDIT
        AXES_TO_AUDIT = AXES_TO_AUDIT[:3]
        print(f"[smoke] auditing only {AXES_TO_AUDIT}")
    summary = run(args)
    print(f"\n[final] {summary['verdict_class']}")


if __name__ == "__main__":
    main()
