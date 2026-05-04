"""Build per-evaluator annotation trajectory cohort from AI2 MultiPref.

Dataset: `allenai/multipref` on HuggingFace (Miranda et al., "Hybrid Preferences:
Learning to Route Instances for Human vs. AI Feedback", arXiv:2410.19133).

Schema (verified 2026-04-17):
  - 10,461 comparisons  ×  4 annotations each  =  41,844 annotations
  - 227 unique evaluators (Docker-style names, e.g. `clever_bardeen`)
  - per-annotation timestamp (UTC-naive) + per-aspect preference + confidence
  - time span: 2024-05-01 → 2024-05-30 (≈29 days, one month of collection)

Because the time span is ~1 month, we bucket by DAY (not month) and fit
`quality_proxy ~ day_num + is_expert + (1 | evaluator)`. Two quality proxies
are emitted so downstream analysis can pick either (both stored as columns):

  - mean_conf : mean of {overall, helpful, truthful, harmless} confidence
                (absolutely=1.0, fairly=0.66, not=0.33). Higher = more careful review.
  - time_spent: seconds the annotator spent. Higher = more effort proxy.

Cohort filter (defaults, overridable):
  n ≥ 30 annotations AND span ≥ 7 days per evaluator.
  Produces ~148 "power evaluators" covering ~34k annotations.

Output:
  - data/multipref_evaluator_quality.parquet  (all rows, daily granularity)
  - data/multipref_evaluator_cohort.parquet   (cohort evaluator index + stats)
  - data/multipref_daily_quality.parquet      (daily aggregate — raw series)

References:
  - Miranda et al. 2024 arXiv:2410.19133 (HYPER dataset paper)
  - Pinheiro & Bates 2000 "Mixed-Effects Models in S and S-PLUS" (MixedLM)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PREF_TO_SCORE = {
    "A-is-clearly-better": 1.0,
    "A-is-slightly-better": 0.75,
    "Tie": 0.5,
    "B-is-slightly-better": 0.25,
    "B-is-clearly-better": 0.0,
}

CONF_TO_SCORE = {
    "absolutely-confident": 1.0,
    "fairly-confident": 0.66,
    "not-confident": 0.33,
}


def iter_annotation_rows(ds):
    """Yield one row per (comparison × annotator) pair."""
    for row in ds:
        for kind, anns in [
            ("normal", row["normal_worker_annotations"]),
            ("expert", row["expert_worker_annotations"]),
        ]:
            for a in anns:
                overall_c = CONF_TO_SCORE.get(a["overall_confidence"], float("nan"))
                helpful_c = CONF_TO_SCORE.get(a["helpful_confidence"], float("nan"))
                truthful_c = CONF_TO_SCORE.get(a["truthful_confidence"], float("nan"))
                harmless_c = CONF_TO_SCORE.get(a["harmless_confidence"], float("nan"))
                mean_conf = sum(
                    c for c in (overall_c, helpful_c, truthful_c, harmless_c)
                    if c == c  # NaN-skip
                )
                valid_count = sum(
                    1 for c in (overall_c, helpful_c, truthful_c, harmless_c)
                    if c == c
                )
                if valid_count > 0:
                    mean_conf /= valid_count
                else:
                    mean_conf = float("nan")

                yield {
                    "comparison_id": row["comparison_id"],
                    "category": row["category"],
                    "source": row["source"],
                    "evaluator": a["evaluator"],
                    "kind": kind,
                    "timestamp": a["timestamp"],
                    "time_spent": a["time_spent"],
                    "overall_pref": a["overall_pref"],
                    "helpful_pref": a["helpful_pref"],
                    "truthful_pref": a["truthful_pref"],
                    "harmless_pref": a["harmless_pref"],
                    "overall_conf": overall_c,
                    "mean_conf": mean_conf,
                }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hf-dataset", default="allenai/multipref",
        help="HuggingFace dataset name",
    )
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--min-annotations-per-evaluator", type=int, default=30)
    ap.add_argument("--min-span-days", type=float, default=7.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    print(f"[load] {args.hf_dataset}")
    ds = load_dataset(args.hf_dataset, split="train")
    print(f"[load] {len(ds)} comparisons")

    df = pd.DataFrame(list(iter_annotation_rows(ds)))
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts", "mean_conf"]).sort_values("ts").reset_index(drop=True)

    t0 = df["ts"].min()
    df["day_num"] = ((df["ts"] - t0).dt.total_seconds() / 86400.0).astype(float)
    df["is_expert"] = (df["kind"] == "expert").astype(float)

    # Rename for cross-script parity (oasst2 uses `user_id`, `quality`).
    df["user_id"] = df["evaluator"]
    df["quality"] = df["mean_conf"]

    all_path = out_dir / "multipref_evaluator_quality.parquet"
    df.to_parquet(all_path, index=False)
    print(f"[save] {all_path} ({len(df)} rows)")

    # Cohort filter
    stats = df.groupby("evaluator").agg(
        n_annotations=("comparison_id", "size"),
        tmin=("ts", "min"),
        tmax=("ts", "max"),
        mean_conf=("mean_conf", "mean"),
        expert_frac=("is_expert", "mean"),
    )
    stats["span_days"] = (stats["tmax"] - stats["tmin"]).dt.total_seconds() / 86400.0
    cohort = stats[
        (stats["n_annotations"] >= args.min_annotations_per_evaluator)
        & (stats["span_days"] >= args.min_span_days)
    ].sort_values("n_annotations", ascending=False)

    cohort_path = out_dir / "multipref_evaluator_cohort.parquet"
    cohort.to_parquet(cohort_path)

    print(f"\n=== MultiPref Evaluator Trajectory Cohort ===")
    print(f"  unique evaluators total: {len(stats)}")
    print(f"  cohort (n≥{args.min_annotations_per_evaluator}, span≥{args.min_span_days}d): {len(cohort)}")
    print(f"  cohort total annotations: {int(cohort['n_annotations'].sum())}")
    print(f"  cohort mean span: {cohort['span_days'].mean():.1f} days")
    print(f"  cohort expert frac: {cohort['expert_frac'].mean():.2f}")
    print(f"  mean_conf mean: {cohort['mean_conf'].mean():.3f}")
    print(f"  saved: {cohort_path}")

    # Daily aggregate (raw drift signal)
    df["day_bucket"] = df["day_num"].astype(int)
    daily = df.groupby("day_bucket").agg(
        n_annotations=("comparison_id", "size"),
        mean_conf=("mean_conf", "mean"),
        std_conf=("mean_conf", "std"),
        time_spent_mean=("time_spent", "mean"),
    )
    daily_path = out_dir / "multipref_daily_quality.parquet"
    daily.to_parquet(daily_path)
    print(f"\n=== Daily aggregate (for raw-signal plot) ===")
    print(daily.head(15).to_string())
    print(f"\nsaved: {daily_path}")


if __name__ == "__main__":
    main()
