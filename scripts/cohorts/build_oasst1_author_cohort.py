"""Build per-author quality trajectory cohort from `OpenAssistant/oasst1`.

OASST1 is the PREDECESSOR to OASST2 and covers a *different* 85-day window
(2023-01-16 → 2023-04-12) — non-overlapping with OASST2 (2023-05 → 2023-11).
This makes OASST1 a genuine 4th real-data test: same platform / same-style
annotation protocol but a distinct time window and reviewer wave.

Like OASST2, OASST1 exposes:
  - `user_id` per authored message (42-char hex ID, stable across tree)
  - `created_date` timestamp (ISO-8601 with TZ)
  - `labels.{name,value}` — averaged anonymous-reviewer Likert scores for
    {quality, toxicity, humor, creativity, violence, spam, ...}. We use the
    `quality` label as the scalar annotation-quality proxy.

Schema verified via `load_dataset('OpenAssistant/oasst1')`:
  - 84,437 rows (messages)
  - 12,917 unique users
  - 83,024 rows carry an aggregated quality label
  - quality range: 0.0–1.0  (mean 0.647, std 0.235)
  - 689 users with ≥20 quality-labelled messages
  - 362 users with ≥20 messages AND ≥7-day author-activity span
    → comfortably exceeds 100 × 20 requirement

Outputs:
  - `data/oasst1_author_quality.parquet`  (all rows)
  - `data/oasst1_author_cohort.parquet`   (cohort index + stats)
  - `data/oasst1_weekly_quality.parquet`  (weekly aggregate, for raw-trend plot)

Refs:
  - Köpf et al. 2023 arXiv:2304.07327 "OpenAssistant Conversations" (dataset)
  - Pinheiro & Bates 2000 Mixed-Effects Models (MixedLM)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _extract_label(row_labels, name: str):
    """Extract a named label value from OASST1's struct-of-arrays labels field.

    HF row['labels'] comes as {'name': [...], 'value': [...], 'count': [...]}
    with optionally None if the message was never reviewed.
    """
    if row_labels is None:
        return None
    try:
        names = list(row_labels["name"])
        values = list(row_labels["value"])
    except (KeyError, TypeError):
        return None
    if name not in names:
        return None
    try:
        return float(values[names.index(name)])
    except (ValueError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dataset", default="OpenAssistant/oasst1")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--min-messages-per-user", type=int, default=20)
    ap.add_argument("--min-span-days", type=float, default=7.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    print(f"[load] {args.hf_dataset}")
    ds = load_dataset(args.hf_dataset, split="train")
    print(f"[load] {len(ds)} rows")

    df = ds.to_pandas()

    # Extract quality label (primary proxy).
    df["quality"] = df["labels"].apply(lambda x: _extract_label(x, "quality"))
    # Extract other potential proxies (not used by default but stored for re-runs).
    for lbl in ("creativity", "humor", "toxicity", "violence", "hate_speech"):
        df[lbl] = df["labels"].apply(lambda x, L=lbl: _extract_label(x, L))

    # Parse timestamp.
    df["ts"] = pd.to_datetime(df["created_date"], utc=True, errors="coerce")

    df = df.dropna(subset=["quality", "ts", "user_id"]).reset_index(drop=True)
    print(f"[filter] rows with quality+ts+user_id: {len(df)}")

    # Day-of-campaign numeric time covariate (same convention as OASST2 script).
    t0 = df["ts"].min()
    df["day_num"] = ((df["ts"] - t0).dt.total_seconds() / 86400.0).astype(float)
    # Also month_num for cross-dataset parity with OASST2 / SHP.
    df["month_num"] = df["day_num"] / 30.437  # avg days/month

    # Role → binary exog for MixedLM (prompter vs assistant).
    df["is_assistant"] = (df["role"].astype(str) == "assistant").astype(float)

    # Rename user_id to keep column naming consistent with multipref/oasst2 scripts.
    # (user_id already matches OASST2 convention.)

    all_path = out_dir / "oasst1_author_quality.parquet"
    # Drop heavy columns we won't use downstream to keep parquet small.
    keep_cols = ["message_id", "parent_id", "user_id", "ts", "day_num", "month_num",
                 "role", "is_assistant", "lang",
                 "quality", "creativity", "humor", "toxicity", "violence",
                 "review_count"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_slim = df[keep_cols].copy()
    df_slim.to_parquet(all_path, index=False)
    print(f"[save] {all_path} ({len(df_slim)} rows)")

    # Cohort stats.
    stats = df.groupby("user_id").agg(
        n_messages=("message_id", "size"),
        tmin=("ts", "min"),
        tmax=("ts", "max"),
        mean_quality=("quality", "mean"),
        assistant_frac=("is_assistant", "mean"),
        mean_review_count=("review_count", "mean"),
    )
    stats["span_days"] = (stats["tmax"] - stats["tmin"]).dt.total_seconds() / 86400.0
    cohort = stats[
        (stats["n_messages"] >= args.min_messages_per_user)
        & (stats["span_days"] >= args.min_span_days)
    ].sort_values("n_messages", ascending=False)

    cohort_path = out_dir / "oasst1_author_cohort.parquet"
    cohort.to_parquet(cohort_path)

    print("\n=== OASST1 Author Trajectory Cohort ===")
    print(f"  unique users total: {len(stats)}")
    print(f"  cohort (n>={args.min_messages_per_user}, span>={args.min_span_days}d): {len(cohort)}")
    print(f"  cohort total messages: {int(cohort['n_messages'].sum())}")
    print(f"  cohort mean span: {cohort['span_days'].mean():.1f} days "
          f"(median {cohort['span_days'].median():.1f})")
    print(f"  cohort assistant frac: {cohort['assistant_frac'].mean():.2f}")
    print(f"  cohort mean quality: {cohort['mean_quality'].mean():.3f}")
    print(f"  saved: {cohort_path}")

    # Weekly aggregate (raw drift signal).
    df["week_bucket"] = (df["day_num"] // 7).astype(int)
    weekly = df.groupby("week_bucket").agg(
        n_messages=("message_id", "size"),
        mean_quality=("quality", "mean"),
        std_quality=("quality", "std"),
        n_users=("user_id", "nunique"),
    )
    weekly_path = out_dir / "oasst1_weekly_quality.parquet"
    weekly.to_parquet(weekly_path)
    print("\n=== Weekly aggregate (raw-signal plot input) ===")
    print(weekly.head(15).to_string())
    print(f"\nsaved: {weekly_path}")


if __name__ == "__main__":
    main()
