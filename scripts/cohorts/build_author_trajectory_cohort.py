"""Build per-author quality trajectory cohort from OASST2 ready trees.

**Pivot note (2026-04-17)**: The OASST2 2023-11-05 export does NOT populate
`events[ranking][].user_id` (discovered empirically this session), contrary
to what the schemas.py declaration suggests. Per-rater longitudinal analysis
is therefore NOT directly possible on OASST2 without upstream reprocessing.

**What IS available**: per-AUTHOR trajectories.
  - 22,348 unique authors
  - 135,174 nodes in ready.trees, 133,087 have aggregated `quality` labels
    (mean of anonymous reviewer ratings + review_count)
  - `created_date` per node, `role` ∈ {prompter, assistant}, `labels.quality.value`

**Pivoted PILSD analysis** (still a valid identification strategy):
  - Hypothesis: as OASST2 grew over 10 months, the REVIEWER POOL's quality
    standards drifted (calibration shift). Measurable via stable-author
    trajectories — a consistent author's quality scores should track reviewer
    drift if author-level quality is approximately stationary after warmup.
  - Detector: detect step-changes / monotone drift in the mean quality score
    of a held-out "anchor cohort" of high-volume authors whose first-month
    quality distribution is close to the overall mean.
  - Identification: same linear calibration idea, but M = quality score on
    anchor authors instead of quality score on anchor responses.

This is a publishable variant of PILSD — same underlying causal graph,
different instrument for identifying reviewer drift.

Reference:
  - OASST1 paper: Köpf et al. 2304.07327 (Köpf et al. 2023) §3.3 quality labels
  - Memory: `oasst2_hidden_annotator_ids.md` (the earlier schema read was
    correct for the codebase but the actual export strips it).
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd


def iter_author_rows(trees_path: Path):
    """Yield one row per OASST2 node with a quality label + timestamp."""
    with gzip.open(trees_path, "rt", encoding="utf-8") as f:
        for line in f:
            tree = json.loads(line)
            root = tree.get("prompt") or {}
            tree_id = tree.get("message_tree_id") or ""

            def walk(n):
                yield n
                for c in n.get("replies") or []:
                    yield from walk(c)

            for node in walk(root):
                labels = node.get("labels") or {}
                quality = labels.get("quality") or {}
                q_value = quality.get("value")
                q_count = quality.get("count")
                if q_value is None or node.get("user_id") is None:
                    continue
                yield {
                    "user_id": node["user_id"],
                    "message_id": node.get("message_id", ""),
                    "parent_id": node.get("parent_id", ""),
                    "tree_id": str(tree_id),
                    "role": node.get("role", ""),
                    "lang": node.get("lang", ""),
                    "created_date": node.get("created_date", ""),
                    "quality": float(q_value),
                    "quality_n_raters": int(q_count) if q_count else 0,
                    "review_count": node.get("review_count"),
                    "rank": node.get("rank"),
                }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees-path", required=True,
                    help="Path to 2023-11-05_oasst2_ready.trees.jsonl.gz")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--min-messages-per-author", type=int, default=20,
                    help="Cohort filter — authors with fewer are pruned.")
    ap.add_argument("--min-span-days", type=int, default=28)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trees_path = Path(args.trees_path)
    print(f"[parse] {trees_path}")

    rows = list(iter_author_rows(trees_path))
    print(f"[parse] {len(rows)} rows with quality label")

    df = pd.DataFrame(rows)
    df["created_ts"] = pd.to_datetime(df["created_date"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_ts"]).sort_values("created_ts").reset_index(drop=True)

    all_path = out_dir / "oasst2_author_quality.parquet"
    df.to_parquet(all_path)
    print(f"[save] {all_path} ({len(df)} rows)")

    # Author-level stats + cohort filter
    user_stats = df.groupby("user_id").agg(
        n_messages=("message_id", "size"),
        mean_quality=("quality", "mean"),
        first_ts=("created_ts", "min"),
        last_ts=("created_ts", "max"),
        assistant_frac=("role", lambda s: (s == "assistant").mean()),
    )
    user_stats["span_days"] = (user_stats["last_ts"] - user_stats["first_ts"]).dt.days

    cohort = user_stats[
        (user_stats["n_messages"] >= args.min_messages_per_author)
        & (user_stats["span_days"] >= args.min_span_days)
    ].sort_values("n_messages", ascending=False)
    cohort_path = out_dir / "oasst2_author_cohort.parquet"
    cohort.to_parquet(cohort_path)

    print(f"\n=== OASST2 Author Trajectory Cohort ===")
    print(f"  unique authors total:   {len(user_stats)}")
    print(f"  cohort size (N>={args.min_messages_per_author}, span>={args.min_span_days}d): {len(cohort)}")
    print(f"  cohort total messages:  {int(cohort['n_messages'].sum())}")
    print(f"  cohort mean span:       {cohort['span_days'].mean():.1f} days")
    print(f"  cohort mean quality:    {cohort['mean_quality'].mean():.3f}")
    print(f"  top 5 authors by #msg:  {cohort.head(5)['n_messages'].tolist()}")
    print(f"  saved: {cohort_path}")

    # Time-binned mean quality (monthly) — proxy for reviewer drift signal
    df["month"] = df["created_ts"].dt.to_period("M")
    monthly = df.groupby("month").agg(
        n_messages=("message_id", "size"),
        mean_quality=("quality", "mean"),
        std_quality=("quality", "std"),
    )
    monthly_path = out_dir / "oasst2_monthly_quality.parquet"
    monthly.to_parquet(monthly_path)
    print(f"\n=== Monthly aggregate (first sign of reviewer drift if monotone) ===")
    print(monthly.to_string())
    print(f"\nsaved: {monthly_path}")


if __name__ == "__main__":
    main()
