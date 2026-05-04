"""Build per-author, per-label-axis quality trajectory cohort from OASST2 ready trees.

Extends `build_author_trajectory_cohort.py` by extracting *all* OASST2 label axes
(quality, humor, creativity, violence, toxicity, helpfulness, ...) into a wide
parquet. Each row = one message, with a column per axis holding the aggregated
reviewer mean for that axis on that message (or NaN if no annotators scored
that axis).

Output:
  - data/oasst2_multiaxis_quality.parquet  (wide: user_id, created_date, role,
    lang, plus one float column per axis + an {axis}_n_raters count column)
  - data/oasst2_multiaxis_cohort.parquet   (same power-author cohort as base
    builder, recomputed from the multi-axis parquet; index=user_id)
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd


# 13 label axes present in 2023-11-05 OASST2 ready.trees. We keep all except
# spam/pii/hate_speech/sexual_content which are near-zero-variance ("safety
# floor" axes — almost all annotators say 0) and would yield trivially null
# drift. We DO keep toxicity + violence + not_appropriate since those have
# moderate variance. The final shortlist is:
AXES = [
    "quality",
    "humor",
    "creativity",
    "violence",
    "toxicity",
    "helpfulness",
    "lang_mismatch",
    "not_appropriate",
    "fails_task",
    # low-variance but kept for completeness; drift test will flag as null
    "hate_speech",
    "sexual_content",
    "pii",
    "spam",
]


def iter_author_rows(trees_path: Path):
    """Yield one row per OASST2 node with *any* label + timestamp."""
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
                if not labels or node.get("user_id") is None:
                    continue
                row = {
                    "user_id": node["user_id"],
                    "message_id": node.get("message_id", ""),
                    "parent_id": node.get("parent_id", ""),
                    "tree_id": str(tree_id),
                    "role": node.get("role", ""),
                    "lang": node.get("lang", ""),
                    "created_date": node.get("created_date", ""),
                    "review_count": node.get("review_count"),
                    "rank": node.get("rank"),
                }
                for axis in AXES:
                    lab = labels.get(axis) or {}
                    v = lab.get("value")
                    c = lab.get("count")
                    row[axis] = float(v) if v is not None else None
                    row[f"{axis}_n_raters"] = int(c) if c else 0
                yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees-path", required=True,
                    help="Path to 2023-11-05_oasst2_ready.trees.jsonl.gz")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--min-messages-per-author", type=int, default=20)
    ap.add_argument("--min-span-days", type=int, default=28)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trees_path = Path(args.trees_path)
    print(f"[parse] {trees_path}")

    rows = list(iter_author_rows(trees_path))
    print(f"[parse] {len(rows)} rows with at least one label")

    df = pd.DataFrame(rows)
    df["created_ts"] = pd.to_datetime(df["created_date"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_ts"]).sort_values("created_ts").reset_index(drop=True)

    # coverage per axis
    print("\n=== Per-axis coverage ===")
    for axis in AXES:
        n_non_na = int(df[axis].notna().sum())
        frac = 100 * n_non_na / len(df)
        print(f"  {axis:<20s}: {n_non_na:>7,} rows ({frac:.1f}%)  "
              f"mean={df[axis].mean():.3f}  std={df[axis].std():.3f}")

    all_path = out_dir / "oasst2_multiaxis_quality.parquet"
    df.to_parquet(all_path)
    print(f"\n[save] {all_path} ({len(df)} rows, {len(df.columns)} cols)")

    # Cohort filter (same as base script: ≥20 msgs, ≥28-day span)
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
    cohort_path = out_dir / "oasst2_multiaxis_cohort.parquet"
    cohort.to_parquet(cohort_path)

    print(f"\n=== OASST2 Multi-axis Author Cohort ===")
    print(f"  authors total:     {len(user_stats)}")
    print(f"  cohort size:       {len(cohort)} "
          f"(N>={args.min_messages_per_author}, span>={args.min_span_days}d)")
    print(f"  cohort messages:   {int(cohort['n_messages'].sum())}")
    print(f"  cohort span (mean):{cohort['span_days'].mean():.1f} days")
    print(f"  saved: {cohort_path}")


if __name__ == "__main__":
    main()
