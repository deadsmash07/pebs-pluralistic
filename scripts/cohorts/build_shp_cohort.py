"""Build per-subreddit preference-pair trajectory cohort from Stanford SHP.

Dataset: `stanfordnlp/SHP` on HuggingFace (Ethayarajh, Choi, Swayamdipta 2022,
"Stanford Human Preferences Dataset" — SHP/SHP-2). 348,718 Reddit-derived
preference pairs across 18 `ask*` / `changemyview` / `explainlikeimfive`
subreddits, collected 2011-02-11 → 2023-01-01 (~12 years).

Schema:
  post_id         str     Reddit submission id
  domain          str     subreddit_train / _validation / _test
  history         str     Reddit post text (the "question")
  upvote_ratio    float   post-level upvote ratio
  c_root_id_A/B   str     comment root ids
  created_at_utc_A/B  int UTC seconds when each comment was posted
  score_A/B       int     upvote count for each comment
  labels          int     0 or 1 preference label
  seconds_difference  f   |ts_A - ts_B|
  score_ratio     f       max(score_A,score_B)/min(...), ≥1

Rater-identity / drift framing
------------------------------
Each SHP pair was "judged" by the aggregate of Reddit upvoters in that
subreddit (no single-user ID exposed in the dataset). The natural random-
effect grouping is therefore the **subreddit domain** — each subreddit is a
distinct community of voters with its own norms, size, and time envelope.
This matches the OASST2/MultiPref pattern "many observations per rater-cohort
over time" but at the community level rather than individual-annotator level.

We treat one pair = one observation with:
  user_id      = stripped domain (e.g. "askengineers" — join train+val+test)
  month_num    = months-since-2010-01-01, based on min(ts_A, ts_B)
  quality      = log1p(max(score_A, score_B))  — winner upvote count as
                 preference-strength proxy; log to tame Reddit's heavy tail.
  log_score_ratio = log(score_ratio)  — relative separation between winner/
                 loser, independent of subreddit popularity.

Both proxies are stored; the detector script picks one (default quality).

Cohort filter (defaults):
  n ≥ 100 pairs per subreddit AND span ≥ 12 months.
  With 18 subreddits, all of them pass easily; the filter is future-proof.

Output:
  - data/shp_domain_quality.parquet   (all pairs, per-pair granularity)
  - data/shp_domain_cohort.parquet    (cohort domain index + stats)
  - data/shp_monthly_quality.parquet  (monthly aggregate, raw series)

References
----------
  - Ethayarajh, Choi, Swayamdipta 2022 SHP (arXiv:2212.09251, NeurIPS
    DaSH-LIT workshop) / huggingface dataset card
  - Pinheiro & Bates 2000 Mixed-Effects Models in S and S-PLUS (MixedLM)
  - Kittur, Chi & Suh 2008 (per-cohort community-norm drift in crowdwork)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _strip_split(domain: str) -> str:
    """`askengineers_train` -> `askengineers` (combine train/val/test)."""
    for suf in ("_train", "_validation", "_test"):
        if domain.endswith(suf):
            return domain[: -len(suf)]
    return domain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dataset", default="stanfordnlp/SHP")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument(
        "--min-pairs-per-domain", type=int, default=100,
        help="Minimum preference pairs per subreddit to qualify for cohort",
    )
    ap.add_argument(
        "--min-span-months", type=float, default=12.0,
        help="Minimum temporal span per subreddit",
    )
    ap.add_argument(
        "--include-splits", default="train,validation,test",
        help="Comma-separated HF splits to union",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    dfs: list[pd.DataFrame] = []
    for split in args.include_splits.split(","):
        split = split.strip()
        if not split:
            continue
        print(f"[load] {args.hf_dataset}:{split}")
        ds = load_dataset(args.hf_dataset, split=split)
        d = ds.to_pandas()
        d["split"] = split
        dfs.append(d)
    raw = pd.concat(dfs, ignore_index=True)
    print(f"[load] total rows: {len(raw)}")

    df = pd.DataFrame()
    df["post_id"] = raw["post_id"]
    df["domain_raw"] = raw["domain"]
    df["user_id"] = raw["domain"].map(_strip_split)
    df["label"] = raw["labels"]
    df["score_A"] = raw["score_A"].astype(float)
    df["score_B"] = raw["score_B"].astype(float)
    df["ts_A"] = raw["created_at_utc_A"].astype("Int64")
    df["ts_B"] = raw["created_at_utc_B"].astype("Int64")
    df["score_ratio"] = raw["score_ratio"].astype(float)

    # Winner properties + pair timestamp (min of A/B — when the thread opened)
    winner_score = np.where(df["label"] == 1, df["score_A"], df["score_B"])
    loser_score = np.where(df["label"] == 1, df["score_B"], df["score_A"])
    ts_min = np.minimum(df["ts_A"].astype(float), df["ts_B"].astype(float))
    df["winner_score"] = winner_score
    df["loser_score"] = loser_score
    df["ts"] = pd.to_datetime(ts_min, unit="s", utc=True)
    df["log_winner_score"] = np.log1p(winner_score)
    df["log_score_ratio"] = np.log(df["score_ratio"].clip(lower=1.0))

    # Time features
    anchor = pd.Timestamp("2010-01-01", tz="UTC")
    df["month_num"] = (
        (df["ts"] - anchor).dt.total_seconds() / (30.4375 * 86400.0)
    ).astype(float)
    df["year"] = df["ts"].dt.year.astype(int)

    # Default "quality" = log_winner_score (rater-cohort's absolute preference
    # strength, monotone in upvote count). log_score_ratio stays as a second
    # proxy.
    df["quality"] = df["log_winner_score"]

    df = df.dropna(subset=["ts", "quality", "user_id"]).sort_values("ts").reset_index(drop=True)

    all_path = out_dir / "shp_domain_quality.parquet"
    df.to_parquet(all_path, index=False)
    print(f"[save] {all_path} ({len(df)} rows)")

    # Cohort filter (span months, not years)
    stats = df.groupby("user_id").agg(
        n_pairs=("post_id", "size"),
        tmin=("ts", "min"),
        tmax=("ts", "max"),
        mean_winner_score=("winner_score", "mean"),
        mean_log_winner=("log_winner_score", "mean"),
        mean_log_ratio=("log_score_ratio", "mean"),
    )
    stats["span_months"] = (
        (stats["tmax"] - stats["tmin"]).dt.total_seconds() / (30.4375 * 86400.0)
    ).astype(float)
    cohort = stats[
        (stats["n_pairs"] >= args.min_pairs_per_domain)
        & (stats["span_months"] >= args.min_span_months)
    ].sort_values("n_pairs", ascending=False)

    cohort_path = out_dir / "shp_domain_cohort.parquet"
    cohort.to_parquet(cohort_path)

    print(f"\n=== SHP Domain Trajectory Cohort ===")
    print(f"  unique domains total: {len(stats)}")
    print(f"  cohort (n≥{args.min_pairs_per_domain}, "
          f"span≥{args.min_span_months}mo): {len(cohort)}")
    print(f"  cohort total pairs: {int(cohort['n_pairs'].sum())}")
    print(f"  cohort mean span: {cohort['span_months'].mean():.1f} months")
    print(f"  mean log_winner: {cohort['mean_log_winner'].mean():.3f}")
    print(f"  saved: {cohort_path}")

    # Monthly aggregate (raw series)
    df["month_bucket"] = df["month_num"].astype(int)
    monthly = df.groupby("month_bucket").agg(
        n_pairs=("post_id", "size"),
        mean_quality=("quality", "mean"),
        std_quality=("quality", "std"),
        mean_log_ratio=("log_score_ratio", "mean"),
    )
    monthly_path = out_dir / "shp_monthly_quality.parquet"
    monthly.to_parquet(monthly_path)
    print(f"\n=== Monthly aggregate (for raw-signal plot) ===")
    print(monthly.tail(10).to_string())
    print(f"\nsaved: {monthly_path}")


if __name__ == "__main__":
    main()
