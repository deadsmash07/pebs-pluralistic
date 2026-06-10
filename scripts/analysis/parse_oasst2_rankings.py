"""Parse OASST2 trees.jsonl.gz → per-rater ranking-event dataframe for PEBS.

**Critical**: per-rater `user_id` is ONLY in the raw `trees.jsonl.gz` (inside
each node's `events` field). The flattened parquet release strips it. See
the schema-discovery notes.

We extract one row per ranking event:
  user_id, tree_id, parent_message_id, created_date, ranking_parent_id,
  ranked_message_ids (list), ranking (list of ints)

Output: `data/oasst2_rankings.parquet` — the master table for all downstream
PEBS analyses (cohort filter, anchor-coherence detector, etc.).

Cohort filter applied separately (`scripts/oasst2_cohort_analysis.py`) so the
full unfiltered data is always preserved.

Reference:
  - Köpf et al. 2023 "OpenAssistant Conversations" arXiv:2304.07327
  - LAION-AI/Open-Assistant `oasst-data/oasst_data/schemas.py`
    specifically `ExportMessageEventRanking` + `ExportMessageNode.events`

Run (CPU-only, ~1-2 min):
  source ~/venv_pebs/bin/activate
  cd ~/IMPLEMENTATION/3_PEBS_Standalone
  python scripts/parse_oasst2_rankings.py --output-dir data
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


def download_oasst2(cache_dir: Path, file_pattern: str = "2023-11-05_oasst2_all.trees.jsonl.gz") -> Path:
    """Snapshot the OASST2 trees file from HF Hub."""
    from huggingface_hub import hf_hub_download

    print(f"[download] HF: OpenAssistant/oasst2 → {file_pattern}")
    local = hf_hub_download(
        repo_id="OpenAssistant/oasst2",
        repo_type="dataset",
        filename=file_pattern,
        cache_dir=str(cache_dir),
    )
    print(f"[download] local: {local} ({os.path.getsize(local) / 1e6:.1f} MB)")
    return Path(local)


def iter_trees(path: Path) -> Iterable[dict]:
    """Stream the trees file (one tree per line)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def walk_nodes(node: dict, parent_msg_id: str | None = None):
    """Recursive DFS over a tree's prompter → assistant → ... structure.

    Yields (node, parent_msg_id_of_this_node) tuples.
    """
    yield node, parent_msg_id
    msg_id = node.get("message_id")
    for child in node.get("replies", []) or []:
        yield from walk_nodes(child, msg_id)


def extract_rankings_from_tree(tree_row: dict) -> list[dict]:
    """Extract all ranking events from one tree JSON object.

    Tree structure (from OASST2 export schema):
      {
        "message_tree_id": str,
        "prompt": ExportMessageNode  # root prompter message, with `replies`
      }

    Each node has an `events` dict/list keyed by event type. Ranking events
    may live at `events["ranking"]` (list[ExportMessageEventRanking]) OR in a
    flat `events: [...]` list depending on the export version. We handle both.
    """
    tree_id = tree_row.get("message_tree_id") or tree_row.get("id")
    root = tree_row.get("prompt") or tree_row.get("tree") or tree_row
    out = []

    for node, parent_msg_id in walk_nodes(root):
        node_id = node.get("message_id")
        created = node.get("created_date") or node.get("created_at")
        events = node.get("events")
        if events is None:
            continue

        # Schema variant A: events is a dict keyed by type
        if isinstance(events, dict):
            ranking_events = events.get("ranking", []) or []
        # Schema variant B: events is a flat list of typed records
        elif isinstance(events, list):
            ranking_events = [e for e in events if isinstance(e, dict) and e.get("type") == "ranking"]
        else:
            ranking_events = []

        for ev in ranking_events:
            if not isinstance(ev, dict):
                continue
            uid = ev.get("user_id")
            if uid is None:
                continue
            out.append({
                "tree_id": str(tree_id) if tree_id else "",
                "parent_message_id": str(parent_msg_id) if parent_msg_id else "",
                "this_node_id": str(node_id) if node_id else "",
                "created_date": str(created) if created else "",
                "user_id": str(uid),
                "ranking_parent_id": str(ev.get("ranking_parent_id") or ""),
                "ranked_message_ids": list(ev.get("ranked_message_ids") or []),
                "ranking": list(ev.get("ranking") or []),
                "not_rankable": bool(ev.get("not_rankable") or False),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/data/hf-cache"))
    ap.add_argument("--file-pattern", default="2023-11-05_oasst2_all.trees.jsonl.gz")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    local_path = download_oasst2(Path(args.cache_dir), args.file_pattern)

    all_rows = []
    n_trees = 0
    for tree in iter_trees(local_path):
        n_trees += 1
        all_rows.extend(extract_rankings_from_tree(tree))
    print(f"[parse] scanned {n_trees} trees → {len(all_rows)} ranking events")

    if len(all_rows) == 0:
        print("[parse] WARNING: zero ranking events extracted — schema may be different. "
              "Dumping one raw tree for inspection:")
        for tree in iter_trees(local_path):
            root = tree.get("prompt") or {}
            print(json.dumps(root, indent=2, default=str)[:3000])
            break
        return

    df = pd.DataFrame(all_rows)
    # Parse timestamp for longitudinal sorting
    df["created_ts"] = pd.to_datetime(df["created_date"], errors="coerce", utc=True)

    out_parquet = out_dir / "oasst2_rankings.parquet"
    df.to_parquet(out_parquet)
    print(f"[save] {out_parquet} ({len(df)} rows)")

    # Summary stats
    n_users = df["user_id"].nunique()
    per_user = df.groupby("user_id").size()
    date_range = (df["created_ts"].min(), df["created_ts"].max())
    print(f"\n=== OASST2 Ranking-Event Summary ===")
    print(f"  total ranking events:  {len(df)}")
    print(f"  unique users:          {n_users}")
    print(f"  mean events/user:      {per_user.mean():.1f}")
    print(f"  median events/user:    {per_user.median():.0f}")
    print(f"  top 1% users contribute: {per_user.sort_values(ascending=False).head(max(1, n_users // 100)).sum() / len(df):.1%}")
    print(f"  timestamp range:       {date_range[0]} → {date_range[1]}")

    # Power-user cohort preview (N>=20 events over >=4 weeks)
    user_stats = df.groupby("user_id").agg(
        n_events=("user_id", "size"),
        first_ts=("created_ts", "min"),
        last_ts=("created_ts", "max"),
    )
    user_stats["span_days"] = (user_stats["last_ts"] - user_stats["first_ts"]).dt.days
    cohort = user_stats[(user_stats["n_events"] >= 20) & (user_stats["span_days"] >= 28)]
    cohort_path = out_dir / "oasst2_cohort_users.parquet"
    cohort.to_parquet(cohort_path)
    print(f"\n=== PEBS Power-User Cohort (N>=20 events, >=28 days span) ===")
    print(f"  cohort size:           {len(cohort)} users")
    print(f"  cohort total events:   {cohort['n_events'].sum()}")
    print(f"  cohort mean span:      {cohort['span_days'].mean():.1f} days")
    print(f"  saved: {cohort_path}")


if __name__ == "__main__":
    main()
