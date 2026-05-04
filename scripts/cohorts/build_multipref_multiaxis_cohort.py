"""Build a per-annotation MultiPref cohort enriched with multiple scalar axes
for the T3 multi-axis drift scan (iter+N+192).

Per-annotation axes emitted (all day-granularity, one row per evaluator x
comparison annotation):

  Continuous (confidence-Likert converted to {0.33, 0.66, 1.0}):
    - overall_conf       (= overall_confidence)   ALREADY in original parquet
    - helpful_conf                                 NEW
    - truthful_conf                                NEW
    - harmless_conf                                NEW
    - mean_conf          (= mean of 4 above)       ALREADY

  Effort / engagement:
    - time_spent         (seconds)                 ALREADY
    - log_time_spent     (ln(time_spent + 1))      NEW
    - total_reasons_checked  (sum of |*_checked_reasons|)  NEW

  Decisiveness (|pref_score − 0.5| ∈ {0, 0.25, 0.5}; aspect-specific):
    - overall_decisiveness                         NEW
    - helpful_decisiveness                         NEW
    - truthful_decisiveness                        NEW
    - harmless_decisiveness                        NEW

  Tie-lazy indicators (binary; an evaluator who "tires" might tie more):
    - is_tie_overall                               NEW
    - is_tie_helpful                               NEW
    - is_tie_truthful                              NEW
    - is_tie_harmless                              NEW

Conceptually these probe four distinct drift modes that may dissociate:
  (a) confidence drift      (careful-review)
  (b) effort drift          (time_spent / reasons)
  (c) decisiveness drift    (polarisation of judgement)
  (d) tie-rate drift        (engagement floor)

If OASST2's multi-dimensional evaluator re-calibration generalises, (a)-(d)
should NOT move in lockstep. If MultiPref's month is too short for genuine
drift, most axes should be null (honest report).

Output:
  - data/multipref_multiaxis_quality.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CONF_TO_SCORE = {
    "absolutely-confident": 1.0,
    "fairly-confident": 0.66,
    "not-confident": 0.33,
}

PREF_TO_SCORE = {
    "A-is-clearly-better": 1.0,
    "A-is-slightly-better": 0.75,
    "Tie": 0.5,
    "B-is-slightly-better": 0.25,
    "B-is-clearly-better": 0.0,
}


def _conf(s):
    return CONF_TO_SCORE.get(s, float("nan"))


def _pref(s):
    return PREF_TO_SCORE.get(s, float("nan"))


def _len_or_zero(x):
    if x is None:
        return 0
    try:
        return len(x)
    except TypeError:
        return 0


def iter_rows(ds):
    for row in ds:
        for kind, anns in [
            ("normal", row["normal_worker_annotations"]),
            ("expert", row["expert_worker_annotations"]),
        ]:
            for a in anns:
                # Confidence axes
                overall_c = _conf(a.get("overall_confidence"))
                helpful_c = _conf(a.get("helpful_confidence"))
                truthful_c = _conf(a.get("truthful_confidence"))
                harmless_c = _conf(a.get("harmless_confidence"))
                confs = [c for c in (overall_c, helpful_c, truthful_c, harmless_c)
                        if c == c]
                mean_conf = float(np.mean(confs)) if confs else float("nan")

                # Preference axes → decisiveness
                op = _pref(a.get("overall_pref"))
                hp = _pref(a.get("helpful_pref"))
                tp = _pref(a.get("truthful_pref"))
                mp = _pref(a.get("harmless_pref"))

                def dec(x):
                    return float("nan") if x != x else abs(x - 0.5)

                def tie(x):
                    return float("nan") if x != x else float(x == 0.5)

                # Effort axes
                ts = float(a.get("time_spent", float("nan")))
                log_ts = float(np.log1p(ts)) if ts == ts and ts >= 0 else float("nan")

                reasons = (
                    _len_or_zero(a.get("harmless_checked_reasons"))
                    + _len_or_zero(a.get("helpful_checked_reasons"))
                    + _len_or_zero(a.get("truthful_checked_reasons"))
                )

                yield {
                    "comparison_id": row["comparison_id"],
                    "category": row["category"],
                    "source": row["source"],
                    "evaluator": a["evaluator"],
                    "kind": kind,
                    "timestamp": a["timestamp"],
                    # Continuous axes
                    "overall_conf": overall_c,
                    "helpful_conf": helpful_c,
                    "truthful_conf": truthful_c,
                    "harmless_conf": harmless_c,
                    "mean_conf": mean_conf,
                    "time_spent": ts,
                    "log_time_spent": log_ts,
                    "total_reasons_checked": reasons,
                    # Decisiveness axes
                    "overall_decisiveness": dec(op),
                    "helpful_decisiveness": dec(hp),
                    "truthful_decisiveness": dec(tp),
                    "harmless_decisiveness": dec(mp),
                    # Tie-rate axes
                    "is_tie_overall": tie(op),
                    "is_tie_helpful": tie(hp),
                    "is_tie_truthful": tie(tp),
                    "is_tie_harmless": tie(mp),
                }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dataset", default="allenai/multipref")
    ap.add_argument("--output-dir", default="data")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    print(f"[load] {args.hf_dataset}")
    ds = load_dataset(args.hf_dataset, split="train")
    print(f"[load] {len(ds)} comparisons")

    df = pd.DataFrame(list(iter_rows(ds)))
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    t0 = df["ts"].min()
    df["day_num"] = ((df["ts"] - t0).dt.total_seconds() / 86400.0).astype(float)
    df["is_expert"] = (df["kind"] == "expert").astype(float)
    df["user_id"] = df["evaluator"]

    out_path = out_dir / "multipref_multiaxis_quality.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[save] {out_path}  ({len(df)} rows)")

    axes = [
        "overall_conf", "helpful_conf", "truthful_conf", "harmless_conf",
        "mean_conf", "time_spent", "log_time_spent", "total_reasons_checked",
        "overall_decisiveness", "helpful_decisiveness",
        "truthful_decisiveness", "harmless_decisiveness",
        "is_tie_overall", "is_tie_helpful",
        "is_tie_truthful", "is_tie_harmless",
    ]

    print("\n=== Per-axis coverage + summary ===")
    for a in axes:
        s = df[a]
        nn = s.notna().sum()
        cov = nn / len(df)
        if s.dtype == bool or s.dtype == object:
            print(f"  {a:<28s} cov={cov:5.1%}  (n={nn})")
        else:
            print(f"  {a:<28s} cov={cov:5.1%}  mean={s.mean():+.4f}  std={s.std():.4f}")


if __name__ == "__main__":
    main()
