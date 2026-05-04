"""Extract policy/clipfrac_avg distributions from 7B PPO backfill logs (iter+N+172).

Motivation: adversarial-review attack #3 — "clipped PPO ≠ Boltzmann fixed
point" — demands we disclose the clipping-binding rate in training to defend
Prop T1.MI's asymptotic-under-non-binding-clipping scope. We parse the
training log for each seed × arm, extract the per-step `policy/clipfrac_avg`
series, and summarise mean / max / p95 / fraction-of-steps-above-5%.

A LOW clipfrac (say mean < 0.02, p95 < 0.05) means the clipped surrogate
was essentially unclipped, so the theoretical Boltzmann-FP argument
applies to very good approximation. A HIGH clipfrac (mean > 0.2) means
the surrogate was consistently clipped and T1.MI's asymptotic scope is
contaminated.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = Path("/tmp/backfill_nohup.log")  # pulled from H100 via scp

# Regex: `'policy/clipfrac_avg': 0.0188964381814003,`
CLIPFRAC_RE = re.compile(r"'policy/clipfrac_avg': ([0-9.eE+-]+)")

# Markers (from grep -nE "seed=|DONE|FAIL|BT-LL|SKIP")
# Section start = "=== seed=X arm=Y ..." or "=== BT-LL ..." (BT-LL is eval, skip)
# Section end   = "DONE seed=X arm=Y", "FAIL seed=X arm=Y", or next "===" line

SECTION_START = re.compile(r"=== seed=(\d+) arm=(\w+) ")
SECTION_END = re.compile(r"(DONE|FAIL) seed=(\d+) arm=(\w+)")


def parse():
    text = LOG_PATH.read_text().splitlines()
    sections = []
    cur = None
    for i, line in enumerate(text):
        m = SECTION_START.search(line)
        if m:
            if cur is not None and cur.get("clipfracs"):
                sections.append(cur)
            cur = {"seed": int(m.group(1)), "arm": m.group(2),
                   "start_line": i, "clipfracs": []}
            continue
        m = SECTION_END.search(line)
        if m and cur is not None:
            cur["end_line"] = i
            cur["status"] = m.group(1)
            sections.append(cur)
            cur = None
            continue
        if cur is not None:
            m = CLIPFRAC_RE.search(line)
            if m:
                cur["clipfracs"].append(float(m.group(1)))
    if cur is not None and cur.get("clipfracs"):
        sections.append(cur)
    return sections


def summarize(sections):
    import numpy as np
    out = []
    for s in sections:
        arr = np.array(s["clipfracs"])
        if len(arr) == 0:
            continue
        out.append({
            "seed": s["seed"],
            "arm": s["arm"],
            "status": s.get("status", "PARTIAL"),
            "n_steps": len(arr),
            "clipfrac_mean": float(arr.mean()),
            "clipfrac_max": float(arr.max()),
            "clipfrac_p95": float(np.percentile(arr, 95)),
            "frac_steps_above_5pct": float((arr > 0.05).mean()),
            "frac_steps_above_10pct": float((arr > 0.10).mean()),
        })
    return out


def main():
    sections = parse()
    summary = summarize(sections)
    print("| seed | arm | status | n_steps | mean | max | p95 | frac>5% | frac>10% |")
    print("|-----:|:----|:-------|--------:|-----:|----:|----:|-------:|--------:|")
    for r in summary:
        print(f"| {r['seed']} | {r['arm']} | {r['status']} | {r['n_steps']} | "
              f"{r['clipfrac_mean']:.4f} | {r['clipfrac_max']:.4f} | "
              f"{r['clipfrac_p95']:.4f} | {r['frac_steps_above_5pct']:.2%} | "
              f"{r['frac_steps_above_10pct']:.2%} |")

    out_path = ROOT / "results" / "track1_ppo_7b" / "clipfrac_distribution.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    # aggregate non-failed
    import numpy as np
    good = [r for r in summary if r["status"] == "DONE"]
    if good:
        agg = {
            "n_runs": len(good),
            "pooled_mean": float(np.mean([r["clipfrac_mean"] for r in good])),
            "pooled_max": float(max(r["clipfrac_max"] for r in good)),
            "pooled_p95": float(np.percentile([r["clipfrac_p95"] for r in good], 50)),
            "worst_frac_above_5pct": float(max(r["frac_steps_above_5pct"] for r in good)),
            "worst_frac_above_10pct": float(max(r["frac_steps_above_10pct"] for r in good)),
        }
        print(f"\n[DONE runs agg]: mean={agg['pooled_mean']:.4f}  "
              f"p95={agg['pooled_p95']:.4f}  max={agg['pooled_max']:.4f}  "
              f"worst-frac>5%={agg['worst_frac_above_5pct']:.2%}  "
              f"worst-frac>10%={agg['worst_frac_above_10pct']:.2%}")
        out_path2 = ROOT / "results" / "track1_ppo_7b" / "clipfrac_aggregate.json"
        out_path2.write_text(json.dumps(agg, indent=2))
        print(f"[ok] wrote {out_path}, {out_path2}")


if __name__ == "__main__":
    main()
