"""Q6 cross-backbone wrapper - run Q6 DPO downstream-impact on multiple base models.

Background
----------
The original Q6 (`q6_dpo_downstream_impact.py`, anchor commit `aa2af97`) was run on
NousResearch/Meta-Llama-3-8B-Instruct and produced verdict
MODERATE-DPO-PILSD-PARTIAL-DOMINATES with held-out pair-accuracy delta
+8.81pp [+7.03, +10.64] (single-seed; G3 silent-bypass PASS frac=0.461).

This wrapper extends Q6 to ADDITIONAL backbones (Qwen-2.5-7B-Instruct + Mistral-7B-
Instruct-v0.3) using the IDENTICAL PRISM-LOCO matched protocol, IDENTICAL PILSD
calibrators (`prism_user_calibrators_shrunk.parquet`, T1 anchor 9c92523), IDENTICAL
DPO hyperparameters, IDENTICAL seed, IDENTICAL data splits. The ONLY thing that
varies across runs is the BASE MODEL.

Hypothesis
----------
PILSD's gain is reward-side (per-user shrinkage of RM scores; backbone-agnostic).
Therefore the downstream pair-accuracy gain (~+8.8pp on Llama-3-8B) should
TRANSFER to other backbones. Predicted ~+5-10pp per backbone.

Combined cross-backbone evidence
--------------------------------
- 1/3 backbones positive: SCOPE-LIMITED (Llama-specific finding; honest disclosure)
- 2/3 backbones positive: MODERATE-CROSS-BACKBONE
- 3/3 backbones positive: ESTABLISHED-CROSS-BACKBONE-DOMINATES

Per-backbone 4-class STRICT verdict assigned by the original
`assign_verdict()` from `q6_dpo_downstream_impact.py` (verbatim re-use; no
verdict-logic divergence).

EXECUTIVE DECISION rationale (wrapper vs script-modify)
-------------------------------------------------------
Chose (B) WRAPPER because:
- Original Q6 script `q6_dpo_downstream_impact.py` is anchor-load-bearing for the
  Llama-3-8B canonical result. Modifying it risks breaking reproducibility of the
  existing verdict.
- Wrapper imports `main()` from the original script per backbone with a different
  `--base-model-id` and `--output-dir`. ZERO duplication of training/eval logic.
- Per-backbone output dirs prevent pair-parquet collision.
- All 12 G-gate properties of the original script (G1-G12) inherit verbatim
  because the underlying training/eval code is unchanged.
- Cross-backbone aggregation is computed POST-HOC by reading per-backbone
  summary.json files; no shared training state.

Script architecture
-------------------
1. Parse `--base_model_ids` (comma-separated list of HF model IDs).
2. For each model:
   a. Set sys.argv to invoke q6_dpo_downstream_impact.main() with backbone-specific
      base_model_id + output_dir.
   b. Call q6_dpo_downstream_impact.main() in-process (preserves all G-gate
      properties; no subprocess overhead; shares HF cache; one-shot Python).
3. After all backbones complete, read per-backbone summary.json files and write a
   combined cross-backbone summary at
   `results/track1_q6_cross_backbone_summary/cross_backbone_summary.json`.

NO INTERNAL KILL SWITCHES per user 2026-05-02 12:08 IST.
- No internal time-based abort branches.
- No silent skip flags that disable an arm.
- No max-hours upper bound on training.
- The only non-zero "TIMEOUT" in the inherited script is the AlpacaEval-2 subprocess
  2h judge timeout, which is bypassed by --skip-alpaca-eval (default for this
  cross-backbone run; AE2 deferred to camera-ready).

Skill citations
---------------
- Skill: research-grade-code-audit-pre-launch v1 G1-G12 (inherited from parent script)
- Skill: launch-runpod-h100-job (h100_v2_backup setsid+nohup+disown)
- Skill: gpu-artifact-sync (verdict-landing rsync after completion)
- Skill: post-experiment-discipline-3-track Step 4-7 (per-backbone addendum)
- Skill: honest-disclosure 4-class STRICT 6.3 (per-backbone verdict; cross-backbone
  combined verdict)

Output
------
results/track1_q6_dpo_downstream_impact_qwen25_7b/{summary.json, ...}
results/track1_q6_dpo_downstream_impact_mistral_7b/{summary.json, ...}
results/track1_q6_cross_backbone_summary/cross_backbone_summary.json

Usage
-----
python paper/scripts/q6_cross_backbone.py \
    --base_model_ids Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3 \
    --skip-alpaca-eval

Or smoke (50 pairs train / 20 pairs eval per backbone):
python paper/scripts/q6_cross_backbone.py \
    --base_model_ids Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3 \
    --skip-alpaca-eval --smoke
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

# Ensure the parent script's directory is on sys.path so we can import it.
_THIS = Path(__file__).resolve()
_SCRIPTS_DIR = _THIS.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Import the parent script. We re-use main() per backbone in-process.
import q6_dpo_downstream_impact as q6  # noqa: E402

ROOT = _THIS.resolve().parents[2]
CROSS_OUT_DIR = ROOT / "results" / "track1_q6_cross_backbone_summary"


def short_name(model_id: str) -> str:
    """Convert HF model id to a short filesystem-safe slug for output dirs.

    Examples:
      'Qwen/Qwen2.5-7B-Instruct' -> 'qwen25_7b'
      'mistralai/Mistral-7B-Instruct-v0.3' -> 'mistral_7b'
      'NousResearch/Meta-Llama-3-8B-Instruct' -> 'llama3_8b'
    """
    s = model_id.split("/")[-1].lower()
    s = re.sub(r"[-./]instruct.*$", "", s)
    s = re.sub(r"[-./]chat.*$", "", s)
    s = s.replace("meta-", "").replace(".", "")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def run_one_backbone(base_model_id: str, common_args: dict, smoke: bool) -> dict:
    """Run Q6 once on one backbone via in-process main() invocation."""
    slug = short_name(base_model_id)
    out_dir = ROOT / "results" / f"track1_q6_dpo_downstream_impact_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build sys.argv equivalent to a CLI invocation of q6_dpo_downstream_impact.py.
    argv = ["q6_dpo_downstream_impact.py"]
    argv += ["--base-model-id", base_model_id]
    argv += ["--output-dir", str(out_dir)]
    if common_args.get("skip_alpaca_eval", True):
        argv += ["--skip-alpaca-eval"]
    if smoke:
        argv += ["--smoke"]
    if "utterances_parquet" in common_args:
        argv += ["--utterances-parquet", common_args["utterances_parquet"]]
    if "rm_scored_parquet" in common_args:
        argv += ["--rm-scored-parquet", common_args["rm_scored_parquet"]]
    if "calibrators_parquet" in common_args:
        argv += ["--calibrators-parquet", common_args["calibrators_parquet"]]
    if "seed" in common_args:
        argv += ["--seed", str(common_args["seed"])]

    # Save and replace sys.argv for argparse inside q6.main().
    saved_argv = sys.argv
    sys.argv = argv
    t0 = time.time()
    try:
        print(
            f"\n========================================================================\n"
            f"[CROSS] backbone {base_model_id} -> {out_dir}\n"
            f"[CROSS] argv: {' '.join(argv)}\n"
            f"========================================================================\n",
            flush=True,
        )
        rc = q6.main()
    except SystemExit as exc:
        rc = int(exc.code) if exc.code is not None else 0
    except Exception:
        traceback.print_exc()
        rc = 99
    finally:
        sys.argv = saved_argv
    elapsed = time.time() - t0

    summary_path = out_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            summary = {"verdict_class": "RUNTIME-ERROR-SUMMARY-PARSE-FAILED",
                       "error": str(exc)}
    return {
        "base_model_id": base_model_id,
        "slug": slug,
        "output_dir": str(out_dir),
        "summary_path": str(summary_path),
        "main_return_code": int(rc),
        "wall_seconds": float(elapsed),
        "summary": summary,
    }


def aggregate_cross_backbone(per_backbone_results: list) -> dict:
    """Compute combined cross-backbone verdict-class.

    Combined classes:
      ESTABLISHED-CROSS-BACKBONE-DOMINATES
        iff all backbones >= MODERATE positive (pair-acc delta>=+1pp + CI excludes 0)
      MODERATE-CROSS-BACKBONE-DOMINATES-MAJORITY
        iff strictly more than half are >= MODERATE positive
      PRELIMINARY-CROSS-BACKBONE-MIXED
        iff some positive + some null + zero strict-negative
      FALSIFIED-CROSS-BACKBONE-NULL-OR-WORSE
        iff zero positive backbones AND >=1 strict-negative
      INCONCLUSIVE-CROSS-BACKBONE-INSUFFICIENT
        iff fewer than 2 backbones produced verdicts (e.g., crashes)
    """
    n_total = len(per_backbone_results)
    n_with_verdict = 0
    pos_classes = (
        "ESTABLISHED-DPO-PILSD-DOMINATES-DOWNSTREAM",
        "MODERATE-DPO-PILSD-PARTIAL-DOMINATES",
        "MODERATE-DPO-PILSD-PARTIAL-DOMINATES-AE2",
    )
    neg_classes = (
        "FALSIFIED-DPO-NEUTRAL-OR-WORSE",
        "FALSIFIED-DPO-AE2-NEGATIVE",
    )
    null_classes = (
        "PRELIMINARY-INCONCLUSIVE",
        "PRELIMINARY-INCONCLUSIVE-G3-SILENT-BYPASS-GATE-FAIL",
    )
    n_pos = n_neg = n_null = n_other = 0
    per_backbone_verdicts = []
    for r in per_backbone_results:
        v = (r.get("summary") or {}).get("verdict_class", "MISSING")
        per_backbone_verdicts.append({
            "backbone": r["base_model_id"],
            "slug": r["slug"],
            "verdict_class": v,
            "main_return_code": r["main_return_code"],
            "wall_seconds": r["wall_seconds"],
        })
        if v == "MISSING" or v.startswith("RUNTIME-ERROR"):
            continue
        n_with_verdict += 1
        if v in pos_classes:
            n_pos += 1
        elif v in neg_classes:
            n_neg += 1
        elif v in null_classes:
            n_null += 1
        else:
            n_other += 1

    if n_with_verdict < 2:
        combined = "INCONCLUSIVE-CROSS-BACKBONE-INSUFFICIENT"
    elif n_pos == n_with_verdict:
        combined = "ESTABLISHED-CROSS-BACKBONE-DOMINATES"
    elif n_pos > (n_with_verdict / 2):
        combined = "MODERATE-CROSS-BACKBONE-DOMINATES-MAJORITY"
    elif n_pos == 0 and n_neg >= 1:
        combined = "FALSIFIED-CROSS-BACKBONE-NULL-OR-WORSE"
    elif n_pos >= 1 and n_neg == 0:
        combined = "PRELIMINARY-CROSS-BACKBONE-MIXED"
    else:
        combined = "PRELIMINARY-CROSS-BACKBONE-MIXED"

    # Pull per-backbone headline pair-accuracy deltas.
    per_backbone_pair_acc = []
    for r in per_backbone_results:
        s = r.get("summary") or {}
        pa = (s.get("stages") or {}).get("held_out_pair_accuracy") or {}
        per_backbone_pair_acc.append({
            "backbone": r["base_model_id"],
            "slug": r["slug"],
            "pair_acc_delta_pp": pa.get("delta_pp"),
            "ci95_lo": pa.get("ci95_lo"),
            "ci95_hi": pa.get("ci95_hi"),
            "headline_pair_accuracy_arm_a": pa.get("headline_pair_accuracy_arm_a"),
            "headline_pair_accuracy_arm_b": pa.get("headline_pair_accuracy_arm_b"),
            "ci_excludes_zero_positive": pa.get("ci_excludes_zero_positive"),
        })

    return {
        "experiment_id": "Q6_cross_backbone_dpo_downstream_impact",
        "combined_verdict_class": combined,
        "n_backbones_total": n_total,
        "n_with_verdict": n_with_verdict,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_null": n_null,
        "n_other": n_other,
        "per_backbone_verdicts": per_backbone_verdicts,
        "per_backbone_pair_acc": per_backbone_pair_acc,
        "interpretation": (
            "Cross-backbone test of Q6 hypothesis: PILSD-corrected reward "
            "improves DPO-trained policy's downstream pair-accuracy. PILSD is "
            "reward-side (per-user shrinkage of RM scores; backbone-agnostic), "
            "so per-backbone gains should be SIMILAR to the Llama-3-8B canonical "
            "(+8.81pp). Combined verdict aggregates per-backbone evidence."
        ),
        "skill_citations": [
            "Skill: research-grade-code-audit-pre-launch v1 G1-G12 (inherited)",
            "Skill: honest-disclosure 4-class STRICT 6.3 (cross-backbone combined)",
            "Skill: post-experiment-discipline-3-track Step 4-7",
            "Skill: launch-runpod-h100-job (h100_v2_backup)",
            "Skill: gpu-artifact-sync",
        ],
        "anchors": {
            "q6_llama_canonical": (
                "MODERATE-DPO-PILSD-PARTIAL-DOMINATES; pair-acc delta +8.81pp "
                "[+7.03, +10.64]; G3 PASS frac=0.461; SCRIPT commit aa2af97"
            ),
            "w_b5_prism_canonical": "ESTABLISHED-COMPOUND-NEEDED canonical=8.55%",
            "calibrators_parquet": (
                "/workspace/1_Causal_RLHF/data/prism_user_calibrators_shrunk.parquet "
                "(T1 anchor 9c92523; identical for all backbones)"
            ),
        },
        "honest_disclosure_caveats": [
            "Single-seed run per backbone; multi-seed deferred to camera-ready.",
            "Each backbone uses its native chat template (Qwen2.5 ChatML / "
            "Mistral [INST]/[/INST] / Llama-3 native). DPOTrainer handles "
            "tokenization per its own conventions; both arms within a backbone "
            "use the IDENTICAL chat template, so the within-backbone delta is "
            "template-agnostic.",
            "PRISM data + PILSD calibrators are reward-side; backbone-agnostic; "
            "ARM A vs ARM B difference is exclusively in chosen/rejected pair "
            "selection (not in DPO training procedure).",
            "Mistral-7B has tokenizer.pad_token=None natively; the parent "
            "script's `if pad_token is None: pad_token = eos_token` covers this; "
            "smoke-test verified at G7 stage.",
            "AlpacaEval-2 SKIPPED for cross-backbone (per Llama-canonical run); "
            "downstream signal is per-backbone pair-accuracy on held-out PRISM "
            "users (user-disjoint 20% split per backbone with same seed).",
        ],
        "completion_timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "git_head_t3": q6.git_head_t3(),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base_model_ids",
        type=str,
        required=True,
        help=("Comma-separated list of HF base model IDs to run Q6 on. "
              "Example: Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3"),
    )
    p.add_argument("--utterances-parquet", default=None)
    p.add_argument("--rm-scored-parquet", default=None)
    p.add_argument("--calibrators-parquet", default=None)
    p.add_argument("--seed", type=int, default=q6.CANONICAL_SEED)
    p.add_argument(
        "--skip-alpaca-eval",
        action="store_true",
        help="Skip AlpacaEval-2 generation+judge per backbone (default for cross-backbone)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-test mode: ~80 users / 200 train pairs / 50 eval pairs / batch=1",
    )
    return p.parse_args()


def main():
    args = parse_args()
    base_model_ids = [s.strip() for s in args.base_model_ids.split(",") if s.strip()]
    if not base_model_ids:
        print("[CROSS] FATAL: no base_model_ids parsed", flush=True)
        return 1

    common_args = {
        "skip_alpaca_eval": bool(args.skip_alpaca_eval),
        "seed": int(args.seed),
    }
    if args.utterances_parquet:
        common_args["utterances_parquet"] = args.utterances_parquet
    if args.rm_scored_parquet:
        common_args["rm_scored_parquet"] = args.rm_scored_parquet
    if args.calibrators_parquet:
        common_args["calibrators_parquet"] = args.calibrators_parquet

    print(
        f"[CROSS] starting at "
        f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        f"[CROSS] backbones: {base_model_ids}\n"
        f"[CROSS] common_args: {common_args}\n"
        f"[CROSS] smoke: {args.smoke}",
        flush=True,
    )

    per_backbone_results = []
    for mid in base_model_ids:
        try:
            r = run_one_backbone(mid, common_args, args.smoke)
        except Exception:
            traceback.print_exc()
            r = {
                "base_model_id": mid,
                "slug": short_name(mid),
                "output_dir": str(ROOT / "results" / f"track1_q6_dpo_downstream_impact_{short_name(mid)}"),
                "main_return_code": 99,
                "wall_seconds": 0.0,
                "summary": {"verdict_class": "RUNTIME-ERROR-EXCEPTION-IN-WRAPPER"},
                "error": traceback.format_exc(),
            }
        per_backbone_results.append(r)
        print(
            f"\n[CROSS] backbone {mid} done: "
            f"verdict={(r.get('summary') or {}).get('verdict_class', 'MISSING')} "
            f"wall={r.get('wall_seconds', 0)/60:.1f}min "
            f"rc={r.get('main_return_code', 'NA')}\n",
            flush=True,
        )

    cross = aggregate_cross_backbone(per_backbone_results)
    cross["per_backbone_runs"] = per_backbone_results
    CROSS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CROSS_OUT_DIR / "cross_backbone_summary.json"
    out_path.write_text(json.dumps(cross, indent=2, default=str))
    print(
        f"\n========================================================================\n"
        f"[CROSS] all backbones done. combined_verdict_class={cross['combined_verdict_class']}\n"
        f"[CROSS] summary written -> {out_path}\n"
        f"========================================================================\n",
        flush=True,
    )
    for v in cross["per_backbone_verdicts"]:
        print(
            f"  [{v['slug']}] {v['verdict_class']} "
            f"(rc={v['main_return_code']}, wall={v['wall_seconds']/60:.1f}min)",
            flush=True,
        )
    for pa in cross["per_backbone_pair_acc"]:
        d = pa.get("pair_acc_delta_pp")
        lo = pa.get("ci95_lo")
        hi = pa.get("ci95_hi")
        if d is not None:
            print(
                f"  [{pa['slug']}] pair-acc delta {d:+.2f}pp "
                f"CI=[{lo:+.2f}, {hi:+.2f}]" if lo is not None else f"  [{pa['slug']}] pair-acc delta {d:+.2f}pp",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
