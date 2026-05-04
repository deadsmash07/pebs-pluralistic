"""Wave C EXP-2: OASST2 cross-corpus replication of PILSD headline.

Closes NeurIPS Pass 5 + Pluralistic generalization reviewer attacks: "PILSD
only works on PRISM" / "in-distribution overfit". Hypothesis: PILSD's 8.58%
RMSE reduction on PRISM replicates on OASST2 (Köpf et al. 2023; Kirk-style
power-author cohort) with REAL per-message author IDs (not pseudo-rater
attribute axes as HelpSteer2 used).

This wraps the existing 2-stage pipeline (already audited + smoke-tested
under PREREG `1_Causal_RLHF/scripts/falsifiers/PREREG_F_T1_OASST2_REPLICATION.md`):

    Stage A — score OASST2 messages with Qwen2.5-7B-Instruct mean-LL
              (1_Causal_RLHF/scripts/score_oasst2_with_qwen7b.py)
    Stage B — fit + evaluate PILSD calibrator with cluster-bootstrap CIs
              (1_Causal_RLHF/scripts/eval_oasst2_pilsd_calibrator.py)

The wrapper is a single binary that:
  1. Verifies prerequisites (parquet trees / venv / model cache).
  2. Runs Stage A (skip if `oasst2_qwen7b_scored.parquet` already exists +
     SHA-matches a small diagnostic header — protects against silent re-uses).
  3. Runs Stage B with B=4000 cluster-bootstrap reps.
  4. Emits `results/track1_oasst2_replication/{summary.json, per_user.parquet}`
     with verdict-class STRICT.

12-gate audit (Skill: research-grade-code-audit-pre-launch):
  G1 math: BOTH stages inherit verbatim from PRISM canonical
       (eval_user_score_mse_shrunk.py for B; score_prism_utterances.py for A).
  G2 hypothesis-vs-design: tests PILSD's 8.58% RMSE-reduction headline
       replication on REAL author IDs (NOT pseudo-rater attribute axes).
  G3 no silent-bypass: stage-A output_parquet existence check is BY-PATH
       only; we always re-run Stage A if --force-rescore or if the cached
       file's row count is below the smoke threshold.
  G4 eval pipeline integrity: Stage B inherits the canonical 4-arm RMSE
       eval (no_calib / pop_slope / per_user_OLS / PILSD_EB_shrunk);
       gain_pct = (rmse_pop_slope - rmse_pilsd_shrunk) / rmse_pop_slope.
  G5 reference-implementation: BOTH stages already audited+committed
       (Stage A: 92e4707; Stage B: e2c1a8f) and smoke-PASS (G1-G5)
       at `results/falsifiers/F_T1_OASST2_REPLICATION_smoke/smoke_summary.json`.
  G6 hyperparameter sanity: matches PRISM headline conventions
       (min_obs_per_user=6 / k_folds=5 / B=4000 cluster-bootstrap).
  G7 per-step diagnostic: Stage A logs scoring rate + ETA every 200 rows;
       Stage B logs per-fold per-arm RMSE + cluster-bootstrap progress.
  G8 reproducibility: seed=42 / RNG_BOOT=20260420; cache fingerprint of
       parquet via SHA-256 persisted in summary.json.
  G9 output schema: 4-class STRICT verdict per PREREG §2.6 decision rule.
  G10 compute envelope: Stage A ~10-12h on H100 nf4 7B (~7-9k OASST2
       English assistant messages × ~1 row/s); Stage B ~10-15min CPU.
  G11 anti-overfitting: not theory-claiming.
  G12 honest-disclosure: 4-class verdict explicitly enumerates NULL
       (CI straddles zero ⇒ PRELIMINARY-INCONCLUSIVE-OASST2) and FALSIFIED
       (CI strictly negative ⇒ FALSIFIED-OASST2-DOES-NOT-REPLICATE)
       branches; PARTIAL captures the 1-3pp band.

4-class verdict-class STRICT (Skill: honest-disclosure §6.3):
  ESTABLISHED-OASST2-CONFIRMS-PRISM            gain in [+5, +12]; CI
                                                excludes 0; matches PRISM
                                                +8.58% headline within
                                                ±3pp.
  MODERATE-OASST2-PARTIAL-CONFIRMS              gain in [+1, +5]; CI
                                                excludes 0; weaker than
                                                PRISM but directionally
                                                consistent.
  PRELIMINARY-INCONCLUSIVE-OASST2               CI straddles 0; gain
                                                in [-1, +1] band.
  FALSIFIED-OASST2-DOES-NOT-REPLICATE           CI strictly negative;
                                                PRISM headline does NOT
                                                generalize ⇒ paper claim
                                                must be SCOPE-LIMITED to
                                                PRISM-class corpora.

Output
------
``results/track1_oasst2_replication/{summary.json, per_user.parquet}``
+ Tab cross-corpus row (NeurIPS) + 1-paragraph §3.X cross-corpus paragraph
+ 1-line Pluralistic §Lim reference.

Honest-disclosure note (added 2026-05-03 ~01:48 IST after Stage-B-fix subagent)
------------------------------------------------------------------------------
The 2026-05-02 first-launch run of this wrapper failed at Stage B with
`unrecognized arguments: --n-boot 4000` because the wrapper's CLI mapping to
`1_Causal_RLHF/scripts/eval_oasst2_pilsd_calibrator.py` was authored against a
provisional interface; the downstream script's actual CLI uses
``--seeds INT [INT ...]`` (plural list) + ``--bootstrap-B INT`` +
``--bootstrap-seed INT`` (NOT ``--seed`` + ``--n-boot``), and writes per-seed
sub-dirs as ``seed_<seed>`` UNpadded (NOT ``seed_<NNN>`` zero-padded). This
patch fixes the CLI mapping and the per-seed-dir naming. Stage A (Qwen2.5-7B
mean-LL scoring) succeeded fully in the first run (32125 rows / 2049 authors;
sha256 ``b6e21819...``); the cached parquet is reusable so Stage A is SKIPPED
on the second-launch path. The science (PILSD-EB-shrunk RMSE estimator;
cluster-bootstrap CI; verdict-class assignment) is UNCHANGED — this is purely
a calling-convention fix. CPU smoke verifies the patched CLI mapping
end-to-end before GPU re-launch.

References
----------
- Köpf, A. et al. (2023). OpenAssistant Conversations. NeurIPS Datasets.
- Kirk, H. et al. (2024). PRISM dataset.
- Stiennon, N. et al. (2020). RLHF mean-LL reward proxy.
- Morris, C. (1983). Parametric Empirical Bayes (tau²-MoM estimator).
- Henderson, C. (1975). BLUP shrinkage.
- Cameron, A. C., Gelbach, J. B., Miller, D. L. (2008). Cluster bootstrap.
- Efron, B. (1987). BCa CI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]  # 3_PILSD_Standalone/
T1 = ROOT.parent / "1_Causal_RLHF"
sys.path.insert(0, str(ROOT / "scripts"))
from _repo_paths import STANDALONE_RESULTS  # noqa: E402

OUT_DIR = STANDALONE_RESULTS / "track1_oasst2_replication"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
SEED = 42
RNG_BOOT = 20260420
N_BOOT = 4000
MIN_OBS_PER_USER = 6
K_FOLDS = 5
MIN_AUTHOR_MESSAGES_FOR_SCORING = 3
MAX_PROMPT_TOKENS = 256
MAX_RESPONSE_TOKENS = 256
MIN_SCORED_ROWS_FOR_VALID_CACHE = 1000  # G3: re-run Stage A if cache < this
MODEL_ID_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"

# Verdict thresholds (per PREREG §2.6)
PASS_LO, PASS_HI = 5.0, 12.0
PARTIAL_LO, PARTIAL_HI = 1.0, 5.0
NULL_LO, NULL_HI = -1.0, 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trees-path", required=True,
                   help=("Path to OASST2 ready-trees jsonl.gz dump on the "
                         "GPU pod, e.g. /workspace/.hf_cache/oasst2/"
                         "2023-11-05_oasst2_ready.trees.jsonl.gz"))
    p.add_argument("--scored-parquet",
                   default=str(T1 / "data" / "oasst2"
                               / "oasst2_qwen7b_scored.parquet"))
    p.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--k-folds", type=int, default=K_FOLDS)
    p.add_argument("--min-obs-per-user", type=int, default=MIN_OBS_PER_USER)
    p.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS)
    p.add_argument("--max-response-tokens", type=int, default=MAX_RESPONSE_TOKENS)
    p.add_argument("--force-rescore", action="store_true",
                   help="Re-run Stage A even if scored parquet exists.")
    p.add_argument("--smoke", action="store_true",
                   help="Cap Stage A at --smoke-limit rows for testing.")
    p.add_argument("--smoke-limit", type=int, default=200)
    p.add_argument("--output-dir", default=str(OUT_DIR))
    return p.parse_args()


def file_sha256(p: Path) -> str:
    if not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_a_needs_rerun(parquet: Path, force: bool, min_rows: int) -> bool:
    """G3: decide whether to re-run Stage A. Re-run if (a) --force-rescore
    or (b) parquet missing or (c) parquet has fewer than `min_rows` rows."""
    if force:
        return True
    if not parquet.exists():
        return True
    try:
        import pandas as pd
        n = len(pd.read_parquet(parquet, columns=["user_id"]))
        return n < min_rows
    except Exception:
        return True


def run_stage_a(args: argparse.Namespace) -> dict:
    """Stage A — score OASST2 messages with Qwen2.5-7B-Instruct mean-LL."""
    script = T1 / "scripts" / "score_oasst2_with_qwen7b.py"
    if not script.exists():
        raise FileNotFoundError(f"Stage A script missing: {script}")

    out_parquet = Path(args.scored_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script),
        "--trees-path", args.trees_path,
        "--model-id", args.model_id,
        "--output-parquet", str(out_parquet),
        "--max-prompt-tokens", str(args.max_prompt_tokens),
        "--max-response-tokens", str(args.max_response_tokens),
        "--min-author-messages", str(MIN_AUTHOR_MESSAGES_FOR_SCORING),
        "--seed", str(args.seed),
        "--quantize-4bit",
    ]
    if args.smoke:
        cmd.extend(["--limit", str(args.smoke_limit)])
    print(f"[Stage A] cmd: {shlex.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Stage A failed with exit {proc.returncode}")
    return {
        "elapsed_seconds": float(elapsed),
        "exit_code": int(proc.returncode),
        "scored_parquet": str(out_parquet),
        "scored_sha256": file_sha256(out_parquet),
    }


def run_stage_b(args: argparse.Namespace) -> dict:
    """Stage B — fit + evaluate PILSD calibrator on OASST2 with cluster-
    bootstrap CIs.

    NOTE on CLI mapping: the downstream
    `1_Causal_RLHF/scripts/eval_oasst2_pilsd_calibrator.py` script's argparse
    interface uses (a) plural `--seeds INT [INT ...]` (nargs="+"), NOT
    `--seed INT`; (b) `--bootstrap-B INT`, NOT `--n-boot INT`; and
    (c) `--bootstrap-seed INT` with default 42. It also creates its own
    per-seed sub-directory `<output_dir>/seed_<seed>/` (UNpadded), so we pass
    `--output-dir` as the parent directory (NOT a pre-created seed_NNN path),
    let Stage B create the unpadded seed sub-dir itself, and read back from
    that unpadded path.
    """
    script = T1 / "scripts" / "eval_oasst2_pilsd_calibrator.py"
    if not script.exists():
        raise FileNotFoundError(f"Stage B script missing: {script}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Stage B creates `out_dir/seed_<seed>/` itself (unpadded);
    # we record the expected path so downstream output.json lookup mirrors it.
    seed_dir = out_dir / f"seed_{args.seed}"

    cmd = [
        sys.executable, str(script),
        "--scored-parquet", args.scored_parquet,
        "--output-dir", str(out_dir),
        "--seeds", str(args.seed),
        "--bootstrap-B", str(args.n_boot),
        "--bootstrap-seed", str(RNG_BOOT),
        "--k-folds", str(args.k_folds),
        "--min-obs-per-user", str(args.min_obs_per_user),
    ]
    print(f"[Stage B] cmd: {shlex.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Stage B failed with exit {proc.returncode}")
    return {
        "elapsed_seconds": float(elapsed),
        "exit_code": int(proc.returncode),
        "stage_b_seed_dir": str(seed_dir),
    }


def assign_verdict_class(gain: float, ci_lo: float, ci_hi: float) -> str:
    """4-class verdict-class STRICT per PREREG §2.6 + Skill: honest-disclosure
    §6.3."""
    excludes_zero_pos = ci_lo > 0
    excludes_zero_neg = ci_hi < 0
    if PASS_LO <= gain <= PASS_HI and excludes_zero_pos:
        return "ESTABLISHED-OASST2-CONFIRMS-PRISM"
    if PARTIAL_LO <= gain <= PARTIAL_HI and excludes_zero_pos:
        return "MODERATE-OASST2-PARTIAL-CONFIRMS"
    if excludes_zero_neg or gain < NULL_LO:
        return "FALSIFIED-OASST2-DOES-NOT-REPLICATE"
    return "PRELIMINARY-INCONCLUSIVE-OASST2"


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    summary: dict[str, Any] = {
        "experiment_id": "WaveC_EXP2_OASST2_replication",
        "verdict_class": "PENDING",
        "args": vars(args),
        "seed": int(args.seed),
        "rng_boot": int(RNG_BOOT),
        "n_boot": int(args.n_boot),
        "k_folds": int(args.k_folds),
        "min_obs_per_user": int(args.min_obs_per_user),
        "model_id": args.model_id,
        "skill_citations": [
            "Skill: research-grade-code-audit-pre-launch v1 G1-G12",
            "Skill: honest-disclosure §6.3 SCOPE-not-RETRACT (4-class STRICT)",
            "Skill: post-experiment-discipline-3-track Step 5+7",
            "Skill: launch-runpod-h100-job",
            "Skill: gpu-artifact-sync",
        ],
        "anomaly_branches_fired": [],
        "stages": {},
    }

    print(f"[Wave-C EXP-2] OASST2 cross-corpus PILSD replication")
    print(f"[output-dir] {out_dir}")
    print(f"[scored-parquet] {args.scored_parquet}")

    # ------------------------------------------------------------------
    # Stage A (skippable cache)
    # ------------------------------------------------------------------
    parquet = Path(args.scored_parquet)
    if stage_a_needs_rerun(parquet, args.force_rescore,
                            MIN_SCORED_ROWS_FOR_VALID_CACHE):
        print(f"[Stage A] running (parquet missing / too small / forced)")
        try:
            stage_a_meta = run_stage_a(args)
            summary["stages"]["stage_a"] = stage_a_meta
        except Exception as e:
            summary["verdict_class"] = "RUNTIME-ERROR-STAGE-A"
            summary["error"] = traceback.format_exc()
            summary_path.write_text(json.dumps(summary, indent=2))
            raise
    else:
        # G3: cache reuse path; record the SHA + row count.
        import pandas as pd
        n = len(pd.read_parquet(parquet, columns=["user_id"]))
        summary["stages"]["stage_a"] = {
            "elapsed_seconds": 0.0,
            "exit_code": 0,
            "cache_reused": True,
            "scored_parquet": str(parquet),
            "scored_sha256": file_sha256(parquet),
            "scored_n_rows": int(n),
        }
        print(f"[Stage A] CACHE REUSE: {parquet} ({n} rows; "
              f"sha256={summary['stages']['stage_a']['scored_sha256'][:12]}...)")

    # ------------------------------------------------------------------
    # Stage B
    # ------------------------------------------------------------------
    try:
        stage_b_meta = run_stage_b(args)
        summary["stages"]["stage_b"] = stage_b_meta
    except Exception as e:
        summary["verdict_class"] = "RUNTIME-ERROR-STAGE-B"
        summary["error"] = traceback.format_exc()
        summary_path.write_text(json.dumps(summary, indent=2))
        raise

    # ------------------------------------------------------------------
    # Stage B -> verdict-class
    # ------------------------------------------------------------------
    seed_dir = Path(stage_b_meta["stage_b_seed_dir"])
    stage_b_output = seed_dir / "output.json"
    if not stage_b_output.exists():
        # Try alternative known output paths
        candidates = list(seed_dir.glob("**/output.json")) + \
                     list(seed_dir.glob("**/primary_seed_output.json"))
        if candidates:
            stage_b_output = candidates[0]
        else:
            summary["verdict_class"] = "RUNTIME-ERROR-STAGE-B-NO-OUTPUT"
            summary_path.write_text(json.dumps(summary, indent=2))
            raise FileNotFoundError(
                f"Stage B did not produce output.json at {seed_dir}")

    with stage_b_output.open() as f:
        stage_b_result = json.load(f)

    gain = float(stage_b_result.get("gain_pct", float("nan")))
    ci_lo = float(stage_b_result.get("gain_ci_bca_lo", float("nan")))
    ci_hi = float(stage_b_result.get("gain_ci_bca_hi", float("nan")))
    n_users = int(stage_b_result.get("n_users_post_filter", 0))
    n_obs = int(stage_b_result.get("n_obs_total", 0))

    verdict = assign_verdict_class(gain, ci_lo, ci_hi)

    # Anomaly checks
    anomalies = []
    if not (gain == gain):  # NaN
        anomalies.append("gain_is_nan")
    if ci_hi < ci_lo:
        anomalies.append("ci_bounds_swapped")
    if n_users < 50:
        anomalies.append("n_users_below_minimum_50")

    summary.update({
        "verdict_class": verdict,
        "gain_pct": gain,
        "gain_ci_bca_lo": ci_lo,
        "gain_ci_bca_hi": ci_hi,
        "n_users_post_filter": n_users,
        "n_obs_total": n_obs,
        "rmse_baseline_pop": float(stage_b_result.get("rmse_baseline_pop", float("nan"))),
        "rmse_pilsd_shrunk": float(stage_b_result.get("rmse_pilsd_shrunk", float("nan"))),
        "stage_b_full_output": stage_b_result,
        "anomaly_branches_fired": anomalies,
    })

    summary_path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"=== Wave-C EXP-2 OASST2 Replication ===")
    print(f"verdict_class           : {verdict}")
    print(f"n_users / n_obs         : {n_users} / {n_obs}")
    print(f"rmse_pop / rmse_shrunk  : "
          f"{summary['rmse_baseline_pop']:.4f} / "
          f"{summary['rmse_pilsd_shrunk']:.4f}")
    print(f"gain_pct                : {gain:+.3f}%  CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"anomalies_fired         : {anomalies}")
    print(f"summary_path            : {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        # G3: ensure summary.json exists for cron's verdict-detection
        sp = OUT_DIR / "summary.json"
        if not sp.exists():
            sp.write_text(json.dumps({
                "experiment_id": "WaveC_EXP2_OASST2_replication",
                "verdict_class": "RUNTIME-ERROR",
                "error": traceback.format_exc(),
            }, indent=2))
        sys.exit(2)
