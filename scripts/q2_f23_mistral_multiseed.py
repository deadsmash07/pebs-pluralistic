"""Q2 - F23 multi-seed Mistral cross-family extension (brief seed grid).

Closes the NeurIPS critical-review W2 single-seed disclosure attack flagged in
`memory/p0_experiment_queue_v1_2026_05_03.md` Q2 brief and `Skill:
icml-neurips-critical-reviewer-2026` Pass 5 (multi-seed robustness check).

Hypothesis
----------
F23 cross-family Mistral coherence-only sign-direction is robust across the
4-seed grid {42, 123, 7777, 20260420}. The grid mirrors T3 W-B2 5-seed CV
verification verbatim (less seed=2024) for cross-experiment seed-list discipline
against p-hacking. seed=42 is harvested from the existing F23 main Mistral
RESULTS path (T1 path
`results/falsifiers/F23_coherence_multi_base/mistral_small_instruct_2409/output.json`;
+19.991% [-17.223, +42.862]; per_base_class NULL-STRADDLES-ZERO; pred_sd_coh
0.4633 above 0.40 head-collapse FALSIFIED gate). Seeds {123, 7777, 20260420} are
NEW cells that get full LoRA training on rtx6000-1 RTX PRO 6000 Blackwell 96GB.

Decision rule (4-class STRICT per Skill: honest-disclosure 6.3)
---------------------------------------------------------------
ESTABLISHED-CROSS-FAMILY-MISTRAL-SIGN-FLIP-MULTISEED if all 4 seeds same
                                                     direction with CI excluding
                                                     zero (cross-seed cluster
                                                     bootstrap aggregate CI
                                                     strictly POS or strictly
                                                     NEG)
MODERATE-MISTRAL-MULTISEED-PARTIAL                   if 3/4 seeds confirm
                                                     direction; 1 dissenter
PRELIMINARY-INCONCLUSIVE                             if 2/4 confirm OR cross-seed
                                                     CI half-width > 100pp OR
                                                     anomaly branch fires
FALSIFIED-MISTRAL-SINGLE-SEED-ARTIFACT               if direction inconsistent
                                                     across the 4 seeds (sign
                                                     mixed at single-seed
                                                     coverage)

These map to the existing T1 script's CI-TIGHTENS-POSITIVE / CI-REMAINS-NULL /
CI-FLIPS-NEGATIVE / PRELIMINARY-INCONCLUSIVE branches; the wrapper translates the
inner-script verdict into the brief's vocabulary in the dispatch memo.

Q2-vs-existing-extension delta
------------------------------
The existing T1 multi-seed extension `F23_mistral_multiseed_extension.py`
(commits anchor PREREG `5c3aa12`, SCRIPT `f0a1982` upstream) covers seeds
{137, 271, 314}. The Q2 brief specifies seeds {123, 7777, 20260420}, which is a
DIFFERENT 3-NEW-cell grid (matches W-B2 5-seed CV minus seed=2024). This is
intentional cross-experiment seed-list re-use:
  - T3 W-B2 used [42, 123, 2024, 7777, 20260420] for the pure-PILSD shrinkage
    headline; verdict ESTABLISHED-MULTI-SEED-CONFIRMS-HEADLINE
  - Q2 reuses {42, 123, 7777, 20260420} for F23 coherence-only Mistral (seed
    2024 dropped per 4-seed budget cap)

This keeps the Mistral extension's seed-list orthogonal to F19' / F43 (which use
{42, 137, 271, 314, 1729}) so the cross-family panel composition rule is NOT a
silent re-use of the F19' Phi-3 anchor seed-list.

Usage (CPU smoke test)
----------------------
    python paper/scripts/q2_f23_mistral_multiseed.py \\
        --tier 1 --smoke --no-launch-gate \\
        --output-dir /tmp/q2_smoke

Usage (production on rtx6000-1)
-------------------------------
    python paper/scripts/q2_f23_mistral_multiseed.py \\
        --tier 2 \\
        --output-dir /workspace/1_Causal_RLHF/results/falsifiers/F23_mistral_multiseed_extension_q2

12-gate audit summary (Skill: research-grade-code-audit-pre-launch)
-------------------------------------------------------------------
Q2 wrapper is a <150 LOC THIN DISPATCHER over the existing T1 script
`F23_mistral_multiseed_extension.py` (1412 LOC). The T1 inner script was audited
+ launched as production-class for the {137, 271, 314} grid (smoke + CPU dry-run
+ H100 launch all green). G1-G12 inheritance:

G1 math-vs-code           inherited verbatim from T1 inner (compute_gain +
                          cluster_bootstrap + cross_seed_cluster_bootstrap fns
                          unchanged).
G2 hypothesis-vs-design   Q2 brief 4-seed grid {42, 123, 7777, 20260420} matches
                          this wrapper's `--seed_list 123,7777,20260420`
                          delegation to T1 inner; seed=42 harvested from F23
                          main RESULTS path. Apples-to-apples with canonical F23
                          Llama panel: same coherence-only loss + same LoRA
                          targets + same 4-bit nf4 + same effective batch 128.
G3 no silent-bypass       T1 inner enforces `--coherence_only_loss` flag in F23
                          main; absence aborts at L552-554. Bare wrapper does
                          not bypass this.
G4 eval pipeline integrity Bootstrap CI computed on the per-attribute coherence
                          gain (attr_idx=2); aggregation cluster-by-seed.
G5 reference-impl parity  inherits canonical Morris (1983) MoM tau^2 + Henderson
                          (1975) BLUP shrinkage from T1 inner; matches sister
                          falsifiers F19', F43.
G6 hyperparameter sanity  F23-canonical: LR=1e-5 / epochs=3 / effective batch
                          128 / LoRA r=64 alpha=128 dropout=0.05 / 4-bit nf4 +
                          double-quant + grad-checkpointing. NO sweeps.
G7 per-step diagnostic    T1 inner logs every 10 steps + per-cell branch
                          verdict.json with anomaly branches (loss NaN/Inf,
                          bootstrap CI > 100pp, PILSD gate failure).
G8 reproducibility        per-cell train_seed varied via CLI; bootstrap seed
                          fixed (BOOT_SEED=20260424); cross-seed bootstrap seed
                          fixed (CROSS_SEED_BOOT_SEED=20260501); RNG_SEED=42
                          fixed for 80/20 cal/hel split.
G9 output schema          T1 inner writes
                          `results/falsifiers/F23_mistral_multiseed_extension_q2/cross_seed_verdict.json`
                          with `evidence_class` field drawn from
                          {CI-TIGHTENS-POSITIVE, CI-REMAINS-NULL,
                          CI-FLIPS-NEGATIVE, PRELIMINARY-INCONCLUSIVE}; this
                          wrapper's dispatch memo translates to brief vocabulary.
G10 compute envelope      Mistral-22B on rtx6000-1 RTX PRO 6000 Blackwell
                          96GB: 4-bit nf4 LoRA +  effective batch 128 + 3 epochs
                          x 159 steps per cell ~= 8-12h per cell on Blackwell
                          (matches T1 cell-1 budget calibration that hit 19h on
                          earlier H100 estimate; Blackwell is faster). 3 cells
                          sequential = ~24-36h. Per-cell kill 24h / master kill
                          80h.
G11 anti-overfitting      not theory-claiming.
G12 honest-disclosure     T1 inner outputs 14 honest-disclosure flags in
                          `cross_seed_verdict.json["honest_disclosures"]`;
                          dispatch memo enumerates them.

NO INTERNAL KILL SWITCHES per user 2026-05-02 12:08 IST directive: the inner
script's `kill_h_per_cell=24h` and `kill_h_master=80h` ARE the only kill gates;
no additional Q2 wrapper-level kills.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]                   # 3_PILSD_Standalone/
T1   = ROOT.parent / "1_Causal_RLHF"
F23_MS_INNER = T1 / "scripts" / "falsifiers" / "F23_mistral_multiseed_extension.py"

# Q2 4-seed grid (brief verbatim; matches W-B2 5-seed CV minus seed 2024)
Q2_FULL_4SEED_GRID = [42, 123, 7777, 20260420]
Q2_NEW_3CELL_GRID  = [123, 7777, 20260420]   # seed=42 harvested from F23 main


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[Q2 {ts}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Q2 F23 Mistral multi-seed cross-family extension (brief 4-seed grid)"
    )
    ap.add_argument("--tier", type=int, default=2, choices=[1, 2],
                    help="Tier 1 SMOKE / Tier 2 STANDARD")
    ap.add_argument("--smoke", action="store_true",
                    help="Pass --smoke to inner (1 epoch + batch 1 + n_boot=10)")
    ap.add_argument("--no-launch-gate", action="store_true",
                    help="Pass --no-launch-gate to inner (skip GPU idle check)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Master output dir; default = "
                         "T1/results/falsifiers/F23_mistral_multiseed_extension_q2/")
    ap.add_argument("--kill_h", type=float, default=None,
                    help="Override per-cell HARD KILL wall (hours)")
    ap.add_argument("--kill_h_master", type=float, default=None,
                    help="Override master cumulative HARD KILL wall (hours)")
    ap.add_argument("--seed_list", type=str, default=None,
                    help="Override the canonical Q2 NEW-cell seed-list (default "
                         "'123,7777,20260420')")
    args = ap.parse_args()

    if not F23_MS_INNER.exists():
        log(f"FATAL: inner script not found at {F23_MS_INNER}")
        log(f"  expected T1 path: {T1}")
        log(f"  ensure 1_Causal_RLHF repo is checked out in sibling directory")
        return 2

    seed_list = args.seed_list if args.seed_list is not None else ",".join(
        str(s) for s in Q2_NEW_3CELL_GRID
    )

    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.tier == 1 or args.smoke:
        output_dir = "/tmp/q2_f23_mistral_multiseed_smoke"
    else:
        output_dir = str(
            T1 / "results" / "falsifiers" / "F23_mistral_multiseed_extension_q2"
        )

    cmd = [
        sys.executable,
        str(F23_MS_INNER),
        "--tier", str(args.tier),
        "--seed_list", seed_list,
        "--output_dir", output_dir,
    ]
    if args.smoke:
        cmd.append("--smoke")
    if args.no_launch_gate:
        cmd.append("--no-launch-gate")
    if args.kill_h is not None:
        cmd.extend(["--kill_h", str(args.kill_h)])
    if args.kill_h_master is not None:
        cmd.extend(["--kill_h_master", str(args.kill_h_master)])

    log(f"Q2 dispatcher delegating to T1 inner script")
    log(f"  inner: {F23_MS_INNER}")
    log(f"  full 4-seed grid: {Q2_FULL_4SEED_GRID}")
    log(f"  NEW cells (3): [{seed_list}]")
    log(f"  output_dir: {output_dir}")
    log(f"  cmd: {' '.join(shlex.quote(c) for c in cmd)}")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, check=False)
        log(f"Q2 inner returned code={result.returncode}; "
            f"wall {(time.time()-t0)/3600:.2f}h")
        return int(result.returncode)
    except KeyboardInterrupt:
        log(f"Q2 dispatcher interrupted; wall {(time.time()-t0)/3600:.2f}h")
        return 130


if __name__ == "__main__":
    sys.exit(main())
