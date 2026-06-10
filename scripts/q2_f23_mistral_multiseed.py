"""Q2 - F23 multi-seed Mistral cross-family extension.

The F23 cross-family Mistral result was previously reported from a single
seed; a single seed cannot support a robustness claim, so this extends the
result to a multi-seed grid.

Hypothesis
----------
F23 cross-family Mistral coherence-only sign-direction is robust across the
4-seed grid {42, 123, 7777, 20260420}. The grid mirrors the W-B2 5-seed CV
verification verbatim (less seed=2024) for cross-experiment seed-list discipline
against p-hacking. seed=42 is harvested from the existing F23 main Mistral
results path (
`results/falsifiers/F23_coherence_multi_base/mistral_small_instruct_2409/output.json`;
+19.991% [-17.223, +42.862]; per_base_class NULL-STRADDLES-ZERO; pred_sd_coh
0.4633 above 0.40 head-collapse REJECTED gate). Seeds {123, 7777, 20260420} are
NEW cells that get full LoRA training on a 96GB GPU.

Decision rule
-------------
CONFIRMED-CROSS-FAMILY-MISTRAL-SIGN-FLIP-MULTISEED if all 4 seeds same
                                                     direction with CI excluding
                                                     zero (cross-seed cluster
                                                     bootstrap aggregate CI
                                                     strictly POS or strictly
                                                     NEG)
PARTIAL-MISTRAL-MULTISEED-PARTIAL                   if 3/4 seeds confirm
                                                     direction; 1 dissenter
TENTATIVE-INCONCLUSIVE                             if 2/4 confirm OR cross-seed
                                                     CI half-width > 100pp OR
                                                     anomaly branch fires
REJECTED-MISTRAL-SINGLE-SEED-ARTIFACT               if direction inconsistent
                                                     across the 4 seeds (sign
                                                     mixed at single-seed
                                                     coverage)

These map to the existing inner script's CI-TIGHTENS-POSITIVE / CI-REMAINS-NULL /
CI-FLIPS-NEGATIVE / TENTATIVE-INCONCLUSIVE branches; the wrapper translates the
inner-script outcome into the vocabulary above.

Q2-vs-existing-extension delta
------------------------------
The existing multi-seed extension `F23_mistral_multiseed_extension.py`
covers seeds
{137, 271, 314}. Q2 specifies seeds {123, 7777, 20260420}, which is a
DIFFERENT 3-NEW-cell grid (matches W-B2 5-seed CV minus seed=2024). This is
intentional cross-experiment seed-list re-use:
  - W-B2 (`scripts/wave_b/wave_b_W_B2_5seed_cv.py`) used
    [42, 123, 2024, 7777, 20260420] for the pure-PEBS shrinkage headline
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

Usage (production)
------------------
    python paper/scripts/q2_f23_mistral_multiseed.py \\
        --tier 2 \\
        --output-dir /workspace/1_Causal_RLHF/results/falsifiers/F23_mistral_multiseed_extension_q2

Design notes
------------
Q2 wrapper is a <150 LOC thin wrapper over the existing script
`F23_mistral_multiseed_extension.py` (1412 LOC), which was previously
validated end-to-end on the {137, 271, 314} grid. Inherited properties:

- Math: inherited verbatim from the inner script (compute_gain +
  cluster_bootstrap + cross_seed_cluster_bootstrap fns unchanged).
- Design: the 4-seed grid {42, 123, 7777, 20260420} matches
  this wrapper's `--seed_list 123,7777,20260420`
  delegation to the inner script; seed=42 harvested from the F23
  main results path. Apples-to-apples with the canonical F23
  Llama panel: same coherence-only loss + same LoRA
  targets + same 4-bit nf4 + same effective batch 128.
- No silent-bypass: the inner script enforces `--coherence_only_loss` in F23
  main; absence aborts at L552-554. Bare wrapper does not bypass this.
- Eval pipeline: bootstrap CI computed on the per-attribute coherence
  gain (attr_idx=2); aggregation cluster-by-seed.
- Reference-impl parity: inherits canonical Morris (1983) MoM tau^2 + Henderson
  (1975) BLUP shrinkage from the inner script; matches sister
  falsifiers F19', F43.
- Hyperparameters: F23-canonical: LR=1e-5 / epochs=3 / effective batch
  128 / LoRA r=64 alpha=128 dropout=0.05 / 4-bit nf4 +
  double-quant + grad-checkpointing. NO sweeps.
- Diagnostics: the inner script logs every 10 steps + per-cell branch
  verdict.json with anomaly branches (loss NaN/Inf,
  bootstrap CI > 100pp, PEBS gate failure).
- Reproducibility: per-cell train_seed varied via CLI; bootstrap seed
  fixed (BOOT_SEED=20260424); cross-seed bootstrap seed
  fixed (CROSS_SEED_BOOT_SEED=20260501); RNG_SEED=42
  fixed for 80/20 cal/hel split.
- Output schema: the inner script writes
  `results/falsifiers/F23_mistral_multiseed_extension_q2/cross_seed_verdict.json`
  with `evidence_class` field drawn from
  {CI-TIGHTENS-POSITIVE, CI-REMAINS-NULL,
  CI-FLIPS-NEGATIVE, TENTATIVE-INCONCLUSIVE}.
- Compute: Mistral-22B on a 96GB GPU: 4-bit nf4 LoRA + effective batch 128
  + 3 epochs x 159 steps per cell ~= 8-12h per cell. 3 cells
  sequential = ~24-36h. Per-cell kill 24h / master kill 80h.
- The inner script outputs 14 disclosure flags in
  `cross_seed_verdict.json["honest_disclosures"]`.

The inner script's `kill_h_per_cell=24h` and `kill_h_master=80h` ARE the only
kill gates; no additional Q2 wrapper-level kills.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]                   # 3_PEBS_Standalone/
T1   = ROOT.parent / "1_Causal_RLHF"
F23_MS_INNER = T1 / "scripts" / "falsifiers" / "F23_mistral_multiseed_extension.py"

# Q2 4-seed grid (matches W-B2 5-seed CV minus seed 2024)
Q2_FULL_4SEED_GRID = [42, 123, 7777, 20260420]
Q2_NEW_3CELL_GRID  = [123, 7777, 20260420]   # seed=42 harvested from F23 main


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[Q2 {ts}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Q2 F23 Mistral multi-seed cross-family extension (4-seed grid)"
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
        log(f"  expected path: {T1}")
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

    log(f"Q2 wrapper delegating to inner script")
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
        log(f"Q2 wrapper interrupted; wall {(time.time()-t0)/3600:.2f}h")
        return 130


if __name__ == "__main__":
    sys.exit(main())
