"""Q6-MS3 - multi-seed Mistral DPO replication for Q6.

This script runs three additional Mistral-7B Q6 DPO seeds after the Q6
cross-backbone run showed a strongly positive Mistral held-out
pair-accuracy delta. It intentionally wraps the existing
`q6_dpo_downstream_impact.py` implementation instead of duplicating DPO
training or evaluation logic.

Protocol
--------
- Base model: mistralai/Mistral-7B-Instruct-v0.3 by default.
- Seeds: 42, 123, 7777 by default.
- Per seed: run Q6 ARM A vs ARM B DPO with identical hyperparameters and only
  the seed-varying user-disjoint split / training RNG.
- Endpoint: held-out PRISM pair-accuracy delta, same as Q6 cross-backbone.
- AlpacaEval-2 is skipped by default, matching Q6 cross-backbone.
- No internal time-based kill switches.

Output
------
results/track1_q6_ms3_dpo_multiseed_mistral/
  summary.json
  cross_seed_summary.json
  per_seed_summaries.json
  preflight.json
  seed_42/summary.json
  seed_123/summary.json
  seed_7777/summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
Q6_SCRIPT = SCRIPTS_DIR / "q6_dpo_downstream_impact.py"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import q6_dpo_downstream_impact as q6  # noqa: E402


DEFAULT_BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_SEEDS = (42, 123, 7777)
DEFAULT_OUT_DIR = ROOT / "results" / "track1_q6_ms3_dpo_multiseed_mistral"
DEFAULT_PRIOR_Q6_DIR = ROOT / "results" / "track1_q6_dpo_downstream_impact_mistral_7b"
CANONICAL_Q6_DIR = ROOT / "results" / "track1_q6_dpo_downstream_impact"
PAIR_A_NAME = "pairs_arm_a_uncorrected.parquet"
PAIR_B_NAME = "pairs_arm_b_pebs_corrected.parquet"

MULTISEED_ESTABLISHED_DELTA_PP = 1.5
MULTISEED_MODERATE_DELTA_PP = 1.0


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def file_sha256(path: Path) -> str:
    if not path.exists():
        return f"FILE_NOT_FOUND:{path}"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_seed_list(raw: str) -> list[int]:
    seeds = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("no seeds parsed")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds are not allowed: {seeds}")
    return seeds


def localize_workspace_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    raw_s = str(path)
    prefix = "/workspace/3_PEBS_Standalone"
    if raw_s.startswith(prefix):
        return ROOT / raw_s[len(prefix):].lstrip("/")
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def prior_pair_paths(prior_dir: Path, prior_summary: dict[str, Any]) -> tuple[Path, Path]:
    pair_stage = (prior_summary.get("stages") or {}).get("pair_construction") or {}
    arm_a = localize_workspace_path(pair_stage.get("arm_a_path"))
    arm_b = localize_workspace_path(pair_stage.get("arm_b_path"))
    return arm_a or prior_dir / PAIR_A_NAME, arm_b or prior_dir / PAIR_B_NAME


def validate_pair_frame(df: pd.DataFrame, label: str) -> dict[str, Any]:
    required = {
        "user_id",
        "conversation_id",
        "turn",
        "prompt",
        "chosen",
        "rejected",
        "chosen_within_turn_id",
        "rejected_within_turn_id",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{label} missing required pair columns: {missing}")
    if len(df) == 0:
        raise ValueError(f"{label} pair frame is empty")
    if df["user_id"].isna().any():
        raise ValueError(f"{label} has null user_id values")
    if (df["chosen"].astype(str) == df["rejected"].astype(str)).any():
        raise ValueError(f"{label} contains identical chosen/rejected strings")
    return {
        "n_pairs": int(len(df)),
        "n_users": int(df["user_id"].nunique()),
        "sha256": None,
        "columns": list(df.columns),
    }


def load_prior_pairs(
    prior_dir: Path,
    prior_summary: dict[str, Any],
    allow_pair_fallback_for_smoke: bool,
    cpu_smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    arm_a_path, arm_b_path = prior_pair_paths(prior_dir, prior_summary)
    source = "prior_mistral_q6"

    if not (arm_a_path.exists() and arm_b_path.exists()):
        if allow_pair_fallback_for_smoke and cpu_smoke:
            fallback_a = CANONICAL_Q6_DIR / PAIR_A_NAME
            fallback_b = CANONICAL_Q6_DIR / PAIR_B_NAME
            if fallback_a.exists() and fallback_b.exists():
                arm_a_path, arm_b_path = fallback_a, fallback_b
                source = "cpu_smoke_fallback_canonical_q6_pairs"
            else:
                raise FileNotFoundError(
                    "prior Mistral pair parquets missing and canonical smoke "
                    f"fallback missing: {arm_a_path}, {arm_b_path}, "
                    f"{fallback_a}, {fallback_b}"
                )
        else:
            raise FileNotFoundError(
                "production preflight requires prior Mistral pair parquets: "
                f"{arm_a_path}, {arm_b_path}"
            )

    arm_a = pd.read_parquet(arm_a_path)
    arm_b = pd.read_parquet(arm_b_path)
    a_diag = validate_pair_frame(arm_a, "ARM A")
    b_diag = validate_pair_frame(arm_b, "ARM B")
    a_diag["sha256"] = file_sha256(arm_a_path)
    b_diag["sha256"] = file_sha256(arm_b_path)

    pair_diag = q6.diagnose_pair_diff(arm_a, arm_b)
    if not pair_diag.get("g3_silent_bypass_pass", False):
        raise ValueError(f"silent-bypass guard failed in prior pairs: {pair_diag}")

    return arm_a, arm_b, {
        "source": source,
        "arm_a_path": str(arm_a_path),
        "arm_b_path": str(arm_b_path),
        "arm_a": a_diag,
        "arm_b": b_diag,
        "g3_silent_bypass_diagnostics": pair_diag,
    }


def split_diagnostics(
    arm_a: pd.DataFrame,
    arm_b: pd.DataFrame,
    seeds: list[int],
    held_out_frac: float,
    smoke_pairs: int,
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        rng_split = np.random.default_rng(seed)
        all_users = sorted(set(arm_a.user_id.unique()) | set(arm_b.user_id.unique()))
        rng_split.shuffle(all_users)
        n_test_users = max(1, int(round(len(all_users) * held_out_frac)))
        test_users = set(all_users[:n_test_users])
        a_train = arm_a[~arm_a.user_id.isin(test_users)]
        a_test = arm_a[arm_a.user_id.isin(test_users)]
        b_train = arm_b[~arm_b.user_id.isin(test_users)]
        b_test = arm_b[arm_b.user_id.isin(test_users)]
        if smoke_pairs > 0:
            a_train = a_train.head(smoke_pairs)
            b_train = b_train.head(smoke_pairs)
            a_test = a_test.head(max(1, smoke_pairs // 4))
            b_test = b_test.head(max(1, smoke_pairs // 4))
        train_users = set(a_train.user_id.astype(str)) | set(b_train.user_id.astype(str))
        eval_users = set(a_test.user_id.astype(str)) | set(b_test.user_id.astype(str))
        overlap = sorted(train_users.intersection(eval_users))
        if overlap:
            raise ValueError(f"user-disjoint split failed for seed={seed}: {overlap[:5]}")
        rows.append({
            "seed": int(seed),
            "n_test_users": int(n_test_users),
            "n_train_users": int(len(all_users) - n_test_users),
            "n_arm_a_train": int(len(a_train)),
            "n_arm_a_test": int(len(a_test)),
            "n_arm_b_train": int(len(b_train)),
            "n_arm_b_test": int(len(b_test)),
            "user_overlap_train_eval": 0,
        })
    return rows


def load_prior_hparams(prior_summary: dict[str, Any]) -> dict[str, Any]:
    args = prior_summary.get("args") or {}
    keys = (
        "held_out_frac",
        "lora_r",
        "lora_alpha",
        "dpo_beta",
        "dpo_lr",
        "dpo_batch",
        "dpo_grad_accum",
        "dpo_max_seq_len",
        "dpo_max_prompt_len",
        "dpo_num_epochs",
    )
    out = {}
    for key in keys:
        if key in args:
            out[key] = args[key]
    return out


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    prior_dir = Path(args.prior_q6_output_dir)
    prior_summary_path = prior_dir / "summary.json"
    prior_summary = read_json(prior_summary_path)
    seeds = parse_seed_list(args.seeds)

    if not Q6_SCRIPT.exists():
        raise FileNotFoundError(f"missing parent Q6 script: {Q6_SCRIPT}")
    if args.base_model_id != DEFAULT_BASE_MODEL_ID:
        log(f"[preflight] non-default base_model_id requested: {args.base_model_id}")

    prior_args = prior_summary.get("args") or {}
    prior_base = prior_args.get("base_model_id")
    prior_verdict = prior_summary.get("verdict_class")
    if prior_summary and prior_base != args.base_model_id:
        raise ValueError(
            f"prior Q6 summary base_model_id={prior_base!r} does not match "
            f"requested base_model_id={args.base_model_id!r}"
        )

    hparams = load_prior_hparams(prior_summary)
    effective_held_out_frac = (
        float(args.held_out_frac)
        if args.held_out_frac is not None
        else float(hparams.get("held_out_frac", q6.PRISM_HELD_OUT_FRAC))
    )

    arm_a, arm_b, pair_info = load_prior_pairs(
        prior_dir=prior_dir,
        prior_summary=prior_summary,
        allow_pair_fallback_for_smoke=bool(args.allow_pair_fallback_for_smoke),
        cpu_smoke=bool(args.cpu_smoke),
    )
    split_info = split_diagnostics(
        arm_a=arm_a,
        arm_b=arm_b,
        seeds=seeds,
        held_out_frac=effective_held_out_frac,
        smoke_pairs=int(args.smoke_pairs if args.cpu_smoke else 0),
    )

    parent_source = Q6_SCRIPT.read_text()
    if "DPOTrainer" not in parent_source or "DPOConfig" not in parent_source:
        raise ValueError("parent Q6 script no longer contains TRL DPOTrainer/DPOConfig")

    return {
        "status": "PASS",
        "timestamp_utc": utc_now(),
        "base_model_id": args.base_model_id,
        "seeds": seeds,
        "prior_q6_summary_path": str(prior_summary_path),
        "prior_q6_verdict_class": prior_verdict,
        "prior_q6_pair_acc_delta_pp": (
            ((prior_summary.get("stages") or {}).get("held_out_pair_accuracy") or {})
            .get("delta_pp")
        ),
        "prior_q6_pair_acc_ci95": [
            ((prior_summary.get("stages") or {}).get("held_out_pair_accuracy") or {})
            .get("ci95_lo"),
            ((prior_summary.get("stages") or {}).get("held_out_pair_accuracy") or {})
            .get("ci95_hi"),
        ],
        "pair_info": pair_info,
        "split_diagnostics": split_info,
        "hparams_inherited_from_prior_q6": hparams,
        "parent_q6_script": str(Q6_SCRIPT),
        "parent_q6_script_sha256": file_sha256(Q6_SCRIPT),
        "git_head_t3_at_preflight": git_head(),
    }


def hparam_cli_args(hparams: dict[str, Any], args: argparse.Namespace) -> list[str]:
    merged = dict(hparams)
    for key in (
        "held_out_frac",
        "lora_r",
        "lora_alpha",
        "dpo_beta",
        "dpo_lr",
        "dpo_batch",
        "dpo_grad_accum",
        "dpo_max_seq_len",
        "dpo_max_prompt_len",
        "dpo_num_epochs",
    ):
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            merged[key] = cli_value

    mapping = {
        "held_out_frac": "--held-out-frac",
        "lora_r": "--lora-r",
        "lora_alpha": "--lora-alpha",
        "dpo_beta": "--dpo-beta",
        "dpo_lr": "--dpo-lr",
        "dpo_batch": "--dpo-batch",
        "dpo_grad_accum": "--dpo-grad-accum",
        "dpo_max_seq_len": "--dpo-max-seq-len",
        "dpo_max_prompt_len": "--dpo-max-prompt-len",
        "dpo_num_epochs": "--dpo-num-epochs",
    }
    out: list[str] = []
    for key, opt in mapping.items():
        if key in merged and merged[key] is not None:
            out += [opt, str(merged[key])]
    return out


def run_seed(seed: int, args: argparse.Namespace, inherited_hparams: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    seed_out = out_dir / f"seed_{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Q6_SCRIPT),
        "--base-model-id",
        args.base_model_id,
        "--seed",
        str(seed),
        "--output-dir",
        str(seed_out),
        "--n-boot",
        str(args.n_boot),
    ]
    cmd += hparam_cli_args(inherited_hparams, args)
    if args.utterances_parquet:
        cmd += ["--utterances-parquet", args.utterances_parquet]
    if args.rm_scored_parquet:
        cmd += ["--rm-scored-parquet", args.rm_scored_parquet]
    if args.calibrators_parquet:
        cmd += ["--calibrators-parquet", args.calibrators_parquet]
    if args.skip_alpaca_eval:
        cmd += ["--skip-alpaca-eval"]

    command_path = seed_out / "q6_parent_command.json"
    command_path.write_text(json.dumps({
        "seed": seed,
        "cmd": cmd,
        "started_utc": utc_now(),
    }, indent=2))

    log(f"[seed:{seed}] launching parent Q6: {' '.join(cmd)}")
    started = time.time()
    proc = subprocess.run(cmd, check=False)
    elapsed = time.time() - started
    summary_path = seed_out / "summary.json"
    if not summary_path.exists():
        return {
            "seed": int(seed),
            "return_code": int(proc.returncode),
            "elapsed_seconds": float(elapsed),
            "summary_path": str(summary_path),
            "verdict_class": "RUNTIME-ERROR-MISSING-SUMMARY",
            "error": "parent Q6 did not write summary.json",
        }

    summary = json.loads(summary_path.read_text())
    pair_acc = (summary.get("stages") or {}).get("held_out_pair_accuracy") or {}
    return {
        "seed": int(seed),
        "return_code": int(proc.returncode),
        "elapsed_seconds": float(elapsed),
        "summary_path": str(summary_path),
        "verdict_class": summary.get("verdict_class"),
        "pair_accuracy": pair_acc,
        "arm_a": (summary.get("stages") or {}).get("arm_a", {}),
        "arm_b": (summary.get("stages") or {}).get("arm_b", {}),
        "split": (summary.get("stages") or {}).get("split", {}),
        "pair_construction": (summary.get("stages") or {}).get("pair_construction", {}),
    }


def t_critical_95(n: int) -> float:
    table = {
        2: 12.7062047364,
        3: 4.3026527299,
        4: 3.1824463053,
        5: 2.7764451052,
        6: 2.5705818366,
        7: 2.4469118511,
        8: 2.3646242510,
        9: 2.3060041350,
        10: 2.2621571628,
    }
    if n <= 1:
        return float("nan")
    return table.get(n, 1.9599639845)


def aggregate_seed_rows(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in seed_rows if int(r.get("return_code", 1)) == 0]
    deltas = []
    pos_ci = 0
    neg_ci = 0
    per_seed = []
    for row in seed_rows:
        pa = row.get("pair_accuracy") or {}
        delta = float(pa.get("delta_pp", float("nan")))
        ci_lo = float(pa.get("ci95_lo", float("nan")))
        ci_hi = float(pa.get("ci95_hi", float("nan")))
        if np.isfinite(delta):
            deltas.append(delta)
        if bool(pa.get("ci_excludes_zero_positive", False)):
            pos_ci += 1
        if bool(pa.get("ci_excludes_zero_negative", False)):
            neg_ci += 1
        per_seed.append({
            "seed": row.get("seed"),
            "return_code": row.get("return_code"),
            "verdict_class": row.get("verdict_class"),
            "delta_pp": delta,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "arm_a_pair_acc": pa.get("headline_pair_accuracy_arm_a"),
            "arm_b_pair_acc": pa.get("headline_pair_accuracy_arm_b"),
            "summary_path": row.get("summary_path"),
        })

    n = len(deltas)
    mean_delta = float(np.mean(deltas)) if deltas else float("nan")
    sd_delta = float(np.std(deltas, ddof=1)) if n >= 2 else float("nan")
    se_delta = sd_delta / math.sqrt(n) if n >= 2 else float("nan")
    tcrit = t_critical_95(n)
    ci_lo = mean_delta - tcrit * se_delta if n >= 2 else float("nan")
    ci_hi = mean_delta + tcrit * se_delta if n >= 2 else float("nan")

    all_finished = len(ok_rows) == len(seed_rows)
    if not all_finished:
        verdict = "RUNTIME-ERROR-SEED-FAILED"
        rule = "one or more per-seed parent Q6 runs returned non-zero"
    elif n == len(seed_rows) and pos_ci == len(seed_rows) and ci_lo > 0 and mean_delta >= MULTISEED_ESTABLISHED_DELTA_PP:
        verdict = "CONFIRMED-DPO-PEBS-DOMINATES-MULTISEED"
        rule = "all seeds positive, per-seed CIs exclude 0, cross-seed CI excludes 0, mean >= +1.5pp"
    elif n == len(seed_rows) and pos_ci >= max(2, len(seed_rows) - 1) and mean_delta >= MULTISEED_MODERATE_DELTA_PP:
        verdict = "PARTIAL-DPO-PEBS-PARTIAL-DOMINATES-MULTISEED"
        rule = "at least 2 seeds positive with mean >= +1.0pp"
    elif n == len(seed_rows) and (mean_delta <= 0 or neg_ci > 0):
        verdict = "REJECTED-DPO-SEED-ARTIFACT"
        rule = "mean delta <= 0 or at least one per-seed CI is strictly negative"
    else:
        verdict = "TENTATIVE-DPO-SEED-VARIABLE"
        rule = "seed effects are mixed or cross-seed uncertainty remains too wide"

    return {
        "verdict_class": verdict,
        "decision_rule": rule,
        "n_seeds_requested": int(len(seed_rows)),
        "n_seeds_completed": int(len(ok_rows)),
        "n_seeds_with_finite_delta": int(n),
        "n_per_seed_ci_positive": int(pos_ci),
        "n_per_seed_ci_negative": int(neg_ci),
        "mean_delta_pp": mean_delta,
        "sd_delta_pp": sd_delta,
        "se_delta_pp": se_delta,
        "cross_seed_ci95_lo_pp": float(ci_lo),
        "cross_seed_ci95_hi_pp": float(ci_hi),
        "t_critical_95": float(tcrit) if np.isfinite(tcrit) else float("nan"),
        "per_seed": per_seed,
    }


def build_summary(args: argparse.Namespace, preflight_result: dict[str, Any], seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = aggregate_seed_rows(seed_rows)
    return {
        "experiment_id": "Q6_MS3_dpo_multiseed_mistral",
        "verdict_class": aggregate["verdict_class"],
        "args": vars(args),
        "anchors": {
            "trigger": "Q6 cross-backbone Mistral strongly positive",
            "base_model": args.base_model_id,
            "prior_q6_dir": str(args.prior_q6_output_dir),
            "no_internal_kill_switches": "no time-based watchdogs or subprocess timeouts",
        },
        "honest_disclosure_caveats": [
            "This is a three-additional-seed Mistral replication of held-out PRISM pair-accuracy.",
            "AlpacaEval-2 is skipped by default, matching Q6 cross-backbone; policy quality claim remains PRISM held-out pair-accuracy unless later KPI runs are added.",
            "Per-seed user-disjoint splits vary with seed; all other DPO hyperparameters inherit from the completed Mistral Q6 run.",
            "REJECTED and TENTATIVE outcome branches are explicit and publishable as seed-sensitivity bounds.",
        ],
        "stages": {
            "preflight": preflight_result,
            "per_seed_runs": seed_rows,
            "cross_seed_aggregate": aggregate,
            "held_out_pair_accuracy": {
                "delta_pp": aggregate["mean_delta_pp"],
                "ci95_lo": aggregate["cross_seed_ci95_lo_pp"],
                "ci95_hi": aggregate["cross_seed_ci95_hi_pp"],
                "n_seeds": aggregate["n_seeds_with_finite_delta"],
                "per_seed": aggregate["per_seed"],
            },
        },
        "completion_timestamp_utc": utc_now(),
        "git_head_t3_at_run": git_head(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_id", "--base-model-id", default=DEFAULT_BASE_MODEL_ID)
    p.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--prior-q6-output-dir", default=str(DEFAULT_PRIOR_Q6_DIR))
    p.add_argument("--utterances-parquet", default=None)
    p.add_argument("--rm-scored-parquet", default=None)
    p.add_argument("--calibrators-parquet", default=None)
    p.add_argument("--n-boot", type=int, default=q6.N_BOOT)
    p.add_argument("--held-out-frac", type=float, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--dpo-beta", type=float, default=None)
    p.add_argument("--dpo-lr", type=float, default=None)
    p.add_argument("--dpo-batch", type=int, default=None)
    p.add_argument("--dpo-grad-accum", type=int, default=None)
    p.add_argument("--dpo-max-seq-len", type=int, default=None)
    p.add_argument("--dpo-max-prompt-len", type=int, default=None)
    p.add_argument("--dpo-num-epochs", type=int, default=None)
    p.add_argument("--run-alpaca-eval", action="store_true")
    p.add_argument("--cpu-smoke", action="store_true")
    p.add_argument("--smoke-pairs", type=int, default=200)
    p.add_argument(
        "--allow-pair-fallback-for-smoke",
        action="store_true",
        help=(
            "Only for local CPU smoke when Mistral pair parquets were not "
            "synced; uses canonical Q6 pair parquets to verify wrapper logic."
        ),
    )
    args = p.parse_args()
    args.skip_alpaca_eval = not bool(args.run_alpaca_eval)
    return args


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        preflight_result = preflight(args)
    except Exception as exc:
        err = {
            "status": "FAIL",
            "timestamp_utc": utc_now(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (out_dir / "preflight.json").write_text(json.dumps(err, indent=2, default=str))
        log(f"[Q6-MS3] preflight FAIL: {exc}")
        log(traceback.format_exc())
        return 2

    (out_dir / "preflight.json").write_text(
        json.dumps(preflight_result, indent=2, default=str)
    )
    log("[Q6-MS3] preflight PASS")

    if args.cpu_smoke:
        smoke_summary = {
            "experiment_id": "Q6_MS3_dpo_multiseed_mistral_cpu_smoke",
            "verdict_class": "CPU-SMOKE-PASS",
            "args": vars(args),
            "stages": {"preflight": preflight_result},
            "completion_timestamp_utc": utc_now(),
            "git_head_t3_at_run": git_head(),
        }
        (out_dir / "summary.json").write_text(json.dumps(smoke_summary, indent=2, default=str))
        log("[Q6-MS3] CPU smoke PASS; no GPU training launched")
        return 0

    inherited_hparams = preflight_result.get("hparams_inherited_from_prior_q6", {})
    seed_rows = []
    for seed in preflight_result["seeds"]:
        row = run_seed(int(seed), args, inherited_hparams)
        seed_rows.append(row)
        (out_dir / "per_seed_summaries.json").write_text(
            json.dumps(seed_rows, indent=2, default=str)
        )
        if int(row.get("return_code", 1)) != 0:
            log(f"[Q6-MS3] seed {seed} failed; stopping before later seeds")
            break

    summary = build_summary(args, preflight_result, seed_rows)
    (out_dir / "cross_seed_summary.json").write_text(
        json.dumps(summary["stages"]["cross_seed_aggregate"], indent=2, default=str)
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    agg = summary["stages"]["cross_seed_aggregate"]
    print()
    print("=" * 72)
    print("Q6-MS3 Mistral multi-seed DPO replication")
    print("=" * 72)
    print(f"verdict_class       : {summary['verdict_class']}")
    print(f"mean delta          : {agg['mean_delta_pp']:+.2f}pp")
    print(
        "cross-seed CI       : "
        f"[{agg['cross_seed_ci95_lo_pp']:+.2f}, {agg['cross_seed_ci95_hi_pp']:+.2f}]"
    )
    print(f"n seeds completed   : {agg['n_seeds_completed']}/{agg['n_seeds_requested']}")
    print(f"summary_path        : {out_dir / 'summary.json'}")
    print(f"decision_rule       : {agg['decision_rule']}")
    print()
    return 0 if summary["verdict_class"] != "RUNTIME-ERROR-SEED-FAILED" else 1


if __name__ == "__main__":
    sys.exit(main())
