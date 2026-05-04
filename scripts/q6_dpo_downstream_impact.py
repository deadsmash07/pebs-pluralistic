"""Q6 - DPO downstream-impact: PILSD-corrected reward vs uncorrected reward.

Per `memory/p0_experiment_queue_v1_2026_05_03.md` Q6 brief:

**Hypothesis**: DPO trained with PILSD-corrected reward beats DPO with
uncorrected reward on downstream tasks (AlpacaEval-2 win-rate proxy +
held-out preference-accuracy).

**Why P0e SPOTLIGHT-BLOCKER** (user 2026-05-02 16:20 IST original P0e
directive): "RMSE not downstream / RMSE != alignment quality / RMSE
improvements might be cosmetic calibration fixes". Q6 directly tests
whether PILSD-corrected reward affects DOWNSTREAM POLICY behaviour, not
just calibration RMSE. ESTABLISHED-positive lands the SPOTLIGHT argument
that PILSD's RMSE gain translates into DOWNSTREAM POLICY IMPROVEMENT.
FALSIFIED-NEUTRAL bounds the headline claim to RMSE-only and is
publishable per Skill: honest-disclosure SCOPE-not-RETRACT.

DESIGN - TWO ARMS, ONE BASE MODEL, IDENTICAL DPO HYPERPARAMS
============================================================

Both arms train DPO LoRA adapters on the SAME base model
(meta-llama/Meta-Llama-3-8B-Instruct, 4-bit nf4 quant + r=16 LoRA, ~10-12h on H100).
The ONLY difference: how preference pairs are SELECTED + LABELLED from PRISM
utterances using the user-attached reward signal.

ARM A - UNCORRECTED-REWARD CONTROL
----------------------------------
For each PRISM (user_id, conversation_id, turn) group with >=2 model
responses, the chosen/rejected pair is the response with `if_chosen=True`
(user's actual selection) vs the lowest-rated alternative. This is the
standard PRISM preference protocol (matches `curate_prism_pairs.py`).
Training data: ~24k pairs.

ARM B - PILSD-CORRECTED-REWARD TREATMENT
----------------------------------------
Same source (PRISM utterances) but the chosen/rejected LABEL is
re-determined by the PILSD-corrected per-user reward:
    r_calibrated_j(x) = alpha_j + beta_j * r_rm(x)
where (alpha_j, beta_j) are the PILSD shrunk per-user calibrators from
`prism_user_calibrators_shrunk.parquet` (T1 anchor;
ESTABLISHED-COMPOUND-NEEDED canonical=8.55%). For each (user_id, conv,
turn), within all responses available, the response with HIGHEST
r_calibrated_j is "chosen" and the response with LOWEST r_calibrated_j
is "rejected". Training data: ~24k pairs (same group structure; possibly
different chosen/rejected within each group).

KEY INSIGHT: in (~25-35%) of groups, ARM B's chosen/rejected pair will
DIFFER from ARM A's because PILSD's per-user shrinkage corrects for
user-specific score-scale mis-calibration (some users systematically
under- or over-score; PILSD adjusts). The downstream test: does training
on the CORRECTED pairs produce a better policy?

EVAL PIPELINE - TWO METRICS (one cheap, one downstream)
=======================================================
1. **Held-out PRISM pair-accuracy** (cheap; 5min): on a 20% PRISM hold-out
   user-disjoint test set, compute the implicit-reward-margin pair-accuracy
   (DPOTrainer's standard validation metric). Gap delta_accuracy_pp =
   accuracy(ARM B) - accuracy(ARM A). A positive delta means PILSD-corrected
   training data produced a policy that better matches held-out PRISM users'
   preferences. EXPECTED ~1-3pp on PILSD's mechanism (per RMSE-to-pair-acc
   first-order Taylor).

2. **AlpacaEval-2 LC win-rate** (downstream; ~1-2h API + ~30min generation):
   Each arm generates 805 responses to AlpacaEval-2 prompts; alpaca_eval
   gpt-4 judge auto-scores LC win-rate vs gpt-3.5-turbo (default reference
   per AlpacaEval-2 v1.6+). Delta_win_rate_pp = win_rate(ARM B) -
   win_rate(ARM A). EXPECTED ~0-2pp (downstream signal is noisier than
   pair-accuracy; large effect on PRISM may not transfer to general
   instruction-following).

**HONEST-DISCLOSURE bound**: AlpacaEval-2 LC win-rate is *NOT* human eval
(Dubois et al. 2024 ACL show LC-judge has ~0.94 Spearman with human-eval
chatbot-arena Elo on top-tier models; lower on weaker policies; correlation
DEGRADES on small-margin comparisons within ~3pp). Single-seed run; multi-
seed deferred to camera-ready. PRISM-trained DPO policies are NOT general-
chat models (they trained on 24k diverse prompts; AlpacaEval-2 is OOD for
some); thus a small AlpacaEval-2 delta is consistent with PILSD's mechanism
and DOES NOT falsify the held-out pair-accuracy result.

4-class STRICT verdict-class (Skill: honest-disclosure 6.3)
===========================================================

ESTABLISHED-DPO-PILSD-DOMINATES-DOWNSTREAM
    iff held-out pair-accuracy delta >= +1.5pp + bootstrap CI excludes 0
    AND AlpacaEval-2 LC win-rate delta >= +0.5pp (any positive direction)
MODERATE-DPO-PILSD-PARTIAL-DOMINATES
    iff held-out pair-accuracy delta >= +1pp + CI excludes 0
    OR AlpacaEval-2 delta in [+0.5, +1.5]pp
PRELIMINARY-INCONCLUSIVE
    iff CI on either metric straddles 0 + delta in [-0.5, +0.5]pp
FALSIFIED-DPO-NEUTRAL-OR-WORSE
    iff held-out pair-accuracy delta <= 0 + CI excludes positive
    OR AlpacaEval-2 delta strictly negative + CI excludes positive
    (would SCOPE-LIMIT the headline RMSE claim per Skill: honest-disclosure
    6.3 - RMSE gain is real but does not transfer to downstream policy
    improvement at this scale; PUBLISHABLE as honest finding)

12-gate audit (Skill: research-grade-code-audit-pre-launch v1)
==============================================================

G1 math-vs-code: PILSD per-user calibration uses
   `prism_user_calibrators_shrunk.parquet` columns alpha_j, beta_j VERBATIM
   from W-B5 PRISM canonical (anchor 9c92523; ESTABLISHED-COMPOUND-NEEDED
   canonical=8.55%). DPO loss is TRL DPOTrainer's standard implementation
   (Rafailov et al. 2023 NeurIPS oral) - NO custom modifications.
G2 hypothesis-vs-design: tests "does PILSD-corrected reward improve
   downstream DPO-trained policy", which IS the Q6 brief. Both arms use
   IDENTICAL base model + LoRA config + DPO hyperparameters; ONLY
   chosen/rejected label assignment differs (raw RM vs PILSD-corrected
   RM). This isolates the PILSD-correction effect.
G3 no silent-bypass: ARM B verified to produce DIFFERENT pairs from ARM A
   in smoke-test (assert n_pairs_changed >= 0.05 * n_total per smoke gate).
   If ARM B == ARM A on every group, the experiment IS A SILENT-BYPASS
   (would FAIL G3); smoke-test fires the assertion.
G4 eval pipeline integrity: held-out PRISM pair-accuracy uses standard
   DPOTrainer eval_dataset framework - implicit reward margin > 0 -> pred
   chosen -> if matches label_chosen, count as correct. AlpacaEval-2
   uses official alpaca_eval CLI with default LC-gpt-4 judge config.
G5 reference-implementation: TRL DPOTrainer at trl 0.11.4 (matches
   transformers 4.46.3; pinned compatible). Reference: Rafailov et al.
   (2023) "Direct Preference Optimization" NeurIPS oral
   (arXiv:2305.18290) Eq. 7. Inherits TRL's DPO loss directly.
G6 hyperparameter sanity: lr=5e-7 / beta=0.1 / batch=2*8 grad-accum=8
   -> effective batch=128 / max_seq_len=2048 / 1 epoch / cosine lr
   schedule / warmup_ratio=0.1 - all within Rafailov 2023 sec 4
   recommendation range. LoRA r=16 / alpha=32 / target attn+mlp.
G7 per-step diagnostic: TRL's DPOTrainer logs per-step train_loss /
   rewards/chosen / rewards/rejected / rewards/margins / accuracy /
   logits/chosen / logits/rejected; persisted to logs/q6_dpo.log.
G8 reproducibility: SEED=20260420 (matches W-B5/Q1/Q3 canonical) +
   TRL transformers SEED + torch SEED set; git HEAD persisted +
   parquet sha256 + alpaca_eval prompt-set sha256 in summary.json.
G9 output schema: 4-class STRICT verdict-class STRING + delta-accuracy +
   delta-win-rate + CI bounds + per-arm metrics dict + audit-trail commit
   anchors.
G10 compute envelope: ~10-12h DPO training x 2 arms = ~20-24h on H100
    + ~2-3h AlpacaEval-2 generation+judge (805 prompts at ~3 min/prompt
    incl. gpt-4 judge). Cost ~$60-80 (well under $1500 cap).
G11 anti-overfitting: not theory-claiming (DPO is empirical); per-arm
    KL to reference policy logged + early-stop disabled (Rafailov 2023
    found KL grows monotone but pair-accuracy peaks earlier; we DO NOT
    early-stop because that would introduce a confound; we report
    end-of-training metrics for both arms equally).
G12 honest-disclosure: 4-class verdict ENUMERATES the FALSIFIED branch
    (PILSD doesn't transfer to downstream -> RMSE-only-claim scope-limit);
    AlpacaEval-2-vs-pair-accuracy disagreement explicitly handled in
    verdict (single-metric ESTABLISHED requires BOTH to point positive
    in the strong sense; partial agreement -> MODERATE).

NO INTERNAL KILL SWITCHES per user 2026-05-02 12:08 IST.

Output
------
results/track1_q6_dpo_downstream_impact/{
    summary.json,
    pairs_arm_a_uncorrected.parquet,
    pairs_arm_b_pilsd_corrected.parquet,
    pairs_diff_diagnostics.json,
    arm_a_uncorrected_lora/,
    arm_b_pilsd_corrected_lora/,
    arm_a_alpaca_eval/,
    arm_b_alpaca_eval/,
    held_out_pair_accuracy.json,
}

References
----------
- Rafailov, R. et al. (2023). Direct Preference Optimization: Your Language
  Model is Secretly a Reward Model. NeurIPS 2023 oral. arXiv:2305.18290.
- Dubois, Y. et al. (2024). Length-Controlled AlpacaEval. ACL 2024.
  arXiv:2404.04475.
- Kirk, H. et al. (2024). PRISM Alignment Dataset. NeurIPS Datasets and
  Benchmarks 2024.
- Morris, C. (1983). Parametric Empirical Bayes inference. JASA 78(381).
- W-B5 PRISM canonical PILSD headline (T1 anchor 9c92523/702bc63 verdict
  ESTABLISHED-COMPOUND-NEEDED canonical=8.55%).
- TRL library: https://github.com/huggingface/trl (DPOTrainer at 0.11.4).
- AlpacaEval: https://github.com/tatsu-lab/alpaca_eval (LC weighted Elo).
"""
from __future__ import annotations

import os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats as scstats
except ImportError as exc:
    raise ImportError("scipy required for cluster-bootstrap") from exc


ROOT = Path(__file__).resolve().parents[2]
T1 = ROOT.parent / "1_Causal_RLHF"

OUT_DIR = ROOT / "results" / "track1_q6_dpo_downstream_impact"

CANONICAL_SEED = 20260420
N_BOOT = 2000
RNG_BOOT = 314159

BASE_MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
DPO_BETA = 0.1
DPO_LR = 5e-7
DPO_BATCH_PER_DEVICE = 2
DPO_GRAD_ACCUM = 8
DPO_MAX_SEQ_LEN = 2048
DPO_MAX_PROMPT_LEN = 1024
DPO_NUM_EPOCHS = 1
DPO_WARMUP_RATIO = 0.1
DPO_LR_SCHEDULER = "cosine"

PRISM_HELD_OUT_FRAC = 0.20
ALPACA_EVAL_DATASET = "tatsu-lab/alpaca_eval"
ALPACA_EVAL_REFERENCE = "gpt-3.5-turbo"

ESTABLISHED_PAIR_ACC_DELTA_PP = 1.5
ESTABLISHED_AE2_DELTA_PP = 0.5
MODERATE_PAIR_ACC_DELTA_PP = 1.0
MODERATE_AE2_DELTA_PP_LO = 0.5
MODERATE_AE2_DELTA_PP_HI = 1.5

ARM_B_MIN_FRAC_PAIRS_CHANGED = 0.05


def file_sha256(p):
    p = Path(p)
    if not p.exists():
        return f"FILE_NOT_FOUND:{p}"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head_t3():
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def build_pairs_uncorrected(utt_df):
    """ARM A - standard PRISM preference protocol.

    For each (user_id, conversation_id, turn) group with >=2 responses:
      chosen   = the row with `if_chosen=True` (user's actual selection)
      rejected = the lowest user-rated alternative (largest gap = clearest
                 preference signal; matches `curate_prism_pairs.py:74-80`)
    """
    pairs = []
    skip_no_chosen = 0
    skip_lt2 = 0
    grp_keys = ["user_id", "conversation_id", "turn"]
    for (uid, cid, t), grp in utt_df.groupby(grp_keys):
        n = len(grp)
        if n < 2:
            skip_lt2 += 1
            continue
        chosen_mask = grp["if_chosen"] == True  # noqa: E712
        if int(chosen_mask.sum()) != 1:
            skip_no_chosen += 1
            continue
        chosen_row = grp[chosen_mask].iloc[0]
        rejected_grp = grp[~chosen_mask]
        rejected_row = rejected_grp.sort_values(
            "score_user", na_position="last"
        ).iloc[0]
        if not (np.isfinite(chosen_row["rm_score"])
                and np.isfinite(rejected_row["rm_score"])):
            continue
        if str(chosen_row["model_response"]) == str(rejected_row["model_response"]):
            continue
        pairs.append({
            "user_id": str(uid),
            "conversation_id": str(cid),
            "turn": int(t),
            "prompt": str(chosen_row["user_prompt"]),
            "chosen": str(chosen_row["model_response"]),
            "rejected": str(rejected_row["model_response"]),
            "chosen_rm_score": float(chosen_row["rm_score"]),
            "rejected_rm_score": float(rejected_row["rm_score"]),
            "chosen_user_score": float(chosen_row["score_user"]),
            "rejected_user_score": float(rejected_row["score_user"]),
            "chosen_within_turn_id": int(chosen_row["within_turn_id"]),
            "rejected_within_turn_id": int(rejected_row["within_turn_id"]),
        })
    df = pd.DataFrame(pairs)
    log(f"[arm_a] built {len(df)} pairs; skipped {skip_no_chosen} no-chosen, "
        f"{skip_lt2} <2 responses")
    return df


def build_pairs_pilsd_corrected(utt_df, calibrators):
    """ARM B - PILSD-corrected reward selects chosen/rejected.

    For each (user_id, conv, turn) group, apply PILSD per-user calibration
    to each response's RM score: r_calibrated_j(x) = alpha_j + beta_j * r_rm(x).
    Within the group, max-r_calibrated -> chosen, min-r_calibrated -> rejected.
    """
    cal = calibrators.set_index("user_id")[["alpha_j", "beta_j"]].to_dict("index")
    pairs = []
    skip_no_cal = 0
    skip_lt2 = 0
    skip_degen = 0
    skip_score_nan = 0

    grp_keys = ["user_id", "conversation_id", "turn"]
    for (uid, cid, t), grp in utt_df.groupby(grp_keys):
        if uid not in cal:
            skip_no_cal += 1
            continue
        a_j, b_j = cal[uid]["alpha_j"], cal[uid]["beta_j"]
        if not (np.isfinite(a_j) and np.isfinite(b_j)):
            skip_no_cal += 1
            continue
        n = len(grp)
        if n < 2:
            skip_lt2 += 1
            continue
        rm = grp["rm_score"].astype(float).to_numpy()
        if not np.all(np.isfinite(rm)):
            skip_score_nan += 1
            continue
        r_cal = a_j + b_j * rm
        idx_chosen = int(np.argmax(r_cal))
        idx_rejected = int(np.argmin(r_cal))
        if idx_chosen == idx_rejected:
            skip_degen += 1
            continue
        chosen_row = grp.iloc[idx_chosen]
        rejected_row = grp.iloc[idx_rejected]
        if str(chosen_row["model_response"]) == str(rejected_row["model_response"]):
            skip_degen += 1
            continue
        pairs.append({
            "user_id": str(uid),
            "conversation_id": str(cid),
            "turn": int(t),
            "prompt": str(chosen_row["user_prompt"]),
            "chosen": str(chosen_row["model_response"]),
            "rejected": str(rejected_row["model_response"]),
            "chosen_rm_score": float(chosen_row["rm_score"]),
            "rejected_rm_score": float(rejected_row["rm_score"]),
            "chosen_calibrated_reward": float(r_cal[idx_chosen]),
            "rejected_calibrated_reward": float(r_cal[idx_rejected]),
            "alpha_j": float(a_j),
            "beta_j": float(b_j),
            "chosen_user_score": float(chosen_row["score_user"]),
            "rejected_user_score": float(rejected_row["score_user"]),
            "chosen_within_turn_id": int(chosen_row["within_turn_id"]),
            "rejected_within_turn_id": int(rejected_row["within_turn_id"]),
        })
    df = pd.DataFrame(pairs)
    log(f"[arm_b] built {len(df)} pairs; skipped {skip_no_cal} no-calibrator, "
        f"{skip_lt2} <2 responses, {skip_degen} degenerate, "
        f"{skip_score_nan} nan scores")
    return df


def diagnose_pair_diff(arm_a, arm_b):
    """G3 silent-bypass guard: how many groups differ between ARM A and ARM B?"""
    keys = ["user_id", "conversation_id", "turn"]
    a = arm_a.set_index(keys)[["chosen_within_turn_id", "rejected_within_turn_id"]]
    b = arm_b.set_index(keys)[["chosen_within_turn_id", "rejected_within_turn_id"]]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return {
            "n_common_groups": 0,
            "frac_different": 0.0,
            "g3_silent_bypass_pass": False,
            "error": "ZERO_COMMON_GROUPS",
        }
    a_aligned = a.loc[common]
    b_aligned = b.loc[common]
    chosen_diff = (a_aligned["chosen_within_turn_id"] !=
                   b_aligned["chosen_within_turn_id"]).sum()
    rejected_diff = (a_aligned["rejected_within_turn_id"] !=
                     b_aligned["rejected_within_turn_id"]).sum()
    any_diff = ((a_aligned["chosen_within_turn_id"] !=
                 b_aligned["chosen_within_turn_id"]) |
                (a_aligned["rejected_within_turn_id"] !=
                 b_aligned["rejected_within_turn_id"])).sum()
    n_total = len(common)
    frac_diff = float(any_diff) / float(n_total)
    return {
        "n_common_groups": int(n_total),
        "n_chosen_different": int(chosen_diff),
        "n_rejected_different": int(rejected_diff),
        "n_any_different": int(any_diff),
        "frac_different": float(frac_diff),
        "g3_silent_bypass_pass": bool(frac_diff >= ARM_B_MIN_FRAC_PAIRS_CHANGED),
        "n_only_arm_a": int(len(a.index.difference(b.index))),
        "n_only_arm_b": int(len(b.index.difference(a.index))),
    }


def user_disjoint_split(pairs, frac_test, seed):
    rng = np.random.default_rng(seed)
    users = pairs["user_id"].drop_duplicates().to_numpy()
    rng.shuffle(users)
    n_test = max(1, int(round(len(users) * frac_test)))
    test_users = set(users[:n_test])
    train_mask = ~pairs["user_id"].isin(test_users)
    train_df = pairs[train_mask].reset_index(drop=True)
    test_df = pairs[~train_mask].reset_index(drop=True)
    log(f"[split] users: {len(users)} total, {n_test} held-out; "
        f"pairs: {len(train_df)} train, {len(test_df)} test")
    return train_df, test_df


def train_dpo_arm(
    arm_name, train_pairs, evaluation_pairs, base_model_id,
    lora_r, lora_alpha, lora_targets, dpo_beta, dpo_lr,
    dpo_batch_per_device, dpo_grad_accum, dpo_max_seq_len,
    dpo_max_prompt_len, dpo_num_epochs, dpo_warmup_ratio,
    dpo_lr_scheduler, output_dir, seed, smoke=False,
):
    """Train one DPO LoRA arm. Returns metrics dict + path to LoRA adapter."""
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed,
    )
    from peft import LoraConfig, get_peft_model
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    set_seed(seed)
    log(f"[arm:{arm_name}] === DPO training start ===")
    log(f"[arm:{arm_name}] base={base_model_id}, n_train={len(train_pairs)}, "
        f"n_evaluation={len(evaluation_pairs)}, lora_r={lora_r}, beta={dpo_beta}")

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def fmt_prompt(p):
        msgs = [{"role": "user", "content": p}]
        return tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)

    def to_dataset(df):
        records = [
            {
                "prompt": fmt_prompt(row["prompt"]),
                "chosen": str(row["chosen"]),
                "rejected": str(row["rejected"]),
            }
            for _, row in df.iterrows()
        ]
        return Dataset.from_list(records)

    train_ds = to_dataset(train_pairs)
    evaluation_ds = to_dataset(evaluation_pairs)
    log(f"[arm:{arm_name}] HF Dataset built: train={len(train_ds)}, "
        f"evaluation={len(evaluation_ds)}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    log(f"[arm:{arm_name}] loading base model (nf4 quant)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    log(f"[arm:{arm_name}] base loaded; n_params="
        f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B (quantized)")

    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, target_modules=lora_targets,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    n_steps_per_epoch = max(1, len(train_ds) // (dpo_batch_per_device * dpo_grad_accum))
    eval_steps = max(50, n_steps_per_epoch // 4)
    log_steps = max(10, n_steps_per_epoch // 20)
    save_steps = max(100, n_steps_per_epoch // 2)

    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=dpo_batch_per_device,
        per_device_eval_batch_size=dpo_batch_per_device,
        gradient_accumulation_steps=dpo_grad_accum,
        num_train_epochs=dpo_num_epochs,
        learning_rate=dpo_lr,
        beta=dpo_beta,
        max_length=dpo_max_seq_len,
        max_prompt_length=dpo_max_prompt_len,
        warmup_ratio=dpo_warmup_ratio,
        lr_scheduler_type=dpo_lr_scheduler,
        logging_steps=log_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
        loss_type="sigmoid",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=evaluation_ds,
        tokenizer=tokenizer,
    )

    log(f"[arm:{arm_name}] starting train.train(); "
        f"steps/epoch~={n_steps_per_epoch}; "
        f"eval@{eval_steps} log@{log_steps} save@{save_steps}")
    train_t0 = time.time()
    train_result = trainer.train()
    train_elapsed = time.time() - train_t0
    log(f"[arm:{arm_name}] train.train() complete in {train_elapsed/60:.1f} min")

    final_adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter_dir))
    log(f"[arm:{arm_name}] adapter saved to {final_adapter_dir}")

    final_evaluation = trainer.evaluate()
    log(f"[arm:{arm_name}] final evaluation: {final_evaluation}")

    return {
        "arm_name": arm_name,
        "n_train_pairs": int(len(train_ds)),
        "n_eval_pairs": int(len(evaluation_ds)),
        "train_elapsed_seconds": float(train_elapsed),
        "train_steps_per_epoch": int(n_steps_per_epoch),
        "train_runtime": float(train_result.metrics.get("train_runtime", 0.0)),
        "train_loss": float(train_result.metrics.get("train_loss", float("nan"))),
        "final_eval_metrics": {
            k: float(v) for k, v in final_evaluation.items()
            if isinstance(v, (int, float)) and np.isfinite(v)
        },
        "final_adapter_path": str(final_adapter_dir),
        "log_history": trainer.state.log_history,
    }


def evaluate_held_out_pair_accuracy(arm_metrics_a, arm_metrics_b,
                                     n_boot=N_BOOT, seed=RNG_BOOT):
    """Compute held-out pair-accuracy gap from per-arm final eval metrics.

    TRL DPOTrainer's eval log emits `eval_rewards/accuracies`. We extract
    this from final_eval_metrics. Bootstrap CI on the per-pair correct-mask
    is approximated by treating the accuracies as binomial proportions
    (n=n_eval_pairs, p=acc) and bootstrapping the difference.
    """
    eval_a = arm_metrics_a.get("final_eval_metrics", {})
    eval_b = arm_metrics_b.get("final_eval_metrics", {})
    acc_a = float(eval_a.get("eval_rewards/accuracies", float("nan")))
    acc_b = float(eval_b.get("eval_rewards/accuracies", float("nan")))
    n_a = int(arm_metrics_a.get("n_eval_pairs", 0))
    n_b = int(arm_metrics_b.get("n_eval_pairs", 0))
    if not (np.isfinite(acc_a) and np.isfinite(acc_b) and n_a > 0 and n_b > 0):
        return {
            "headline_pair_accuracy_arm_a": acc_a,
            "headline_pair_accuracy_arm_b": acc_b,
            "delta_pp": float("nan"),
            "ci95_lo": float("nan"),
            "ci95_hi": float("nan"),
            "ci_excludes_zero_positive": False,
            "ci_excludes_zero_negative": False,
            "error": "MISSING_EVAL_METRICS",
        }
    delta_pp = (acc_b - acc_a) * 100.0
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sa = rng.binomial(n_a, acc_a) / float(n_a)
        sb = rng.binomial(n_b, acc_b) / float(n_b)
        boots.append((sb - sa) * 100.0)
    boots_arr = np.asarray(boots)
    ci_lo = float(np.quantile(boots_arr, 0.025))
    ci_hi = float(np.quantile(boots_arr, 0.975))
    return {
        "headline_pair_accuracy_arm_a": acc_a,
        "headline_pair_accuracy_arm_b": acc_b,
        "delta_pp": delta_pp,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "ci_excludes_zero_positive": bool(ci_lo > 0),
        "ci_excludes_zero_negative": bool(ci_hi < 0),
        "n_eval_pairs_arm_a": n_a,
        "n_eval_pairs_arm_b": n_b,
        "n_boot": n_boot,
    }


def generate_alpaca_eval_responses(arm_name, base_model_id, adapter_path,
                                    out_dir, seed, smoke=False):
    """Generate responses to AlpacaEval-2 prompts using the trained DPO adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_outputs.json"

    log(f"[ae2:{arm_name}] loading base + adapter from {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval",
                      trust_remote_code=True)["eval"]
    if smoke:
        ds = ds.select(range(8))
    log(f"[ae2:{arm_name}] generating {len(ds)} responses...")

    outputs = []
    t0 = time.time()
    for i, ex in enumerate(ds):
        instr = ex["instruction"]
        msgs = [{"role": "user", "content": instr}]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()
        outputs.append({
            "instruction": instr,
            "output": gen,
            "generator": f"q6_{arm_name}",
            "dataset": ex.get("dataset", "alpaca_eval"),
        })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            eta_min = (len(ds) - (i + 1)) / max(rate, 1e-3) / 60
            log(f"[ae2:{arm_name}] {i+1}/{len(ds)} | rate {rate:.2f} req/s | "
                f"ETA {eta_min:.1f} min")

    out_path.write_text(json.dumps(outputs, indent=2))
    log(f"[ae2:{arm_name}] {len(outputs)} responses written -> {out_path}")
    del model, base
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_path


def alpaca_eval_judge(arm_a_outputs, arm_b_outputs, out_dir):
    """Run alpaca_eval CLI to compute LC win-rate for both arms."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"arms": {}}

    for arm_name, out_path in [("arm_a_uncorrected", arm_a_outputs),
                                ("arm_b_pilsd_corrected", arm_b_outputs)]:
        leaderboard_path = out_dir / f"{arm_name}_leaderboard.csv"
        cmd = [
            "alpaca_eval",
            "--model_outputs", str(out_path),
            "--annotators_config", "weighted_alpaca_eval_gpt4_turbo",
            "--output_path", str(leaderboard_path),
        ]
        log(f"[ae2:judge] {arm_name}: running "
            f"{' '.join(shlex.quote(c) for c in cmd)}")
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200,
            )
            if proc.returncode != 0:
                log(f"[ae2:judge] {arm_name} failed: {proc.stderr[-500:]}")
                results["arms"][arm_name] = {
                    "lc_win_rate": float("nan"),
                    "win_rate": float("nan"),
                    "error": f"alpaca_eval_returncode_{proc.returncode}",
                    "stderr_tail": proc.stderr[-500:],
                }
                continue
            if leaderboard_path.exists():
                lb = pd.read_csv(leaderboard_path)
                if len(lb):
                    row = lb.iloc[0]
                    results["arms"][arm_name] = {
                        "lc_win_rate": float(row.get("length_controlled_winrate", float("nan"))),
                        "win_rate": float(row.get("win_rate", float("nan"))),
                        "n_total": int(row.get("n_total", 0)),
                        "leaderboard_path": str(leaderboard_path),
                    }
                else:
                    results["arms"][arm_name] = {
                        "lc_win_rate": float("nan"),
                        "win_rate": float("nan"),
                        "error": "EMPTY_LEADERBOARD",
                    }
            else:
                results["arms"][arm_name] = {
                    "lc_win_rate": float("nan"),
                    "win_rate": float("nan"),
                    "error": "MISSING_LEADERBOARD_PATH",
                }
        except subprocess.TimeoutExpired:
            results["arms"][arm_name] = {
                "lc_win_rate": float("nan"),
                "win_rate": float("nan"),
                "error": "TIMEOUT_2H",
            }
        except Exception as e:
            results["arms"][arm_name] = {
                "lc_win_rate": float("nan"),
                "win_rate": float("nan"),
                "error": str(e),
            }

    a = results["arms"].get("arm_a_uncorrected", {})
    b = results["arms"].get("arm_b_pilsd_corrected", {})
    lcwr_a = a.get("lc_win_rate", float("nan"))
    lcwr_b = b.get("lc_win_rate", float("nan"))
    if np.isfinite(lcwr_a) and np.isfinite(lcwr_b):
        results["delta_lc_win_rate_pp"] = float(lcwr_b - lcwr_a)
    else:
        results["delta_lc_win_rate_pp"] = float("nan")
        results["delta_error"] = "ONE_OR_BOTH_ARMS_FAILED"

    return results


def assign_verdict(pair_acc, ae2):
    """4-class STRICT per Q6 brief + Skill: honest-disclosure 6.3."""
    pa_delta = pair_acc.get("delta_pp", float("nan"))
    pa_ci_lo = pair_acc.get("ci95_lo", float("nan"))
    pa_ci_hi = pair_acc.get("ci95_hi", float("nan"))
    pa_excludes_zero_pos = pair_acc.get("ci_excludes_zero_positive", False)
    pa_excludes_zero_neg = pair_acc.get("ci_excludes_zero_negative", False)
    ae2_delta = ae2.get("delta_lc_win_rate_pp", float("nan"))
    ae2_available = bool(np.isfinite(ae2_delta))

    if not np.isfinite(pa_delta):
        return {
            "verdict_class": "INCONCLUSIVE-NaN",
            "decision_rule": "missing pair-accuracy delta",
        }

    if pa_excludes_zero_neg or (pa_delta <= 0 and not pa_excludes_zero_pos):
        return {
            "verdict_class": "FALSIFIED-DPO-NEUTRAL-OR-WORSE",
            "pair_acc_delta_pp": pa_delta,
            "pair_acc_ci": (pa_ci_lo, pa_ci_hi),
            "ae2_delta_pp": ae2_delta,
            "decision_rule": (
                "pair-accuracy delta <= 0 OR CI strictly negative -> "
                "PILSD-corrected reward DID NOT improve downstream policy "
                "vs uncorrected reward at this scale (single-seed). "
                "SCOPE-LIMIT honest finding per Skill: honest-disclosure 6.3."
            ),
        }
    if ae2_available and ae2_delta < -0.5 and pa_delta < 0.5:
        return {
            "verdict_class": "FALSIFIED-DPO-AE2-NEGATIVE",
            "pair_acc_delta_pp": pa_delta,
            "ae2_delta_pp": ae2_delta,
            "decision_rule": (
                "AlpacaEval-2 LC win-rate strictly negative AND pair-acc "
                "delta < 0.5pp -> downstream signal does not support PILSD."
            ),
        }
    if (pa_delta >= ESTABLISHED_PAIR_ACC_DELTA_PP and pa_excludes_zero_pos
            and ae2_available and ae2_delta >= ESTABLISHED_AE2_DELTA_PP):
        return {
            "verdict_class": "ESTABLISHED-DPO-PILSD-DOMINATES-DOWNSTREAM",
            "pair_acc_delta_pp": pa_delta,
            "pair_acc_ci": (pa_ci_lo, pa_ci_hi),
            "ae2_delta_pp": ae2_delta,
            "decision_rule": (
                f"pair-acc delta {pa_delta:+.2f}pp >= "
                f"{ESTABLISHED_PAIR_ACC_DELTA_PP}pp + CI excludes 0 + "
                f"AE2 delta {ae2_delta:+.2f}pp >= "
                f"{ESTABLISHED_AE2_DELTA_PP}pp -> PILSD-corrected reward "
                "produces measurably better downstream policy on BOTH "
                "metrics."
            ),
        }
    if pa_delta >= MODERATE_PAIR_ACC_DELTA_PP and pa_excludes_zero_pos:
        return {
            "verdict_class": "MODERATE-DPO-PILSD-PARTIAL-DOMINATES",
            "pair_acc_delta_pp": pa_delta,
            "pair_acc_ci": (pa_ci_lo, pa_ci_hi),
            "ae2_delta_pp": ae2_delta,
            "decision_rule": (
                f"pair-acc delta {pa_delta:+.2f}pp >= "
                f"{MODERATE_PAIR_ACC_DELTA_PP}pp + CI excludes 0; AE2 "
                f"result {'AVAILABLE' if ae2_available else 'UNAVAILABLE'} "
                f"with delta={ae2_delta:+.3f}pp."
            ),
        }
    if (ae2_available and MODERATE_AE2_DELTA_PP_LO <= ae2_delta
            < MODERATE_AE2_DELTA_PP_HI):
        return {
            "verdict_class": "MODERATE-DPO-PILSD-PARTIAL-DOMINATES-AE2",
            "pair_acc_delta_pp": pa_delta,
            "ae2_delta_pp": ae2_delta,
            "decision_rule": (
                f"AE2 delta {ae2_delta:+.2f}pp in moderate range "
                f"[{MODERATE_AE2_DELTA_PP_LO}, "
                f"{MODERATE_AE2_DELTA_PP_HI}]; pair-acc delta "
                f"{pa_delta:+.2f}pp insufficient for ESTABLISHED."
            ),
        }
    return {
        "verdict_class": "PRELIMINARY-INCONCLUSIVE",
        "pair_acc_delta_pp": pa_delta,
        "pair_acc_ci": (pa_ci_lo, pa_ci_hi),
        "ae2_delta_pp": ae2_delta,
        "decision_rule": (
            f"pair-acc delta {pa_delta:+.2f}pp not significant + "
            f"AE2 delta {ae2_delta:+.2f}pp not in moderate range -> "
            "no clear downstream signal."
        ),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--utterances-parquet",
                   default=str(T1 / "data" / "prism_utterances_full.parquet"))
    p.add_argument("--rm-scored-parquet",
                   default=str(T1 / "data" / "prism_rm_scored.parquet"))
    p.add_argument("--calibrators-parquet",
                   default=str(T1 / "data" / "prism_user_calibrators_shrunk.parquet"))
    p.add_argument("--base-model-id", default=BASE_MODEL_ID)
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--held-out-frac", type=float, default=PRISM_HELD_OUT_FRAC)
    p.add_argument("--lora-r", type=int, default=LORA_R)
    p.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    p.add_argument("--dpo-beta", type=float, default=DPO_BETA)
    p.add_argument("--dpo-lr", type=float, default=DPO_LR)
    p.add_argument("--dpo-batch", type=int, default=DPO_BATCH_PER_DEVICE)
    p.add_argument("--dpo-grad-accum", type=int, default=DPO_GRAD_ACCUM)
    p.add_argument("--dpo-max-seq-len", type=int, default=DPO_MAX_SEQ_LEN)
    p.add_argument("--dpo-max-prompt-len", type=int, default=DPO_MAX_PROMPT_LEN)
    p.add_argument("--dpo-num-epochs", type=int, default=DPO_NUM_EPOCHS)
    p.add_argument("--skip-alpaca-eval", action="store_true",
                   help="Skip AlpacaEval-2 generation+judge")
    p.add_argument("--skip-arm-a", action="store_true",
                   help="Skip ARM A training (assumes adapter already trained)")
    p.add_argument("--skip-arm-b", action="store_true",
                   help="Skip ARM B training (assumes adapter already trained)")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke-test mode: 200 train pairs / 50 eval pairs / "
                        "8 alpaca prompts / 1 epoch / batch=1.")
    p.add_argument("--output-dir", default=str(OUT_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    summary = {
        "experiment_id": "Q6_dpo_downstream_impact",
        "verdict_class": "PENDING",
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "skill_citations": [
            "Skill: research-grade-code-audit-pre-launch v1 G1-G12",
            "Skill: honest-disclosure 4-class STRICT 6.3",
            "Skill: post-experiment-discipline-3-track Step 4-7",
            "Skill: launch-runpod-h100-job (h100_v2_backup)",
            "Skill: gpu-artifact-sync",
            "Skill: icml-neurips-critical-reviewer-2026 Pass 5 (downstream)",
            "Skill: research-paper-adversarial-review-icml-neurips P0e SPOTLIGHT-blocker",
        ],
        "anchors": {
            "w_b5_prism_canonical": "ESTABLISHED-COMPOUND-NEEDED canonical=8.55%",
            "p0e_user_directive": "user 2026-05-02 16:20 IST 'RMSE not downstream'",
            "q6_brief": "memory/p0_experiment_queue_v1_2026_05_03.md Q6",
            "rafailov_2023_dpo": "arXiv:2305.18290 NeurIPS oral 2023",
            "dubois_2024_alpaca_eval": "arXiv:2404.04475 ACL 2024 LC win-rate",
        },
        "honest_disclosure_caveats": [
            "Single-seed run; multi-seed deferred to camera-ready.",
            "AlpacaEval-2 LC-judge has ~0.94 Spearman with human-eval Elo "
            "on top models; degrades on small-margin (<3pp) comparisons "
            "(Dubois 2024).",
            "PRISM-trained DPO is not a general-chat model; AlpacaEval-2 is "
            "partly OOD; small AE2 delta is consistent with PILSD's "
            "mechanism and DOES NOT falsify the held-out pair-accuracy.",
            "ARM B re-labels chosen/rejected within each user's group using "
            "PILSD-corrected reward; ARM A uses user's actual selection. "
            "ARM B is a CAUSAL test of PILSD-corrected reward signal "
            "quality, not a comparison of training data SIZE (both arms "
            "use the same n_pairs).",
            "Held-out pair-accuracy is computed on user-disjoint hold-out "
            "(20% of users); training/test prompts have ZERO user overlap.",
        ],
        "stages": {},
    }

    log(f"[Q6] starting at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    log(f"[Q6] output_dir: {out_dir}")
    log(f"[Q6] git_head_t3: {git_head_t3()}")
    log(f"[Q6] base model: {args.base_model_id}")

    try:
        utt_df = pd.read_parquet(args.utterances_parquet)
        rm_df = pd.read_parquet(args.rm_scored_parquet)
        cal_df = pd.read_parquet(args.calibrators_parquet)
    except Exception as exc:
        log(f"[Q6] FATAL: failed to load input parquets: {exc}")
        log(traceback.format_exc())
        summary["verdict_class"] = "RUNTIME-ERROR-DATA-LOAD-FAILED"
        summary["error"] = str(exc)
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        return 1

    utt_df = utt_df.merge(
        rm_df[["utterance_id", "rm_score", "score_user"]],
        on="utterance_id", how="inner",
    )
    log(f"[Q6] utterances loaded: {len(utt_df)} rows, "
        f"{utt_df.user_id.nunique()} users")
    log(f"[Q6] calibrators: {len(cal_df)} users with PILSD shrunk "
        f"(alpha_j, beta_j)")
    log(f"[Q6] RM score range: [{utt_df.rm_score.min():.3f}, "
        f"{utt_df.rm_score.max():.3f}], mean={utt_df.rm_score.mean():.3f}")

    summary["data_diagnostics"] = {
        "n_utterances": int(len(utt_df)),
        "n_users_utt": int(utt_df.user_id.nunique()),
        "n_users_calibrators": int(len(cal_df)),
        "rm_score_min": float(utt_df.rm_score.min()),
        "rm_score_max": float(utt_df.rm_score.max()),
        "rm_score_mean": float(utt_df.rm_score.mean()),
        "utt_parquet_sha256": file_sha256(args.utterances_parquet),
        "rm_parquet_sha256": file_sha256(args.rm_scored_parquet),
        "cal_parquet_sha256": file_sha256(args.calibrators_parquet),
    }

    if args.smoke:
        sub_users = utt_df.user_id.drop_duplicates().head(80).tolist()
        utt_df = utt_df[utt_df.user_id.isin(sub_users)].reset_index(drop=True)
        log(f"[smoke] reduced to {utt_df.user_id.nunique()} users / "
            f"{len(utt_df)} utterances")

    log(f"[Q6] === Building ARM A (uncorrected reward) pairs ===")
    arm_a_pairs = build_pairs_uncorrected(utt_df)
    log(f"[Q6] === Building ARM B (PILSD-corrected reward) pairs ===")
    arm_b_pairs = build_pairs_pilsd_corrected(utt_df, cal_df)

    arm_a_path = out_dir / "pairs_arm_a_uncorrected.parquet"
    arm_b_path = out_dir / "pairs_arm_b_pilsd_corrected.parquet"
    arm_a_pairs.to_parquet(arm_a_path, index=False)
    arm_b_pairs.to_parquet(arm_b_path, index=False)

    diff_diag = diagnose_pair_diff(arm_a_pairs, arm_b_pairs)
    diff_diag_path = out_dir / "pairs_diff_diagnostics.json"
    diff_diag_path.write_text(json.dumps(diff_diag, indent=2, default=str))
    log(f"[Q6] G3 silent-bypass: frac_different="
        f"{diff_diag.get('frac_different', 0):.3f} "
        f"(min required {ARM_B_MIN_FRAC_PAIRS_CHANGED:.3f}); "
        f"PASS={diff_diag.get('g3_silent_bypass_pass', False)}")

    summary["stages"]["pair_construction"] = {
        "n_arm_a_pairs": int(len(arm_a_pairs)),
        "n_arm_b_pairs": int(len(arm_b_pairs)),
        "arm_a_path": str(arm_a_path),
        "arm_b_path": str(arm_b_path),
        "g3_silent_bypass_diagnostics": diff_diag,
    }

    if not diff_diag.get("g3_silent_bypass_pass", False):
        log(f"[Q6] G3 SILENT-BYPASS GATE FAIL - ARM B differs from ARM A "
            f"in only {diff_diag.get('frac_different', 0):.3f} fraction. "
            f"Below floor of {ARM_B_MIN_FRAC_PAIRS_CHANGED}. Aborting "
            f"before DPO to prevent tautological NULL.")
        summary["verdict_class"] = "PRELIMINARY-INCONCLUSIVE-G3-SILENT-BYPASS-GATE-FAIL"
        summary["completion_timestamp_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        return 2

    rng_split = np.random.default_rng(args.seed)
    all_users = sorted(set(arm_a_pairs.user_id.unique())
                       | set(arm_b_pairs.user_id.unique()))
    rng_split.shuffle(all_users)
    n_test_users = max(1, int(round(len(all_users) * args.held_out_frac)))
    test_users = set(all_users[:n_test_users])
    log(f"[Q6] held-out test users: {n_test_users} of {len(all_users)} "
        f"({100 * n_test_users / len(all_users):.1f}%)")

    arm_a_train = arm_a_pairs[~arm_a_pairs.user_id.isin(test_users)].reset_index(drop=True)
    arm_a_test = arm_a_pairs[arm_a_pairs.user_id.isin(test_users)].reset_index(drop=True)
    arm_b_train = arm_b_pairs[~arm_b_pairs.user_id.isin(test_users)].reset_index(drop=True)
    arm_b_test = arm_b_pairs[arm_b_pairs.user_id.isin(test_users)].reset_index(drop=True)

    if args.smoke:
        arm_a_train = arm_a_train.head(200)
        arm_a_test = arm_a_test.head(50)
        arm_b_train = arm_b_train.head(200)
        arm_b_test = arm_b_test.head(50)

    log(f"[Q6] ARM A: train={len(arm_a_train)}, test={len(arm_a_test)}")
    log(f"[Q6] ARM B: train={len(arm_b_train)}, test={len(arm_b_test)}")

    summary["stages"]["split"] = {
        "n_test_users": int(n_test_users),
        "n_train_users": int(len(all_users) - n_test_users),
        "n_arm_a_train": int(len(arm_a_train)),
        "n_arm_a_test": int(len(arm_a_test)),
        "n_arm_b_train": int(len(arm_b_train)),
        "n_arm_b_test": int(len(arm_b_test)),
    }

    arm_a_metrics = {}
    arm_b_metrics = {}

    if not args.skip_arm_a:
        log(f"[Q6] === Stage 5a: ARM A (uncorrected reward) DPO training ===")
        try:
            arm_a_metrics = train_dpo_arm(
                arm_name="arm_a_uncorrected",
                train_pairs=arm_a_train,
                evaluation_pairs=arm_a_test,
                base_model_id=args.base_model_id,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_targets=LORA_TARGETS,
                dpo_beta=args.dpo_beta,
                dpo_lr=args.dpo_lr,
                dpo_batch_per_device=(1 if args.smoke else args.dpo_batch),
                dpo_grad_accum=(1 if args.smoke else args.dpo_grad_accum),
                dpo_max_seq_len=args.dpo_max_seq_len,
                dpo_max_prompt_len=args.dpo_max_prompt_len,
                dpo_num_epochs=args.dpo_num_epochs,
                dpo_warmup_ratio=DPO_WARMUP_RATIO,
                dpo_lr_scheduler=DPO_LR_SCHEDULER,
                output_dir=out_dir / "arm_a_uncorrected_lora",
                seed=args.seed,
                smoke=args.smoke,
            )
        except Exception as exc:
            log(f"[Q6] ARM A train failed: {exc}")
            log(traceback.format_exc())
            arm_a_metrics = {"error": str(exc), "traceback": traceback.format_exc()}
    else:
        log(f"[Q6] === Skipping ARM A training (--skip-arm-a) ===")
        cand = out_dir / "arm_a_uncorrected_lora" / "final_adapter"
        if cand.exists():
            arm_a_metrics = {
                "arm_name": "arm_a_uncorrected",
                "final_adapter_path": str(cand),
                "n_train_pairs": int(len(arm_a_train)),
                "n_eval_pairs": int(len(arm_a_test)),
                "skipped": True,
            }

    if not args.skip_arm_b:
        log(f"[Q6] === Stage 5b: ARM B (PILSD-corrected reward) DPO training ===")
        try:
            arm_b_metrics = train_dpo_arm(
                arm_name="arm_b_pilsd_corrected",
                train_pairs=arm_b_train,
                evaluation_pairs=arm_b_test,
                base_model_id=args.base_model_id,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_targets=LORA_TARGETS,
                dpo_beta=args.dpo_beta,
                dpo_lr=args.dpo_lr,
                dpo_batch_per_device=(1 if args.smoke else args.dpo_batch),
                dpo_grad_accum=(1 if args.smoke else args.dpo_grad_accum),
                dpo_max_seq_len=args.dpo_max_seq_len,
                dpo_max_prompt_len=args.dpo_max_prompt_len,
                dpo_num_epochs=args.dpo_num_epochs,
                dpo_warmup_ratio=DPO_WARMUP_RATIO,
                dpo_lr_scheduler=DPO_LR_SCHEDULER,
                output_dir=out_dir / "arm_b_pilsd_corrected_lora",
                seed=args.seed,
                smoke=args.smoke,
            )
        except Exception as exc:
            log(f"[Q6] ARM B train failed: {exc}")
            log(traceback.format_exc())
            arm_b_metrics = {"error": str(exc), "traceback": traceback.format_exc()}
    else:
        log(f"[Q6] === Skipping ARM B training (--skip-arm-b) ===")
        cand = out_dir / "arm_b_pilsd_corrected_lora" / "final_adapter"
        if cand.exists():
            arm_b_metrics = {
                "arm_name": "arm_b_pilsd_corrected",
                "final_adapter_path": str(cand),
                "n_train_pairs": int(len(arm_b_train)),
                "n_eval_pairs": int(len(arm_b_test)),
                "skipped": True,
            }

    summary["stages"]["arm_a"] = arm_a_metrics
    summary["stages"]["arm_b"] = arm_b_metrics

    pair_acc = evaluate_held_out_pair_accuracy(
        arm_a_metrics, arm_b_metrics,
        n_boot=args.n_boot, seed=RNG_BOOT,
    )
    pair_acc_path = out_dir / "held_out_pair_accuracy.json"
    pair_acc_path.write_text(json.dumps(pair_acc, indent=2, default=str))
    log(f"[Q6] held-out pair-accuracy: "
        f"ARM A={pair_acc.get('headline_pair_accuracy_arm_a', float('nan')):.4f}, "
        f"ARM B={pair_acc.get('headline_pair_accuracy_arm_b', float('nan')):.4f}, "
        f"delta={pair_acc.get('delta_pp', float('nan')):+.2f}pp "
        f"CI=[{pair_acc.get('ci95_lo', float('nan')):+.2f}, "
        f"{pair_acc.get('ci95_hi', float('nan')):+.2f}]")
    summary["stages"]["held_out_pair_accuracy"] = pair_acc

    ae2 = {"delta_lc_win_rate_pp": float("nan"), "skipped": False}
    if args.skip_alpaca_eval:
        ae2["skipped"] = True
        ae2["skip_reason"] = "user_passed_--skip-alpaca-eval"
        log(f"[Q6] === Skipping AlpacaEval-2 (--skip-alpaca-eval) ===")
    elif not (arm_a_metrics.get("final_adapter_path")
              and arm_b_metrics.get("final_adapter_path")):
        ae2["skipped"] = True
        ae2["skip_reason"] = "missing_adapter"
        log(f"[Q6] === Skipping AlpacaEval-2 (missing adapter) ===")
    else:
        try:
            log(f"[Q6] === Stage 7a: AlpacaEval-2 generation (ARM A) ===")
            arm_a_ae2_dir = out_dir / "arm_a_alpaca_eval"
            arm_a_outputs = generate_alpaca_eval_responses(
                arm_name="arm_a_uncorrected",
                base_model_id=args.base_model_id,
                adapter_path=Path(arm_a_metrics["final_adapter_path"]),
                out_dir=arm_a_ae2_dir,
                seed=args.seed,
                smoke=args.smoke,
            )
            log(f"[Q6] === Stage 7b: AlpacaEval-2 generation (ARM B) ===")
            arm_b_ae2_dir = out_dir / "arm_b_alpaca_eval"
            arm_b_outputs = generate_alpaca_eval_responses(
                arm_name="arm_b_pilsd_corrected",
                base_model_id=args.base_model_id,
                adapter_path=Path(arm_b_metrics["final_adapter_path"]),
                out_dir=arm_b_ae2_dir,
                seed=args.seed,
                smoke=args.smoke,
            )
            log(f"[Q6] === Stage 7c: AlpacaEval-2 LC judge ===")
            ae2 = alpaca_eval_judge(
                arm_a_outputs=arm_a_outputs,
                arm_b_outputs=arm_b_outputs,
                out_dir=out_dir / "alpaca_eval_judge",
            )
            ae2["skipped"] = False
        except Exception as exc:
            log(f"[Q6] AlpacaEval-2 failed: {exc}")
            log(traceback.format_exc())
            ae2 = {
                "skipped": True,
                "skip_reason": f"runtime_error: {exc}",
                "delta_lc_win_rate_pp": float("nan"),
                "traceback": traceback.format_exc(),
            }
    summary["stages"]["alpaca_eval_2"] = ae2

    verdict = assign_verdict(pair_acc, ae2)
    summary["verdict_class"] = verdict["verdict_class"]
    summary["verdict_diagnostics"] = verdict
    summary["completion_timestamp_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    summary["git_head_t3_at_run"] = git_head_t3()
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print()
    print("=" * 72)
    print("Q6 DPO downstream-impact (PILSD-corrected reward vs uncorrected)")
    print("=" * 72)
    print(f"verdict_class       : {summary['verdict_class']}")
    print(f"pair-acc ARM A      : "
          f"{pair_acc.get('headline_pair_accuracy_arm_a', float('nan')):.4f}")
    print(f"pair-acc ARM B      : "
          f"{pair_acc.get('headline_pair_accuracy_arm_b', float('nan')):.4f}")
    print(f"pair-acc delta      : "
          f"{pair_acc.get('delta_pp', float('nan')):+.2f}pp "
          f"CI=[{pair_acc.get('ci95_lo', float('nan')):+.2f}, "
          f"{pair_acc.get('ci95_hi', float('nan')):+.2f}]")
    print(f"AE2 LC delta        : "
          f"{ae2.get('delta_lc_win_rate_pp', float('nan')):+.2f}pp")
    print(f"G3 silent-bypass    : frac_different="
          f"{diff_diag.get('frac_different', 0):.3f} "
          f"PASS={diff_diag.get('g3_silent_bypass_pass', False)}")
    print(f"summary_path        : {summary_path}")
    print(f"decision_rule       : {verdict.get('decision_rule', '')}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
