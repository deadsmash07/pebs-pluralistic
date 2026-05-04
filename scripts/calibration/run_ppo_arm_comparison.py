"""Track 1 downstream: PPO with vanilla / PILSD-calibrated / delayed PILSD reward.

Three arms are supported via --arm:
  vanilla         reward = r_raw (7B+LoRA RM output, no affine)
  pilsd           reward = alpha_ref + beta_ref * r_raw, no delay
  pilsd_delay     reward = alpha_ref + beta_ref * r_raw, delayed K optimizer
                  steps via a FIFO buffer around the RM's score head.

Delay semantics (T1+T2 integration experiment, 2026-04-18):
  At each RM forward call (once per PPO optimizer step, since TRL batches
  the rollout scoring into a single RM pass), the computed score tensor
  is pushed onto a FIFO buffer of length K and the OLDEST tensor in the
  buffer is returned. For the first K calls, the buffer returns zeros
  of the same shape — effectively "no reward signal yet" at early steps.

This simulates delayed feedback (e.g. when human ratings are not available
immediately), and connects to Track 2's RAC (Retroactive Advantage
Correction) premise: if delay degrades policy quality at fixed step count,
the RAC fix is needed; if it doesn't, delay is a cheap trade-off.

Hypothesis (H_ppo, H_delay):
  H_ppo:   PILSD β-scaling changes advantage magnitude -> moves policy in KL.
  H_delay: Delay K=5 reduces effective reward-credit-assignment resolution,
           EXPECTED to (i) reduce reward accumulation at step t<K+buffer
           because zeros are fed in; (ii) produce a different final policy
           vs no-delay at fixed step budget; (iii) ONLY show a large effect
           if step count is not much larger than K (here K=5 vs 500 steps,
           so the effect may be small — honest reporting required).

Arm A: reward = alpha_ref + beta_ref * r_raw, delay=0 (pilsd)
Arm B: reward = alpha_ref + beta_ref * r_raw, delay=K (pilsd_delay)

Policy: Qwen2.5-0.5B-Instruct (base + reset head from RM architecture).
Ref: frozen copy of policy init.
Value: AutoModelForSequenceClassification init from Qwen2.5-0.5B.
Reward: existing 7B+LoRA SEQ_CLS RM from results/e2_canonical_7b/.

Outputs:
  results/{outdir}/{arm}_seed{S}/metrics.csv
  results/{outdir}/{arm}_seed{S}/policy/
  results/{outdir}/{arm}_seed{S}/summary.json
  results/{outdir}/{arm}_seed{S}/reward_trace.json  (per-step RM calls)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import PPOConfig, PPOTrainer


class AffineScore(nn.Module):
    """Wrap an existing score head (Linear) with r' = alpha + beta * score(x).

    Records per-call mean for symmetry with DelayedAffineScore (delay=0 case).
    """

    def __init__(self, inner_score: nn.Module, alpha: float, beta: float):
        super().__init__()
        self.inner = inner_score
        self.register_buffer("alpha_buf", torch.tensor(float(alpha)))
        self.register_buffer("beta_buf", torch.tensor(float(beta)))
        self.call_idx = 0
        self.call_log = []

    def forward(self, x):
        out = self.alpha_buf + self.beta_buf * self.inner(x)
        self.call_log.append({
            "call": self.call_idx, "outer_step": self.call_idx,
            "true_mean": float(out.float().mean().detach().cpu()),
            "emitted_mean": float(out.float().mean().detach().cpu()),
            "delayed": False, "is_new_outer": True,
        })
        self.call_idx += 1
        return out


class DelayedAffineScore(nn.Module):
    """Affine-calibrated RM score with a K-OUTER-STEP FIFO delay on emissions.

    Semantics (per-unique-batch delay — TRL's PPOTrainer calls the RM
    multiple times on the same rollout batch within one optimizer step,
    so we dedup consecutive identical calls to map delay to OUTER steps):

      unique_call t=0..K-1   -> returns zero tensor (shape matching inner output)
      unique_call t=K,K+1,.. -> returns the score computed at unique_call (t-K).

    The `call_log` records EVERY raw forward call with true_mean/emitted_mean/
    delayed/outer_step so we can quantify "reward accumulates differently
    when delayed".

    Deduplication uses the hash of the input hidden-state tensor's first row's
    sum as a signature (cheap, stable within an optimizer step).
    """

    def __init__(self, inner_score: nn.Module, alpha: float, beta: float, delay: int):
        super().__init__()
        self.inner = inner_score
        self.register_buffer("alpha_buf", torch.tensor(float(alpha)))
        self.register_buffer("beta_buf", torch.tensor(float(beta)))
        assert delay >= 0
        self.delay = int(delay)
        self.buffer = []          # FIFO of per-outer-step score tensors
        self.current_outer_emit = None  # cached emit for duplicate calls
        self.last_sig = None      # signature of last UNIQUE input
        self.outer_step = 0       # increments only on UNIQUE calls
        self.call_idx = 0
        self.call_log = []        # per-raw-call log

    @staticmethod
    def _sig(x):
        # Cheap content signature: sum of first row. Different outer-step rollouts
        # produce different hidden states; within one outer step the RM is called
        # with the same inputs, producing the same signature.
        try:
            flat = x.detach().float().reshape(x.shape[0], -1)
            return float(flat[0].sum().item())
        except Exception:
            return None

    def forward(self, x):
        true = self.alpha_buf + self.beta_buf * self.inner(x)  # (B, S, 1)
        if self.delay == 0:
            self.call_log.append({
                "call": self.call_idx, "outer_step": self.outer_step,
                "true_mean": float(true.float().mean().detach().cpu()),
                "emitted_mean": float(true.float().mean().detach().cpu()),
                "delayed": False, "is_new_outer": True,
            })
            self.call_idx += 1
            self.outer_step += 1
            return true

        # Reduce per-token scores to per-sample scalar for shape-invariant FIFO
        # storage. TRL indexes these at eos position later anyway; broadcasting
        # a per-sample scalar across seq dim preserves its value after indexing.
        # Stored shape is (B,) — we emit a per-token tensor of matching shape.
        per_sample_scalar = true.detach().float().mean(dim=tuple(range(1, true.ndim)))

        sig = self._sig(x)
        is_new = (sig is None) or (sig != self.last_sig)
        if is_new:
            self.buffer.append(per_sample_scalar.clone())
            if len(self.buffer) > self.delay:
                old_scalar = self.buffer.pop(0)
                self.current_outer_emit = old_scalar
                delayed_flag = True
            else:
                self.current_outer_emit = None  # signal zeros
                delayed_flag = False
            self.last_sig = sig
            self.outer_step += 1
        else:
            delayed_flag = self.current_outer_emit is not None

        # Build per-token emit tensor with same shape as true.
        if self.current_outer_emit is None:
            emit = torch.zeros_like(true)
        else:
            scalar = self.current_outer_emit  # (B_old,)
            B = true.shape[0]
            if scalar.shape[0] != B:
                # shape mismatch (rare): fall back to zeros
                emit = torch.zeros_like(true)
            else:
                # Broadcast scalar across all non-batch dims of true.
                expand_shape = [B] + [1] * (true.ndim - 1)
                emit = scalar.to(true.dtype).to(true.device).view(*expand_shape)
                emit = emit.expand_as(true).contiguous()

        self.call_log.append({
            "call": self.call_idx, "outer_step": self.outer_step - 1,
            "true_mean": float(true.float().mean().detach().cpu()),
            "emitted_mean": float(emit.float().mean().detach().cpu()),
            "delayed": bool(delayed_flag), "is_new_outer": bool(is_new),
        })
        self.call_idx += 1
        return emit


def apply_pilsd_to_rm(rm_model, alpha: float, beta: float, delay: int = 0):
    """Patch rm_model.score with (possibly delayed) affine calibration.

    Returns the wrapper module so caller can access call_log.
    """
    original = rm_model.score
    dev = next(original.parameters()).device
    if delay > 0:
        wrapper = DelayedAffineScore(original, alpha, beta, delay).to(dev)
    else:
        wrapper = AffineScore(original, alpha, beta).to(dev)
    rm_model.score = wrapper
    return wrapper


def build_prompt_dataset(prism_pairs_parquet, tokenizer, n_train, n_eval, seed,
                         max_prompt_tokens=128):
    df = pd.read_parquet(prism_pairs_parquet)
    df = df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    df = df[df["prompt"].str.len() > 0].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    train_idx = idx[:n_train]
    eval_idx = idx[n_train : n_train + n_eval]

    def _apply(df_slice):
        out = []
        for _, row in df_slice.iterrows():
            messages = [{"role": "user", "content": str(row["prompt"])[:800]}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            enc = tokenizer(text, truncation=True, max_length=max_prompt_tokens,
                            padding=False, add_special_tokens=False)
            out.append({"input_ids": enc["input_ids"], "lengths": len(enc["input_ids"])})
        return out

    train_rows = _apply(df.iloc[train_idx])
    eval_rows = _apply(df.iloc[eval_idx])
    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["vanilla", "pilsd", "pilsd_delay"], required=True)
    ap.add_argument("--reward-delay-steps", type=int, default=0,
                    help="Delay the reward K optimizer steps (used iff arm=pilsd_delay).")
    ap.add_argument("--policy-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--value-model-name", default=None,
                    help="Name of the base model for the value head. Defaults to --policy-name. "
                         "Use a smaller model (e.g. Qwen/Qwen2.5-0.5B-Instruct) when training 7B "
                         "policies on a single 80GB H100 to avoid OOM.")
    ap.add_argument("--rm-adapter-dir",
                    default="<DATA_ROOT>/1_Causal_RLHF/results/e2_canonical_7b")
    ap.add_argument("--rm-base-name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--pairs-parquet",
                    default="<DATA_ROOT>/1_Causal_RLHF/data/prism_pairs.parquet")
    ap.add_argument("--calibrators-parquet",
                    default="<DATA_ROOT>/1_Causal_RLHF/data/prism_user_calibrators_shrunk.parquet")
    ap.add_argument("--ref-user-id", default=None,
                    help="Exact user id. Takes precedence over --reference-user-mode.")
    ap.add_argument("--reference-user-mode", choices=["median", "random"], default="median",
                    help="If --ref-user-id unset: 'median' picks nearest-to-median-alpha user "
                         "(legacy 0.5B behaviour); 'random' picks uniformly at random using "
                         "--reference-user-seed so the 7B sweep is not contaminated with user344.")
    ap.add_argument("--reference-user-seed", type=int, default=None,
                    help="RNG seed for --reference-user-mode=random. Defaults to --seed.")
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--total-episodes", type=int, default=None,
                    help="Override total_episodes in PPOConfig. Defaults to --n-train-prompts. "
                         "Increase to run more PPO outer steps. Outer steps = "
                         "total_episodes / (per-device-batch * gradient-accumulation).")
    ap.add_argument("--n-train-prompts", type=int, default=256)
    ap.add_argument("--n-eval-prompts", type=int, default=100)
    ap.add_argument("--per-device-batch", type=int, default=8)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--num-train-epochs", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=1.0e-6)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--response-length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir",
                    default="<DATA_ROOT>/1_Causal_RLHF/results/track1_ppo_pilsd_vs_vanilla")
    args = ap.parse_args()

    out_dir = Path(args.output_dir) / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)

    cal = pd.read_parquet(args.calibrators_parquet)
    if args.arm in ("pilsd", "pilsd_delay"):
        if args.ref_user_id is not None:
            ref_row = cal[cal["user_id"] == args.ref_user_id].iloc[0]
            ref_mode = "explicit"
        elif args.reference_user_mode == "random":
            ref_seed = args.reference_user_seed if args.reference_user_seed is not None else args.seed
            rng_ref = np.random.default_rng(ref_seed)
            i = int(rng_ref.integers(0, len(cal)))
            ref_row = cal.iloc[i]
            ref_mode = f"random(seed={ref_seed},i={i})"
        else:
            med = cal["alpha_j"].median()
            ref_row = cal.iloc[(cal["alpha_j"] - med).abs().argsort()].iloc[0]
            ref_mode = "median"
        alpha_ref, beta_ref = float(ref_row["alpha_j"]), float(ref_row["beta_j"])
        ref_user = str(ref_row["user_id"])
        print(f"[{args.arm}] ref_user={ref_user} alpha={alpha_ref:.3f} "
              f"beta={beta_ref:.3f} mode={ref_mode}")
    else:
        alpha_ref, beta_ref, ref_user = 0.0, 1.0, None
        ref_mode = "n/a"
        print("[vanilla] no affine (alpha=0, beta=1)")

    # Delay is only active for pilsd_delay arm.
    effective_delay = args.reward_delay_steps if args.arm == "pilsd_delay" else 0
    print(f"[arm] {args.arm} effective_delay_steps={effective_delay}")

    print("[load] tokenizers")
    policy_tok = AutoTokenizer.from_pretrained(args.policy_name)
    if policy_tok.pad_token_id is None:
        policy_tok.pad_token = policy_tok.eos_token
    rm_tok = AutoTokenizer.from_pretrained(args.rm_base_name)
    if rm_tok.pad_token_id is None:
        rm_tok.pad_token = rm_tok.eos_token

    print("[data] building PRISM prompt datasets")
    train_ds, eval_ds = build_prompt_dataset(
        args.pairs_parquet, policy_tok,
        n_train=args.n_train_prompts, n_eval=args.n_eval_prompts, seed=args.seed,
    )
    print(f"[data] train={len(train_ds)} eval={len(eval_ds)}")

    print(f"[load] policy {args.policy_name}")
    policy = AutoModelForCausalLM.from_pretrained(
        args.policy_name, torch_dtype=torch.bfloat16
    )
    policy.generation_config.pad_token_id = policy_tok.pad_token_id
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(policy, lora_cfg)
    policy.print_trainable_parameters()

    print("[load] ref_policy (frozen)")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        args.policy_name, torch_dtype=torch.bfloat16
    )
    ref_policy.generation_config.pad_token_id = policy_tok.pad_token_id
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    value_model_name = args.value_model_name or args.policy_name
    print(f"[load] value_model {value_model_name}")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        value_model_name, num_labels=1, torch_dtype=torch.bfloat16
    )
    value_model.config.pad_token_id = policy_tok.pad_token_id

    print(f"[load] reward_model base={args.rm_base_name}")
    rm_base = AutoModelForSequenceClassification.from_pretrained(
        args.rm_base_name, num_labels=1, torch_dtype=torch.bfloat16
    )
    rm_base.config.pad_token_id = rm_tok.pad_token_id
    print(f"[load] reward_model adapter={args.rm_adapter_dir}")
    reward_model = PeftModel.from_pretrained(rm_base, args.rm_adapter_dir)
    reward_model.requires_grad_(False)

    if policy_tok.vocab_size != rm_tok.vocab_size:
        print(f"[warn] vocab size mismatch policy={policy_tok.vocab_size} "
              f"rm={rm_tok.vocab_size}.")

    score_wrapper = None  # populated iff we patch .score
    if args.arm in ("pilsd", "pilsd_delay"):
        print(f"[{args.arm}] wrapping RM.score with affine "
              f"(alpha={alpha_ref}, beta={beta_ref}, delay={effective_delay})")
        try:
            inner = reward_model.base_model.model
            _ = inner.score  # probe
            score_wrapper = apply_pilsd_to_rm(
                inner, alpha_ref, beta_ref, delay=effective_delay
            )
            print(f"[{args.arm}] patched inner.score -> {type(inner.score).__name__}")
        except Exception as e:
            print(f"[{args.arm}] inner patch failed: {e}; trying reward_model.score")
            score_wrapper = apply_pilsd_to_rm(
                reward_model, alpha_ref, beta_ref, delay=effective_delay
            )

    ppo_cfg = PPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_ppo_epochs=args.ppo_epochs,
        num_train_epochs=args.num_train_epochs,
        total_episodes=args.total_episodes if args.total_episodes is not None else args.n_train_prompts,
        response_length=args.response_length,
        kl_coef=args.kl_coef,
        temperature=0.9,
        num_mini_batches=1,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=True,
        seed=args.seed,
        num_sample_generations=0,
    )

    print("[trainer] init PPOTrainer")
    trainer = PPOTrainer(
        config=ppo_cfg,
        processing_class=policy_tok,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    print("[trainer] train")
    trainer.train()

    hist = trainer.state.log_history
    pd.DataFrame(hist).to_csv(out_dir / "metrics.csv", index=False)
    print(f"[save] {out_dir / 'metrics.csv'}")

    policy_out = out_dir / "policy"
    policy_out.mkdir(parents=True, exist_ok=True)
    try:
        # PPOTrainer wraps policy as PolicyAndValueWrapper; unwrap then save adapter.
        inner_policy = trainer.model.policy
        # inner_policy is the PEFT-wrapped CausalLM; save_pretrained writes adapter.
        inner_policy.save_pretrained(policy_out)
        print(f"[save] policy adapter -> {policy_out}")
    except Exception as e:
        print(f"[save] policy save failed: {e}")
        # Fallback: full model save
        try:
            trainer.save_model(str(policy_out))
        except Exception as e2:
            print(f"[save] trainer.save_model failed: {e2}")

    # Save per-call reward trace if we patched a wrapper that logs (delay arms).
    reward_trace = None
    if score_wrapper is not None and hasattr(score_wrapper, "call_log"):
        reward_trace = score_wrapper.call_log
        (out_dir / "reward_trace.json").write_text(json.dumps(reward_trace, indent=2))
        print(f"[save] {out_dir / 'reward_trace.json'} ({len(reward_trace)} calls)")

    summary = {
        "arm": args.arm,
        "ref_user": ref_user,
        "ref_user_mode": ref_mode,
        "alpha": alpha_ref,
        "beta": beta_ref,
        "reward_delay_steps": effective_delay,
        "policy_name": args.policy_name,
        "value_model_name": value_model_name,
        "rm_adapter": args.rm_adapter_dir,
        "n_train_prompts": args.n_train_prompts,
        "n_eval_prompts": args.n_eval_prompts,
        "learning_rate": args.learning_rate,
        "kl_coef": args.kl_coef,
        "response_length": args.response_length,
        "ppo_epochs": args.ppo_epochs,
        "seed": args.seed,
        "final_log": hist[-1] if hist else {},
        "n_rm_calls": len(reward_trace) if reward_trace is not None else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[save] {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
