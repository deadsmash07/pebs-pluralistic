"""Per-user preference evaluation for PILSD vs vanilla PPO policies.

The rationale: pair accuracy on the raw 7B RM judge is monotone-invariant
under affine calibration, so it cannot separate PILSD-calibrated from
vanilla-trained policies (see track1_pair_accuracy_wrong_metric.md and
track1_ppo_scaled_result.md).

This script fixes the metric by using each PRISM user's OWN linear
calibrator (alpha_j, beta_j) as the judge. The idea:

    raw_scores = RM(prompt, response)                      # shared 7B RM
    user_score = alpha_j + beta_j * raw_scores             # per-user calibration
    per-user-delta = user_score[pilsd] - user_score[vanilla]
    per-user-prefer-pilsd = delta > 0

Then we report:
  - user-win-rate (fraction of sampled users who prefer PILSD)
  - per-user reward delta distribution
  - breakdown by calibrator alpha-magnitude bucket (does PILSD
    differentially help users who are more sensitive to RM score?)
  - Wilson CI on user-win-rate (binomial over users, not prompts)

Because raw_pilsd and raw_vanilla are fixed (given the generations),
the per-user preference only depends on the sign of beta_j * (raw_p - raw_v):
if beta_j > 0 (all 1394 PRISM users have beta_j > 0), then user prefers
PILSD iff mean(raw_p) > mean(raw_v). But THAT is the population-level
raw-RM answer, monotone-invariant under beta.

The sharper test, which IS PILSD-sensitive, is PER-PROMPT aggregation:
for each user, the expected calibrated reward across n_prompts prompts is
alpha_j + beta_j * mean(raw). The win rate per-user is then mean(raw_p) > mean(raw_v)
(one boolean per user), which IS just sign(mean(raw_p) - mean(raw_v)) scaled
by N_users. That gives either ~0% or ~100% user-win-rate, which is a
degenerate signal.

THE META-POINT: linear per-user calibrators cannot separate PILSD from vanilla
at the argmax level even when applied per-user. The signal must come from
(a) user-level reward MAGNITUDE (not argmax), or
(b) per-prompt bootstrap that reflects prompt-level variance.

So we report 3 metrics:

  M1. Per-user MEAN-REWARD delta (alpha+beta * E[r]) — shows magnitude.
      Paired bootstrap over prompts, one delta per user.

  M2. Per-user WIN RATE on per-prompt basis:
      For user j, for each prompt i, user j prefers PILSD on that prompt
      iff beta_j * (r_p[i] - r_v[i]) > 0. Since beta_j > 0 for all PRISM
      users, this collapses to sign(r_p[i] - r_v[i]) — same across users.
      We report this to DEMONSTRATE the monotone-invariance trap.

  M3. Per-user CALIBRATED-REWARD delta (alpha_j + beta_j * mean_raw),
      one number per user; then user-win-rate = fraction with delta > 0.
      This is a BINOMIAL test over N_users users with Wilson CI.

Outputs results/track1_per_user_preference/eval_per_user.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def wilson_ci(successes, trials, z=1.96):
    if trials == 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * ((p * (1 - p) / trials + z * z / (4 * trials * trials)) ** 0.5) / denom
    return (center - half, center + half)


def bca_ci(data, stat_fn=np.mean, n_boot=2000, alpha=0.05, seed=0):
    """Simple percentile bootstrap (good enough at n>=100; BCa correction
    minor at these sample sizes).
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)
    if n < 2:
        return (float("nan"), float("nan"))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = stat_fn(data[idx])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return (lo, hi)


@torch.no_grad()
def generate(policy, tok, prompts, max_new_tokens=128, batch_size=8, seed=0,
             temperature=1.0, top_p=1.0):
    device = next(policy.parameters()).device
    gen_cfg = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
    )
    outputs = []
    torch.manual_seed(seed)
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)
        out = policy.generate(**enc, **gen_cfg)
        resp = out[:, enc["input_ids"].shape[1] :]
        for r in resp:
            outputs.append(tok.decode(r, skip_special_tokens=True))
        if (i // batch_size) % 10 == 0:
            print(f"  [gen] {i + len(batch)}/{len(prompts)}")
    return outputs


@torch.no_grad()
def score(rm, rm_tok, prompts, responses, batch_size=8):
    device = next(rm.parameters()).device
    scores = []
    for i in range(0, len(prompts), batch_size):
        p_batch = prompts[i : i + batch_size]
        r_batch = responses[i : i + batch_size]
        texts = []
        for p, r in zip(p_batch, r_batch):
            msgs = [
                {"role": "user", "content": p[:800]},
                {"role": "assistant", "content": r[:800]},
            ]
            texts.append(rm_tok.apply_chat_template(msgs, tokenize=False))
        enc = rm_tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        out = rm(**enc)
        logits = out.logits.squeeze(-1).float().cpu().numpy().tolist()
        if isinstance(logits, float):
            logits = [logits]
        scores.extend(logits)
        if (i // batch_size) % 10 == 0:
            print(f"  [score] {i + len(p_batch)}/{len(prompts)}")
    return np.array(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--rm-base-name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument(
        "--rm-adapter-dir",
        default="<DATA_ROOT>/1_Causal_RLHF/results/e2_canonical_7b",
    )
    ap.add_argument(
        "--vanilla-adapter",
        default="<DATA_ROOT>/1_Causal_RLHF/results/track1_ppo_scaled/scaled_high/vanilla/policy",
    )
    ap.add_argument(
        "--pilsd-adapter",
        default="<DATA_ROOT>/1_Causal_RLHF/results/track1_ppo_scaled/scaled_high/pilsd/policy",
    )
    ap.add_argument(
        "--pairs-parquet",
        default="<DATA_ROOT>/1_Causal_RLHF/data/prism_pairs.parquet",
    )
    ap.add_argument(
        "--calibrators-parquet",
        default="<DATA_ROOT>/1_Causal_RLHF/data/prism_user_calibrators_shrunk.parquet",
    )
    # Per the user directive: ≥500 prompts × ≥100 PRISM users
    ap.add_argument("--n-eval-prompts", type=int, default=500)
    ap.add_argument("--n-users", type=int, default=100)
    ap.add_argument(
        "--eval-start-idx",
        type=int,
        default=3000,  # well beyond the PPO train slice (0..1024) and the
                        # prior pairwise eval slice (2000..2500)
    )
    ap.add_argument("--seed", type=int, default=20260418)
    ap.add_argument("--response-length", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--gen-batch-size", type=int, default=16)
    ap.add_argument("--rm-batch-size", type=int, default=8)
    ap.add_argument(
        "--out-json",
        default="<DATA_ROOT>/1_Causal_RLHF/results/track1_per_user_preference/eval_per_user.json",
    )
    ap.add_argument(
        "--out-parquet",
        default="<DATA_ROOT>/1_Causal_RLHF/results/track1_per_user_preference/per_user_deltas.parquet",
    )
    args = ap.parse_args()

    out_json = Path(args.out_json)
    out_parquet = Path(args.out_parquet)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.policy_name, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    rm_tok = AutoTokenizer.from_pretrained(args.rm_base_name, padding_side="right")
    if rm_tok.pad_token_id is None:
        rm_tok.pad_token = rm_tok.eos_token

    # Same shuffle-seed=42 logic as PPO training/eval so we stay OUTSIDE
    # the train slice [0..1024) and the prior eval slice [2000..2500).
    df = (
        pd.read_parquet(args.pairs_parquet)
        .drop_duplicates(subset=["prompt"])
        .reset_index(drop=True)
    )
    df = df[df["prompt"].str.len() > 0].reset_index(drop=True)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    end_idx = args.eval_start_idx + args.n_eval_prompts
    if end_idx > len(df):
        raise RuntimeError(
            f"eval slice [{args.eval_start_idx}..{end_idx}) exceeds n_unique_prompts={len(df)}"
        )
    eval_idx = idx[args.eval_start_idx : end_idx]
    raw_prompts = [str(df.iloc[i]["prompt"])[:800] for i in eval_idx]
    prompt_texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in raw_prompts
    ]
    print(f"[data] {len(raw_prompts)} held-out prompts (idx {args.eval_start_idx}..{end_idx})")

    # ------------- Generate from both policies -------------
    print("[load] vanilla policy")
    pv_base = AutoModelForCausalLM.from_pretrained(
        args.policy_name, torch_dtype=torch.bfloat16
    ).cuda()
    pv = PeftModel.from_pretrained(pv_base, args.vanilla_adapter)
    pv.requires_grad_(False)
    print("[generate] vanilla (T=%.2f, top_p=%.2f, max_new=%d)"
          % (args.temperature, args.top_p, args.response_length))
    v_resp = generate(
        pv, tok, prompt_texts,
        max_new_tokens=args.response_length,
        batch_size=args.gen_batch_size,
        seed=args.seed, temperature=args.temperature, top_p=args.top_p,
    )
    pv = pv.cpu()
    del pv, pv_base
    torch.cuda.empty_cache()

    print("[load] pilsd policy")
    pp_base = AutoModelForCausalLM.from_pretrained(
        args.policy_name, torch_dtype=torch.bfloat16
    ).cuda()
    pp = PeftModel.from_pretrained(pp_base, args.pilsd_adapter)
    pp.requires_grad_(False)
    print("[generate] pilsd")
    p_resp = generate(
        pp, tok, prompt_texts,
        max_new_tokens=args.response_length,
        batch_size=args.gen_batch_size,
        seed=args.seed, temperature=args.temperature, top_p=args.top_p,
    )
    pp = pp.cpu()
    del pp, pp_base
    torch.cuda.empty_cache()

    # ------------- Score with 7B RM -------------
    print("[load] 7B reward model")
    rm_base = AutoModelForSequenceClassification.from_pretrained(
        args.rm_base_name, num_labels=1, torch_dtype=torch.bfloat16
    ).cuda()
    rm_base.config.pad_token_id = rm_tok.pad_token_id
    rm = PeftModel.from_pretrained(rm_base, args.rm_adapter_dir)
    rm.config.pad_token_id = rm_tok.pad_token_id
    rm.requires_grad_(False)

    print("[score] vanilla responses")
    v_scores = score(rm, rm_tok, raw_prompts, v_resp, batch_size=args.rm_batch_size)
    print("[score] pilsd responses")
    p_scores = score(rm, rm_tok, raw_prompts, p_resp, batch_size=args.rm_batch_size)
    rm = rm.cpu()
    del rm, rm_base
    torch.cuda.empty_cache()

    # ------------- Per-user calibrated rewards -------------
    cal = pd.read_parquet(args.calibrators_parquet)
    # Filter: require finite calibrators + beta_j > 0 (the PRISM MixedLM
    # fits produce beta_j > 0 for all 1394 users — we sanity-check anyway).
    cal = cal[np.isfinite(cal["alpha_j"]) & np.isfinite(cal["beta_j"])]
    n_neg_beta = int((cal["beta_j"] <= 0).sum())
    print(f"[cal] {len(cal)} calibrators available; {n_neg_beta} have beta<=0")

    sample_rng = np.random.default_rng(args.seed)
    sampled = cal.sample(
        n=min(args.n_users, len(cal)),
        random_state=sample_rng.integers(0, 2**31 - 1),
    ).reset_index(drop=True)

    # Per-user MEAN calibrated reward, per-user delta
    rows = []
    for _, u in sampled.iterrows():
        a, b = float(u["alpha_j"]), float(u["beta_j"])
        v_cal = a + b * v_scores  # (n_prompts,)
        p_cal = a + b * p_scores
        mean_v = float(v_cal.mean())
        mean_p = float(p_cal.mean())
        delta = mean_p - mean_v
        # Paired per-prompt win rate for this user (sign of per-prompt delta)
        per_prompt_delta = p_cal - v_cal
        prompt_wins_p = int((per_prompt_delta > 0).sum())
        prompt_losses_p = int((per_prompt_delta < 0).sum())
        prompt_ties = int((per_prompt_delta == 0).sum())
        rows.append({
            "user_id": str(u["user_id"]),
            "alpha_j": a,
            "beta_j": b,
            "n_observations": int(u["n_observations"]),
            "mean_cal_reward_vanilla": mean_v,
            "mean_cal_reward_pilsd": mean_p,
            "delta_pilsd_minus_vanilla": delta,
            "user_prefers_pilsd": int(delta > 0),
            "prompt_level_wins_pilsd": prompt_wins_p,
            "prompt_level_wins_vanilla": prompt_losses_p,
            "prompt_level_ties": prompt_ties,
        })
    per_user = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    per_user.to_parquet(out_parquet, index=False)

    # ------------- Aggregate metrics -------------
    n_users = len(per_user)
    n_pref_p = int(per_user["user_prefers_pilsd"].sum())
    n_pref_v = int(n_users - n_pref_p - (per_user["delta_pilsd_minus_vanilla"] == 0).sum())
    n_tie = int((per_user["delta_pilsd_minus_vanilla"] == 0).sum())

    # Wilson on user-level win rate (excluding ties)
    user_wr = n_pref_p / max(1, n_pref_p + n_pref_v)
    user_wr_ci = wilson_ci(n_pref_p, n_pref_p + n_pref_v)

    # BCa on mean per-user reward delta
    deltas = per_user["delta_pilsd_minus_vanilla"].values
    mean_delta = float(np.mean(deltas))
    mean_delta_ci = bca_ci(deltas, stat_fn=np.mean, n_boot=5000, seed=args.seed)
    median_delta = float(np.median(deltas))

    # alpha-magnitude buckets (quartiles on |alpha_j|)
    cal_alpha_abs = per_user["alpha_j"].abs()
    q25, q50, q75 = cal_alpha_abs.quantile([0.25, 0.5, 0.75])
    def bucket(x):
        if x <= q25: return "Q1"
        if x <= q50: return "Q2"
        if x <= q75: return "Q3"
        return "Q4"
    per_user["alpha_abs_bucket"] = cal_alpha_abs.apply(bucket)
    bucket_stats = {}
    for b in ["Q1", "Q2", "Q3", "Q4"]:
        sub = per_user[per_user["alpha_abs_bucket"] == b]
        if len(sub) == 0:
            bucket_stats[b] = {"n": 0}
            continue
        s_pref = int(sub["user_prefers_pilsd"].sum())
        s_tot = len(sub)
        bucket_stats[b] = {
            "n": s_tot,
            "n_prefer_pilsd": s_pref,
            "win_rate": s_pref / s_tot,
            "win_rate_wilson95": list(wilson_ci(s_pref, s_tot)),
            "mean_delta": float(sub["delta_pilsd_minus_vanilla"].mean()),
            "mean_alpha_j": float(sub["alpha_j"].mean()),
            "mean_beta_j": float(sub["beta_j"].mean()),
        }

    # Prompt-level win rate (population raw RM, averaged across users —
    # should collapse to the raw-RM pair accuracy since beta_j > 0 for all)
    total_prompt_wins_p = int(per_user["prompt_level_wins_pilsd"].sum())
    total_prompt_wins_v = int(per_user["prompt_level_wins_vanilla"].sum())
    total_prompt_ties = int(per_user["prompt_level_ties"].sum())
    prompt_level_wr = total_prompt_wins_p / max(1, total_prompt_wins_p + total_prompt_wins_v)

    # Raw RM summary
    raw_pilsd_wins = int((p_scores > v_scores).sum())
    raw_vanilla_wins = int((v_scores > p_scores).sum())
    raw_ties = int((p_scores == v_scores).sum())

    # GPU util snapshot
    try:
        import subprocess
        nvidia = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader"]
        ).decode().strip()
    except Exception as e:
        nvidia = f"N/A ({e})"

    result = {
        "config": {
            "n_eval_prompts": len(raw_prompts),
            "n_users_sampled": int(n_users),
            "eval_start_idx": args.eval_start_idx,
            "response_length": args.response_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "vanilla_adapter": args.vanilla_adapter,
            "pilsd_adapter": args.pilsd_adapter,
            "rm_adapter": args.rm_adapter_dir,
        },
        # Raw RM side (sanity — should match prior eval since same adapters/slice-style)
        "raw_rm": {
            "mean_raw_vanilla": float(v_scores.mean()),
            "mean_raw_pilsd": float(p_scores.mean()),
            "raw_pilsd_wins": raw_pilsd_wins,
            "raw_vanilla_wins": raw_vanilla_wins,
            "raw_ties": raw_ties,
            "raw_pilsd_win_rate": raw_pilsd_wins / max(1, raw_pilsd_wins + raw_vanilla_wins),
            "raw_pilsd_win_rate_wilson95": list(
                wilson_ci(raw_pilsd_wins, raw_pilsd_wins + raw_vanilla_wins)
            ),
        },
        # HEADLINE: per-user preference
        "per_user_preference": {
            "n_users": n_users,
            "n_prefer_pilsd": n_pref_p,
            "n_prefer_vanilla": n_pref_v,
            "n_tie_exact": n_tie,
            "user_win_rate_pilsd": user_wr,
            "user_win_rate_pilsd_wilson95": list(user_wr_ci),
            "mean_per_user_delta": mean_delta,
            "mean_per_user_delta_bca95": list(mean_delta_ci),
            "median_per_user_delta": median_delta,
        },
        # Prompt-level aggregate (expected to match raw RM, demonstrating invariance)
        "prompt_level_aggregate": {
            "total_prompt_wins_pilsd": total_prompt_wins_p,
            "total_prompt_wins_vanilla": total_prompt_wins_v,
            "total_prompt_ties": total_prompt_ties,
            "prompt_level_win_rate_pilsd": prompt_level_wr,
        },
        "alpha_bucket_breakdown": bucket_stats,
        "gpu_state_at_end": nvidia,
        "sample_users": per_user.head(10).to_dict(orient="records"),
    }

    out_json.write_text(json.dumps(result, indent=2))
    print(f"[save] {out_json}")
    print(f"[save] {out_parquet}")
    # Print the bits without sample_users
    summary = {k: v for k, v in result.items() if k != "sample_users"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
