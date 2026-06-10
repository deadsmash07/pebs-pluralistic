"""Score every PRISM utterance with a trained reward model → per-user calibration data.

**Pipeline insight**: PRISM users do NOT share reference responses —
each (interaction_id, turn) comparison is unique to one user, so traditional
anchor-vignette identification (designated anchors rated by all users) is not
directly available. We use an ALTERNATE identification strategy:

  The trained reward model acts as a *shared anchor*. Each utterance i gets a
  `rm_score_i` from the 7B LoRA RM, and each user j's PRISM `score_ij` is a
  noisy linear function of it:

      score_ij = α_j · rm_score_i + β_j + ε_ij

  Fitting per-user (α_j, β_j) via MixedLM partial pooling (Pinheiro & Bates
  2000) gives a per-user scale/shift → we compute
  `corrected_ij = (score_ij - β_j) / α_j` as M_corrected.

This is mathematically equivalent to the anchor-calibration story: the RM
provides a common reference scale, and per-user OLS extracts scale drift.

**Inputs**:
  - `data/prism_pairs.parquet` (26,876 curated preference pairs; curate_prism_pairs.py)
  - LoRA adapter directory `results/e2_canonical_7b/` (trained arm A)
  - PRISM utterances full table (for per-user score column) — we re-load from HF

**Output**:
  - `data/prism_rm_scored.parquet` — one row per utterance with
    (utterance_id, user_id, interaction_id, score_user, score_rm, role, ...)
  - Downstream: `fit_user_calibrators.py` fits MixedLM on this.

**Backing refs**:
  - Pinheiro & Bates 2000 Mixed-Effects Models §2
  - Baker 2001 IRT — linear anchor calibration specialization
  - Kirk et al. 2024 PRISM §4 (per-utterance score metadata)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", required=True,
                   help="Path to trained LoRA adapter (e.g., results/e2_canonical_7b)")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output-parquet", default="data/prism_rm_scored.parquet")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Only score first N utterances (for smoke).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_inference_mode(model):
    """Disable dropout/batchnorm-tracking across the model tree.

    Equivalent to the inference-mode switch that PyTorch exposes via
    `.training = False` on every submodule.
    """
    for module in model.modules():
        module.training = False


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[load] base model {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1, torch_dtype=torch.bfloat16,
    )
    base.config.pad_token_id = tok.pad_token_id

    print(f"[load] LoRA adapter {args.adapter_dir}")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    set_inference_mode(model)
    model = model.cuda()

    # Load PRISM utterances
    print("[load] PRISM utterances from HF")
    from datasets import load_dataset
    utt_ds = load_dataset("HannahRoseKirk/prism-alignment", "utterances",
                          split="train")
    df = utt_ds.to_pandas()
    # Keep only rows with non-null user_prompt + model_response
    df = df[df["user_prompt"].notna() & df["model_response"].notna()].reset_index(drop=True)
    if args.limit is not None:
        df = df.head(args.limit)
    print(f"[load] {len(df)} utterances to score")

    # Score in batches
    results = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(df), args.batch_size):
            batch_df = df.iloc[start:start + args.batch_size]
            texts = [
                f"{row.user_prompt}\n\n{row.model_response}"
                for row in batch_df.itertuples()
            ]
            enc = tok(texts, truncation=True, max_length=args.max_length,
                      padding=True, return_tensors="pt").to(device)
            scores = model(**enc).logits.squeeze(-1).float().cpu().numpy()
            for row_idx, (_, row) in enumerate(batch_df.iterrows()):
                results.append({
                    "utterance_id": row["utterance_id"],
                    "interaction_id": row["interaction_id"],
                    "conversation_id": row["conversation_id"],
                    "user_id": row["user_id"],
                    "turn": int(row["turn"]),
                    "within_turn_id": int(row.get("within_turn_id", 0)),
                    "model_name": row.get("model_name", ""),
                    "score_user": float(row["score"]) if pd.notna(row["score"]) else None,
                    "if_chosen": bool(row["if_chosen"]),
                    "rm_score": float(scores[row_idx]),
                })
            if (start // args.batch_size) % 20 == 0:
                print(f"[score] {start + len(batch_df)}/{len(df)}  "
                      f"rm_score last batch min={scores.min():.3f} "
                      f"max={scores.max():.3f}")

    out_df = pd.DataFrame(results)
    out_path = Path(args.output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path)
    print(f"[save] {out_path} ({len(out_df)} rows)")

    # Summary stats
    user_df = out_df.dropna(subset=["score_user"])
    if len(user_df) == 0:
        print("[warn] no rows with user-score — can't compute correlation")
        return
    corr = np.corrcoef(user_df["rm_score"], user_df["score_user"])[0, 1]
    print(f"\n=== RM vs user-score diagnostic ===")
    print(f"  n rows with user score: {len(user_df)}")
    print(f"  corr(rm_score, score_user) = {corr:.4f}")
    print(f"  rm_score range: [{out_df.rm_score.min():.3f}, {out_df.rm_score.max():.3f}]")
    print(f"  score_user range: [{user_df.score_user.min()}, {user_df.score_user.max()}]")
    print(f"  unique users with scores: {user_df.user_id.nunique()}")


if __name__ == "__main__":
    main()
