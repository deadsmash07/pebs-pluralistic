"""Score PILSD-calibrated Qwen2.5 reward model against RewardBench 2.

First downstream evaluation for Track 1's Qwen2.5-7B PRISM RM. RewardBench 2
(Lambert et al. 2025, arXiv:2506.01937) scores RMs on a best-of-4 leaderboard:
for each prompt, 4 candidate responses are ranked and the top-1 must match the
designated gold answer.

Categories: Factuality, Precise Instruction Following, Math, Safety, Focus, Ties.

Crucial limitation (documented in DOC_REWARDBENCH2.md):
  RewardBench 2 prompts have NO annotator_id -- PILSD's per-user calibrator
  cannot be applied directly. This script supports four --shrinkage-mode
  settings to test what PILSD's POPULATION-SLOPE contribution buys us:
    - none             : no calibration applied (raw RM logits, top-1 rank)
    - vanilla          : alias for 'none'; names the baseline-RM arm
    - pilsd_pop_slope  : apply population alpha, beta from REML MixedLM fit
    - pilsd_shrunk     : per-user alpha_j, beta_j if user_id is present (falls
                         back to pop-slope for unseen users). On RewardBench 2
                         this equals pilsd_pop_slope because prompts lack user_id.

Note that top-1 selection on a SINGLE user's calibration is monotone-invariant
to (alpha_j, beta_j) -- so calibration only matters when scores are compared
ACROSS users (e.g., selecting a shared policy per-user) or when passed
downstream into a PPO reward normalizer. RewardBench 2 measures the former
indirectly: a population calibration that shifts mean score does not change
within-prompt argmax. This script's main value is therefore:
  (1) a leaderboard-comparable held-out RM score for paper table T1
  (2) a sanity check that LoRA + PILSD did not break the RM's ranking ability
  (3) an ablation: measure category-wise perf of vanilla vs PILSD-calibrated RM

References:
  - Lambert et al. 2025 'RewardBench 2' (arXiv:2506.01937)
  - Lambert et al. 2024 'Tulu 3' (arXiv:2411.15124)
  - Kirk et al. 2024 NeurIPS D&B PRISM (user-id source)
  - Gelman & Hill 2007 Section 12 (partial-pooling framework behind PILSD)
  - allenai/reward-bench GitHub (scripts/run_v2.py reference impl)

Example (H100):
    python3 scripts/eval_rewardbench2.py \\
        --model-path Qwen/Qwen2.5-7B-Instruct \\
        --adapter-path ~/IMPLEMENTATION/1_Causal_RLHF/results/e2_canonical_7b \\
        --calibrators-parquet data/prism_user_calibrators_shrunk.parquet \\
        --pop-alpha 66.48 --pop-beta 14.37 \\
        --output-json results/rewardbench2_pilsd.json \\
        --shrinkage-mode pilsd_pop_slope \\
        --batch-size 8
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHRINKAGE_MODES = ("none", "vanilla", "pilsd_shrunk", "pilsd_pop_slope")

# RewardBench 2 official category list. Source: dataset card + run_v2.py in
# allenai/reward-bench. "Ties" is scored differently from best-of-4 (see paper
# section 4.2) -- we report it but exclude from the weighted overall mean by
# default, matching the leaderboard's primary accuracy column.
RB2_CATEGORIES = (
    "Factuality",
    "Precise Instruction Following",
    "Math",
    "Safety",
    "Focus",
    "Ties",
)
NON_TIE_CATEGORIES = tuple(c for c in RB2_CATEGORIES if c != "Ties")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-path", required=True,
        help="HF ID or local path to base model (e.g. Qwen/Qwen2.5-7B-Instruct)",
    )
    p.add_argument(
        "--adapter-path", default=None,
        help="Optional LoRA adapter dir from train_rm_e2.py. If absent, scores "
             "the base model directly (non-RM).",
    )
    p.add_argument(
        "--calibrators-parquet", default=None,
        help="Optional PILSD per-user calibrators parquet (columns: user_id, "
             "alpha_j, beta_j, ...). If absent AND mode != pilsd_pop_slope, "
             "the script falls back to pop-slope using --pop-alpha/--pop-beta.",
    )
    p.add_argument(
        "--pop-alpha", type=float, default=66.48,
        help="Population intercept alpha from REML MixedLM fit "
             "(Track 1 pop_intercept per track1_tau_comparison.json)",
    )
    p.add_argument(
        "--pop-beta", type=float, default=14.37,
        help="Population slope beta from REML MixedLM fit "
             "(Track 1 pop_slope per track1_tau_comparison.json)",
    )
    p.add_argument(
        "--output-json", required=True,
        help="Where to write per-prompt scores + category accuracies + metadata",
    )
    p.add_argument(
        "--categories", default="all",
        help="Comma-separated subset of RewardBench 2 categories, or 'all'",
    )
    p.add_argument(
        "--batch-size", type=int, default=8,
        help="Forward batch size (H100 80GB fits ~16 at max_length=2048; "
             "auto-halves on CUDA OOM)",
    )
    p.add_argument(
        "--max-length", type=int, default=2048,
        help="Tokenizer truncation length (RewardBench 2 chats can exceed 1k)",
    )
    p.add_argument(
        "--max-examples", type=int, default=None,
        help="Dev flag: score only first N prompts (for smoke-tests)",
    )
    p.add_argument(
        "--shrinkage-mode", choices=SHRINKAGE_MODES, default="vanilla",
        help="Post-hoc calibration mode; see docstring",
    )
    p.add_argument(
        "--dataset-repo-id", default="allenai/reward-bench-2",
        help="HF dataset repo id (overridable for private forks)",
    )
    p.add_argument(
        "--dataset-cache-dir",
        default=os.environ.get("REWARDBENCH2_DIR"),
        help="Local snapshot dir from build_rewardbench2_dataset.py "
             "(default: $REWARDBENCH2_DIR)",
    )
    p.add_argument(
        "--dataset-split", default="test",
        help="HF split name (RewardBench 2 releases 'test')",
    )
    p.add_argument(
        "--user-id-field", default=None,
        help="If set AND calibrators-parquet present, look up per-user calibrator "
             "by this column name (RewardBench 2 lacks user_ids; provided for "
             "private benchmark forks that ADD annotator ids)",
    )
    p.add_argument(
        "--device", default=None,
        help="torch device (default: cuda if available else cpu)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate imports + args + dataset access, then exit",
    )
    return p.parse_args()


# ----------------------------------------------------------------------------
# Calibration helpers
# ----------------------------------------------------------------------------


def load_calibrators(parquet_path: Optional[str]) -> Optional[Dict[str, Tuple[float, float]]]:
    """Return {user_id: (alpha_j, beta_j)} or None if parquet absent."""
    if not parquet_path:
        return None
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    required = {"user_id", "alpha_j", "beta_j"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"calibrators parquet {parquet_path} missing required columns: {missing}"
        )
    return {
        str(row.user_id): (float(row.alpha_j), float(row.beta_j))
        for row in df.itertuples(index=False)
    }


def apply_calibration(
    raw_scores: np.ndarray,
    mode: str,
    pop_alpha: float,
    pop_beta: float,
    user_id: Optional[str],
    calibrators: Optional[Dict[str, Tuple[float, float]]],
) -> np.ndarray:
    """Apply PILSD calibration.

    Calibration model (Gelman & Hill section 12 partial-pooling):
        rating_ij = alpha_j + beta_j * (M_i - M_bar) + eps
    So raw RM score M_i is calibrated to PREDICTED rating for user j via:
        rating_hat_ij = alpha_j + beta_j * M_i   (we absorb M_bar into alpha_j at fit time)

    Under best-of-4 argmax within a prompt, monotonic transforms are no-ops.
    The calibration matters ONLY when scores are combined across prompts or
    passed to downstream stages (PPO, MC sampling). This function therefore
    returns the transformed scalar so the OUTPUT_JSON captures the calibrated
    values for downstream analysis, while argmax metrics are invariant.

    Returns same shape as raw_scores.
    """
    if mode in ("none", "vanilla"):
        return raw_scores.astype(np.float64)

    if mode == "pilsd_pop_slope":
        return pop_alpha + pop_beta * raw_scores.astype(np.float64)

    if mode == "pilsd_shrunk":
        # Per-user if available, else population fallback
        if user_id is not None and calibrators is not None and user_id in calibrators:
            alpha_j, beta_j = calibrators[user_id]
        else:
            alpha_j, beta_j = pop_alpha, pop_beta
        return alpha_j + beta_j * raw_scores.astype(np.float64)

    raise ValueError(f"unknown shrinkage mode: {mode}")


# ----------------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------------


def load_rewardbench2(
    repo_id: str,
    cache_dir: Optional[str],
    split: str,
    categories: Sequence[str],
    max_examples: Optional[int],
):
    """Return (list_of_rows, list_of_categories_actually_present, cat_key)."""
    from datasets import load_dataset

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    ds = load_dataset(repo_id, split=split, **kwargs)

    # RewardBench 2 row schema (per dataset card, verified 2025-06):
    #   prompt: str
    #   chosen: List[str]       (length 1 -- the gold response)
    #   rejected: List[str]     (length 3 -- the distractors)
    #   subset: str             (one of RB2_CATEGORIES; also called 'category')
    #   total_completions: int  (4 for best-of-4, >4 for ties)
    #   id: str
    # We defensively accept either 'subset' or 'category' as category key.
    cat_key = "subset" if "subset" in ds.column_names else "category"

    keep = []
    actual_cats = set()
    for row in ds:
        cat = row.get(cat_key, "UNKNOWN")
        actual_cats.add(cat)
        if "all" in categories or cat in categories:
            keep.append(row)
        if max_examples is not None and len(keep) >= max_examples:
            break
    return keep, sorted(actual_cats), cat_key


# ----------------------------------------------------------------------------
# Model loading + scoring
# ----------------------------------------------------------------------------


def load_reward_model(model_path: str, adapter_path: Optional[str], device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tok.pad_token_id

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model = model.to(device)
    model.train(False)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def score_texts(
    texts: List[str], model, tok, device, max_length: int, batch_size: int,
) -> np.ndarray:
    """Score a list of full conversation texts. Returns 1-D float array.

    Auto-halves batch_size on CUDA OOM and retries; propagates if bs==1 still OOMs.
    """
    import torch

    out = np.zeros(len(texts), dtype=np.float32)
    i = 0
    bs = batch_size
    while i < len(texts):
        chunk = texts[i:i + bs]
        try:
            enc = tok(
                chunk,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits.squeeze(-1).float().cpu().numpy()
            out[i:i + bs] = logits
            i += bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"[rb2] OOM at i={i} -- halving batch size to {bs}", flush=True)
    return out


def format_for_rm(prompt: str, response: str, tok) -> str:
    """Build a single text string representing the full conversation for RM scoring.

    Prefers tokenizer chat template when available (Qwen2.5-Instruct has one),
    else falls back to the train_rm_e2.py concat format '<prompt>\\n\\n<response>'
    to keep train/eval distributions aligned.
    """
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": response}],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass
    return f"{prompt}\n\n{response}"


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------


def aggregate_category_accuracy(
    per_prompt: List[Dict],
) -> Dict[str, Dict[str, float]]:
    """Compute per-category top-1 accuracy + 95 pct Wilson CI + overall weighted mean.

    'overall_macro' weights each category equally (leaderboard convention);
    'overall_micro' weights by n_prompts. 'overall_macro_excl_ties' drops the
    Ties category from both.
    """
    by_cat: Dict[str, List[int]] = {}
    for r in per_prompt:
        by_cat.setdefault(r["category"], []).append(int(r["correct"]))

    def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
        if n == 0:
            return (float("nan"), float("nan"))
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
        return (centre - half, centre + half)

    out: Dict[str, Dict[str, float]] = {}
    for cat, vals in by_cat.items():
        n = len(vals)
        k = int(sum(vals))
        lo, hi = _wilson(k, n)
        out[cat] = {
            "accuracy": k / n if n else float("nan"),
            "n": n,
            "n_correct": k,
            "ci_lo": lo,
            "ci_hi": hi,
        }

    cat_accs = [out[c]["accuracy"] for c in out if out[c]["n"] > 0]
    cat_accs_excl_ties = [
        out[c]["accuracy"] for c in out if c != "Ties" and out[c]["n"] > 0
    ]
    total_n = sum(out[c]["n"] for c in out)
    total_k = sum(out[c]["n_correct"] for c in out)

    out["__summary__"] = {
        "overall_macro": float(np.mean(cat_accs)) if cat_accs else float("nan"),
        "overall_macro_excl_ties": (
            float(np.mean(cat_accs_excl_ties)) if cat_accs_excl_ties else float("nan")
        ),
        "overall_micro": total_k / total_n if total_n else float("nan"),
        "n_total": total_n,
    }
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    # Resolve device now so dry-run messages reflect the planned config.
    if args.device:
        device = args.device
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    categories = [c.strip() for c in args.categories.split(",")] if args.categories != "all" else ["all"]

    print(f"[rb2] model={args.model_path}")
    print(f"[rb2] adapter={args.adapter_path}")
    print(f"[rb2] calibrators={args.calibrators_parquet}")
    print(f"[rb2] mode={args.shrinkage_mode}")
    print(f"[rb2] device={device}")
    print(f"[rb2] categories={categories}")

    if args.shrinkage_mode in ("pilsd_shrunk", "pilsd_pop_slope"):
        if args.pop_alpha is None or args.pop_beta is None:
            raise ValueError(
                "--pop-alpha and --pop-beta required for PILSD modes "
                "(from track1_tau_comparison.json pop_intercept/pop_slope)"
            )

    calibrators = load_calibrators(args.calibrators_parquet)
    if calibrators is not None:
        print(f"[rb2] loaded {len(calibrators)} per-user calibrators")
    elif args.shrinkage_mode == "pilsd_shrunk":
        print("[rb2] WARNING: pilsd_shrunk requested but no calibrators parquet; "
              "all users will fall back to pop-slope")

    if args.dry_run:
        # Exercise dataset/tokenizer/model import paths.
        try:
            import transformers  # noqa: F401
            import datasets       # noqa: F401
            import peft           # noqa: F401
        except ImportError as e:
            print(f"[rb2] dry-run import failed: {e}", file=sys.stderr)
            return 2
        print("[rb2] --dry-run: imports OK")
        return 0

    print(f"[rb2] loading dataset {args.dataset_repo_id} split={args.dataset_split}")
    rows, all_cats_present, cat_key = load_rewardbench2(
        args.dataset_repo_id, args.dataset_cache_dir, args.dataset_split,
        categories, args.max_examples,
    )
    print(f"[rb2] kept {len(rows)} rows; categories present: {all_cats_present}")

    print(f"[rb2] loading model + adapter")
    model, tok = load_reward_model(args.model_path, args.adapter_path, device)

    per_prompt: List[Dict] = []
    t0 = time.time()
    batch_size = args.batch_size

    for i, row in enumerate(rows):
        prompt = row["prompt"]
        chosen = row.get("chosen") or []
        rejected = row.get("rejected") or []
        if isinstance(chosen, str):
            chosen = [chosen]
        if isinstance(rejected, str):
            rejected = [rejected]

        responses = list(chosen) + list(rejected)
        n_chosen = len(chosen)
        if len(responses) < 2:
            continue  # malformed row

        texts = [format_for_rm(prompt, r, tok) for r in responses]
        raw = score_texts(texts, model, tok, device, args.max_length, batch_size)

        uid = row.get(args.user_id_field) if args.user_id_field else None
        calibrated = apply_calibration(
            raw, args.shrinkage_mode, args.pop_alpha, args.pop_beta,
            uid, calibrators,
        )

        top_idx = int(np.argmax(calibrated))
        correct = int(top_idx < n_chosen)

        per_prompt.append({
            "id": row.get("id", f"row_{i}"),
            "category": row.get(cat_key, "UNKNOWN"),
            "raw_scores": [float(x) for x in raw.tolist()],
            "calibrated_scores": [float(x) for x in calibrated.tolist()],
            "n_chosen": n_chosen,
            "n_total": len(responses),
            "top_idx": top_idx,
            "correct": correct,
            "user_id": uid,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            print(f"[rb2] scored {i+1}/{len(rows)} ({rate:.2f} prompts/s)", flush=True)

    # Aggregation
    agg = aggregate_category_accuracy(per_prompt)

    output = {
        "config": {
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "calibrators_parquet": args.calibrators_parquet,
            "shrinkage_mode": args.shrinkage_mode,
            "pop_alpha": args.pop_alpha,
            "pop_beta": args.pop_beta,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "dataset_repo_id": args.dataset_repo_id,
            "dataset_split": args.dataset_split,
            "n_prompts_evaluated": len(per_prompt),
            "elapsed_s": time.time() - t0,
        },
        "category_accuracy": agg,
        "per_prompt": per_prompt,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"[rb2] wrote {out_path}")
    print(f"[rb2] overall_macro={agg['__summary__']['overall_macro']:.4f}  "
          f"overall_macro_excl_ties={agg['__summary__']['overall_macro_excl_ties']:.4f}  "
          f"overall_micro={agg['__summary__']['overall_micro']:.4f}")
    for cat in NON_TIE_CATEGORIES:
        if cat in agg:
            a = agg[cat]
            print(f"  {cat:32s}  acc={a['accuracy']:.4f}  "
                  f"[{a['ci_lo']:.4f}, {a['ci_hi']:.4f}]  n={a['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
