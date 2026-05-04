"""Modal A10 smoke-test — Track 1 TRL RewardTrainer end-to-end.

Run: `modal run scripts/modal_smoke_test_reward_model.py`

Validates the reward-model training path with a real transformer (Qwen2.5-0.5B,
fits comfortably on A10 24GB) on synthetic PRISM-like preference pairs. Smoke
test only — goal is to confirm:
  (a) TRL v0.26+ RewardTrainer actually runs with our pipeline
  (b) compute_metrics receives annotator_id through extra_cols (per Agent A survey)
  (c) Checkpoint round-trips cleanly via save_pretrained/from_pretrained

Smoke-test scale: 128 pairs, 8 annotators, 50 training steps, ~2 minutes on A10.
NOT a real training run — that lives on Lambda with Qwen2.5-7B + PRISM 8k pairs.

Per feedback_sota_implementations.md: we reuse TRL/transformers rather than
hand-rolling. Per adversarial-theorem-review/SKILL.md §4.5: we verify the stack
end-to-end before burning GPU-hours on Lambda.
"""
from __future__ import annotations

import modal

app = modal.App("pilsd-rm-smoke")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.5.0",
        "transformers>=4.47",
        "trl>=0.26",
        "accelerate>=1.0",
        "datasets>=3.0",
        "numpy",
        "scipy",
    )
    .env({"HF_HOME": "/hf-cache", "TRANSFORMERS_OFFLINE": "0"})
)

vol = modal.Volume.from_name("pilsd-hf-cache", create_if_missing=True)


@app.function(
    gpu="A10",
    image=image,
    volumes={"/hf-cache": vol},
    timeout=600,
)
def smoke_test():
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import RewardTrainer, RewardConfig

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    n_pairs = 128
    n_annotators = 8

    print(f"[setup] loading tokenizer + model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir="/hf-cache")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, cache_dir="/hf-cache",
        torch_dtype=torch.bfloat16,
    )

    print(f"[setup] synthesizing {n_pairs} preference pairs from {n_annotators} annotators")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_pairs):
        ann_id = int(rng.integers(0, n_annotators))
        chosen = f"Response A #{i}: this is a quality answer to the prompt."
        rejected = f"Response B #{i}: low quality, off-topic response."
        # Encode via tokenizer
        c = tok(chosen, truncation=True, max_length=64)
        r = tok(rejected, truncation=True, max_length=64)
        rows.append({
            "input_ids_chosen": c["input_ids"],
            "attention_mask_chosen": c["attention_mask"],
            "input_ids_rejected": r["input_ids"],
            "attention_mask_rejected": r["attention_mask"],
            "annotator_id": ann_id,
        })
    ds = Dataset.from_list(rows)

    print("[setup] configuring RewardTrainer (50 steps)")
    config = RewardConfig(
        output_dir="/tmp/reward-model-smoke",
        num_train_epochs=1,
        max_steps=50,
        per_device_train_batch_size=4,
        learning_rate=1e-5,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,  # keep annotator_id for compute_metrics
        bf16=True,
    )

    # annotator_id tracking: use a TrainerCallback that intercepts each batch
    # and records which annotator_ids flowed through. This is the REAL test
    # of whether the extra column survives — a compute_metrics stub alone
    # proves nothing (it only runs at eval time, not on training batches).
    from transformers import TrainerCallback
    annotator_counts: dict[int, int] = {}

    class AnnotatorIdProbe(TrainerCallback):
        def on_step_begin(self, args, state, control, **kw):
            # no-op; the actual probe is inside the collator below
            pass

    # Monkey-wrap the trainer's data collator so we can see the batch keys
    # before they're stripped for the forward pass.
    def probing_collator(features):
        # Record annotator_ids in this batch (proves they survived HF Datasets)
        for f in features:
            if "annotator_id" in f:
                annotator_counts[int(f["annotator_id"])] = annotator_counts.get(int(f["annotator_id"]), 0) + 1
        # Call TRL's default collator to produce the actual model inputs
        from trl.trainer.utils import RewardDataCollatorWithPadding
        default = RewardDataCollatorWithPadding(tokenizer=tok)
        return default(features)

    trainer = RewardTrainer(
        model=model,
        args=config,
        train_dataset=ds,
        processing_class=tok,
        data_collator=probing_collator,
    )

    print("[train] stepping for 50 updates")
    trainer.train()

    print("[checkpoint] saving + reloading")
    trainer.save_model("/tmp/reward-model-smoke/final")
    model_reloaded = AutoModelForSequenceClassification.from_pretrained(
        "/tmp/reward-model-smoke/final", num_labels=1,
    )
    with torch.no_grad():
        inputs = tok("test prompt", return_tensors="pt")
        score = model_reloaded(**inputs).logits.item()
    print(f"[checkpoint] loaded model scored 'test prompt' = {score:.4f}")

    # Verify annotator_id survived the pipeline (the actual extra_cols test)
    print(f"[annotator-id-check] observed annotator_ids in training batches: "
          f"{sorted(annotator_counts.keys())}")
    if len(annotator_counts) < n_annotators // 2:
        raise RuntimeError(
            f"annotator_id was stripped: saw only {len(annotator_counts)} of "
            f"{n_annotators} expected annotators. Set remove_unused_columns=False."
        )

    print("[PASS] smoke test completed — TRL RewardTrainer pipeline validated on A10")
    return {
        "pairs": n_pairs,
        "annotators_seen": len(annotator_counts),
        "final_score": score,
    }


@app.local_entrypoint()
def main():
    result = smoke_test.remote()
    print(f"\n[local] smoke-test result: {result}")
    print("\n✅ Ready for Lambda scale-up with Qwen2.5-7B + PRISM 8k pairs")
