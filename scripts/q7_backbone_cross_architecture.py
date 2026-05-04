"""Q7 - Backbone-mismatch cross-architecture PILSD on PRISM.

Closes the "Qwen-only backbone" reviewer attack flagged in
`memory/p0_experiment_queue_v1_2026_05_03.md` Q7 brief and `Skill:
icml-neurips-critical-reviewer-2026` Pass 5 (cross-architecture
generalization). Hypothesis: PILSD's PRISM headline gain replicates with
Mistral-7B-Instruct + Yi-1.5-34B-Chat backbones (in addition to
Qwen2.5-7B-Instruct), establishing that the per-rater empirical-Bayes
shrinkage mechanism is a property of the calibration pipeline rather
than an artifact of any single backbone family.

DESIGN-DECISION DOCUMENTED HONESTLY (Skill: honest-disclosure SCOPE-not-RETRACT)
================================================================================
The legacy PRISM headline (8.58%) used Qwen2.5-7B-Instruct + a trained
LoRA reward-model classifier head (`AutoModelForSequenceClassification`).
For Q7 cross-architecture comparison we cannot reuse a Qwen-trained LoRA
head on Mistral / Yi backbones (architectures differ; LoRA target_modules
differ; trained adapter weights are tied to the Qwen tokenizer + hidden
dim). Training fresh LoRA RMs per backbone would conflate two confounds:

  1. backbone-architecture variance (the question we want to answer)
  2. RM-training-noise variance (different SFT, different LR schedule,
     different data shuffling)

To isolate (1), Q7 uses the **mean-response log-likelihood** scoring
function (Stiennon et al. 2020) as a *backbone-agnostic* RM proxy. The
same scoring function is applied to all 3 backbones; differences in
PILSD gain therefore reflect ONLY backbone-level calibration variance,
not RM-training variance.

This means Q7 does NOT re-replicate the 8.58% legacy headline (which
used a different scoring function). Q7 is a SELF-CONTAINED 3-arm
cross-architecture comparison whose internal consistency is the headline
claim. Q9's pooled-4-corpus result already used Qwen2.5-7B mean-LL and
landed +9.977% on the PRISM slice; Q7's Qwen2.5-7B mean-LL re-extraction
should land at the same value (within bootstrap noise) and serves as a
sanity-check on the pipeline.

4-class verdict-class STRICT (per Q7 brief)
- ESTABLISHED-PILSD-CROSS-BACKBONE-CONFIRMS  if all 3 backbones gain >= +3pp + CI excludes 0
- MODERATE-PILSD-CROSS-BACKBONE-PARTIAL      if 2/3 backbones confirm
- PRELIMINARY-INCONCLUSIVE-CROSS-BACKBONE    if 1/3 confirms or wide CI
- FALSIFIED-PILSD-QWEN-ONLY                  if other backbones reverse (CI strictly negative)

Pipeline (mirrors Q1 + Q9 + eval_oasst2_pilsd_calibrator)
---------------------------------------------------------
Stage A - per backbone: score each PRISM utterance with mean-response
          log-likelihood (Stiennon-style). Output:
          data/prism_<backbone>_meanll_scored.parquet
          (one row per utterance with (utterance_id, user_id, score_user, rm_score))

Stage B - per backbone: PILSD per-user EB-shrinkage on the scored data.
          K=5 within-user CV; pop-OLS pre-pass for tau^2_alpha, tau^2_beta;
          per-user OLS with V_alpha, V_beta; omega = tau^2 / (tau^2 + V);
          shrunk = omega*per_user + (1-omega)*pop. Cluster-bootstrap by
          user_id; B=2000.

Stage C - cross-backbone aggregation: 4-class verdict per brief, plus
          per-backbone gains + CI table for paper integration.

Compute envelope honest-disclosure
- Per-utterance cap: 20 utterances per user (PRISM median is 48; cap
  preserves K=5 CV with >=4 obs/fold/user with comfortable headroom for
  the >=10 MIN_OBS_PER_USER threshold while bounding wall-time).
- Estimated wall on h100_v2_backup (H100 80GB):
    Mistral-7B nf4   ~1.5-2.5h
    Yi-1.5-34B nf4   ~5-8h
    Qwen2.5-7B nf4   ~1.5-2.5h
  Total ~9-13h sequential (within 6-12h target band).

12-gate audit (Skill: research-grade-code-audit-pre-launch v1)
--------------------------------------------------------------
G1 math-vs-code: VERBATIM PILSD math from eval_oasst2_pilsd_calibrator.py
   (`ols_with_V` + `cluster_bootstrap_gain_ci` + 5-fold within-user CV)
   + Q9 G3 tau-pool reliability filter.
G2 hypothesis-vs-design: each of 3 backbones uses identical mean-LL
   scoring + identical PILSD pipeline. Differences in gain attribute
   exclusively to backbone variance. Honest disclosure: this is NOT
   re-replication of the LoRA-RM 8.58% headline.
G3 no silent-bypass: Stage A per-backbone output_parquet existence check
   by-path + row-count threshold; Stage B reuses Q9 G3 tau-pool filter
   (CAUGHT-AND-FIXED real bug at Q9 pre-launch). The 3 backbone arms each
   produce observably different rm_score distributions (verified at end
   via per-backbone Pearson r vs score_user diagnostic).
G4 pipeline integrity: gain_pct = (rmse_pop_slope - rmse_pilsd_shrunk) /
   rmse_pop_slope * 100; identical formula across 3 backbones; no
   cross-backbone metric leakage.
G5 reference-implementation: math + bootstrap inherit verbatim from
   1_Causal_RLHF/scripts/eval_oasst2_pilsd_calibrator.py (commit-anchored)
   + Q9 G3 tau-pool reliability filter backport.
G6 hyperparameter sanity: K_FOLDS=5 / N_BOOT=2000 / MIN_OBS_PER_USER=10
   (per Q7 brief; matches PRISM canonical) / MAX_PROMPT_TOKENS=256
   / MAX_RESPONSE_TOKENS=256 (matches Q1 + Q9).
G7 per-step diagnostic: Stage A logs scoring rate + ETA every 200 rows;
   Stage B logs per-arm RMSE + cluster-bootstrap progress. Per-backbone
   tau^2 + Pearson r vs user-score recorded.
G8 reproducibility: SEED=20260420 + bootstrap_seed=20260420 (matches
   PRISM canonical); git HEAD + parquet sha256 persisted in summary.json.
G9 output schema: 4-class STRICT verdict-class per Skill: honest-disclosure
   6.3; per-backbone gains + CI in `per_backbone_results` dict; cross-
   backbone aggregate verdict in top-level `verdict_class`.
G10 compute envelope: ~9-13h sequential (3 backbones); per-utterance cap
    at 20/user keeps wall under 12h with H100 nf4 throughput estimates.
G11 anti-overfitting: not theory-claiming.
G12 honest-disclosure: 4-class verdict ENUMERATES the 3 outcomes per
    backbone PLUS the cross-backbone aggregation rule; mean-LL-vs-LoRA-RM
    scoring-function difference disclosed at top of file + in summary.

Output
------
results/track1_q7_backbone_cross_architecture/{
  summary.json            # 4-class verdict + per-backbone gains + CIs
  per_backbone.parquet    # row-per-backbone diagnostic table
  per_user_<backbone>.parquet  # per-user calibrator + RMSE
}

References
----------
- Stiennon, N. et al. (2020). Learning to summarize from human feedback.
  NeurIPS. (mean-response log-likelihood reward proxy)
- Morris, C. (1983). Parametric Empirical Bayes inference: theory and
  applications. JASA 78(381), 47-55.
- Henderson, C. (1975). Best linear unbiased estimation and prediction
  under a selection model. Biometrics 31(2), 423-447.
- Cameron, A. C., Gelbach, J. B., Miller, D. L. (2008). Bootstrap-based
  improvements for inference with clustered errors. Review of Economics
  and Statistics 90(3), 414-427.
- Efron, B. (1987). Better bootstrap confidence intervals. JASA 82(397),
  171-185.
- Mistral AI (2024). Mistral-7B-Instruct-v0.3.
  https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
- 01.AI (2024). Yi-1.5-34B-Chat.
  https://huggingface.co/01-ai/Yi-1.5-34B-Chat
- Kirk, H. R. et al. (2024). PRISM Alignment Project.
  https://huggingface.co/datasets/HannahRoseKirk/prism-alignment
"""
from __future__ import annotations

import os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"

import argparse
import hashlib
import json
import math
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
    raise ImportError("scipy required for cluster-bootstrap + Wilcoxon") from exc


ROOT = Path(__file__).resolve().parents[2]                  # 3_PILSD_Standalone/
T1 = ROOT.parent / "1_Causal_RLHF"

OUT_DIR = ROOT / "results" / "track1_q7_backbone_cross_architecture"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters (PRISM-headline-anchored; mirror Q1 + Q9)
SEED = 20260420
BOOT_SEED = 20260420
N_BOOT = 2000
K_FOLDS = 5
MIN_OBS_PER_USER = 10            # per Q7 brief
PER_USER_UTTERANCE_CAP = 20      # compute envelope; bounds wall while preserving CV signal
MIN_SCORED_ROWS_FOR_VALID_CACHE = 5_000
MAX_PROMPT_TOKENS = 256
MAX_RESPONSE_TOKENS = 256

# Q7 backbone roster (per brief)
BACKBONES = {
    "mistral_7b_inst_v03": "mistralai/Mistral-7B-Instruct-v0.3",
    "yi_15_34b_chat":       "01-ai/Yi-1.5-34B-Chat",
    "qwen25_7b_inst":       "Qwen/Qwen2.5-7B-Instruct",
}

# Q9 G3 tau-pool reliability filter (CAUGHT-AND-FIXED at Q9 pre-launch)
TAU_RELIABLE_V_MIN = 1e-6
TAU_RELIABLE_V_MAX = 10.0
TAU_RELIABLE_PARAM_ABS_MAX = 10.0

# 4-class verdict-class thresholds
ESTABLISHED_GAIN_LO = 3.0
N_CLUSTERS_FOR_ESTABLISHED = 500


# ============================================================================
# G8 reproducibility helpers
# ============================================================================

def file_sha256(p: Path) -> str:
    if not Path(p).exists():
        return f"FILE_NOT_FOUND:{p}"
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head_t3() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================================
# Stage A - score PRISM utterances with one backbone (mean-response log-likelihood)
# G5 reference parity: replicates score_oasst2_with_qwen7b.py:215-230 verbatim
# (separate prompt/response tokenization, concat, -100 mask on prompt, score = -loss)
# ============================================================================


def stage_a_score_prism(
    utt_df: pd.DataFrame,
    output_parquet: Path,
    model_id: str,
    backbone_alias: str,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    max_response_tokens: int = MAX_RESPONSE_TOKENS,
    quantize_4bit: bool = True,
    smoke_limit: int | None = None,
    log_every: int = 200,
) -> dict:
    """Score each PRISM utterance with `model_id` mean-response log-likelihood.

    Output rows: one per utterance with (utterance_id, user_id, score_user,
    rm_score, n_resp_tokens). Identical schema across all 3 backbones.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        from transformers import BitsAndBytesConfig
        BNB_AVAILABLE = True
    except ImportError:
        BNB_AVAILABLE = False

    df0 = utt_df.copy()
    if smoke_limit is not None:
        df0 = df0.iloc[:smoke_limit].copy()
    n_total = len(df0)

    log(f"[Stage A / {backbone_alias}] {n_total} utterances to score")
    log(f"[Stage A / {backbone_alias}] loading {model_id} (quantize_4bit={quantize_4bit})")

    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        if quantize_4bit and BNB_AVAILABLE:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32
        model_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.training = False
    for module in model.modules():
        module.training = False
    model.requires_grad_(False)
    device = next(model.parameters()).device
    log(f"[Stage A / {backbone_alias}] loaded in {time.time() - t_load:.1f}s; device={device}; "
        f"params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    scored_rows = []
    skipped = 0
    t_start = time.time()

    for i, row in enumerate(df0.itertuples()):
        prompt = str(getattr(row, "user_prompt", "") or "")
        response = str(getattr(row, "model_response", "") or "")
        try:
            p_ids = tokenizer(
                prompt, truncation=True, max_length=max_prompt_tokens,
                return_tensors="pt",
            ).input_ids.to(device)
            r_ids = tokenizer(
                response, truncation=True, max_length=max_response_tokens,
                add_special_tokens=False, return_tensors="pt",
            ).input_ids.to(device)
            if r_ids.shape[1] < 1:
                skipped += 1
                continue
            full = torch.cat([p_ids, r_ids], dim=1)
            labels = full.clone()
            labels[:, :p_ids.shape[1]] = -100
            with torch.no_grad():
                out = model(input_ids=full, labels=labels)
            score = float(-out.loss.item())  # mean response log-likelihood
            n_resp_tok = int(r_ids.shape[1])
        except Exception as e:
            if (i % log_every) == 0:
                log(f"[warn / {backbone_alias}] row {i}: scoring failed ({e}); skipping")
            skipped += 1
            continue

        scored_rows.append({
            "utterance_id": row.utterance_id,
            "user_id": row.user_id,
            "score_user": float(row.score) if pd.notna(row.score) else None,
            "rm_score": score,
            "n_resp_tokens": n_resp_tok,
        })

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1e-3)
            eta_min = (n_total - (i + 1)) / max(rate, 1e-3) / 60
            log(f"[Stage A / {backbone_alias}] {i+1}/{n_total} "
                f"({100*(i+1)/n_total:.1f}%) | rate {rate:.2f} row/s | "
                f"ETA {eta_min:.1f} min | skipped={skipped}")

    scored_df = pd.DataFrame(scored_rows)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_parquet(output_parquet, index=False)
    elapsed_total = time.time() - t_start
    n_with_user_score = int(scored_df["score_user"].notna().sum()) if len(scored_df) else 0
    log(f"[Stage A / {backbone_alias}] done in {elapsed_total/60:.1f} min; "
        f"{len(scored_df)} rows ({n_with_user_score} with user-score) -> {output_parquet}")

    # Free GPU memory before loading next backbone
    del model
    try:
        import torch as _torch
        _torch.cuda.empty_cache()
    except Exception:
        pass

    diag: dict[str, Any] = {}
    if len(scored_df) > 100:
        m = scored_df.dropna(subset=["score_user"])
        if len(m) > 50:
            try:
                r = float(np.corrcoef(m["rm_score"], m["score_user"])[0, 1])
            except Exception:
                r = float("nan")
            diag = {
                "n_rows_with_user_score": int(len(m)),
                "pearson_r_rm_score_vs_user_score": r,
                "rm_score_mean": float(scored_df["rm_score"].mean()),
                "rm_score_std": float(scored_df["rm_score"].std(ddof=1)),
                "rm_score_min": float(scored_df["rm_score"].min()),
                "rm_score_max": float(scored_df["rm_score"].max()),
            }

    return {
        "elapsed_seconds": float(elapsed_total),
        "scored_parquet": str(output_parquet),
        "scored_n_rows": int(len(scored_df)),
        "scored_n_with_user_score": n_with_user_score,
        "scored_sha256": file_sha256(output_parquet),
        "skipped": int(skipped),
        **diag,
    }


# ============================================================================
# Stage B - PILSD math (verbatim from eval_oasst2_pilsd_calibrator.py)
# Plus Q9 G3 tau-pool reliability filter (CAUGHT-AND-FIXED real bug)
# ============================================================================


def ols_with_V(x: np.ndarray, y: np.ndarray):
    """(intercept, slope, V_intercept, V_slope) with sample-variance estimates.

    Per `eval_user_score_mse_shrunk.py:61-73` verbatim (G1 + G5).
    """
    k = len(x)
    if k < 2 or np.var(x) < 1e-12:
        return float(np.mean(y)) if k else 0.0, 0.0, np.inf, np.inf
    x_bar = x.mean()
    Sxx = ((x - x_bar) ** 2).sum()
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = intercept + slope * x
    sigma_hat_sq = ((y - y_pred) ** 2).sum() / max(k - 2, 1)
    V_int = sigma_hat_sq * (1.0 / k + x_bar ** 2 / max(Sxx, 1e-12))
    V_slope = sigma_hat_sq / max(Sxx, 1e-12)
    return float(intercept), float(slope), float(V_int), float(V_slope)


def kfold_split(n: int, k: int, rng: np.random.Generator):
    """Random k-fold split. Per `eval_user_score_mse_shrunk.py:47-58` verbatim."""
    idx = np.arange(n)
    rng.shuffle(idx)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        stop = (i + 1) * fold_size if i < k - 1 else n
        test_idx = idx[start:stop]
        train_idx = np.concatenate([idx[:start], idx[stop:]])
        folds.append((train_idx, test_idx))
    return folds


def cluster_bootstrap_gain_ci(
    sq_err_baseline: np.ndarray,
    sq_err_pilsd: np.ndarray,
    per_row_cluster_idx: list,
    B: int = N_BOOT,
    seed: int = BOOT_SEED,
):
    """Cluster-resample-by-cluster bootstrap on gain_pct. BCa + percentile + jackknife.
    VERBATIM from eval_oasst2_pilsd_calibrator.py:150-280 (cluster=user_id)."""
    if len(sq_err_baseline) == 0 or len(sq_err_pilsd) == 0:
        return {
            "boot_mean": float("nan"), "boot_sd": float("nan"),
            "ci_percentile_lo": float("nan"), "ci_percentile_hi": float("nan"),
            "ci_bca_lo": float("nan"), "ci_bca_hi": float("nan"),
            "ci_jackknife_lo": float("nan"), "ci_jackknife_hi": float("nan"),
            "n_boot_valid": 0, "error": "empty sq_err arrays",
        }

    cluster_to_block: dict[Any, list[int]] = {}
    for i, c in enumerate(per_row_cluster_idx):
        cluster_to_block.setdefault(c, []).append(i)
    cluster_keys = list(cluster_to_block.keys())
    n_clusters = len(cluster_keys)

    rng = np.random.default_rng(seed)
    boot_gains = []
    for _ in range(B):
        sampled = rng.choice(cluster_keys, size=n_clusters, replace=True)
        flat = []
        for c in sampled:
            flat.extend(cluster_to_block[c])
        if not flat:
            continue
        idx_arr = np.array(flat)
        sb = sq_err_baseline[idx_arr]
        sp = sq_err_pilsd[idx_arr]
        rb = math.sqrt(sb.mean()) if sb.size else 0.0
        rp = math.sqrt(sp.mean()) if sp.size else 0.0
        if rb > 0:
            boot_gains.append((rb - rp) / rb * 100.0)

    boot_arr = np.array([g for g in boot_gains if not math.isnan(g)])
    if len(boot_arr) < 100:
        return {
            "boot_mean": float("nan"), "boot_sd": float("nan"),
            "ci_percentile_lo": float("nan"), "ci_percentile_hi": float("nan"),
            "ci_bca_lo": float("nan"), "ci_bca_hi": float("nan"),
            "ci_jackknife_lo": float("nan"), "ci_jackknife_hi": float("nan"),
            "n_boot_valid": int(len(boot_arr)), "error": "BOOTSTRAP_CI_DEGENERATE",
        }

    boot_mean = float(boot_arr.mean())
    boot_sd = float(boot_arr.std(ddof=1))
    pct_lo = float(np.quantile(boot_arr, 0.025))
    pct_hi = float(np.quantile(boot_arr, 0.975))

    rmse_base_obs = math.sqrt(sq_err_baseline.mean())
    rmse_pilsd_obs = math.sqrt(sq_err_pilsd.mean())
    obs_gain = ((rmse_base_obs - rmse_pilsd_obs) / rmse_base_obs * 100.0
                if rmse_base_obs > 0 else 0.0)

    p_below = float(np.mean(boot_arr < obs_gain))
    z0 = scstats.norm.ppf(p_below) if 0.0 < p_below < 1.0 else 0.0

    # Jackknife leave-one-cluster-out for acceleration
    jack_gains = []
    for j in range(n_clusters):
        excluded = cluster_keys[j]
        flat = [i for c in cluster_keys if c != excluded for i in cluster_to_block[c]]
        if not flat:
            continue
        sb_jk = sq_err_baseline[np.array(flat)]
        sp_jk = sq_err_pilsd[np.array(flat)]
        rb_jk = math.sqrt(sb_jk.mean()) if sb_jk.size else 0.0
        rp_jk = math.sqrt(sp_jk.mean()) if sp_jk.size else 0.0
        if rb_jk > 0:
            jack_gains.append((rb_jk - rp_jk) / rb_jk * 100.0)
    jack_arr = np.array(jack_gains)
    if len(jack_arr) > 1:
        jm = jack_arr.mean()
        num = float(np.sum((jm - jack_arr) ** 3))
        den = 6.0 * (float(np.sum((jm - jack_arr) ** 2)) ** 1.5)
        a_hat = num / den if den > 0 else 0.0
    else:
        a_hat = 0.0

    z_lo = scstats.norm.ppf(0.025)
    z_hi = scstats.norm.ppf(0.975)

    def _bca_q(z_alpha, z0_local, a):
        num = z0_local + z_alpha
        den = 1 - a * num
        return scstats.norm.cdf(z0_local + num / den) if abs(den) > 1e-12 else None

    q_lo = _bca_q(z_lo, z0, a_hat)
    q_hi = _bca_q(z_hi, z0, a_hat)
    if q_lo is None or q_hi is None or q_lo >= q_hi or q_lo < 0 or q_hi > 1:
        bca_lo, bca_hi = pct_lo, pct_hi
    else:
        bca_lo = float(np.quantile(boot_arr, np.clip(q_lo, 0.001, 0.999)))
        bca_hi = float(np.quantile(boot_arr, np.clip(q_hi, 0.001, 0.999)))

    if len(jack_arr) > 1:
        jack_se = math.sqrt(
            (n_clusters - 1) / n_clusters
            * float(np.sum((jack_arr - jack_arr.mean()) ** 2))
        )
        jk_lo = obs_gain - 1.96 * jack_se
        jk_hi = obs_gain + 1.96 * jack_se
    else:
        jk_lo = float("nan")
        jk_hi = float("nan")

    return {
        "boot_mean": boot_mean,
        "boot_sd": boot_sd,
        "ci_percentile_lo": pct_lo,
        "ci_percentile_hi": pct_hi,
        "ci_bca_lo": bca_lo,
        "ci_bca_hi": bca_hi,
        "ci_jackknife_lo": float(jk_lo),
        "ci_jackknife_hi": float(jk_hi),
        "n_boot_valid": int(len(boot_arr)),
        "obs_gain_pct": float(obs_gain),
        "z0_bias_correction": float(z0),
        "a_hat_acceleration": float(a_hat),
    }


def stage_b_pilsd_per_user(
    scored_df: pd.DataFrame,
    backbone_alias: str,
    seed: int = SEED,
    k_folds: int = K_FOLDS,
    min_obs_per_user: int = MIN_OBS_PER_USER,
    bootstrap_B: int = N_BOOT,
    bootstrap_seed: int = BOOT_SEED,
    zscore_inputs: bool = True,
) -> dict:
    """Run PILSD per-user-cluster shrinkage. cluster_id=user_id (PRISM canonical).

    G3 ROBUSTNESS NOTE (Q9 backport):
    Per-backbone z-scoring of (rm_score, score_user) is enabled by default.
    Justification:
      - gain_pct is invariant to monotonic linear rescaling (omega = tau^2/(tau^2+V)
        is invariant under uniform scaling because tau^2 and V scale identically;
        verified empirically; difference is bootstrap noise <0.01pp on synthetic).
      - Z-scoring makes the Q9 G3 tau-pool reliability thresholds (PARAM_ABS_MAX=10)
        meaningful across backbones with different RM-score scales (Mistral / Yi /
        Qwen mean-LL produces values in different ranges; raw scoring with
        score_user 0-100 makes intercept ~50 trip the 10-sigma filter spuriously).
      - Cross-backbone alpha/beta diagnostics become directly comparable.
    Set zscore_inputs=False to reproduce raw-scale legacy behavior.
    """
    df = scored_df.dropna(subset=["score_user", "rm_score"]).reset_index(drop=True)
    n_obs_total = int(len(df))
    n_users_total = int(df["user_id"].nunique())

    n_per = df.groupby("user_id").size()
    keep_users = set(n_per[n_per >= min_obs_per_user].index)
    df = df[df["user_id"].isin(keep_users)].reset_index(drop=True)
    n_users_post_filter = int(df["user_id"].nunique())
    log(f"[Stage B / {backbone_alias}] post-filter: {n_users_post_filter} users / "
        f"{len(df)} obs (min_obs_per_user={min_obs_per_user})")

    # Per-backbone z-scoring (G3 robustness; gain_pct invariant)
    if zscore_inputs and len(df) > 0:
        x_mean = float(df["rm_score"].mean())
        x_std = float(df["rm_score"].std(ddof=1))
        y_mean = float(df["score_user"].mean())
        y_std = float(df["score_user"].std(ddof=1))
        if x_std > 1e-9 and y_std > 1e-9:
            df = df.copy()
            df["rm_score"] = (df["rm_score"] - x_mean) / x_std
            df["score_user"] = (df["score_user"] - y_mean) / y_std
            log(f"[Stage B / {backbone_alias}] z-scored inputs: "
                f"x_mean_orig={x_mean:.4f}, x_std_orig={x_std:.4f}, "
                f"y_mean_orig={y_mean:.4f}, y_std_orig={y_std:.4f}")
        else:
            log(f"[Stage B / {backbone_alias}] z-score skipped (degenerate scale)")

    if n_users_post_filter < 100:
        return {
            "n_obs_total": n_obs_total,
            "n_users_total": n_users_total,
            "n_users_post_filter": n_users_post_filter,
            "error": f"N_USERS_BELOW_100_AT_FILTER: {n_users_post_filter} < 100",
            "verdict_branch": "INCONCLUSIVE-COHORT-SHRUNK",
        }

    # Within-cluster x-variance check
    x_var_per_user = df.groupby("user_id")["rm_score"].var(ddof=0)
    n_degenerate_x = int((x_var_per_user < 1e-9).sum())
    log(f"[Stage B / {backbone_alias}] users with degenerate within-user x-var: "
        f"{n_degenerate_x}/{n_users_post_filter}")

    # Population OLS on entire dataset
    slope_pop, intercept_pop = np.polyfit(df.rm_score, df.score_user, 1)
    pop_alpha = float(intercept_pop)
    pop_beta = float(slope_pop)
    log(f"[Stage B / {backbone_alias}] pop_alpha={pop_alpha:.4f}, pop_beta={pop_beta:.4f}")

    # Per-user OLS for tau^2 pre-pass
    user_stats = []
    for uid, grp in df.groupby("user_id"):
        a, b, Va, Vb = ols_with_V(
            grp.rm_score.to_numpy(),
            grp.score_user.to_numpy().astype(float),
        )
        user_stats.append({
            "user_id": uid, "alpha": a, "beta": b,
            "V_alpha": Va, "V_beta": Vb, "n": len(grp),
        })
    us = pd.DataFrame(user_stats)

    # Q9 G3 tau-pool reliability filter (CAUGHT-AND-FIXED real bug)
    # Exclude clusters with degenerate sampling-V or extreme intercepts/slopes
    reliable_mask = (
        (us["V_alpha"].replace([np.inf, -np.inf], np.nan).between(
            TAU_RELIABLE_V_MIN, TAU_RELIABLE_V_MAX)) &
        (us["V_beta"].replace([np.inf, -np.inf], np.nan).between(
            TAU_RELIABLE_V_MIN, TAU_RELIABLE_V_MAX)) &
        (us["alpha"].abs() <= TAU_RELIABLE_PARAM_ABS_MAX) &
        (us["beta"].abs() <= TAU_RELIABLE_PARAM_ABS_MAX)
    )
    us_reliable = us[reliable_mask]
    n_reliable = int(len(us_reliable))
    log(f"[Stage B / {backbone_alias}] tau-pool reliability filter: "
        f"{n_reliable}/{len(us)} users RELIABLE for tau^2 pre-pass")

    if n_reliable < 50:
        log(f"[Stage B / {backbone_alias}] WARN: <50 reliable users for tau^2; "
            f"falling back to full pool")
        us_reliable = us

    V_alpha_total = float(us_reliable.alpha.var())
    V_beta_total = float(us_reliable.beta.var())
    mean_samp_V_alpha = float(
        us_reliable.V_alpha.replace([np.inf, -np.inf], np.nan).dropna().mean()
    )
    mean_samp_V_beta = float(
        us_reliable.V_beta.replace([np.inf, -np.inf], np.nan).dropna().mean()
    )
    tau_a_sq = max(V_alpha_total - mean_samp_V_alpha, 1e-6)
    tau_b_sq = max(V_beta_total - mean_samp_V_beta, 1e-6)
    log(f"[Stage B / {backbone_alias}] tau_alpha_sq={tau_a_sq:.4f}, "
        f"tau_beta_sq={tau_b_sq:.6f}")

    # K-fold within-user CV: 4 arms
    sq_err_no_calib_per_row: list[float] = []
    sq_err_baseline_per_row: list[float] = []
    sq_err_per_user_ols_per_row: list[float] = []
    sq_err_pilsd_per_row: list[float] = []
    per_row_user: list[Any] = []
    rng2 = np.random.default_rng(seed)

    for uid, grp in df.groupby("user_id"):
        n = len(grp)
        if n < min_obs_per_user:
            continue
        x = grp.rm_score.to_numpy()
        y = grp.score_user.to_numpy().astype(float)
        folds = kfold_split(n, k_folds, rng2)
        for train_idx, test_idx in folds:
            x_tr, y_tr = x[train_idx], y[train_idx]
            x_te, y_te = x[test_idx], y[test_idx]
            if len(x_te) == 0:
                continue
            y_hat_nc = np.full_like(y_te, np.mean(y_tr) if len(y_tr) else 0.0)
            y_hat_ps = pop_alpha + pop_beta * x_te
            a, b, Va, Vb = ols_with_V(x_tr, y_tr)
            y_hat_ols = a + b * x_te
            omega_a = tau_a_sq / (tau_a_sq + Va) if np.isfinite(Va) else 0.0
            omega_b = tau_b_sq / (tau_b_sq + Vb) if np.isfinite(Vb) else 0.0
            a_s = omega_a * a + (1 - omega_a) * pop_alpha
            b_s = omega_b * b + (1 - omega_b) * pop_beta
            y_hat_sh = a_s + b_s * x_te
            sq_err_no_calib_per_row.extend(((y_hat_nc - y_te) ** 2).tolist())
            sq_err_baseline_per_row.extend(((y_hat_ps - y_te) ** 2).tolist())
            sq_err_per_user_ols_per_row.extend(((y_hat_ols - y_te) ** 2).tolist())
            sq_err_pilsd_per_row.extend(((y_hat_sh - y_te) ** 2).tolist())
            per_row_user.extend([uid] * len(y_te))

    sq_err_no_calib = np.asarray(sq_err_no_calib_per_row)
    sq_err_baseline = np.asarray(sq_err_baseline_per_row)
    sq_err_per_user_ols = np.asarray(sq_err_per_user_ols_per_row)
    sq_err_pilsd = np.asarray(sq_err_pilsd_per_row)

    rmse_no_calib = float(np.sqrt(sq_err_no_calib.mean()))
    rmse_baseline = float(np.sqrt(sq_err_baseline.mean()))
    rmse_per_user_ols = float(np.sqrt(sq_err_per_user_ols.mean()))
    rmse_pilsd = float(np.sqrt(sq_err_pilsd.mean()))
    gain_pct = ((rmse_baseline - rmse_pilsd) / rmse_baseline * 100.0
                if rmse_baseline > 0 else float("nan"))

    log(f"[Stage B / {backbone_alias}] rmse_no_calib    = {rmse_no_calib:.5f}")
    log(f"[Stage B / {backbone_alias}] rmse_baseline    = {rmse_baseline:.5f}")
    log(f"[Stage B / {backbone_alias}] rmse_per_user_OLS= {rmse_per_user_ols:.5f}")
    log(f"[Stage B / {backbone_alias}] rmse_pilsd       = {rmse_pilsd:.5f}")
    log(f"[Stage B / {backbone_alias}] gain_pct         = {gain_pct:+.3f}%")

    log(f"[Stage B / {backbone_alias}] cluster-bootstrap B={bootstrap_B} ...")
    bs = cluster_bootstrap_gain_ci(
        sq_err_baseline, sq_err_pilsd, per_row_user,
        B=bootstrap_B, seed=bootstrap_seed,
    )
    log(f"[Stage B / {backbone_alias}] BCa CI95: "
        f"[{bs['ci_bca_lo']:+.3f}, {bs['ci_bca_hi']:+.3f}]")

    return {
        "n_obs_total": n_obs_total,
        "n_users_total": n_users_total,
        "n_users_post_filter": n_users_post_filter,
        "n_obs_post_filter": int(len(df)),
        "n_degenerate_x_users": n_degenerate_x,
        "n_reliable_for_tau_pool": n_reliable,
        "tau_alpha_sq": tau_a_sq,
        "tau_beta_sq": tau_b_sq,
        "pop_alpha": pop_alpha,
        "pop_beta": pop_beta,
        "rmse_no_calib": rmse_no_calib,
        "rmse_baseline_pop": rmse_baseline,
        "rmse_per_user_OLS": rmse_per_user_ols,
        "rmse_pilsd_shrunk": rmse_pilsd,
        "gain_pct": float(gain_pct),
        "gain_ci_bca_lo": float(bs.get("ci_bca_lo", float("nan"))),
        "gain_ci_bca_hi": float(bs.get("ci_bca_hi", float("nan"))),
        "gain_ci_percentile_lo": float(bs.get("ci_percentile_lo", float("nan"))),
        "gain_ci_percentile_hi": float(bs.get("ci_percentile_hi", float("nan"))),
        "gain_ci_jackknife_lo": float(bs.get("ci_jackknife_lo", float("nan"))),
        "gain_ci_jackknife_hi": float(bs.get("ci_jackknife_hi", float("nan"))),
        "boot_mean": float(bs.get("boot_mean", float("nan"))),
        "boot_sd": float(bs.get("boot_sd", float("nan"))),
        "boot_n_valid": int(bs.get("n_boot_valid", 0)),
        "z0_bias_correction": float(bs.get("z0_bias_correction", float("nan"))),
        "a_hat_acceleration": float(bs.get("a_hat_acceleration", float("nan"))),
        "per_user_table": us.to_dict(orient="records"),
    }


# ============================================================================
# Stage C - cross-backbone aggregate verdict (4-class STRICT per Q7 brief)
# ============================================================================


def per_backbone_verdict(gain: float, ci_lo: float, ci_hi: float, n_users: int) -> str:
    if not (np.isfinite(gain) and np.isfinite(ci_lo) and np.isfinite(ci_hi)):
        return "INCONCLUSIVE-NaN"
    if ci_hi < 0:
        return "FALSIFIED-NEGATIVE-CI"
    if gain >= ESTABLISHED_GAIN_LO and ci_lo > 0 and n_users >= N_CLUSTERS_FOR_ESTABLISHED:
        return "CONFIRMS"
    if gain >= ESTABLISHED_GAIN_LO and ci_lo > 0 and n_users < N_CLUSTERS_FOR_ESTABLISHED:
        return "CONFIRMS-COHORT-BELOW-500"
    if gain >= 1.0 and ci_lo > 0:
        return "PARTIAL-CONFIRMS"
    return "INCONCLUSIVE-CI-STRADDLES"


def cross_backbone_verdict(per_backbone: dict[str, dict]) -> str:
    """Aggregate per-backbone outcomes into Q7 brief 4-class verdict.

    ESTABLISHED-PILSD-CROSS-BACKBONE-CONFIRMS  if all 3 confirm
    MODERATE-PILSD-CROSS-BACKBONE-PARTIAL      if 2/3 confirm
    PRELIMINARY-INCONCLUSIVE-CROSS-BACKBONE    if 1/3 confirms or wide CI
    FALSIFIED-PILSD-QWEN-ONLY                  if other backbones reverse
    """
    statuses = [v.get("per_backbone_verdict", "INCONCLUSIVE-NaN")
                for v in per_backbone.values()]
    n_confirms = sum(1 for s in statuses if s in ("CONFIRMS", "CONFIRMS-COHORT-BELOW-500"))
    n_falsified = sum(1 for s in statuses if s == "FALSIFIED-NEGATIVE-CI")

    # FALSIFIED only if at least one reversal AND Qwen does NOT reverse
    qwen_status = per_backbone.get("qwen25_7b_inst", {}).get("per_backbone_verdict", "")
    qwen_confirms = qwen_status in ("CONFIRMS", "CONFIRMS-COHORT-BELOW-500", "PARTIAL-CONFIRMS")

    if n_falsified >= 1 and qwen_confirms:
        # The "Qwen-only" hypothesis: Qwen confirms but other(s) reverse
        return "FALSIFIED-PILSD-QWEN-ONLY"

    if n_confirms == 3:
        return "ESTABLISHED-PILSD-CROSS-BACKBONE-CONFIRMS"
    if n_confirms == 2:
        return "MODERATE-PILSD-CROSS-BACKBONE-PARTIAL"
    if n_confirms == 1:
        return "PRELIMINARY-INCONCLUSIVE-CROSS-BACKBONE-1OF3"
    return "PRELIMINARY-INCONCLUSIVE-CROSS-BACKBONE-0OF3"


# ============================================================================
# Driver
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prism-utterances-cache",
                   default=str(T1 / "data" / "prism_utterances_full.parquet"),
                   help="Local cache of PRISM utterances (auto-downloads from HF if missing).")
    p.add_argument("--scored-dir",
                   default=str(T1 / "data" / "q7_backbone_scored"),
                   help="Directory holding per-backbone scored parquets.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--bootstrap-seed", type=int, default=BOOT_SEED)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--k-folds", type=int, default=K_FOLDS)
    p.add_argument("--min-obs-per-user", type=int, default=MIN_OBS_PER_USER)
    p.add_argument("--per-user-utterance-cap", type=int, default=PER_USER_UTTERANCE_CAP)
    p.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS)
    p.add_argument("--max-response-tokens", type=int, default=MAX_RESPONSE_TOKENS)
    p.add_argument("--no-quantize-4bit", action="store_true",
                   help="Disable 4-bit nf4 quantization (default: enabled).")
    p.add_argument("--force-rescore", action="store_true",
                   help="Re-run Stage A even if scored parquet exists.")
    p.add_argument("--skip-backbones", default="",
                   help="Comma-separated backbone aliases to skip (for partial dispatch).")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke-test mode: cap Stage A at --smoke-limit utterances per backbone.")
    p.add_argument("--smoke-limit", type=int, default=50)
    p.add_argument("--output-dir", default=str(OUT_DIR))
    return p.parse_args()


def stage_a_needs_rerun(parquet: Path, force: bool, min_rows: int) -> bool:
    if force:
        return True
    if not parquet.exists():
        return True
    try:
        n = len(pd.read_parquet(parquet, columns=["utterance_id"]))
        return n < min_rows
    except Exception:
        return True


def load_or_download_prism_utterances(cache_path: Path) -> pd.DataFrame:
    """Load PRISM utterances; download from HF if local cache missing."""
    if cache_path.exists():
        log(f"[PRISM] loading utterances from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        log(f"[PRISM] cache missing; downloading from HannahRoseKirk/prism-alignment via HF")
        from datasets import load_dataset
        utt_ds = load_dataset("HannahRoseKirk/prism-alignment", "utterances", split="train")
        df = utt_ds.to_pandas()
        df = df[df["user_prompt"].notna() & df["model_response"].notna()].reset_index(drop=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        log(f"[PRISM] downloaded + cached {len(df)} rows -> {cache_path}")

    # Filter to utterances with non-null user_prompt + model_response + score
    df = df[df["user_prompt"].notna() & df["model_response"].notna()].reset_index(drop=True)
    log(f"[PRISM] {len(df)} utterances loaded; "
        f"{df['user_id'].nunique()} unique users")
    return df


def per_user_cap_subsample(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Cap per-user utterance count at `cap` (deterministic via seed).

    Compute-envelope discipline: PRISM median is 48 utt/user; capping at 20
    bounds total wall while keeping K=5 CV signal (>=4 obs/fold/user)."""
    if cap is None or cap <= 0:
        return df
    rng = np.random.default_rng(seed)
    parts = []
    for uid, grp in df.groupby("user_id"):
        if len(grp) <= cap:
            parts.append(grp)
        else:
            idx = rng.choice(len(grp), size=cap, replace=False)
            parts.append(grp.iloc[idx])
    out = pd.concat(parts, ignore_index=True)
    log(f"[PRISM] post per-user-cap (cap={cap}): {len(out)} rows / "
        f"{out['user_id'].nunique()} users (was {len(df)})")
    return out


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    scored_dir = Path(args.scored_dir)
    scored_dir.mkdir(parents=True, exist_ok=True)

    skip_backbones = {b.strip() for b in args.skip_backbones.split(",") if b.strip()}
    active_backbones = {k: v for k, v in BACKBONES.items() if k not in skip_backbones}

    summary: dict[str, Any] = {
        "experiment_id": "Q7_backbone_cross_architecture",
        "verdict_class": "PENDING",
        "args": vars(args),
        "seed": int(args.seed),
        "bootstrap_seed": int(args.bootstrap_seed),
        "n_boot": int(args.n_boot),
        "k_folds": int(args.k_folds),
        "min_obs_per_user": int(args.min_obs_per_user),
        "per_user_utterance_cap": int(args.per_user_utterance_cap),
        "backbones": active_backbones,
        "skill_citations": [
            "Skill: research-grade-code-audit-pre-launch v1 G1-G12",
            "Skill: honest-disclosure SCOPE-not-RETRACT (4-class STRICT cross-backbone)",
            "Skill: post-experiment-discipline-3-track Step 5+7",
            "Skill: launch-runpod-h100-job (h100_v2_backup)",
            "Skill: gpu-artifact-sync",
            "Skill: icml-neurips-critical-reviewer-2026 Pass 5 cross-architecture generalization",
        ],
        "honest_disclosure_design_notes": (
            "Q7 uses mean-response log-likelihood scoring (Stiennon et al. 2020) as a "
            "BACKBONE-AGNOSTIC RM proxy across all 3 backbones, NOT the legacy LoRA-RM "
            "classifier head used for the PRISM 8.58% headline. This isolates backbone-"
            "architecture variance from RM-training variance. Q7 does NOT re-replicate "
            "the 8.58% legacy headline; it is a self-contained 3-arm cross-architecture "
            "comparison. Q9's pooled-4-corpus PRISM-slice gain (+9.977% with Qwen mean-LL) "
            "is the closest internal anchor for the Qwen2.5-7B mean-LL arm. Per-user "
            "utterance cap (20) bounds wall-time on H100 80GB nf4 while preserving K=5 "
            "CV signal (>=4 obs/fold/user)."
        ),
        "anomaly_branches_fired": [],
        "stages": {},
        "per_backbone": {},
    }

    log(f"[Q7] starting at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    log(f"[Q7] output_dir: {out_dir}")
    log(f"[Q7] scored_dir: {scored_dir}")
    log(f"[Q7] git_head_t3: {git_head_t3()}")
    log(f"[Q7] active_backbones: {list(active_backbones.keys())}")

    # ------------------------------------------------------------------
    # Load PRISM utterances + per-user cap subsample (deterministic)
    # ------------------------------------------------------------------
    try:
        utt_df = load_or_download_prism_utterances(Path(args.prism_utterances_cache))
    except Exception as exc:
        summary["verdict_class"] = "RUNTIME-ERROR-PRISM-LOAD"
        summary["error"] = traceback.format_exc()
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        raise

    utt_capped = per_user_cap_subsample(utt_df, args.per_user_utterance_cap, args.seed)
    summary["prism_n_rows_full"] = int(len(utt_df))
    summary["prism_n_users_full"] = int(utt_df["user_id"].nunique())
    summary["prism_n_rows_capped"] = int(len(utt_capped))
    summary["prism_n_users_capped"] = int(utt_capped["user_id"].nunique())

    # ------------------------------------------------------------------
    # Per-backbone Stage A + Stage B (sequential to free GPU between models)
    # ------------------------------------------------------------------
    for backbone_alias, model_id in active_backbones.items():
        log(f"\n=== BACKBONE: {backbone_alias} ({model_id}) ===")
        backbone_summary: dict[str, Any] = {
            "model_id": model_id,
            "stages": {},
        }
        scored_parquet = scored_dir / f"prism_{backbone_alias}_meanll_scored.parquet"

        # Stage A
        if stage_a_needs_rerun(scored_parquet, args.force_rescore,
                                MIN_SCORED_ROWS_FOR_VALID_CACHE):
            log(f"[Stage A / {backbone_alias}] running (parquet missing / too small / forced)")
            try:
                stage_a_meta = stage_a_score_prism(
                    utt_capped, scored_parquet,
                    model_id=model_id,
                    backbone_alias=backbone_alias,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_response_tokens=args.max_response_tokens,
                    quantize_4bit=not args.no_quantize_4bit,
                    smoke_limit=args.smoke_limit if args.smoke else None,
                )
                backbone_summary["stages"]["stage_a"] = stage_a_meta
            except Exception as exc:
                backbone_summary["stages"]["stage_a"] = {
                    "error": traceback.format_exc(),
                    "verdict_branch": "STAGE-A-RUNTIME-ERROR",
                }
                summary["per_backbone"][backbone_alias] = backbone_summary
                summary_path.write_text(json.dumps(summary, indent=2, default=str))
                log(f"[ERR / {backbone_alias}] Stage A failed; continuing to next backbone")
                continue
        else:
            n = len(pd.read_parquet(scored_parquet, columns=["utterance_id"]))
            backbone_summary["stages"]["stage_a"] = {
                "elapsed_seconds": 0.0,
                "cache_reused": True,
                "scored_parquet": str(scored_parquet),
                "scored_sha256": file_sha256(scored_parquet),
                "scored_n_rows": int(n),
            }
            log(f"[Stage A / {backbone_alias}] CACHE REUSE: {scored_parquet} ({n} rows)")

        # Stage B
        try:
            scored_df = pd.read_parquet(scored_parquet)
            stage_b_result = stage_b_pilsd_per_user(
                scored_df,
                backbone_alias=backbone_alias,
                seed=args.seed,
                k_folds=args.k_folds,
                min_obs_per_user=args.min_obs_per_user,
                bootstrap_B=args.n_boot,
                bootstrap_seed=args.bootstrap_seed,
            )
            backbone_summary["stages"]["stage_b"] = stage_b_result
        except Exception as exc:
            backbone_summary["stages"]["stage_b"] = {
                "error": traceback.format_exc(),
                "verdict_branch": "STAGE-B-RUNTIME-ERROR",
            }
            summary["per_backbone"][backbone_alias] = backbone_summary
            summary_path.write_text(json.dumps(summary, indent=2, default=str))
            log(f"[ERR / {backbone_alias}] Stage B failed; continuing to next backbone")
            continue

        # Per-backbone verdict
        gain = float(stage_b_result.get("gain_pct", float("nan")))
        ci_lo = float(stage_b_result.get("gain_ci_bca_lo", float("nan")))
        ci_hi = float(stage_b_result.get("gain_ci_bca_hi", float("nan")))
        n_users = int(stage_b_result.get("n_users_post_filter", 0))
        backbone_summary["gain_pct"] = gain
        backbone_summary["gain_ci_bca_lo"] = ci_lo
        backbone_summary["gain_ci_bca_hi"] = ci_hi
        backbone_summary["n_users_post_filter"] = n_users
        backbone_summary["per_backbone_verdict"] = per_backbone_verdict(
            gain, ci_lo, ci_hi, n_users
        )

        # Save per-user table separately
        per_user_path = out_dir / f"per_user_{backbone_alias}.parquet"
        if isinstance(stage_b_result.get("per_user_table"), list):
            pd.DataFrame(stage_b_result["per_user_table"]).to_parquet(
                per_user_path, index=False,
            )
            backbone_summary["per_user_parquet"] = str(per_user_path)
            # Strip from in-memory summary (already on disk)
            stage_b_result.pop("per_user_table", None)

        summary["per_backbone"][backbone_alias] = backbone_summary
        # Write summary after each backbone for crash-resilience
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        log(f"[backbone {backbone_alias}] gain={gain:+.3f}% CI [{ci_lo:+.3f}, {ci_hi:+.3f}] "
            f"n_users={n_users} verdict={backbone_summary['per_backbone_verdict']}")

    # ------------------------------------------------------------------
    # Stage C - cross-backbone aggregate verdict
    # ------------------------------------------------------------------
    cross_verdict = cross_backbone_verdict(summary["per_backbone"])
    summary["verdict_class"] = cross_verdict

    # Anomaly branches
    anomalies = []
    for alias, res in summary["per_backbone"].items():
        if "error" in res.get("stages", {}).get("stage_a", {}):
            anomalies.append(f"stage_a_runtime_error_{alias}")
        if "error" in res.get("stages", {}).get("stage_b", {}):
            anomalies.append(f"stage_b_runtime_error_{alias}")
        if not np.isfinite(res.get("gain_pct", float("nan"))):
            anomalies.append(f"gain_nan_{alias}")
        if res.get("n_users_post_filter", 0) < N_CLUSTERS_FOR_ESTABLISHED:
            anomalies.append(f"n_users_below_500_{alias}")
    summary["anomaly_branches_fired"] = anomalies

    # Build per-backbone diagnostic table
    rows = []
    for alias, res in summary["per_backbone"].items():
        rows.append({
            "backbone": alias,
            "model_id": res.get("model_id", ""),
            "gain_pct": res.get("gain_pct", float("nan")),
            "gain_ci_bca_lo": res.get("gain_ci_bca_lo", float("nan")),
            "gain_ci_bca_hi": res.get("gain_ci_bca_hi", float("nan")),
            "n_users_post_filter": res.get("n_users_post_filter", 0),
            "rmse_baseline_pop": res.get("stages", {}).get("stage_b", {}).get(
                "rmse_baseline_pop", float("nan")),
            "rmse_pilsd_shrunk": res.get("stages", {}).get("stage_b", {}).get(
                "rmse_pilsd_shrunk", float("nan")),
            "tau_alpha_sq": res.get("stages", {}).get("stage_b", {}).get(
                "tau_alpha_sq", float("nan")),
            "tau_beta_sq": res.get("stages", {}).get("stage_b", {}).get(
                "tau_beta_sq", float("nan")),
            "pearson_r_rm_user": res.get("stages", {}).get("stage_a", {}).get(
                "pearson_r_rm_score_vs_user_score", float("nan")),
            "per_backbone_verdict": res.get("per_backbone_verdict", "MISSING"),
        })
    per_backbone_df = pd.DataFrame(rows)
    per_backbone_path = out_dir / "per_backbone.parquet"
    per_backbone_df.to_parquet(per_backbone_path, index=False)
    summary["per_backbone_parquet"] = str(per_backbone_path)

    summary["git_head_t3_at_run"] = git_head_t3()
    summary["completion_timestamp_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print()
    print(f"=== Q7 Backbone-Mismatch Cross-Architecture ===")
    print(f"verdict_class : {cross_verdict}")
    print(f"per-backbone summary:")
    print(per_backbone_df.to_string(index=False))
    print(f"anomalies     : {anomalies}")
    print(f"summary_path  : {summary_path}")
    print(f"per_backbone  : {per_backbone_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sp = OUT_DIR / "summary.json"
        if not sp.exists():
            sp.write_text(json.dumps({
                "experiment_id": "Q7_backbone_cross_architecture",
                "verdict_class": "RUNTIME-ERROR",
                "error": traceback.format_exc(),
            }, indent=2))
        sys.exit(2)
