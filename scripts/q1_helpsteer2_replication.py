"""Q1 - HelpSteer2 cross-corpus PEBS replication.

Addresses the out-of-distribution generalization question: does PEBS's
+8.58% PRISM headline replicate on NVIDIA HelpSteer2 (Wang et al. 2024)
with REAL annotator-rating disagreement data (rather than treating the five
quality attributes as pseudo-raters)?

DESIGN DECISION DOCUMENTED HONESTLY
===================================
The original design assumed an "annotator_id field" exists in HelpSteer2. **It does not.**
NVIDIA's released HelpSteer2 (`nvidia/HelpSteer2`) anonymizes annotators across rows;
each row's `helpfulness` / `correctness` / `coherence` / `complexity` / `verbosity`
is reduced to a single mean Likert score in the main config. The `disagreements`
config exposes the per-annotator ratings as a list `[r_1, r_2, r_3]` per row,
**but list-position is NOT a stable cross-row annotator identifier** (the HelpSteer2
paper, Wang et al. 2024 Section 3.4, explicitly states annotator order is not
preserved across rows). Therefore PRISM-style per-rater PEBS (cluster=user_id with
n_obs=10..500 per user) is **structurally infeasible** on HelpSteer2.

Closest defensible PEBS analog the dataset DOES support:
  cluster_id = prompt_hash
  observations = (response_idx, annotator_slot) tuples within each prompt
  target y = annotator's helpfulness rating for that response
  feature x = RM score for that response (replicated across annotators)

Each prompt has K=2-4 responses x 3 annotators = approx 6 obs per cluster. This
satisfies MIN_UTT_PER_CLUSTER=6 (matches PRISM-OASST2 headline convention) but
is WEAKER than PRISM's median-50 obs/user. We note up front that this difference
is expected to **bound the gain magnitude downward** by Morris (1983) eq. 2.9
(small-cluster MoM bias + high sampling-V_j -> omega_j -> 0 -> PEBS approx pop-slope).
Predicted gain: 1-3pp (PARTIAL-HELPSTEER2-PARTIAL-CONFIRMS) most likely; >=3pp
CONFIRMED would be a positive surprise; CI straddling zero would be the honest
null-by-construction outcome.

This is falsifier-class evidence - testing
PEBS's APPLICABILITY BOUNDARY on a corpus whose annotation schema differs from
PRISM. A clean POSITIVE generalizes the headline; a clean NULL bounds it.

Pipeline (mirrors wave_c_exp2_oasst2_replication + eval_oasst2_pebs_calibrator)
------------------------------------------------------------------------------
Stage A - score HelpSteer2 disagreements responses with Qwen2.5-7B-Instruct
          mean-response log-likelihood (matches PRISM headline RM):
          out: data/helpsteer2/helpsteer2_qwen7b_disagreements_scored.parquet
          (one row per (prompt_hash, response_hash) - approx 23k unique pairs)

Stage B - Per-prompt PEBS calibrator on the scored parquet:
          5-fold within-prompt CV; pop-OLS pre-pass for tau^2_alpha, tau^2_beta;
          per-prompt OLS with V_alpha, V_beta; omega_alpha=tau^2/(tau^2+V); same beta;
          PEBS-shrunk = omega*per_cluster + (1-omega)*pop_slope.
          Cluster-bootstrap by prompt_hash; B=2000.

Design notes
------------
- Math: VERBATIM PEBS math from eval_oasst2_pebs_calibrator.py
  (`ols_with_V` + `cluster_bootstrap_gain_ci` + 5-fold within-cluster CV).
- Design mismatch documented honestly: the original design assumed
  per-rater clusters; HelpSteer2 doesn't expose them. The closest defensible
  analog (per-prompt clusters) is implemented + the limitation stated up front.
- No silent-bypass: Stage-A output_parquet existence check is BY-PATH only;
  re-run if --force-rescore or scored row count below MIN_SCORED_ROWS_FOR_VALID_CACHE.
- Pipeline integrity: gain_pct = (rmse_pop_slope - rmse_pebs_shrunk) /
  rmse_pop_slope * 100; identical formula to PRISM headline.
- Reference implementation: math + bootstrap inherit verbatim from
  eval_oasst2_pebs_calibrator.py.
- Hyperparameters: K_FOLDS=5 / N_BOOT=2000 / MIN_OBS_PER_CLUSTER=6 /
  MAX_SEQ_LEN=2048 - matches PRISM headline + OASST2 replication conventions.
- Diagnostics: Stage A logs scoring rate + ETA every 200 rows; Stage B
  logs per-fold per-arm RMSE + cluster-bootstrap progress.
- Reproducibility: SEED=20260420 + bootstrap_seed=20260420 (matches OASST2);
  git HEAD + parquet sha256 persisted in summary.json.
- Compute: Stage A ~3-4h (23k unique pairs x 7B nf4 mean-LL on a
  96GB GPU ~2 row/s); Stage B ~5-10min CPU.
- The decision rule enumerates the design-vs-data mismatch and the
  small-cluster Morris-bound expectation explicitly.

Outcome classification
----------------------
CONFIRMED-HELPSTEER2-CONFIRMS-PRISM            gain >= +3pp + CI excludes 0 +
                                                  n_clusters_post_filter >= 500;
                                                  generalizes PRISM headline to
                                                  HelpSteer2 schema (with caveat
                                                  on per-prompt-vs-per-rater
                                                  cluster type).
PARTIAL-HELPSTEER2-PARTIAL-CONFIRMS              gain in [+1, +3]pp + CI
                                                  excludes 0; small-cluster
                                                  Morris-bound regime; consistent
                                                  with PEBS mechanism but
                                                  attenuated by limited within-
                                                  cluster degrees of freedom.
TENTATIVE-INCONCLUSIVE-DATASET-SCHEMA-LIMIT     CI straddles zero; default
                                                  expectation given n_obs=6
                                                  per cluster + per-prompt
                                                  (not per-rater) cluster
                                                  structure.
REJECTED-HELPSTEER2-DOES-NOT-REPLICATE           CI strictly negative;
                                                  PEBS mechanism does NOT
                                                  generalize to per-prompt
                                                  clusters; paper claim must be
                                                  SCOPE-LIMITED to per-rater
                                                  RLHF preference corpora.

Output
------
results/track1_helpsteer2_replication/{summary.json, per_cluster.parquet}
+ a cross-corpus table row + a Limitations clause.

References
----------
- Wang, Z. et al. (2024). HelpSteer2: Open-source dataset for top-performing
    reward models. NeurIPS Datasets and Benchmarks 2024.
    https://huggingface.co/datasets/nvidia/HelpSteer2
- Stiennon, N. et al. (2020). Learning to summarize from human feedback. NeurIPS.
    (mean-response log-likelihood reward proxy)
- Morris, C. (1983). Parametric Empirical Bayes inference: theory and applications.
    JASA 78(381), 47-55.
- Henderson, C. (1975). Best linear unbiased estimation and prediction under a
    selection model. Biometrics 31(2), 423-447.
- Cameron, A. C., Gelbach, J. B., Miller, D. L. (2008). Bootstrap-based
    improvements for inference with clustered errors. Review of Economics and
    Statistics 90(3), 414-427.
- Efron, B. (1987). Better bootstrap confidence intervals. JASA 82(397), 171-185.
"""
from __future__ import annotations

import os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"

import argparse
import gzip
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
    raise ImportError("scipy required for cluster-bootstrap + Wilcoxon") from exc


ROOT = Path(__file__).resolve().parents[2]                  # 3_PEBS_Standalone/
T1   = ROOT.parent / "1_Causal_RLHF"

OUT_DIR = ROOT / "results" / "track1_helpsteer2_replication"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters (PRISM-headline-anchored; mirror eval_oasst2_pebs_calibrator.py)
SEED = 20260420                          # matches PRISM canonical
BOOT_SEED = 20260420                     # matches PRISM canonical
N_BOOT = 2000
K_FOLDS = 5
MIN_OBS_PER_CLUSTER = 6                  # matches OASST2 headline (PRISM analog: 10)
MIN_SCORED_ROWS_FOR_VALID_CACHE = 5_000  # re-run Stage A if cache below
MAX_PROMPT_TOKENS = 256
MAX_RESPONSE_TOKENS = 256
MODEL_ID_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"

# Outcome thresholds
ESTABLISHED_GAIN_LO = 3.0    # CONFIRMED if gain >= 3pp + CI excludes 0
MODERATE_GAIN_LO = 1.0       # PARTIAL if 1-3pp + CI excludes 0
MODERATE_GAIN_HI = 3.0
N_CLUSTERS_FOR_ESTABLISHED = 500

# HelpSteer2 5 attribute axes - Q1 restricts to helpfulness only (per-row
# Likert 0-4) for apples-to-apples with PRISM 0-100 scalar. Verbosity = closer
# to a covariate; coherence/complexity/correctness are rater-content axes that
# DO NOT correspond to the helpfulness target.
PRIMARY_TARGET_AXIS = "helpfulness"


# ============================================================================
# Reproducibility helpers
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
# Stage A - score HelpSteer2 disagreements responses with Qwen2.5-7B-Instruct
# mean-response log-likelihood (matches PRISM headline RM proxy)
# ============================================================================


def _md5_short(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def _load_disagreements_jsonl(jsonl_gz_path: str) -> list[dict]:
    """Load HelpSteer2 disagreements config raw JSONL."""
    out = []
    with gzip.open(jsonl_gz_path, "rt") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def _build_observation_rows(disagreements: list[dict]) -> pd.DataFrame:
    """Flatten the disagreements list-per-row into one observation per (prompt,
    response, annotator_slot)."""
    rows = []
    for r in disagreements:
        prompt_hash = _md5_short(r["prompt"])
        response_hash = _md5_short(r["response"])
        helpfulness_list = r.get(PRIMARY_TARGET_AXIS, [])
        if not isinstance(helpfulness_list, list) or len(helpfulness_list) < 1:
            continue
        for slot, h in enumerate(helpfulness_list):
            rows.append({
                "prompt_hash": prompt_hash,
                "response_hash": response_hash,
                "prompt": r["prompt"],
                "response": r["response"],
                "annotator_slot": int(slot),  # NOT stable across prompts
                "helpfulness": int(h),
                # Other 4 axes for diagnostic only (NOT used in headline)
                "correctness": int(r.get("correctness", [])[slot]) if slot < len(r.get("correctness", [])) else None,
                "coherence":   int(r.get("coherence", [])[slot])   if slot < len(r.get("coherence", []))   else None,
                "complexity":  int(r.get("complexity", [])[slot])  if slot < len(r.get("complexity", []))  else None,
                "verbosity":   int(r.get("verbosity", [])[slot])   if slot < len(r.get("verbosity", []))   else None,
            })
    return pd.DataFrame(rows)


def stage_a_score_with_qwen7b(
    obs_df: pd.DataFrame,
    output_parquet: Path,
    model_id: str = MODEL_ID_DEFAULT,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    max_response_tokens: int = MAX_RESPONSE_TOKENS,
    quantize_4bit: bool = True,
    smoke_limit: int | None = None,
) -> dict:
    """Score each unique (prompt, response) with Qwen2.5-7B-Instruct mean-LL.

    Reference-parity: replicates `score_oasst2_with_qwen7b.py`
    lines 215-230 verbatim --- separate prompt/response tokenization, concat,
    -100 mask on prompt positions, score = -loss (mean response log-likelihood).
    This is the canonical Stiennon (2020) / Ouyang (2022) RLHF reward proxy
    using shifted-CE loss, simpler and more robust than chat-template
    boundary-finding on a per-tokenizer basis.

    Output rows: one per UNIQUE (prompt_hash, response_hash) --- replication into
    annotator-slot rows happens in the merge in Stage B prep.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        from transformers import BitsAndBytesConfig
        BNB_AVAILABLE = True
    except ImportError:
        BNB_AVAILABLE = False

    # Unique (prompt, response) pairs - score each once
    unique_pairs = obs_df.drop_duplicates(
        subset=["prompt_hash", "response_hash"]
    )[["prompt_hash", "response_hash", "prompt", "response"]].reset_index(drop=True)

    if smoke_limit is not None:
        unique_pairs = unique_pairs.iloc[:smoke_limit].copy()

    log(f"[stage A] {len(unique_pairs)} unique (prompt, response) pairs to score")
    log(f"[stage A] loading model: {model_id} (quantize_4bit={quantize_4bit})")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
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
    log(f"[stage A] model loaded; device={device}; "
        f"n_params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    scored_rows = []
    skipped = 0
    t0 = time.time()
    n_total = len(unique_pairs)

    for idx, row in unique_pairs.iterrows():
        prompt = str(row["prompt"]) or ""
        response = str(row["response"]) or ""
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
            if (idx % 200) == 0:
                log(f"[warn] row {idx}: scoring failed ({e}); skipping")
            skipped += 1
            continue

        scored_rows.append({
            "prompt_hash": row["prompt_hash"],
            "response_hash": row["response_hash"],
            "rm_score": score,
            "n_tokens_scored": n_resp_tok,
        })

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1e-3)
            eta_min = (n_total - (idx + 1)) / max(rate, 1e-3) / 60
            log(f"[stage A] {idx+1}/{n_total} ({100*(idx+1)/n_total:.1f}%) | "
                f"rate {rate:.2f} row/s | ETA {eta_min:.1f} min | skipped={skipped}")

    scored_df = pd.DataFrame(scored_rows)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_parquet(output_parquet, index=False)
    elapsed = time.time() - t0
    log(f"[stage A] done in {elapsed/60:.1f} min; {len(scored_df)} rows -> {output_parquet}")

    return {
        "elapsed_seconds": float(elapsed),
        "scored_parquet": str(output_parquet),
        "scored_n_unique_pairs": int(len(scored_df)),
        "scored_sha256": file_sha256(output_parquet),
    }


# ============================================================================
# Stage B - PEBS math (verbatim from eval_oasst2_pebs_calibrator.py + minor
# rename: user_id -> cluster_id where cluster = prompt_hash for HelpSteer2)
# ============================================================================


def ols_with_V(x: np.ndarray, y: np.ndarray):
    """Returns (intercept, slope, V_intercept, V_slope) with sample-variance estimates.
    VERBATIM from eval_oasst2_pebs_calibrator.py:90-105."""
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
    """Random k-fold split. VERBATIM from eval_oasst2_pebs_calibrator.py:108-120."""
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
    sq_err_pebs: np.ndarray,
    per_row_cluster_idx: list,
    B: int = 2000,
    seed: int = 20260420,
):
    """Cluster-resample-by-cluster bootstrap on gain_pct. Returns BCa + percentile +
    jackknife CIs. VERBATIM from eval_oasst2_pebs_calibrator.py:150-280 with
    user_id -> cluster_id rename."""
    if len(sq_err_baseline) == 0 or len(sq_err_pebs) == 0:
        return {
            "boot_mean": float("nan"), "boot_sd": float("nan"),
            "ci_percentile_lo": float("nan"), "ci_percentile_hi": float("nan"),
            "ci_bca_lo": float("nan"), "ci_bca_hi": float("nan"),
            "ci_jackknife_lo": float("nan"), "ci_jackknife_hi": float("nan"),
            "n_boot_valid": 0, "error": "empty sq_err arrays",
        }

    cluster_to_block = {}
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
        sp = sq_err_pebs[idx_arr]
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
            "n_boot_valid": int(len(boot_arr)),
            "error": "BOOTSTRAP_CI_DEGENERATE",
        }

    boot_mean = float(boot_arr.mean())
    boot_sd = float(boot_arr.std(ddof=1))
    pct_lo = float(np.quantile(boot_arr, 0.025))
    pct_hi = float(np.quantile(boot_arr, 0.975))

    rmse_base_obs = math.sqrt(sq_err_baseline.mean())
    rmse_pebs_obs = math.sqrt(sq_err_pebs.mean())
    obs_gain = ((rmse_base_obs - rmse_pebs_obs) / rmse_base_obs * 100.0
                if rmse_base_obs > 0 else 0.0)

    p_below = float(np.mean(boot_arr < obs_gain))
    z0 = scstats.norm.ppf(p_below) if 0.0 < p_below < 1.0 else 0.0

    # Jackknife leave-one-cluster-out
    jack_gains = []
    for j in range(n_clusters):
        excluded = cluster_keys[j]
        flat = [i for c in cluster_keys if c != excluded for i in cluster_to_block[c]]
        if not flat:
            continue
        sb_jk = sq_err_baseline[np.array(flat)]
        sp_jk = sq_err_pebs[np.array(flat)]
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


def stage_b_pebs_per_prompt(
    obs_df: pd.DataFrame,
    seed: int = SEED,
    k_folds: int = K_FOLDS,
    min_obs_per_cluster: int = MIN_OBS_PER_CLUSTER,
    bootstrap_B: int = N_BOOT,
    bootstrap_seed: int = BOOT_SEED,
) -> dict:
    """Run PEBS per-prompt-cluster shrinkage on annotator-rating observations.

    cluster_id = prompt_hash, observation = (response_idx, annotator_slot).
    target y = helpfulness rating (0-4 Likert).
    feature x = RM score (mean-response log-likelihood; replicated across
                annotators of the same response).

    Pipeline mirrors `eval_oasst2_pebs_calibrator.py:run_pebs_oasst2`.
    """
    rng = np.random.default_rng(seed)

    df = obs_df.dropna(subset=["helpfulness", "rm_score"]).reset_index(drop=True)
    n_obs_total = int(len(df))
    n_clusters_total = int(df["prompt_hash"].nunique())

    # Sanity check - feature x should NOT be constant within cluster
    # (would force beta degenerate)
    n_per = df.groupby("prompt_hash").size()
    keep_clusters = set(n_per[n_per >= min_obs_per_cluster].index)
    df = df[df["prompt_hash"].isin(keep_clusters)].reset_index(drop=True)
    n_clusters_post_filter = int(df["prompt_hash"].nunique())
    log(f"[stage B] post-filter: {n_clusters_post_filter} clusters / {len(df)} obs "
        f"(min_obs_per_cluster={min_obs_per_cluster})")

    if n_clusters_post_filter < 100:
        return {
            "n_obs_total": n_obs_total,
            "n_clusters_total": n_clusters_total,
            "n_clusters_post_filter": n_clusters_post_filter,
            "error": f"N_CLUSTERS_BELOW_100_AT_FILTER: {n_clusters_post_filter} < 100",
            "verdict_branch": "INCONCLUSIVE-COHORT-SHRUNK",
        }

    # Within-cluster x-variance check
    x_var_per_cluster = df.groupby("prompt_hash")["rm_score"].var(ddof=0)
    n_degenerate_x = int((x_var_per_cluster < 1e-9).sum())
    log(f"[stage B] clusters with degenerate within-cluster x-variance: {n_degenerate_x}/{n_clusters_post_filter}")

    # Population OLS on entire dataset
    slope_pop, intercept_pop = np.polyfit(df.rm_score, df.helpfulness, 1)
    pop_alpha = float(intercept_pop)
    pop_beta = float(slope_pop)
    log(f"[stage B] pop_alpha={pop_alpha:.4f}, pop_beta={pop_beta:.4f}")

    # Per-cluster OLS for tau^2 pre-pass
    cluster_stats = []
    for pid, grp in df.groupby("prompt_hash"):
        a, b, Va, Vb = ols_with_V(
            grp.rm_score.to_numpy(),
            grp.helpfulness.to_numpy().astype(float),
        )
        cluster_stats.append({
            "cluster_id": pid, "alpha": a, "beta": b,
            "V_alpha": Va, "V_beta": Vb, "n": len(grp),
        })
    us = pd.DataFrame(cluster_stats)
    V_alpha_total = float(us.alpha.var())
    V_beta_total = float(us.beta.var())
    mean_samp_V_alpha = float(
        us.V_alpha.replace([np.inf, -np.inf], np.nan).dropna().mean()
    )
    mean_samp_V_beta = float(
        us.V_beta.replace([np.inf, -np.inf], np.nan).dropna().mean()
    )
    tau_a_sq = max(V_alpha_total - mean_samp_V_alpha, 1e-6)
    tau_b_sq = max(V_beta_total - mean_samp_V_beta, 1e-6)
    log(f"[stage B] tau_alpha_sq={tau_a_sq:.4f}, tau_beta_sq={tau_b_sq:.6f}")

    # K-fold within-cluster CV: 4 arms
    sq_err_no_calib_per_row = []
    sq_err_baseline_per_row = []
    sq_err_per_user_ols_per_row = []
    sq_err_pebs_per_row = []
    per_row_cluster = []
    rng2 = np.random.default_rng(seed)

    for pid, grp in df.groupby("prompt_hash"):
        n = len(grp)
        if n < min_obs_per_cluster:
            continue
        x = grp.rm_score.to_numpy()
        y = grp.helpfulness.to_numpy().astype(float)
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
            sq_err_pebs_per_row.extend(((y_hat_sh - y_te) ** 2).tolist())
            per_row_cluster.extend([pid] * len(y_te))

    sq_err_no_calib_per_row = np.asarray(sq_err_no_calib_per_row)
    sq_err_baseline_per_row = np.asarray(sq_err_baseline_per_row)
    sq_err_per_user_ols_per_row = np.asarray(sq_err_per_user_ols_per_row)
    sq_err_pebs_per_row = np.asarray(sq_err_pebs_per_row)

    rmse_no_calib = float(np.sqrt(sq_err_no_calib_per_row.mean()))
    rmse_baseline = float(np.sqrt(sq_err_baseline_per_row.mean()))
    rmse_per_user_ols = float(np.sqrt(sq_err_per_user_ols_per_row.mean()))
    rmse_pebs = float(np.sqrt(sq_err_pebs_per_row.mean()))
    gain_pct = ((rmse_baseline - rmse_pebs) / rmse_baseline * 100.0
                if rmse_baseline > 0 else float("nan"))

    log(f"[stage B] rmse_no_calib   = {rmse_no_calib:.5f}")
    log(f"[stage B] rmse_baseline    = {rmse_baseline:.5f}")
    log(f"[stage B] rmse_per_user_ols= {rmse_per_user_ols:.5f}")
    log(f"[stage B] rmse_pebs       = {rmse_pebs:.5f}")
    log(f"[stage B] gain_pct         = {gain_pct:+.3f}%")

    # Cluster-bootstrap CI on gain_pct (cluster by prompt_hash)
    log(f"[stage B] cluster-bootstrap B={bootstrap_B} ...")
    bs = cluster_bootstrap_gain_ci(
        sq_err_baseline_per_row, sq_err_pebs_per_row, per_row_cluster,
        B=bootstrap_B, seed=bootstrap_seed,
    )
    log(f"[stage B] BCa CI95: [{bs['ci_bca_lo']:+.3f}, {bs['ci_bca_hi']:+.3f}]")

    # Per-cluster diagnostics
    diag = {
        "n_clusters_post_filter": n_clusters_post_filter,
        "n_clusters_total": n_clusters_total,
        "n_obs_total": n_obs_total,
        "n_obs_post_filter": int(len(df)),
        "n_degenerate_x_clusters": n_degenerate_x,
        "tau_alpha_sq": tau_a_sq,
        "tau_beta_sq": tau_b_sq,
        "sigma_alpha": float(np.sqrt(tau_a_sq)),
        "sigma_beta": float(np.sqrt(tau_b_sq)),
        "pop_alpha": pop_alpha,
        "pop_beta": pop_beta,
        "per_cluster_alpha_mean": float(us.alpha.mean()),
        "per_cluster_alpha_std": float(us.alpha.std(ddof=1)),
        "per_cluster_beta_mean": float(us.beta.mean()),
        "per_cluster_beta_std": float(us.beta.std(ddof=1)),
    }

    return {
        "rmse_no_calib": rmse_no_calib,
        "rmse_baseline_pop": rmse_baseline,
        "rmse_per_user_OLS": rmse_per_user_ols,
        "rmse_pebs_shrunk": rmse_pebs,
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
        "per_cluster_table": us.to_dict(orient="records"),
        **diag,
    }


# ============================================================================
# Outcome classification
# ============================================================================


def assign_verdict_class(
    gain: float, ci_lo: float, ci_hi: float, n_clusters: int,
) -> str:
    """Pre-specified outcome classification.

    CONFIRMED-HELPSTEER2-CONFIRMS-PRISM            gain >= +3 + CI excludes 0
                                                      + n_clusters >= 500
    PARTIAL-HELPSTEER2-PARTIAL-CONFIRMS              gain in [+1, +3] + CI excludes 0
    TENTATIVE-INCONCLUSIVE-DATASET-SCHEMA-LIMIT     CI straddles zero
    REJECTED-HELPSTEER2-DOES-NOT-REPLICATE           CI strictly negative
    """
    if not (np.isfinite(gain) and np.isfinite(ci_lo) and np.isfinite(ci_hi)):
        return "INCONCLUSIVE-NaN"

    excludes_zero_pos = ci_lo > 0
    excludes_zero_neg = ci_hi < 0

    if excludes_zero_neg:
        return "REJECTED-HELPSTEER2-DOES-NOT-REPLICATE"
    if (gain >= ESTABLISHED_GAIN_LO
            and excludes_zero_pos
            and n_clusters >= N_CLUSTERS_FOR_ESTABLISHED):
        return "CONFIRMED-HELPSTEER2-CONFIRMS-PRISM"
    if (gain >= MODERATE_GAIN_LO
            and gain < ESTABLISHED_GAIN_LO
            and excludes_zero_pos):
        return "PARTIAL-HELPSTEER2-PARTIAL-CONFIRMS"
    if (gain >= ESTABLISHED_GAIN_LO
            and excludes_zero_pos
            and n_clusters < N_CLUSTERS_FOR_ESTABLISHED):
        return "PARTIAL-HELPSTEER2-PARTIAL-CONFIRMS-COHORT-BELOW-500"
    return "TENTATIVE-INCONCLUSIVE-DATASET-SCHEMA-LIMIT"


# ============================================================================
# Driver
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--disagreements-jsonl-gz",
        default="/workspace/.hf_cache/hub/datasets--nvidia--HelpSteer2/snapshots/990b2711a36180dd19d9c94b8627844866f8982a/disagreements/disagreements.jsonl.gz",
        help="Path to HelpSteer2 disagreements config jsonl.gz on the GPU host.",
    )
    p.add_argument(
        "--scored-parquet",
        default=str(T1 / "data" / "helpsteer2"
                    / "helpsteer2_qwen7b_disagreements_scored.parquet"),
    )
    p.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--bootstrap-seed", type=int, default=BOOT_SEED)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--k-folds", type=int, default=K_FOLDS)
    p.add_argument("--min-obs-per-cluster", type=int, default=MIN_OBS_PER_CLUSTER)
    p.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS)
    p.add_argument("--max-response-tokens", type=int, default=MAX_RESPONSE_TOKENS)
    p.add_argument("--no-quantize-4bit", action="store_true",
                   help="Disable 4-bit nf4 quantization (default: enabled).")
    p.add_argument("--force-rescore", action="store_true",
                   help="Re-run Stage A even if scored parquet exists.")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke-test mode: cap Stage A at --smoke-limit unique pairs.")
    p.add_argument("--smoke-limit", type=int, default=50)
    p.add_argument("--output-dir", default=str(OUT_DIR))
    return p.parse_args()


def stage_a_needs_rerun(parquet: Path, force: bool, min_rows: int) -> bool:
    """Re-run Stage A if --force-rescore OR parquet missing OR too small."""
    if force:
        return True
    if not parquet.exists():
        return True
    try:
        n = len(pd.read_parquet(parquet, columns=["prompt_hash"]))
        return n < min_rows
    except Exception:
        return True


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    summary: dict[str, Any] = {
        "experiment_id": "Q1_HelpSteer2_replication",
        "verdict_class": "PENDING",
        "args": vars(args),
        "seed": int(args.seed),
        "bootstrap_seed": int(args.bootstrap_seed),
        "n_boot": int(args.n_boot),
        "k_folds": int(args.k_folds),
        "min_obs_per_cluster": int(args.min_obs_per_cluster),
        "model_id": args.model_id,
        "honest_disclosure_design_vs_data_mismatch": (
            "The original design assumed an 'annotator_id field' exists in HelpSteer2. "
            "It does not (NVIDIA anonymized annotators across rows; Wang et al. "
            "2024 Section 3.4 explicitly states list-position is not a stable cross-row "
            "annotator identifier). Closest defensible PEBS analog: "
            "cluster_id=prompt_hash with (response_idx, annotator_slot) "
            "observations within cluster. n_obs_per_cluster ~6 vs PRISM's median-50 "
            "-> Morris (1983) eq. 2.9 predicts omega_j -> 0 -> PEBS approx pop-slope "
            "-> gain magnitude bounded downward. CONFIRMED outcome (>=3pp + CI "
            "excludes 0 + n_clusters >= 500) would be a positive surprise; "
            "PARTIAL-PARTIAL (1-3pp + CI excludes 0) is the expected best case; "
            "TENTATIVE-INCONCLUSIVE is the default expectation; REJECTED would "
            "scope-limit the paper claim to per-rater corpora."
        ),
        "anomaly_branches_fired": [],
        "stages": {},
    }

    log(f"[Q1 HelpSteer2] starting at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    log(f"[Q1 HelpSteer2] output_dir: {out_dir}")
    log(f"[Q1 HelpSteer2] git_head_t3: {git_head_t3()}")
    log(f"[Q1 HelpSteer2] disagreements_jsonl_gz: {args.disagreements_jsonl_gz}")
    log(f"[Q1 HelpSteer2] scored_parquet: {args.scored_parquet}")

    # ------------------------------------------------------------------
    # Load disagreements config
    # ------------------------------------------------------------------
    if not Path(args.disagreements_jsonl_gz).exists():
        # Auto-download via HF if missing
        try:
            from huggingface_hub import hf_hub_download
            log(f"[Q1] disagreements not at {args.disagreements_jsonl_gz}; downloading via HF")
            local = hf_hub_download(
                "nvidia/HelpSteer2",
                "disagreements/disagreements.jsonl.gz",
                repo_type="dataset",
            )
            log(f"[Q1] downloaded to {local}")
            args.disagreements_jsonl_gz = local
        except Exception as exc:
            summary["verdict_class"] = "RUNTIME-ERROR-DISAGREEMENTS-MISSING"
            summary["error"] = traceback.format_exc()
            summary_path.write_text(json.dumps(summary, indent=2))
            raise

    log(f"[Q1] loading disagreements from {args.disagreements_jsonl_gz}")
    disagreements = _load_disagreements_jsonl(args.disagreements_jsonl_gz)
    log(f"[Q1] {len(disagreements)} disagreement rows")

    obs_df = _build_observation_rows(disagreements)
    log(f"[Q1] flattened to {len(obs_df)} observation rows "
        f"({obs_df['prompt_hash'].nunique()} unique prompts; "
        f"{obs_df.drop_duplicates(['prompt_hash','response_hash']).shape[0]} unique pairs)")

    # ------------------------------------------------------------------
    # Stage A - score with Qwen2.5-7B-Instruct (skippable cache)
    # ------------------------------------------------------------------
    parquet = Path(args.scored_parquet)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    if stage_a_needs_rerun(parquet, args.force_rescore,
                            MIN_SCORED_ROWS_FOR_VALID_CACHE):
        log(f"[Stage A] running (parquet missing / too small / forced)")
        try:
            stage_a_meta = stage_a_score_with_qwen7b(
                obs_df, parquet,
                model_id=args.model_id,
                max_prompt_tokens=args.max_prompt_tokens,
                max_response_tokens=args.max_response_tokens,
                quantize_4bit=not args.no_quantize_4bit,
                smoke_limit=args.smoke_limit if args.smoke else None,
            )
            summary["stages"]["stage_a"] = stage_a_meta
        except Exception as exc:
            summary["verdict_class"] = "RUNTIME-ERROR-STAGE-A"
            summary["error"] = traceback.format_exc()
            summary_path.write_text(json.dumps(summary, indent=2))
            raise
    else:
        n = len(pd.read_parquet(parquet, columns=["prompt_hash"]))
        summary["stages"]["stage_a"] = {
            "elapsed_seconds": 0.0,
            "cache_reused": True,
            "scored_parquet": str(parquet),
            "scored_sha256": file_sha256(parquet),
            "scored_n_unique_pairs": int(n),
        }
        log(f"[Stage A] CACHE REUSE: {parquet} ({n} pairs; "
            f"sha256={summary['stages']['stage_a']['scored_sha256'][:12]}...)")

    # Merge scored RM scores back into obs_df
    scored_df = pd.read_parquet(parquet)
    obs_df = obs_df.merge(scored_df, on=["prompt_hash", "response_hash"], how="inner")
    log(f"[Q1] post-merge: {len(obs_df)} observation rows with rm_score")

    # ------------------------------------------------------------------
    # Stage B - PEBS per-prompt-cluster shrinkage
    # ------------------------------------------------------------------
    try:
        stage_b_result = stage_b_pebs_per_prompt(
            obs_df,
            seed=args.seed,
            k_folds=args.k_folds,
            min_obs_per_cluster=args.min_obs_per_cluster,
            bootstrap_B=args.n_boot,
            bootstrap_seed=args.bootstrap_seed,
        )
        summary["stages"]["stage_b"] = stage_b_result
    except Exception as exc:
        summary["verdict_class"] = "RUNTIME-ERROR-STAGE-B"
        summary["error"] = traceback.format_exc()
        summary_path.write_text(json.dumps(summary, indent=2))
        raise

    # ------------------------------------------------------------------
    # Stage B -> outcome classification
    # ------------------------------------------------------------------
    gain = float(stage_b_result.get("gain_pct", float("nan")))
    ci_lo = float(stage_b_result.get("gain_ci_bca_lo", float("nan")))
    ci_hi = float(stage_b_result.get("gain_ci_bca_hi", float("nan")))
    n_clusters_post_filter = int(stage_b_result.get("n_clusters_post_filter", 0))
    n_obs_total = int(stage_b_result.get("n_obs_total", 0))

    verdict = assign_verdict_class(gain, ci_lo, ci_hi, n_clusters_post_filter)

    anomalies = []
    if not (gain == gain):
        anomalies.append("gain_is_nan")
    if ci_hi < ci_lo:
        anomalies.append("ci_bounds_swapped")
    if n_clusters_post_filter < 100:
        anomalies.append("n_clusters_below_minimum_100")
    if n_clusters_post_filter < N_CLUSTERS_FOR_ESTABLISHED:
        anomalies.append(f"n_clusters_below_established_threshold_{N_CLUSTERS_FOR_ESTABLISHED}")
    if stage_b_result.get("n_degenerate_x_clusters", 0) > 0.1 * n_clusters_post_filter:
        anomalies.append("degenerate_x_in_>10pct_clusters")

    summary.update({
        "verdict_class": verdict,
        "gain_pct": gain,
        "gain_ci_bca_lo": ci_lo,
        "gain_ci_bca_hi": ci_hi,
        "n_clusters_post_filter": n_clusters_post_filter,
        "n_obs_total": n_obs_total,
        "rmse_baseline_pop": float(stage_b_result.get("rmse_baseline_pop", float("nan"))),
        "rmse_pebs_shrunk": float(stage_b_result.get("rmse_pebs_shrunk", float("nan"))),
        "stage_b_full_output": stage_b_result,
        "anomaly_branches_fired": anomalies,
        "git_head_t3_at_run": git_head_t3(),
        "completion_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    # Save per-cluster table separately for downstream analysis
    per_cluster_path = out_dir / "per_cluster.parquet"
    if isinstance(stage_b_result.get("per_cluster_table"), list):
        per_cluster_df = pd.DataFrame(stage_b_result["per_cluster_table"])
        per_cluster_df.to_parquet(per_cluster_path, index=False)
        # Strip per-cluster from in-memory summary (already on disk)
        summary["stage_b_full_output"].pop("per_cluster_table", None)
        summary["per_cluster_parquet"] = str(per_cluster_path)

    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print()
    print(f"=== Q1 HelpSteer2 Cross-Corpus Replication ===")
    print(f"verdict_class                   : {verdict}")
    print(f"n_clusters_post_filter / n_obs  : {n_clusters_post_filter} / {n_obs_total}")
    print(f"rmse_pop / rmse_pebs_shrunk    : "
          f"{summary['rmse_baseline_pop']:.5f} / {summary['rmse_pebs_shrunk']:.5f}")
    print(f"gain_pct                        : {gain:+.3f}%  CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"anomalies_fired                 : {anomalies}")
    print(f"summary_path                    : {summary_path}")
    print(f"per_cluster_path                : {per_cluster_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sp = OUT_DIR / "summary.json"
        if not sp.exists():
            sp.write_text(json.dumps({
                "experiment_id": "Q1_HelpSteer2_replication",
                "verdict_class": "RUNTIME-ERROR",
                "error": traceback.format_exc(),
            }, indent=2))
        sys.exit(2)
