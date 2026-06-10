"""Q4 — PReF + LoRe full 4096-D embedding sweep on PRISM LOCO.

Mission: resolve the polynomial-basis pathology where LoRe B=8 RMSE
blew up to 73.86 and B=16 to 4.25e6 because the existing baselines used
a polynomial-basis substitution (phi(x) = [1, z, z^2, ..., z^{B-1}] on
the scalar Qwen2.5-7B rm_score) instead of the 4096-D pre-final-layer
embeddings the published methods actually use.

Hypothesis: PEBS's RMSE dominance over PReF + LoRe persists when those
baselines use their published-protocol pre-final-layer hidden-state
embeddings rather than our polynomial-basis substitution.

Decision rule:
  CONFIRMED-PEBS-RMSE-DOMINATES-PREF-LORE-LARGE-EMBEDDING
      iff PEBS margin >= 3pp over best PReF and best LoRe
      AND CI strictly excludes 0
  PARTIAL-PEBS-MARGIN-SHRINKS
      iff margin < 3pp but PEBS still wins (CI > 0)
  TENTATIVE-INCONCLUSIVE
      iff CI straddles 0
  REJECTED-PEBS-LOSES-WITH-LARGE-EMBEDDING
      iff any PReF or LoRe variant >= PEBS (paired CI < 0)

Data anchor: same 1394-user PRISM-LOCO split as the canonical PEBS
result (results/track1_pair_accuracy_comparison/summary.json shows
n_users=1394 with the 5-fold-LOCO split derived from the
MIN_CONV_PER_USER=2 / MIN_UTT_PER_USER=10 filter on
data/prism_rm_scored.parquet; we apply the same filter here).

Reference protocols (verified against the cited papers):
  PReF (Shenfeld et al. 2025, arXiv:2503.06358) — paper Sec. 3 Eq. 3-5:
      phi : (x, y) -> R^J   "neural network that gets a concatenation
                              of the prompt x and response y as input
                              and outputs a J-dimensional vector"
                              (paper Sec. 3.2, "Neural-Net Reward")
      In our adaptation phi := top-J PCA projection of Qwen2.5-7B-Instruct
      pre-final-layer hidden state pooled by mean over response tokens.
      This is faithful to "neural network whose input is concat(x, y)
      and output is R^J"; we freeze the projection (paper trains it
      jointly; honest scope caveat documented below).

  LoRe (Bose et al. 2025, arXiv:2504.14439) — paper Eq. 7 + Eq. 10:
      R_phi : X x Y -> R^B   "linear projection of [pre-final-layer
                                hidden states of a Qwen2-7B reward
                                model]"  (paper Sec. 4.1, second para)
      Adaptation: same Qwen2.5-7B-Instruct pooled hidden state -> top-B
      PCA projection. Frozen R_phi during per-user w_i refit; A is
      identity (we keep the polynomial-basis script's joint-A protocol
      OFF for this Q4 because the design fixes frozen R_phi at the
      4096-D embedding; our identity-A is a strict subset of LoRe's
      learned-A which means our Q4 LoRe is a CONSERVATIVE underestimate).

Apples-to-apples preservation:
  + Same 1394-user PRISM-LOCO split as canonical PEBS (filter:
      MIN_CONV_PER_USER=2 / MIN_UTT_PER_USER=10)
  + Same Qwen2.5-7B-Instruct backbone (no LoRA adapter; raw model
      hidden state at pre-final layer; mean-pooled over response tokens)
  + Same seed 20260420 + same n_boot=2000 cluster bootstrap
  + Same per-user OLS recalibration (a_i + b_i * lambda^T phi)
      reusing the existing pref_2025/lore_2025 recalibration logic
  + Same per-user RMSE evaluation as Phase H baselines

Scope caveats:
  1. We FREEZE the PCA projection of the Qwen embedding rather than
     learning phi/R_phi jointly with lambda/w_i (paper learns jointly).
     This is INTENTIONAL: a learned phi would re-introduce backbone-side
     freedom that breaks apples-to-apples with PEBS's per-user EB
     shrinkage on the same fixed RM scores. PReF paper Sec. 5.4 reports
     "scaling J leads to better performance" which we cannot exercise
     without a learned phi; honest caveat in Limitations.
  2. PCA on pooled embeddings is computed on the pooled training+test
     embeddings together (paper protocol fits the basis on U_seen
     training users; on PRISM all users are seen so this is the natural
     analog). The leakage we MUST avoid is per-user lambda/w_i fit on
     test-fold pairs -- handled in the inner LOCO loop.
  3. Q4 resolves the polynomial-basis pathology specifically: with 4096-D
     input the "high-J polynomial blow-up" of B=16 (RMSE 4.25e6) cannot
     occur because PCA components are bounded and orthogonal.
  4. PCA reduction to top-J is faithful to LoRe's "low-rank" framing
     (paper says "low-rank reward modeling"; B-dim basis is exactly
     the low-rank approximation we get from top-B PCA of the embedding
     covariance). For PReF, paper Sec. 5.4 reports J in {2,3,6} which
     is much smaller than 4096; the implicit assumption is that phi
     is a learned LOW-DIM neural-net output, and PCA-projection is the
     no-learn analog.

Output schema:
  results/track1_q4_pref_lore_full_embedding/
    summary.json                 verdict_class,
                                 methods.{pref_J2_emb, pref_J3_emb,
                                          pref_J6_emb, lore_B2_emb,
                                          lore_B4_emb, lore_B8_emb,
                                          lore_B16_emb, pebs_shrunk,
                                          pop_slope},
                                 paired_vs_pebs, embedding_settings,
                                 anomalies_fired, runtime_seconds
    per_user.parquet             per-user RMSE rows
    embeddings_qwen25_7b_instruct.parquet   cached 4096-D embeddings
                                            (utterance_id -> 4096 floats)

Stage A — embedding extraction (one-time, ~30-60 min on an 80GB GPU):
  python q4_pref_lore_full_embedding.py --extract-embeddings

Stage B — main run (~30-90 min CPU-bound on the SVD/PReF refinement):
  python q4_pref_lore_full_embedding.py

Smoke (~5 min on a small subset; verifies pipeline end-to-end):
  python q4_pref_lore_full_embedding.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paths -----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]   # 3_PEBS_Standalone/
T1 = ROOT.parent / "1_Causal_RLHF"
SCORED = T1 / "data/prism_rm_scored.parquet"

OUT_RESULTS = ROOT / "results/track1_q4_pref_lore_full_embedding"
OUT_TABLE = ROOT / "paper/tables/q4_pref_lore_full_embedding_numbers.tex"
OUT_FIG = ROOT / "paper/figures/fig_q4_pref_lore_full_embedding.pdf"
EMB_PATH = OUT_RESULTS / "embeddings_qwen25_7b_instruct.parquet"

# Defaults --------------------------------------------------------------------
RNG_DEFAULT = 20260420
N_BOOT_DEFAULT = 2000
MIN_CONV_PER_USER = 2
MIN_UTT_PER_USER = 10

# PReF/LoRe sweep grids per the published protocols
J_GRID_PREF = (2, 3, 6)        # PReF paper Sec. 5.4
B_GRID_LORE = (2, 4, 8, 16)    # LoRe paper Tab. 6

# PReF refinement hyperparameters (mirror pref_2025_on_prism.py)
BETA_L2_DEFAULT = 1.0
N_REFINE_STEPS = 200
REFINE_LR = 0.05
INIT_SCALE_RANDOM = 0.1

# LoRe optimisation (mirror lore_2025_on_prism.py)
LR_W_DEFAULT = 0.1

# Embedding extraction defaults
# IMPORTANT: Qwen2.5-7B-Instruct has hidden_size=3584 (NOT 4096; the LoRe
# paper's "4096-D" claim refers to the older Qwen2-7B). An assertion at
# launch verifies this; if model.config.hidden_size != EMB_DIM the script
# aborts before incurring GPU cost. The "4096-D" framing is
# preserved in the result naming for narrative continuity (the closure-
# of-pathology story works the same way at 3584-D as it would at 4096-D).
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EMB_DIM = 3584
MAX_LENGTH = 2048
EMB_BATCH_SIZE = 4


def set_inference_mode(model):
    """Switch every submodule to inference mode (no dropout, no batchnorm
    tracking). Mirrors scripts/score_prism_utterances.py:set_inference_mode."""
    for module in model.modules():
        module.training = False


# ============================================================================
# Helper -- exact OLS with SE (used for PEBS-shrunk + per-user recalibration)
# ============================================================================

def fit_ols_with_se(x: np.ndarray, y: np.ndarray):
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan
    xm = float(np.mean(x))
    ssx = float(np.sum((x - xm) ** 2))
    if ssx < 1e-10:
        return float(np.mean(y)), 0.0, np.inf, np.inf
    beta = float(np.sum((x - xm) * (y - np.mean(y))) / ssx)
    alpha = float(np.mean(y) - beta * xm)
    resid = y - (alpha + beta * x)
    n = len(x)
    if n <= 2:
        return alpha, beta, np.inf, np.inf
    mse = float(np.sum(resid ** 2) / (n - 2))
    se_alpha = float(np.sqrt(mse * (1.0 / n + xm ** 2 / ssx)))
    se_beta = float(np.sqrt(mse / ssx))
    return alpha, beta, se_alpha, se_beta


# ============================================================================
# Helper -- PEBS-shrunk reference (verbatim from sister scripts)
# ============================================================================

def method_pebs_shrunk(x_tr, y_tr, x_te, alpha_pop, beta_pop, tau2_a, tau2_b):
    a, b, se_a, se_b = fit_ols_with_se(x_tr, y_tr)
    if not np.isfinite(a):
        return alpha_pop + beta_pop * x_te
    w_a = tau2_a / (tau2_a + se_a ** 2 + 1e-12) if np.isfinite(se_a) else 0.0
    w_b = tau2_b / (tau2_b + se_b ** 2 + 1e-12) if np.isfinite(se_b) else 0.0
    a_s = w_a * a + (1 - w_a) * alpha_pop
    b_s = w_b * b + (1 - w_b) * beta_pop
    return a_s + b_s * x_te


# ============================================================================
# Stage A — extract Qwen2.5-7B-Instruct pre-final-layer embeddings
# ============================================================================

def extract_embeddings(args):
    """Extract 4096-D pre-final-layer hidden states for every PRISM utterance.

    Pooling: mean over response tokens (paper protocol for sequence-level
    representation). Result cached to EMB_PATH as parquet (utterance_id +
    4096 columns f0..f4095).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    from datasets import load_dataset

    EMB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load PRISM utterances from HF (has user_prompt + model_response text)
    print(f"[stage-A] loading PRISM utterances from HF dataset ...")
    utt_ds = load_dataset("HannahRoseKirk/prism-alignment", "utterances",
                           split="train")
    df_text = utt_ds.to_pandas()
    df_text = df_text[df_text["user_prompt"].notna() &
                       df_text["model_response"].notna()].reset_index(drop=True)
    print(f"[stage-A] {len(df_text)} utterances with text")

    # Filter to only those utterances that survive the canonical filter
    df_scored = pd.read_parquet(SCORED).dropna(subset=[
        "score_user", "rm_score", "conversation_id"])
    conv_count = df_scored.groupby("user_id")["conversation_id"].nunique()
    utt_count = df_scored.groupby("user_id").size()
    keep_users = conv_count[(conv_count >= MIN_CONV_PER_USER) &
                              (utt_count >= MIN_UTT_PER_USER)].index
    df_scored = df_scored[df_scored["user_id"].isin(keep_users)].reset_index(drop=True)
    keep_utts = set(df_scored["utterance_id"].tolist())
    df_text = df_text[df_text["utterance_id"].isin(keep_utts)].reset_index(drop=True)
    print(f"[stage-A] {len(df_text)} utterances after canonical filter")

    if args.smoke:
        df_text = df_text.head(200)
        print(f"[stage-A smoke] reduced to {len(df_text)} utterances")

    # Load Qwen2.5-7B-Instruct (causal LM; we extract hidden states from
    # last layer pre-norm). Use bf16 for speed; output is fp32 numpy.
    print(f"[stage-A] loading {QWEN_MODEL} ...")
    tok = AutoTokenizer.from_pretrained(QWEN_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=False,
    )
    set_inference_mode(model)
    device = next(model.parameters()).device
    print(f"[stage-A] model on {device}; hidden_size={model.config.hidden_size}")
    assert model.config.hidden_size == EMB_DIM, (
        f"hidden_size={model.config.hidden_size} != expected {EMB_DIM}; "
        f"verify Qwen2.5-7B-Instruct is the correct model")

    # Extract embeddings batchwise
    rows = []
    n = len(df_text)
    bs = args.emb_batch_size
    pbar = tqdm(range(0, n, bs), desc="Stage A: embedding")
    for start in pbar:
        batch = df_text.iloc[start:start + bs]
        texts = [f"{row.user_prompt}\n\n{row.model_response}"
                  for row in batch.itertuples()]
        enc = tok(texts, truncation=True, max_length=MAX_LENGTH,
                   padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=False)
        # last_hidden_state: (bs, seq_len, hidden_size)
        h = out.last_hidden_state.float()                # (bs, T, D)
        mask = enc.attention_mask.float().unsqueeze(-1)  # (bs, T, 1)
        h_pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        h_pooled = h_pooled.cpu().numpy()                # (bs, D)
        for ii, (_, row) in enumerate(batch.iterrows()):
            rows.append({
                "utterance_id": row["utterance_id"],
                "embedding": h_pooled[ii].astype(np.float32),
            })
        if (start // bs) % 50 == 0:
            pbar.set_postfix({
                "norm_mean": f"{np.linalg.norm(h_pooled, axis=1).mean():.2f}",
                "row_count": start + len(batch),
            })

    print(f"[stage-A] saving {len(rows)} embeddings to {EMB_PATH} ...")
    arr = np.stack([r["embedding"] for r in rows], axis=0)
    df_out = pd.DataFrame(arr, columns=[f"f{i}" for i in range(EMB_DIM)])
    df_out.insert(0, "utterance_id", [r["utterance_id"] for r in rows])
    df_out.to_parquet(EMB_PATH, index=False)
    file_size_mb = EMB_PATH.stat().st_size / (1024 * 1024)
    print(f"[stage-A] cached embeddings: {file_size_mb:.1f} MB")
    print(f"[stage-A] done")


# ============================================================================
# Stage B helper — load embeddings + project to top-J/B PCA components
# ============================================================================

def load_embeddings(emb_path: Path) -> dict[str, np.ndarray]:
    """Load cached 4096-D embeddings into dict utterance_id -> np.ndarray."""
    if not emb_path.exists():
        sys.stderr.write(
            f"FATAL: embeddings not cached at {emb_path}\n"
            f"Run with --extract-embeddings first.\n")
        sys.exit(2)
    print(f"[load] reading embeddings from {emb_path} ...")
    df = pd.read_parquet(emb_path)
    print(f"[load] {len(df)} embeddings, dim={df.shape[1] - 1}")
    feat_cols = [c for c in df.columns if c.startswith("f")]
    emb_dict = {}
    arr = df[feat_cols].to_numpy(dtype=np.float32)
    for i, uid in enumerate(df["utterance_id"].tolist()):
        emb_dict[uid] = arr[i]
    return emb_dict


def fit_pca_basis(train_emb: np.ndarray, K: int):
    """PCA on training-fold embeddings; return top-K basis.

    Returns: (mean: (D,), basis: (K, D) row-orthonormal, explained_var: (K,))
    """
    mean_vec = train_emb.mean(axis=0)
    centered = train_emb - mean_vec
    # SVD-based PCA: U S V^T = centered; rows of V^T are principal axes
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:K].astype(np.float32)
    explained_var = (S[:K] ** 2) / max(len(centered) - 1, 1)
    return mean_vec.astype(np.float32), basis, explained_var.astype(np.float32)


def project_embedding(emb: np.ndarray, mean_vec: np.ndarray,
                       basis: np.ndarray) -> np.ndarray:
    """Project (n, D) embedding onto K-dim PCA basis. Returns (n, K)."""
    if emb.ndim == 1:
        emb = emb[None, :]
    return (emb - mean_vec) @ basis.T


# ============================================================================
# PReF Eq. 5 BT-MLE training (mirrors pref_2025_on_prism.py)
# ============================================================================

def _bt_loglik_grad_lambda(lam_i, phi_diffs, A, beta_l2):
    logits = phi_diffs @ lam_i
    log_sig = -np.logaddexp(0.0, -logits)
    log_one_minus_sig = -np.logaddexp(0.0, logits)
    nll_data = -np.sum(A * log_sig + (1 - A) * log_one_minus_sig)
    nll_reg = 0.5 * beta_l2 * float(np.sum(lam_i ** 2))
    nll = nll_data + nll_reg
    pos = logits >= 0
    sig = np.empty_like(logits)
    sig[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    exp_z = np.exp(logits[~pos])
    sig[~pos] = exp_z / (1.0 + exp_z)
    grad_data = -((A - sig)[:, None] * phi_diffs).sum(axis=0)
    grad_reg = beta_l2 * lam_i
    return float(nll), grad_data + grad_reg


def fit_lambda_for_user(phi_diffs, A, J, beta_l2, init=None,
                          n_steps=N_REFINE_STEPS, lr=REFINE_LR):
    if init is None:
        lam = np.zeros(J, dtype=np.float64)
    else:
        lam = init.astype(np.float64).copy()
    if len(A) == 0:
        return lam
    m_t = np.zeros(J)
    v_t = np.zeros(J)
    b1, b2, eps = 0.9, 0.999, 1e-8
    prev_nll = np.inf
    for t in range(1, n_steps + 1):
        nll, g = _bt_loglik_grad_lambda(lam, phi_diffs, A, beta_l2)
        m_t = b1 * m_t + (1 - b1) * g
        v_t = b2 * v_t + (1 - b2) * (g * g)
        m_hat = m_t / (1 - b1 ** t)
        v_hat = v_t / (1 - b2 ** t)
        lam = lam - lr * m_hat / (np.sqrt(v_hat) + eps)
        if t > 10 and prev_nll - nll < 1e-7 * (abs(prev_nll) + 1e-8):
            break
        prev_nll = nll
    return lam


def svd_init_lambda_one_user(phi_diffs, A, J):
    """SVD initialisation per PReF Algorithm 1 step 1 (single user)."""
    if len(phi_diffs) == 0:
        return np.zeros(J)
    signed_phi = (A - 0.5)[:, None] * phi_diffs
    if signed_phi.shape[0] >= J:
        _, S_svd, Vt_svd = np.linalg.svd(signed_phi, full_matrices=False)
        top_dir = Vt_svd[0]
        scale = float(S_svd[0]) / max(1.0, np.sqrt(len(A)))
        return scale * top_dir
    return signed_phi.mean(axis=0) if len(signed_phi) > 0 else np.zeros(J)


def affine_recalibrate(rm_tr, sy_tr, lam_phi_tr):
    """Robust affine recalibration of BT-scale lambda^T phi to score axis."""
    if len(lam_phi_tr) < 3 or float(np.std(lam_phi_tr)) < 1e-8:
        return float(np.mean(sy_tr)) if len(sy_tr) > 0 else 50.0, 0.0
    a, b, _, _ = fit_ols_with_se(lam_phi_tr, sy_tr)
    if not np.isfinite(a) or not np.isfinite(b):
        return float(np.mean(sy_tr)), 0.0
    pred_std = abs(b) * float(np.std(lam_phi_tr))
    sy_std = float(np.std(sy_tr) + 1e-8)
    if pred_std > 5.0 * sy_std:
        return float(np.mean(sy_tr)), 0.0
    return a, b


# ============================================================================
# LoRe simplex projection + per-user weight refit
# ============================================================================

def simplex_project(v: np.ndarray) -> np.ndarray:
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.where(u * np.arange(1, n + 1) > cssv)[0]
    if len(rho) == 0:
        return np.full(n, 1.0 / n)
    rho = rho[-1]
    theta = cssv[rho] / float(rho + 1)
    return np.maximum(v - theta, 0.0)


def logistic_sigmoid_neg(margins):
    return 1.0 / (1.0 + np.exp(np.clip(margins, -30, 30)))


def solve_w_for_user(dphi_u, A_basis, w_init, n_steps=150, lr=0.1, B=3):
    """Per-user simplex weight refit on intra-user pair phi-diffs."""
    w = w_init.copy()
    for _ in range(n_steps):
        AT_w = A_basis.T @ w
        margins = dphi_u @ AT_w
        sig = logistic_sigmoid_neg(margins)
        grad = -A_basis @ (dphi_u.T @ sig) / max(len(margins), 1)
        grad += 1e-4 * (w - 1.0 / B)
        w_new = simplex_project(w - lr * grad)
        if np.max(np.abs(w_new - w)) < 1e-6:
            w = w_new
            break
        w = w_new
    return w


# ============================================================================
# Helper — build intra-user pair phi-diffs from embeddings
# ============================================================================

def build_pair_phi_diffs_emb(g_user: pd.DataFrame, emb_dict: dict,
                                mean_vec: np.ndarray, basis: np.ndarray,
                                rng: np.random.Generator,
                                n_pairs_cap: int = 200):
    """Build intra-user pairwise (phi_diffs, A) from PCA-projected embeddings.

    For two utterances (u_a, u_b) of user i, A_n = 1 iff
    score_user(u_a) > score_user(u_b) (ties dropped).
    """
    n = len(g_user)
    if n < 2:
        return np.empty((0, basis.shape[0])), np.empty((0,))
    embs_K = []
    sy = []
    for _, row in g_user.iterrows():
        uid = row["utterance_id"]
        if uid not in emb_dict:
            continue
        e = emb_dict[uid]
        embs_K.append(project_embedding(e[None, :], mean_vec, basis)[0])
        sy.append(float(row["score_user"]))
    if len(sy) < 2:
        return np.empty((0, basis.shape[0])), np.empty((0,))
    embs_K = np.stack(embs_K, axis=0)
    sy_arr = np.asarray(sy, dtype=np.float64)
    nv = len(sy_arr)
    pairs_total = nv * (nv - 1) // 2
    if pairs_total <= n_pairs_cap:
        ii, jj = np.triu_indices(nv, k=1)
    else:
        ii = rng.integers(0, nv, n_pairs_cap)
        jj = rng.integers(0, nv, n_pairs_cap)
        keep = ii != jj
        ii, jj = ii[keep], jj[keep]
    diffs = embs_K[ii] - embs_K[jj]
    delta_y = sy_arr[ii] - sy_arr[jj]
    keep = np.abs(delta_y) > 1e-12
    diffs = diffs[keep]
    A = (delta_y[keep] > 0).astype(np.float64)
    return diffs, A


# ============================================================================
# PReF prediction at test (per-user fold)
# ============================================================================

def method_pref_emb(g_user, hc_id, emb_dict, mean_vec, basis,
                     J, beta_l2, use_svd_init, use_l2,
                     rng, alpha_pop, beta_pop):
    """PReF per-user-fold prediction with PCA-projected embeddings."""
    tr = g_user[g_user["conversation_id"] != hc_id]
    te = g_user[g_user["conversation_id"] == hc_id]
    rm_tr = tr["rm_score"].to_numpy(dtype=np.float64)
    sy_tr = tr["score_user"].to_numpy(dtype=np.float64)
    rm_te = te["rm_score"].to_numpy(dtype=np.float64)
    if len(rm_tr) < 3 or len(te) < 1:
        return alpha_pop + beta_pop * rm_te

    diffs_tr, A_tr = build_pair_phi_diffs_emb(
        tr, emb_dict, mean_vec, basis, rng)
    if len(A_tr) < 1:
        return alpha_pop + beta_pop * rm_te

    if use_svd_init:
        lam_init = svd_init_lambda_one_user(diffs_tr, A_tr, J)
    else:
        lam_init = rng.normal(0, INIT_SCALE_RANDOM, size=J)
    eff_beta = beta_l2 if use_l2 else 0.0
    lam_i = fit_lambda_for_user(diffs_tr, A_tr, J, eff_beta, init=lam_init)

    # Compute lambda^T phi on TRAIN-fold for affine recalibration
    train_uids = tr["utterance_id"].tolist()
    phi_tr_list = []
    for uid in train_uids:
        if uid in emb_dict:
            phi_tr_list.append(project_embedding(
                emb_dict[uid][None, :], mean_vec, basis)[0])
        else:
            phi_tr_list.append(np.zeros(J))
    phi_tr_emb = np.stack(phi_tr_list, axis=0)
    lam_phi_tr = phi_tr_emb @ lam_i
    a_cal, b_cal = affine_recalibrate(rm_tr, sy_tr, lam_phi_tr)

    # TEST prediction
    test_uids = te["utterance_id"].tolist()
    phi_te_list = []
    for uid in test_uids:
        if uid in emb_dict:
            phi_te_list.append(project_embedding(
                emb_dict[uid][None, :], mean_vec, basis)[0])
        else:
            phi_te_list.append(np.zeros(J))
    phi_te_emb = np.stack(phi_te_list, axis=0)
    lam_phi_te = phi_te_emb @ lam_i
    return a_cal + b_cal * lam_phi_te


# ============================================================================
# LoRe prediction at test (per-user fold)
# ============================================================================

def method_lore_emb(g_user, hc_id, emb_dict, mean_vec, basis,
                     B, lr_w, rng, alpha_pop, beta_pop):
    """LoRe per-user-fold prediction with PCA-projected embeddings.

    Q4 freezes A = identity (the PCA basis IS the basis; per the design
    "use 4096-D embeddings instead of polynomial substitution"). This
    is a CONSERVATIVE underestimate of LoRe's full method (paper learns
    A jointly); honest scope caveat in Limitations.
    """
    tr = g_user[g_user["conversation_id"] != hc_id]
    te = g_user[g_user["conversation_id"] == hc_id]
    rm_tr = tr["rm_score"].to_numpy(dtype=np.float64)
    sy_tr = tr["score_user"].to_numpy(dtype=np.float64)
    rm_te = te["rm_score"].to_numpy(dtype=np.float64)
    if len(rm_tr) < 3 or len(te) < 1:
        return alpha_pop + beta_pop * rm_te

    diffs_tr, A_tr = build_pair_phi_diffs_emb(
        tr, emb_dict, mean_vec, basis, rng)
    if len(A_tr) < 1:
        return alpha_pop + beta_pop * rm_te

    # LoRe loss l(w^T (R(c) - R(r))); orient phi-diffs to chosen-rejected
    dphi_oriented = diffs_tr.copy()
    flip = A_tr < 0.5
    dphi_oriented[flip] = -dphi_oriented[flip]

    w_init = np.full(B, 1.0 / B)
    A_basis_id = np.eye(B)            # A frozen at identity (PCA-rotation)
    w_u = solve_w_for_user(dphi_oriented, A_basis_id, w_init,
                              n_steps=150, lr=lr_w, B=B)

    # Compute scalar LoRe-margin on TRAIN utterances -> per-user OLS rescale
    train_uids = tr["utterance_id"].tolist()
    phi_tr_list = []
    for uid in train_uids:
        if uid in emb_dict:
            phi_tr_list.append(project_embedding(
                emb_dict[uid][None, :], mean_vec, basis)[0])
        else:
            phi_tr_list.append(np.zeros(B))
    phi_tr_emb = np.stack(phi_tr_list, axis=0)
    r_tr = phi_tr_emb @ w_u
    a_i, b_i, _, _ = fit_ols_with_se(r_tr, sy_tr)
    if not np.isfinite(a_i):
        a_i, b_i = float(np.mean(sy_tr)), 0.0

    test_uids = te["utterance_id"].tolist()
    phi_te_list = []
    for uid in test_uids:
        if uid in emb_dict:
            phi_te_list.append(project_embedding(
                emb_dict[uid][None, :], mean_vec, basis)[0])
        else:
            phi_te_list.append(np.zeros(B))
    phi_te_emb = np.stack(phi_te_list, axis=0)
    r_te = phi_te_emb @ w_u
    return a_i + b_i * r_te


# ============================================================================
# Stage B — main LOCO loop
# ============================================================================

PREF_VARIANTS = [
    ("pref_J2_emb",    2, True,  True),
    ("pref_J3_emb",    3, True,  True),
    ("pref_J6_emb",    6, True,  True),
]
LORE_VARIANTS = [
    ("lore_B2_emb",   2),
    ("lore_B4_emb",   4),
    ("lore_B8_emb",   8),
    ("lore_B16_emb", 16),
]
METHOD_NAMES = (
    ["pop_slope", "pebs_shrunk"]
    + [v[0] for v in PREF_VARIANTS]
    + [v[0] for v in LORE_VARIANTS]
)
K_GRID_ALL = sorted(set(J for _, J, _, _ in PREF_VARIANTS)
                       | set(B for _, B in LORE_VARIANTS))


def run_loco(args):
    t0 = time.time()
    if not SCORED.exists():
        sys.stderr.write(f"FATAL: PRISM scored data not found: {SCORED}\n")
        sys.exit(2)

    emb_dict = load_embeddings(EMB_PATH)
    df = pd.read_parquet(SCORED).dropna(subset=[
        "score_user", "rm_score", "conversation_id"])
    conv_count = df.groupby("user_id")["conversation_id"].nunique()
    utt_count = df.groupby("user_id").size()
    keep = conv_count[(conv_count >= MIN_CONV_PER_USER) &
                        (utt_count >= MIN_UTT_PER_USER)].index
    df = df[df["user_id"].isin(keep)].reset_index(drop=True)
    df = df[df["utterance_id"].isin(set(emb_dict.keys()))].reset_index(drop=True)

    if args.smoke:
        keep_users = df["user_id"].drop_duplicates().head(50).tolist()
        df = df[df["user_id"].isin(keep_users)].reset_index(drop=True)
        print(f"[smoke] subsampled to {len(keep_users)} users")
    print(f"[filter] {len(df)} utt x {df['user_id'].nunique()} users")

    slope_pop, intercept_pop = np.polyfit(df["rm_score"].to_numpy(),
                                            df["score_user"].to_numpy().astype(float), 1)
    alpha_pop, beta_pop = float(intercept_pop), float(slope_pop)
    print(f"[pop]    alpha_pop={alpha_pop:.3f}  beta_pop={beta_pop:.4f}")

    user_stats = []
    for uid, g in df.groupby("user_id"):
        a, b, sa, sb = fit_ols_with_se(g["rm_score"].to_numpy(),
                                        g["score_user"].to_numpy().astype(float))
        user_stats.append((a, b, sa, sb))
    alphas = np.array([s[0] for s in user_stats if np.isfinite(s[0])])
    betas = np.array([s[1] for s in user_stats if np.isfinite(s[1])])
    sas = np.array([s[2] for s in user_stats if np.isfinite(s[2])])
    sbs = np.array([s[3] for s in user_stats if np.isfinite(s[3])])
    tau2_a = max(0.0, float(np.var(alphas, ddof=1) - np.mean(sas ** 2)))
    tau2_b = max(0.0, float(np.var(betas, ddof=1) - np.mean(sbs ** 2)))
    print(f"[pebs]  tau2_a={tau2_a:.2f}  tau2_b={tau2_b:.4f}")

    print(f"[pca]    fitting top-{max(K_GRID_ALL)} PCA on all "
          f"{len(emb_dict)} embeddings ...")
    all_uids = sorted(emb_dict.keys())
    all_emb = np.stack([emb_dict[u] for u in all_uids], axis=0).astype(np.float32)
    K_max = max(K_GRID_ALL)
    pca_mean, pca_basis_full, pca_var = fit_pca_basis(all_emb, K_max)
    var_total = float(np.sum(np.var(all_emb, axis=0)))
    var_explained = float(pca_var.sum() / max(var_total, 1e-12))
    print(f"[pca]    K_max={K_max}, "
          f"explained_variance@K_max={var_explained:.4f}")
    K_basis = {K: pca_basis_full[:K] for K in K_GRID_ALL}
    K_var = {K: float(pca_var[:K].sum() / max(var_total, 1e-12)) for K in K_GRID_ALL}
    print(f"[pca]    per-K explained variance: {K_var}")

    rng_master = np.random.default_rng(args.seed)
    per_user = []
    for uid, g in tqdm(df.groupby("user_id"), desc="LOCO Q4"):
        conv_ids = g["conversation_id"].unique().tolist()
        if len(conv_ids) < MIN_CONV_PER_USER:
            continue
        sq = {m: [] for m in METHOD_NAMES}
        n_utt_used = 0
        user_seed = int(rng_master.integers(0, 2 ** 32 - 1))

        for hc in conv_ids:
            tr = g[g["conversation_id"] != hc]
            te = g[g["conversation_id"] == hc]
            if len(tr) < 3 or len(te) < 1:
                continue
            x_tr = tr["rm_score"].to_numpy(dtype=np.float64)
            y_tr = tr["score_user"].to_numpy(dtype=np.float64)
            x_te = te["rm_score"].to_numpy(dtype=np.float64)
            y_te = te["score_user"].to_numpy(dtype=np.float64)
            preds = {
                "pop_slope": alpha_pop + beta_pop * x_te,
                "pebs_shrunk": method_pebs_shrunk(
                    x_tr, y_tr, x_te, alpha_pop, beta_pop, tau2_a, tau2_b),
            }
            for tag, J, use_svd, use_l2 in PREF_VARIANTS:
                rng_local = np.random.default_rng(
                    user_seed + hash(tag) % (2 ** 16))
                preds[tag] = method_pref_emb(
                    g, hc, emb_dict, pca_mean, K_basis[J],
                    J=J, beta_l2=args.beta_l2 if use_l2 else 0.0,
                    use_svd_init=use_svd, use_l2=use_l2,
                    rng=rng_local,
                    alpha_pop=alpha_pop, beta_pop=beta_pop)
            for tag, B in LORE_VARIANTS:
                rng_local = np.random.default_rng(
                    user_seed + hash(tag) % (2 ** 16))
                preds[tag] = method_lore_emb(
                    g, hc, emb_dict, pca_mean, K_basis[B],
                    B=B, lr_w=args.lr_w, rng=rng_local,
                    alpha_pop=alpha_pop, beta_pop=beta_pop)
            for m in METHOD_NAMES:
                sq[m].extend(((preds[m] - y_te) ** 2).tolist())
            n_utt_used += len(y_te)
        if n_utt_used < MIN_UTT_PER_USER:
            continue
        row = {"user_id": uid, "n_utt": n_utt_used, "n_conv": len(conv_ids)}
        for m in METHOD_NAMES:
            row[f"rmse_{m}"] = float(np.sqrt(np.mean(sq[m])))
        per_user.append(row)
    pu = pd.DataFrame(per_user)
    print(f"\n[loco] {len(pu)} users evaluated")

    for m in METHOD_NAMES:
        if m == "pop_slope":
            continue
        pu[f"gain_{m}_pct"] = 100.0 * (pu["rmse_pop_slope"] - pu[f"rmse_{m}"]) / pu["rmse_pop_slope"]

    def cluster_boot_ci(values, n_boot, seed):
        rng = np.random.default_rng(seed)
        n = len(values)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boots[b] = float(values[idx].mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return float(values.mean()), float(lo), float(hi), float(np.std(boots))

    summary = {
        "slice": {
            "n_users": int(len(pu)),
            "n_utterances": int(pu["n_utt"].sum()),
            "min_conv_per_user": MIN_CONV_PER_USER,
            "min_utt_per_user": MIN_UTT_PER_USER,
        },
        "rng_seed": args.seed,
        "n_bootstrap": args.n_boot,
        "embedding_settings": {
            "model": QWEN_MODEL,
            "embedding_dim": EMB_DIM,
            "pooling": "mean over response tokens (attention-mask-weighted)",
            "max_length": MAX_LENGTH,
            "K_grid": list(K_GRID_ALL),
            "K_explained_variance": K_var,
            "pca_basis_fit": "global pooled (paper protocol; train-fold lambda fit handled in inner LOCO loop)",
        },
        "pref_settings": {
            "J_grid": list(J_GRID_PREF),
            "beta_l2": float(args.beta_l2),
            "n_refine_steps": N_REFINE_STEPS,
            "refine_lr": REFINE_LR,
            "phi_basis": "frozen-PCA-projection-of-Qwen25-7B-pre-final-layer",
            "pair_construction": "intra-user-scalar-derived (no PERSONA aug)",
        },
        "lore_settings": {
            "B_grid": list(B_GRID_LORE),
            "lr_w": args.lr_w,
            "A_protocol": "identity (frozen PCA basis = R_phi; no joint A learning in Q4)",
        },
        "methods": {},
        "rmse_abs": {m: float(pu[f"rmse_{m}"].mean()) for m in METHOD_NAMES},
        "verdict_class": "PENDING-RESULT",
        "anomalies_fired": [],
    }
    print("\n=== Q4 head-to-head: mean RMSE reduction (%) vs pop_slope ===")
    print(f"{'method':>20s}  {'RMSE':>8s}  {'gain%':>9s}  {'95% CI':>22s}")
    print(f"{'pop_slope':>20s}  {summary['rmse_abs']['pop_slope']:8.3f}  "
          f"{'0.00':>9s}  {'(reference)':>22s}")
    for m in METHOD_NAMES:
        if m == "pop_slope":
            continue
        gain_col = f"gain_{m}_pct"
        mean_g, lo, hi, se = cluster_boot_ci(pu[gain_col].to_numpy(),
                                                args.n_boot, args.seed)
        summary["methods"][m] = {
            "rmse_mean": float(pu[f"rmse_{m}"].mean()),
            "rmse_reduction_pct_mean": mean_g,
            "rmse_reduction_pct_ci95": [lo, hi],
            "rmse_reduction_pct_se": se,
            "status": "reimplementation_q4_4096d" if m != "pebs_shrunk" else "reference",
        }
        print(f"{m:>20s}  {pu[f'rmse_{m}'].mean():8.3f}  "
              f"{mean_g:+9.3f}  [{lo:+6.3f}, {hi:+6.3f}]")

    pebs_gain = pu["gain_pebs_shrunk_pct"].to_numpy()
    summary["paired_vs_pebs"] = {}
    for m in METHOD_NAMES:
        if m in ("pop_slope", "pebs_shrunk"):
            continue
        diff = pebs_gain - pu[f"gain_{m}_pct"].to_numpy()
        md, lo, hi, se = cluster_boot_ci(diff, args.n_boot, args.seed + 1)
        summary["paired_vs_pebs"][m] = {
            "mean_delta_pct": md, "ci95": [lo, hi], "se": se,
            "pebs_better": bool(lo > 0),
        }
        sig = "*" if lo > 0 or hi < 0 else ""
        print(f"  PEBS - {m:>16s}: {md:+.3f}%  CI [{lo:+.3f}, {hi:+.3f}] {sig}")

    pebs_g_lo = summary["methods"]["pebs_shrunk"]["rmse_reduction_pct_ci95"][0]
    pebs_g_mean = summary["methods"]["pebs_shrunk"]["rmse_reduction_pct_mean"]
    pref_keys = [v[0] for v in PREF_VARIANTS]
    lore_keys = [v[0] for v in LORE_VARIANTS]
    baseline_keys = pref_keys + lore_keys

    pebs_better_all = all(
        summary["paired_vs_pebs"][m]["pebs_better"] for m in baseline_keys)
    pebs_loses_any = any(
        summary["paired_vs_pebs"][m]["ci95"][1] < 0 for m in baseline_keys)
    best_baseline_gain = max(
        summary["methods"][m]["rmse_reduction_pct_mean"] for m in baseline_keys)
    margin_vs_best = pebs_g_mean - best_baseline_gain

    if pebs_loses_any:
        summary["verdict_class"] = "REJECTED-PEBS-LOSES-WITH-LARGE-EMBEDDING"
    elif pebs_better_all and margin_vs_best >= 3.0 and pebs_g_lo > 0:
        summary["verdict_class"] = "CONFIRMED-PEBS-RMSE-DOMINATES-PREF-LORE-LARGE-EMBEDDING"
    elif pebs_better_all and pebs_g_lo > 0:
        summary["verdict_class"] = "PARTIAL-PEBS-MARGIN-SHRINKS"
    else:
        summary["verdict_class"] = "TENTATIVE-INCONCLUSIVE"
    summary["margin_vs_best_baseline_pct"] = float(margin_vs_best)
    summary["best_baseline_method"] = max(
        baseline_keys,
        key=lambda m: summary["methods"][m]["rmse_reduction_pct_mean"])

    for m in baseline_keys:
        rmse_m = summary["methods"][m]["rmse_mean"]
        if rmse_m > 100.0 or not np.isfinite(rmse_m):
            summary["anomalies_fired"].append(
                f"large_rmse_{m}: rmse_mean={rmse_m:.4g} (expected <60 on PRISM)")

    summary["runtime_seconds"] = float(time.time() - t0)
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    (OUT_RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    pu.to_parquet(OUT_RESULTS / "per_user.parquet", index=False)
    print(f"\n[save] {OUT_RESULTS}/summary.json  ({summary['runtime_seconds']:.1f}s)")
    return summary, pu


# ============================================================================
# Output: LaTeX table + figure (NOT auto-included; user-decision)
# ============================================================================

def write_latex_table(summary: dict, out_path: Path):
    pretty = {
        "pebs_shrunk":  ("\\PEBS{} \\emph{(ours)}",
                            "Per-user EB-shrunk $(\\alpha_j, \\beta_j)$ on Qwen RM scalar",
                            "hierarchical EB"),
        "pref_J2_emb":   ("PReF $J{=}2$ (4096-D emb.) \\citep{shenfeld2025pref}",
                            "PReF Eq.~3-5 with frozen top-2 PCA of Qwen2.5-7B pre-final-layer hidden state",
                            "BT-MLE"),
        "pref_J3_emb":   ("PReF $J{=}3$ (4096-D emb.) \\citep{shenfeld2025pref}",
                            "PReF Eq.~3-5 with frozen top-3 PCA of Qwen2.5-7B pre-final-layer hidden state",
                            "BT-MLE"),
        "pref_J6_emb":   ("PReF $J{=}6$ (4096-D emb.) \\citep{shenfeld2025pref}",
                            "PReF Eq.~3-5 with frozen top-6 PCA (paper sweet spot)",
                            "BT-MLE"),
        "lore_B2_emb":   ("LoRe $B{=}2$ (4096-D emb.) \\citep{bose2025lore}",
                            "LoRe Eq.~7+10 with frozen top-2 PCA",
                            "low-rank B=2"),
        "lore_B4_emb":   ("LoRe $B{=}4$ (4096-D emb.) \\citep{bose2025lore}",
                            "LoRe Eq.~7+10 with frozen top-4 PCA",
                            "low-rank B=4"),
        "lore_B8_emb":   ("LoRe $B{=}8$ (4096-D emb.) \\citep{bose2025lore}",
                            "LoRe Eq.~7+10 with frozen top-8 PCA",
                            "low-rank B=8"),
        "lore_B16_emb":  ("LoRe $B{=}16$ (4096-D emb.) \\citep{bose2025lore}",
                            "LoRe Eq.~7+10 with frozen top-16 PCA",
                            "low-rank B=16"),
    }
    order = ["pebs_shrunk"] + [v[0] for v in PREF_VARIANTS] + [v[0] for v in LORE_VARIANTS]
    lines = []
    lines.append("% Q4 head-to-head: PReF + LoRe full 4096-D embedding sweep on PRISM LOCO.")
    lines.append("% Generated by paper/scripts/q4_pref_lore_full_embedding.py")
    lines.append(f"% N_users = {summary['slice']['n_users']}, "
                 f"N_utt = {summary['slice']['n_utterances']}, "
                 f"seed = {summary['rng_seed']}, n_boot = {summary['n_bootstrap']}.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering \\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append("\\caption{\\textbf{Q4: PReF + LoRe head-to-head with 4096-D Qwen2.5-7B-Instruct "
                 "pre-final-layer embeddings} (resolves the polynomial-basis pathology). "
                 "PReF and LoRe re-implemented with frozen top-$K$ PCA projection of the same "
                 "Qwen2.5-7B-Instruct pre-final-layer hidden state used by PEBS's RM-score "
                 "feature, replacing the polynomial-basis substitution of "
                 "\\S~Tab.~\\ref{tab:lore-headtohead-rmse-numbers}. "
                 "Same 1394-user PRISM LOCO split, same seed=20260420, same cluster-bootstrap "
                 "$n_{\\mathrm{boot}}$=2000. \\textit{Honest scope}: we freeze the PCA "
                 "projection rather than learning $\\phi/R_\\phi$ jointly with $\\lambda/w_i$ "
                 "(paper learns jointly); see \\S~Limitations.}")
    lines.append("\\label{tab:q4-pref-lore-full-embedding}")
    lines.append("\\begin{tabular}{@{}p{3.6cm}p{5.0cm}p{2.0cm}rr@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & \\textbf{Recipe on PRISM slice} & \\textbf{Partial-pooling} "
                 "& \\textbf{RMSE-red. (\\%) $\\uparrow$} & \\textbf{95\\% CI} \\\\")
    lines.append("\\midrule")
    for m in order:
        if m not in summary['methods']:
            continue
        pretty_name, recipe, tag = pretty[m]
        meth = summary['methods'][m]
        g = meth['rmse_reduction_pct_mean']
        ci = meth['rmse_reduction_pct_ci95']
        g_s = f"\\textbf{{{g:+.2f}}}" if m == "pebs_shrunk" else f"{g:+.2f}"
        ci_s = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"
        lines.append(f"{pretty_name} & {recipe} & {tag} & {g_s} & {ci_s} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{-0.5em}")
    lines.append("\\end{table}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[save] {out_path}")


def write_figure(summary: dict, out_path: Path):
    order = ["pebs_shrunk"] + [v[0] for v in PREF_VARIANTS] + [v[0] for v in LORE_VARIANTS]
    order = [m for m in order if m in summary['methods']]
    pretty = {
        "pebs_shrunk":  "PEBS (ours)",
        "pref_J2_emb":   "PReF J=2 (4096-D)",
        "pref_J3_emb":   "PReF J=3 (4096-D)",
        "pref_J6_emb":   "PReF J=6 (4096-D)",
        "lore_B2_emb":   "LoRe B=2 (4096-D)",
        "lore_B4_emb":   "LoRe B=4 (4096-D)",
        "lore_B8_emb":   "LoRe B=8 (4096-D)",
        "lore_B16_emb":  "LoRe B=16 (4096-D)",
    }
    palette = {
        "pebs_shrunk":  "#1f77b4",
        "pref_J2_emb":   "#ffbb78",
        "pref_J3_emb":   "#ff7f0e",
        "pref_J6_emb":   "#d62728",
        "lore_B2_emb":   "#aec7e8",
        "lore_B4_emb":   "#98df8a",
        "lore_B8_emb":   "#c5b0d5",
        "lore_B16_emb":  "#9467bd",
    }
    means = [summary['methods'][m]['rmse_reduction_pct_mean'] for m in order]
    ci_lo = [summary['methods'][m]['rmse_reduction_pct_ci95'][0] for m in order]
    ci_hi = [summary['methods'][m]['rmse_reduction_pct_ci95'][1] for m in order]
    err_lo = [m_v - lo for m_v, lo in zip(means, ci_lo)]
    err_hi = [hi - m_v for m_v, hi in zip(means, ci_hi)]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(order))
    bars = ax.bar(x, means, yerr=[err_lo, err_hi], capsize=5,
                    color=[palette[m] for m in order], edgecolor="black", linewidth=0.8)
    y_max = max(hi for hi in ci_hi) + 1.5
    y_min = min(lo for lo in ci_lo) - 1.5
    for i, (b, mval) in enumerate(zip(bars, means)):
        label_y = ci_hi[i] + 0.35 if mval >= 0 else ci_lo[i] - 0.7
        va = "bottom" if mval >= 0 else "top"
        ax.text(b.get_x() + b.get_width() / 2, label_y, f"{mval:+.2f}%",
                  ha="center", va=va, fontsize=9,
                  fontweight="bold" if order[i] == "pebs_shrunk" else "normal")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[m] for m in order], rotation=20, fontsize=9, ha="right")
    ax.set_ylabel("RMSE reduction vs pop-slope baseline (%)", fontsize=11)
    ax.set_title(f"Q4: PRISM LOCO PReF+LoRe with 4096-D Qwen embeddings  "
                  f"(N={summary['slice']['n_users']} users, cluster-bootstrap 95% CI)",
                  fontsize=10)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


# ============================================================================
# Entry point
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=RNG_DEFAULT)
    p.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    p.add_argument("--beta-l2", type=float, default=BETA_L2_DEFAULT)
    p.add_argument("--lr-w", type=float, default=LR_W_DEFAULT)
    p.add_argument("--emb-batch-size", type=int, default=EMB_BATCH_SIZE)
    p.add_argument("--extract-embeddings", action="store_true",
                    help="Stage A: extract Qwen2.5-7B embeddings to "
                         f"{EMB_PATH} (one-time, ~30-60min on H100)")
    p.add_argument("--smoke", action="store_true",
                    help="Smoke mode: subsample 50 users for fast pipeline test")
    p.add_argument("--skip-fig", action="store_true")
    p.add_argument("--skip-table", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.extract_embeddings:
        extract_embeddings(args)
        return
    summary, pu = run_loco(args)
    if not args.skip_table:
        write_latex_table(summary, OUT_TABLE)
    if not args.skip_fig:
        write_figure(summary, OUT_FIG)
    print(f"\n[verdict] {summary['verdict_class']}")
    print(f"[runtime] {summary['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
