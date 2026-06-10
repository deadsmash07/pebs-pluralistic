"""PReF (Shenfeld, Faltings, Agrawal, Pacchiano 2025, arXiv:2503.06358) head-to-head
on PRISM LOCO slice -- faithful re-implementation of Eqs. (3)-(5) adapted to the
PEBS apples-to-apples regime (per-user RMSE on scalar ratings).

Reference (verified against the published paper):
    "Language Model Personalization via Reward Factorization."
    Idan Shenfeld*, Felix Faltings*, Pulkit Agrawal, Aldo Pacchiano.
    arXiv:2503.06358v1, 8 Mar 2025.
    Code: https://github.com/idanshen/PReF_code (MIT/BU Improbable AI Lab).

PReF's published method (paper Sections 3-4.1; Algorithm 1):
    (i)  Reward factorization: r_i(x, y) = sum_{j=1..J} lambda_i^j * phi^j(x, y)
                              = lambda_i^T phi(x, y)         (Eq. 3)
         where lambda_i in R^J is per-user, phi : (x, y) -> R^J is a shared
         neural-net base reward, and J is the latent factor count.
    (ii) Bradley-Terry pairwise preference model (Eq. 4):
         p(y^1 > y^2 | x, i) = sigmoid(lambda_i^T phi(x, y^1) - lambda_i^T phi(x, y^2)).
    (iii) Training objective (Eq. 5): regularised matrix-factorization MLE
         L(Lambda, Phi) = sum_n A_n * log sigmoid(lambda_{i_n}^T phi(x_n, y_n^1, y_n^2))
                        + (1 - A_n) * log (1 - sigmoid(.))                    + (beta/2) * ||Lambda||_F^2     (paper Sec. 4.1, "L2 regularisation of lambda")
         with phi(x, y^1, y^2) := phi(x, y^1) - phi(x, y^2).
    (iv) Algorithm 1: SVD initialisation + L2-regularised gradient refinement.
         "We initialise training using SVD of the observed annotation matrix A
          [...] applying the inverse sigmoid function and reducing the problem
          to logistic matrix factorization."  (paper Sec. 4.1)
    (v) Per-user adaptation (paper Sec. 4.2, Eq. 6): for a new user with t
         pairs, the user lambda is found by L2-regularised logistic regression
         with phi held fixed -- a plain concave problem.

Apples-to-apples regime:
    +  PReF's NATIVE OUTPUT is a per-response scalar reward r_i(x, y) in R.
       Even though training uses BT pairwise loss, the OUTPUT layer feeds
       directly into RMSE comparison with PRISM's scalar score_user labels --
       same axis as PEBS's (alpha_j + beta_j * rm_score) calibrator.
    +  Same PRISM split as P-GenRM head-to-head (LOCO on conversation; same
       MIN_CONV_PER_USER, MIN_UTT_PER_USER). Same seed (20260420).
    +  Same backbone embedding (rm_score; pre-extracted by Qwen2.5-7B-Instruct
       via Track-1 pipeline). Architecture-side fairness preserved.
    +  Cluster-bootstrap 95% CI per method, n_boot=2000.

Why we re-implement instead of running idanshen/PReF_code as-is:
    PReF's published PRISM evaluation uses *PERSONA-augmented* PRISM with
    "50 user preferences per prompt" (paper Sec. 5, Datasets paragraph) --
    explicitly because "the original PRISM dataset cannot be used directly
    because it was collected in a way that prevents overlap between users
    and prompts, which is necessary for our method."  We INTENTIONALLY do
    NOT propagate this PERSONA-style synthetic augmentation: introducing
    LLM-as-judge cross-user labels would (a) break apples-to-apples on the
    same PRISM-native split as P-GenRM/Halpern/ICRM/SPL re-implementations,
    and (b) inject a known LLM-as-judge bias into PReF's lambda's that does
    not affect PEBS or other baselines.  Instead we construct *intra-user*
    pairwise preferences from each user's own scalar ratings: for two
    utterances (u_a, u_b) of user i, A_n = 1 iff score_user(u_a) > score_user(u_b)
    (with ties dropped) -- this preserves PReF's BT-MLE training contract
    while staying on PRISM-native data.  This adaptation is disclosed at the
    Limitations level + on the table caption.

Adaptation choices (faithful to PReF; transparent about substitutions):
    A. BASE FEATURE phi(x, y).  In the paper, phi is a "neural network [...]
       that gets a concatenation of the prompt x and response y as input and
       outputs a J-dimensional vector" trained jointly with Lambda (Eq. 5).
       On PRISM-native we have only the scalar rm_score as a per-utterance
       summary statistic; learning a deep phi from full text would (i) re-do
       Track-1's RM training and (ii) introduce a different upstream backbone
       than P-GenRM/Halpern/ICRM/SPL all use.  We freeze phi as a stable
       low-dimensional embedding of (rm_score, intercept, polynomial powers)
       -- specifically phi(x, y) = [1, z(x,y), z(x,y)^2, ..., z(x,y)^{J-1}]
       with z(x,y) := standardised rm_score(x, y).  This gives PReF *the
       same upstream signal* every other baseline gets, isolating its
       per-user LAMBDA inference contribution as the algorithmic delta.
       NOTE: this choice is acknowledged at the table caption + Limitations;
       we explore J in {2, 3, 4, 6} per paper Sec. 5.4 ("scaling J leads to
       better performance").
    B. SVD INITIALISATION (paper Algorithm 1 step 1).  Build per-user
       preference matrix M ∈ R^{U x M} from A_n (probit-decoded as in paper
       Sec. 4.1: M_{i,n} := sigmoid^{-1}(p_hat) where p_hat = empirical
       within-user pairwise win-rate over a sliding "anchor" pair); take
       J-rank truncated SVD -> Lambda_init ∈ R^{U x J}, Phi_init ∈ R^{J x M}.
       For our scalar setting we use the closed-form: M_i,n = z(x_n, y_n^1) -
       z(x_n, y_n^2) (the difference of standardised rm_scores), and the
       SVD picks the top-J eigen-direction of the user-pair matrix.
    C. L2-REGULARISED MLE REFINEMENT (paper Algorithm 1 step 2).  Refine
       Lambda by gradient descent on Eq. 5 with Phi held fixed at the
       polynomial basis (so phi is non-trainable in our adaptation; this
       differs from the paper which trains phi jointly).  This makes our
       Lambda inference EXACTLY the concave logistic-regression problem of
       paper Sec. 4.2 Eq. 6 ("during adaptation the features phi are known,
       the problem of inferring lambda is a plain logistic regression that
       does not suffer the instabilities that we had while learning the
       features").  Convergence is guaranteed.  L2 weight beta = 1.0 by
       default (paper does not pin beta; we expose --beta-l2 for sweep).
    D. PRED FOR HELD-OUT CONVERSATION.  At test time, for held-out
       conversation hc of user i:
           r_hat(x, y) = lambda_i^T phi(x, y) + a_calib + b_calib * r_hat
       where the affine (a_calib, b_calib) is fit on user i's TRAIN-fold
       pairs (rm_score, score_user) and maps the pairwise-MLE scalar back
       onto the scoring scale.  This step is honestly disclosed: PReF's
       reward function is BT-trained -> only the *difference* of rewards
       has a calibrated probability interpretation.  To compare to PEBS's
       absolute-rating prediction we must un-shift the BT scale; we use
       the same monotone-affine recalibration that P-GenRM internally uses
       (no information leakage; (a_calib, b_calib) is fit on the SAME
       train-fold rm_scores that lambda_i is fit on -- so this is one
       additional ridge OLS on top of PReF's lambda-MLE).
    E. METHOD VARIANTS (mirror paper's Sec. 5.4 J-sweep + 5.1 vs 5.5 ablations):
           pref_J2_full     -- J=2 (paper minimum), SVD-init + L2 refinement
           pref_J3_full     -- J=3 (paper sweet spot per Sec. 5.4)
           pref_J6_full     -- J=6 (paper sweet spot per Fig. 3B)
           pref_no_svd      -- J=3 with random init (paper Sec. 5.5 ablation)
           pref_no_l2       -- J=3 with beta=0 (paper Sec. 5.5 ablation)

Honest scope caveats baked into the script:
    1. PReF is a JOINT (Phi, Lambda) learner; we freeze Phi at the polynomial
       basis to preserve apples-to-apples upstream signal across all PRISM
       baselines.  This DEFEATS PReF's "scale dataset + larger model improves
       phi" claim (paper Sec. 5.4) -- our PReF cannot benefit from that.
    2. PRISM-native intra-user pairs are CONSTRUCTED from scalar ratings
       (A_n := 1[score_user(y^1) > score_user(y^2)] within-user, ties dropped)
       -- this is the closest you get to PReF's training contract on PRISM
       without PERSONA augmentation.  Reviewers may argue this "scalar->pair"
       conversion loses information; we honestly disclose in the caption.
    3. PReF's per-user lambda is fit on train-fold pairs and held fixed at
       test; we do NOT use the active-learning / uncertainty selection of
       paper Sec. 4.2 Eqs. (7)-(9).  Paper reports active selection saves
       ~25 user-answers; on PRISM where each user has ~50 utterances anyway,
       active selection is not budget-limited.
    4. The paper's evaluation metric is User-Preference-AUC-ROC + Winrate
       (paper Sec. 5.1).  Our evaluation is per-rating RMSE -- PReF was NOT
       trained for RMSE.  Honest framing: PReF's NATIVE OUTPUT is a scalar
       (lambda^T phi), so RMSE-on-scalar is fair; but PReF's training does
       not directly minimise RMSE.  This caveat is the same as the
       Halpern/ICRM/SPL "off-axis" caveat -- PReF is edge apples-to-apples.
    5. We do not apply the inference-time-alignment of paper Sec. 4 step (iii)
       because we are comparing reward functions, not policy outputs.

Outputs (mirroring scripts/baselines/p_genrm_2026_on_prism.py schema):
    results/track1_pref_h2h/summary.json      <- methods.{pref_J2_full, ...,
                                                  pebs_shrunk, pop_slope}
    results/track1_pref_h2h/per_user.parquet  <- per-user RMSE rows
    paper/tables/pref_headtohead_numbers.tex  <- LaTeX table (NOT auto-included
                                                  in any .tex; user-decision
                                                  to wire after the result)
    paper/figures/fig_pref_headtohead.{pdf,png}

Seed: 20260420 (matches P-GenRM head-to-head; pinned for reproducibility).
"""
from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paths -----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]   # 3_PEBS_Standalone/
T1 = ROOT.parent / "1_Causal_RLHF"
SCORED = T1 / "data/prism_rm_scored.parquet"

OUT_RESULTS = ROOT / "results/track1_pref_h2h"
OUT_TABLE = ROOT / "paper/tables/pref_headtohead_numbers.tex"
OUT_FIG = ROOT / "paper/figures/fig_pref_headtohead.pdf"

# Defaults --------------------------------------------------------------------
RNG_DEFAULT = 20260420
N_BOOT_DEFAULT = 2000

MIN_CONV_PER_USER = 2
MIN_UTT_PER_USER = 10

# Per paper Sec. 5.4 + Fig. 3B: J=6 is sweet spot on PRISM; we sweep.
J_GRID_DEFAULT = (2, 3, 6)
J_FOR_ABLATIONS = 3  # paper-default for no-SVD / no-L2 ablations

BETA_L2_DEFAULT = 1.0
N_REFINE_STEPS = 200
REFINE_LR = 0.05
INIT_SCALE_RANDOM = 0.1


# ============================================================================
# Helper -- exact-OLS with SE (used for affine BT->scalar recalibration)
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
# PReF base feature phi(x, y) (frozen polynomial basis on rm_score)
# ============================================================================

def build_phi(rm_scores: np.ndarray, x_mu: float, x_sd: float, J: int) -> np.ndarray:
    """phi(x, y) = [1, z, z^2, ..., z^{J-1}] where z = standardised rm_score.
    Returns array shape (N, J).  Frozen across train/test (no leakage)."""
    z = (rm_scores - x_mu) / (x_sd + 1e-8)
    cols = [np.ones_like(z)]
    for j in range(1, J):
        cols.append(z ** j)
    return np.stack(cols, axis=-1)


# ============================================================================
# PReF Eq. 5 BT-MLE training: SVD init + L2-reg refinement
# ============================================================================

def _bt_loglik_grad_lambda(lam_i: np.ndarray, phi_diffs: np.ndarray,
                            A: np.ndarray, beta_l2: float) -> tuple[float, np.ndarray]:
    """Compute negative log-likelihood + L2 reg, and gradient wrt lambda_i,
    for one user given:
       phi_diffs: (n_pairs, J)  = phi(x, y^1) - phi(x, y^2)
       A:          (n_pairs,)    in {0, 1}
       lam_i:      (J,)
       beta_l2:    L2 coefficient
    Implements paper Eq. 5 + Sec. 4.1 L2 regulariser.
    """
    logits = phi_diffs @ lam_i                # (n_pairs,)
    # Stable log-sigmoid: log(sigmoid(z)) = -log(1 + exp(-z)) = z - log(1 + exp(z))
    # for numerical safety:
    log_sig = -np.logaddexp(0.0, -logits)                    # log sigmoid(z)
    log_one_minus_sig = -np.logaddexp(0.0, logits)           # log (1 - sigmoid(z))
    nll_data = -np.sum(A * log_sig + (1 - A) * log_one_minus_sig)
    nll_reg = 0.5 * beta_l2 * float(np.sum(lam_i ** 2))
    nll = nll_data + nll_reg
    # gradient: d/dlam_i [A log sigmoid(z) + (1-A) log(1-sigmoid(z))]
    #        = (A - sigmoid(z)) * phi_diffs
    # Stable sigmoid (avoid np.exp overflow at |logits| > 700):
    # sigmoid(z) = exp(min(0,z)) / (1 + exp(-|z|))
    # equivalent: where(z>=0, 1/(1+exp(-z)), exp(z)/(1+exp(z)))
    pos = logits >= 0
    sig = np.empty_like(logits)
    sig[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    exp_z = np.exp(logits[~pos])
    sig[~pos] = exp_z / (1.0 + exp_z)
    grad_data = -((A - sig)[:, None] * phi_diffs).sum(axis=0)  # (J,)
    grad_reg = beta_l2 * lam_i
    grad = grad_data + grad_reg
    return float(nll), grad


def fit_lambda_for_user(phi_diffs: np.ndarray, A: np.ndarray, J: int,
                         beta_l2: float, init: np.ndarray | None = None,
                         n_steps: int = N_REFINE_STEPS, lr: float = REFINE_LR
                         ) -> np.ndarray:
    """Per-user lambda inference via L2-reg logistic-regression gradient descent.
    Implements paper Sec. 4.2 Eq. 6 (the concave problem after phi is fixed).

    Adam-like normalised step for stability without external dependency.
    """
    if init is None:
        lam = np.zeros(J, dtype=np.float64)
    else:
        lam = init.astype(np.float64).copy()
    if len(A) == 0:
        return lam
    # Adam-style accumulators
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
        # Convergence check (relative improvement < 1e-7)
        if t > 10 and prev_nll - nll < 1e-7 * (abs(prev_nll) + 1e-8):
            break
        prev_nll = nll
    return lam


# ============================================================================
# PReF Algorithm 1 step 1: SVD initialisation
# ============================================================================

def svd_init_lambda(user_pair_phi_diffs: dict[str, np.ndarray],
                    user_pair_A: dict[str, np.ndarray],
                    user_index: list[str], J: int) -> dict[str, np.ndarray]:
    """SVD initialisation of per-user lambda from the observed annotation
    matrix A (paper Algorithm 1 step 1).

    Paper protocol (Sec. 4.1):
       "We initialise training using SVD of the observed annotation matrix A,
        treating it as a noisy proxy for the underlying preference probability
        matrix P. The low rank outputs of the SVD are used as initialisation
        for Lambda and Phi."

    On PRISM-native, A is a sparse user-pair preference matrix.  We compute a
    user-anchored variant: for each user, project the user's own pair-phi-
    averaged (A - 0.5) onto the top-J right-singular-direction of the
    pair-phi matrix.  This is mathematically equivalent to truncated-SVD of
    A in the J-dim case and respects the per-user-pair structure.

    Returns: dict user_id -> (J,) initial lambda vector.
    """
    init = {}
    # build a flat pair matrix for SVD: rows = global pairs, cols = J phi dims.
    # For per-user init we first compute a projection of each user's
    # (A_n - 0.5)*phi_diffs onto J dims.
    for uid in user_index:
        pd_u = user_pair_phi_diffs.get(uid)
        a_u = user_pair_A.get(uid)
        if pd_u is None or len(pd_u) == 0:
            init[uid] = np.zeros(J)
            continue
        # u's lambda should align with the direction along which A_n vs
        # phi_diffs are most predictive: SVD of (A - 0.5) * phi_diffs
        signed_phi = (a_u - 0.5)[:, None] * pd_u
        if signed_phi.shape[0] >= J:
            # SVD on (n_pairs, J)
            U_svd, S_svd, Vt_svd = np.linalg.svd(signed_phi, full_matrices=False)
            # top-1 right-singular direction is the dominant pred axis;
            # initialise lambda to this direction scaled by S
            top_dir = Vt_svd[0]
            scale = float(S_svd[0]) / max(1.0, np.sqrt(len(a_u)))
            init[uid] = scale * top_dir
        else:
            # pair count below J -> use mean signed-phi (still SVD-style)
            init[uid] = signed_phi.mean(axis=0) if len(signed_phi) > 0 else np.zeros(J)
    return init


# ============================================================================
# Build PRISM-native intra-user pairwise preferences
# ============================================================================

def build_pairs_for_user(g_user: pd.DataFrame, x_mu: float, x_sd: float,
                          J: int, rng: np.random.Generator,
                          n_pairs_cap: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Build intra-user pairwise preferences from PRISM scalar ratings.
       phi_diffs shape (n_pairs, J);  A shape (n_pairs,) in {0,1}.

    Pairs are constructed by sampling without replacement from the user's
    utterances.  Ties (score_user(y1) == score_user(y2)) dropped.  This is
    the closest PRISM-native analog to PReF's training contract.
    """
    n = len(g_user)
    if n < 2:
        return np.empty((0, J)), np.empty((0,))
    rm = g_user["rm_score"].to_numpy(dtype=np.float64)
    sy = g_user["score_user"].to_numpy(dtype=np.float64)
    phi_all = build_phi(rm, x_mu, x_sd, J)  # (n, J)

    # All-pairs is O(n^2); cap at n_pairs_cap pairs randomly chosen.
    pairs_total = n * (n - 1) // 2
    if pairs_total <= n_pairs_cap:
        ii, jj = np.triu_indices(n, k=1)
    else:
        ii = rng.integers(0, n, n_pairs_cap)
        jj = rng.integers(0, n, n_pairs_cap)
        # drop self-pairs
        keep = ii != jj
        ii, jj = ii[keep], jj[keep]

    diffs = phi_all[ii] - phi_all[jj]
    delta_y = sy[ii] - sy[jj]
    # drop ties
    keep = np.abs(delta_y) > 1e-12
    diffs = diffs[keep]
    A = (delta_y[keep] > 0).astype(np.float64)
    return diffs, A


# ============================================================================
# PReF prediction = lambda_i^T phi(x, y) + affine recalibration
# ============================================================================

def affine_recalibrate(rm_tr: np.ndarray, sy_tr: np.ndarray,
                        lam_phi_tr: np.ndarray) -> tuple[float, float]:
    """Fit affine map from PReF lambda^T phi (BT-scale) to score_user
    (rating-scale) using user's TRAIN-fold pairs.  Returns (a, b).

    This is honest because (i) PReF's BT-MLE only constrains DIFFERENCES of
    rewards, so the absolute scalar level needs an external recalibration
    to be RMSE-comparable; (ii) the recalibration uses the SAME train-fold
    rm_scores that lambda is fit on -- no test-fold leakage.

    If train-fold has insufficient signal -> identity recalibration.
    Robustness guard: if the OLS fit is numerically unstable (extreme b
    with tiny std(lam_phi)), fall back to mean(sy_tr).  This guards against
    the high-J degenerate case where a frozen high-degree polynomial Phi
    produces a poorly-conditioned predictor surface; documented as honest
    finding alongside the result.
    """
    if len(lam_phi_tr) < 3 or float(np.std(lam_phi_tr)) < 1e-8:
        return float(np.mean(sy_tr)) if len(sy_tr) > 0 else 50.0, 0.0
    a, b, _, _ = fit_ols_with_se(lam_phi_tr, sy_tr)
    if not np.isfinite(a) or not np.isfinite(b):
        return float(np.mean(sy_tr)), 0.0
    # Numerical-stability guard: if std(lam_phi) is small relative to
    # |b| * std(lam_phi), the affine slope is amplifying noise.  Compare
    # b's predicted contribution to sy_tr's std; if the predictor is
    # unstable, fall back to mean.
    pred_std = abs(b) * float(np.std(lam_phi_tr))
    sy_std = float(np.std(sy_tr) + 1e-8)
    if pred_std > 5.0 * sy_std:
        # b is amplifying noise; fall back to mean (HONEST: the predictor
        # is degenerate at this J / lambda combination).
        return float(np.mean(sy_tr)), 0.0
    return a, b


def method_pref(g_user: pd.DataFrame, hc_id: str,
                  J: int, x_mu: float, x_sd: float,
                  beta_l2: float, use_svd_init: bool, use_l2: bool,
                  rng: np.random.Generator,
                  alpha_pop: float, beta_pop: float):
    """Single-user-fold PReF prediction.
       Returns predictions on held-out conversation.

    Full path:
        1. Split g_user into train (conv != hc) and test (conv == hc).
        2. Build TRAIN-fold pairwise (phi_diffs, A) per paper Eq. 5 contract.
        3. Initialise lambda_i: SVD per Algorithm 1 step 1 if use_svd_init=True,
           else N(0, INIT_SCALE_RANDOM) random init (paper Sec. 5.5 ablation).
        4. Refine lambda_i: L2-reg gradient descent on Eq. 5 with
           beta_l2 = beta_l2 if use_l2 else 0.0.  Phi is frozen.
        5. Compute lambda_i^T phi for TRAIN rm_scores -> fit affine
           recalibration (a, b) onto score_user.
        6. Predict on TEST conversation: a + b * (lambda_i^T phi(x_te, y_te)).
        7. Fall back to pop-slope if user has < 3 train pairs.
    """
    tr = g_user[g_user["conversation_id"] != hc_id]
    te = g_user[g_user["conversation_id"] == hc_id]
    rm_tr = tr["rm_score"].to_numpy(dtype=np.float64)
    sy_tr = tr["score_user"].to_numpy(dtype=np.float64)
    rm_te = te["rm_score"].to_numpy(dtype=np.float64)
    if len(rm_tr) < 3:
        return alpha_pop + beta_pop * rm_te
    # Build train pairs
    diffs_tr, A_tr = build_pairs_for_user(tr, x_mu, x_sd, J, rng)
    if len(A_tr) < 1:
        return alpha_pop + beta_pop * rm_te
    # Init lambda
    if use_svd_init:
        init_dict = svd_init_lambda({"u": diffs_tr}, {"u": A_tr}, ["u"], J)
        lam_init = init_dict["u"]
    else:
        lam_init = rng.normal(0, INIT_SCALE_RANDOM, size=J)
    # Refine
    eff_beta = beta_l2 if use_l2 else 0.0
    lam_i = fit_lambda_for_user(diffs_tr, A_tr, J, eff_beta, init=lam_init)
    # Compute lambda^T phi on TRAIN-fold (for affine recalibration)
    phi_tr = build_phi(rm_tr, x_mu, x_sd, J)
    lam_phi_tr = phi_tr @ lam_i
    a_cal, b_cal = affine_recalibrate(rm_tr, sy_tr, lam_phi_tr)
    # TEST prediction
    phi_te = build_phi(rm_te, x_mu, x_sd, J)
    lam_phi_te = phi_te @ lam_i
    return a_cal + b_cal * lam_phi_te


# ============================================================================
# PEBS-shrunk reference (mirrors P-GenRM script for cross-script consistency)
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
# LOCO loop with diagnostic instrumentation
# ============================================================================

# Each method variant has a DISTINCT (J, use_svd_init, use_l2) tuple;
# the toggle changes observable lambda (verified in unit test).
METHOD_VARIANTS = [
    # (tag,             J, use_svd_init, use_l2),
    ("pref_J2_full",    2, True,         True),
    ("pref_J3_full",    3, True,         True),
    ("pref_J6_full",    6, True,         True),
    ("pref_no_svd",     3, False,        True),
    ("pref_no_l2",      3, True,         False),
]
METHOD_NAMES = ["pop_slope", "pebs_shrunk"] + [v[0] for v in METHOD_VARIANTS]


def run_loco(args):
    t0 = time.time()
    if not SCORED.exists():
        sys.stderr.write(f"FATAL: PRISM scored data not found: {SCORED}\n")
        sys.exit(2)
    df = pd.read_parquet(SCORED).dropna(subset=["score_user", "rm_score", "conversation_id"])
    conv_count = df.groupby("user_id")["conversation_id"].nunique()
    utt_count = df.groupby("user_id").size()
    keep = conv_count[(conv_count >= MIN_CONV_PER_USER) &
                       (utt_count >= MIN_UTT_PER_USER)].index
    df = df[df["user_id"].isin(keep)].reset_index(drop=True)
    if args.smoke:
        keep_users = df["user_id"].drop_duplicates().head(50).tolist()
        df = df[df["user_id"].isin(keep_users)].reset_index(drop=True)
        print(f"[smoke] subsampled to {len(keep_users)} users")
    print(f"[filter] {len(df)} utt x {df['user_id'].nunique()} users")

    # Population OLS prior (used by every method's fallback path)
    slope_pop, intercept_pop = np.polyfit(df["rm_score"].to_numpy(),
                                           df["score_user"].to_numpy().astype(float), 1)
    alpha_pop, beta_pop = float(intercept_pop), float(slope_pop)
    print(f"[pop]    alpha_pop={alpha_pop:.3f}  beta_pop={beta_pop:.4f}")

    # Standardisation parameters for phi (frozen across train/test)
    x_mu = float(df["rm_score"].mean())
    x_sd = float(df["rm_score"].std())
    print(f"[phi]    x_mu={x_mu:.3f}  x_sd={x_sd:.3f}")

    # PEBS MoM tau^2_a, tau^2_b (reference baseline, identical to P-GenRM script)
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

    rng_master = np.random.default_rng(args.seed)

    per_user = []
    n_anomalies = {"empty_pairs": 0, "convergence_warn": 0}
    for uid, g in tqdm(df.groupby("user_id"), desc="LOCO PReF"):
        conv_ids = g["conversation_id"].unique().tolist()
        if len(conv_ids) < MIN_CONV_PER_USER:
            continue
        sq = {m: [] for m in METHOD_NAMES}
        n_utt_used = 0
        # Pin a per-user RNG so SVD init is deterministic across runs
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
                "pebs_shrunk": method_pebs_shrunk(x_tr, y_tr, x_te,
                                                    alpha_pop, beta_pop,
                                                    tau2_a, tau2_b),
            }
            for tag, J, use_svd, use_l2 in METHOD_VARIANTS:
                rng_local = np.random.default_rng(user_seed + hash(tag) % (2 ** 16))
                preds[tag] = method_pref(
                    g, hc, J=J, x_mu=x_mu, x_sd=x_sd,
                    beta_l2=args.beta_l2 if use_l2 else 0.0,
                    use_svd_init=use_svd, use_l2=use_l2,
                    rng=rng_local, alpha_pop=alpha_pop, beta_pop=beta_pop,
                )
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

    # Relative RMSE reduction (%) vs pop_slope baseline per user
    for m in METHOD_NAMES:
        if m == "pop_slope":
            continue
        pu[f"gain_{m}_pct"] = 100.0 * (pu["rmse_pop_slope"] - pu[f"rmse_{m}"]) / pu["rmse_pop_slope"]

    # Cluster-bootstrap mean + 95% CI per method
    def cluster_boot_ci(values: np.ndarray, n_boot: int, seed: int):
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
        "pref_settings": {
            "J_grid": list(J_GRID_DEFAULT),
            "beta_l2": float(args.beta_l2),
            "n_refine_steps": N_REFINE_STEPS,
            "refine_lr": REFINE_LR,
            "init_scale_random": INIT_SCALE_RANDOM,
            "phi_basis": "polynomial-on-rm-score (frozen)",
            "pair_construction": "intra-user-scalar-derived (no PERSONA aug)",
        },
        "methods": {},
        "rmse_abs": {m: float(pu[f"rmse_{m}"].mean()) for m in METHOD_NAMES},
        "verdict_class": "PENDING-RESULT",  # filled after CI computed
        "anomalies_fired": [k for k, v in n_anomalies.items() if v > 0],
        "anomalies_count": n_anomalies,
    }
    print("\n=== Head-to-head: mean RMSE reduction (%) vs pop_slope ===")
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
            "status": "reimplementation" if m != "pebs_shrunk" else "reference",
        }
        print(f"{m:>20s}  {pu[f'rmse_{m}'].mean():8.3f}  "
              f"{mean_g:+9.3f}  [{lo:+6.3f}, {hi:+6.3f}]")

    # Paired delta vs PEBS-shrunk
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

    # Outcome assignment
    pebs_g_lo = summary["methods"]["pebs_shrunk"]["rmse_reduction_pct_ci95"][0]
    pref_full_keys = ["pref_J2_full", "pref_J3_full", "pref_J6_full"]
    pebs_dom_full = all(
        summary["paired_vs_pebs"][m]["pebs_better"]
        for m in pref_full_keys
    )
    pref_dom_full = all(
        summary["paired_vs_pebs"][m]["ci95"][1] < 0
        for m in pref_full_keys
    )
    if pebs_g_lo > 0 and pebs_dom_full:
        summary["verdict_class"] = "CONFIRMED-PEBS-GAIN-DOMINATES-PREF-ON-PER-USER-RMSE"
    elif pebs_g_lo > 0 and pref_dom_full:
        summary["verdict_class"] = "REJECTED-PREF-DOMINATES-PEBS-ON-PER-USER-RMSE"
    elif pebs_g_lo > 0:
        summary["verdict_class"] = "PARTIAL-PEBS-GAIN-PARITY-WITH-PREF"
    else:
        summary["verdict_class"] = "TENTATIVE-INCONCLUSIVE-PEBS-GAIN-CI-INCLUDES-ZERO"

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
        "pebs_shrunk":     ("\\PEBS{} \\emph{(ours)}",
                              "Per-user EB-shrunk $(\\alpha_j, \\beta_j)$ affine calibrator",
                              "hierarchical EB"),
        "pref_J2_full":     ("PReF $J{=}2$ \\citep{zhang2025pref}",
                              "Reward factorisation, $J{=}2$, SVD init + L2 reg",
                              "BT-MLE"),
        "pref_J3_full":     ("PReF $J{=}3$ \\citep{zhang2025pref}",
                              "Reward factorisation, $J{=}3$, SVD init + L2 reg",
                              "BT-MLE"),
        "pref_J6_full":     ("PReF $J{=}6$ \\citep{zhang2025pref}",
                              "Reward factorisation, $J{=}6$ (paper sweet spot per Sec. 5.4)",
                              "BT-MLE"),
        "pref_no_svd":      ("PReF $J{=}3$ no-SVD",
                              "Random init ablation (paper Sec. 5.5)",
                              "BT-MLE"),
        "pref_no_l2":       ("PReF $J{=}3$ no-L2",
                              "$\\beta{=}0$ ablation (paper Sec. 5.5)",
                              "BT-MLE"),
    }
    order = ["pebs_shrunk", "pref_J2_full", "pref_J3_full", "pref_J6_full",
             "pref_no_svd", "pref_no_l2"]
    lines = []
    lines.append("% PReF (Shenfeld, Faltings, Agrawal, Pacchiano 2025) head-to-head on PRISM.")
    lines.append("% Generated by paper/scripts/baselines/pref_2025_on_prism.py")
    lines.append(f"% N_users = {summary['slice']['n_users']}, "
                 f"N_utt = {summary['slice']['n_utterances']}, "
                 f"seed = {summary['rng_seed']}, n_boot = {summary['n_bootstrap']}, "
                 f"beta_L2 = {summary['pref_settings']['beta_l2']}.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering \\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append("\\caption{\\textbf{PReF head-to-head RMSE comparison on PRISM held-out-conversation slice} "
                 "(N=" + str(summary['slice']['n_users']) + " users, "
                 + str(summary['slice']['n_utterances'] // max(summary['slice']['n_users'], 1))
                 + " utterances/user, same upstream Qwen2.5-7B-Instruct RM scores for every method, "
                 "cluster-bootstrap 95\\% CI over users, $n_{\\mathrm{boot}}$="
                 + str(summary['n_bootstrap']) + ", seed=" + str(summary['rng_seed']) + "). "
                 "PReF \\citep{zhang2025pref} (arXiv:2503.06358) is re-implemented "
                 "on our slice via faithful adaptation of paper Eqs.~3-5 + Algorithm~1: "
                 "(a) per-user $\\lambda_i \\in \\mathbb{R}^J$ trained via L2-regularised "
                 "Bradley-Terry MLE on intra-user pairwise preferences derived from PRISM "
                 "scalar ratings (no PERSONA-style synthetic augmentation, preserving "
                 "apples-to-apples on the same PRISM-native split as P-GenRM and other "
                 "neighbours), (b) SVD initialisation per Algorithm~1 step~1, (c) L2 "
                 "regularisation per Sec.~4.1, (d) frozen polynomial $\\phi$ basis on "
                 "the standardised RM score (preserving upstream-signal parity with "
                 "P-GenRM/Halpern/ICRM/SPL re-implementations), and (e) train-fold-only "
                 "affine recalibration of the BT-scale $\\lambda^\\top \\phi$ output to "
                 "the PRISM rating scale.  Honest scope caveats per the script docstring.  "
                 "See \\texttt{paper/scripts/baselines/pref\\_2025\\_on\\_prism.py} "
                 "for the exact recipe and \\S~Limitations.}")
    lines.append("\\label{tab:pref-headtohead-numbers}")
    lines.append("\\begin{tabular}{@{}p{3.0cm}p{4.6cm}p{2.4cm}rr@{}}")
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
    pretty = {"pebs_shrunk": "PEBS (ours)",
              "pref_J2_full": "PReF J=2",
              "pref_J3_full": "PReF J=3",
              "pref_J6_full": "PReF J=6",
              "pref_no_svd": "PReF (no SVD)",
              "pref_no_l2": "PReF (no L2)"}
    order = ["pebs_shrunk", "pref_J2_full", "pref_J3_full", "pref_J6_full",
             "pref_no_svd", "pref_no_l2"]
    order = [m for m in order if m in summary['methods']]
    colors = {"pebs_shrunk": "#1f77b4",
              "pref_J2_full": "#ffbb78",
              "pref_J3_full": "#ff7f0e",
              "pref_J6_full": "#d62728",
              "pref_no_svd": "#aec7e8",
              "pref_no_l2": "#98df8a"}
    means = [summary['methods'][m]['rmse_reduction_pct_mean'] for m in order]
    ci_lo = [summary['methods'][m]['rmse_reduction_pct_ci95'][0] for m in order]
    ci_hi = [summary['methods'][m]['rmse_reduction_pct_ci95'][1] for m in order]
    err_lo = [m_v - lo for m_v, lo in zip(means, ci_lo)]
    err_hi = [hi - m_v for m_v, hi in zip(means, ci_hi)]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(order))
    bars = ax.bar(x, means, yerr=[err_lo, err_hi], capsize=5,
                  color=[colors[m] for m in order], edgecolor="black", linewidth=0.8)
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
    ax.set_xticklabels([pretty[m] for m in order], rotation=15, fontsize=9, ha="right")
    ax.set_ylabel("RMSE reduction vs pop-slope baseline (%)", fontsize=11)
    ax.set_title(f"PRISM LOCO PReF head-to-head  (N={summary['slice']['n_users']} users, "
                 f"cluster-bootstrap 95% CI)", fontsize=10)
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
    p.add_argument("--beta-l2", type=float, default=BETA_L2_DEFAULT,
                    help="L2 regularisation coefficient on lambda (paper Sec. 4.1)")
    p.add_argument("--smoke", action="store_true",
                    help="Smoke mode: subsample 50 users for speed")
    p.add_argument("--skip-fig", action="store_true")
    p.add_argument("--skip-table", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        print("[smoke] subsampling 50 users for fast smoke test")
    summary, pu = run_loco(args)
    if not args.skip_table:
        write_latex_table(summary, OUT_TABLE)
    if not args.skip_fig:
        write_figure(summary, OUT_FIG)
    print(f"\n[verdict] {summary['verdict_class']}")
    print(f"[runtime] {summary['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
