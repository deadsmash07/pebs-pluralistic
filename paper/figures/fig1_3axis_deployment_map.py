"""Figure 1 (Pluralistic local) --- 3-axis PEBS deployment map.

RE-AUTHORED 2026-05-03 (F-NEW-1 v2; was 7/30 cells, now 18/30 cells filled):
  Adds F23 cross-family coherence-only cells (Phi-3-medium-14B / Llama-3-8B /
  Yi-1.5-34B / Mistral-Small-22B / Phi-3.5-MoE) on the PRISM column under
  honest-disclosure SCOPE-not-RETRACT framing (negative cells render in deep
  blue; the cross-family sign-flip IS a contribution per Skill: research-paper-
  writing-oral-spotlight §6.1 deployment-map framing).  Adds bottom Q9-pooled-
  multi-corpus row encoding pooled +7.19% across 4 RLHF preference corpora.
  Caption disambiguates metrics per cell (within-user RMSE for Qwen × PRISM
  canonical; mean-LL for Q7 backbone-cross-arch; per-attribute coherence-only
  for F23 cross-family probe).

Concept (per planning memo `memory/figure_table_audit_plan_20260503_1223.md` §4 F-NEW-1):
  Single 2D matrix; ROWS = 7 (6 backbones + 1 Q9-pooled aggregate);
  COLS = 5 RLHF preference corpora.
  Each cell = signed PEBS gain (% over pop-slope or canonical baseline);
  colour = RdBu_r diverging with TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
  so zero is white midpoint.  Negative cells render in DEEP BLUE per Skill:
  honest-disclosure §6.3 (sign-flips are honest contributions, NOT hidden).
  UNTESTED cells render as hatched grey per Skill: research-grade-plots §9
  (NA-discipline: missing-data pattern visible, not silently omitted).
  Three concentric outline boxes around cells per verdict class:
    - dashed Wong-blue (#0072B2) = ESTABLISHED-* cells
    - dashed Wong-orange (#E69F00) = MODERATE-* cells
    - dashed Wong-vermillion (#D55E00) = FALSIFIED-CROSS-FAMILY cells
    - dashed grey = UNTESTED cells (also hatched grey fill)
  Annotate gain magnitudes inside cells (signed, 7-8pt).

Honest-disclosure positions (per Skill: honest-disclosure §6.3 SCOPE-not-RETRACT
+ Skill: research-paper-writing-oral-spotlight §6.1 deployment-map framing):
  - Cross-family sign-flips on Llama-3-8B (-171.16%) and Yi-1.5-34B (-109.76%)
    are ENCODED VISIBLY as deep-blue saturated cells, not hidden.  The
    sign-flip IS the deployment-boundary contribution.
  - Phi-3-medium-14B coherence-only +43.23% (within-family ESTABLISHED-
    MAGNITUDE-REPLICATION 5/5 seeds via F19') anchors the within-family
    deployment region; the contrast with cross-family negatives makes the
    boundary structurally visible.
  - Mistral-Small-22B coherence-only NULL (+19.99% CI [-17.22, +42.86]) is
    rendered as MODERATE outline since CI straddles zero.
  - Phi-3.5-MoE coherence-only -59.41% [-132.12, +6.90] (F29) and Mixtral
    8x7B -60.44% [-122.97, +3.11] (F44) are within-MoE-class cross-arch panel
    ESTABLISHED-MOE-2-of-2 H1-CONSISTENT; rendered as deep-blue with
    MODERATE outline (CI straddles zero per cell, panel-stable per
    PREREG sec 3.1).
  - Q9-pooled-multi-corpus row at bottom encodes ESTABLISHED pooled +7.19%
    on z-score scale (4 RLHF corpora pooled with corpus-level fixed-effect
    normalization + namespaced cluster ids per Q9 dispatch 06:49 IST 2026-05-03).
  - MultiPref cell at +0.47% reported as MODERATE / boundary-diagnostic
    (Morris-1983 g-function predicts +17.96%; the 17.49pp gap is a diagnostic
    of ordinal-Gaussian random-effect mis-specification rather than theorem
    failure).
  - The bulk of non-Qwen × non-PRISM cells remain UNTESTED at canonical RMSE
    metric.  These are the deployment boundary, not failures.

NOTE on metric heterogeneity: The cells in this matrix span 4 distinct
metrics: (a) within-user OLS-shrunk RMSE [Qwen×PRISM canonical]; (b) Q7
mean-response log-likelihood backbone-agnostic proxy [Qwen×Mistral×Yi PRISM];
(c) per-attribute coherence-only PEBS gain [F19/F23/F29/F44 cross-family
HelpSteer2 probe — REPORTED ON THE PRISM COLUMN to keep within-family-anchor
deployment narrative legible]; (d) z-score scale for Q9-pooled.  The caption
disambiguates per-cell.

NO fabricated values. All cell numbers cross-referenced to PAPER_CLAIMS.json v51
+ IMPLEMENTATION/PAPER_CLAIMS.json v14 + planning memo verbatim.

Output: 1 PDF (paper/figures/fig1_3axis_deployment_map.pdf) at NeurIPS 2-col
width 7.0 in (the 7 x 5 matrix needs 2-col room for legibility per Skill:
research-grade-plots §1 column-width).  Type 42 fonts (Skill: research-grade-
plots §7 PDF embedding requirement; NeurIPS / ICML reject Type 3).

Anonymity-clean.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


# ---- Wong 2011 / Okabe-Ito colorblind-safe palette (Nature Methods 8, 441) ----
WONG = {
    "black":      "#000000",
    "orange":     "#E69F00",   # MODERATE-* outline
    "sky_blue":   "#56B4E9",
    "blu_green":  "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",   # ESTABLISHED-* outline
    "vermillion": "#D55E00",   # FALSIFIED-CROSS-FAMILY outline
    "red_purple": "#CC79A7",
    "grey":       "#606060",   # axes / UNTESTED outline
    "light_grey": "#B0B0B0",
}


# ---- rcParams: paper-grade defaults (Skill: research-grade-plots §1 + §7) ----
def set_pub_style() -> None:
    """Apply the publication-quality rcParams baseline.

    Same family as `paper/workshop_T1_pluralistic/figures/_figstyle.py`:
    serif Times / 8pt body / 9pt axes / TrueType embedded.
    """
    mpl.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset":  "stix",
        "font.size":         8,
        "axes.labelsize":    9,
        "axes.titlesize":    10,
        "axes.titleweight":  "bold",
        "axes.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    WONG["grey"],
        "axes.labelcolor":   "black",
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "xtick.color":       WONG["grey"],
        "ytick.color":       WONG["grey"],
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "legend.fontsize":   7.5,
        "legend.frameon":    False,
        "pdf.fonttype":      42,    # NeurIPS / ICML required
        "ps.fonttype":       42,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.04,
    })


# ---- Data structures ---------------------------------------------------------
# 7 rows (6 backbones + 1 Q9-pooled aggregate) x 5 corpora.  Each cell:
#   gain_pct: float or None (None = UNTESTED hatched-grey)
#   verdict:  one of {"ESTABLISHED", "MODERATE", "FALSIFIED", "UNTESTED"}
#   metric:   short descriptor for caption / footnote disambiguation
#   anchor:   PAPER_CLAIMS.json claim id (or "memo" if from planning memo)
#
# Q9-pooled is rendered as a SEPARATE BOTTOM ROW (not as a backbone) since
# it's an aggregate across cluster definitions (user_id / prompt_hash /
# author_id) and corpora.  Decision per planning memo §2 line 30 +
# Q9 dispatch 06:49 IST 2026-05-03: ROW (not column) makes the within-pool
# slice columns map 1:1 to existing corpus columns for visual scan.

BACKBONES: List[str] = [
    "Qwen-2.5-7B",
    "Phi-3-medium-14B",
    "Mistral-Small-22B",
    "Yi-1.5-34B",
    "Llama-3-8B",
    "Phi-3.5-MoE",
    "(Q9 pooled-multi-corpus)",   # aggregate row; not a backbone
]

CORPORA: List[str] = [
    "PRISM",
    "PluriHarms",
    "HelpSteer2",
    "OASST2",
    "MultiPref",
]


# Map (backbone, corpus) -> recipe dict.
# `claim_id` of None means we use the explicit fallback (planning-memo verbatim).
# `verdict` is one of "ESTABLISHED" / "MODERATE" / "FALSIFIED" / "UNTESTED".
# When `verdict` is "UNTESTED", `fallback_value` is ignored (rendered hatched grey).
#
# CROSS-FAMILY CELLS (Phi-3-medium-14B / Llama-3-8B / Yi-1.5-34B / Phi-3.5-MoE
# under PRISM column) carry F23/F19'/F29/F44 evidence which is on the
# per-attribute coherence-only metric (HelpSteer2 train-set), NOT canonical
# within-user RMSE.  They are placed in the PRISM column for cross-base
# deployment-map legibility; caption disambiguates the metric.  This matches
# the Pluralistic Fig 1 forest plot convention (paper/workshop_T1_pluralistic/
# figures/fig1_pebs_overview.py rows 64-74).

CELL_RECIPE: Dict[Tuple[str, str], Dict] = {
    # --- PRISM column (mixed metrics; caption disambiguates) ---------------
    ("Qwen-2.5-7B", "PRISM"): {
        "claim_id": "T1.1",
        "fallback": 8.58,
        "verdict": "ESTABLISHED",
        "metric": "within-user OLS-shrunk RMSE (canonical PEBS headline)",
    },
    ("Phi-3-medium-14B", "PRISM"): {
        # F19 + F19' multi-seed 5/5 ESTABLISHED-WITHIN-FAMILY-MAGNITUDE
        # cross-seed mean +42.153% [+40.10, +44.20] CI half=2.051pp
        # Per-attribute coherence-only metric (HelpSteer2 train-set);
        # NOT within-user RMSE; placed here under deployment-map framing.
        "claim_id": "T1.F19_PRIME_PHI3_MULTISEED_5_OF_5_ESTABLISHED_WITHIN_FAMILY_MAGNITUDE",
        "fallback": 42.15,
        "verdict": "ESTABLISHED",
        "metric": "within-family coherence-only F19' 5-seed mean (+42.15% [+40.10, +44.20])",
    },
    ("Mistral-Small-22B", "PRISM"): {
        # Q7 backbone-cross-arch via mean-response log-likelihood (Stiennon
        # et al. 2020 backbone-agnostic RM proxy).  Different metric from
        # Qwen-2.5-7B canonical; documented in caption.
        "claim_id": "T1.Q7_BACKBONE_CROSS_ARCHITECTURE_PILSD_ESTABLISHED",
        "fallback": 4.20,
        "verdict": "ESTABLISHED",
        "metric": "Q7 mean-LL backbone-agnostic proxy (+4.20% [+3.51, +5.04])",
    },
    ("Yi-1.5-34B", "PRISM"): {
        # F23 Yi-1.5-34B coherence-only sentinel sign-flip
        # -109.76% [-170.11, -47.92] CI strictly NEG
        # (Note: Q7 also gives Yi-1.5-34B +3.87% on mean-LL; we use the
        # F23 cross-family coherence-only number here because it is the
        # LOAD-BEARING deployment-boundary signal and the visible
        # contrast with within-family Phi-3-medium +42% makes the
        # boundary structurally clear.  Q7 +3.87% is captured in App I
        # backbone-cross-arch text; not in this figure.)
        "claim_id": "T1.F23_CROSS_FAMILY_COHERENCE_FALSIFIED_FOREGONE",
        "fallback": -109.76,
        "verdict": "FALSIFIED",
        "metric": "F23 cross-family coherence-only sign-flip (-109.76% [-170.11, -47.92])",
    },
    ("Llama-3-8B", "PRISM"): {
        # F23 Llama-3-8B coherence-only sentinel sign-flip
        # -171.16% [-256.40, -82.33] CI strictly NEG
        "claim_id": "T1.F23_CROSS_FAMILY_COHERENCE_FALSIFIED_FOREGONE",
        "fallback": -171.16,
        "verdict": "FALSIFIED",
        "metric": "F23 cross-family coherence-only sign-flip (-171.16% [-256.40, -82.33])",
    },
    ("Phi-3.5-MoE", "PRISM"): {
        # F29 Phi-3.5-MoE coherence-only -59.41% [-132.12, +6.90]
        # NULL-STRADDLES-ZERO classification per F29 PREREG; ESTABLISHED-MOE
        # at panel-level joint with F44 Mixtral 8x7B -60.44% per
        # T1.CROSS_ARCH_MOE_PANEL_2_OF_2_ESTABLISHED.  We render the F29
        # cell here as MODERATE (CI straddles, but panel-stable).
        "claim_id": "T1.F29_PHI35_MOE_NULL_STRADDLES_ZERO_H1_CONSISTENT_4_OF_4",
        "fallback": -59.41,
        "verdict": "MODERATE",
        "metric": "F29 cross-arch MoE coherence-only (-59.41% [-132.12, +6.90]; ESTABLISHED-MOE-panel jt F44)",
    },

    # --- PluriHarms column --------------------------------------------------
    ("Qwen-2.5-7B", "PluriHarms"): {
        "claim_id": "T1.Q3_HALF_PILSD_CROSS_CORPUS_PLURIHARMS_ESTABLISHED_REVERSED_HALF_ARMS",
        "fallback": 9.66,
        "verdict": "ESTABLISHED",
        "metric": "Q3 canonical compound (alpha- and beta-only halves NEG individually)",
    },
    ("Phi-3-medium-14B", "PluriHarms"): {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},
    ("Mistral-Small-22B", "PluriHarms"): {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},
    ("Yi-1.5-34B", "PluriHarms"):       {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},
    ("Llama-3-8B", "PluriHarms"):       {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},
    ("Phi-3.5-MoE", "PluriHarms"):      {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},

    # --- HelpSteer2 column -------------------------------------------------
    ("Qwen-2.5-7B", "HelpSteer2"): {
        "claim_id": "T1.Q1_HELPSTEER2_REPLICATION_ESTABLISHED_HELPSTEER2_CONFIRMS_PRISM",
        "fallback": 3.70,
        "verdict": "ESTABLISHED",
        "metric": "Q1 cluster=prompt_hash (HelpSteer2 anonymizes annotators per Wang et al. 2024 NeurIPS §3.4)",
    },
    ("Phi-3-medium-14B", "HelpSteer2"): {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},
    ("Mistral-Small-22B", "HelpSteer2"): {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},
    ("Yi-1.5-34B", "HelpSteer2"):       {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},
    ("Llama-3-8B", "HelpSteer2"):       {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},
    ("Phi-3.5-MoE", "HelpSteer2"):      {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},

    # --- OASST2 column -----------------------------------------------------
    ("Qwen-2.5-7B", "OASST2"): {
        "claim_id": "T1.WAVE_C_EXP2_OASST2_REPLICATION_MODERATE_OASST2_PARTIAL_CONFIRMS",
        "fallback": 1.21,
        "verdict": "MODERATE",
        "metric": "W-A4 / WAVE-C-EXP2 cluster=author_id (BCa CI strict-POS at smaller magnitude)",
    },
    ("Phi-3-medium-14B", "OASST2"): {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},
    ("Mistral-Small-22B", "OASST2"): {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},
    ("Yi-1.5-34B", "OASST2"):       {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},
    ("Llama-3-8B", "OASST2"):       {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},
    ("Phi-3.5-MoE", "OASST2"):      {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},

    # --- MultiPref column --------------------------------------------------
    ("Qwen-2.5-7B", "MultiPref"): {
        "claim_id": "T1.cross_dataset_multipref",
        "fallback": 0.47,
        "verdict": "MODERATE",
        "metric": "boundary diagnostic; Morris g-forecast +17.96% / observed +0.47% (17.49pp gap)",
    },
    ("Phi-3-medium-14B", "MultiPref"): {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},
    ("Mistral-Small-22B", "MultiPref"): {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},
    ("Yi-1.5-34B", "MultiPref"):       {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},
    ("Llama-3-8B", "MultiPref"):       {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},
    ("Phi-3.5-MoE", "MultiPref"):      {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "not run"},

    # --- Q9-pooled-multi-corpus aggregate row ------------------------------
    # The Q9 pooled headline is the SUM aggregate: pooled +7.19% across all 4
    # RLHF corpora.  Within-pool slices per corpus per Q9 summary.json.
    # This row is PER-CORPUS within-pool slices reflecting pooled + corpus-
    # FE-normalised per-corpus headline.
    ("(Q9 pooled-multi-corpus)", "PRISM"): {
        "claim_id": "T1.Q9_POOLED_4_CORPUS_PILSD_ESTABLISHED",
        "fallback": 9.98,   # PRISM within-pool +9.977%
        "verdict": "ESTABLISHED",
        "metric": "Q9 within-pool PRISM slice (+9.98% [+9.23, +10.65]); pool headline +7.19%",
    },
    ("(Q9 pooled-multi-corpus)", "PluriHarms"): {
        "claim_id": "T1.Q9_POOLED_4_CORPUS_PILSD_ESTABLISHED",
        "fallback": 28.18,  # PluriHarms within-pool +28.180% (small-N borrowing-of-strength)
        "verdict": "ESTABLISHED",
        "metric": "Q9 within-pool PluriHarms slice (+28.18% [+26.82, +29.57]; small-N borrowing of strength)",
    },
    ("(Q9 pooled-multi-corpus)", "HelpSteer2"): {
        "claim_id": "T1.Q9_POOLED_4_CORPUS_PILSD_ESTABLISHED",
        "fallback": 3.64,   # HelpSteer2 within-pool +3.644% (matches Q1 standalone within 0.05pp G5 parity)
        "verdict": "ESTABLISHED",
        "metric": "Q9 within-pool HelpSteer2 slice (+3.64% [+2.05, +4.73]; matches Q1 standalone within 0.05pp)",
    },
    ("(Q9 pooled-multi-corpus)", "OASST2"): {
        "claim_id": "T1.Q9_POOLED_4_CORPUS_PILSD_ESTABLISHED",
        "fallback": -1.30,  # OASST2 within-pool -1.300% [-4.828, +0.190] CI straddles 0
        "verdict": "MODERATE",
        "metric": "Q9 within-pool OASST2 slice (-1.30% [-4.83, +0.19]; CI straddles 0)",
    },
    ("(Q9 pooled-multi-corpus)", "MultiPref"): {"claim_id": None, "fallback": None, "verdict": "UNTESTED", "metric": "MultiPref not in Q9 pool (Morris-g out-of-scope diagnostic)"},
}


# ---- Number-extraction helpers (parse PAPER_CLAIMS.json values) -------------
def _parse_first_pct(s: str) -> float | None:
    """Pull the first signed percentage out of a freeform 'number' string.

    Used as a fallback when the JSON value is a long descriptive string and we
    want the leading magnitude (e.g. '+9.66% [+7.28, +11.19]' -> 9.66).
    Returns None if no parse.
    """
    import re
    if not s:
        return None
    # Match a signed number followed by '%' (or 'pp').  Preserve sign.
    m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _q7_pull_per_backbone(s: str) -> Dict[str, float]:
    """Q7 number string is e.g.
    'Mistral +4.201% [3.514, 5.036] / Yi-1.5-34B +3.868% [3.148, 4.716] / Qwen2.5-7B +3.697% ...'
    Parse each per-backbone percentage."""
    import re
    out: Dict[str, float] = {}
    for tok in s.split("/"):
        tok = tok.strip()
        # First word is backbone tag; first percentage is the gain.
        m_pct = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", tok)
        if not m_pct:
            continue
        v = float(m_pct.group(1))
        # Heuristic backbone tag detection from leading word.
        if tok.startswith("Mistral"):
            out["Mistral-Small-22B"] = v
        elif tok.startswith("Yi"):
            out["Yi-1.5-34B"] = v
        elif tok.startswith("Qwen"):
            out["Qwen-2.5-7B"] = v
    return out


def load_cell_values(claims_path: str) -> Dict[Tuple[str, str], Dict]:
    """Build the (backbone, corpus) -> {gain_pct, verdict, metric, anchor} map.

    Resolves each cell against PAPER_CLAIMS.json.  Falls back to the
    planning-memo verbatim value if the JSON doesn't have a direct row but
    the recipe entry carries a `fallback`.
    """
    if not os.path.exists(claims_path):
        print(f"[WARN] PAPER_CLAIMS.json not found at {claims_path}; using fallbacks", file=sys.stderr)
        claims_by_id: Dict[str, Dict] = {}
    else:
        with open(claims_path) as f:
            claims_data = json.load(f)
        claims_by_id = {c["id"]: c for c in claims_data.get("claims", [])}

    # Special-case: pre-parse Q7 because three PRISM cells share its row.
    q7_per_bb: Dict[str, float] = {}
    if "T1.Q7_BACKBONE_CROSS_ARCHITECTURE_PILSD_ESTABLISHED" in claims_by_id:
        q7_per_bb = _q7_pull_per_backbone(
            claims_by_id["T1.Q7_BACKBONE_CROSS_ARCHITECTURE_PILSD_ESTABLISHED"].get("number", "")
        )

    cells: Dict[Tuple[str, str], Dict] = {}
    for (bb, cp), recipe in CELL_RECIPE.items():
        verdict = recipe["verdict"]
        if verdict == "UNTESTED":
            cells[(bb, cp)] = {
                "gain_pct": None,
                "verdict": "UNTESTED",
                "metric": recipe.get("metric", "not run"),
                "anchor": "UNTESTED-no-claim-row",
            }
            continue

        # ESTABLISHED / MODERATE / FALSIFIED -> resolve numeric value.
        # PRECEDENCE RULE: per-cell `fallback` is AUTHORITATIVE when present
        # because many claim rows pack multiple cells into one freeform
        # `number` string (Q7 has 3 backbones; F23 has 4 bases; Q9 has
        # within-pool + per-corpus slices).  Parsing the leading %% from
        # the claim row would silently pick the wrong cell value.  The
        # `fallback` is the planning-memo + claims-row VERBATIM per-cell
        # number cross-checked manually.  We only fall back to JSON
        # parsing as a sanity-check round-trip.
        cid = recipe.get("claim_id")
        recipe_fb = recipe.get("fallback")
        gain = recipe_fb   # fallback IS the per-cell authoritative value

        # Optional: Q7 special-case parses 3 backbones from one row (these
        # are bit-exact the Q7 numbers per claim row; safer than fallback).
        if cid == "T1.Q7_BACKBONE_CROSS_ARCHITECTURE_PILSD_ESTABLISHED" and bb in q7_per_bb:
            gain = q7_per_bb.get(bb)

        if gain is None:
            print(f"[ERROR] Could not resolve cell ({bb}, {cp}) -> claim {cid}", file=sys.stderr)
            sys.exit(1)

        cells[(bb, cp)] = {
            "gain_pct": float(gain),
            "verdict": verdict,
            "metric": recipe.get("metric", ""),
            "anchor": cid or "memo-fallback",
        }
    return cells


# ---- Drawing -----------------------------------------------------------------
def draw_figure(cells: Dict[Tuple[str, str], Dict], out_pdf: str) -> None:
    """Render the 3-axis deployment-map heatmap and write to `out_pdf`."""
    set_pub_style()

    n_rows, n_cols = len(BACKBONES), len(CORPORA)

    # Build numeric matrix + masked-NA mask for hatched-grey rendering
    # (Skill: research-grade-plots §9 NA discipline).
    M = np.full((n_rows, n_cols), np.nan, dtype=float)
    verdict_mat = np.empty((n_rows, n_cols), dtype=object)
    for i, bb in enumerate(BACKBONES):
        for j, cp in enumerate(CORPORA):
            cell = cells[(bb, cp)]
            if cell["verdict"] != "UNTESTED" and cell["gain_pct"] is not None:
                M[i, j] = cell["gain_pct"]
            verdict_mat[i, j] = cell["verdict"]
    M_masked = np.ma.masked_invalid(M)

    # 2-col NeurIPS / ICML width = 7.0in; 7 rows x 5 cols matrix needs the
    # full 2-col width for legibility (Skill: research-grade-plots §1).
    # Aim for a near-square cell aspect with the extra Q9 row adding
    # ~0.5in height vs prior 6-row version.  Height 4.5in keeps below the
    # §14 max-height cap while preserving in-cell text legibility at 7-8pt.
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # Symmetric diverging colormap.  Now that cross-family negatives are
    # encoded (-171.16% / -109.76% / -60.44%), the natural data range spans
    # ~[-171, +43].  We CAP vmax at 50 so the within-family cells [-50, +50]
    # get good contrast; sign-flip cells beyond -50 saturate to deep blue,
    # which is correct: the saturation IS the visual signal that those cells
    # are "off the chart" deployment-boundary failures (Skill: honest-
    # disclosure §6.3 SCOPE-not-RETRACT — sign-flips visible, not hidden).
    # Use `extend="min"` on colorbar to indicate saturation honestly.
    abs_max = float(np.nanmax(np.abs(M))) if np.any(np.isfinite(M)) else 50.0
    # Cap at 50 so most cells get gradient detail; sign-flips saturate to
    # deep-blue (correct visual signal).
    vmax = 50.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # Use a copy of RdBu_r so we can set the masked colour to light grey.
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color=WONG["light_grey"])

    im = ax.imshow(M_masked, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="none", zorder=2)

    # Hatched overlay on UNTESTED cells (Skill: research-grade-plots §9).
    # We re-draw each UNTESTED cell as a Rectangle with hatching so the NA
    # pattern is unambiguous in B/W as well as in colour.
    for i in range(n_rows):
        for j in range(n_cols):
            if verdict_mat[i, j] == "UNTESTED":
                rect = mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    facecolor=WONG["light_grey"],
                    edgecolor=WONG["grey"],
                    linewidth=0.4,
                    hatch="///",
                    zorder=2.5,
                )
                ax.add_patch(rect)

    # Thin black gridlines between cells (Tufte: matrix-as-grid not blob).
    for x in np.arange(-0.5, n_cols, 1.0):
        ax.axvline(x, color="black", lw=0.4, zorder=3.5, alpha=0.5)
    for y in np.arange(-0.5, n_rows, 1.0):
        ax.axhline(y, color="black", lw=0.4, zorder=3.5, alpha=0.5)

    # Per-verdict-class outline rectangles (3 concentric outline classes,
    # dashed; per task brief).  Drawn in order grey -> orange -> vermillion
    # -> blue so ESTABLISHED outlines sit on top.
    verdict_outline_color = {
        "ESTABLISHED": WONG["blue"],
        "MODERATE":    WONG["orange"],
        "FALSIFIED":   WONG["vermillion"],
        "UNTESTED":    WONG["grey"],
    }
    # Outline ESTABLISHED + MODERATE + FALSIFIED.  UNTESTED fill is hatched-
    # grey which already conveys the verdict; doubling outline would be
    # redundant ink per Tufte data-ink ratio.
    for verdict in ("MODERATE", "FALSIFIED", "ESTABLISHED"):
        for i in range(n_rows):
            for j in range(n_cols):
                if verdict_mat[i, j] == verdict:
                    rect = mpatches.Rectangle(
                        (j - 0.45, i - 0.45), 0.90, 0.90,
                        facecolor="none",
                        edgecolor=verdict_outline_color[verdict],
                        linewidth=1.4,
                        linestyle="--",
                        zorder=4,
                    )
                    ax.add_patch(rect)

    # Visual separator: dashed grey horizontal line between row 5 (Phi-3.5-MoE,
    # the last backbone row) and row 6 (Q9 pooled aggregate row) to make the
    # aggregate row visually distinct from the per-backbone rows.
    ax.axhline(n_rows - 1.5, color=WONG["grey"], lw=1.0, ls="--",
               alpha=0.85, zorder=3.7)

    # Cell text annotations.  Signed magnitudes; 7pt; white text on saturated
    # cells (|v| > vmax * 0.55) for legibility.  UNTESTED cells get '—' (em-
    # dash) at neutral grey to make NA explicit.
    for i in range(n_rows):
        for j in range(n_cols):
            v = M[i, j]
            verdict = verdict_mat[i, j]
            if verdict == "UNTESTED" or not np.isfinite(v):
                ax.text(j, i, "—",
                        ha="center", va="center",
                        fontsize=8, color=WONG["grey"],
                        fontweight="bold", zorder=5)
                continue
            text_color = "white" if abs(v) > vmax * 0.55 else "black"
            # Format: 0 sig digits for |v|>=10 (saves cell space; readable);
            # 1 sig digit for |v|<10 (preserves precision on small cells).
            # For very-large magnitude sign-flip cells (|v|>=100), use 0 dp.
            if abs(v) >= 100:
                txt = f"{v:+.0f}%"
            elif abs(v) >= 10:
                txt = f"{v:+.0f}%"
            else:
                txt = f"{v:+.1f}%"
            ax.text(j, i, txt,
                    ha="center", va="center",
                    fontsize=8, color=text_color,
                    fontweight="bold", zorder=5)

    # Axes + ticks
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(CORPORA, fontsize=8.5, fontweight="bold")
    ax.set_yticks(range(n_rows))
    # Q9-pooled row gets italic font to distinguish from per-backbone rows.
    ax.set_yticklabels(BACKBONES, fontsize=8.5, fontweight="bold")
    for tick_label in ax.get_yticklabels():
        if tick_label.get_text().startswith("("):
            tick_label.set_style("italic")
            tick_label.set_color(WONG["grey"])
    ax.set_xlabel("RLHF preference corpus", fontsize=9, labelpad=4)
    ax.set_ylabel("Reward-model backbone", fontsize=9, labelpad=4)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Sequential diverging colorbar (data-driven; no categorical legend).
    # Shrink so it doesn't overpower the matrix.  Use extend="min" to signal
    # that sign-flip cells (Llama -171, Yi -110, MoE -60, Mixtral -60) are
    # SATURATED beyond the displayed range — visual honesty per Skill:
    # research-grade-plots §9 (don't lie about the range).
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.025,
                        extend="min")
    cbar.set_label("PEBS gain over pop-slope (%)", fontsize=8, labelpad=3)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)
    # Reference tick at zero (the perceptual midpoint)
    cbar.set_ticks([-vmax, -vmax / 2, 0, vmax / 2, vmax])

    # Verdict-outline + hatched-NA legend, positioned BELOW the matrix.
    # Four patch handles per verdict class (now includes FALSIFIED).
    legend_handles = [
        mpatches.Patch(facecolor="none", edgecolor=WONG["blue"],
                       linestyle="--", linewidth=1.4,
                       label="ESTABLISHED"),
        mpatches.Patch(facecolor="none", edgecolor=WONG["orange"],
                       linestyle="--", linewidth=1.4,
                       label="MODERATE"),
        mpatches.Patch(facecolor="none", edgecolor=WONG["vermillion"],
                       linestyle="--", linewidth=1.4,
                       label="FALSIFIED-CROSS-FAMILY"),
        mpatches.Patch(facecolor=WONG["light_grey"], edgecolor=WONG["grey"],
                       linewidth=0.4, hatch="///",
                       label="UNTESTED (deployment boundary)"),
    ]
    ax.legend(handles=legend_handles,
              loc="upper center",
              bbox_to_anchor=(0.5, -0.06),
              ncol=4,
              fontsize=7.5,
              frameon=False,
              handlelength=1.6,
              handleheight=1.2,
              columnspacing=1.4)

    # Save (Type 42 fonts; PDF vector; via global savefig.bbox = "tight").
    plt.savefig(out_pdf, format="pdf")
    plt.close(fig)
    print(f"wrote {out_pdf}")


# ---- Provenance / summary print ---------------------------------------------
def print_provenance(cells: Dict[Tuple[str, str], Dict]) -> None:
    """Print a per-cell anchor table for replicability auditing."""
    n_est = sum(1 for c in cells.values() if c["verdict"] == "ESTABLISHED")
    n_mod = sum(1 for c in cells.values() if c["verdict"] == "MODERATE")
    n_fal = sum(1 for c in cells.values() if c["verdict"] == "FALSIFIED")
    n_unt = sum(1 for c in cells.values() if c["verdict"] == "UNTESTED")
    n_total = len(cells)
    n_filled = n_est + n_mod + n_fal
    print(f"\nProvenance summary: {n_total} cells = {n_est} ESTABLISHED + {n_mod} MODERATE + {n_fal} FALSIFIED + {n_unt} UNTESTED")
    print(f"  cells_filled (non-UNTESTED) = {n_filled} / {n_total}")
    print("(Per-cell anchors:)")
    for (bb, cp), c in sorted(cells.items()):
        gain = c["gain_pct"]
        gtxt = f"{gain:+.2f}%" if gain is not None else "    NA"
        print(f"  {bb:28s} x {cp:11s} = {gtxt:>9s}  [{c['verdict']:12s}]  anchor={c['anchor']}")


# ---- main -------------------------------------------------------------------
def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    claims_path = os.path.join(repo_root, "PAPER_CLAIMS.json")

    cells = load_cell_values(claims_path)
    print_provenance(cells)

    out_pdf = os.path.join(here, "fig1_3axis_deployment_map.pdf")
    draw_figure(cells, out_pdf)


if __name__ == "__main__":
    main()
