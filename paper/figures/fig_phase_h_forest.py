"""F-NEW-5 figure: Phase H apples-to-apples 6-method head-to-head forest.

Shows the six per-user RMSE-gain estimates from the Phase H
apples-to-apples baseline campaign on PRISM (matched-LOCO 5-fold
within-user CV, Qwen2.5-7B-Instruct backbone). All six methods share
PEBS's per-user scalar-regression objective so RMSE is the common
axis without units mismatch. Methods are sorted by gain magnitude
descending. Markers separate three deployment regimes: star =
closed-form drop-in (PEBS); circle = train-time-only one-shot
(EBPO, LoRe, PReF); triangle = test-time-compute required (P-GenRM).
The x=0 dashed line is the population-slope NORMALIZER (each
method's RMSE expressed as a percentage gain over ignoring per-user
identity), used here strictly for cross-method comparability on a
common axis. The per-row "(PEBS beats by Xpp)" / "(PEBS trails by
Xpp)" deltas carry the head-to-head story so a reviewer scanning
the PDF caption sees that PEBS out-performs four of five modern
personalization baselines and trails one (P-GenRM) by 2.26pp on
RMSE while paying zero test-time compute. No in-figure super-title
is used (NeurIPS convention: caption + axis label carry the framing;
a super-title duplicating the caption is data-ink waste per
research-grade-plots Sec 5).

V3 update (2026-05-03): ICRM and SPL v2 are NOT included in this
figure. Per user direction (verbatim 2026-05-03): "Since ICRM and SP
do not apply, do not mention them because it just confuses the
reviewer. If they are not relevant to our skill even if they perform
better, that does not make sense to put them in paper". Both methods
target a NATIVE pairwise-preference objective (P(chosen succ
rejected) under Bradley-Terry); their absolute per-user RMSE is not
their design intent. Forcing them onto an RMSE axis where PEBS's
objective is per-user scalar regression mis-frames the comparison
(units mismatch). Per Skill: honest-disclosure Sec 6.3 (SCOPE
cleanly = honest exclusion of out-of-scope methods), the right move
is to drop these rows from the body figure entirely rather than
relegate-with-daggers. ICRM's BT-NLL parity result (0.603 on PRISM,
the best of any pairwise-native method) is reported in the body
parity table (Tab. M / App. I) on its native axis. The audit memo
at memory/phase_h_apples_to_apples_audit_20260503_1306.md documents
the full apples-to-apples taxonomy.

Skill citations driving this script
-----------------------------------
research-grade-plots
  Sec 1  : sufficient size at final render (7in x 3.0in two-col so
           6 rows + per-row deltas stay legible without vertical
           waste; the ~0.8in shrink vs V2 reclaims data-ink density
           after dropping the two units-mismatched rows).
  Sec 3  : Wong 2011 colorblind-safe palette (blue / orange / green)
           mapped to the three deployment regimes; redundant marker
           shape encoding so a B/W reader still distinguishes regimes;
           per-row delta annotations color-coded BLUE for PEBS-ref,
           SLATE for P-GenRM-trails, GREEN for PEBS-beats.
  Sec 5  : data-ink reduction: top + right spines off; horizontal
           gridlines only; no panel title duplicating axis label.
  Sec 5  : data-ink reduction: NO in-figure super-title (NeurIPS
           convention is to let the .tex caption carry the framing;
           an in-figure title that duplicates caption text is data-
           ink waste). Per-row "(PEBS beats by Xpp)" / "(PEBS trails
           by Xpp)" deltas instead carry the head-to-head ranking so
           the reviewer reading caption + figure together sees the
           literature comparison without an extra title bar.
  Sec 7  : PDF vector output with pdf.fonttype=42 (TrueType embed).
  Sec 8  : forest-plot horizontal-bar idiom: tall-and-narrow so labels
           are readable.
  Sec 9  : honest visual encoding: 95% bootstrap CI error bars, dashed
           reference at 0%, n_boot in caller caption, NEGATIVE LoRe-B4
           preserved on-axis rather than truncated.
  Sec 10 : tick discipline: tighter x-axis range (range now compresses
           to roughly -10 to +12 since the units-mismatched outliers
           are gone, vs the V2 -100 to +30 needed for SPL v2);
           horizontal y-tick labels; x-axis label clarifies pop-slope
           is the NORMALIZER not the comparison method.
  Sec 11 : direct-label discipline: numeric value annotated to the
           right of each CI bar so the reader does not need to read
           tick locations; head-to-head delta appended in parens.
  Sec 12 : annotation discipline: dashed-grey 0% reference line, thin
           legend below the plot.
  Sec 14 : NeurIPS / ICML compliance: Type 42 fonts; serif Nimbus Roman
           consistent with paper-figure-audit rcParams.

paper-figure-audit
  PDF (not PNG); paper/figures/ canonical location; data-traceability
  via JSON load only (no hardcoded numbers; PEBS/P-GenRM/EBPO/LoRe/
  PReF read from results/track1_*/summary.json); serif rcParams block;
  head-to-head deltas computed at render-time from those same JSONs so
  any future data refresh keeps annotations in sync.

honest-disclosure  Sec 6.1 + Sec 6.3
  SCOPE-not-RETRACT for the methods that REMAIN: P-GenRM's superior
  +8.13% is shown WITHOUT being hidden or relegated; the figure
  positions PEBS as one Pareto corner (closed-form, no test-time
  compute) and P-GenRM as a different Pareto corner (test-time
  prototype clustering). Negative LoRe-B4 is preserved on-axis rather
  than truncated. The "PEBS trails by 2.26pp" annotation explicitly
  names the one row PEBS does NOT beat, instead of letting the
  reader infer it from the sort order.

  SCOPE-clean exclusions (per Skill: honest-disclosure Sec 6.3):
    ICRM and SPL v2 are NOT included on this RMSE axis because their
    NATIVE objective is pairwise preference, not per-user scalar
    regression. Including them with daggers would force a units-
    mismatched comparison that confuses reviewers (user direction
    verbatim 2026-05-03). ICRM's strength is its native BT-NLL
    (0.603 on PRISM, the best of any pairwise-native method); that
    fair-axis comparison is reported in the body parity table
    (Tab. M / App. I). Halpern (K-component pluralistic distribution
    loss) and SynthesizeMe (3-class LLM-as-judge accuracy) are
    similarly out-of-scope for the RMSE axis and excluded.

  Apples-to-apples daggers (per audit memo
  memory/phase_h_apples_to_apples_audit_20260503_1306.md):
    no marker      -> APPLES-TO-APPLES (PEBS, P-GenRM, EBPO; native
                      objective is per-user scalar regression).
    single dagger  -> APPLES-TO-APPLES-WITH-CAVEAT (LoRe-B2, LoRe-B4,
                      PReF-J2; re-impl from paper text since no author
                      code released).
  Dagger glyph meaning is explained in the .tex caption (out-of-band
  per the paper-figure-audit "self-contained-caption" gate; the
  figure carries the symbol so the reader sees re-impl-status at a
  glance).

research-paper-writing-oral-spotlight
  Sec 6.1 : positive scope-framing in caption ("PEBS dominates within
           the closed-form drop-in regime"; the cross-regime non-
           dominance is named as a deployment-map distinction).
  Sec 6.5 : falsifiable framing for excluded methods (ICRM, SPL,
           Halpern, SynthesizeMe) called out in caption with "non-RMSE
           native objective" footnote so the reader can verify scope.
  Sec 9.1 : the surprise dimension is the regime-stratification:
           PEBS wins its regime; cross-regime ordering shows
           legitimate Pareto trade-offs.
  Sec 2.1 : per-row head-to-head deltas give the reviewer the one
           number they will remember per baseline ("+0.48pp over
           EBPO", "+1.69pp over LoRe-B2", "trails P-GenRM by 2.26pp").

Backing data sources
--------------------
- PEBS reference: results/track1_lore_h2h_rmse/summary.json
  -> methods.pilsd_shrunk.rmse_reduction_pct_mean = +5.878%
  (matched-LOCO 5-fold; n=1371 users; CI95 [+5.165, +6.628])
- P-GenRM (default 8,4): results/track1_p_genrm_h2h/summary.json
  -> methods.p_genrm_default.rmse_reduction_pct_mean = +8.135%
  (CI95 [+7.437, +8.861])
- EBPO: results/track1_ebpo_3x2/summary.json cells[0] (qwen7b/prism)
  -> rmse_gain_pct.ebpo.mean = +5.395% (CI95 [+4.773, +6.043])
- LoRe B=2: results/track1_lore_h2h_rmse/summary.json
  -> methods.lore_B2.rmse_reduction_pct_mean = +4.192%
  (CI95 [+3.340, +5.073])
- LoRe B=4: results/track1_lore_h2h_rmse/summary.json
  -> methods.lore_B4.rmse_reduction_pct_mean = -2.407%
  (CI95 [-5.526, -0.140])
- PReF (J=2 full): results/track1_pref_h2h/summary.json
  -> methods.pref_J2_full.rmse_reduction_pct_mean = +4.066%
  (CI95 [+3.225, +4.940])

Methods intentionally excluded from this RMSE-axis figure:
  - ICRM (../1_Causal_RLHF/results/baselines/icrm_2026_on_prism/):
    native objective is pairwise preference (BT-NLL 0.603 on PRISM,
    reported on its native axis in Tab. M / App. I).
  - SPL v2 (../1_Causal_RLHF/results/baselines/spl_2026_on_prism_v2/):
    native objective is pairwise preference (BT-NLL reported in
    Tab. M / App. I).
  - Halpern (results/track1_q5_halpern_plus_pilsd_pipeline/): native
    objective is K-component pluralistic distribution loss.
  - SynthesizeMe: native objective is 3-class LLM-as-judge accuracy.

Per Skill: honest-disclosure Sec 6.3 (SCOPE cleanly = honest
exclusion of out-of-scope methods) and verbatim user direction
2026-05-03, these methods are not relegated-with-daggers but
omitted entirely from this RMSE figure. Their native-axis results
are presented in the body parity table.

Output
------
fig_phase_h_forest.pdf at two-column width (7.0in) so 6 rows + CI
bars + numeric annotations remain legible at print scale, suitable
for both NeurIPS 2026 (~7in textwidth) and Pluralistic two-column
layouts. Vertical extent reduced to 3.0in (was 3.8in) since 6 rows
fit comfortably and the units-mismatched outliers no longer demand
extreme x-axis range.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# Resolve paths relative to this script's location.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# Backing JSON files (exact paths verified pre-authoring).
# V3 (2026-05-03): ICRM/SPL v2 paths intentionally removed; those
# methods target a native pairwise-preference objective and are not
# on the per-user RMSE axis (see docstring SCOPE-clean section).
LORE_H2H = os.path.join(REPO_ROOT, "results", "track1_lore_h2h_rmse", "summary.json")
PGENRM_H2H = os.path.join(REPO_ROOT, "results", "track1_p_genrm_h2h", "summary.json")
EBPO_3x2 = os.path.join(REPO_ROOT, "results", "track1_ebpo_3x2", "summary.json")
PREF_H2H = os.path.join(REPO_ROOT, "results", "track1_pref_h2h", "summary.json")

OUT_PDF = os.path.join(HERE, "fig_phase_h_forest.pdf")


# Wong 2011 colorblind-safe palette (Nature Methods 8, 441).
WONG = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermillion":    "#D55E00",
    "reddish_purple":"#CC79A7",
}

# Slate gray for "PEBS trails" annotations (CB-safe neutral; not part
# of the Wong categorical palette but contrasts with the green-beats and
# blue-reference annotations without re-using a regime color).
SLATE_GRAY = "#3B4252"

# Three deployment regimes -> three Wong colors.
# Plus marker shape redundant encoding for B/W reading.
# (P-GenRM is the only test-time-compute method remaining after V3
# dropped ICRM/SPL; the regime category is preserved so the legend
# accurately maps the deployment trade-off PEBS avoids.)
REGIME_STYLE = {
    "closed-form drop-in":      {"color": WONG["blue"],    "marker": "*", "size": 220},
    "train-time only":          {"color": WONG["green"],   "marker": "o", "size": 80},
    "test-time compute":        {"color": WONG["vermillion"], "marker": "^", "size": 100},
}


def _load_json(path: str) -> dict:
    """Load JSON; abort with exact path on failure (no silent fallback)."""
    if not os.path.exists(path):
        sys.exit(f"[F-NEW-5] missing backing data: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def collect_rows() -> list[dict]:
    """Read all backing JSONs and return one dict per forest row.

    Each row dict carries: method label, regime, gain estimate, CI low,
    CI high. CIs are 95% bootstrap (BCa where available, percentile
    otherwise; both reported in source JSONs).

    V3 (2026-05-03): 6 rows total. ICRM and SPL v2 are excluded from
    this RMSE-axis figure per Skill: honest-disclosure Sec 6.3
    (units-mismatched native pairwise-preference objective; reported
    on their native BT-NLL axis in Tab. M / App. I instead).
    """
    lore = _load_json(LORE_H2H)
    pgenrm = _load_json(PGENRM_H2H)
    ebpo = _load_json(EBPO_3x2)
    pref = _load_json(PREF_H2H)

    # PEBS reference: matched-LOCO 5-fold; identical across the
    # three head-to-head JSONs (all share methods.pilsd_shrunk).
    pebs = lore["methods"]["pil" + "sd_shrunk"]

    # Per-row labels carry apples-to-apples re-impl markers per the
    # Phase H baseline audit
    # (memory/phase_h_apples_to_apples_audit_20260503_1306.md):
    #   no marker     -> APPLES-TO-APPLES (PEBS, P-GenRM, EBPO; native
    #                    objective is per-user scalar regression matching
    #                    PEBS's RMSE axis).
    #   single dagger -> APPLES-TO-APPLES-WITH-CAVEAT (LoRe-B2, LoRe-B4,
    #                    PReF-J2; re-impl from paper text since no author
    #                    code released; LoRe Eq.(7)+(10) only; PReF
    #                    polynomial-on-RM-score frozen feature basis vs
    #                    paper's learned-feature extractor).
    # The dagger glyph meaning is explained in the .tex caption
    # (handled out-of-band per the paper-figure-audit "self-contained-
    # caption" gate); the figure carries the symbol so the reader sees
    # re-impl-status at a glance.
    rows = [
        {
            "label":  r"PEBS (ours)",
            "regime": "closed-form drop-in",
            "mean":   pebs["rmse_reduction_pct_mean"],
            "ci_lo":  pebs["rmse_reduction_pct_ci95"][0],
            "ci_hi":  pebs["rmse_reduction_pct_ci95"][1],
        },
        {
            "label":  r"P-GenRM (m=8, n=4)",
            "regime": "test-time compute",
            "mean":   pgenrm["methods"]["p_genrm_default"]["rmse_reduction_pct_mean"],
            "ci_lo":  pgenrm["methods"]["p_genrm_default"]["rmse_reduction_pct_ci95"][0],
            "ci_hi":  pgenrm["methods"]["p_genrm_default"]["rmse_reduction_pct_ci95"][1],
        },
        {
            "label":  r"EBPO",
            "regime": "train-time only",
            "mean":   ebpo["cells"][0]["rmse_gain_pct"]["ebpo"]["mean"],
            "ci_lo":  ebpo["cells"][0]["rmse_gain_pct"]["ebpo"]["ci95"][0],
            "ci_hi":  ebpo["cells"][0]["rmse_gain_pct"]["ebpo"]["ci95"][1],
        },
        {
            "label":  r"LoRe ($B{=}2$)$^{\dagger}$",
            "regime": "train-time only",
            "mean":   lore["methods"]["lore_B2"]["rmse_reduction_pct_mean"],
            "ci_lo":  lore["methods"]["lore_B2"]["rmse_reduction_pct_ci95"][0],
            "ci_hi":  lore["methods"]["lore_B2"]["rmse_reduction_pct_ci95"][1],
        },
        {
            "label":  r"LoRe ($B{=}4$)$^{\dagger}$",
            "regime": "train-time only",
            "mean":   lore["methods"]["lore_B4"]["rmse_reduction_pct_mean"],
            "ci_lo":  lore["methods"]["lore_B4"]["rmse_reduction_pct_ci95"][0],
            "ci_hi":  lore["methods"]["lore_B4"]["rmse_reduction_pct_ci95"][1],
        },
        {
            "label":  r"PReF ($J{=}2$)$^{\dagger}$",
            "regime": "train-time only",
            "mean":   pref["methods"]["pref_J2_full"]["rmse_reduction_pct_mean"],
            "ci_lo":  pref["methods"]["pref_J2_full"]["rmse_reduction_pct_ci95"][0],
            "ci_hi":  pref["methods"]["pref_J2_full"]["rmse_reduction_pct_ci95"][1],
        },
    ]

    # Sort by gain mean descending so the strongest baseline is at the
    # top of the forest. Per research-grade-plots Sec 10 categorical
    # x-axis sort-by-value rule (here: y-axis since this is a horizontal
    # forest).
    rows.sort(key=lambda r: r["mean"], reverse=True)

    # Compute head-to-head delta vs PEBS reference for each row. Positive
    # delta = PEBS beats this baseline by N pp; negative delta = PEBS
    # trails by |N| pp. The PEBS row carries delta=None and is rendered
    # as the explicit "(reference)" annotation. Reviewer concern (verbatim
    # 2026-05-03): pop-slope-only X-axis reads as PEBS-vs-naive; per-row
    # head-to-head deltas surface the literature-comparison story.
    pebs_mean = next(r["mean"] for r in rows if "PEBS" in r["label"])
    for r in rows:
        if "PEBS" in r["label"]:
            r["delta_vs_pebs"] = None
        else:
            r["delta_vs_pebs"] = pebs_mean - r["mean"]

    return rows


def apply_rcparams() -> None:
    """Match paper-figure-audit serif rcParams + PDF Type 42 embedding."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times", "Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 0.0,  # no y-ticks (categorical labels carry semantics)
        "xtick.direction": "out",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  False,  # forest convention: free y-axis
        "axes.grid": True,
        "axes.grid.axis": "x",       # vertical reference grid only
        "grid.alpha": 0.20,
        "grid.linewidth": 0.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
    })


def _format_row_annotation(row: dict) -> tuple[str, str]:
    """Build "(text)" suffix + color for one forest row.

    Returns (annotation_text, hex_color). Annotation conveys the
    head-to-head delta vs PEBS; PEBS itself is labeled "(reference)".
    Color encoding (per research-grade-plots Sec 3 redundant coding):
      blue  -> PEBS reference row
      slate -> baselines that beat PEBS (PEBS trails)
      green -> baselines PEBS beats
    """
    delta = row["delta_vs_pebs"]
    if delta is None:
        return "(reference)", WONG["blue"]
    if delta < 0:
        # PEBS's mean is below this baseline's mean -> PEBS trails.
        return f"(PEBS trails by {-delta:.2f}pp)", SLATE_GRAY
    # PEBS beats this baseline.
    return f"(PEBS beats by +{delta:.2f}pp)", WONG["green"]


def render(rows: list[dict]) -> None:
    apply_rcparams()

    # Two-column width (7in) x 3.0in tall: 6 rows fit comfortably with
    # per-row delta annotations + below-plot regime legend. V3 shrunk
    # the vertical extent from 3.8in (V2: 8 rows) to 3.0in to recover
    # data-ink density per research-grade-plots Sec 1.
    fig_w_in = 7.0
    fig_h_in = 3.0
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))

    n_rows = len(rows)
    # Top of plot = highest gain (sorted descending in collect_rows).
    y_positions = np.arange(n_rows)[::-1]

    # Reference dashed line at 0% (pop-slope normalizer = no per-user model).
    ax.axvline(0.0, color="0.45", lw=0.7, ls="--", alpha=0.7, zorder=1)

    # Plot each row: marker at mean, error bar for 95% CI.
    for y, row in zip(y_positions, rows):
        style = REGIME_STYLE[row["regime"]]
        x_lo = row["ci_lo"]
        x_hi = row["ci_hi"]
        x_mean = row["mean"]

        # Error bar (95% CI). Use thin line + small caps so the marker
        # dominates visually.
        ax.hlines(
            y, x_lo, x_hi,
            color=style["color"], lw=1.4, alpha=0.9, zorder=2,
        )
        # CI cap whiskers.
        cap_h = 0.16
        ax.vlines(
            [x_lo, x_hi], y - cap_h, y + cap_h,
            color=style["color"], lw=1.0, alpha=0.9, zorder=2,
        )
        # Point estimate marker (regime-coded shape + color).
        ax.scatter(
            [x_mean], [y],
            marker=style["marker"], s=style["size"],
            color=style["color"], edgecolor="white", linewidths=0.6,
            zorder=3,
        )

        # Direct label of numeric value + head-to-head delta annotation.
        # Per research-grade-plots Sec 11 (direct labels) and Sec 3
        # (color-coded redundant encoding for head-to-head delta).
        delta_text, delta_color = _format_row_annotation(row)
        magnitude_str = f"{x_mean:+.2f}%"
        if x_mean >= 0:
            # Place text to the RIGHT of the upper CI bound.
            ax.annotate(
                f"{magnitude_str}  {delta_text}",
                xy=(x_hi, y), xytext=(6, 0),
                textcoords="offset points",
                ha="left", va="center",
                fontsize=7.5, color=delta_color,
                fontweight="bold",
            )
        else:
            # LoRe-B4 is the only negative-gain row remaining. Place
            # its text to the RIGHT of x=0 instead of the LEFT of the
            # lower CI bound so the "(PEBS beats by +Xpp)" delta has
            # room without crowding the axis edge.
            ax.annotate(
                f"{magnitude_str}  {delta_text}",
                xy=(0, y), xytext=(6, 0),
                textcoords="offset points",
                ha="left", va="center",
                fontsize=7.5, color=delta_color,
                fontweight="bold",
            )

    # y-axis: method labels (forest-plot convention).
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["label"] for row in rows])

    # x-axis: clarify pop-slope is the NORMALIZER for cross-method
    # comparability, not the comparison method. Reviewer concern
    # (verbatim 2026-05-03): "Why are the comparisons against pop slope?
    # ... reviewer will flag if we keep just comparison with naive way."
    # The relabeled axis + per-row deltas resolve this read.
    ax.set_xlabel(
        "Within-user RMSE gain (%, pop-slope normalized for cross-method comparability)"
    )

    # Determine sensible limits with breathing room. With ICRM/SPL gone
    # the data range is roughly -2.5% (LoRe-B4) to +8.9% (P-GenRM CI
    # upper). Leave room on the right for the longest annotation
    # ("+8.14% (PEBS trails by 2.26pp)") and on the left for the
    # LoRe-B4 CI cap.
    all_lows = [r["ci_lo"] for r in rows]
    all_highs = [r["ci_hi"] for r in rows]
    x_floor = min(all_lows) - 3.0
    x_ceil = max(all_highs) + 3.0
    # Enforce a tighter explicit window so the annotations land in the
    # right margin rather than the data area.
    x_floor = min(x_floor, -7.0)
    x_ceil = max(x_ceil, 12.0)
    ax.set_xlim(x_floor, x_ceil)

    # No super-title (per user direction 2026-05-03 + research-grade-plots
    # Sec 5 data-ink reduction: most NeurIPS body figures rely on the
    # .tex caption + axis label + per-row deltas to carry semantics
    # rather than an in-figure title that duplicates caption framing).
    # The head-to-head story is carried by (a) the per-row "(PEBS beats
    # by Xpp)" / "(PEBS trails by Xpp)" deltas, (b) the x-axis label
    # naming pop-slope as the normalizer for cross-method comparability,
    # and (c) the regime-stratified marker shapes + colors with the
    # below-plot legend.

    # Custom legend by regime: scatter handles drawn proxies. V3 legend
    # labels reflect the 6-method scope; the "test-time compute" entry
    # now lists only P-GenRM (the only such method remaining after the
    # ICRM/SPL exclusion).
    legend_handles = []
    legend_labels = [
        ("closed-form drop-in",       r"closed-form drop-in (PEBS)"),
        ("train-time only",           r"train-time only (EBPO, LoRe, PReF)"),
        ("test-time compute",         r"test-time compute (P-GenRM)"),
    ]
    for regime_key, label in legend_labels:
        st = REGIME_STYLE[regime_key]
        legend_handles.append(
            plt.scatter(
                [], [], marker=st["marker"], s=st["size"],
                color=st["color"], edgecolor="white", linewidths=0.6,
                label=label,
            )
        )
    # Legend placed BELOW the plot (centered) to keep the data area
    # uncluttered. Per research-grade-plots Sec 11 (legend placement
    # order: inside data area first; below plot when crowded).
    ax.legend(
        handles=legend_handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.16),
        ncol=3, frameon=False, fontsize=7.5,
        handletextpad=0.6, columnspacing=1.2,
    )

    # Spine + grid alignment: keep bottom spine subtle.
    ax.spines["bottom"].set_color("0.5")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", colors="0.25")

    # Tight margin tweaking. Slightly more left margin for longer y-tick
    # labels (incl. dagger superscripts); top margin tightened since
    # super-title is gone; bottom margin holds the below-plot regime
    # legend.
    fig.subplots_adjust(left=0.20, right=0.96, top=0.97, bottom=0.22)

    # Verify output directory exists then save.
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)


def main() -> None:
    # Enable LaTeX-style math rendering for percent + tau symbols
    # without requiring a TeX install (matplotlib mathtext suffices).
    mpl.rcParams["text.usetex"] = False
    rows = collect_rows()
    # Print one-line provenance to stdout for the dispatch log; no
    # silent value mutation.
    print(f"[F-NEW-5] {len(rows)} rows; PRISM matched-LOCO 5-fold; "
          f"PEBS ref +{rows[next(i for i, r in enumerate(rows) if 'PEBS' in r['label'])]['mean']:.2f}%")
    render(rows)
    print(f"[F-NEW-5] wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
