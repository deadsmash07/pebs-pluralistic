"""Figure 8 (P1 Pluralistic) --- head-collapse signature across bases.

Bar chart of the trained-coherence prediction-spread (sigma_pred_coh)
on the 208-row HelpSteer2 slice for the 3 cross-family bases plus
the F23 main anchor. The 0.40 moderate-collapse threshold is shown
as a dashed reference. The two sign-flipping bases (Llama-3-8B,
Yi-1.5-34B) lie below the threshold; the null base (Mistral-Small-22B)
lies above. This is the head-collapse signature underlying
Figure 1's verdict-correlated prediction-collapse claim.

Reads pred_sd_coh from F23'' / F23''' / F23'''' adapter-inspection
predictions_summary.json files in 1_Causal_RLHF.

Anonymity-clean. Wong 2011 colorblind-safe palette.
"""
from __future__ import annotations

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from _figstyle import WONG, set_pub_style, save_fig, here


def load_pred_sd_coh(base_dir: str) -> tuple[float, float]:
    """Return (mean_coh, sd_coh) from predictions_summary.json."""
    path = os.path.join(base_dir, "predictions_summary.json")
    with open(path) as f:
        d = json.load(f)
    coh = d["per_attribute_pred_summary"]["coherence"]
    return float(coh["mean"]), float(coh["sd"])


def main() -> None:
    set_pub_style()

    f23_double_prime_root = os.path.realpath(os.path.join(
        here(__file__), "..", "..", "..", "..", "1_Causal_RLHF",
        "results", "falsifiers", "F23_double_prime_adapter_inspection",
    ))

    bases = [
        ("Llama-3-8B", "nous_meta_llama_3_8b_instruct", "sign-flip"),
        ("Yi-1.5-34B", "yi_1p5_34b_chat", "sign-flip"),
        ("Mistral-22B", "mistral_small_instruct_2409", "null"),
    ]

    sds = []
    for label, dirname, _verdict in bases:
        _, sd = load_pred_sd_coh(os.path.join(f23_double_prime_root, dirname))
        sds.append(sd)

    # F23 main anchor (Llama-family-dense reference; from predictions_summary
    # field anchor_pred_sd_coh_F23_main).
    anchor_path = os.path.join(
        f23_double_prime_root, "mistral_small_instruct_2409",
        "predictions_summary.json",
    )
    with open(anchor_path) as f:
        anchor = json.load(f)["anchor_pred_sd_coh_F23_main"]

    moderate_threshold = 0.40

    fig, ax = plt.subplots(figsize=(5.2, 3.2))

    xs = np.arange(len(bases))
    bar_colors = []
    for _, _, verdict in bases:
        if verdict == "sign-flip":
            bar_colors.append(WONG["vermillion"])
        elif verdict == "null":
            bar_colors.append(WONG["orange"])
        else:
            bar_colors.append(WONG["blu_green"])

    bars = ax.bar(xs, sds, color=bar_colors,
                  edgecolor="black", linewidth=0.5, zorder=2)

    # Threshold line --- raised above bars (zorder=4) so the dashed
    # rule reads cleanly across the Mistral-22B bar (the only bar
    # that exceeds the threshold). Without zorder=4 the dashed line
    # is hidden behind the bar where they overlap.
    ax.axhline(moderate_threshold, color="black", lw=0.9, ls="--",
               zorder=4)
    ax.text(len(bases) - 0.55, moderate_threshold - 0.022,
            f"moderate-collapse threshold = {moderate_threshold:.2f}",
            ha="right", va="top", fontsize=7.5,
            color="black", style="italic", zorder=5)

    # Reference line: dense in-family anchor value (Llama-family-dense
    # reference; same protocol, drawn from the in-family adapter run).
    ax.axhline(anchor, color=WONG["grey"], lw=0.9, ls=":", zorder=4)
    ax.text(0.03, anchor + 0.014,
            f"in-family reference = {anchor:.2f}",
            ha="left", va="bottom", fontsize=7.5,
            color=WONG["grey"], style="italic", zorder=5)

    # Bar value labels
    for bar, sd, (_, _, verdict) in zip(bars, sds, bases):
        ax.text(bar.get_x() + bar.get_width() / 2,
                sd + 0.012,
                f"{sd:.4f}\n({verdict})",
                ha="center", va="bottom", fontsize=8,
                color=bar.get_facecolor())

    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bases], fontsize=9)
    ax.set_ylabel("Trained-coherence prediction\n"
                  "standard deviation $\\sigma_{\\mathrm{pred,coh}}$",
                  fontsize=9)
    ax.set_title(
        "Head-collapse signature: prediction-spread by base model",
        fontsize=10, fontweight="bold", pad=8,
    )
    ax.set_ylim(0, max(sds + [anchor, moderate_threshold]) * 1.40)

    save_fig(fig, here(__file__) + "/fig8_head_collapse_signature.pdf")


if __name__ == "__main__":
    main()
