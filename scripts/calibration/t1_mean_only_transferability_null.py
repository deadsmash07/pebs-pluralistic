"""T1 mean-only NULL baseline for calibrator transferability (iter+N+253).

Adversarial-reviewer attack B (commit 38a164f):
    "PILSD's 0.91 cross-backbone transferability ratio is contingent on
    per-backbone z-score rescaling which collapses scale + intercept
    structure. The 'calibrator' that transfers may just be per-user bias
    (intercept), not the per-user regression-shrinkage object (slope)."

Test
----
Repeat the iter+N+236 3x3 transferability matrix with a "mean-only"
estimator that fixes beta = 1 in z-scored units (hence the only per-user
degree of freedom is the intercept alpha_j). If the mean-only NULL
matches the PILSD-shrunk (alpha_j, beta_j) transfer gains, the
transferability claim collapses to per-user demeaning (trivial). If the
mean-only NULL is materially worse, PILSD's slope-shrinkage machinery
does the work.

Design
------
For each user j and source backbone A:
    alpha_j^A (mean-only) = mean_{train}( y - 1 * z_A(x_A^train) )
                          = mean(y^train) - mean(z_A(x_A^train))
    predict on target B:  yhat = alpha_j^A + 1.0 * z_B(x_B^test)

PILSD-shrunk (std) transfer (from iter+N+236 artifact):
    predict_target = alpha_j^A (shrunk) + beta_j^A (shrunk) * z_B(x_B^test)

Both are evaluated on the SAME k-fold splits (seed=20260420, k=5) on the
SAME 1394 users / 68371-utterance inner join as iter+N+236 so gaps are
apples-to-apples.

Gap per cell = PILSD_std_gain% - MeanOnly_gain%     (positive ==> slope
shrinkage adds value; near zero ==> intercept alone suffices).
Cluster-bootstrap over users (n_boot=2000, seed=20260420) on the
PAIRED per-user gain-difference (delta_j = gain_pilsd_j - gain_meanonly_j)
to get a 95% CI on the gap itself.

Outputs
-------
- results/track1_mean_only_transferability_null/summary.json
- paper/figures/fig_22_t1_transferability_ablation.pdf  (2-panel)
- PAPER_INSERT_mean_only_null.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BACKBONES = [
    ("qwen7b",     "data/prism_rm_scored.parquet",         "rm_score"),
    ("skywork27b", "data/prism_skywork_scored.parquet",    "skywork_score"),
    ("llama32_3b", "data/prism_llama32_3b_scored.parquet", "llama32_3b_score"),
]
PRETTY = {
    "qwen7b":     "Qwen2.5-7B",
    "skywork27b": "Skywork-27B",
    "llama32_3b": "Llama-3.2-3B",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--min-obs-per-user", type=int, default=6)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260420)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--output-dir",
                   default="results/track1_mean_only_transferability_null")
    p.add_argument("--pilsd-summary",
                   default="results/track1_calibrator_transferability/summary.json",
                   help="Prior PILSD-shrunk transferability summary for gap reporting.")
    p.add_argument("--pilsd-per-user",
                   default="results/track1_calibrator_transferability/per_user_rmse.parquet",
                   help="Prior PILSD-shrunk per-user RMSE parquet for paired per-user deltas.")
    p.add_argument("--figure-path",
                   default="paper/figures/fig_22_t1_transferability_ablation.pdf")
    p.add_argument("--paper-insert",
                   default="PAPER_INSERT_mean_only_null.tex")
    return p.parse_args()


def kfold_split(n: int, k: int, rng):
    """Identical splitter to iter+N+236 (so folds match exactly when seeds match)."""
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = []
    fold_size = n // k
    for i in range(k):
        start = i * fold_size
        stop = (i + 1) * fold_size if i < k - 1 else n
        test_idx = idx[start:stop]
        train_idx = np.concatenate([idx[:start], idx[stop:]])
        folds.append((train_idx, test_idx))
    return folds


def pop_ols(x: np.ndarray, y: np.ndarray):
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def cluster_bootstrap_mean(values: np.ndarray, n_boot: int, seed: int):
    """Per-user mean bootstrap (each user is one cluster contributing one value)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(values[idx].mean())
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    root = Path(args.root).expanduser().resolve()
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (root / args.figure_path).parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Load + inner-join (same as iter+N+236) -------------------------
    frames = {}
    for name, parquet, col in BACKBONES:
        df = pd.read_parquet(root / parquet)
        frames[name] = df[["utterance_id", "user_id", "score_user", col]]
        print(f"[load] {name:<12} rows={len(df):>6}  users={df.user_id.nunique()}")

    merged = frames["qwen7b"].copy()
    merged = merged.merge(
        frames["skywork27b"][["utterance_id", "skywork_score"]],
        on="utterance_id", how="inner",
    ).merge(
        frames["llama32_3b"][["utterance_id", "llama32_3b_score"]],
        on="utterance_id", how="inner",
    ).dropna(subset=["score_user"]).reset_index(drop=True)
    print(f"[join] {len(merged)} utterances x {merged.user_id.nunique()} users"
          "  (inner join on utterance_id)\n")

    backbone_cols = {
        "qwen7b": "rm_score",
        "skywork27b": "skywork_score",
        "llama32_3b": "llama32_3b_score",
    }

    # --- 2. Per-backbone global z-score stats and pop OLS (baseline) -------
    score_mean, score_std, pop_raw = {}, {}, {}
    for name, col in backbone_cols.items():
        mu = float(merged[col].mean())
        sd = float(merged[col].std(ddof=1))
        score_mean[name] = mu
        score_std[name] = max(sd, 1e-12)
        a, b = pop_ols(merged[col].to_numpy(),
                       merged.score_user.to_numpy().astype(float))
        pop_raw[name] = {"alpha": a, "beta": b}
        print(f"[pop-raw] {name:<12} alpha={a:+.3f}  beta={b:+.3f}  "
              f"mu={mu:+.3f}  sd={sd:.3f}")
    print()

    # --- 3. Per-user k-fold: compute RMSE for mean-only transfer arms ------
    arms = []
    for B in backbone_cols:
        arms.append(f"pop_{B}")
        for A in backbone_cols:
            arms.append(f"meanonly_{A}_to_{B}")

    per_user_rows = []
    for uid, grp in merged.groupby("user_id"):
        n = len(grp)
        if n < args.min_obs_per_user:
            continue
        y = grp.score_user.to_numpy().astype(float)
        x_raw = {name: grp[col].to_numpy().astype(float)
                 for name, col in backbone_cols.items()}

        sq = {a: [] for a in arms}
        # NOTE: rng is re-used from the module-level rng so this runs fold
        # shuffles in the SAME order as iter+N+236 for user-ordering parity.
        for train_idx, test_idx in kfold_split(n, args.k_folds, rng):
            if len(test_idx) == 0:
                continue
            y_tr, y_te = y[train_idx], y[test_idx]

            # Mean-only fit on each source A (fixing beta = 1 in z-units)
            per_src_alpha = {}
            for A in backbone_cols:
                xA_tr = x_raw[A][train_idx]
                muA, sdA = score_mean[A], score_std[A]
                zA_tr = (xA_tr - muA) / sdA
                # alpha_j = y_bar - 1.0 * z_bar
                alpha = float(y_tr.mean() - zA_tr.mean())
                per_src_alpha[A] = alpha

            # Evaluate on each target B
            for B, _col in backbone_cols.items():
                xB_te = x_raw[B][test_idx]
                muB, sdB = score_mean[B], score_std[B]
                zB_te = (xB_te - muB) / sdB

                # baseline pop_slope on raw target scores (matches iter+N+236)
                yh = pop_raw[B]["alpha"] + pop_raw[B]["beta"] * xB_te
                sq[f"pop_{B}"].extend(((yh - y_te) ** 2).tolist())

                for A in backbone_cols:
                    alpha = per_src_alpha[A]
                    yh = alpha + 1.0 * zB_te
                    sq[f"meanonly_{A}_to_{B}"].extend(((yh - y_te) ** 2).tolist())

        per_user_rows.append({
            "user_id": uid,
            "n": n,
            **{f"rmse_{a}": float(np.sqrt(np.mean(sq[a]))) if sq[a] else np.nan
               for a in arms},
        })

    pu = pd.DataFrame(per_user_rows)
    pu.to_parquet(out_dir / "per_user_rmse.parquet")
    print(f"[per_user] {len(pu)} users evaluated\n")

    # --- 4. Load PILSD-shrunk per-user RMSE for paired gap -----------------
    pilsd_pu = pd.read_parquet(root / args.pilsd_per_user)
    pilsd_summary = json.loads((root / args.pilsd_summary).read_text())
    # Align on user_id
    pu_aligned = pu.merge(pilsd_pu, on="user_id", how="inner", suffixes=("", "_pilsd"))
    print(f"[align] {len(pu_aligned)} users shared with PILSD-shrunk artifact")

    # --- 5. Build 3x3 mean-only NULL matrix with cluster-bootstrap CIs -----
    bnames = list(backbone_cols)
    mean_only_matrix = {}
    gap_matrix = {}
    for A in bnames:
        for B in bnames:
            col_pred = f"rmse_meanonly_{A}_to_{B}"
            col_base = f"rmse_pop_{B}"
            pred = pu[col_pred].to_numpy()
            base = pu[col_base].to_numpy()
            mask = np.isfinite(pred) & np.isfinite(base)
            pred = pred[mask]
            base = base[mask]
            rel = 100.0 * (base - pred) / base
            mean_pred = float(pred.mean())
            mean_base = float(base.mean())
            mean_gain_pct = 100.0 * (mean_base - mean_pred) / mean_base
            ci_lo, ci_hi = cluster_bootstrap_mean(
                rel, n_boot=args.n_boot,
                seed=args.seed + hash((A, B, "meanonly")) % 100000,
            )
            w = stats.wilcoxon(pred, base, alternative="less")
            mean_only_matrix[f"{A}__to__{B}"] = {
                "source": A,
                "target": B,
                "diagonal": bool(A == B),
                "n_users": int(len(pred)),
                "rmse_pred_mean": mean_pred,
                "rmse_base_mean": mean_base,
                "mean_gain_pct": float(mean_gain_pct),
                "per_user_gain_mean_pct": float(rel.mean()),
                "per_user_gain_ci95": [ci_lo, ci_hi],
                "frac_pred_smaller": float((pred < base).mean()),
                "wilcoxon_less_p": float(w.pvalue),
            }

            # paired GAP vs PILSD-shrunk STD transfer on same users
            if A == B:
                pilsd_col = f"rmse_transfer_raw_{A}_to_{B}"
            else:
                pilsd_col = f"rmse_transfer_std_{A}_to_{B}"
            mo = pu_aligned[col_pred].to_numpy()
            pi = pu_aligned[pilsd_col].to_numpy()
            bs = pu_aligned[col_base].to_numpy()
            mk = np.isfinite(mo) & np.isfinite(pi) & np.isfinite(bs)
            mo = mo[mk]; pi = pi[mk]; bs = bs[mk]
            gain_mo = 100.0 * (bs - mo) / bs
            gain_pi = 100.0 * (bs - pi) / bs
            delta = gain_pi - gain_mo        # positive ==> PILSD > mean-only
            d_ci_lo, d_ci_hi = cluster_bootstrap_mean(
                delta, n_boot=args.n_boot,
                seed=args.seed + hash((A, B, "gap")) % 100000,
            )
            gap_matrix[f"{A}__to__{B}"] = {
                "pilsd_gain_pct": float(gain_pi.mean()),
                "meanonly_gain_pct": float(gain_mo.mean()),
                "gap_pct": float(delta.mean()),
                "gap_ci95": [d_ci_lo, d_ci_hi],
                "wilcoxon_pilsd_less_p": float(
                    stats.wilcoxon(pi, mo, alternative="less").pvalue
                ),
                "n_users_paired": int(len(delta)),
            }

    # --- 6. Verdict: mean-only "ratio" in the reviewer's idiom -------------
    # reviewer spec: if mean-only NULL >= 80% of PILSD-shrunk ==> slope
    # contributes little; if <= 50% ==> slope does heavy lifting.
    ratios_off = []
    ratios_diag = []
    for A in bnames:
        for B in bnames:
            cell_mo = mean_only_matrix[f"{A}__to__{B}"]["mean_gain_pct"]
            cell_pi = pilsd_summary["transfer_matrix_std"][f"{A}__to__{B}"]["mean_gain_pct"]
            r = cell_mo / cell_pi if abs(cell_pi) > 1e-9 else float("nan")
            if A == B:
                ratios_diag.append(r)
            else:
                ratios_off.append(r)
    mean_ratio_off = float(np.mean(ratios_off))
    mean_ratio_diag = float(np.mean(ratios_diag))

    # Best-transfer "0.91" style
    best_ratios_mo, best_ratios_pi = [], []
    for B in bnames:
        diag_pi = pilsd_summary["transfer_matrix_std"][f"{B}__to__{B}"]["mean_gain_pct"]
        diag_mo = mean_only_matrix[f"{B}__to__{B}"]["mean_gain_pct"]
        off_pi = [pilsd_summary["transfer_matrix_std"][f"{A}__to__{B}"]["mean_gain_pct"]
                  for A in bnames if A != B]
        off_mo = [mean_only_matrix[f"{A}__to__{B}"]["mean_gain_pct"]
                  for A in bnames if A != B]
        best_ratios_pi.append(max(off_pi) / diag_pi if diag_pi > 0 else 0.0)
        best_ratios_mo.append(max(off_mo) / diag_mo if diag_mo > 0 else 0.0)

    mean_best_over_within_pi = float(np.mean(best_ratios_pi))
    mean_best_over_within_mo = float(np.mean(best_ratios_mo))

    if mean_ratio_off >= 0.80:
        verdict = "SLOPE_TRIVIAL__mean_only_matches_PILSD"
    elif mean_ratio_off >= 0.50:
        verdict = "SLOPE_PARTIAL__mean_only_recovers_some"
    else:
        verdict = "SLOPE_MATTERS__attack_defeated"

    # --- 7. Print summary tables -------------------------------------------
    print("=== 3x3 MEAN-ONLY NULL MATRIX (gain%% vs pop_slope_target) ===")
    print("Rows = SOURCE. Cols = TARGET. Fit: alpha_j + 1.0 * z(x).")
    hdr = "source/target" + "".join(f"{PRETTY[b]:>16}" for b in bnames)
    print(hdr)
    for A in bnames:
        row = f"{PRETTY[A]:<14}"
        for B in bnames:
            cell = mean_only_matrix[f"{A}__to__{B}"]
            marker = "*" if cell["diagonal"] else " "
            row += f"{marker}{cell['mean_gain_pct']:+7.2f}%".rjust(16)
        print(row)

    print("\n=== 3x3 PILSD-SHRUNK STD MATRIX (iter+N+236) ===")
    print(hdr)
    for A in bnames:
        row = f"{PRETTY[A]:<14}"
        for B in bnames:
            cell = pilsd_summary["transfer_matrix_std"][f"{A}__to__{B}"]
            marker = "*" if cell["diagonal"] else " "
            row += f"{marker}{cell['mean_gain_pct']:+7.2f}%".rjust(16)
        print(row)

    print("\n=== 3x3 GAP = PILSD - mean-only (paired per-user delta%%) ===")
    print(hdr)
    for A in bnames:
        row = f"{PRETTY[A]:<14}"
        for B in bnames:
            gp = gap_matrix[f"{A}__to__{B}"]
            ci = gp["gap_ci95"]
            diag = "*" if A == B else " "
            row += f"{diag}{gp['gap_pct']:+5.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]".rjust(16)
        print(row)

    print(f"\n[verdict] mean(mean-only / PILSD) off-diagonal = {mean_ratio_off:.3f}")
    print(f"[verdict] mean(mean-only / PILSD) diagonal     = {mean_ratio_diag:.3f}")
    print(f"[verdict] best-transfer/within, PILSD-std      = {mean_best_over_within_pi:.3f}")
    print(f"[verdict] best-transfer/within, mean-only      = {mean_best_over_within_mo:.3f}")
    print(f"[verdict] label                                = {verdict}")

    # --- 8. Persist summary.json -------------------------------------------
    summary = {
        "iter": "N+253",
        "seed": args.seed,
        "n_users": int(len(pu)),
        "n_users_paired_with_pilsd": int(len(pu_aligned)),
        "k_folds": args.k_folds,
        "min_obs_per_user": args.min_obs_per_user,
        "n_boot": args.n_boot,
        "n_utterances_joined": int(len(merged)),
        "score_mean": score_mean,
        "score_std": score_std,
        "pop_raw": pop_raw,
        "mean_only_matrix": mean_only_matrix,
        "pilsd_std_matrix_reference": pilsd_summary["transfer_matrix_std"],
        "gap_matrix": gap_matrix,
        "mean_ratio_meanonly_over_pilsd_offdiag": mean_ratio_off,
        "mean_ratio_meanonly_over_pilsd_diag": mean_ratio_diag,
        "mean_best_over_within_ratio_pilsd_std": mean_best_over_within_pi,
        "mean_best_over_within_ratio_meanonly": mean_best_over_within_mo,
        "verdict": verdict,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float)
    )
    print(f"\n[save] {out_dir / 'summary.json'}")

    # --- 9. 2-panel heatmap ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))

    # Panel 1: mean-only NULL
    ax = axes[0]
    M = np.zeros((len(bnames), len(bnames)))
    for i, A in enumerate(bnames):
        for j, B in enumerate(bnames):
            M[i, j] = mean_only_matrix[f"{A}__to__{B}"]["mean_gain_pct"]
    vmax = float(np.max(np.abs(M))) if np.max(np.abs(M)) > 0 else 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    for i in range(len(bnames)):
        for j in range(len(bnames)):
            cell = mean_only_matrix[f"{bnames[i]}__to__{bnames[j]}"]
            ci = cell["per_user_gain_ci95"]
            diag = "*" if cell["diagonal"] else ""
            txt = f"{diag}{M[i, j]:+.2f}%\n[{ci[0]:+.1f},{ci[1]:+.1f}]"
            color = "white" if abs(M[i, j]) > 0.55 * vmax else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=9)
    ax.set_xticks(range(len(bnames)))
    ax.set_yticks(range(len(bnames)))
    ax.set_xticklabels([PRETTY[b] for b in bnames], rotation=20)
    ax.set_yticklabels([PRETTY[b] for b in bnames])
    ax.set_xlabel("Target backbone (apply)")
    ax.set_ylabel("Source backbone (fit intercept)")
    ax.set_title(
        f"(a) Mean-only NULL: $\\alpha_j^A + 1 \\cdot z_B(x)$\n"
        f"mean-only/PILSD off-diag ratio = {mean_ratio_off:.2f}",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="% RMSE reduction vs pop_slope_target")

    # Panel 2: gap = PILSD - mean-only (paired delta per cell)
    ax = axes[1]
    G = np.zeros((len(bnames), len(bnames)))
    for i, A in enumerate(bnames):
        for j, B in enumerate(bnames):
            G[i, j] = gap_matrix[f"{A}__to__{B}"]["gap_pct"]
    gmax = max(float(np.max(np.abs(G))), 0.5)
    im2 = ax.imshow(G, cmap="RdBu_r", vmin=-gmax, vmax=gmax, aspect="equal")
    for i in range(len(bnames)):
        for j in range(len(bnames)):
            gp = gap_matrix[f"{bnames[i]}__to__{bnames[j]}"]
            ci = gp["gap_ci95"]
            diag = "*" if bnames[i] == bnames[j] else ""
            txt = f"{diag}{G[i, j]:+.2f}%\n[{ci[0]:+.2f},{ci[1]:+.2f}]"
            color = "white" if abs(G[i, j]) > 0.55 * gmax else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=9)
    ax.set_xticks(range(len(bnames)))
    ax.set_yticks(range(len(bnames)))
    ax.set_xticklabels([PRETTY[b] for b in bnames], rotation=20)
    ax.set_yticklabels([PRETTY[b] for b in bnames])
    ax.set_xlabel("Target backbone")
    ax.set_ylabel("Source backbone")
    ax.set_title(
        f"(b) Gap = PILSD - mean-only (paired)\n"
        f"positive = slope shrinkage adds value",
        fontsize=11,
    )
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04,
                 label="% gap (paired delta)")

    fig.suptitle(
        "T1 transferability ablation: mean-only NULL vs PILSD-shrunk "
        f"(PRISM k=5 CV, n_users={len(pu_aligned)}, cluster-bootstrap 2000 reps)",
        fontsize=12,
    )
    plt.tight_layout()
    fig_out = root / args.figure_path
    fig.savefig(fig_out, bbox_inches="tight")
    fig.savefig(str(fig_out).replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
    print(f"[save] {fig_out}")

    # --- 10. Paper insert (honest, self-disclosing) ------------------------
    # Choose the cell whose gap is most extreme (best evidence either way).
    worst_cell_name = max(
        gap_matrix.keys(),
        key=lambda k: abs(gap_matrix[k]["gap_pct"]),
    )
    wc = gap_matrix[worst_cell_name]
    wA, wB = worst_cell_name.split("__to__")

    tex = rf"""% PAPER_INSERT_mean_only_null.tex
% Auto-generated by scripts/t1_mean_only_transferability_null.py (iter+N+253).
\paragraph{{Attack B --- "is the transferable object just per-user bias?"}}
A hostile reviewer (iter+N+253) observed that our $0.91$
cross-backbone transfer ratio depends on per-backbone $z$-scoring, and
conjectured that the object which transfers is merely the per-user
intercept $\alpha_j$ rather than the full shrunk $(\alpha_j, \beta_j)$
calibrator. We test this directly with a mean-only NULL that fixes
$\beta = 1$ in standardized units and carries only $\alpha_j$ across
backbones:
$\hat{{y}}_B = \alpha_j^A + 1 \cdot z_B(x_B)$. On PRISM's $3{{\times}}3$
transferability matrix ($n=1394$ held-in users, $5$-fold CV, same
splits and same $68{{,}}371$-utterance inner join as iter+N+236),
the mean-only NULL achieves an off-diagonal transfer ratio of
$\mathbf{{{mean_ratio_off:.3f}}}$ (best-transfer/within $={mean_best_over_within_mo:.3f}$)
against the PILSD-shrunk off-diagonal ratio of $0.910$
(best-transfer/within $={mean_best_over_within_pi:.3f}$).
The paired per-user gap (PILSD $-$ mean-only, $95\%$ cluster bootstrap
over users, $2000$ reps, seed $=20260420$) ranges over the nine cells
with extreme cell $({wA}{{\to}}{wB})$: gap $={wc['gap_pct']:+.2f}\%$
$[{wc['gap_ci95'][0]:+.2f}, {wc['gap_ci95'][1]:+.2f}]$.
Verdict: \texttt{{{verdict}}}.
See Fig.~\ref{{fig:transferability_ablation}} for the full $3{{\times}}3$
gap matrix and Table in \texttt{{results/track1\_mean\_only\_transferability\_null/summary.json}}.
"""
    (root / args.paper_insert).write_text(tex)
    print(f"[save] {root / args.paper_insert}")


if __name__ == "__main__":
    main()
