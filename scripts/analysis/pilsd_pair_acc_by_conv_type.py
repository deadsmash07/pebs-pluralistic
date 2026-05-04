"""PILSD pair-accuracy stratified by PRISM conversation type.

Does PILSD gain differ across task types (unguided vs values-guided
vs controversy-guided)? Reviewer-relevant for "does it generalize
across instruction types?".

Workflow:
  1. Load track1_quadratic_calibrator/per_pair.parquet (40929 pairs)
  2. Load prism_rm_scored.parquet (68371 utterances) — map interaction_id → conversation_id
  3. Load _prism_conversation_meta.parquet — map conversation_id → conversation_type
  4. For each conv_type, compute pair-accuracy under raw vs PILSD-affine vs PILSD-quadratic
  5. Cluster-bootstrap over users (2000 reps) for CIs
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

T1_DIR = (Path(__file__).resolve().parents[2] / "1_Causal_RLHF")
OUT_DIR = (Path(__file__).resolve().parents[2] / "3_PILSD_Standalone/results/track1_pairacc_by_convtype")

N_BOOT = 2000
RNG = 20260420


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_pair = pd.read_parquet(T1_DIR / "results/track1_quadratic_calibrator/per_pair.parquet")
    rm_scored = pd.read_parquet(T1_DIR / "data/prism_rm_scored.parquet")
    meta = pd.read_parquet(T1_DIR / "data/_prism_conversation_meta.parquet")

    int_to_conv = rm_scored[["interaction_id", "conversation_id"]].drop_duplicates()
    conv_to_type = meta.set_index("conversation_id")["conversation_type"]
    int_to_type = int_to_conv.set_index("interaction_id")["conversation_id"].map(conv_to_type)

    per_pair["conv_type"] = per_pair["interaction_id"].map(int_to_type)
    df = per_pair.dropna(subset=["conv_type"])
    print(f"[merge] {len(df)} pairs with conv_type (out of {len(per_pair)})")
    print(f"conv_type counts:\n{df['conv_type'].value_counts()}")

    rng = np.random.default_rng(RNG)

    def pair_acc(m: np.ndarray) -> float:
        v = np.isfinite(m)
        return float(np.mean(m[v] > 0))

    def boot_pair_acc(df_sub: pd.DataFrame, col: str) -> tuple[float, float, float]:
        users = df_sub["user_id"].to_numpy()
        uniq = np.unique(users)
        groups = {u: np.where(users == u)[0] for u in uniq}
        m = df_sub[col].to_numpy(dtype=np.float64)
        point = pair_acc(m)
        out = np.empty(N_BOOT, dtype=np.float64)
        rng_boot = np.random.default_rng(RNG + hash(col) % 10000)
        for b in range(N_BOOT):
            s = rng_boot.choice(uniq, size=len(uniq), replace=True)
            rows = np.concatenate([groups[u] for u in s])
            out[b] = pair_acc(m[rows])
        lo, hi = np.percentile(out, [2.5, 97.5])
        return point, float(lo), float(hi)

    results = []
    for ctype, g in df.groupby("conv_type"):
        ent = {
            "conv_type": ctype,
            "n_pairs": int(len(g)),
            "n_users": int(g["user_id"].nunique()),
        }
        for cal, col in [("raw", "margin_raw"),
                         ("affine", "margin_affine"),
                         ("quadratic", "margin_quadratic")]:
            p, lo, hi = boot_pair_acc(g, col)
            ent[f"{cal}_pair_acc"] = p
            ent[f"{cal}_ci95"] = [lo, hi]
        results.append(ent)

    print("\n=== Pair-accuracy by conversation type (cluster-boot over users, 2000 reps) ===")
    print(f"{'type':24s} {'n_pairs':>8s} {'raw':>8s} {'affine':>8s} {'quadratic':>9s}")
    for r in results:
        print(f"{r['conv_type']:24s} {r['n_pairs']:>8d}  "
              f"{r['raw_pair_acc']:.4f}  {r['affine_pair_acc']:.4f}  {r['quadratic_pair_acc']:.4f}")
        print(f"{'  CI95':24s} {'':8s} "
              f"[{r['raw_ci95'][0]:.4f},{r['raw_ci95'][1]:.4f}] "
              f"[{r['affine_ci95'][0]:.4f},{r['affine_ci95'][1]:.4f}] "
              f"[{r['quadratic_ci95'][0]:.4f},{r['quadratic_ci95'][1]:.4f}]")

    # Check: are affine vs raw within CI overlap for each type (T1.MI holds)?
    t1mi_per_type = []
    for r in results:
        r_lo, r_hi = r["raw_ci95"]
        a_lo, a_hi = r["affine_ci95"]
        overlap = not (a_hi < r_lo or a_lo > r_hi)
        t1mi_per_type.append({"conv_type": r["conv_type"], "affine_ci_overlaps_raw_ci": overlap})
    all_overlap = all(x["affine_ci_overlaps_raw_ci"] for x in t1mi_per_type)
    print(f"\n[T1.MI stability across conv_types] all 3 affine CIs overlap raw CI: {all_overlap}")

    summary = {
        "per_conv_type": results,
        "n_bootstrap": N_BOOT,
        "rng_seed": RNG,
        "t1mi_per_type": t1mi_per_type,
        "t1mi_stable_across_types": bool(all_overlap),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
