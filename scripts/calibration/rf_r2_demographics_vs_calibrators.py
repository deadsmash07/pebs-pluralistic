"""RandomForest R² predicting per-user PILSD calibrators (α_j, β_j) from PRISM demographics.

Reviewer W5 ask: the paper reports ANOVA η²≤0.018 for each demographic
individually (only gender→β_j survives Bonferroni), which is evidence
that LINEAR partitions don't explain much calibrator variance. But a
cleaner claim is the NON-LINEAR upper bound: can a flexible model
(RandomForest) jointly predict (α_j, β_j) from all six demographic
variables? If 5-fold-CV R² is near zero, the "idiosyncratic calibrator"
claim holds non-parametrically.

Output: `results/track1_rf_r2_demographics.json` with per-target R²
summaries (α, β, naive-OLS α, naive-OLS β) at 5-fold CV.

Methodology:
- One-hot encode categorical demographics; drop rows with missing user_id match
- RandomForestRegressor(n_estimators=500, min_samples_leaf=5, n_jobs=-1)
- Seed=0 fixed; report mean +/- std across folds
- Negative R² means worse-than-mean-baseline (genuinely null predictive signal)

Scope: closes the adversarial-review claim that the 6-PRISM-demographic
features miss most of the per-user calibrator signal.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score


DEMOGRAPHIC_COLS = [
    "age", "gender", "education", "employment_status",
    "english_proficiency", "religion",
]

TARGETS = ["alpha_j", "beta_j", "alpha_naive_ols", "beta_naive_ols"]


def main() -> None:
    cal = pd.read_parquet("data/prism_user_calibrators_shrunk.parquet")
    dem = pd.read_parquet("data/prism_demographics.parquet")

    df = cal.merge(dem[["user_id"] + DEMOGRAPHIC_COLS], on="user_id", how="inner")
    # Coerce dict / list cells to their string repr (religion is stored as dict)
    for c in DEMOGRAPHIC_COLS:
        df[c] = df[c].astype(str)
    df = df[~df[DEMOGRAPHIC_COLS].isin(["None", "nan", "NaN"]).any(axis=1)]
    n = len(df)

    # One-hot encode
    X = pd.get_dummies(df[DEMOGRAPHIC_COLS], drop_first=False, dtype=float).values
    print(f"[W5] merged + dropna: n={n}  X.shape={X.shape}")

    results = {"n_users": int(n), "n_features_one_hot": int(X.shape[1]),
               "demographic_cols": DEMOGRAPHIC_COLS, "per_target": {}}

    for target in TARGETS:
        y = df[target].astype(float).values
        rf = RandomForestRegressor(
            n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=0,
        )
        # 5-fold CV R²
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        r2 = cross_val_score(rf, X, y, scoring="r2", cv=cv, n_jobs=1)
        rf.fit(X, y)
        importances = sorted(
            zip(pd.get_dummies(df[DEMOGRAPHIC_COLS], drop_first=False, dtype=float).columns,
                rf.feature_importances_),
            key=lambda p: -p[1],
        )[:10]
        results["per_target"][target] = {
            "r2_mean": float(r2.mean()),
            "r2_std": float(r2.std(ddof=1)),
            "r2_folds": [float(x) for x in r2],
            "target_var": float(np.var(y, ddof=1)),
            "top_10_importances": [(c, float(v)) for c, v in importances],
        }
        print(f"[W5] {target:20s}  R2 = {r2.mean():+.4f} +/- {r2.std(ddof=1):.4f}  "
              f"(folds: {', '.join(f'{x:+.3f}' for x in r2)})")

    out = Path("results/track1_rf_r2_demographics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[W5] wrote {out}")

    print("\n=== Interpretation ===")
    for t in TARGETS:
        r2 = results["per_target"][t]["r2_mean"]
        verdict = "NON-PREDICTIVE (near-zero R^2)" if r2 < 0.05 else \
                  "WEAKLY PREDICTIVE" if r2 < 0.15 else "PREDICTIVE"
        print(f"  {t}: R^2 = {r2:+.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
