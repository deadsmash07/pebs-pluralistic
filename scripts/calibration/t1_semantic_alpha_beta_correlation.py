#!/usr/bin/env python3
"""
t1_semantic_alpha_beta_correlation.py
--------------------------------------
Tests whether PEBS's fitted per-user (alpha_j, beta_j) calibration parameters
correlate with SEMANTIC features of PRISM users' self-descriptions and stated
preferences.

Hypothesis: the 8.58% RMSE improvement from per-user calibration has NO
demographic explanation. This script extends that
null-finding test to SEMANTIC features:

    1. Free-text TF-IDF -> PCA top-5 PCs   (self_description, system_string)
    2. Structured preference vectors       (stated_prefs: 10-D, lm_usecases: 19-D)
    3. Hand-crafted simple features:
          - text length
          - sentiment (VADER-style via positive/negative word lists; no network)
          - technical-vocabulary indicator (regex)

For every feature we compute Spearman rho vs alpha_j and vs beta_j and report
Bonferroni-adjusted p-values.

CPU only, ~1 min expected.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Simple sentiment & tech-vocab lexicons (offline, no downloads).
# ---------------------------------------------------------------------------
POS_WORDS = {
    "good", "great", "positive", "friendly", "happy", "kind", "helpful",
    "honest", "respectful", "respect", "love", "care", "support", "supportive",
    "open", "open-minded", "understanding", "compassionate", "compassion",
    "empathetic", "empathy", "patient", "warm", "welcoming", "polite",
    "thoughtful", "generous", "fair", "fun", "joy", "optimistic", "trust",
    "trustworthy", "loyal", "sincere", "grateful", "gratitude",
}
NEG_WORDS = {
    "bad", "negative", "hate", "angry", "sad", "rude", "lazy", "harsh",
    "abrasive", "aggressive", "mean", "cold", "distant", "selfish", "cruel",
    "unkind", "judgmental", "judgement", "dishonest", "liar", "stupid",
    "terrible", "awful", "horrible", "pessimistic", "skeptical", "suspicious",
    "distrust", "frustrated", "annoyed", "angry", "bitter", "resentful",
}
CRITICAL_WORDS = {
    "critical", "discerning", "analytical", "rigorous", "skeptical",
    "evaluate", "assessment", "scrutiny", "examine", "question", "challenge",
    "precise", "precision", "accurate", "accuracy", "detail-oriented",
    "thorough", "exacting", "pedantic", "meticulous",
}
TECH_TERMS = {
    "code", "coding", "programming", "program", "python", "java", "javascript",
    "algorithm", "developer", "engineer", "engineering", "software", "debug",
    "debugging", "api", "database", "sql", "ml", "ai", "machine",
    "statistics", "statistical", "math", "mathematics", "physics",
    "research", "researcher", "scientist", "scientific", "technical",
    "technology", "data", "dataset", "analysis", "analyst", "linux",
    "git", "github", "compiler", "framework", "library", "stack",
    "frontend", "backend", "function", "variable", "numpy", "pandas",
    "tensor", "neural", "network",
}


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def text_length(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.len().to_numpy(dtype=float)


def word_count(series: pd.Series) -> np.ndarray:
    return np.asarray([len(tokenize(t)) for t in series], dtype=float)


def lexicon_fraction(series: pd.Series, lex: set[str]) -> np.ndarray:
    out = np.zeros(len(series), dtype=float)
    for i, t in enumerate(series):
        toks = tokenize(t)
        if not toks:
            continue
        out[i] = sum(1 for w in toks if w in lex) / len(toks)
    return out


def sentiment_score(series: pd.Series) -> np.ndarray:
    """pos-frac minus neg-frac (in [-1, 1])."""
    return lexicon_fraction(series, POS_WORDS) - lexicon_fraction(series, NEG_WORDS)


def tech_indicator(series: pd.Series) -> np.ndarray:
    """Fraction of tokens hitting the TECH_TERMS set (continuous 0-1)."""
    return lexicon_fraction(series, TECH_TERMS)


def critical_indicator(series: pd.Series) -> np.ndarray:
    return lexicon_fraction(series, CRITICAL_WORDS)


def tfidf_pca(series: pd.Series, n_components: int = 5, max_features: int = 2000,
              min_df: int = 5) -> tuple[np.ndarray, list[str]]:
    """TF-IDF -> TruncatedPCA on dense matrix. Returns (PC_matrix, feature_names)."""
    vec = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"[a-zA-Z][a-zA-Z\-']+",
        stop_words="english",
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, 2),
    )
    X = vec.fit_transform(series.astype(str)).toarray()
    if X.shape[1] == 0:
        return np.zeros((len(series), 0)), []
    n_comp = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
    if n_comp < 1:
        return np.zeros((len(series), 0)), []
    pca = PCA(n_components=n_comp, random_state=0)
    PCs = pca.fit_transform(X)
    # Name each PC by its top-3 loading tokens (magnitude).
    vocab = vec.get_feature_names_out()
    names = []
    for i in range(n_comp):
        top_idx = np.argsort(np.abs(pca.components_[i]))[::-1][:3]
        names.append(f"PC{i+1}[{','.join(vocab[j] for j in top_idx)}]")
    return PCs, names


def dict_to_matrix(series: pd.Series, numeric_only: bool = True) -> tuple[np.ndarray, list[str]]:
    """Convert a column of dicts into a dense matrix with stable column order."""
    keys: list[str] = []
    seen: set[str] = set()
    for d in series:
        if isinstance(d, dict):
            for k, v in d.items():
                if k in seen:
                    continue
                if numeric_only and not isinstance(v, (int, float, bool)):
                    continue
                if numeric_only and isinstance(v, bool):
                    # treat bool as numeric
                    pass
                keys.append(k)
                seen.add(k)
    X = np.full((len(series), len(keys)), np.nan, dtype=float)
    for i, d in enumerate(series):
        if not isinstance(d, dict):
            continue
        for j, k in enumerate(keys):
            v = d.get(k, None)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                X[i, j] = float(v)
            elif isinstance(v, bool):
                X[i, j] = float(v)
    # Replace NaN with column mean (only missing value handling).
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = col_mean[j]
    return X, keys


# ---------------------------------------------------------------------------
# Correlation + Bonferroni
# ---------------------------------------------------------------------------
def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return (np.nan, np.nan)
    res = stats.spearmanr(x[mask], y[mask])
    return (float(res.statistic), float(res.pvalue))


def bonferroni_adjust(rows: list[dict], p_col: str = "p_raw") -> list[dict]:
    m = sum(1 for r in rows if np.isfinite(r[p_col]))
    for r in rows:
        p = r[p_col]
        r["p_bonf"] = min(1.0, p * m) if np.isfinite(p) else np.nan
        r["bonf_sig_0_05"] = bool(np.isfinite(r["p_bonf"]) and r["p_bonf"] < 0.05)
    return rows


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run(args):
    data_dir = Path(args.data_dir)
    demo = pd.read_parquet(data_dir / "prism_demographics.parquet")
    cal = pd.read_parquet(data_dir / "prism_user_calibrators_shrunk.parquet")

    # Join on user_id
    df = cal.merge(demo, on="user_id", how="inner")
    print(f"[t1_semantic] merged: {len(df)} users "
          f"(cal={len(cal)}, demo={len(demo)}, inter={len(df)})")

    alpha = df["alpha_j"].to_numpy(dtype=float)
    beta = df["beta_j"].to_numpy(dtype=float)

    rows: list[dict] = []

    # ---- Simple features for 2 free-text columns
    for col in ("self_description", "system_string"):
        txt = df[col].astype(str)
        feats = {
            f"{col}.len_chars": text_length(txt),
            f"{col}.word_count": word_count(txt),
            f"{col}.pos_frac": lexicon_fraction(txt, POS_WORDS),
            f"{col}.neg_frac": lexicon_fraction(txt, NEG_WORDS),
            f"{col}.sentiment": sentiment_score(txt),
            f"{col}.tech_frac": tech_indicator(txt),
            f"{col}.critical_frac": critical_indicator(txt),
        }
        for name, x in feats.items():
            for target, y in (("alpha_j", alpha), ("beta_j", beta)):
                rho, p = spearman(x, y)
                rows.append({
                    "family": "simple_text",
                    "feature": name,
                    "target": target,
                    "rho": rho,
                    "p_raw": p,
                    "n_nonzero": int(np.sum(x != 0)) if not np.all(np.isnan(x)) else 0,
                })

    # ---- TF-IDF PCA on the two free-text fields
    for col in ("self_description", "system_string"):
        PCs, names = tfidf_pca(df[col], n_components=5, max_features=2000, min_df=5)
        for i, pc_name in enumerate(names):
            x = PCs[:, i]
            for target, y in (("alpha_j", alpha), ("beta_j", beta)):
                rho, p = spearman(x, y)
                rows.append({
                    "family": "tfidf_pca",
                    "feature": f"{col}.{pc_name}",
                    "target": target,
                    "rho": rho,
                    "p_raw": p,
                    "n_nonzero": int(PCs.shape[0]),
                })

    # ---- Structured-dict features (stated_prefs, lm_usecases)
    for col in ("stated_prefs", "lm_usecases"):
        X, keys = dict_to_matrix(df[col])
        print(f"[t1_semantic] {col}: dict expanded to {X.shape[1]} numeric keys")
        # Test each key directly
        for j, k in enumerate(keys):
            x = X[:, j]
            for target, y in (("alpha_j", alpha), ("beta_j", beta)):
                rho, p = spearman(x, y)
                rows.append({
                    "family": f"dict_{col}",
                    "feature": f"{col}.{k}",
                    "target": target,
                    "rho": rho,
                    "p_raw": p,
                    "n_nonzero": int(np.sum(x != 0)),
                })
        # Also PCA the structured matrix
        if X.shape[1] >= 2:
            Xs = StandardScaler().fit_transform(X)
            n_comp = min(5, Xs.shape[1] - 1)
            pca = PCA(n_components=n_comp, random_state=0)
            PCs = pca.fit_transform(Xs)
            for i in range(n_comp):
                top = np.argsort(np.abs(pca.components_[i]))[::-1][:3]
                pc_name = f"PC{i+1}[{','.join(keys[j] for j in top)}]"
                for target, y in (("alpha_j", alpha), ("beta_j", beta)):
                    rho, p = spearman(PCs[:, i], y)
                    rows.append({
                        "family": f"dict_pca_{col}",
                        "feature": f"{col}.{pc_name}",
                        "target": target,
                        "rho": rho,
                        "p_raw": p,
                        "n_nonzero": int(PCs.shape[0]),
                    })

    # ---- Aggregate scale-use controls for stated_prefs and lm_usecases.
    # These test the "response-scale use" hypothesis: if alpha_j captures baseline
    # leniency, it should correlate with stated_prefs_MEAN (not any single rating).
    for col in ("stated_prefs", "lm_usecases"):
        X, _ = dict_to_matrix(df[col])
        if X.shape[1] == 0:
            continue
        # NaN-safe aggregates (dict_to_matrix already fills NaN with col mean, so
        # these are robust to a couple of missing fields).
        agg = {
            f"{col}.AGGREGATE_mean":  np.nanmean(X, axis=1),
            f"{col}.AGGREGATE_std":   np.nanstd(X, axis=1),
            f"{col}.AGGREGATE_count": np.nansum(X > 0, axis=1).astype(float),
        }
        for name, x in agg.items():
            for target, y in (("alpha_j", alpha), ("beta_j", beta)):
                rho, p = spearman(x, y)
                rows.append({
                    "family": "aggregate_scale_use",
                    "feature": name,
                    "target": target,
                    "rho": rho,
                    "p_raw": p,
                    "n_nonzero": int(np.sum(x != 0)),
                })

    # ---- Bonferroni adjustment across ALL tests
    rows = bonferroni_adjust(rows, p_col="p_raw")
    res = pd.DataFrame(rows).sort_values("p_raw")
    print(f"[t1_semantic] Total tests: {len(res)}  "
          f"Bonferroni-sig at 0.05: {int(res['bonf_sig_0_05'].sum())}")

    out_csv = data_dir.parent / "T1_SEMANTIC_CORRELATIONS.csv"
    res.to_csv(out_csv, index=False)
    print(f"[t1_semantic] Wrote {out_csv}")

    # Print top hits (smallest raw p) for both targets
    print("\n=== TOP 10 by raw p (alpha_j) ===")
    print(res[res.target == "alpha_j"].head(10).to_string(index=False))
    print("\n=== TOP 10 by raw p (beta_j) ===")
    print(res[res.target == "beta_j"].head(10).to_string(index=False))

    print("\n=== Bonferroni-significant hits (p_bonf < 0.05) ===")
    sig = res[res["bonf_sig_0_05"]]
    if len(sig) == 0:
        print("  NONE  (confirms 'idiosyncratic' hypothesis)")
    else:
        print(sig.to_string(index=False))

    return res


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",
                   default="<DATA_ROOT>/"
                           "1_Causal_RLHF/data",
                   help="Directory with prism_demographics.parquet and "
                        "prism_user_calibrators_shrunk.parquet")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
