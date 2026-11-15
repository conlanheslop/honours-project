"""Rank peptides with a weighted score from MHC binding, cleavage, and expression."""

import os
import math
import pandas as pd

IN_PATH = "data/processed/ranking_features.csv"
OUT_PATH = "results/linear/rankings.csv"

REQUIRED_COLUMNS = ["peptide", "protein", "allele", "BA_rank", "cleavage_score", "expression_tpm"]

W_MHC = 0.50
W_CLEAVAGE = 0.25
W_EXPRESSION = 0.25

assert abs(W_MHC + W_CLEAVAGE + W_EXPRESSION - 1.0) < 1e-9, "Weights must sum to 1.0"


def min_max_normalize(series):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def build_features(df):
    df = df.copy()

    # --- Global normalisation (used for linear_score / the mixed ranking) ---
    df["mhc_binding_feature"] = 1 - min_max_normalize(df["BA_rank"])
    df["cleavage_feature"] = min_max_normalize(df["cleavage_score"])

    log_expr = (df["expression_tpm"] + 1).apply(lambda x: math.log2(x) if x > 0 else 0.0)
    df["expression_feature"] = min_max_normalize(log_expr)

    # --- Within-protein versions, kept for direct comparison against baseline ---
    df["mhc_binding_feature_within_protein"] = 1 - df.groupby("protein")["BA_rank"].transform(min_max_normalize)
    df["cleavage_feature_within_protein"] = df.groupby("protein")["cleavage_score"].transform(min_max_normalize)
    # expression_tpm is a per-protein constant, so within-protein normalisation
    # is degenerate by construction (every peptide in a group shares one value) —
    # not computed here for that reason.

    return df


def compute_linear_score(df):
    df = df.copy()
    df["linear_score"] = (
        W_MHC * df["mhc_binding_feature"]
        + W_CLEAVAGE * df["cleavage_feature"]
        + W_EXPRESSION * df["expression_feature"]
    )

    # Global rank/percentile — the primary output, comparable across proteins.
    df["linear_rank"] = df["linear_score"].rank(ascending=False, method="min").astype(int)
    df["linear_percentile"] = df["linear_score"].rank(pct=True, ascending=True) * 100

    # Within-protein rank kept too, for direct baseline comparison later.
    df["linear_rank_within_protein"] = df.groupby("protein")["linear_score"].rank(ascending=False, method="min").astype(int)

    return df


def main():
    df = pd.read_csv(IN_PATH)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing_cols, f"Missing required columns: {missing_cols}"

    missing_vals = df[REQUIRED_COLUMNS].isna().sum()
    assert missing_vals.sum() == 0, f"Missing values found:\n{missing_vals[missing_vals > 0]}"

    n_in = len(df)

    df = build_features(df)
    df = compute_linear_score(df)

    assert len(df) == n_in, "Row count changed during processing"

    # Sort the output file itself by linear_score, mixed across proteins.
    df = df.sort_values("linear_score", ascending=False).reset_index(drop=True)

    os.makedirs("results/linear", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(df)} rows to {OUT_PATH}, sorted by linear_score (all proteins mixed)")
    print(f"\nWeights: MHC={W_MHC}, cleavage={W_CLEAVAGE}, expression={W_EXPRESSION}")
    print("Normalisation: GLOBAL across all proteins\n")

    print("Top 10 peptides overall:")
    print(df[["protein", "peptide", "BA_rank", "cleavage_score", "expression_tpm",
              "linear_score", "linear_rank"]].head(10).to_string(index=False))

    print("\nProtein representation in top 100:")
    print(df.head(100)["protein"].value_counts())


if __name__ == "__main__":
    main()