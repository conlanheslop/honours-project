"""Aggregate TCGA expression by protein and add it to the peptide feature table."""

import pandas as pd

PER_SAMPLE_PATH = "data/processed/tcga_laml_expression_per_sample.csv"
AGG_PATH = "data/processed/expression_tpm_by_protein.csv"
MHC_PATH = "data/processed/mhc_cleavage_processing.csv"
OUT_PATH = "data/processed/ranking_features.csv"


def aggregate_expression():
    expr_long = pd.read_csv(PER_SAMPLE_PATH)
    agg = expr_long.groupby("protein")["tpm"].agg(["mean", "median", "std", "count"]).reset_index()
    agg = agg.rename(columns={"mean": "expression_tpm"})
    agg.to_csv(AGG_PATH, index=False)
    print(agg)
    return agg


def merge_into_ranking_features(agg):
    mhc = pd.read_csv(MHC_PATH)
    expr = agg[["protein", "expression_tpm"]]

    merged = mhc.merge(expr, on="protein", how="left")

    assert merged["expression_tpm"].isna().sum() == 0, "Some proteins missing expression_tpm"
    assert len(merged) == len(mhc), "Row count changed after merge"

    merged.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(merged)} rows to {OUT_PATH}")


def main():
    agg = aggregate_expression()
    merge_into_ranking_features(agg)


if __name__ == "__main__":
    main()