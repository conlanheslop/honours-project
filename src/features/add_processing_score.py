"""Merge MHCflurry processing predictions into the cleavage-score table."""

import pandas as pd


INPUT_PATH = "data/processed/mhc_cleavage.csv"
MHCFLURRY_PATH = "data/raw/mhcflurry/mhcflurry_output.csv"
OUTPUT_PATH = "data/processed/mhc_cleavage_processing.csv"


def merge_processing_scores(mhc, mhcflurry):
    duplicate_keys = mhcflurry.duplicated(
        subset=["protein", "start", "end"], keep=False
    )
    if duplicate_keys.any():
        print(
            f"Warning: {duplicate_keys.sum()} rows share protein+start+end — "
            "investigate before trusting merge."
        )

    merged = mhc.merge(
        mhcflurry[["protein", "start", "end", "mhcflurry_processing_score"]],
        on=["protein", "start", "end"],
        how="left",
        validate="many_to_one",
    )

    if merged["mhcflurry_processing_score"].isna().sum() > 0:
        raise ValueError("Some peptides didn't get a processing score — check the merge")
    if len(merged) != len(mhc):
        raise ValueError(
            "Row count changed after merge — check for duplicate protein+start+end"
        )

    return merged


def main():
    mhc = pd.read_csv(INPUT_PATH)
    mhcflurry = pd.read_csv(MHCFLURRY_PATH)
    merged = merge_processing_scores(mhc, mhcflurry)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(merged)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()