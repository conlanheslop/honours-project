"""Add NetChop cleavage scores to the cleaned MHC peptide table."""

import re
from pathlib import Path

import pandas as pd


# ============================================================
# File paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MHC_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_mhc.csv"
NETCHOP_FILE = PROJECT_ROOT / "data" / "raw" / "netchop" / "netchop_output.txt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "mhc_cleavage.csv"


# ============================================================
# NetChop ID -> project protein mapping
# ============================================================

PROTEIN_MAP = {
    "sp_P78395_": "PRAME",
    "sp_P19544_": "WT1",
    "sp_P78396_": "CCNA1",
    "sp_P31269_": "HOXA9",
}


# ============================================================
# Read NetChop output
# ============================================================

def parse_netchop_output(file_path):
    """
    Read NetChop output and extract:

        protein
        position
        amino_acid
        cleavage_prediction
        cleavage_score

    Returns a pandas DataFrame.
    """

    records = []

    # Look for the important five pieces anywhere in a line:
    #
    # position  amino_acid  prediction  score  NetChop_ID
    #
    # Example:
    # 1 M S 0.740804 sp_P78395_

    prediction_pattern = re.compile(
        r"(\d+)\s+([A-Z])\s+([S.])\s+"
        r"([0-9]*\.?[0-9]+)\s+"
        r"(sp_[A-Za-z0-9]+_)"
    )

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:

        for line in file:

            # Search anywhere in the line rather than requiring
            # the line to have a particular exact format.
            match = prediction_pattern.search(line)

            if not match:
                continue

            position = int(match.group(1))
            amino_acid = match.group(2)
            cleavage_prediction = match.group(3)
            cleavage_score = float(match.group(4))
            netchop_id = match.group(5)

            # Check that this is one of our four proteins
            if netchop_id not in PROTEIN_MAP:
                print(
                    f"WARNING: Unknown NetChop identifier: {netchop_id}"
                )
                continue

            protein = PROTEIN_MAP[netchop_id]

            records.append(
                {
                    "protein": protein,
                    "position": position,
                    "amino_acid": amino_acid,
                    "cleavage_prediction": cleavage_prediction,
                    "cleavage_score": cleavage_score,
                }
            )

    return pd.DataFrame(records)


# ============================================================
# Main processing
# ============================================================

def main():

    print("=" * 60)
    print("ADDING NETCHOP CLEAVAGE SCORES")
    print("=" * 60)

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    if not MHC_FILE.exists():
        raise FileNotFoundError(
            f"Could not find MHC file:\n{MHC_FILE}"
        )

    if not NETCHOP_FILE.exists():
        raise FileNotFoundError(
            f"Could not find NetChop file:\n{NETCHOP_FILE}"
        )

    # --------------------------------------------------------
    # Load MHC dataset
    # --------------------------------------------------------

    print("\nLoading MHC dataset...")

    mhc_df = pd.read_csv(MHC_FILE)

    print(f"Peptides loaded: {len(mhc_df)}")
    print(f"Proteins: {mhc_df['protein'].unique().tolist()}")

    # --------------------------------------------------------
    # Parse NetChop output
    # --------------------------------------------------------

    print("\nParsing NetChop output...")

    cleavage_df = parse_netchop_output(NETCHOP_FILE)

    if cleavage_df.empty:
        raise ValueError(
            "No NetChop prediction lines were found. "
            "Check the contents of netchop_output.txt."
        )

    print(f"NetChop predictions parsed: {len(cleavage_df)}")

    print("\nNetChop predictions by protein:")
    print(
        cleavage_df.groupby("protein")
        .size()
        .to_string()
    )

    # --------------------------------------------------------
    # Check for duplicate protein/position combinations
    # --------------------------------------------------------

    duplicates = cleavage_df.duplicated(
        subset=["protein", "position"]
    )

    if duplicates.any():

        duplicate_count = duplicates.sum()

        raise ValueError(
            f"Found {duplicate_count} duplicate "
            "protein/position combinations in NetChop output."
        )

    # --------------------------------------------------------
    # Prepare the cleavage table
    # --------------------------------------------------------

    # For each peptide, the relevant NetChop prediction is the
    # score at the peptide's C-terminal position (the 'end').
    #
    # Therefore:
    #
    #     peptide end -> NetChop position
    #
    cleavage_scores = cleavage_df[
        ["protein", "position", "cleavage_score"]
    ].copy()

    cleavage_scores = cleavage_scores.rename(
        columns={"position": "end"}
    )

    # --------------------------------------------------------
    # Merge cleavage scores onto the MHC dataset
    # --------------------------------------------------------

    print("\nMatching peptides to NetChop predictions...")

    original_count = len(mhc_df)

    result_df = mhc_df.merge(
        cleavage_scores,
        on=["protein", "end"],
        how="left",
        validate="many_to_one"
    )

    # --------------------------------------------------------
    # Check that the number of peptides did not change
    # --------------------------------------------------------

    if len(result_df) != original_count:
        raise ValueError(
            "ERROR: Number of peptides changed after merging."
        )

    # --------------------------------------------------------
    # Check for missing cleavage scores
    # --------------------------------------------------------

    missing = result_df["cleavage_score"].isna()

    missing_count = missing.sum()

    print("\nMatching results:")
    print(f"Total peptides: {len(result_df)}")
    print(f"Matched peptides: {len(result_df) - missing_count}")
    print(f"Missing cleavage scores: {missing_count}")

    if missing_count > 0:

        print("\nMissing matches by protein:")

        print(
            result_df.loc[missing, "protein"]
            .value_counts()
            .to_string()
        )

        print("\nExample unmatched peptides:")

        print(
            result_df.loc[
                missing,
                ["protein", "start", "end", "peptide"]
            ]
            .head(20)
            .to_string(index=False)
        )

        raise ValueError(
            "\nSome peptides could not be matched to NetChop. "
            "The output file was NOT created. "
            "Check the protein sequences/positions."
        )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)

    print(f"\nOutput file:")
    print(OUTPUT_FILE)

    print(f"\nRows: {len(result_df)}")
    print(f"Columns: {len(result_df.columns)}")

    print("\nNew column:")
    print("cleavage_score")

    print("\nCleavage score statistics:")
    print(
        result_df["cleavage_score"].describe()
    )

    print("\nFirst five cleavage scores:")
    print(
        result_df[
            ["protein", "start", "end", "peptide", "cleavage_score"]
        ].head().to_string(index=False)
    )


if __name__ == "__main__":
    main()