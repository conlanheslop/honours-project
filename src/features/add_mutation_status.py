"""Add protein-level mutation status to the ranking feature table."""

import pandas as pd


# ============================================================
# File paths
# ============================================================

RANKING_FEATURES_PATH = (
    "data/processed/ranking_features.csv"
)

MUTATION_PATH = (
    "data/processed/mutation_status_by_protein.csv"
)

OUT_PATH = (
    "data/processed/ranking_features.csv"
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("ADDING MUTATION STATUS")
    print("=" * 60)

    # Load existing ranking features

    ranking = pd.read_csv(
        RANKING_FEATURES_PATH
    )

    print(
        f"Existing ranking features: "
        f"{len(ranking)} rows"
    )

    # Load mutation results

    mutation = pd.read_csv(
        MUTATION_PATH
    )

    print("\nMutation data:")

    print(
        mutation.to_string(
            index=False
        )
    )

    # Keep only the columns needed
    # for the ranking feature table

    mutation = mutation[
        [
            "protein",
            "mutation_status"
        ]
    ]

    # Make sure every protein appears
    # only once

    assert not mutation[
        "protein"
    ].duplicated().any(), (
        "Mutation data contains "
        "duplicate proteins."
    )

    # Merge mutation status by protein

    merged = ranking.merge(
        mutation,
        on="protein",
        how="left"
    )

    # Make sure the merge did not
    # change the number of peptides

    assert len(merged) == len(
        ranking
    ), (
        "Row count changed after "
        "mutation merge."
    )

    # Make sure every peptide received
    # a mutation value

    missing = merged[
        "mutation_status"
    ].isna().sum()

    assert missing == 0, (
        f"{missing} rows are missing "
        f"mutation_status."
    )

    # Save updated ranking features

    merged.to_csv(
        OUT_PATH,
        index=False
    )

    print(
        f"\nAdded mutation_status to "
        f"{len(merged)} peptides."
    )

    print(
        f"Updated:\n{OUT_PATH}"
    )

    print("\nFeature columns:")

    print(
        merged.columns.tolist()
    )


if __name__ == "__main__":
    main()