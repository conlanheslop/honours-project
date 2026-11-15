"""Retrieve cBioPortal mutation data and summarize status by target protein."""

import os
import requests
import pandas as pd


# ============================================================
# Configuration
# ============================================================

STUDY_ID = "laml_tcga_pan_can_atlas_2018"

GENES = {
    "CCNA1",
    "HOXA9",
    "PRAME",
    "WT1",
}

BASE_URL = "https://www.cbioportal.org/api"

OUT_DIR = "data/processed"

OUT_PATH = os.path.join(
    OUT_DIR,
    "mutation_status_by_protein.csv"
)


# ============================================================
# Find mutation profile
# ============================================================

def get_mutation_profile():

    url = f"{BASE_URL}/molecular-profiles"

    params = {
        "studyId": STUDY_ID,
        "limit": 100,
    }

    response = requests.get(
        url,
        params=params
    )

    response.raise_for_status()

    profiles = response.json()

    for profile in profiles:

        # Filter for the correct study
        if profile.get("studyId") != STUDY_ID:
            continue

        if profile.get(
            "molecularAlterationType"
        ) == "MUTATION_EXTENDED":

            profile_id = profile[
                "molecularProfileId"
            ]

            print(
                f"Using mutation profile: "
                f"{profile_id}"
            )

            return profile_id

    raise RuntimeError(
        f"No mutation profile found for "
        f"{STUDY_ID}"
    )


# ============================================================
# Find sample list
# ============================================================

def get_sample_list():

    url = f"{BASE_URL}/sample-lists"

    params = {
        "studyId": STUDY_ID,
        "limit": 100,
    }

    response = requests.get(
        url,
        params=params
    )

    response.raise_for_status()

    sample_lists = response.json()

    for sample_list in sample_lists:

        # Filter for the correct study
        if sample_list.get("studyId") != STUDY_ID:
            continue

        sample_list_id = sample_list.get(
            "sampleListId",
            ""
        )

        if sample_list_id.endswith("_all"):

            print(
                f"Using sample list: "
                f"{sample_list_id}"
            )

            return sample_list_id

    raise RuntimeError(
        f"No all-samples list found for "
        f"{STUDY_ID}"
    )


# ============================================================
# Find Entrez Gene IDs
# ============================================================

def get_gene_entrez_ids():

    url = f"{BASE_URL}/genes"

    params = {
        "geneId": ",".join(
            sorted(GENES)
        ),
        "limit": 100,
    }

    response = requests.get(
        url,
        params=params
    )

    response.raise_for_status()

    genes = response.json()

    mapping = {}

    for gene in genes:

        symbol = gene.get(
            "hugoGeneSymbol"
        )

        entrez_id = gene.get(
            "entrezGeneId"
        )

        if symbol in GENES:
            mapping[symbol] = entrez_id

    missing = GENES - set(
        mapping.keys()
    )

    if missing:

        raise RuntimeError(
            f"Could not find Entrez IDs for: "
            f"{sorted(missing)}"
        )

    print("\nGene mapping:")

    for symbol, entrez_id in sorted(
        mapping.items()
    ):

        print(
            f"  {symbol}: {entrez_id}"
        )

    return mapping


# ============================================================
# Download mutations
# ============================================================

def get_mutations(
    molecular_profile_id,
    sample_list_id,
    gene_ids
):

    url = (
        f"{BASE_URL}/molecular-profiles/"
        f"{molecular_profile_id}/mutations"
    )

    # Send entrezGeneId as a list - requests handles this
    # by repeating the parameter
    params = {
        "sampleListId": sample_list_id,
        "entrezGeneId": [
            str(gene_ids[gene])
            for gene in sorted(gene_ids)
        ],
        "projection": "SUMMARY",
        "pageSize": 100000,
    }

    response = requests.get(
        url,
        params=params
    )

    response.raise_for_status()

    mutations = response.json()

    print(
        f"\nRetrieved "
        f"{len(mutations)} mutation records"
    )

    return mutations


# ============================================================
# Calculate mutation frequency
# ============================================================

def calculate_mutation_frequency(
    mutations,
    gene_ids
):

    # Convert Entrez ID -> gene symbol

    reverse_mapping = {
        entrez_id: gene
        for gene, entrez_id
        in gene_ids.items()
    }

    # Store unique mutated samples
    # for each gene

    mutated_samples = {
        gene: set()
        for gene in GENES
    }

    # Store all samples represented
    # in the mutation data

    all_samples = set()

    for mutation in mutations:

        entrez_id = mutation.get(
            "entrezGeneId"
        )

        sample_id = mutation.get(
            "sampleId"
        )

        if sample_id:

            all_samples.add(
                sample_id
            )

        gene = reverse_mapping.get(
            entrez_id
        )

        if gene is not None and sample_id:

            mutated_samples[
                gene
            ].add(sample_id)

    cohort_size = len(
        all_samples
    )

    print(
        f"\nSamples represented in "
        f"mutation data: {cohort_size}"
    )

    if cohort_size == 0:

        raise RuntimeError(
            "No sample IDs were returned."
        )

    records = []

    for gene in sorted(GENES):

        mutated_count = len(
            mutated_samples[gene]
        )

        frequency = (
            mutated_count / cohort_size
        )

        records.append(
            {
                "protein": gene,
                "mutated_samples": mutated_count,
                "cohort_samples": cohort_size,
                "mutation_status": frequency,
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    print("=" * 60)
    print("cBioPortal AML MUTATION STATUS")
    print("=" * 60)

    print("\nFinding mutation profile...")

    mutation_profile_id = (
        get_mutation_profile()
    )

    print("\nFinding sample list...")

    sample_list_id = (
        get_sample_list()
    )

    print("\nFinding gene IDs...")

    gene_ids = (
        get_gene_entrez_ids()
    )

    print("\nDownloading mutations...")

    mutations = get_mutations(
        mutation_profile_id,
        sample_list_id,
        gene_ids
    )

    print(
        "\nCalculating mutation frequencies..."
    )

    result = (
        calculate_mutation_frequency(
            mutations,
            gene_ids
        )
    )

    print("\nMutation statistics:")

    print(
        result.to_string(
            index=False
        )
    )

    result.to_csv(
        OUT_PATH,
        index=False
    )

    print(
        f"\nSaved mutation statistics to:"
        f"\n{OUT_PATH}"
    )


if __name__ == "__main__":
    main()