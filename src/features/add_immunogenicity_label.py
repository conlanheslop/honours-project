"""Merge positive HLA-A*02:01 T-cell and MHC-ligand evidence from IEDB."""

import os
import pandas as pd

TCELL_ASSAY_PATH = "data/raw/iedb/tcell_table_export.csv"
MHC_LIGAND_ASSAY_PATH = "data/raw/iedb/mhc_ligand_table_export.csv"
RANKING_FEATURES_PATH = "data/processed/ranking_features.csv"
LABELS_OUT_PATH = "data/processed/immunogenicity_labels.csv"
RANKING_FEATURES_OUT_PATH = "data/processed/ranking_features.csv"  # overwrite in place

TARGET_ALLELE = "HLA-A*02:01"

PROTEIN_MAP = {
    "Melanoma antigen preferentially expressed in tumors": "PRAME",
    "melanoma antigen preferentially expressed in tumors": "PRAME",
    "Preferentially expressed antigen in melanoma": "PRAME",
    "preferentially expressed antigen in melanoma": "PRAME",
    "Cyclin-A1": "CCNA1",
    "Isoform 3 of Cyclin-A1": "CCNA1",
    "cyclin-A1 [Homo sapiens]": "CCNA1",
    "Homeobox protein Hox-A9": "HOXA9",
    "Wilms tumor protein": "WT1",
    "Isoform 7 of Wilms tumor protein": "WT1",
    "Wilms tumor protein isoform A [Homo sapiens]": "WT1",
    "Wilms tumor protein isoform D [Homo sapiens]": "WT1",
    "WT1, partial [Homo sapiens]": "WT1",
}


def clean_peptide_set(df, protein_col, name_col, start_col, end_col, mhc_col, label):
    df = df.copy()

    if mhc_col in df.columns:
        before = len(df)
        df = df[df[mhc_col] == TARGET_ALLELE]
        print(f"[{label}] Filtered to {TARGET_ALLELE}: {before} -> {len(df)} rows.")

    df["protein"] = df[protein_col].map(PROTEIN_MAP)
    unmapped = df[df["protein"].isna()][protein_col].unique()
    if len(unmapped) > 0:
        print(f"[{label}] WARNING: unmapped source molecule(s), dropped: {list(unmapped)}")
    df = df.dropna(subset=["protein"])

    df["peptide"] = df[name_col].str.replace(r"\s*\+.*", "", regex=True)
    df["start"] = df[start_col]
    df["end"] = df[end_col]

    lengths = df["peptide"].str.len()
    out_of_window = (~lengths.between(8, 11)).sum()
    if out_of_window > 0:
        print(f"[{label}] Dropping {out_of_window} epitope(s) outside the 8-11mer window.")
    df = df[lengths.between(8, 11)]

    before = len(df)
    df = df.drop_duplicates(subset=["protein", "peptide", "start", "end"])
    print(f"[{label}] {before} -> {len(df)} unique peptide-protein-position rows.\n")

    return df[["protein", "peptide", "start", "end"]].reset_index(drop=True)


def load_tcell_assay_export(path):
    df = pd.read_csv(path, header=[0, 1])
    df.columns = [
        f"{a.strip()} - {b.strip()}" if "Unnamed" not in str(b) else a.strip()
        for a, b in df.columns
    ]

    qual_col = "Assay - Qualitative Measurement"
    if qual_col in df.columns:
        before = len(df)
        df = df[df[qual_col].astype(str).str.startswith("Positive")]
        print(f"[tcell] Filtered to Positive* qualitative measurement: {before} -> {len(df)} rows.")

    return clean_peptide_set(
        df, "Epitope - Source Molecule", "Epitope - Name",
        "Epitope - Starting Position", "Epitope - Ending Position",
        "MHC Restriction - Name", "tcell",
    )


def load_mhc_ligand_assay_export(path):
    df = pd.read_csv(path)

    qual_col = "Assay - Qualitative Measurement"
    if qual_col in df.columns:
        before = len(df)
        df = df[df[qual_col].astype(str).str.startswith("Positive")]
        print(f"[mhc_ligand] Filtered to Positive* qualitative measurement: {before} -> {len(df)} rows.")

    return clean_peptide_set(
        df, "Epitope - Source Molecule", "Epitope - Name",
        "Epitope - Starting Position", "Epitope - Ending Position",
        "MHC Restriction - Name", "mhc_ligand",
    )


def main():
    tcell = load_tcell_assay_export(TCELL_ASSAY_PATH)
    ligand = load_mhc_ligand_assay_export(MHC_LIGAND_ASSAY_PATH)

    tcell_keys = set(zip(tcell["protein"], tcell["start"], tcell["end"]))
    ligand_keys = set(zip(ligand["protein"], ligand["start"], ligand["end"]))
    both = tcell_keys & ligand_keys

    print(f"tcell positives: {len(tcell_keys)}")
    print(f"mhc_ligand positives: {len(ligand_keys)}")
    print(f"confirmed by BOTH methods: {len(both)}")
    if both:
        print(f"  {both}")

    ranking_features = pd.read_csv(RANKING_FEATURES_PATH)

    # Drop any previous versions of these columns so the merge below replaces
    # them cleanly rather than creating _x/_y suffix duplicates.
    ranking_features = ranking_features.drop(
        columns=["tcell_evidence", "mhc_ligand_evidence"], errors="ignore"
    )

    labels = ranking_features[["peptide", "protein", "start", "end"]].copy()
    labels["tcell_evidence"] = labels.apply(
        lambda r: int((r["protein"], r["start"], r["end"]) in tcell_keys), axis=1
    )
    labels["mhc_ligand_evidence"] = labels.apply(
        lambda r: int((r["protein"], r["start"], r["end"]) in ligand_keys), axis=1
    )

    print(f"\nMatched to candidate rows:")
    print(f"  tcell_evidence = 1 for {labels['tcell_evidence'].sum()} rows")
    print(f"  mhc_ligand_evidence = 1 for {labels['mhc_ligand_evidence'].sum()} rows")
    print(f"  either = 1 for {((labels['tcell_evidence']==1) | (labels['mhc_ligand_evidence']==1)).sum()} rows")

    print("\nBy protein:")
    for col in ["tcell_evidence", "mhc_ligand_evidence"]:
        print(f"  {col}:")
        print(labels[labels[col] == 1].groupby("protein").size().to_string())

    os.makedirs(os.path.dirname(LABELS_OUT_PATH), exist_ok=True)
    labels.to_csv(LABELS_OUT_PATH, index=False)
    print(f"\nWrote {len(labels)} rows to {LABELS_OUT_PATH}")

    merged = ranking_features.merge(
        labels[["protein", "start", "end", "tcell_evidence", "mhc_ligand_evidence"]],
        on=["protein", "start", "end"], how="left"
    )
    assert len(merged) == len(ranking_features), "Row count changed during merge"
    assert merged["tcell_evidence"].isna().sum() == 0
    assert merged["mhc_ligand_evidence"].isna().sum() == 0

    merged.to_csv(RANKING_FEATURES_OUT_PATH, index=False)
    print(f"Updated {RANKING_FEATURES_OUT_PATH} with tcell_evidence and mhc_ligand_evidence columns.")


if __name__ == "__main__":
    main()