"""Calculate GRAVY hydrophobicity scores for unique peptides."""

import pandas as pd
from pathlib import Path
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Paths
DATA_DIR = Path("data/processed")
INPUT_FILE = DATA_DIR / "ranking_features.csv"
OUTPUT_FILE = DATA_DIR / "hydrophobicity.csv"

def calculate_gravy(sequence: str) -> float:
    """Calculate GRAVY score using Kyte-Doolittle hydropathy scale."""
    try:
        analysed_seq = ProteinAnalysis(sequence)
        return analysed_seq.gravy()
    except Exception as e:
        print(f"Error calculating GRAVY for sequence {sequence}: {e}")
        return None

def main():
    print(f"Loading peptides from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    if "peptide" not in df.columns:
        raise KeyError("Expected column 'peptide' not found in input CSV.")
    
    # Extract unique peptides to avoid redundant calculations
    unique_peptides = df["peptide"].unique()
    print(f"Calculating GRAVY hydrophobicity for {len(unique_peptides)} unique peptides...")
    
    hydro_data = []
    for seq in unique_peptides:
        gravy_score = calculate_gravy(seq)
        hydro_data.append({"peptide": seq, "hydrophobicity": gravy_score})
        
    hydro_df = pd.DataFrame(hydro_data)
    hydro_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved hydrophobicity scores to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()