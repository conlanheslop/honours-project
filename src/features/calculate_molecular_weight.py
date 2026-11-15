"""Calculate molecular weights for unique peptides."""

import pandas as pd
from pathlib import Path
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Paths
DATA_DIR = Path("data/processed")
INPUT_FILE = DATA_DIR / "ranking_features.csv"
OUTPUT_FILE = DATA_DIR / "molecular_weight.csv"

def calculate_mw(sequence: str) -> float:
    """Calculate molecular weight in Daltons."""
    try:
        analysed_seq = ProteinAnalysis(sequence)
        return analysed_seq.molecular_weight()
    except Exception as e:
        print(f"Error calculating molecular weight for sequence {sequence}: {e}")
        return None

def main():
    print(f"Loading peptides from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    if "peptide" not in df.columns:
        raise KeyError("Expected column 'peptide' not found in input CSV.")
        
    unique_peptides = df["peptide"].unique()
    print(f"Calculating molecular weight for {len(unique_peptides)} unique peptides...")
    
    mw_data = []
    for seq in unique_peptides:
        mw = calculate_mw(seq)
        mw_data.append({"peptide": seq, "molecular_weight": mw})
        
    mw_df = pd.DataFrame(mw_data)
    mw_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved molecular weights to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()