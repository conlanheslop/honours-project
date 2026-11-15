"""Add the configured HLA allele frequency to the ranking feature table."""

import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path("data/processed")
INPUT_FILE = DATA_DIR / "ranking_features.csv"
OUTPUT_FILE = DATA_DIR / "ranking_features.csv"

# Parameters
ALLELE_FREQ = 0.23758  # Global frequency for HLA-A*02:01 from AFND

def main():
    print(f"Loading feature table from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Add or update hla_frequency column
    df["hla_frequency"] = ALLELE_FREQ
    
    # Save updated feature table
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully added 'hla_frequency' ({ALLELE_FREQ}) to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()