"""Merge peptide hydrophobicity and molecular weight into the feature table."""

import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path("data/processed")
MAIN_FEATURES_FILE = DATA_DIR / "ranking_features.csv"
HYDRO_FILE = DATA_DIR / "hydrophobicity.csv"
MW_FILE = DATA_DIR / "molecular_weight.csv"
OUTPUT_FILE = DATA_DIR / "ranking_features.csv"

def main():
    print(f"Loading main feature dataset from {MAIN_FEATURES_FILE}...")
    main_df = pd.read_csv(MAIN_FEATURES_FILE)
    
    print(f"Loading hydrophobicity from {HYDRO_FILE}...")
    hydro_df = pd.read_csv(HYDRO_FILE)
    
    print(f"Loading molecular weight from {MW_FILE}...")
    mw_df = pd.read_csv(MW_FILE)
    
    # Merge hydrophobicity
    if "hydrophobicity" in main_df.columns:
        main_df = main_df.drop(columns=["hydrophobicity"])
    main_df = main_df.merge(hydro_df, on="peptide", how="left")
    
    # Merge molecular weight
    if "molecular_weight" in main_df.columns:
        main_df = main_df.drop(columns=["molecular_weight"])
    main_df = main_df.merge(mw_df, on="peptide", how="left")
    
    # Validation checks
    missing_hydro = main_df["hydrophobicity"].isna().sum()
    missing_mw = main_df["molecular_weight"].isna().sum()
    
    if missing_hydro > 0 or missing_mw > 0:
        print(f"Warning: Missing values detected! (Hydrophobicity: {missing_hydro}, MW: {missing_mw})")
    else:
        print("Merge successful with zero missing values.")
        
    main_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Updated feature table saved to {OUTPUT_FILE} (Total rows: {len(main_df)})")

if __name__ == "__main__":
    main()