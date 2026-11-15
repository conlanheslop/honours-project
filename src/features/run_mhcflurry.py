"""Predict MHCflurry processing scores for the prepared peptide inputs."""

import pandas as pd
from mhcflurry import Class1PresentationPredictor

predictor = Class1PresentationPredictor.load()

df = pd.read_csv("data/raw/mhcflurry/mhcflurry_input.csv")

df["n_flank"] = df["n_flank"].fillna("").astype(str)
df["c_flank"] = df["c_flank"].fillna("").astype(str)

# Processing score is allele-independent, so call it directly rather than
# going through predict(), which expects a genotype (<=6 alleles) for
# combined affinity+processing scoring.
processing_scores = predictor.predict_processing(
    peptides=df["peptide"].tolist(),
    n_flanks=df["n_flank"].tolist(),
    c_flanks=df["c_flank"].tolist(),
)

df["mhcflurry_processing_score"] = processing_scores
df.to_csv("data/raw/mhcflurry/mhcflurry_output.csv", index=False)
print(f"Wrote {len(df)} predictions to data/raw/mhcflurry/mhcflurry_output.csv")