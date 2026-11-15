# Computational Peptide Prioritisation for T-Cell Receptor Discovery

This project builds a ranked list of candidate peptides from four target
proteins: PRAME, WT1, CCNA1, and HOXA9. It combines MHC-binding and
proteasomal-processing predictions with expression, mutation, and IEDB
T-cell evidence, then ranks candidates using linear, Random Forest, or
XGBoost models.

## Setup

Use a Python environment compatible with the versions in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

MHCflurry is used by `run_mhcflurry.py` but is not listed in
`requirements.txt`; install and configure it separately, including its
prediction models, before running that step. The feature retrieval scripts
also need network access to the TCGA Genomic Data Commons and cBioPortal APIs.

## Input data

The pipeline expects these inputs in the paths shown:

| Input | Purpose |
| --- | --- |
| `data/processed/cleaned_mhc.csv` | Candidate peptide table with protein, allele, and peptide positions |
| `data/raw/netchop/netchop_output.txt` | Precomputed NetChop output for the target proteins |
| `data/raw/iedb/tcell_table_export.csv` | IEDB T-cell assay evidence |
| `data/raw/iedb/mhc_ligand_table_export.csv` | IEDB MHC-ligand assay evidence |

`cleaned_mhc.csv` and the NetChop output are inputs to this repository's
feature scripts; those scripts do not create the cleaned candidate table or
run NetChop. The IEDB exports must contain the columns expected by
`add_immunogenicity_label.py`.

## Feature pipeline

Run commands from the repository root. Each command reads and writes the
project's `data/` paths.

1. Add cleavage scores, prepare MHCflurry inputs, predict processing scores,
   and merge predictions:

   ```bash
   python src/features/add_cleavage_score.py
   python src/features/prepare_mhcflurry_input.py
   python src/features/run_mhcflurry.py
   python src/features/add_processing_score.py
   ```

2. Retrieve protein expression and mutation data:

   ```bash
   python src/features/fetch_tcga_expression.py
   python src/features/get_mutation_status.py
   ```

3. Build the ranking feature table and add further features and evidence:

   ```bash
   python src/features/add_expression_score.py
   python src/features/calculate_hydrophobicity.py
   python src/features/calculate_molecular_weight.py
   python src/features/add_physiochemical_features.py
   python src/features/add_mutation_status.py
   python src/features/add_hla_frequency.py
   python src/features/add_immunogenicity_label.py
   ```

The resulting `data/processed/ranking_features.csv` is used by the ranking
models. The model scripts expect the relevant feature columns and evidence
labels to have been generated first.

## Rank candidates

Run one or more ranking models:

```bash
python src/models/linear.py
python src/models/random_forest.py
python src/models/xgboost_model.py
```

Ranked tables are written to `results/linear/rankings.csv`,
`results/random_forest/rankings.csv`, and `results/xgboost/rankings.csv`.
These outputs are generated locally and excluded from Git.

The linear model combines MHC binding, cleavage, and expression using
configured weights. The supervised models use `tcell_evidence` as their
training target; candidates without recorded positive evidence are treated as
unlabeled, not experimentally confirmed negatives. A persisted positive
holdout split is stored in `data/processed/evaluation_split.csv` and excluded
from model fitting.

## Model evaluation

Run the evaluation scripts after generating the corresponding rankings:

```bash
python src/models/random_forest_evaluation.py
python src/models/xgboost_evaluation.py
python src/models/random_forest_stability.py
python src/models/feature_ablation.py
```

The Random Forest and XGBoost evaluation scripts accept `--fast` for a
reduced diagnostic run. Run Random Forest evaluation before XGBoost
evaluation if you want XGBoost's side-by-side comparison with the Random
Forest results. Evaluation tables and plots are saved under each model's
`results/<model>/evaluation/` directory.

The feature-ablation script reuses the grouped evaluation folds and the
persisted holdout split without using the holdout positives for feature
selection. It writes repeat-level baseline and ablation metrics, plus
heatmaps, under `results/feature_ablation/`. `--fast` is available for a
non-reportable smoke test; the default run uses all three evaluation repeats.
It also writes `all_model_feature_effect_heatmap.png`, a four-model summary
for the binding-only baseline, Linear, Random Forest, and XGBoost.
`feature_effect_summary.csv` contains the metric changes behind that plot.
The summary averages the bounded AUC, average-precision, and precision@100
changes in percentage points (precision@100 represents the top-100 metrics
without counting its equivalent recall and enrichment measures again);
positive values mean performance improved when the feature was removed.
Linear and baseline ablations are deterministic score removals, while the
supervised models use their repeated grouped-CV ablations. The direct-model
scores are calculated on the non-holdout pool only. Use `--summary-only` to
rebuild this plot from existing ablation CSVs without retraining the
supervised models.

## Project layout

```text
src/features/   Data retrieval and feature-generation scripts
src/models/     Linear, Random Forest, and XGBoost ranking and evaluation
data/raw/       Source exports and external prediction inputs
data/processed/ Intermediate tables and the assembled ranking features
results/        Generated rankings, evaluation tables, and plots
tools/          Bundled external-tool data
```

The `.gitignore` excludes the local Python environment, generated results,
MHCflurry intermediates, and the large downloaded TCGA archive. Supply any
ignored input files locally before running the relevant pipeline steps.