"""Measure Random Forest ranking stability across seeds using the saved split."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from random_forest import (
    RANKING_FEATURES_PATH,
    TARGET_LABEL_COLUMN,
    select_feature_columns,
    build_model_matrix,
    load_or_create_split,
)

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def run_one_seed(X_full, y_full, train_mask, seed):
    clf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )
    clf.fit(X_full.loc[train_mask], y_full.loc[train_mask])

    scores = clf.predict_proba(X_full)[:, 1]
    ranks = pd.Series(scores).rank(ascending=False, method="min")
    importances = pd.Series(clf.feature_importances_, index=X_full.columns)

    return ranks, importances


def main():
    df = pd.read_csv(RANKING_FEATURES_PATH).reset_index(drop=True)

    assert TARGET_LABEL_COLUMN in df.columns, (
        f"'{TARGET_LABEL_COLUMN}' not found in {RANKING_FEATURES_PATH} -- "
        f"run add_immunogenicity_label.py first."
    )

    df[TARGET_LABEL_COLUMN] = df[TARGET_LABEL_COLUMN].fillna(0).astype(int)

    labels_for_split = df[["peptide", "protein", TARGET_LABEL_COLUMN]]
    split_df = load_or_create_split(labels_for_split)
    split_lookup = {(r.peptide, r.protein): r.split for r in split_df.itertuples()}
    df["split"] = df.apply(lambda r: split_lookup.get((r["peptide"], r["protein"]), "unlabeled"), axis=1)

    holdout_mask = df["split"] == "holdout"
    train_mask = ~holdout_mask

    print(f"Holdout set: {holdout_mask.sum()} peptides")
    print(f"Training on: {train_mask.sum()} peptides")

    feature_cols = select_feature_columns(df)
    X_full = build_model_matrix(df, feature_cols)
    y_full = df[TARGET_LABEL_COLUMN]

    all_ranks = {}
    all_importances = {}

    for seed in SEEDS:
        print(f"Running seed {seed}...")
        ranks, importances = run_one_seed(X_full, y_full, train_mask, seed)
        all_ranks[seed] = ranks
        all_importances[seed] = importances

    rank_df = pd.DataFrame(all_ranks)

    print("\n--- Rank stability (Spearman correlation between seed pairs) ---")
    corr_matrix = rank_df.corr(method="spearman")
    print(corr_matrix.round(3))
    off_diag = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    print(f"\nMean pairwise Spearman correlation: {off_diag.mean():.3f} "
          f"(min {off_diag.min():.3f}, max {off_diag.max():.3f})")

    print("\n--- Feature importance stability ---")
    imp_df = pd.DataFrame(all_importances)
    imp_summary = pd.DataFrame({
        "mean": imp_df.mean(axis=1),
        "std": imp_df.std(axis=1),
    }).sort_values("mean", ascending=False)
    print(imp_summary)

    print("\n--- Held-out positives: rank stability across seeds ---")
    holdout_positive_idx = df[(df[TARGET_LABEL_COLUMN] == 1) & holdout_mask].index.tolist()
    for idx in holdout_positive_idx:
        peptide_ranks = rank_df.loc[idx]
        print(f"  {df.loc[idx, 'protein']} {df.loc[idx, 'peptide']}: "
              f"mean rank {peptide_ranks.mean():.0f} ± {peptide_ranks.std():.0f} "
              f"(range {peptide_ranks.min():.0f}-{peptide_ranks.max():.0f})")


if __name__ == "__main__":
    main()