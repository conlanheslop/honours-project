"""Train an XGBoost model on T-cell evidence and rank candidate peptides.

Excludes persisted holdout positives from training and writes scores for the
candidate set.
"""

import os

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from random_forest import (
    RANKING_FEATURES_PATH,
    TARGET_LABEL_COLUMN,
    build_model_matrix,
    load_or_create_split,
    select_feature_columns,
)

OUT_DIR = "results/xgboost"
OUT_PATH = os.path.join(OUT_DIR, "rankings.csv")
RANDOM_SEED = 42


def make_model(y, n_estimators=500):
    """Create a regularized classifier with imbalance weighting for this fold."""
    n_positive = int(y.sum())
    n_negative = len(y) - n_positive
    if n_positive == 0:
        raise ValueError("XGBoost training data must contain at least one positive label.")

    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=2,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=n_negative / n_positive,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
        importance_type="gain",
    )


def leave_one_positive_out_cv(dev_positive_idx, X, y, holdout_mask):
    if len(dev_positive_idx) < 2:
        print("\nSkipping CV diagnostic: fewer than 2 dev positives available.")
        return

    print("\nRunning leave-one-positive-out cross-validation (diagnostic)...")
    ranks = []
    for pep_idx in dev_positive_idx:
        train_mask = (X.index != pep_idx) & (~holdout_mask)
        score_mask = ~holdout_mask

        y_train = y.loc[train_mask]
        clf = make_model(y_train, n_estimators=300)
        clf.fit(X.loc[train_mask], y_train)

        scores = pd.Series(
            clf.predict_proba(X.loc[score_mask])[:, 1], index=X.loc[score_mask].index
        )
        rank = scores.rank(ascending=False, method="min").loc[pep_idx]
        ranks.append(rank)
        print(f"  held-out positive idx {pep_idx}: rank {int(rank)} of {len(scores)}")

    print(f"\nMean held-out rank: {np.mean(ranks):.1f} "
          f"(lower is better; {len(scores)} candidates scored per fold)")


def main():
    df = pd.read_csv(RANKING_FEATURES_PATH).reset_index(drop=True)

    assert TARGET_LABEL_COLUMN in df.columns, (
        f"'{TARGET_LABEL_COLUMN}' not found in {RANKING_FEATURES_PATH} -- "
        "run add_immunogenicity_label.py first."
    )
    df[TARGET_LABEL_COLUMN] = df[TARGET_LABEL_COLUMN].fillna(0).astype(int)

    split_df = load_or_create_split(df[["peptide", "protein", TARGET_LABEL_COLUMN]])
    split_lookup = {(row.peptide, row.protein): row.split for row in split_df.itertuples()}
    df["split"] = [
        split_lookup.get((peptide, protein), "unlabeled")
        for peptide, protein in zip(df["peptide"], df["protein"])
    ]

    holdout_mask = df["split"] == "holdout"
    print(f"\nHoldout set: {holdout_mask.sum()} peptides (permanently excluded from training)")
    print(f"Positives in dev pool: {((df[TARGET_LABEL_COLUMN] == 1) & (~holdout_mask)).sum()}")
    print(f"Positives in holdout:  {((df[TARGET_LABEL_COLUMN] == 1) & holdout_mask).sum()}")

    feature_cols = select_feature_columns(df)
    X_full = build_model_matrix(df, feature_cols)
    y_full = df[TARGET_LABEL_COLUMN]

    dev_positive_idx = df[(y_full == 1) & (~holdout_mask)].index.tolist()
    leave_one_positive_out_cv(dev_positive_idx, X_full, y_full, holdout_mask)

    train_mask = ~holdout_mask
    y_train = y_full.loc[train_mask]
    clf = make_model(y_train)
    clf.fit(X_full.loc[train_mask], y_train)

    df["xgb_score"] = clf.predict_proba(X_full)[:, 1]
    df["xgb_rank"] = df["xgb_score"].rank(ascending=False, method="min").astype(int)
    df["xgb_percentile"] = df["xgb_score"].rank(pct=True, ascending=True) * 100
    df = df.sort_values("xgb_score", ascending=False).reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_PATH}")

    importances = pd.Series(
        clf.feature_importances_, index=X_full.columns
    ).sort_values(ascending=False)
    print("\nFeature importances (gain):")
    print(importances.to_string())

    print("\nTop 10 peptides overall:")
    print(df[["protein", "peptide", "xgb_score", "xgb_rank", "split"]]
          .head(10).to_string(index=False))


if __name__ == "__main__":
    main()
