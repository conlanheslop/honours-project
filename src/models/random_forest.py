"""Train a Random Forest on T-cell evidence and rank candidate peptides.

Persists a positive holdout split, excludes holdout cases from training, and
writes scores for the candidate set.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

RANKING_FEATURES_PATH = "data/processed/ranking_features.csv"
SPLIT_PATH = "data/processed/evaluation_split.csv"
OUT_DIR = "results/random_forest"
OUT_PATH = os.path.join(OUT_DIR, "rankings.csv")

TARGET_LABEL_COLUMN = "tcell_evidence"  # primary supervised training target

HOLDOUT_SIZE = 3
RANDOM_SEED = 42

NON_FEATURE_COLUMNS = {
    "peptide", "protein", "allele", "seq_num", "start", "end",
    "BA_core", "BA_icore", "EL_core", "EL_icore",
    "tcell_evidence", "mhc_ligand_evidence", "split",
}

# With only 4 proteins, RF could otherwise learn "favour CCNA1" directly
# from protein identity rather than the biological features. Off by default.
INCLUDE_PROTEIN_AS_FEATURE = False


def load_or_create_split(labels_df):
    positives = labels_df[labels_df[TARGET_LABEL_COLUMN] == 1].copy()

    if os.path.exists(SPLIT_PATH):
        split_df = pd.read_csv(SPLIT_PATH)
        known = set(zip(split_df["peptide"], split_df["protein"]))
        new_positives = positives[
            ~positives.apply(lambda r: (r["peptide"], r["protein"]) in known, axis=1)
        ]
        if len(new_positives) > 0:
            print(f"Found {len(new_positives)} new positive(s) not in the split file. "
                  f"Adding to 'dev' by default.")
            new_rows = new_positives[["peptide", "protein"]].copy()
            new_rows["split"] = "dev"
            split_df = pd.concat([split_df, new_rows], ignore_index=True)
            split_df.to_csv(SPLIT_PATH, index=False)
        return split_df

    positives = positives.sample(frac=1.0, random_state=RANDOM_SEED)
    n_holdout = min(HOLDOUT_SIZE, max(0, len(positives) - 1))
    holdout = positives.iloc[:n_holdout]
    dev = positives.iloc[n_holdout:]

    split_df = pd.concat([
        holdout[["peptide", "protein"]].assign(split="holdout"),
        dev[["peptide", "protein"]].assign(split="dev"),
    ], ignore_index=True)

    os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
    split_df.to_csv(SPLIT_PATH, index=False)
    print(f"Created new evaluation split: {n_holdout} holdout, {len(dev)} dev positives.")
    return split_df


def select_feature_columns(df):
    exclude = set(NON_FEATURE_COLUMNS)
    if not INCLUDE_PROTEIN_AS_FEATURE:
        exclude.add("protein")
    feature_cols = [c for c in df.columns if c not in exclude]
    print(f"Using {len(feature_cols)} feature columns: {feature_cols}")
    return feature_cols


def build_model_matrix(df, feature_cols):
    X = df[feature_cols].copy()
    if INCLUDE_PROTEIN_AS_FEATURE and "protein" in df.columns:
        X["protein"] = df["protein"]
    X = pd.get_dummies(X, drop_first=False)
    return X


def leave_one_positive_out_cv(dev_positive_idx, X, y, holdout_mask):
    if len(dev_positive_idx) < 2:
        print("\nSkipping CV diagnostic: fewer than 2 dev positives available.")
        return

    print("\nRunning leave-one-positive-out cross-validation (diagnostic)...")
    ranks = []
    for pep_idx in dev_positive_idx:
        train_mask = (X.index != pep_idx) & (~holdout_mask)
        score_mask = ~holdout_mask

        clf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1,
        )
        clf.fit(X.loc[train_mask], y.loc[train_mask])

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
        f"run add_immunogenicity_label.py first."
    )

    df[TARGET_LABEL_COLUMN] = df[TARGET_LABEL_COLUMN].fillna(0).astype(int)

    labels_for_split = df[["peptide", "protein", TARGET_LABEL_COLUMN]]
    split_df = load_or_create_split(labels_for_split)
    split_lookup = {(r.peptide, r.protein): r.split for r in split_df.itertuples()}
    df["split"] = df.apply(lambda r: split_lookup.get((r["peptide"], r["protein"]), "unlabeled"), axis=1)

    holdout_mask = df["split"] == "holdout"
    print(f"\nHoldout set: {holdout_mask.sum()} peptides (permanently excluded from training)")
    print(f"Positives in dev pool: {((df[TARGET_LABEL_COLUMN] == 1) & (~holdout_mask)).sum()}")
    print(f"Positives in holdout:  {((df[TARGET_LABEL_COLUMN] == 1) & (holdout_mask)).sum()}")

    feature_cols = select_feature_columns(df)
    X_full = build_model_matrix(df, feature_cols)
    y_full = df[TARGET_LABEL_COLUMN]

    dev_positive_idx = df[(df[TARGET_LABEL_COLUMN] == 1) & (~holdout_mask)].index.tolist()
    leave_one_positive_out_cv(dev_positive_idx, X_full, y_full, holdout_mask)

    train_mask = ~holdout_mask
    clf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    clf.fit(X_full.loc[train_mask], y_full.loc[train_mask])

    df["rf_score"] = clf.predict_proba(X_full)[:, 1]
    df["rf_rank"] = df["rf_score"].rank(ascending=False, method="min").astype(int)
    df["rf_percentile"] = df["rf_score"].rank(pct=True, ascending=True) * 100

    df = df.sort_values("rf_score", ascending=False).reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_PATH}")

    importances = pd.Series(clf.feature_importances_, index=X_full.columns).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances.to_string())

    print("\nTop 10 peptides overall:")
    print(df[["protein", "peptide", "rf_score", "rf_rank", "split"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()