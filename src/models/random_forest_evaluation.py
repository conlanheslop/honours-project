"""Evaluate Random Forest rankings with grouped cross-validation and diagnostics.

Reuses the persisted holdout split and writes performance, stability,
feature-importance, and candidate-ranking results under the RF evaluation folder.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from random_forest import (
    RANKING_FEATURES_PATH,
    TARGET_LABEL_COLUMN,
    build_model_matrix,
    load_or_create_split,
    select_feature_columns,
)

OUT_DIR = "results/random_forest/evaluation"
LINEAR_RANKINGS_PATH = "results/linear/rankings.csv"

RANDOM_SEED = 42
GROUP_BIN_SIZE = 30          # residues; windows in the same bin stay in one fold
FIXED_KS = [10, 50, 100]     # plus top 1% and top 5% of the pool

# Columns this script adds for its own bookkeeping. They must NEVER reach the
# model as features (baseline/linear scores would be leakage of other models).
HELPER_COLUMNS = ["baseline_score", "linear_score", "group"]

# Defaults (overridden by --fast)
N_SPLITS = 5
N_REPEATS = 3
CV_TREES = 300
FINAL_TREES = 500
PERM_TREES = 200
N_PERMUTATIONS = 20
IMPORTANCE_SEEDS = list(range(10))
LOSS_STEP = 10


def apply_fast_mode():
    global N_REPEATS, CV_TREES, FINAL_TREES, PERM_TREES, N_PERMUTATIONS
    global IMPORTANCE_SEEDS, LOSS_STEP
    N_REPEATS, CV_TREES, FINAL_TREES, PERM_TREES = 2, 60, 100, 50
    N_PERMUTATIONS, IMPORTANCE_SEEDS, LOSS_STEP = 4, [0, 1, 2], 25


def make_model(seed, n_estimators):
    # Same hyperparameters as random_forest.py
    return RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )


# ---------------------------------------------------------------- data

def load_data():
    df = pd.read_csv(RANKING_FEATURES_PATH).reset_index(drop=True)
    assert TARGET_LABEL_COLUMN in df.columns, (
        f"'{TARGET_LABEL_COLUMN}' not in {RANKING_FEATURES_PATH}; "
        f"run add_immunogenicity_label.py first.")
    df[TARGET_LABEL_COLUMN] = df[TARGET_LABEL_COLUMN].fillna(0).astype(int)

    split_df = load_or_create_split(df[["peptide", "protein", TARGET_LABEL_COLUMN]])
    lookup = {(r.peptide, r.protein): r.split for r in split_df.itertuples()}
    df["split"] = [lookup.get((p, q), "unlabeled")
                   for p, q in zip(df["peptide"], df["protein"])]

    n = len(df)
    # Baseline = MHC binding only (BA_rank, lower is better); global ordering.
    df["baseline_score"] = -df["BA_rank"]

    if os.path.exists(LINEAR_RANKINGS_PATH):
        lin = pd.read_csv(LINEAR_RANKINGS_PATH)[["protein", "start", "end", "linear_score"]]
        df = df.merge(lin, on=["protein", "start", "end"], how="left")
        assert len(df) == n, "Linear merge changed row count (duplicate keys?)"
    else:
        print(f"WARNING: {LINEAR_RANKINGS_PATH} not found; Linear excluded from comparison.")
        df["linear_score"] = np.nan

    df["group"] = df["protein"] + "_" + (df["start"] // GROUP_BIN_SIZE).astype(int).astype(str)
    return df


# ------------------------------------------------------------- metrics

def pool_ks(n):
    return ([(f"top{k}", k) for k in FIXED_KS]
            + [("top1pct", int(np.ceil(0.01 * n))), ("top5pct", int(np.ceil(0.05 * n)))])


def compute_metrics(y, score, seed=RANDOM_SEED):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    n, n_pos = len(y), int(y.sum())
    base_rate = n_pos / n
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(n), -score))   # ties broken randomly

    out = {"n_pool": n, "n_positives": n_pos,
           "auc": roc_auc_score(y, score),
           "avg_precision": average_precision_score(y, score)}
    for label, k in pool_ks(n):
        hits = y[order[:k]].sum()
        out[f"precision@{label}"] = hits / k
        out[f"recall@{label}"] = hits / n_pos
        out[f"enrichment@{label}"] = (hits / k) / base_rate
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    out["median_positive_rank_pct"] = np.median(ranks[y == 1]) / n * 100
    return out


# ------------------------------------------------------------------ CV

def run_grouped_cv(X, y, groups, pool_mask, holdout_idx, seed, n_estimators,
                   perm_importance=False):
    """One repeat of grouped, stratified CV over the non-holdout pool.

    Returns out-of-fold scores (NaN for holdout rows), per-fold scores for
    the holdout peptides, the held-out negatives of each fold, and
    (optionally) per-fold permutation importances.
    """
    pool_idx = np.where(pool_mask)[0]
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    oof = np.full(len(y), np.nan)
    holdout_scores, fold_negatives, perm_imps = [], [], []

    for fold, (tr, te) in enumerate(sgkf.split(pool_idx, y[pool_idx], groups[pool_idx])):
        tr_idx, te_idx = pool_idx[tr], pool_idx[te]
        clf = make_model(seed * 100 + fold, n_estimators)
        clf.fit(X.iloc[tr_idx], y[tr_idx])

        oof[te_idx] = clf.predict_proba(X.iloc[te_idx])[:, 1]
        if len(holdout_idx):
            holdout_scores.append(clf.predict_proba(X.iloc[holdout_idx])[:, 1])
        fold_negatives.append(te_idx[y[te_idx] == 0])

        if perm_importance and len(np.unique(y[te_idx])) == 2:
            r = permutation_importance(clf, X.iloc[te_idx], y[te_idx],
                                       scoring="roc_auc", n_repeats=5,
                                       random_state=seed, n_jobs=-1)
            perm_imps.append(r.importances_mean)

    return oof, holdout_scores, fold_negatives, perm_imps


def holdout_percentiles(neg_scores_per_fold, holdout_scores_per_fold, holdout_ids):
    """For each held-out positive: fraction of unseen negatives scoring >= it,
    averaged over folds. 0.02 means it outranks 98% of unseen negatives."""
    pcts = np.zeros((len(neg_scores_per_fold), len(holdout_ids)))
    for f, (negs, hs) in enumerate(zip(neg_scores_per_fold, holdout_scores_per_fold)):
        for j in range(len(holdout_ids)):
            pcts[f, j] = np.mean(negs >= hs[j])
    return pcts.mean(axis=0)


def run_repeats(X, y, groups, pool_mask, holdout_idx, n_repeats, n_estimators,
                perm_importance_first=False):
    results = []
    for rep in range(n_repeats):
        seed = RANDOM_SEED + rep
        results.append(run_grouped_cv(
            X, y, groups, pool_mask, holdout_idx, seed, n_estimators,
            perm_importance=(perm_importance_first and rep == 0)))
        print(f"    repeat {rep + 1}/{n_repeats} done")
    return results


# --------------------------------------------------------- loss curve

def oob_loss_curve(X, y, train_mask):
    """OOB vs training log-loss / average precision as trees are added."""
    Xt, yt = X[train_mask], y[train_mask]
    clf = RandomForestClassifier(
        n_estimators=LOSS_STEP, warm_start=True, oob_score=True,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
    eps, rows = 1e-15, []
    for n_trees in range(LOSS_STEP, FINAL_TREES + 1, LOSS_STEP):
        clf.set_params(n_estimators=n_trees)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xt, yt)
        oob = clf.oob_decision_function_[:, 1]
        ok = ~np.isnan(oob)
        train_p = clf.predict_proba(Xt)[:, 1]
        rows.append({
            "n_trees": n_trees,
            "oob_logloss": log_loss(yt[ok], np.clip(oob[ok], eps, 1 - eps), labels=[0, 1]),
            "train_logloss": log_loss(yt, np.clip(train_p, eps, 1 - eps), labels=[0, 1]),
            "oob_avg_precision": average_precision_score(yt[ok], oob[ok]),
            "train_avg_precision": average_precision_score(yt, train_p),
        })
    return pd.DataFrame(rows)


def plot_loss_curve(curve, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(curve["n_trees"], curve["train_logloss"], label="Training (in-sample)")
    axes[0].plot(curve["n_trees"], curve["oob_logloss"], label="Out-of-bag")
    axes[0].set_xlabel("Number of trees"); axes[0].set_ylabel("Log-loss")
    axes[0].set_title("Log-loss vs number of trees"); axes[0].legend()
    axes[1].plot(curve["n_trees"], curve["train_avg_precision"], label="Training (in-sample)")
    axes[1].plot(curve["n_trees"], curve["oob_avg_precision"], label="Out-of-bag")
    axes[1].set_xlabel("Number of trees"); axes[1].set_ylabel("Average precision")
    axes[1].set_title("Average precision vs number of trees"); axes[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ------------------------------------------------- permutation test

def permutation_test(X, y, groups, pool_mask, real_auc, real_ap):
    rng = np.random.default_rng(RANDOM_SEED)
    pool_idx = np.where(pool_mask)[0]
    null = []
    for i in range(N_PERMUTATIONS):
        y_perm = y.copy()
        y_perm[pool_idx] = rng.permutation(y[pool_idx])
        oof, _, _, _ = run_grouped_cv(X, y_perm, groups, pool_mask, [],
                                      seed=1000 + i, n_estimators=PERM_TREES)
        null.append({"auc": roc_auc_score(y_perm[pool_idx], oof[pool_idx]),
                     "avg_precision": average_precision_score(y_perm[pool_idx], oof[pool_idx])})
        print(f"    permutation {i + 1}/{N_PERMUTATIONS}")
    null = pd.DataFrame(null)
    p_auc = (1 + (null["auc"] >= real_auc).sum()) / (len(null) + 1)
    p_ap = (1 + (null["avg_precision"] >= real_ap).sum()) / (len(null) + 1)
    return null, p_auc, p_ap


def plot_permutation(null, real_auc, real_ap, p_auc, p_ap, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, real, p, name in [(axes[0], "auc", real_auc, p_auc, "AUC"),
                                   (axes[1], "avg_precision", real_ap, p_ap, "Average precision")]:
        ax.hist(null[col], bins=12, alpha=0.7, label="Shuffled labels")
        ax.axvline(real, color="red", label=f"Real labels (p = {p:.3f})")
        ax.set_xlabel(name); ax.set_ylabel("Count"); ax.legend()
        ax.set_title(f"Label-permutation test: {name}")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ------------------------------------------------ feature diagnostics

def diagnose_features(df, feature_cols):
    rows = []
    for c in feature_cols:
        n_unique = df[c].nunique(dropna=True)
        within = df.groupby("protein")[c].nunique(dropna=True).max()
        if n_unique <= 1:
            kind = "constant"
        elif within <= 1:
            kind = "protein-level"
        else:
            kind = "peptide-level"
        rows.append({"feature": c, "n_unique": n_unique,
                     "max_unique_within_protein": within, "kind": kind})
    return pd.DataFrame(rows)


def plot_importance(imp, path):
    imp = imp.sort_values("gini_mean")
    fig, axes = plt.subplots(1, 2, figsize=(11, max(4, 0.35 * len(imp))))
    axes[0].barh(imp.index, imp["gini_mean"], xerr=imp["gini_std"])
    axes[0].set_title("Gini importance (mean +/- sd over seeds)")
    if imp["perm_auc_drop_mean"].notna().any():
        axes[1].barh(imp.index, imp["perm_auc_drop_mean"], xerr=imp["perm_auc_drop_std"])
    axes[1].set_title("Permutation importance (AUC drop, out-of-fold)")
    axes[1].set_yticklabels([])
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# --------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="quick smoke test")
    args = parser.parse_args()
    if args.fast:
        apply_fast_mode()
        print("*** FAST MODE: reduced trees/repeats, numbers are NOT for reporting ***\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()

    # Select features from the ORIGINAL columns only, so this matches
    # random_forest.py exactly.
    original = df.drop(columns=HELPER_COLUMNS)
    feature_cols = select_feature_columns(original)
    X = build_model_matrix(original, feature_cols).astype(float)
    y = df[TARGET_LABEL_COLUMN].values.astype(int)
    groups = df["group"].values

    holdout_mask = (df["split"] == "holdout").values
    pool_mask = ~holdout_mask
    holdout_idx = np.where(holdout_mask & (y == 1))[0]
    pool_idx = np.where(pool_mask)[0]

    print(f"\nPool (CV) peptides: {pool_mask.sum()} | positives in pool: {y[pool_idx].sum()}")
    print(f"Holdout positives: {len(holdout_idx)} (never used in training)")

    # ---- feature diagnostics ------------------------------------------
    print("\n[1/6] Feature diagnostics")
    diag = diagnose_features(df, feature_cols)
    diag.to_csv(os.path.join(OUT_DIR, "feature_diagnostics.csv"), index=False)
    print(diag.to_string(index=False))
    flagged = diag[diag["kind"] != "peptide-level"]["feature"].tolist()
    if flagged:
        print(f"\n  NOTE: {flagged} are constant or protein-level. Protein-level features "
              f"act as a protein identifier, which partly undoes excluding 'protein' as a "
              f"feature. See the ablation below.")

    # ---- grouped CV ----------------------------------------------------
    print("\n[2/6] Grouped out-of-fold cross-validation")
    cv = run_repeats(X, y, groups, pool_mask, holdout_idx, N_REPEATS, CV_TREES,
                     perm_importance_first=True)

    rf_rows, hold_rows = [], []
    for rep, (oof, hold_scores, fold_negs, _) in enumerate(cv):
        m = compute_metrics(y[pool_idx], oof[pool_idx])
        m["repeat"] = rep
        rf_rows.append(m)
        if len(holdout_idx):
            neg_scores = [oof[n] for n in fold_negs]
            hold_rows.append(holdout_percentiles(neg_scores, hold_scores, holdout_idx))
    rf_by_repeat = pd.DataFrame(rf_rows)
    rf_by_repeat.to_csv(os.path.join(OUT_DIR, "cv_metrics_by_repeat.csv"), index=False)

    metric_cols = [c for c in rf_by_repeat.columns if c not in ("repeat", "n_pool", "n_positives")]
    summary = pd.DataFrame({"random_forest_mean": rf_by_repeat[metric_cols].mean(),
                            "random_forest_sd": rf_by_repeat[metric_cols].std()})
    for name, col in [("linear", "linear_score"), ("baseline", "baseline_score")]:
        if df[col].notna().all():
            m = compute_metrics(y[pool_idx], df[col].values[pool_idx])
            summary[name] = pd.Series({k: m[k] for k in metric_cols})
    summary.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"))
    print("\n  Out-of-fold performance (pool of non-holdout peptides):")
    print(summary.round(4).to_string())

    real_auc = rf_by_repeat["auc"].mean()
    real_ap = rf_by_repeat["avg_precision"].mean()

    # ---- holdout positives --------------------------------------------
    if len(holdout_idx):
        print("\n  Held-out positives (fraction of unseen negatives that outrank them; "
              "lower = better):")
        hold_df = pd.DataFrame({
            "protein": df.loc[holdout_idx, "protein"].values,
            "peptide": df.loc[holdout_idx, "peptide"].values,
            "rf_pct_outranking": np.mean(hold_rows, axis=0),
        })
        for name, col in [("linear", "linear_score"), ("baseline", "baseline_score")]:
            if df[col].notna().all():
                fixed = [df[col].values[holdout_idx]] * N_SPLITS
                reps = []
                for _, _, fold_negs, _ in cv:
                    reps.append(holdout_percentiles([df[col].values[n] for n in fold_negs],
                                                    fixed, holdout_idx))
                hold_df[f"{name}_pct_outranking"] = np.mean(reps, axis=0)
        hold_df.to_csv(os.path.join(OUT_DIR, "holdout_percentiles.csv"), index=False)
        print(hold_df.round(4).to_string(index=False))
        print("  (Only a handful of positives: treat as illustrative, not as a headline number.)")

    # ---- loss curve ----------------------------------------------------
    print("\n[3/6] OOB loss curve")
    curve = oob_loss_curve(X, y, pool_mask)
    curve.to_csv(os.path.join(OUT_DIR, "loss_curve.csv"), index=False)
    plot_loss_curve(curve, os.path.join(OUT_DIR, "loss_curve.png"))
    last = curve.iloc[-1]
    print(f"  At {int(last.n_trees)} trees: train log-loss {last.train_logloss:.3f} vs "
          f"OOB {last.oob_logloss:.3f}; train AP {last.train_avg_precision:.3f} vs "
          f"OOB AP {last.oob_avg_precision:.3f}")
    print("  Note: class_weight='balanced' inflates predicted probabilities, so absolute "
          "log-loss is not calibrated; use the curve shape (plateau) and the AP panel.")

    # ---- permutation test ----------------------------------------------
    print("\n[4/6] Label-permutation test")
    null, p_auc, p_ap = permutation_test(X, y, groups, pool_mask, real_auc, real_ap)
    null.to_csv(os.path.join(OUT_DIR, "permutation_null.csv"), index=False)
    plot_permutation(null, real_auc, real_ap, p_auc, p_ap,
                     os.path.join(OUT_DIR, "permutation_test.png"))
    print(f"  Real AUC {real_auc:.3f} vs shuffled mean {null['auc'].mean():.3f} "
          f"(p = {p_auc:.3f}); real AP {real_ap:.3f} vs shuffled mean "
          f"{null['avg_precision'].mean():.3f} (p = {p_ap:.3f})")

    # ---- feature importance --------------------------------------------
    print("\n[5/6] Feature importance")
    gini = []
    for s in IMPORTANCE_SEEDS:
        clf = make_model(s, FINAL_TREES)
        clf.fit(X[pool_mask], y[pool_mask])
        gini.append(pd.Series(clf.feature_importances_, index=X.columns))
    gini = pd.DataFrame(gini)
    imp = pd.DataFrame({"gini_mean": gini.mean(), "gini_std": gini.std()})

    perm_list = cv[0][3]
    if perm_list:
        perm = pd.DataFrame(perm_list, columns=X.columns)
        imp["perm_auc_drop_mean"] = perm.mean()
        imp["perm_auc_drop_std"] = perm.std()
    else:
        imp["perm_auc_drop_mean"] = np.nan
        imp["perm_auc_drop_std"] = np.nan
    imp = imp.sort_values("gini_mean", ascending=False)
    imp.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"))
    plot_importance(imp, os.path.join(OUT_DIR, "feature_importance.png"))
    print(imp.round(4).to_string())

    # ---- ablation ------------------------------------------------------
    if flagged:
        print(f"\n  Ablation: removing {flagged}")
        keep_cols = [c for c in X.columns
                     if not any(c == f or c.startswith(f + "_") for f in flagged)]
        abl = run_repeats(X[keep_cols], y, groups, pool_mask, holdout_idx,
                          N_REPEATS, CV_TREES)
        abl_m = pd.DataFrame([compute_metrics(y[pool_idx], r[0][pool_idx]) for r in abl])
        ablation = pd.DataFrame({
            "all_features": rf_by_repeat[["auc", "avg_precision"]].mean(),
            "without_flagged": abl_m[["auc", "avg_precision"]].mean()})
        ablation.to_csv(os.path.join(OUT_DIR, "ablation.csv"))
        print(ablation.round(4).to_string())

    # ---- model agreement + candidates ----------------------------------
    print("\n[6/6] Model agreement and candidate list")
    oof_stack = np.vstack([r[0] for r in cv])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # holdout rows are all-NaN here; filled below
        rf_score = np.nanmean(oof_stack, axis=0)
    # The holdout set contains only the held-out positives. None of the fold
    # models trained on them, so average the fold models' predictions.
    for j, i in enumerate(holdout_idx):
        rf_score[i] = np.mean([hs[j] for r in cv for hs in r[1]])
    df["rf_oof_score"] = rf_score
    df["rf_oof_rank"] = df["rf_oof_score"].rank(ascending=False, method="min").astype(int)
    # Full out-of-fold scores, so other models' evaluations can compare against RF.
    df[["protein", "peptide", "start", "end", "rf_oof_score", "rf_oof_rank"]].to_csv(
        os.path.join(OUT_DIR, "oof_scores.csv"), index=False)

    score_cols = {"random_forest": "rf_oof_score", "baseline": "baseline_score"}
    if df["linear_score"].notna().all():
        score_cols["linear"] = "linear_score"
    names = list(score_cols)
    agree = []
    for a in names:
        for b in names:
            if a >= b:
                continue
            row = {"model_a": a, "model_b": b,
                   "spearman": spearmanr(df[score_cols[a]], df[score_cols[b]])[0]}
            for k in (50, 100, 250):
                ta = set(df.nlargest(k, score_cols[a]).index)
                tb = set(df.nlargest(k, score_cols[b]).index)
                row[f"top{k}_overlap"] = len(ta & tb)
                row[f"top{k}_jaccard"] = len(ta & tb) / len(ta | tb)
            agree.append(row)
    agree = pd.DataFrame(agree)
    agree.to_csv(os.path.join(OUT_DIR, "model_agreement.csv"), index=False)
    print(agree.round(3).to_string(index=False))

    keep = [c for c in ["protein", "peptide", "start", "end", "length", "rf_oof_score",
                        "rf_oof_rank", "BA_rank", "cleavage_score",
                        "mhcflurry_processing_score", "expression_tpm",
                        "tcell_evidence", "mhc_ligand_evidence", "split"] if c in df.columns]
    top = df.sort_values("rf_oof_rank").head(100)[keep].copy()
    ev_cols = [c for c in ("tcell_evidence", "mhc_ligand_evidence") if c in top.columns]
    top["known_evidence"] = top[ev_cols].max(axis=1).astype(bool)
    top.to_csv(os.path.join(OUT_DIR, "top_candidates.csv"), index=False)

    print(f"\n  Top 100 by out-of-fold RF score: {int(top['known_evidence'].sum())} have known "
          f"evidence, {int((~top['known_evidence']).sum())} are novel candidates.")
    print("  By protein:", top["protein"].value_counts().to_dict())
    if "length" in top.columns:
        print("  By length: ", top["length"].value_counts().sort_index().to_dict())
    print(f"\nAll outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()