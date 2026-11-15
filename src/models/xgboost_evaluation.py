"""Evaluate XGBoost rankings with grouped cross-validation and diagnostics.

Uses the shared evaluation split and writes performance, robustness,
feature-importance, and model-comparison results under the XGBoost evaluation folder.
"""

import argparse
import inspect
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from random_forest import build_model_matrix, select_feature_columns, TARGET_LABEL_COLUMN
from random_forest_evaluation import (
    HELPER_COLUMNS,
    RANDOM_SEED,
    compute_metrics,
    diagnose_features,
    holdout_percentiles,
    load_data,
)
from xgboost_model import make_model

OUT_DIR = "results/xgboost/evaluation"
RF_EVAL_DIR = "results/random_forest/evaluation"

N_SPLITS = 5                      # must match random_forest_evaluation.py
N_REPEATS = 3                     # must match random_forest_evaluation.py
CONFIGURED_ROUNDS = inspect.signature(make_model).parameters["n_estimators"].default
CURVE_ROUNDS = 2 * CONFIGURED_ROUNDS
N_ROUNDS = CONFIGURED_ROUNDS      # rounds used for all reported performance
N_PERMUTATIONS = 20
IMPORTANCE_SEEDS = list(range(10))
SMOOTH_WINDOW = 10


def apply_fast_mode():
    global N_REPEATS, CURVE_ROUNDS, N_ROUNDS, N_PERMUTATIONS, IMPORTANCE_SEEDS
    N_REPEATS, CURVE_ROUNDS, N_PERMUTATIONS = 2, 120, 4
    N_ROUNDS, IMPORTANCE_SEEDS = 80, [0, 1, 2]


def new_model(y_train, seed, n_estimators, spw_power=None):
    clf = make_model(y_train, n_estimators=n_estimators)
    clf.set_params(random_state=seed, eval_metric=["logloss", "aucpr"])
    if spw_power is not None:
        ratio = (y_train == 0).sum() / max(1, (y_train == 1).sum())
        clf.set_params(scale_pos_weight=float(ratio) ** spw_power)
    return clf


# ------------------------------------------------------------------ CV

def run_cv(X, y, groups, pool_mask, holdout_idx, seed,
           record_curves=False, spw_power=None):
    """One repeat of grouped, stratified CV. Folds are identical to the
    Random Forest evaluation (same seed formula, same splitter)."""
    pool_idx = np.where(pool_mask)[0]
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    n_fit = CURVE_ROUNDS if record_curves else N_ROUNDS
    use = (0, N_ROUNDS)

    oof = np.full(len(y), np.nan)
    holdout_scores, fold_negatives, curves = [], [], []

    for fold, (tr, te) in enumerate(sgkf.split(pool_idx, y[pool_idx], groups[pool_idx])):
        tr_idx, te_idx = pool_idx[tr], pool_idx[te]
        clf = new_model(y[tr_idx], seed * 100 + fold, n_fit, spw_power)

        if record_curves and y[te_idx].sum() > 0:
            clf.fit(X.iloc[tr_idx], y[tr_idx], verbose=False,
                    eval_set=[(X.iloc[tr_idx], y[tr_idx]), (X.iloc[te_idx], y[te_idx])])
            res = clf.evals_result()
            curves.append(pd.DataFrame({
                "round": np.arange(1, n_fit + 1),
                "train_logloss": res["validation_0"]["logloss"],
                "val_logloss": res["validation_1"]["logloss"],
                "train_aucpr": res["validation_0"]["aucpr"],
                "val_aucpr": res["validation_1"]["aucpr"]}))
        else:
            clf.fit(X.iloc[tr_idx], y[tr_idx], verbose=False)

        oof[te_idx] = clf.predict_proba(X.iloc[te_idx], iteration_range=use)[:, 1]
        if len(holdout_idx):
            holdout_scores.append(
                clf.predict_proba(X.iloc[holdout_idx], iteration_range=use)[:, 1])
        fold_negatives.append(te_idx[y[te_idx] == 0])

    return {"oof": oof, "holdout_scores": holdout_scores,
            "fold_negatives": fold_negatives, "curves": curves}


def fold_permutation_importance(X, y, groups, pool_mask, seed):
    pool_idx = np.where(pool_mask)[0]
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    out = []
    for fold, (tr, te) in enumerate(sgkf.split(pool_idx, y[pool_idx], groups[pool_idx])):
        tr_idx, te_idx = pool_idx[tr], pool_idx[te]
        if len(np.unique(y[te_idx])) < 2:
            continue
        clf = new_model(y[tr_idx], seed * 100 + fold, N_ROUNDS)
        clf.fit(X.iloc[tr_idx], y[tr_idx], verbose=False)
        r = permutation_importance(clf, X.iloc[te_idx], y[te_idx], scoring="roc_auc",
                                   n_repeats=5, random_state=seed)
        out.append(r.importances_mean)
    return out


# ------------------------------------------------------------- plots

def plot_loss_curve(curves, path):
    g = curves.groupby("round")
    mean, sd = g.mean(), g.std()
    best_round = int(mean["val_aucpr"].rolling(SMOOTH_WINDOW, center=True,
                                               min_periods=1).mean().idxmax())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, key, title in [(axes[0], "logloss", "Log-loss per boosting round"),
                           (axes[1], "aucpr", "AUCPR per boosting round")]:
        for kind, label in [("train", "Training"), ("val", "Validation (held-out fold)")]:
            m, s = mean[f"{kind}_{key}"], sd[f"{kind}_{key}"]
            ax.plot(m.index, m, label=label)
            ax.fill_between(m.index, m - s, m + s, alpha=0.2)
        ax.axvline(N_ROUNDS, color="grey", ls="--", label=f"Configured rounds ({N_ROUNDS})")
        ax.axvline(best_round, color="red", ls=":", label=f"Best validation AUCPR ({best_round})")
        ax.set_xlabel("Boosting round"); ax.set_title(title); ax.legend(fontsize=8)
    axes[0].set_ylabel("Log-loss (unweighted)"); axes[1].set_ylabel("AUCPR")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return best_round


def plot_permutation(null, real_auc, real_ap, p_auc, p_ap, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, real, p, name in [(axes[0], "auc", real_auc, p_auc, "AUC"),
                                   (axes[1], "avg_precision", real_ap, p_ap, "Average precision")]:
        ax.hist(null[col], bins=12, alpha=0.7, label="Shuffled labels")
        ax.axvline(real, color="red", label=f"Real labels (p = {p:.3f})")
        ax.set_xlabel(name); ax.set_ylabel("Count"); ax.legend()
        ax.set_title(f"Label-permutation test: {name}")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_importance(imp, path):
    imp = imp.sort_values("gain_mean")
    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, 0.35 * len(imp))))
    axes[0].barh(imp.index, imp["gain_mean"], xerr=imp["gain_std"])
    axes[0].set_title("Gain importance (mean +/- sd over seeds)")
    axes[1].barh(imp.index, imp["perm_auc_drop_mean"].fillna(0), xerr=imp["perm_auc_drop_std"].fillna(0))
    axes[1].set_title("Permutation importance (AUC drop, out-of-fold)")
    axes[2].barh(imp.index, imp["mean_abs_shap"])
    axes[2].set_title("Mean |SHAP| (log-odds)")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# --------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="quick smoke test")
    args = parser.parse_args()
    if args.fast:
        apply_fast_mode()
        print("*** FAST MODE: reduced rounds/repeats, numbers are NOT for reporting ***\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()
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
    print(f"Holdout positives: {len(holdout_idx)} | rounds evaluated: {N_ROUNDS} "
          f"(curve explored to {CURVE_ROUNDS})")

    # ---- [1] feature diagnostics
    print("\n[1/7] Feature diagnostics")
    diag = diagnose_features(df, feature_cols)
    diag.to_csv(os.path.join(OUT_DIR, "feature_diagnostics.csv"), index=False)
    print(diag.to_string(index=False))
    flagged = diag[diag["kind"] != "peptide-level"]["feature"].tolist()

    # ---- [2] grouped CV + loss curves (one pass)
    print("\n[2/7] Grouped out-of-fold CV and boosting loss curves")
    cv = []
    for rep in range(N_REPEATS):
        cv.append(run_cv(X, y, groups, pool_mask, holdout_idx,
                         seed=RANDOM_SEED + rep, record_curves=True))
        print(f"    repeat {rep + 1}/{N_REPEATS} done")

    rows, hold_rows = [], []
    for rep, r in enumerate(cv):
        m = compute_metrics(y[pool_idx], r["oof"][pool_idx])
        m["repeat"] = rep
        rows.append(m)
        if len(holdout_idx):
            hold_rows.append(holdout_percentiles(
                [r["oof"][n] for n in r["fold_negatives"]], r["holdout_scores"], holdout_idx))
    by_repeat = pd.DataFrame(rows)
    by_repeat.to_csv(os.path.join(OUT_DIR, "cv_metrics_by_repeat.csv"), index=False)

    metric_cols = [c for c in by_repeat.columns if c not in ("repeat", "n_pool", "n_positives")]
    summary = pd.DataFrame({"xgboost_mean": by_repeat[metric_cols].mean(),
                            "xgboost_sd": by_repeat[metric_cols].std()})
    for name, col in [("linear", "linear_score"), ("baseline", "baseline_score")]:
        if df[col].notna().all():
            m = compute_metrics(y[pool_idx], df[col].values[pool_idx])
            summary[name] = pd.Series({k: m[k] for k in metric_cols})
    summary.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"))
    print("\n  Out-of-fold performance (pool of non-holdout peptides):")
    print(summary.round(4).to_string())
    real_auc, real_ap = by_repeat["auc"].mean(), by_repeat["avg_precision"].mean()

    # side-by-side with Random Forest
    rf_path = os.path.join(RF_EVAL_DIR, "metrics_summary.csv")
    if os.path.exists(rf_path):
        rf = pd.read_csv(rf_path, index_col=0)
        comp = pd.DataFrame({"baseline": summary.get("baseline"), "linear": summary.get("linear"),
                             "random_forest": rf.get("random_forest_mean"),
                             "random_forest_sd": rf.get("random_forest_sd"),
                             "xgboost": summary["xgboost_mean"], "xgboost_sd": summary["xgboost_sd"]})
        comp.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"))
        print("\n  Four-model comparison (identical folds):")
        print(comp.loc[["auc", "avg_precision", "precision@top100", "recall@top100",
                        "enrichment@top100", "median_positive_rank_pct"]].round(4).to_string())
    else:
        print(f"\n  ({rf_path} not found; run random_forest_evaluation.py for a 4-model table.)")

    # holdout positives
    if len(holdout_idx):
        hold_df = pd.DataFrame({"protein": df.loc[holdout_idx, "protein"].values,
                                "peptide": df.loc[holdout_idx, "peptide"].values,
                                "xgb_pct_outranking": np.mean(hold_rows, axis=0)})
        for name, col in [("linear", "linear_score"), ("baseline", "baseline_score")]:
            if df[col].notna().all():
                fixed = [df[col].values[holdout_idx]] * N_SPLITS
                reps = [holdout_percentiles([df[col].values[n] for n in r["fold_negatives"]],
                                            fixed, holdout_idx) for r in cv]
                hold_df[f"{name}_pct_outranking"] = np.mean(reps, axis=0)
        hold_df.to_csv(os.path.join(OUT_DIR, "holdout_percentiles.csv"), index=False)
        print("\n  Held-out positives (fraction of unseen negatives that outrank them; lower = better):")
        print(hold_df.round(4).to_string(index=False))
        print("  (Only a handful of positives: illustrative, not a headline number.)")

    # loss curves
    curves = pd.concat([c for r in cv for c in r["curves"]], ignore_index=True)
    mean_curve = curves.groupby("round").mean().reset_index()
    mean_curve.to_csv(os.path.join(OUT_DIR, "loss_curve.csv"), index=False)
    best_round = plot_loss_curve(curves, os.path.join(OUT_DIR, "loss_curve.png"))
    at = mean_curve[mean_curve["round"] == N_ROUNDS].iloc[0]
    peak = mean_curve["val_aucpr"].rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean().max()
    print(f"\n  At {N_ROUNDS} rounds: train AUCPR {at.train_aucpr:.3f} vs validation {at.val_aucpr:.3f} "
          f"(gap {at.train_aucpr - at.val_aucpr:.3f})")
    print(f"  Validation AUCPR peaks at round {best_round} (smoothed {peak:.3f}); "
          f"configured rounds = {N_ROUNDS}.")
    if best_round < 0.6 * N_ROUNDS:
        print("  -> Validation peaks well before the configured rounds: consider fewer rounds.")
    elif best_round > 1.4 * N_ROUNDS:
        print("  -> Validation still improving past the configured rounds: could use more.")
    else:
        print("  -> Configured rounds are close to the validation optimum.")

    # stability between repeats (different folds AND seeds)
    stab = []
    for i in range(len(cv)):
        for j in range(i + 1, len(cv)):
            stab.append({"repeat_a": i, "repeat_b": j,
                         "spearman_oof": spearmanr(cv[i]["oof"][pool_idx], cv[j]["oof"][pool_idx])[0]})
    stab = pd.DataFrame(stab)
    stab.to_csv(os.path.join(OUT_DIR, "stability.csv"), index=False)
    print(f"\n  Out-of-fold ranking stability across repeats: mean Spearman "
          f"{stab['spearman_oof'].mean():.3f} (min {stab['spearman_oof'].min():.3f})")

    # ---- [3] permutation test
    print("\n[3/7] Label-permutation test")
    rng = np.random.default_rng(RANDOM_SEED)
    null = []
    for i in range(N_PERMUTATIONS):
        y_perm = y.copy()
        y_perm[pool_idx] = rng.permutation(y[pool_idx])
        r = run_cv(X, y_perm, groups, pool_mask, np.array([], dtype=int), seed=1000 + i)
        null.append({"auc": roc_auc_score(y_perm[pool_idx], r["oof"][pool_idx]),
                     "avg_precision": average_precision_score(y_perm[pool_idx], r["oof"][pool_idx])})
        print(f"    permutation {i + 1}/{N_PERMUTATIONS}")
    null = pd.DataFrame(null)
    null.to_csv(os.path.join(OUT_DIR, "permutation_null.csv"), index=False)
    p_auc = (1 + (null["auc"] >= real_auc).sum()) / (len(null) + 1)
    p_ap = (1 + (null["avg_precision"] >= real_ap).sum()) / (len(null) + 1)
    plot_permutation(null, real_auc, real_ap, p_auc, p_ap, os.path.join(OUT_DIR, "permutation_test.png"))
    print(f"  Real AUC {real_auc:.3f} vs shuffled mean {null['auc'].mean():.3f} (p = {p_auc:.3f}); "
          f"real AP {real_ap:.3f} vs shuffled mean {null['avg_precision'].mean():.3f} (p = {p_ap:.3f})")

    # ---- [4] feature importance
    print("\n[4/7] Feature importance (gain, permutation, SHAP)")
    gain = []
    for s in IMPORTANCE_SEEDS:
        clf = new_model(y[pool_idx], s, N_ROUNDS)
        clf.fit(X.iloc[pool_idx], y[pool_idx], verbose=False)
        gain.append(pd.Series(clf.feature_importances_, index=X.columns))
    gain = pd.DataFrame(gain)
    imp = pd.DataFrame({"gain_mean": gain.mean(), "gain_std": gain.std()})

    perm = fold_permutation_importance(X, y, groups, pool_mask, RANDOM_SEED)
    if perm:
        perm = pd.DataFrame(perm, columns=X.columns)
        imp["perm_auc_drop_mean"], imp["perm_auc_drop_std"] = perm.mean(), perm.std()
    else:
        imp["perm_auc_drop_mean"] = imp["perm_auc_drop_std"] = np.nan

    final = new_model(y[pool_idx], RANDOM_SEED, N_ROUNDS)
    final.fit(X.iloc[pool_idx], y[pool_idx], verbose=False)
    contribs = final.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    imp["mean_abs_shap"] = pd.Series(np.abs(contribs[pool_idx]).mean(axis=0), index=X.columns)
    imp = imp.sort_values("gain_mean", ascending=False)
    imp.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"))
    plot_importance(imp, os.path.join(OUT_DIR, "feature_importance.png"))
    print(imp.round(4).to_string())

    # ---- [5] ablation of protein-level features
    if flagged:
        print(f"\n[5/7] Ablation: removing {flagged}")
        keep = [c for c in X.columns
                if not any(c == f or c.startswith(f + "_") for f in flagged)]
        res = [run_cv(X[keep], y, groups, pool_mask, np.array([], dtype=int), seed=RANDOM_SEED + rep)
               for rep in range(N_REPEATS)]
        abl = pd.DataFrame([compute_metrics(y[pool_idx], r["oof"][pool_idx]) for r in res])
        ablation = pd.DataFrame({"all_features": by_repeat[["auc", "avg_precision"]].mean(),
                                 "without_flagged": abl[["auc", "avg_precision"]].mean()})
        ablation.to_csv(os.path.join(OUT_DIR, "ablation.csv"))
        print(ablation.round(4).to_string())
    else:
        print("\n[5/7] Ablation skipped (no constant or protein-level features found).")

    # ---- [6] scale_pos_weight sensitivity
    print("\n[6/7] Sensitivity to scale_pos_weight")
    sens = []
    for label, power in [("full (as configured)", None), ("square root", 0.5), ("none (=1)", 0.0)]:
        res = [run_cv(X, y, groups, pool_mask, np.array([], dtype=int),
                      seed=RANDOM_SEED + rep, spw_power=power) for rep in range(N_REPEATS)]
        m = pd.DataFrame([compute_metrics(y[pool_idx], r["oof"][pool_idx]) for r in res])
        sens.append({"scale_pos_weight": label, "auc": m["auc"].mean(),
                     "avg_precision": m["avg_precision"].mean(),
                     "precision@top100": m["precision@top100"].mean()})
    sens = pd.DataFrame(sens)
    sens.to_csv(os.path.join(OUT_DIR, "sensitivity.csv"), index=False)
    print(sens.round(4).to_string(index=False))

    # ---- [7] agreement + candidates
    print("\n[7/7] Model agreement and candidate list")
    stack = np.vstack([r["oof"] for r in cv])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = np.nanmean(stack, axis=0)
    for j, i in enumerate(holdout_idx):
        score[i] = np.mean([hs[j] for r in cv for hs in r["holdout_scores"]])
    df["xgb_oof_score"] = score
    df["xgb_oof_rank"] = df["xgb_oof_score"].rank(ascending=False, method="min").astype(int)

    rf_oof = os.path.join(RF_EVAL_DIR, "oof_scores.csv")
    has_rf = os.path.exists(rf_oof)
    if has_rf:
        n = len(df)
        df = df.merge(pd.read_csv(rf_oof), on=["protein", "peptide", "start", "end"], how="left")
        assert len(df) == n, "RF scores merge changed the row count"
    else:
        print(f"  ({rf_oof} not found: rerun random_forest_evaluation.py to include RF here.)")

    cols = {"xgboost": "xgb_oof_score", "baseline": "baseline_score"}
    if df["linear_score"].notna().all():
        cols["linear"] = "linear_score"
    if has_rf and df["rf_oof_score"].notna().all():
        cols["random_forest"] = "rf_oof_score"
    agree = []
    names = sorted(cols)
    for a in names:
        for b in names:
            if a >= b:
                continue
            row = {"model_a": a, "model_b": b,
                   "spearman": spearmanr(df[cols[a]], df[cols[b]])[0]}
            for k in (50, 100, 250):
                ta, tb = set(df.nlargest(k, cols[a]).index), set(df.nlargest(k, cols[b]).index)
                row[f"top{k}_overlap"] = len(ta & tb)
                row[f"top{k}_jaccard"] = len(ta & tb) / len(ta | tb)
            agree.append(row)
    agree = pd.DataFrame(agree)
    agree.to_csv(os.path.join(OUT_DIR, "model_agreement.csv"), index=False)
    print(agree.round(3).to_string(index=False))

    keep_cols = [c for c in ["protein", "peptide", "start", "end", "length", "xgb_oof_score",
                             "xgb_oof_rank", "rf_oof_rank", "BA_rank", "cleavage_score",
                             "mhcflurry_processing_score", "expression_tpm", "tcell_evidence",
                             "mhc_ligand_evidence", "split"] if c in df.columns]
    top = df.sort_values("xgb_oof_rank").head(100)[keep_cols].copy()
    ev = [c for c in ("tcell_evidence", "mhc_ligand_evidence") if c in top.columns]
    top["known_evidence"] = top[ev].max(axis=1).astype(bool)
    if "rf_oof_rank" in top.columns:
        top["also_rf_top100"] = top["rf_oof_rank"] <= 100
    # per-peptide drivers (TreeSHAP of the final model; log-odds scale)
    drivers = []
    for i in top.index:
        c = pd.Series(contribs[i], index=X.columns)
        t = c.reindex(c.abs().sort_values(ascending=False).index).head(3)
        drivers.append("; ".join(f"{k}({v:+.2f})" for k, v in t.items()))
    top["top_drivers"] = drivers
    top.to_csv(os.path.join(OUT_DIR, "top_candidates.csv"), index=False)

    print(f"\n  Top 100 by out-of-fold XGBoost score: {int(top['known_evidence'].sum())} have known "
          f"evidence, {int((~top['known_evidence']).sum())} are novel candidates.")
    if "also_rf_top100" in top.columns:
        print(f"  {int(top['also_rf_top100'].sum())} of these are also in the RF top 100.")
    print("  By protein:", top["protein"].value_counts().to_dict())
    if "length" in top.columns:
        print("  By length: ", top["length"].value_counts().sort_index().to_dict())
    print(f"\nAll outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()