"""Run grouped-CV feature ablations for the Random Forest and XGBoost models."""

import argparse
import itertools
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import random_forest_evaluation as rf_evaluation
import xgboost_evaluation as xgb_evaluation
import linear as linear_model
from random_forest import (
    RANKING_FEATURES_PATH,
    SPLIT_PATH,
    TARGET_LABEL_COLUMN,
    build_model_matrix,
    select_feature_columns,
)

OUT_DIR = Path("results/feature_ablation")
GROUP_BIN_SIZE = rf_evaluation.GROUP_BIN_SIZE
RANDOM_SEED = rf_evaluation.RANDOM_SEED
N_SPLITS = rf_evaluation.N_SPLITS
N_REPEATS = rf_evaluation.N_REPEATS
RF_TREES = rf_evaluation.CV_TREES
XGB_ROUNDS = xgb_evaluation.N_ROUNDS
METRICS = {
    "auc": "auc",
    "average_precision": "avg_precision",
    "precision@100": "precision@top100",
    "recall@100": "recall@top100",
    "enrichment@100": "enrichment@top100",
}

FEATURE_GROUPS = {
    "HLA binding": ("BA_ic50", "BA_rank", "EL_score", "EL_rank", "hla_frequency"),
    "Cleavage": ("cleavage_score",),
    "Antigen processing": ("mhcflurry_processing_score",),
    "Expression": ("expression_tpm",),
    "Mutation status": ("mutation_status",),
    "Peptide properties": ("length", "hydrophobicity", "molecular_weight"),
}
SUMMARY_MODEL_ORDER = [
    "Binding baseline",
    "Linear",
    "Random Forest",
    "XGBoost",
]
SUMMARY_METRICS = ("auc", "average_precision", "precision@100")


def load_data():
    """Load features and the existing split without changing either file."""
    df = pd.read_csv(RANKING_FEATURES_PATH).reset_index(drop=True)
    if TARGET_LABEL_COLUMN not in df:
        raise ValueError(
            f"'{TARGET_LABEL_COLUMN}' not found in {RANKING_FEATURES_PATH}"
        )
    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(
            f"{SPLIT_PATH} is required; run the existing evaluation setup first."
        )

    df[TARGET_LABEL_COLUMN] = df[TARGET_LABEL_COLUMN].fillna(0).astype(int)
    split_df = pd.read_csv(SPLIT_PATH)
    required = {"peptide", "protein", "split"}
    if not required.issubset(split_df.columns):
        raise ValueError(f"{SPLIT_PATH} must contain {sorted(required)}")
    if split_df.duplicated(["peptide", "protein"]).any():
        raise ValueError(f"{SPLIT_PATH} contains duplicate peptide/protein keys.")

    split_lookup = {
        (row.peptide, row.protein): row.split for row in split_df.itertuples()
    }
    positive_keys = set(
        map(
            tuple,
            df.loc[df[TARGET_LABEL_COLUMN] == 1, ["peptide", "protein"]].itertuples(
                index=False, name=None
            ),
        )
    )
    missing_positives = positive_keys - split_lookup.keys()
    if missing_positives:
        raise ValueError(
            f"{len(missing_positives)} positive(s) are missing from {SPLIT_PATH}; "
            "update the persisted split before evaluating."
        )

    df["split"] = [
        split_lookup.get((peptide, protein), "unlabeled")
        for peptide, protein in zip(df["peptide"], df["protein"])
    ]
    holdout_count = int(
        ((df[TARGET_LABEL_COLUMN] == 1) & (df["split"] == "holdout")).sum()
    )
    if holdout_count != 3:
        raise ValueError(
            f"Expected the three permanent holdout positives, found {holdout_count}."
        )
    return df


def validate_feature_groups(feature_cols):
    memberships = {}
    for group, features in FEATURE_GROUPS.items():
        for feature in features:
            if feature in memberships:
                raise ValueError(f"Feature {feature!r} belongs to multiple groups.")
            memberships[feature] = group

    unassigned = sorted(set(feature_cols) - memberships.keys())
    if unassigned:
        raise ValueError(
            "Add these model inputs to FEATURE_GROUPS before running ablations: "
            + ", ".join(unassigned)
        )
    return {
        group: [feature for feature in features if feature in feature_cols]
        for group, features in FEATURE_GROUPS.items()
    }


def make_matrix(df, feature_cols):
    return build_model_matrix(df, feature_cols).astype(float)


def run_repeat(model, X, y, groups, pool_mask, seed, rf_trees, xgb_rounds):
    pool_idx = np.flatnonzero(pool_mask)
    if model == "random_forest":
        oof, _, _, _ = rf_evaluation.run_grouped_cv(
            X,
            y,
            groups,
            pool_mask,
            np.array([], dtype=int),
            seed=seed,
            n_estimators=rf_trees,
        )
    else:
        result = xgb_evaluation.run_cv(
            X,
            y,
            groups,
            pool_mask,
            np.array([], dtype=int),
            seed=seed,
        )
        oof = result["oof"]
        if xgb_rounds != xgb_evaluation.N_ROUNDS:
            raise ValueError(
                "XGBoost rounds do not match the evaluation configuration."
            )

    metrics = rf_evaluation.compute_metrics(
        y[pool_idx], oof[pool_idx], seed=RANDOM_SEED
    )
    return {name: float(metrics[source]) for name, source in METRICS.items()}


def run_scenario(
    model, df, y, groups, pool_mask, feature_cols, repeats, rf_trees, xgb_rounds
):
    X = make_matrix(df, feature_cols)
    return [
        run_repeat(
            model,
            X,
            y,
            groups,
            pool_mask,
            seed=RANDOM_SEED + repeat,
            rf_trees=rf_trees,
            xgb_rounds=xgb_rounds,
        )
        for repeat in range(repeats)
    ]


def metric_records(kind, name, removed, model, baseline, ablated, group_deltas=None):
    rows = []
    for repeat, (base, result) in enumerate(zip(baseline, ablated)):
        row = {
            "ablation": name,
            "removed_features": ";".join(removed),
            "model": model,
            "repeat": repeat,
        }
        if kind == "interaction":
            group_a, group_b = name.split(" + ", maxsplit=1)
            row["group_a"] = group_a
            row["group_b"] = group_b
        for metric in METRICS:
            delta = result[metric] - base[metric]
            row[f"baseline_{metric}"] = base[metric]
            row[f"ablated_{metric}"] = result[metric]
            row[f"delta_{metric}"] = delta
            if kind == "interaction":
                group_a, group_b = name.split(" + ", maxsplit=1)
                single_a = group_deltas[(group_a, model, repeat)][metric]
                single_b = group_deltas[(group_b, model, repeat)][metric]
                row[f"individual_delta_sum_{metric}"] = single_a + single_b
                row[f"interaction_effect_{metric}"] = delta - single_a - single_b
        rows.append(row)
    return rows


def add_effect_direction(frame, tolerance):
    mean_ap = frame.groupby(["ablation", "model"])["delta_average_precision"].mean()

    def direction(row):
        delta = mean_ap[(row["ablation"], row["model"])]
        if delta > tolerance:
            return "improved"
        if delta < -tolerance:
            return "decreased"
        return "little_change"

    frame["average_precision_effect"] = frame.apply(direction, axis=1)
    return frame


def symmetric_norm(values):
    limit = float(np.nanmax(np.abs(values))) if values.size else 0.0
    limit = max(limit, 1e-12)
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def plot_ablation_heatmap(frame, output_path, metrics, title):
    models = ["random_forest", "xgboost"]
    ablations = frame["ablation"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(8 * len(metrics), max(4, 0.42 * len(ablations) + 2)),
        squeeze=False,
    )
    for ax, metric in zip(axes[0], metrics):
        matrix = (
            frame.groupby(["ablation", "model"])[f"delta_{metric}"]
            .mean()
            .unstack("model")
            .reindex(index=ablations, columns=models)
        )
        values = matrix.to_numpy(dtype=float)
        image = ax.imshow(
            values, cmap="RdBu", norm=symmetric_norm(values), aspect="auto"
        )
        ax.set_xticks(range(len(models)), ["Random Forest", "XGBoost"])
        ax.set_yticks(range(len(ablations)), ablations)
        ax.set_title(metric.replace("_", " "))
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{values[row_idx, col_idx]:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        fig.colorbar(image, ax=ax, label="Ablated - full model")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_interaction_heatmap(frame, output_path):
    models = ["random_forest", "xgboost"]
    interactions = frame["ablation"].drop_duplicates().tolist()
    matrix = (
        frame.groupby(["ablation", "model"])["delta_average_precision"]
        .mean()
        .unstack("model")
        .reindex(index=interactions, columns=models)
    )
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.38 * len(interactions) + 2)))
    image = ax.imshow(values, cmap="RdBu", norm=symmetric_norm(values), aspect="auto")
    ax.set_xticks(range(len(models)), ["Random Forest", "XGBoost"])
    ax.set_yticks(range(len(interactions)), interactions)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{values[row_idx, col_idx]:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    ax.set_title("Average precision change after removing feature-group pairs")
    fig.colorbar(image, ax=ax, label="Ablated - full model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def summary_metric_values(y, scores):
    metrics = rf_evaluation.compute_metrics(y, scores, seed=RANDOM_SEED)
    return {metric: float(metrics[METRICS[metric]]) for metric in SUMMARY_METRICS}


def make_model_feature_row(model, feature, repeat, baseline, ablated):
    row = {"model": model, "feature": feature, "repeat": repeat}
    deltas = []
    for metric in SUMMARY_METRICS:
        delta = ablated[metric] - baseline[metric]
        row[f"baseline_{metric}"] = baseline[metric]
        row[f"ablated_{metric}"] = ablated[metric]
        row[f"delta_{metric}"] = delta
        deltas.append(delta)
    row["mean_normalized_delta_pp"] = 100 * float(np.mean(deltas))
    return row


def direct_model_feature_rows(df, pool_mask, features):
    pool_df = df.loc[pool_mask].copy().reset_index(drop=True)
    y = pool_df[TARGET_LABEL_COLUMN].to_numpy(dtype=int)
    rows = []

    baseline_scores = -pool_df["BA_rank"].to_numpy(dtype=float)
    baseline_metrics = summary_metric_values(y, baseline_scores)
    for feature in features:
        ablated_scores = (
            np.zeros(len(pool_df), dtype=float)
            if feature == "BA_rank"
            else baseline_scores
        )
        rows.append(
            make_model_feature_row(
                "Binding baseline",
                feature,
                "single",
                baseline_metrics,
                summary_metric_values(y, ablated_scores),
            )
        )

    linear_features = linear_model.build_features(pool_df)
    full_linear_scores = linear_model.compute_linear_score(linear_features)[
        "linear_score"
    ].to_numpy(dtype=float)
    linear_contributions = {
        "BA_rank": (
            linear_model.W_MHC * linear_features["mhc_binding_feature"].to_numpy()
        ),
        "cleavage_score": (
            linear_model.W_CLEAVAGE * linear_features["cleavage_feature"].to_numpy()
        ),
        "expression_tpm": (
            linear_model.W_EXPRESSION * linear_features["expression_feature"].to_numpy()
        ),
    }
    linear_metrics = summary_metric_values(y, full_linear_scores)
    for feature in features:
        ablated_scores = full_linear_scores - linear_contributions.get(
            feature, np.zeros(len(pool_df), dtype=float)
        )
        rows.append(
            make_model_feature_row(
                "Linear",
                feature,
                "single",
                linear_metrics,
                summary_metric_values(y, ablated_scores),
            )
        )
    return rows


def build_all_model_feature_summary(df, pool_mask, features, individual_results):
    rows = direct_model_feature_rows(df, pool_mask, features)
    display_names = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    for _, result in individual_results.iterrows():
        baseline = {
            metric: float(result[f"baseline_{metric}"]) for metric in SUMMARY_METRICS
        }
        ablated = {
            metric: float(result[f"ablated_{metric}"]) for metric in SUMMARY_METRICS
        }
        rows.append(
            make_model_feature_row(
                display_names[result["model"]],
                result["ablation"],
                str(result["repeat"]),
                baseline,
                ablated,
            )
        )
    return pd.DataFrame(rows)


def plot_all_model_feature_heatmap(frame, features, output_path):
    matrix = (
        frame.groupby(["model", "feature"])["mean_normalized_delta_pp"]
        .mean()
        .unstack("feature")
        .reindex(index=SUMMARY_MODEL_ORDER, columns=features)
    )
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(15, 5.5))
    image = ax.imshow(
        values,
        cmap="RdBu",
        norm=symmetric_norm(values),
        aspect="auto",
    )
    ax.set_xticks(
        range(len(features)),
        [feature.replace("_", "\n") for feature in features],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(len(SUMMARY_MODEL_ORDER)), SUMMARY_MODEL_ORDER)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{values[row_idx, col_idx]:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    ax.set_title(
        "Mean feature-removal effect across four models\n"
        "Blue: removal improved metrics; red: removal reduced metrics"
    )
    fig.colorbar(
        image,
        ax=ax,
        label="Mean change in AUC, average precision, and precision@100 (pp)",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_all_model_feature_summary(
    df, pool_mask, features, output_dir, individual_results
):
    summary = build_all_model_feature_summary(
        df, pool_mask, features, individual_results
    )
    csv_path = output_dir / "feature_effect_summary.csv"
    png_path = output_dir / "all_model_feature_effect_heatmap.png"
    summary.to_csv(csv_path, index=False)
    plot_all_model_feature_heatmap(summary, features, png_path)
    print(f"Saved four-model feature summary to {csv_path} and {png_path}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="run the full-model reference CV only and compare it with prior evaluations",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use one repeat and fewer trees/rounds for a smoke test; not reportable",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="regenerate the four-model summary from existing ablation CSVs without retraining",
    )
    parser.add_argument("--negligible-delta", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if args.negligible_delta < 0:
        parser.error("--negligible-delta must be non-negative")
    if args.summary_only and (args.fast or args.baseline_only):
        parser.error("--summary-only cannot be combined with --fast or --baseline-only")

    repeats = 1 if args.fast else N_REPEATS
    rf_trees = 30 if args.fast else RF_TREES
    xgb_rounds = 40 if args.fast else XGB_ROUNDS
    if args.fast:
        xgb_evaluation.N_ROUNDS = xgb_rounds

    df = load_data()
    feature_cols = select_feature_columns(df)
    feature_groups = validate_feature_groups(feature_cols)
    y = df[TARGET_LABEL_COLUMN].to_numpy(dtype=int)
    holdout_mask = df["split"].eq("holdout").to_numpy()
    pool_mask = ~holdout_mask
    groups = (
        df["protein"].astype(str)
        + "_"
        + (df["start"] // GROUP_BIN_SIZE).astype(int).astype(str)
    ).to_numpy()
    pool_idx = np.flatnonzero(pool_mask)

    if pool_mask.sum() <= 100:
        raise ValueError(
            "The non-holdout evaluation pool must contain more than 100 rows."
        )
    if np.unique(y[pool_idx]).size != 2:
        raise ValueError("The non-holdout pool must contain both label classes.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        individual_path = args.output_dir / "individual_ablation.csv"
        if not individual_path.exists():
            raise FileNotFoundError(
                f"{individual_path} is required for --summary-only."
            )
        individual_results = pd.read_csv(individual_path)
        write_all_model_feature_summary(
            df,
            pool_mask,
            [feature for group in feature_groups.values() for feature in group],
            args.output_dir,
            individual_results,
        )
        return

    print(
        f"CV pool: {len(pool_idx)} rows ({int(y[pool_idx].sum())} positives); "
        f"holdout positives excluded: {int((holdout_mask & (y == 1)).sum())}"
    )
    print(
        f"Using {N_SPLITS} grouped folds x {repeats} repeat(s), "
        f"RF trees={rf_trees}, XGBoost rounds={xgb_rounds}."
    )

    models = ("random_forest", "xgboost")
    baseline_results = {}
    for model in models:
        print(f"Running {model} full-feature reference...")
        baseline_results[model] = run_scenario(
            model,
            df,
            y,
            groups,
            pool_mask,
            feature_cols,
            repeats,
            rf_trees,
            xgb_rounds,
        )

    baseline_rows = []
    for model in models:
        for repeat, metrics in enumerate(baseline_results[model]):
            baseline_rows.append({"model": model, "repeat": repeat, **metrics})
    baseline_frame = pd.DataFrame(baseline_rows)
    baseline_path = args.output_dir / "baseline_reference.csv"
    baseline_frame.to_csv(baseline_path, index=False)
    print(f"Saved full-model repeat metrics to {baseline_path}.")

    if args.fast:
        print("FAST MODE: these smoke-test metrics are not for reporting.")
    else:
        compare_prior_results(baseline_results)
    if args.baseline_only:
        return

    group_deltas = {}
    result_frames = {}
    for kind in ("group", "individual", "interaction"):
        print(f"Running {kind} ablations...")
        if kind == "group":
            scenarios = [
                (group, features) for group, features in feature_groups.items()
            ]
        elif kind == "individual":
            scenarios = [
                (feature, [feature])
                for features in feature_groups.values()
                for feature in features
            ]
        else:
            scenarios = [
                (
                    f"{group_a} + {group_b}",
                    feature_groups[group_a] + feature_groups[group_b],
                )
                for group_a, group_b in itertools.combinations(feature_groups, 2)
            ]

        rows = []
        for name, removed in scenarios:
            kept = [feature for feature in feature_cols if feature not in removed]
            if not kept:
                raise ValueError(f"Removing {name!r} would leave no features.")
            for model in models:
                ablated = run_scenario(
                    model,
                    df,
                    y,
                    groups,
                    pool_mask,
                    kept,
                    repeats,
                    rf_trees,
                    xgb_rounds,
                )
                rows.extend(
                    metric_records(
                        kind,
                        name,
                        removed,
                        model,
                        baseline_results[model],
                        ablated,
                        group_deltas=group_deltas,
                    )
                )
                if kind == "group":
                    for repeat, (base, result) in enumerate(
                        zip(baseline_results[model], ablated)
                    ):
                        group_deltas[(name, model, repeat)] = {
                            metric: result[metric] - base[metric] for metric in METRICS
                        }
        frame = pd.DataFrame(rows)
        if kind == "individual":
            frame = add_effect_direction(frame, args.negligible_delta)
        result_frames[kind] = frame
        csv_path = args.output_dir / f"{kind}_ablation.csv"
        frame.to_csv(csv_path, index=False)
        print(f"Saved {kind} repeat-level results to {csv_path}.")

    plot_ablation_heatmap(
        result_frames["group"],
        args.output_dir / "group_ablation_heatmap.png",
        ["average_precision", "auc", "enrichment@100"],
        "Feature-group ablation",
    )
    plot_ablation_heatmap(
        result_frames["individual"],
        args.output_dir / "individual_ablation_heatmap.png",
        ["average_precision", "auc", "enrichment@100"],
        "Individual-feature ablation",
    )
    plot_interaction_heatmap(
        result_frames["interaction"],
        args.output_dir / "interaction_heatmap.png",
    )
    write_all_model_feature_summary(
        df,
        pool_mask,
        [feature for group in feature_groups.values() for feature in group],
        args.output_dir,
        result_frames["individual"],
    )
    print(f"Heatmaps saved under {args.output_dir}.")


def compare_prior_results(baseline_results):
    prior_paths = {
        "random_forest": Path(
            "results/random_forest/evaluation/cv_metrics_by_repeat.csv"
        ),
        "xgboost": Path("results/xgboost/evaluation/cv_metrics_by_repeat.csv"),
    }
    for model, path in prior_paths.items():
        if not path.exists():
            print(
                f"No prior evaluation found for {model} at {path}; skipping comparison."
            )
            continue
        prior = pd.read_csv(path)
        if len(prior) != len(baseline_results[model]):
            print(f"Prior repeat count differs for {model}; cannot compare directly.")
            continue
        prior_metrics = {
            name: source for name, source in METRICS.items() if source in prior.columns
        }
        if len(prior_metrics) != len(METRICS):
            print(f"Prior metrics are incomplete for {model}; cannot compare directly.")
            continue
        deltas = []
        for repeat, result in enumerate(baseline_results[model]):
            old = prior.iloc[repeat]
            deltas.extend(
                abs(result[name] - float(old[source]))
                for name, source in prior_metrics.items()
            )
        max_delta = max(deltas, default=0.0)
        print(
            f"Prior {model} reference comparison: max absolute metric difference "
            f"across repeats = {max_delta:.6g}."
        )


if __name__ == "__main__":
    main()
