from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import brier_score_loss, f1_score
from sklearn.tree import DecisionTreeClassifier

from .benchmark import (
    TabularPreprocessor,
    bootstrap_mean_ci,
    explain_lime,
    explanation_indices,
    extract_shap_vector,
    make_lime_explainer,
    make_model,
    predictive_metrics,
    prepare_dataset,
    safe_cosine,
    stratified_indices,
)
from .transparency import _ranking_score

matplotlib.use("Agg")


def _artifact_path(output_dir: Path, dataset: str, task: str, model_name: str) -> Path:
    name = f"{dataset}_{task}_{model_name}".replace("-", "_")
    return output_dir / "models" / f"{name}.joblib"


def _calibration_rows(
    dataset: str,
    model_name: str,
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    bins: int = 10,
) -> tuple[list[dict], dict]:
    attack_index = classes.index("Attack")
    probability = model.predict_proba(x_test)[:, attack_index]
    target = (y_test == attack_index).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    assigned = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    rows = []
    weighted_gap = 0.0
    for bin_index in range(bins):
        mask = assigned == bin_index
        if not np.any(mask):
            continue
        mean_probability = float(probability[mask].mean())
        observed_frequency = float(target[mask].mean())
        count = int(mask.sum())
        weighted_gap += count * abs(mean_probability - observed_frequency)
        rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "bin": bin_index,
                "count": count,
                "mean_attack_probability": mean_probability,
                "observed_attack_frequency": observed_frequency,
            }
        )
    summary = {
        "dataset": dataset,
        "model": model_name,
        "brier_score": float(brier_score_loss(target, probability)),
        "expected_calibration_error": float(weighted_gap / len(target)),
    }
    return rows, summary


def _masking_baseline_rows(
    dataset: str,
    task: str,
    model_name: str,
    model,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    preprocessor: TabularPreprocessor,
    seed: int,
    n: int = 12,
) -> list[dict]:
    prediction = model.predict(x_test).astype(int)
    indices = explanation_indices(y_test, prediction, min(n, len(y_test)), seed)
    explained = x_test[indices]
    explained_prediction = prediction[indices]
    tree_explainer = shap.TreeExplainer(model)
    train_probe_indices = stratified_indices(model.predict(x_train), min(5000, len(x_train)), seed)
    train_probe = x_train[train_probe_indices]
    train_probe_prediction = model.predict(train_probe).astype(int)
    scale = np.std(train_probe, axis=0)
    scale[scale < 1e-9] = 1.0
    rng = np.random.default_rng(seed)
    rows = []
    for local_id, (source_index, row, predicted_class) in enumerate(
        zip(indices, explained, explained_prediction)
    ):
        vector = extract_shap_vector(
            tree_explainer(row.reshape(1, -1), check_additivity=False), int(predicted_class)
        )
        ranking = np.argsort(-np.abs(vector))
        candidates = train_probe[train_probe_prediction == predicted_class]
        if len(candidates):
            distance = np.linalg.norm((candidates - row) / scale, axis=1)
            nearest = candidates[int(np.argmin(distance))]
        else:
            nearest = np.asarray(preprocessor.baseline_)
        opposite_candidates = train_probe[train_probe_prediction != predicted_class]
        if len(opposite_candidates):
            opposite_distance = np.linalg.norm((opposite_candidates - row) / scale, axis=1)
            nearest_opposite = opposite_candidates[int(np.argmin(opposite_distance))]
        else:
            nearest_opposite = np.asarray(preprocessor.baseline_)
        baselines = {
            "training_median_mode": np.asarray(preprocessor.baseline_, dtype=float),
            "zero": np.zeros_like(row),
            "marginal_training_row": train_probe[int(rng.integers(0, len(train_probe)))],
            "nearest_same_prediction": nearest,
            "nearest_opposite_prediction": nearest_opposite,
        }
        for baseline_name, baseline in baselines.items():
            score, details = _ranking_score(
                model, row, int(predicted_class), baseline, ranking, [1, 3, 5, 10]
            )
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "model": model_name,
                    "source_test_index": int(source_index),
                    "baseline": baseline_name,
                    "deletion_score": score,
                    "any_prediction_flip": int(any(item["prediction_flipped"] for item in details)),
                }
            )
    return rows


def _lime_budget_rows(
    dataset: str,
    task: str,
    model_name: str,
    model,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    preprocessor: TabularPreprocessor,
    classes: list[str],
    seed: int,
    budgets: tuple[int, ...] = (100, 350, 700),
    repeats: int = 5,
    n: int = 6,
) -> list[dict]:
    prediction = model.predict(x_test).astype(int)
    indices = explanation_indices(y_test, prediction, min(n, len(y_test)), seed)
    background_indices = stratified_indices(
        model.predict(x_train), min(5000, len(x_train)), seed
    )
    background = x_train[background_indices]
    explainers = [
        make_lime_explainer(background, preprocessor, classes, repeat_seed)
        for repeat_seed in range(seed, seed + repeats)
    ]
    rows = []
    for source_index in indices:
        row = x_test[int(source_index)]
        predicted_class = int(prediction[int(source_index)])
        for budget in budgets:
            vectors = []
            latencies = []
            fidelity_scores = []
            for explainer in explainers:
                start = time.perf_counter_ns()
                vector = explain_lime(
                    explainer,
                    row,
                    model.predict_proba,
                    predicted_class,
                    len(row),
                    budget,
                )
                latencies.append((time.perf_counter_ns() - start) / 1_000_000)
                vectors.append(vector)
                ranking = np.argsort(-np.abs(vector))
                score, _ = _ranking_score(
                    model,
                    row,
                    predicted_class,
                    np.asarray(preprocessor.baseline_, dtype=float),
                    ranking,
                    [1, 3, 5, 10],
                )
                fidelity_scores.append(score)
            similarities = [
                safe_cosine(vectors[a], vectors[b])
                for a in range(len(vectors))
                for b in range(a + 1, len(vectors))
            ]
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "model": model_name,
                    "source_test_index": int(source_index),
                    "lime_samples": budget,
                    "mean_deletion_fidelity": float(np.mean(fidelity_scores)),
                    "mean_self_stability_cosine": float(np.mean(similarities)),
                    "median_latency_ms": float(np.median(latencies)),
                }
            )
    return rows


def _summarize_existing_outputs(output_dir: Path, extended_dir: Path) -> None:
    fidelity = pd.read_csv(output_dir / "fidelity.csv")
    top = fidelity[fidelity["control"] == "top"].copy()
    per_instance = (
        top.groupby(
            ["dataset", "task", "model", "explainer", "explanation_id", "actual_class", "predicted_class", "correct"],
            as_index=False,
        )["probability_drop"].mean()
    )
    class_summary = (
        per_instance.groupby(["dataset", "task", "model", "explainer", "actual_class"], as_index=False)
        .agg(n=("explanation_id", "size"), mean_fidelity=("probability_drop", "mean"), median_fidelity=("probability_drop", "median"))
    )
    class_summary.to_csv(extended_dir / "class_specific_fidelity.csv", index=False)

    def error_type(row):
        if row.task != "binary":
            return "correct" if row.correct else "incorrect"
        actual_attack = row.actual_class == "Attack"
        predicted_attack = row.predicted_class == "Attack"
        return "TP" if actual_attack and predicted_attack else "FN" if actual_attack else "FP" if predicted_attack else "TN"

    per_instance["error_type"] = per_instance.apply(error_type, axis=1)
    error_summary = (
        per_instance.groupby(["dataset", "task", "model", "explainer", "error_type"], as_index=False)
        .agg(n=("explanation_id", "size"), mean_fidelity=("probability_drop", "mean"), median_fidelity=("probability_drop", "median"))
    )
    error_summary.to_csv(extended_dir / "error_type_fidelity.csv", index=False)

    stability = pd.read_csv(output_dir / "stability.csv")
    stability_summary = (
        stability.groupby(["dataset", "task", "model", "explainer"], as_index=False)
        .agg(
            n=("cosine_similarity", "count"),
            cosine_q10=("cosine_similarity", lambda x: x.quantile(0.10)),
            cosine_median=("cosine_similarity", "median"),
            cosine_q90=("cosine_similarity", lambda x: x.quantile(0.90)),
            jaccard_q10=("top5_jaccard", lambda x: x.quantile(0.10)),
            jaccard_median=("top5_jaccard", "median"),
            jaccard_q90=("top5_jaccard", lambda x: x.quantile(0.90)),
        )
    )
    stability_summary.to_csv(extended_dir / "stability_distributions.csv", index=False)


def _make_plots(extended_dir: Path) -> None:
    figures = extended_dir / "figures"
    figures.mkdir(exist_ok=True)
    calibration = pd.read_csv(extended_dir / "calibration_bins.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), sharex=True, sharey=True)
    for ax, dataset in zip(axes, sorted(calibration.dataset.unique())):
        subset = calibration[calibration.dataset == dataset]
        for model_name, group in subset.groupby("model"):
            ax.plot(group.mean_attack_probability, group.observed_attack_frequency, marker="o", label=model_name)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(title=dataset, xlabel="Predicted attack probability", ylabel="Observed attack frequency")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "calibration_audit.png", dpi=240)
    plt.close(fig)

    masking = pd.read_csv(extended_dir / "masking_baseline_ablation.csv")
    summary = masking.groupby("baseline", as_index=False).deletion_score.mean().sort_values("deletion_score")
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.barh(summary.baseline, summary.deletion_score, color="#39739d")
    ax.set(xlabel="Mean TreeSHAP top-k deletion score", ylabel="", title="Masking baseline sensitivity")
    fig.tight_layout()
    fig.savefig(figures / "masking_baseline_audit.png", dpi=240)
    plt.close(fig)

    lime = pd.read_csv(extended_dir / "lime_budget_ablation.csv")
    summary = lime.groupby("lime_samples", as_index=False).agg(
        fidelity=("mean_deletion_fidelity", "mean"),
        stability=("mean_self_stability_cosine", "mean"),
        latency=("median_latency_ms", "median"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    for ax, column, label in zip(
        axes,
        ("fidelity", "stability", "latency"),
        ("Deletion fidelity", "Self-stability cosine", "Median latency (ms)"),
    ):
        ax.plot(summary.lime_samples, summary[column], marker="o", color="#d07c43")
        ax.set(xlabel="LIME samples", ylabel=label)
    fig.tight_layout()
    fig.savefig(figures / "lime_budget_audit.png", dpi=240)
    plt.close(fig)


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    output_dir = Path(config["output_dir"])
    extended_dir = output_dir / "extended"
    extended_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = []
    balance_rows = []
    calibration_rows = []
    calibration_summaries = []
    masking_rows = []
    lime_rows = []

    for dataset in ("CIC-IDS2017", "UNSW-NB15"):
        for task in ("binary", "multiclass"):
            print(f"[audit prepare] {dataset} / {task}", flush=True)
            train, test, y_train, y_test, classes, metadata = prepare_dataset(dataset, task, config, seed)
            feature_names = metadata["feature_names"]
            first_artifact = joblib.load(_artifact_path(output_dir, dataset, task, "random_forest"))
            preprocessor: TabularPreprocessor = first_artifact["preprocessor"]
            x_train = preprocessor.transform(train)
            x_test = preprocessor.transform(test)

            class_distribution = np.bincount(y_train, minlength=len(classes)) / len(y_train)
            majority = int(np.argmax(class_distribution))
            majority_prediction = np.full(len(y_test), majority, dtype=int)
            majority_probabilities = np.tile(class_distribution, (len(y_test), 1))
            metrics, _, _ = predictive_metrics(y_test, majority_prediction, majority_probabilities, classes)
            baseline_rows.append({"dataset": dataset, "task": task, "baseline": "majority", **metrics})

            shallow = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=seed)
            shallow.fit(x_train, y_train)
            prediction = shallow.predict(x_test)
            probabilities = shallow.predict_proba(x_test)
            metrics, _, _ = predictive_metrics(y_test, prediction, probabilities, classes)
            baseline_rows.append({"dataset": dataset, "task": task, "baseline": "shallow_tree_depth5", **metrics})

            for model_name in ("random_forest", "xgboost"):
                print(f"[audit] {dataset} / {task} / {model_name}", flush=True)
                artifact = joblib.load(_artifact_path(output_dir, dataset, task, model_name))
                model = artifact["model"]
                if task == "binary":
                    rows, summary = _calibration_rows(dataset, model_name, model, x_test, y_test, classes)
                    calibration_rows.extend(rows)
                    calibration_summaries.append(summary)

                # Hold the stochastic seed fixed so this ablation changes only
                # the weighting treatment rather than weighting and RNG state.
                unweighted = make_model(model_name, config["models"], seed, len(classes))
                if model_name == "random_forest":
                    unweighted.set_params(class_weight=None)
                unweighted.fit(x_train, y_train)
                unweighted_prediction = unweighted.predict(x_test)
                balance_rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "model": model_name,
                        "training_weighting": "none",
                        "seed": seed,
                        "macro_f1": float(f1_score(y_test, unweighted_prediction, average="macro", zero_division=0)),
                    }
                )
                weighted_prediction = model.predict(x_test)
                balance_rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "model": model_name,
                        "training_weighting": "class_weighted",
                        "seed": seed,
                        "macro_f1": float(f1_score(y_test, weighted_prediction, average="macro", zero_division=0)),
                    }
                )

                masking_rows.extend(
                    _masking_baseline_rows(
                        dataset, task, model_name, model, x_train, x_test, y_test, preprocessor, seed
                    )
                )
                lime_rows.extend(
                    _lime_budget_rows(
                        dataset, task, model_name, model, x_train, x_test, y_test, preprocessor, classes, seed
                    )
                )

    pd.DataFrame(baseline_rows).to_csv(extended_dir / "predictive_baselines.csv", index=False)
    pd.DataFrame(balance_rows).to_csv(extended_dir / "balancing_ablation.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(extended_dir / "calibration_bins.csv", index=False)
    pd.DataFrame(calibration_summaries).to_csv(extended_dir / "calibration_summary.csv", index=False)
    pd.DataFrame(masking_rows).to_csv(extended_dir / "masking_baseline_ablation.csv", index=False)
    pd.DataFrame(lime_rows).to_csv(extended_dir / "lime_budget_ablation.csv", index=False)
    _summarize_existing_outputs(output_dir, extended_dir)

    controls = pd.read_csv(output_dir / "fidelity_control_tests.csv")
    paired = pd.read_csv(output_dir / "paired_comparisons.csv")
    coverage = pd.DataFrame(
        [
            ("Modern dual-dataset evaluation", "complete", "CIC-IDS2017 and official UNSW-NB15 split"),
            ("Binary and multiclass tasks", "complete", "Both tasks on both datasets"),
            ("Simple predictive baselines", "complete", "Majority and depth-5 decision tree"),
            ("Five predictive seeds", "complete", "Seeds 42, 52, 62, 72, and 82"),
            ("Leakage-safe preprocessing", "complete", "Training-only medians and category dictionaries"),
            ("Random and bottom fidelity controls", "complete", "Paired deletion controls with Holm correction"),
            ("Alternative masking baselines", "complete", "Median/mode, zero, marginal row, and nearest same/opposite predictions"),
            ("LIME sampling sensitivity", "complete", "100, 350, and 700 samples with five repeats"),
            ("Explanation by class and error type", "complete", "Class and TP/FP/TN/FN summaries"),
            ("Stability distributions", "complete", "Q10, median, and Q90 for cosine and top-5 overlap"),
            ("Probability calibration audit", "complete", "Brier score, ECE, and reliability curves"),
            ("Balancing sensitivity", "partial", "Class weighting versus none; SMOTE omitted to avoid synthetic categorical flows"),
            ("Human analyst usefulness", "future work", "Requires ethics-approved task-based user study"),
            ("Causal explanation", "out of scope", "Post-hoc model evidence cannot establish network causality"),
        ],
        columns=["literature_requirement", "status", "evidence"],
    )
    coverage.to_csv(extended_dir / "literature_purpose_coverage.csv", index=False)
    summary = {
        "tree_shap_random_controls_passed": int(
            ((controls.explainer == "TreeSHAP") & (controls.holm_adjusted_p < 0.05) & (controls.ci_low > 0)).sum()
        ),
        "tree_shap_random_controls_total": int((controls.explainer == "TreeSHAP").sum()),
        "paired_tree_shap_superiority_intervals_above_zero": int((paired.ci_low > 0).sum()),
        "paired_comparisons_total": len(paired),
        "claim": "Opaque tree-IDS decisions are converted into auditable and rejectable model evidence; causal and universal interpretability are not claimed.",
    }
    (extended_dir / "black_box_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _make_plots(extended_dir)
    print(f"[audit done] {extended_dir.resolve()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run literature-driven robustness audits.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
