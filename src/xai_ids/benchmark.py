from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lime.lime_tabular
import matplotlib
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import shap
import sklearn
import xgboost
import yaml
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class TabularPreprocessor:
    feature_names: list[str]
    categorical_columns: list[str] = field(default_factory=list)
    numeric_medians: dict[str, float] = field(default_factory=dict)
    category_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    categorical_names: dict[int, list[str]] = field(default_factory=dict)
    baseline_: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        self.categorical_columns = [
            c
            for c in self.feature_names
            if pd.api.types.is_object_dtype(frame[c])
            or isinstance(frame[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(frame[c])
        ]
        baseline = []
        for idx, column in enumerate(self.feature_names):
            if column in self.categorical_columns:
                values = frame[column].astype("string").fillna("<MISSING>").astype(str)
                categories = sorted(values.unique().tolist())
                mapping = {value: i for i, value in enumerate(categories)}
                self.category_maps[column] = mapping
                self.categorical_names[idx] = categories
                mode = values.mode(dropna=False)
                baseline.append(float(mapping[mode.iloc[0]]))
            else:
                values = pd.to_numeric(frame[column], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                median = float(values.median()) if values.notna().any() else 0.0
                self.numeric_medians[column] = median
                baseline.append(median)
        self.baseline_ = np.asarray(baseline, dtype=np.float64)
        return self

    @property
    def categorical_indices(self) -> list[int]:
        return [
            i for i, name in enumerate(self.feature_names) if name in self.categorical_columns
        ]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns = []
        for column in self.feature_names:
            if column in self.categorical_columns:
                mapping = self.category_maps[column]
                values = (
                    frame[column]
                    .astype("string")
                    .fillna("<MISSING>")
                    .astype(str)
                    .map(mapping)
                    .fillna(-1)
                    .astype(float)
                    .to_numpy()
                )
            else:
                values = (
                    pd.to_numeric(frame[column], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(self.numeric_medians[column])
                    .astype(float)
                    .to_numpy()
                )
            columns.append(values)
        return np.column_stack(columns).astype(np.float64, copy=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_cic_label(value: Any) -> str:
    return (
        str(value)
        .strip()
        .replace("�", "-")
        .replace("  ", " ")
    )


def load_cic(config: dict[str, Any], seed: int) -> pd.DataFrame:
    pieces = []
    label_column = config["label_column"]
    cap = int(config["per_class_cap"])
    for shard_index, shard in enumerate(config["shards"]):
        frame = pd.read_parquet(shard)
        frame.columns = [str(c).strip() for c in frame.columns]
        frame[label_column] = frame[label_column].map(clean_cic_label)
        for label, group in frame.groupby(label_column, sort=True):
            n = min(cap, len(group))
            pieces.append(group.sample(n=n, random_state=seed + shard_index))
    frame = pd.concat(pieces, ignore_index=True)
    # A second cap is needed because BENIGN occurs in both Parquet shards.
    capped = []
    for label, group in frame.groupby(label_column, sort=True):
        capped.append(group.sample(n=min(cap, len(group)), random_state=seed))
    frame = pd.concat(capped, ignore_index=True)
    before = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    frame.attrs["duplicates_removed"] = before - len(frame)
    return frame


def clean_unsw_partitions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    label_column: str,
    multiclass_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Remove within-partition duplicates and test vectors seen in training.

    Targets are deliberately excluded from the overlap key: the same network
    flow must not occur on both sides of evaluation even if its supplied label
    differs. The returned audit makes every removal visible in the run manifest.
    """
    original_train_rows = len(train)
    original_test_rows = len(test)
    train = train.drop_duplicates().reset_index(drop=True)
    test = test.drop_duplicates().reset_index(drop=True)
    train_duplicates_removed = original_train_rows - len(train)
    test_duplicates_removed = original_test_rows - len(test)

    # The accessible preprocessed mirror contains exact feature vectors in both
    # predefined partitions. Remove every overlapping test vector so an input
    # seen during fitting cannot reappear during evaluation, even when its
    # supplied label differs. Hashing is used only for membership detection;
    # the original rows and dtypes are retained.
    target_columns = {label_column, multiclass_column}
    feature_columns = [column for column in train.columns if column not in target_columns]
    train_feature_hashes = set(
        pd.util.hash_pandas_object(train[feature_columns], index=False).astype("uint64")
    )
    test_feature_hashes = pd.util.hash_pandas_object(
        test[feature_columns], index=False
    ).astype("uint64")
    overlap = test_feature_hashes.isin(train_feature_hashes)
    cross_partition_feature_overlaps_removed = int(overlap.sum())
    test = test.loc[~overlap].reset_index(drop=True)

    audit = {
        "original_training_rows": original_train_rows,
        "original_testing_rows": original_test_rows,
        "training_exact_duplicates_removed": train_duplicates_removed,
        "testing_exact_duplicates_removed": test_duplicates_removed,
        "cross_partition_feature_overlaps_removed_from_test": cross_partition_feature_overlaps_removed,
        "clean_training_rows": len(train),
        "clean_testing_rows": len(test),
    }
    return train, test, audit


def load_unsw(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    train = pd.read_parquet(config["train"])
    test = pd.read_parquet(config["test"])
    if len(train) != 175341 or len(test) != 82332:
        raise ValueError(
            "UNSW standard split row counts are not 175,341 train / 82,332 test."
        )
    for frame in (train, test):
        frame.columns = [str(c).strip() for c in frame.columns]
        frame[config["multiclass_column"]] = (
            frame[config["multiclass_column"]]
            .fillna(config["normal_label"])
            .astype(str)
            .str.strip()
            .replace({"": config["normal_label"]})
        )

    return clean_unsw_partitions(
        train,
        test,
        config["label_column"],
        config["multiclass_column"],
    )


def stratified_indices(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    if n >= len(labels):
        return np.arange(len(labels))
    classes, counts = np.unique(labels, return_counts=True)
    if n < len(classes):
        selected_classes = rng.choice(
            classes, size=n, replace=False, p=counts / counts.sum()
        )
        return np.asarray(
            [rng.choice(np.flatnonzero(labels == label)) for label in selected_classes],
            dtype=int,
        )
    chosen: list[int] = []
    remaining = n
    for class_index, (label, count) in enumerate(zip(classes, counts)):
        if class_index == len(classes) - 1:
            take = remaining
        else:
            take = max(1, int(round(n * count / len(labels))))
            take = min(take, count, remaining - (len(classes) - class_index - 1))
        options = np.flatnonzero(labels == label)
        chosen.extend(rng.choice(options, size=take, replace=False).tolist())
        remaining -= take
    if remaining > 0:
        pool = np.setdiff1d(np.arange(len(labels)), np.asarray(chosen), assume_unique=False)
        chosen.extend(rng.choice(pool, size=remaining, replace=False).tolist())
    return np.asarray(chosen[:n], dtype=int)


def explanation_indices(
    y_true: np.ndarray, y_pred: np.ndarray, n: int, seed: int
) -> np.ndarray:
    """Select label-stratified flows while covering available binary error types."""
    selected = stratified_indices(y_true, min(n, len(y_true)), seed).tolist()
    if len(np.unique(y_true)) != 2:
        if np.any(y_true != y_pred) and not np.any(y_true[selected] != y_pred[selected]):
            candidates = np.flatnonzero(y_true != y_pred)
            selected[-1] = int(np.random.default_rng(seed).choice(candidates))
        return np.asarray(list(dict.fromkeys(selected)), dtype=int)

    rng = np.random.default_rng(seed)
    required = []
    for actual in np.unique(y_true):
        for predicted in np.unique(y_pred):
            candidates = np.flatnonzero((y_true == actual) & (y_pred == predicted))
            if len(candidates):
                required.append(int(rng.choice(candidates)))
    required_set = set(required)
    merged = required + [idx for idx in selected if idx not in required_set]
    if len(merged) < min(n, len(y_true)):
        pool = np.setdiff1d(np.arange(len(y_true)), np.asarray(merged), assume_unique=False)
        merged.extend(rng.choice(pool, size=min(n, len(y_true)) - len(merged), replace=False).tolist())
    return np.asarray(merged[: min(n, len(y_true))], dtype=int)


def split_cic(
    frame: pd.DataFrame, labels: np.ndarray, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(frame))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )
    return (
        frame.iloc[train_idx].reset_index(drop=True),
        frame.iloc[test_idx].reset_index(drop=True),
        labels[train_idx],
        labels[test_idx],
    )


def make_model(name: str, config: dict[str, Any], seed: int, n_classes: int):
    if name == "random_forest":
        return RandomForestClassifier(
            **config[name],
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "xgboost":
        params = dict(config[name])
        params.update(
            random_state=seed,
            n_jobs=max(1, min(8, os.cpu_count() or 1)),
            tree_method="hist",
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
        )
        return XGBClassifier(**params)
    raise KeyError(name)


def predictive_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray, classes: list[str]
) -> tuple[dict[str, float], list[dict[str, Any]], np.ndarray]:
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    if len(classes) == 2:
        if "Attack" not in classes or "Normal" not in classes:
            raise ValueError("Binary IDS metrics require named Attack and Normal classes.")
        attack_index = classes.index("Attack")
        normal_index = classes.index("Normal")
        tp = matrix[attack_index, attack_index]
        fn = matrix[attack_index, normal_index]
        fp = matrix[normal_index, attack_index]
        tn = matrix[normal_index, normal_index]
        result["false_positive_rate"] = float(fp / max(1, fp + tn))
        result["false_negative_rate"] = float(fn / max(1, fn + tp))
        result["pr_auc"] = float(
            average_precision_score((y_true == attack_index).astype(int), probabilities[:, attack_index])
        )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(classes)), zero_division=0
    )
    per_class = [
        {
            "class": classes[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(classes))
    ]
    return result, per_class, matrix


def extract_shap_vector(explanation: shap.Explanation, predicted_class: int) -> np.ndarray:
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        return values[0, :, predicted_class].astype(float)
    if values.ndim == 2:
        vector = values[0].astype(float)
        return vector if predicted_class == 1 else -vector
    raise ValueError(f"Unsupported SHAP value shape: {values.shape}")


def explain_lime(
    explainer: lime.lime_tabular.LimeTabularExplainer,
    row: np.ndarray,
    predict_fn,
    predicted_class: int,
    n_features: int,
    n_samples: int,
) -> np.ndarray:
    explanation = explainer.explain_instance(
        row,
        predict_fn,
        labels=(int(predicted_class),),
        num_features=n_features,
        num_samples=n_samples,
    )
    vector = np.zeros(n_features, dtype=float)
    for index, weight in explanation.local_exp[int(predicted_class)]:
        vector[int(index)] = float(weight)
    return vector


def make_lime_explainer(
    x_train: np.ndarray,
    preprocessor: TabularPreprocessor,
    class_names: list[str],
    seed: int,
):
    return lime.lime_tabular.LimeTabularExplainer(
        x_train,
        feature_names=preprocessor.feature_names,
        class_names=class_names,
        categorical_features=preprocessor.categorical_indices,
        categorical_names=preprocessor.categorical_names,
        discretize_continuous=True,
        mode="classification",
        random_state=seed,
        sample_around_instance=True,
    )


def fidelity_rows(
    model,
    row: np.ndarray,
    attribution: np.ndarray,
    predicted_class: int,
    baseline: np.ndarray,
    k_values: list[int],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    original_probability = float(model.predict_proba(row.reshape(1, -1))[0, predicted_class])
    ranked = np.argsort(-np.abs(attribution))
    bottom = ranked[::-1]
    random_rank = rng.permutation(len(ranked))
    results = []
    for k in k_values:
        k_eff = min(int(k), len(row))
        for control, indices in (
            ("top", ranked[:k_eff]),
            ("bottom", bottom[:k_eff]),
            ("random", random_rank[:k_eff]),
        ):
            perturbed = row.copy()
            perturbed[indices] = baseline[indices]
            probabilities = model.predict_proba(perturbed.reshape(1, -1))[0]
            new_prediction = int(np.argmax(probabilities))
            results.append(
                {
                    "k": int(k_eff),
                    "control": control,
                    "probability_drop": original_probability
                    - float(probabilities[predicted_class]),
                    "flipped": int(new_prediction != predicted_class),
                }
            )
    return results


def safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    if np.linalg.norm(left) == 0 and np.linalg.norm(right) == 0:
        return 1.0
    if np.linalg.norm(left) == 0 or np.linalg.norm(right) == 0:
        return 0.0
    return float(1.0 - cosine(left, right))


def nearest_neighbor_stability(
    x: np.ndarray, predicted: np.ndarray, attributions: np.ndarray, k: int = 5
) -> list[dict[str, float]]:
    scaled = StandardScaler().fit_transform(x)
    rows = []
    for i in range(len(x)):
        candidates = np.flatnonzero(predicted == predicted[i])
        candidates = candidates[candidates != i]
        if not len(candidates):
            continue
        distances = np.linalg.norm(scaled[candidates] - scaled[i], axis=1)
        j = int(candidates[np.argmin(distances)])
        top_i = set(np.argsort(-np.abs(attributions[i]))[: min(k, x.shape[1])])
        top_j = set(np.argsort(-np.abs(attributions[j]))[: min(k, x.shape[1])])
        rows.append(
            {
                "cosine_similarity": safe_cosine(attributions[i], attributions[j]),
                "top5_jaccard": float(len(top_i & top_j) / max(1, len(top_i | top_j))),
                "input_distance": float(np.linalg.norm(scaled[i] - scaled[j])),
            }
        )
    return rows


def bootstrap_mean_ci(values: np.ndarray, seed: int, repetitions: int = 2000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def evaluate_explainer(
    dataset: str,
    task: str,
    model_name: str,
    explainer_name: str,
    model,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    preprocessor: TabularPreprocessor,
    classes: list[str],
    config: dict[str, Any],
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], dict[str, Any]]:
    n = int(config["explanation_samples"])
    full_prediction = model.predict(x_test).astype(int)
    indices = explanation_indices(y_test, full_prediction, min(n, len(y_test)), seed)
    selected_x = x_test[indices]
    selected_y = y_test[indices]
    predicted = model.predict(selected_x).astype(int)
    n_features = selected_x.shape[1]
    k_values = list(config["fidelity_k"])
    rng = np.random.default_rng(seed)
    attribution_vectors = []
    attribution_rows: list[dict] = []
    fidelity: list[dict] = []
    speed_ms = []

    if explainer_name == "TreeSHAP":
        explainer = shap.TreeExplainer(model)
        _ = explainer(selected_x[:1], check_additivity=False)
    else:
        background_idx = stratified_indices(
            model.predict(x_train), min(5000, len(x_train)), seed
        )
        explainer = make_lime_explainer(
            x_train[background_idx], preprocessor, classes, seed
        )
        _ = explain_lime(
            explainer,
            selected_x[0],
            model.predict_proba,
            int(predicted[0]),
            n_features,
            int(config["lime_samples"]),
        )

    for local_index, (source_index, row, actual, prediction) in enumerate(
        zip(indices, selected_x, selected_y, predicted)
    ):
        start = time.perf_counter_ns()
        if explainer_name == "TreeSHAP":
            vector = extract_shap_vector(
                explainer(row.reshape(1, -1), check_additivity=False), int(prediction)
            )
        else:
            vector = explain_lime(
                explainer,
                row,
                model.predict_proba,
                int(prediction),
                n_features,
                int(config["lime_samples"]),
            )
        elapsed = (time.perf_counter_ns() - start) / 1_000_000
        speed_ms.append(elapsed)
        attribution_vectors.append(vector)
        confidence = float(model.predict_proba(row.reshape(1, -1))[0, prediction])
        common = {
            "dataset": dataset,
            "task": task,
            "model": model_name,
            "explainer": explainer_name,
            "explanation_id": int(local_index),
            "source_test_index": int(source_index),
            "actual_class": classes[int(actual)],
            "predicted_class": classes[int(prediction)],
            "correct": int(actual == prediction),
            "confidence": confidence,
        }
        for feature_name, value in zip(preprocessor.feature_names, vector):
            attribution_rows.append(
                {**common, "feature": feature_name, "attribution": float(value)}
            )
        for item in fidelity_rows(
            model,
            row,
            vector,
            int(prediction),
            np.asarray(preprocessor.baseline_),
            k_values,
            rng,
        ):
            fidelity.append({**common, **item})

    attribution_array = np.asarray(attribution_vectors)
    stability = [
        {
            "dataset": dataset,
            "task": task,
            "model": model_name,
            "explainer": explainer_name,
            **row,
        }
        for row in nearest_neighbor_stability(selected_x, predicted, attribution_array)
    ]

    self_stability_values: list[float] = []
    if explainer_name == "LIME":
        repeat_n = min(int(config["self_stability_samples"]), len(selected_x))
        for row, prediction in zip(selected_x[:repeat_n], predicted[:repeat_n]):
            variants = []
            repeat_count = int(config.get("lime_repeat_seeds", 3))
            for repeat_seed in range(seed, seed + repeat_count):
                local_explainer = make_lime_explainer(
                    x_train[stratified_indices(model.predict(x_train), min(5000, len(x_train)), repeat_seed)],
                    preprocessor,
                    classes,
                    repeat_seed,
                )
                variants.append(
                    explain_lime(
                        local_explainer,
                        row,
                        model.predict_proba,
                        int(prediction),
                        n_features,
                        int(config["lime_samples"]),
                    )
                )
            self_stability_values.extend(
                safe_cosine(variants[a], variants[b])
                for a in range(len(variants))
                for b in range(a + 1, len(variants))
            )

    top_fidelity = pd.DataFrame(fidelity)
    top_fidelity = top_fidelity[top_fidelity["control"] == "top"]
    per_instance = top_fidelity.groupby("explanation_id")["probability_drop"].mean()
    fidelity_mean, fidelity_low, fidelity_high = bootstrap_mean_ci(
        per_instance.to_numpy(), seed
    )
    random_mean = float(
        pd.DataFrame(fidelity)
        .query("control == 'random'")
        .groupby("explanation_id")["probability_drop"]
        .mean()
        .mean()
    )
    bottom_mean = float(
        pd.DataFrame(fidelity)
        .query("control == 'bottom'")
        .groupby("explanation_id")["probability_drop"]
        .mean()
        .mean()
    )
    stability_frame = pd.DataFrame(stability)
    stability_cosine = (
        float(stability_frame["cosine_similarity"].mean())
        if "cosine_similarity" in stability_frame
        else np.nan
    )
    stability_jaccard = (
        float(stability_frame["top5_jaccard"].mean())
        if "top5_jaccard" in stability_frame
        else np.nan
    )
    summary = {
        "dataset": dataset,
        "task": task,
        "model": model_name,
        "explainer": explainer_name,
        "n_explanations": len(selected_x),
        "fidelity_mean": fidelity_mean,
        "fidelity_ci_low": fidelity_low,
        "fidelity_ci_high": fidelity_high,
        "random_fidelity_mean": random_mean,
        "bottom_fidelity_mean": bottom_mean,
        "fidelity_lift_vs_random": fidelity_mean - random_mean,
        "flip_rate": float(top_fidelity["flipped"].mean()),
        "stability_cosine_mean": stability_cosine,
        "stability_jaccard_mean": stability_jaccard,
        "self_stability_cosine_mean": float(np.mean(self_stability_values))
        if self_stability_values
        else np.nan,
        "latency_median_ms": float(np.median(speed_ms)),
        "latency_p90_ms": float(np.percentile(speed_ms, 90)),
        "latency_p95_ms": float(np.percentile(speed_ms, 95)),
        "latency_p99_ms": float(np.percentile(speed_ms, 99)),
        "throughput_per_second": float(1000.0 / max(np.mean(speed_ms), 1e-12)),
    }
    return attribution_rows, fidelity, stability, summary


def paired_bootstrap_comparisons(fidelity: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    top = fidelity[fidelity["control"] == "top"]
    keys = ["dataset", "task", "model"]
    for group_key, group in top.groupby(keys):
        per = (
            group.groupby(["explainer", "source_test_index"])["probability_drop"]
            .mean()
            .unstack("explainer")
            .dropna()
        )
        if not {"TreeSHAP", "LIME"}.issubset(per.columns) or per.empty:
            continue
        differences = (per["TreeSHAP"] - per["LIME"]).to_numpy()
        mean, low, high = bootstrap_mean_ci(differences, seed)
        if np.allclose(differences, 0):
            p_value = 1.0
        else:
            p_value = float(
                wilcoxon(differences, alternative="greater", zero_method="wilcox").pvalue
            )
        rows.append(
            dict(
                zip(keys, group_key),
                n_pairs=len(differences),
                mean_difference_shap_minus_lime=mean,
                median_difference_shap_minus_lime=float(np.median(differences)),
                paired_effect_dz=float(mean / np.std(differences, ddof=1))
                if len(differences) > 1 and np.std(differences, ddof=1) > 0
                else np.nan,
                ci_low=low,
                ci_high=high,
                wilcoxon_one_sided_p=p_value,
            )
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        order = np.argsort(result["wilcoxon_one_sided_p"].to_numpy())
        adjusted = np.empty(len(result), dtype=float)
        running = 0.0
        for rank, index in enumerate(order):
            value = min(1.0, (len(result) - rank) * result.iloc[index]["wilcoxon_one_sided_p"])
            running = max(running, value)
            adjusted[index] = running
        result["holm_adjusted_p"] = adjusted
    return result


def fidelity_control_tests(fidelity: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "task", "model", "explainer"]
    for group_key, group in fidelity.groupby(keys):
        per = (
            group.groupby(["source_test_index", "control"])["probability_drop"]
            .mean()
            .unstack("control")
            .dropna()
        )
        if not {"top", "random"}.issubset(per.columns) or per.empty:
            continue
        differences = (per["top"] - per["random"]).to_numpy()
        mean, low, high = bootstrap_mean_ci(differences, seed)
        p_value = (
            1.0
            if np.allclose(differences, 0)
            else float(wilcoxon(differences, alternative="greater", zero_method="wilcox").pvalue)
        )
        rows.append(
            dict(
                zip(keys, group_key),
                n_pairs=len(differences),
                mean_lift_top_minus_random=mean,
                ci_low=low,
                ci_high=high,
                wilcoxon_one_sided_p=p_value,
            )
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        order = np.argsort(result["wilcoxon_one_sided_p"].to_numpy())
        adjusted = np.empty(len(result), dtype=float)
        running = 0.0
        for rank, index in enumerate(order):
            value = min(1.0, (len(result) - rank) * result.iloc[index]["wilcoxon_one_sided_p"])
            running = max(running, value)
            adjusted[index] = running
        result["holm_adjusted_p"] = adjusted
    return result


def feature_ablation(
    dataset: str,
    task: str,
    model_name: str,
    model_config: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    ranking: np.ndarray,
    k_values: list[int],
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for k in [*k_values, len(feature_names)]:
        k_eff = min(k, len(feature_names))
        selected = ranking[:k_eff]
        model = make_model(model_name, model_config, seed, len(np.unique(y_train)))
        sample_weight = (
            compute_sample_weight("balanced", y_train) if model_name == "xgboost" else None
        )
        start = time.perf_counter()
        model.fit(x_train[:, selected], y_train, sample_weight=sample_weight)
        training_seconds = time.perf_counter() - start
        prediction = model.predict(x_test[:, selected])
        explainer = shap.TreeExplainer(model)
        probe = x_test[: min(10, len(x_test)), selected]
        start = time.perf_counter_ns()
        _ = explainer(probe, check_additivity=False)
        shap_ms = (time.perf_counter_ns() - start) / 1_000_000 / max(1, len(probe))
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "model": model_name,
                "feature_count": int(k_eff),
                "macro_f1": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
                "weighted_f1": float(
                    f1_score(y_test, prediction, average="weighted", zero_division=0)
                ),
                "training_seconds": float(training_seconds),
                "shap_latency_ms": float(shap_ms),
                "selected_features": " | ".join(feature_names[i] for i in selected),
            }
        )
    return rows


def save_confusion_plot(
    matrix: np.ndarray, classes: list[str], title: str, path: Path
) -> None:
    size = max(5.0, min(9.0, 4.0 + len(classes) * 0.45))
    fig, ax = plt.subplots(figsize=(size, size * 0.82))
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    sns.heatmap(
        normalized,
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Row-normalized proportion"},
        ax=ax,
    )
    ax.set(title=title, xlabel="Predicted", ylabel="Actual")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_summary_plots(output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    predictive = pd.read_csv(output_dir / "predictive_metrics.csv")
    xai = pd.read_csv(output_dir / "explanation_summary.csv")
    ablation = pd.read_csv(output_dir / "feature_ablation.csv")
    attributions = pd.read_csv(output_dir / "attributions.csv")

    pred_summary = (
        predictive.groupby(["dataset", "task", "model"], as_index=False)
        .agg(macro_f1=("macro_f1", "mean"), sd=("macro_f1", "std"))
    )
    datasets = sorted(pred_summary["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(9.0, 3.8), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        sns.barplot(
            data=pred_summary[pred_summary["dataset"] == dataset],
            x="task",
            y="macro_f1",
            hue="model",
            ax=ax,
        )
        ax.set_ylim(0, 1.03)
        ax.set_ylabel("Macro-F1 (mean over 5 seeds)")
        ax.set_xlabel("")
        ax.set_title(dataset)
    fig.tight_layout()
    fig.savefig(figures / "predictive_performance.png", dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    sns.barplot(data=xai, x="task", y="fidelity_lift_vs_random", hue="explainer", ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(title="Deletion fidelity lift (datasets/models pooled)", xlabel="", ylabel="Probability-drop lift")
    sns.barplot(data=xai, x="task", y="stability_cosine_mean", hue="explainer", ax=axes[1])
    axes[1].set(title="Neighborhood stability (datasets/models pooled)", xlabel="", ylabel="Cosine similarity", ylim=(-0.1, 1.0))
    if axes[1].legend_:
        axes[1].legend_.remove()
    fig.tight_layout()
    fig.savefig(figures / "explanation_quality.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    sns.boxplot(data=xai, x="dataset", y="latency_median_ms", hue="explainer", ax=ax)
    ax.set_yscale("log")
    ax.set(title="Explanation latency", xlabel="", ylabel="Median latency (ms, log scale)")
    fig.tight_layout()
    fig.savefig(figures / "explanation_latency.png", dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), sharey=True)
    for ax, dataset in zip(axes, sorted(ablation["dataset"].unique())):
        subset = ablation[ablation["dataset"] == dataset]
        sns.lineplot(
            data=subset,
            x="feature_count",
            y="macro_f1",
            hue="model",
            style="task",
            markers=True,
            dashes=False,
            ax=ax,
        )
        ax.set(title=dataset, xlabel="Features", ylabel="Macro-F1")
        ax.set_ylim(0, 1.03)
    fig.tight_layout()
    fig.savefig(figures / "feature_ablation.png", dpi=240)
    plt.close(fig)

    subset = attributions[
        (attributions["task"] == "binary") & (attributions["model"] == "xgboost")
    ].copy()
    subset["absolute_attribution"] = subset["attribution"].abs()
    global_imp = (
        subset.groupby(["dataset", "explainer", "feature"], as_index=False)[
            "absolute_attribution"
        ]
        .mean()
    )
    datasets = sorted(global_imp["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(10.0, 4.6))
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        data = global_imp[global_imp["dataset"] == dataset]
        top_features = (
            data.groupby("feature")["absolute_attribution"].mean().nlargest(8).index
        )
        plot_data = data[data["feature"].isin(top_features)]
        sns.barplot(
            data=plot_data,
            y="feature",
            x="absolute_attribution",
            hue="explainer",
            ax=ax,
        )
        ax.set(title=dataset, xlabel="Mean |attribution|", ylabel="")
    fig.tight_layout()
    fig.savefig(figures / "global_feature_importance.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def prepare_dataset(
    dataset: str, task: str, config: dict[str, Any], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    if dataset == "CIC-IDS2017":
        source = load_cic(config["data"]["cicids2017"], seed)
        label_col = config["data"]["cicids2017"]["label_column"]
        raw_labels = source[label_col].to_numpy()
        labels = np.where(raw_labels == "BENIGN", "Normal", "Attack") if task == "binary" else raw_labels
        train, test, y_train_raw, y_test_raw = split_cic(
            source,
            np.asarray(labels),
            float(config["data"]["cicids2017"]["test_size"]),
            seed,
        )
        metadata = {
            "sampled_rows_after_deduplication": len(source),
            "duplicates_removed_after_sampling": source.attrs.get("duplicates_removed", 0),
            "split_protocol": "stratified random 80/20 after per-class cap and deduplication",
        }
        excluded = {label_col}
    else:
        train, test, unsw_audit = load_unsw(config["data"]["unsw_nb15"])
        label_col = config["data"]["unsw_nb15"]["label_column"]
        multiclass_col = config["data"]["unsw_nb15"]["multiclass_column"]
        if task == "binary":
            y_train_raw = np.where(train[label_col].astype(int).to_numpy() == 0, "Normal", "Attack")
            y_test_raw = np.where(test[label_col].astype(int).to_numpy() == 0, "Normal", "Attack")
        else:
            y_train_raw = train[multiclass_col].astype(str).to_numpy()
            y_test_raw = test[multiclass_col].astype(str).to_numpy()
        metadata = {
            **unsw_audit,
            "split_protocol": "predefined training/testing partition with within-partition exact deduplication and cross-partition feature-overlap removal",
        }
        excluded = {label_col, multiclass_col}

    encoder = LabelEncoder().fit(y_train_raw)
    unknown = sorted(set(np.unique(y_test_raw)) - set(encoder.classes_))
    if unknown:
        raise ValueError(f"Test labels absent from training for {dataset}/{task}: {unknown}")
    y_train = encoder.transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)
    feature_names = [c for c in train.columns if c not in excluded]
    return train, test, y_train, y_test, encoder.classes_.tolist(), {
        **metadata,
        "feature_names": feature_names,
    }


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    output_dir = Path(config["output_dir"])
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    predictive_rows: list[dict] = []
    per_class_rows: list[dict] = []
    confusion_rows: list[dict] = []
    attribution_rows: list[dict] = []
    fidelity_records: list[dict] = []
    stability_records: list[dict] = []
    explanation_summaries: list[dict] = []
    ablation_rows: list[dict] = []
    dataset_metadata: dict[str, Any] = {}

    for dataset in ("CIC-IDS2017", "UNSW-NB15"):
        for task in ("binary", "multiclass"):
            print(f"[prepare] {dataset} / {task}", flush=True)
            train, test, y_train, y_test, classes, metadata = prepare_dataset(
                dataset, task, config, seed
            )
            feature_names = metadata.pop("feature_names")
            dataset_metadata[f"{dataset}/{task}"] = metadata
            preprocessor = TabularPreprocessor(feature_names).fit(train)
            x_train = preprocessor.transform(train)
            x_test = preprocessor.transform(test)

            for model_name in ("random_forest", "xgboost"):
                frozen_model = None
                frozen_prediction = None
                for model_seed in config["predictive_seeds"]:
                    print(
                        f"[train] {dataset} / {task} / {model_name} / seed={model_seed}",
                        flush=True,
                    )
                    model = make_model(
                        model_name, config["models"], int(model_seed), len(classes)
                    )
                    sample_weight = (
                        compute_sample_weight("balanced", y_train)
                        if model_name == "xgboost"
                        else None
                    )
                    start = time.perf_counter()
                    model.fit(x_train, y_train, sample_weight=sample_weight)
                    training_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    probabilities = model.predict_proba(x_test)
                    prediction_seconds = time.perf_counter() - start
                    prediction = np.argmax(probabilities, axis=1)
                    metrics, per_class, matrix = predictive_metrics(
                        y_test, prediction, probabilities, classes
                    )
                    common = {
                        "dataset": dataset,
                        "task": task,
                        "model": model_name,
                        "seed": int(model_seed),
                    }
                    predictive_rows.append(
                        {
                            **common,
                            **metrics,
                            "training_seconds": training_seconds,
                            "prediction_microseconds_per_flow": prediction_seconds
                            * 1_000_000
                            / len(x_test),
                            "train_rows": len(x_train),
                            "test_rows": len(x_test),
                            "feature_count": len(feature_names),
                        }
                    )
                    per_class_rows.extend({**common, **row} for row in per_class)
                    if int(model_seed) == seed:
                        frozen_model = model
                        frozen_prediction = prediction
                        for i, actual in enumerate(classes):
                            for j, predicted_class in enumerate(classes):
                                confusion_rows.append(
                                    {
                                        **common,
                                        "actual_class": actual,
                                        "predicted_class": predicted_class,
                                        "count": int(matrix[i, j]),
                                    }
                                )
                        safe_name = f"{dataset}_{task}_{model_name}".replace("-", "_")
                        save_confusion_plot(
                            matrix,
                            classes,
                            f"{dataset}: {task}, {model_name}",
                            output_dir / "figures" / f"confusion_{safe_name}.png",
                        )
                assert frozen_model is not None and frozen_prediction is not None
                artifact = {
                    "model": frozen_model,
                    "preprocessor": preprocessor,
                    "classes": classes,
                    "dataset": dataset,
                    "task": task,
                    "seed": seed,
                }
                safe_name = f"{dataset}_{task}_{model_name}".replace("-", "_")
                joblib.dump(artifact, output_dir / "models" / f"{safe_name}.joblib", compress=3)

                for explainer_name in ("TreeSHAP", "LIME"):
                    print(
                        f"[explain] {dataset} / {task} / {model_name} / {explainer_name}",
                        flush=True,
                    )
                    attrs, fidelity, stability, summary = evaluate_explainer(
                        dataset,
                        task,
                        model_name,
                        explainer_name,
                        frozen_model,
                        x_train,
                        x_test,
                        y_test,
                        preprocessor,
                        classes,
                        config["evaluation"],
                        seed,
                    )
                    attribution_rows.extend(attrs)
                    fidelity_records.extend(fidelity)
                    stability_records.extend(stability)
                    explanation_summaries.append(summary)

                ranking = np.argsort(-np.asarray(frozen_model.feature_importances_))
                ablation_rows.extend(
                    feature_ablation(
                        dataset,
                        task,
                        model_name,
                        config["models"],
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        feature_names,
                        ranking,
                        list(config["evaluation"]["feature_ablation_k"]),
                        seed,
                    )
                )

    pd.DataFrame(predictive_rows).to_csv(output_dir / "predictive_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(output_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output_dir / "confusion_matrices.csv", index=False)
    pd.DataFrame(attribution_rows).to_csv(output_dir / "attributions.csv", index=False)
    fidelity_frame = pd.DataFrame(fidelity_records)
    fidelity_frame.to_csv(output_dir / "fidelity.csv", index=False)
    pd.DataFrame(stability_records).to_csv(output_dir / "stability.csv", index=False)
    pd.DataFrame(explanation_summaries).to_csv(
        output_dir / "explanation_summary.csv", index=False
    )
    pd.DataFrame(ablation_rows).to_csv(output_dir / "feature_ablation.csv", index=False)
    paired_bootstrap_comparisons(fidelity_frame, seed).to_csv(
        output_dir / "paired_comparisons.csv", index=False
    )
    fidelity_control_tests(fidelity_frame, seed).to_csv(
        output_dir / "fidelity_control_tests.csv", index=False
    )

    raw_paths = [
        Path(item)
        for item in config["data"]["cicids2017"]["shards"]
        + [config["data"]["unsw_nb15"]["train"], config["data"]["unsw_nb15"]["test"]]
    ]
    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "config": config,
        "datasets": dataset_metadata,
        "files": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in raw_paths
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "shap": shap.__version__,
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    make_summary_plots(output_dir)
    print(f"[done] Results written to {output_dir.resolve()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    run(args.config)
