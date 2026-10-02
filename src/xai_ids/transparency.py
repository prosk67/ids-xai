from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from .benchmark import extract_shap_vector


def evidence_decision(
    global_control_passed: bool, local_lift: float, local_empirical_p: float
) -> tuple[str, str]:
    """Return a conservative evidence tier and its operational meaning."""
    if not global_control_passed:
        return (
            "UNSUPPORTED",
            "The explainer did not pass the benchmark-level corrected random-ranking control.",
        )
    if local_lift > 0 and local_empirical_p <= 0.05:
        return (
            "SUPPORTED_FOR_THIS_ALERT",
            "The validated explainer also beat randomized rankings for this alert under the primary training median/mode intervention.",
        )
    return (
        "REVIEW_REQUIRED",
        "The explainer passed globally, but this alert did not clear the strict local random-ranking test.",
    )


def _ranking_score(
    model,
    row: np.ndarray,
    predicted_class: int,
    baseline: np.ndarray,
    ranking: np.ndarray,
    k_values: list[int],
) -> tuple[float, list[dict[str, Any]]]:
    original = float(model.predict_proba(row.reshape(1, -1))[0, predicted_class])
    records = []
    drops = []
    for k in k_values:
        k_eff = min(int(k), len(row))
        masked = row.copy()
        masked[ranking[:k_eff]] = baseline[ranking[:k_eff]]
        probability = float(model.predict_proba(masked.reshape(1, -1))[0, predicted_class])
        drop = original - probability
        drops.append(drop)
        records.append(
            {
                "k": k_eff,
                "probability_after_masking": probability,
                "probability_drop": drop,
                "prediction_flipped": bool(np.argmax(model.predict_proba(masked.reshape(1, -1))[0]) != predicted_class),
            }
        )
    return float(np.mean(drops)), records


def local_deletion_evidence(
    model,
    row: np.ndarray,
    attribution: np.ndarray,
    predicted_class: int,
    baseline: np.ndarray,
    k_values: list[int],
    random_trials: int,
    seed: int,
) -> dict[str, Any]:
    """Test an alert's attribution ranking against bottom and randomized rankings."""
    ranking = np.argsort(-np.abs(attribution))
    bottom_ranking = ranking[::-1]
    top_score, top_details = _ranking_score(
        model, row, predicted_class, baseline, ranking, k_values
    )
    bottom_score, bottom_details = _ranking_score(
        model, row, predicted_class, baseline, bottom_ranking, k_values
    )
    rng = np.random.default_rng(seed)
    random_scores = []
    for _ in range(random_trials):
        random_ranking = rng.permutation(len(row))
        score, _ = _ranking_score(
            model, row, predicted_class, baseline, random_ranking, k_values
        )
        random_scores.append(score)
    random_array = np.asarray(random_scores, dtype=float)
    empirical_p = float((1 + np.sum(random_array >= top_score)) / (random_trials + 1))
    return {
        "top_deletion_score": top_score,
        "bottom_deletion_score": bottom_score,
        "random_mean_deletion_score": float(random_array.mean()),
        "random_p95_deletion_score": float(np.quantile(random_array, 0.95)),
        "top_minus_random_lift": float(top_score - random_array.mean()),
        "local_empirical_one_sided_p": empirical_p,
        "random_trials": random_trials,
        "top_masking_details": top_details,
        "bottom_masking_details": bottom_details,
    }


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = [payload]
        return pd.DataFrame(payload)
    raise ValueError("Input must be CSV, Parquet, or JSON.")


def _benchmark_context(results_dir: Path, dataset: str, task: str, model_name: str) -> dict[str, Any]:
    controls = pd.read_csv(results_dir / "fidelity_control_tests.csv")
    summaries = pd.read_csv(results_dir / "explanation_summary.csv")
    selector = (
        controls["dataset"].eq(dataset)
        & controls["task"].eq(task)
        & controls["model"].eq(model_name)
        & controls["explainer"].eq("TreeSHAP")
    )
    control = controls.loc[selector].iloc[0]
    selector = (
        summaries["dataset"].eq(dataset)
        & summaries["task"].eq(task)
        & summaries["model"].eq(model_name)
        & summaries["explainer"].eq("TreeSHAP")
    )
    summary = summaries.loc[selector].iloc[0]
    passed = bool(control["holm_adjusted_p"] < 0.05 and control["ci_low"] > 0)
    context = {
        "global_random_control_passed": passed,
        "global_top_minus_random_lift": float(control["mean_lift_top_minus_random"]),
        "global_lift_ci": [float(control["ci_low"]), float(control["ci_high"])],
        "global_holm_adjusted_p": float(control["holm_adjusted_p"]),
        "neighborhood_cosine_mean": float(summary["stability_cosine_mean"]),
        "neighborhood_top5_jaccard_mean": float(summary["stability_jaccard_mean"]),
        "median_explanation_latency_ms": float(summary["latency_median_ms"]),
        "p95_explanation_latency_ms": float(summary["latency_p95_ms"]),
        "validation_sample_count": int(summary["n_explanations"]),
    }
    masking_path = results_dir / "extended" / "masking_baseline_ablation.csv"
    if masking_path.exists():
        masking = pd.read_csv(masking_path)
        subset = masking[
            masking.dataset.eq(dataset)
            & masking.task.eq(task)
            & masking.model.eq(model_name)
        ]
        context["reference_sensitivity_mean_deletion"] = {
            str(name): float(value)
            for name, value in subset.groupby("baseline").deletion_score.mean().items()
        }
    return context


def explain_alert(
    artifact_path: Path,
    input_path: Path,
    row_index: int,
    output_base: Path,
    top_n: int = 10,
    random_trials: int = 200,
    seed: int = 42,
) -> tuple[Path, Path]:
    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    classes = list(artifact["classes"])
    dataset = str(artifact["dataset"])
    task = str(artifact["task"])
    model_name = artifact_path.stem.replace(f"{dataset}_{task}_", "").replace("-", "_")
    if model_name not in {"random_forest", "xgboost"}:
        model_name = "random_forest" if "random_forest" in artifact_path.stem else "xgboost"

    frame = _load_frame(input_path)
    if row_index < 0 or row_index >= len(frame):
        raise IndexError(f"row_index={row_index} is outside the input's {len(frame)} rows.")
    missing = [name for name in preprocessor.feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"Input is missing required features: {missing}")
    raw = frame.iloc[row_index]
    transformed = preprocessor.transform(frame.iloc[[row_index]])[0]
    probabilities = model.predict_proba(transformed.reshape(1, -1))[0]
    predicted_index = int(np.argmax(probabilities))
    explainer = shap.TreeExplainer(model)
    attribution = extract_shap_vector(
        explainer(transformed.reshape(1, -1), check_additivity=False), predicted_index
    )
    order = np.argsort(-np.abs(attribution))[: min(top_n, len(attribution))]
    top_features = []
    for rank, index in enumerate(order, start=1):
        value = raw[preprocessor.feature_names[index]]
        top_features.append(
            {
                "rank": rank,
                "feature": preprocessor.feature_names[index],
                "observed_value": None if pd.isna(value) else str(value),
                "transformed_value": float(transformed[index]),
                "attribution": float(attribution[index]),
                "direction": "supports prediction" if attribution[index] >= 0 else "opposes prediction",
            }
        )

    local = local_deletion_evidence(
        model,
        transformed,
        attribution,
        predicted_index,
        np.asarray(preprocessor.baseline_, dtype=float),
        [1, 3, 5, 10],
        random_trials,
        seed,
    )
    results_dir = artifact_path.parent.parent
    global_context = _benchmark_context(results_dir, dataset, task, model_name)
    tier, meaning = evidence_decision(
        global_context["global_random_control_passed"],
        local["top_minus_random_lift"],
        local["local_empirical_one_sided_p"],
    )
    report = {
        "black_box_status": "MITIGATED_WITH_TESTED_LOCAL_EVIDENCE_NOT_ELIMINATED",
        "evidence_scope": "PRIMARY_TRAINING_MEDIAN_MODE_INTERVENTION",
        "evidence_tier": tier,
        "evidence_meaning": meaning,
        "scope_warning": (
            "This report explains the fitted model under explicit masking tests. "
            "It is not a causal explanation and does not replace analyst review."
        ),
        "model": {
            "artifact": str(artifact_path.resolve()),
            "dataset": dataset,
            "task": task,
            "model_family": model_name,
            "seed": int(artifact["seed"]),
        },
        "alert": {
            "input": str(input_path.resolve()),
            "row_index": row_index,
            "predicted_class": classes[predicted_index],
            "predicted_probability": float(probabilities[predicted_index]),
            "class_probabilities": {
                str(label): float(probabilities[i]) for i, label in enumerate(classes)
            },
        },
        "top_signed_features": top_features,
        "local_fidelity_evidence": local,
        "benchmark_validation_context": global_context,
    }

    output_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_base.with_suffix(".json")
    markdown_path = output_base.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    feature_lines = "\n".join(
        f"| {item['rank']} | {item['feature']} | {item['observed_value']} | "
        f"{item['attribution']:.6f} | {item['direction']} |"
        for item in top_features
    )
    probability_lines = "\n".join(
        f"- {label}: {probability:.6f}"
        for label, probability in report["alert"]["class_probabilities"].items()
    )
    reference_context = global_context.get("reference_sensitivity_mean_deletion", {})
    reference_lines = "\n".join(
        f"- {name}: {value:.6f}" for name, value in reference_context.items()
    ) or "- Extended reference audit not available."
    markdown = f"""# Evidence-Backed IDS Alert Explanation

## Decision transparency status

- **Evidence tier:** `{tier}`
- **Meaning:** {meaning}
- **Black-box status:** mitigated with tested local evidence; not theoretically eliminated
- **Evidence scope:** primary training median/mode masking intervention
- **Required warning:** {report['scope_warning']}

## Prediction

- Dataset/model context: {dataset} / {task} / {model_name}
- Predicted class: **{classes[predicted_index]}**
- Predicted probability: **{probabilities[predicted_index]:.6f}**
- Model seed: {artifact['seed']}

Class probabilities:

{probability_lines}

## Top signed TreeSHAP evidence

| Rank | Feature | Observed value | Attribution | Direction |
|---:|---|---:|---:|---|
{feature_lines}

Positive attribution supports the reported class; negative attribution opposes it. Attribution is model evidence, not real-world causality.

## Alert-specific fidelity test

- Top-ranked deletion score: {local['top_deletion_score']:.6f}
- Bottom-ranked deletion score: {local['bottom_deletion_score']:.6f}
- Mean of {random_trials} random rankings: {local['random_mean_deletion_score']:.6f}
- Top-minus-random lift: {local['top_minus_random_lift']:.6f}
- Empirical one-sided local p-value: {local['local_empirical_one_sided_p']:.6f}

The deletion score is the mean predicted-probability drop after replacing the top 1, 3, 5, and 10 features with training-derived baseline values.

## Benchmark validation context

- TreeSHAP passed corrected random control: {global_context['global_random_control_passed']}
- Mean top-minus-random lift: {global_context['global_top_minus_random_lift']:.6f}
- 95% bootstrap interval: [{global_context['global_lift_ci'][0]:.6f}, {global_context['global_lift_ci'][1]:.6f}]
- Holm-adjusted p-value: {global_context['global_holm_adjusted_p']:.6f}
- Mean neighbor cosine stability: {global_context['neighborhood_cosine_mean']:.6f}
- Mean top-5 neighbor overlap: {global_context['neighborhood_top5_jaccard_mean']:.6f}
- Median/P95 explanation latency: {global_context['median_explanation_latency_ms']:.3f} / {global_context['p95_explanation_latency_ms']:.3f} ms

Mean TreeSHAP deletion scores from the separate reference-sensitivity audit:

{reference_lines}

## Interpretation rule

- `SUPPORTED_FOR_THIS_ALERT`: benchmark control passed and this alert beats randomized rankings at empirical p <= 0.05.
- `REVIEW_REQUIRED`: benchmark control passed, but the alert-specific ranking did not clear the strict local test.
- `UNSUPPORTED`: the explainer failed benchmark-level validation.

This report makes the model's decision auditable and tests whether its stated evidence changes the model output under a declared intervention. It does not claim causal truth, intervention-invariant fidelity, complete human comprehension, or immunity to dataset shift.
"""
    markdown_path.write_text(markdown, encoding="utf-8")
    return markdown_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an evidence-backed IDS alert explanation.")
    parser.add_argument("--model", required=True, type=Path, help="Path to a saved .joblib model artifact.")
    parser.add_argument("--input", required=True, type=Path, help="CSV, Parquet, or JSON containing flow features.")
    parser.add_argument("--row", type=int, default=0, help="Zero-based input row index.")
    parser.add_argument("--output", type=Path, default=Path("outputs/alert_reports/alert_0"))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--random-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    markdown_path, json_path = explain_alert(
        args.model, args.input, args.row, args.output, args.top_n, args.random_trials, args.seed
    )
    print(markdown_path.resolve())
    print(json_path.resolve())


if __name__ == "__main__":
    main()
