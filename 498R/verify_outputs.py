from pathlib import Path
import json

import pandas as pd
from pypdf import PdfReader


root = Path(__file__).resolve().parent
outputs = root / "outputs"
required = [
    "predictive_metrics.csv",
    "per_class_metrics.csv",
    "explanation_summary.csv",
    "fidelity.csv",
    "stability.csv",
    "paired_comparisons.csv",
    "feature_ablation.csv",
    "fidelity_control_tests.csv",
    "run_manifest.json",
]
for name in required:
    path = outputs / name
    assert path.exists() and path.stat().st_size > 0, path

extended_required = [
    "predictive_baselines.csv",
    "balancing_ablation.csv",
    "calibration_summary.csv",
    "masking_baseline_ablation.csv",
    "lime_budget_ablation.csv",
    "class_specific_fidelity.csv",
    "error_type_fidelity.csv",
    "stability_distributions.csv",
    "literature_purpose_coverage.csv",
    "black_box_audit_summary.json",
]
for name in extended_required:
    path = outputs / "extended" / name
    assert path.exists() and path.stat().st_size > 0, path

predictive = pd.read_csv(outputs / "predictive_metrics.csv")
explanations = pd.read_csv(outputs / "explanation_summary.csv")
paired = pd.read_csv(outputs / "paired_comparisons.csv")
fidelity = pd.read_csv(outputs / "fidelity.csv")
controls = pd.read_csv(outputs / "fidelity_control_tests.csv")
assert len(predictive) == 40
assert len(explanations) == 16
assert len(paired) == 8
assert predictive["seed"].nunique() == 5
assert predictive["macro_f1"].between(0, 1).all()
assert explanations["fidelity_mean"].notna().all()
assert (paired["ci_low"] > 0).all()
assert (paired["n_pairs"] == 40).all()

# TreeSHAP and LIME must be evaluated on the same 40 source flows in every
# configuration, otherwise the paired statistical comparisons are invalid.
for _, group in fidelity.groupby(["dataset", "task", "model"]):
    source_sets = {
        explainer: set(rows["source_test_index"].unique())
        for explainer, rows in group.groupby("explainer")
    }
    assert source_sets.keys() == {"TreeSHAP", "LIME"}
    assert source_sets["TreeSHAP"] == source_sets["LIME"]
    assert len(source_sets["TreeSHAP"]) == 40

assert len(controls) == 16
tree_controls = controls[controls["explainer"] == "TreeSHAP"]
lime_controls = controls[controls["explainer"] == "LIME"]
assert (tree_controls["holm_adjusted_p"] < 0.05).sum() == 8
assert (lime_controls["holm_adjusted_p"] < 0.05).sum() == 7

coverage = pd.read_csv(outputs / "extended" / "literature_purpose_coverage.csv")
assert (coverage["status"] == "complete").sum() >= 11
calibration = pd.read_csv(outputs / "extended" / "calibration_summary.csv")
assert calibration["expected_calibration_error"].between(0, 1).all()
masking = pd.read_csv(outputs / "extended" / "masking_baseline_ablation.csv")
assert masking["baseline"].nunique() == 5

manifest = json.loads((outputs / "run_manifest.json").read_text(encoding="utf-8"))
assert len(manifest["files"]) == 4
for path, metadata in manifest["files"].items():
    assert len(metadata["sha256"]) == 64, path

for task in ("binary", "multiclass"):
    audit = manifest["datasets"][f"UNSW-NB15/{task}"]
    assert audit["clean_training_rows"] == 96822
    assert audit["clean_testing_rows"] == 47716
    assert audit["cross_partition_feature_overlaps_removed_from_test"] == 2255
    subset = predictive[
        (predictive["dataset"] == "UNSW-NB15") & (predictive["task"] == task)
    ]
    assert (subset["train_rows"] == audit["clean_training_rows"]).all()
    assert (subset["test_rows"] == audit["clean_testing_rows"]).all()

balancing = pd.read_csv(outputs / "extended" / "balancing_ablation.csv")
assert balancing["seed"].nunique() == 1
assert balancing["seed"].iloc[0] == manifest["config"]["seed"]

paper = root / "output" / "pdf" / "xai_ids_ieee_paper.pdf"
reader = PdfReader(paper)
text = "\n".join(page.extract_text() or "" for page in reader.pages)
assert len(reader.pages) == 5
for heading in ("INTRODUCTION", "METHODOLOGY", "EXPERIMENTAL RESULTS", "CONCLUSION", "REFERENCES"):
    assert heading in text
for forbidden in ("temporal UNSW", "preregistered", "CSE 498R Research Project"):
    assert forbidden not in text
print("All output, manifest, and paper checks passed.")
