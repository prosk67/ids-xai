from pathlib import Path
import json

import pandas as pd
import yaml
from pypdf import PdfReader


root = Path(__file__).resolve().parent
outputs = root / "outputs"
config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
expected_flows = int(config["evaluation"]["explanation_samples"])
expected_seeds = {int(seed) for seed in config["evaluation"]["explanation_seeds"]}
required = [
    "predictive_metrics.csv",
    "per_class_metrics.csv",
    "explanation_summary.csv",
    "attributions.csv.gz",
    "fidelity.csv.gz",
    "explanation_flows.csv.gz",
    "stability.csv",
    "paired_comparisons.csv",
    "feature_ablation.csv",
    "fidelity_control_tests.csv",
    "tuning.csv",
    "run_manifest.json",
]
for name in required:
    path = outputs / name
    assert path.exists() and path.stat().st_size > 0, path

extended_required = [
    "predictive_baselines.csv",
    "balancing_ablation.csv",
    "calibration_summary.csv",
    "recalibration.csv",
    "masking_baseline_ablation.csv",
    "lime_budget_ablation.csv",
    "shap_background_ablation.csv",
    "randomization_sanity.csv",
    "label_conflicts.csv",
    "class_specific_fidelity.csv",
    "error_type_fidelity.csv",
    "stability_distributions.csv",
    "literature_purpose_coverage.csv",
    "black_box_audit_summary.json",
]
for name in extended_required:
    path = outputs / "extended" / name
    assert path.exists() and path.stat().st_size > 0, path

contexts = ["dataset", "task", "model"]
predictive = pd.read_csv(outputs / "predictive_metrics.csv")
explanations = pd.read_csv(outputs / "explanation_summary.csv")
paired = pd.read_csv(outputs / "paired_comparisons.csv")
fidelity = pd.read_csv(outputs / "fidelity.csv.gz")
flows = pd.read_csv(outputs / "explanation_flows.csv.gz")
controls = pd.read_csv(outputs / "fidelity_control_tests.csv")
tuning = pd.read_csv(outputs / "tuning.csv")
ablation = pd.read_csv(outputs / "feature_ablation.csv")
assert len(predictive) == 8 * len(config["predictive_seeds"])
assert len(explanations) == 16
assert predictive["seed"].nunique() == len(config["predictive_seeds"])
assert predictive["macro_f1"].between(0, 1).all()
assert explanations["fidelity_mean"].notna().all()
assert explanations["n_explanations"].eq(expected_flows).all()
assert explanations["n_model_seeds"].eq(len(expected_seeds)).all()
assert explanations["faithfulness_correlation_mean"].between(-1, 1).all()

# These checks are about internal consistency only. Which explainer wins is a
# research outcome and must not be asserted here.
assert paired.groupby("metric").size().to_dict() == {"deletion": 8, "faithfulness_correlation": 8}
assert paired["n_pairs"].eq(expected_flows).all(), paired["n_pairs"].unique()
assert paired["ci_low"].le(paired["mean_difference_shap_minus_lime"]).all()
assert paired["mean_difference_shap_minus_lime"].le(paired["ci_high"]).all()
assert paired["holm_adjusted_p"].between(0, 1).all()
assert paired["holm_adjusted_p"].ge(paired["wilcoxon_p"] - 1e-12).all()

# Every explainer and model seed must explain exactly the same flows in each
# configuration, otherwise the paired statistical comparisons are invalid.
for key, group in fidelity.groupby(contexts):
    source_sets = {
        name: frozenset(rows["source_test_index"])
        for name, rows in group.groupby(["explainer", "model_seed"])
    }
    assert {name for name, _ in source_sets} == {"TreeSHAP", "LIME"}, key
    assert {seed for _, seed in source_sets} == expected_seeds, key
    assert len(set(source_sets.values())) == 1, key
    assert len(next(iter(source_sets.values()))) == expected_flows, key
assert len(flows) == 8 * 2 * len(expected_seeds) * expected_flows

assert controls.groupby("reference").size().to_dict() == {"global": 16, "random": 16}
assert controls["n_pairs"].eq(expected_flows).all()
assert controls["holm_adjusted_p"].between(0, 1).all()
assert controls["ci_low"].le(controls["ci_high"]).all()

# Exactly one validation-selected candidate and feature count per context.
assert tuning.groupby(contexts)["selected"].sum().eq(1).all()
assert ablation.groupby(contexts)["selected"].sum().eq(1).all()

coverage = pd.read_csv(outputs / "extended" / "literature_purpose_coverage.csv")
assert (coverage["status"] == "complete").sum() >= 18
recalibration = pd.read_csv(outputs / "extended" / "recalibration.csv")
assert recalibration[["ece_raw", "ece_isotonic"]].stack().between(0, 1).all()
sanity = pd.read_csv(outputs / "extended" / "randomization_sanity.csv")
assert set(sanity["explainer"]) == {"TreeSHAP", "LIME"}
conflicts = pd.read_csv(outputs / "extended" / "label_conflicts.csv")
assert conflicts["conflicting_share"].between(0, 1).all()
calibration = pd.read_csv(outputs / "extended" / "calibration_summary.csv")
assert calibration["expected_calibration_error"].between(0, 1).all()
masking = pd.read_csv(outputs / "extended" / "masking_baseline_ablation.csv")
assert masking["baseline"].nunique() == 5

manifest = json.loads((outputs / "run_manifest.json").read_text(encoding="utf-8"))
assert len(manifest["files"]) == 4
assert len(manifest["selected_hyperparameters"]) == 8
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
# Six pages is the current draft; allow growth up to a typical IEEE
# conference limit until the target venue is chosen.
assert 6 <= len(reader.pages) <= 8, len(reader.pages)
assert float(reader.pages[0].mediabox.width) == 612
assert float(reader.pages[0].mediabox.height) == 792
for heading in (
    "INTRODUCTION",
    "METHODOLOGY",
    "EXPERIMENTAL RESULTS",
    "THREATS TO VALIDITY",
    "REPRODUCIBILITY",
    "CONCLUSION",
    "REFERENCES",
):
    assert heading in text
assert "Ethics and Artifact Availability" in text
assert "[16]" in text
assert "explanation fidelity" in (reader.metadata.keywords or "")

# IEEE Xplore requires every font resource to be embedded or subset. TrueType
# resources expose /FontDescriptor directly; composite fonts expose it through
# /DescendantFonts.
for page in reader.pages:
    for font_ref in page.get("/Resources", {}).get("/Font", {}).values():
        font = font_ref.get_object()
        embedded = bool(font.get("/FontDescriptor"))
        if font.get("/DescendantFonts"):
            descendant = font["/DescendantFonts"][0].get_object()
            embedded = embedded or bool(descendant.get("/FontDescriptor"))
        assert embedded, font.get("/BaseFont")
for forbidden in ("temporal UNSW", "preregistered", "CSE 498R Research Project"):
    assert forbidden not in text
print("All output, manifest, and paper checks passed.")
