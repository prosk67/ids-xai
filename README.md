# Quantitative XAI for Network Intrusion Detection

Reproducible research code and artifacts for **A Quantitative Framework for Evaluating Explainable AI in Network Intrusion Detection Systems**.

This project asks a practical question: when an intrusion-detection model raises an alert, can its feature explanation be measured rather than merely visualized? Random Forest and XGBoost are evaluated on CIC-IDS2017 and UNSW-NB15 for binary and multiclass detection. TreeSHAP and LIME are then compared using matched local explanations, deletion fidelity, random and bottom-ranked controls, neighborhood stability, repeated-seed stability, latency, and robustness audits.

> **Research status:** reproducible manuscript artifact prepared for peer review. It has not yet been peer reviewed or accepted by an IEEE venue.

## Main contribution

The framework treats detection and explanation as one auditable system:

1. train and evaluate the detector without fitting preprocessing on test data, choosing hyperparameters and feature counts on a training-only validation split;
2. explain the same 200 stratified test flows with TreeSHAP and LIME under each of five model seeds;
3. mask top-ranked features and measure the predicted-class confidence drop, and correlate attribution mass with the masking effect on random feature subsets (faithfulness correlation);
4. compare the drop against random, bottom-ranked, and global-importance feature controls;
5. check that explanations change when the model is trained on permuted labels (data-randomization sanity check);
6. report paired uncertainty, stability, latency, calibration, masking and SHAP-background sensitivity, and the label-conflict ceiling of each dataset; and
7. generate alert reports that reject weak explanation evidence.

This mitigates the operational black-box problem by making individual decisions inspectable and testable. It does **not** claim causal explanations, universal interpretability, or safe autonomous response.

## Headline results

Mean macro-F1 over five model seeds:

| Dataset | Task | Random Forest | XGBoost |
|---|---|---:|---:|
| CIC-IDS2017 | Binary | 0.9975 | 0.9977 |
| CIC-IDS2017 | Multiclass | 0.9299 | 0.9334 |
| UNSW-NB15, leakage controlled | Binary | 0.8148 | 0.8201 |
| UNSW-NB15, leakage controlled | Multiclass | 0.5323 | 0.5292 |

Hyperparameters and feature counts are chosen on a training-only validation split; the test partition is never used for selection.

TreeSHAP and LIME explain the same 200 test flows per configuration under all five model seeds. TreeSHAP had higher paired deletion fidelity (mean of k = 1, 3, 5, 10) than LIME in all 8 dataset/task/model configurations (differences 0.087-0.187; Holm-adjusted one-sided Wilcoxon p <= 8.8e-17) and higher faithfulness correlation in all 8 (differences 0.248-0.351). TreeSHAP beat random rankings in 8/8 configurations and the model's own global-importance ranking in 8/8. LIME beat random rankings in 7/8 (failing for CIC-IDS2017 binary XGBoost) but the global ranking in only 3/8, so in most settings LIME's per-alert ranking adds little beyond a fixed importance list.

The accessible UNSW mirror required a publication-critical leakage correction. The loader removes 78,519 duplicate training rows, 32,361 duplicate testing rows, and 2,255 test feature vectors also present in training. The resulting predefined partition has 96,822 training and 47,716 testing rows with zero exact feature-vector overlap. These lower corrected scores supersede earlier contaminated results.

## Start here

- [IEEE-style research manuscript](output/pdf/xai_ids_ieee_paper.pdf)
- [Publication-readiness audit](PUBLICATION_READINESS_REPORT.md)
- [How the project was done (full details, beginner to end)](HOW_THE_PROJECT_WAS_DONE.md)
- [Run everything on Google Colab](COLAB_RUN.ipynb)
- [Executable research walkthrough](PROJECT_WALKTHROUGH.ipynb)
- [Literature-to-evidence traceability](LITERATURE_REQUIREMENTS_TRACEABILITY.md)
- [Example conservative alert report](outputs/alert_reports/unsw_binary_xgb_example.md)

## Repository structure

```text
src/xai_ids/                 benchmark, audits, reporting, and paper generation
tests/                       unit tests, including leakage-cleaning behavior
outputs/                     machine-readable results and generated figures
output/pdf/                  generated research manuscript
config.yaml                  fixed experiment configuration
run_benchmark.py             predictive and primary XAI experiments
run_extended_audit.py        robustness and literature-driven audits
explain_alert.py             per-alert transparency report
verify_outputs.py            cross-artifact integrity checks
PROJECT_WALKTHROUGH.ipynb    guided exploration of all research tasks
COLAB_RUN.ipynb              full run on Google Colab
HOW_THE_PROJECT_WAS_DONE.md  complete explanation of the project
```

## Reproduce

Python 3.12 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_benchmark.py --config config.yaml
python run_extended_audit.py --config config.yaml
python build_paper.py
python -B verify_outputs.py
python -B -m unittest discover -s tests -v
```

On a 16-core machine the benchmark takes about 28 minutes and the extended audit about 15. `PROJECT_WALKTHROUGH.ipynb` reads the committed result tables and is the easiest way to understand the workflow without rerunning anything.

**Google Colab:** pack the code with `git archive -o ids-xai.zip HEAD` and the data with `zip -r ids-xai-data.zip data/raw`, upload both to `MyDrive/ids-xai/`, and run [`COLAB_RUN.ipynb`](COLAB_RUN.ipynb). It installs the pinned packages, checks the data hashes, and runs the tests, benchmark, audit, paper build, and verifier, then saves the results to Drive. Set `QUICK = True` for a short smoke run. Details are in [HOW_THE_PROJECT_WAS_DONE.md](HOW_THE_PROJECT_WAS_DONE.md#18-running-the-project-on-google-colab).

## Dataset preparation

Raw datasets are not redistributed in this repository. Obtain CIC-IDS2017 and UNSW-NB15 from their official providers under their respective terms, create the four parquet files declared in `config.yaml`, and compare their SHA-256 hashes with `outputs/run_manifest.json`.

- **CIC-IDS2017:** the local representation contains all 2,830,743 flows and 28 selected flow features. Because its day/timestamp metadata is unavailable, the stratified evaluation is not described as temporal generalization.
- **UNSW-NB15:** uses the predefined 175,341-row training and 82,332-row testing files before the documented duplicate and overlap audit.

Preprocessing is learned only from training data. Numeric missing/infinite values use training medians; categorical variables use training-derived ordinal dictionaries with an explicit unseen category. The five predictive seeds are 42, 52, 62, 72, and 82.

## Explanation evidence

Fidelity is an intervention-specific behavioral check: top-ranked features are replaced by declared reference values and the predicted-class probability change is measured. It is not proof that a feature caused an attack. The extended audit therefore repeats the analysis with zero, marginal-row, nearest-opposite-prediction, and nearest-same-prediction references.

An alert receives `SUPPORTED_FOR_THIS_ALERT` only when TreeSHAP passed the corrected benchmark control and that alert's top ranking beats randomized rankings at empirical p <= 0.05 under the declared median/mode intervention. Otherwise, the report returns `REVIEW_REQUIRED` or `UNSUPPORTED`.

```bash
python explain_alert.py \
  --model outputs/models/UNSW_NB15_binary_xgboost.joblib \
  --input path/to/flows.csv \
  --row 0 \
  --output outputs/alert_reports/alert_0
```

Serialized models are generated locally and intentionally excluded from GitHub.

## Limitations

- Neither benchmark is a live security-operations deployment or a cross-organization test.
- CIC evaluation is stratified, not temporal.
- UNSW is a predefined testbed partition, not a documented temporal holdout.
- Neighborhood stability is measured within the selected 200-flow explanation subset.
- Deletion fidelity depends on the masking distribution and should not be interpreted causally.
- UNSW multiclass recall remains weak for several minority attacks.
- The project does not include a human analyst study or production drift evaluation.

## Publication

The committed PDF is anonymous for peer review. Before camera-ready submission, update `paper_metadata.yaml` with the authors and affiliations, add an archival repository URL or DOI, and apply the exact target venue template and page limit. The named authors must independently approve every claim and comply with the venue's disclosure and artifact policies.
