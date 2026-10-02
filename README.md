# Quantitative XAI for Network Intrusion Detection

Reproducible research code and artifacts for **A Quantitative Framework for Evaluating Explainable AI in Network Intrusion Detection Systems**.

This project asks a practical question: when an intrusion-detection model raises an alert, can its feature explanation be measured rather than merely visualized? Random Forest and XGBoost are evaluated on CIC-IDS2017 and UNSW-NB15 for binary and multiclass detection. TreeSHAP and LIME are then compared using matched local explanations, deletion fidelity, random and bottom-ranked controls, neighborhood stability, repeated-seed stability, latency, and robustness audits.

> **Research status:** reproducible manuscript artifact prepared for peer review. It has not yet been peer reviewed or accepted by an IEEE venue.

## Main contribution

The framework treats detection and explanation as one auditable system:

1. train and evaluate the detector without fitting preprocessing on test data;
2. explain the same 40 stratified test flows with TreeSHAP and LIME;
3. mask top-ranked features and measure the predicted-class confidence drop;
4. compare the drop against random and bottom-ranked feature controls;
5. report paired uncertainty, stability, latency, calibration, and masking sensitivity; and
6. generate alert reports that reject weak explanation evidence.

This mitigates the operational black-box problem by making individual decisions inspectable and testable. It does **not** claim causal explanations, universal interpretability, or safe autonomous response.

## Corrected headline results

Mean macro-F1 over five model seeds:

| Dataset | Task | Random Forest | XGBoost |
|---|---|---:|---:|
| CIC-IDS2017 | Binary | 0.9965 | 0.9974 |
| CIC-IDS2017 | Multiclass | 0.9254 | 0.9324 |
| UNSW-NB15, leakage controlled | Binary | 0.8013 | 0.8156 |
| UNSW-NB15, leakage controlled | Multiclass | 0.5140 | 0.5083 |

TreeSHAP achieved higher paired top-5 deletion fidelity than LIME in all 8 dataset/task/model configurations. All paired 95% bootstrap intervals were above zero; Holm-adjusted one-sided Wilcoxon p-values were at most 0.00319. TreeSHAP passed the top-versus-random control in 8/8 configurations, while LIME passed 7/8.

The accessible UNSW mirror required a publication-critical leakage correction. The loader removes 78,519 duplicate training rows, 32,361 duplicate testing rows, and 2,255 test feature vectors also present in training. The resulting predefined partition has 96,822 training and 47,716 testing rows with zero exact feature-vector overlap. These lower corrected scores supersede earlier contaminated results.

## Start here

- [IEEE-style research manuscript](output/pdf/xai_ids_ieee_paper.pdf)
- [Publication-readiness audit](PUBLICATION_READINESS_REPORT.md)
- [Beginner-to-end project handbook](PROJECT_HANDBOOK.md)
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

The complete benchmark is computationally heavier than the notebook. The notebook reads the committed result tables and is the easiest way to understand the workflow. In Google Colab, upload the locally generated `498R_COLAB_BUNDLE.zip` when prompted.

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
- Neighborhood stability is measured within the selected 40-flow audit subset.
- Deletion fidelity depends on the masking distribution and should not be interpreted causally.
- UNSW multiclass recall remains weak for several minority attacks.
- The project does not include a human analyst study or production drift evaluation.

## Publication

The committed PDF is anonymous for peer review. Before camera-ready submission, update `paper_metadata.yaml` with the authors and affiliations, add an archival repository URL or DOI, and apply the exact target venue template and page limit. The named authors must independently approve every claim and comply with the venue's disclosure and artifact policies.
