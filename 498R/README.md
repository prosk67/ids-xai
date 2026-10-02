# Quantitative XAI Benchmark for Network Intrusion Detection

This repository implements the CSE 498R project **A Quantitative Framework for Evaluating Explainable AI in Network Intrusion Detection Systems**. It trains Random Forest and XGBoost classifiers on CIC-IDS2017 and UNSW-NB15, evaluates binary and multiclass detection, and compares TreeSHAP with LIME using deletion fidelity, neighborhood stability, repeated-seed self-stability, and explanation latency.

## Reproduce

1. Create a Python 3.12 environment and install `requirements.txt`.
2. Obtain the source datasets under their original terms, convert them to the four Parquet paths listed in `config.yaml`, and verify their hashes against `outputs/run_manifest.json`.
3. Run:

```powershell
.\.venv\Scripts\python.exe run_benchmark.py --config config.yaml
.\.venv\Scripts\python.exe run_extended_audit.py --config config.yaml
.\.venv\Scripts\python.exe build_paper.py
```

The benchmark writes machine-readable tables, plots, model artifacts, a run manifest, and logs to `outputs/`. The paper builder writes the IEEE-style manuscript to `output/pdf/`.

The extended audit implements the additional checks requested by the literature review: majority/shallow-tree baselines, class-weight sensitivity, binary calibration, alternative masking references, LIME sample-budget sensitivity, class/error-type fidelity, and stability distributions. Its outputs are written to `outputs/extended/`.

## Explain one alert

The project includes an operational black-box mitigation layer. It loads a frozen model and produces both human-readable Markdown and machine-readable JSON with the prediction, signed TreeSHAP evidence, observed feature values, alert-specific top/bottom/random deletion tests, benchmark validation evidence, stability/latency context, and a conservative evidence tier.

```powershell
.\.venv\Scripts\python.exe explain_alert.py `
  --model outputs/models/UNSW_NB15_binary_xgboost.joblib `
  --input path/to/flows.csv `
  --row 0 `
  --output outputs/alert_reports/alert_0
```

The tier is `SUPPORTED_FOR_THIS_ALERT` only if TreeSHAP passed the corrected benchmark control and the alert's ranking beats randomized rankings at empirical p <= 0.05 under the declared training median/mode intervention. The report also displays the multi-reference sensitivity audit. Otherwise it requires review or marks the explanation unsupported. This makes the fitted decision auditable; it does not claim causality, intervention-invariant fidelity, or elimination of the general theoretical black-box problem.

## Experimental protocol

- **UNSW-NB15:** starts from the predefined 175,341-row training and 82,332-row testing files. The loader removes 78,519/32,361 within-file exact duplicates and 2,255 test feature vectors duplicated across partitions, leaving 96,822 training and 47,716 leakage-controlled test rows.
- **CIC-IDS2017:** the full 2.83M-flow, 28-feature public preprocessing is stratified-capped per class, deduplicated, and split 80/20. The source omits day/timestamp metadata, so temporal evaluation is not claimed.
- Preprocessing is fitted on training data only. Numeric missing/infinite values use training medians; categorical values use training-derived ordinal dictionaries with an explicit unseen value.
- Predictive results are repeated for five model seeds. Explanation metrics use the seed-42 frozen model and 40 identical, label-stratified instances for SHAP and LIME, with available binary TP/FP/TN/FN cases explicitly covered.
- Fidelity masks top-ranked features to training medians/modes and compares the confidence drop with random and bottom-ranked controls.
- Stability reports signed cosine similarity and top-5 Jaccard overlap for nearest same-prediction neighbors within each 40-flow evaluation subset. LIME additionally reports repeated-seed self-stability.
- Latency times explainer calls after one warm-up; it excludes ingestion, preprocessing, prediction, serialization, and report rendering.

## Data provenance

- CIC-IDS2017: Canadian Institute for Cybersecurity, University of New Brunswick. The local Parquet mirror contains all 2,830,743 published flows and 28 selected flow features.
- UNSW-NB15: UNSW Canberra. The predefined files contain 175,341 training and 82,332 testing flows before the documented duplicate/overlap audit.

Raw datasets and serialized models are excluded from the public repository. The generated `outputs/run_manifest.json` records paths, source row counts, cleaning counts, SHA-256 hashes, package versions, configuration, and hardware details.

## Publication status

The generated PDF is an anonymous peer-review manuscript. Before camera-ready submission, edit `paper_metadata.yaml` with the real author names and affiliation, add the archival repository URL/DOI, and apply the exact target venue template and page limit. See `PUBLICATION_READINESS_REPORT.md` for the completed audit and remaining submission-dependent items.
