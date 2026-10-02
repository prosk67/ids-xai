# Literature Review Requirements Traceability

## Purpose

This document maps the implementation plan and success criteria in `Lit_Review_498R__IDS_.pdf` to concrete project evidence. It prevents the final paper from claiming that a requested objective was completed when it was not measured.

## Overall result

- 11 literature requirements are completed with machine-readable evidence.
- 1 requirement is partially completed with a documented safety reason.
- Human usefulness remains future work because it requires a task-based analyst study.
- Universal causal explanation remains outside scope because post-hoc XAI cannot establish real-world causality.

## Requirement matrix

| Literature requirement | Status | Project evidence |
|---|---|---|
| CIC-IDS2017 and UNSW-NB15 | Complete | `config.yaml`, dataset hashes in `outputs/run_manifest.json` |
| Binary and multiclass tasks | Complete | 4 dataset/task combinations in all predictive outputs |
| Random Forest and gradient boosting | Complete | Random Forest and XGBoost artifacts in `outputs/models/` |
| Leakage-safe preprocessing | Complete | Training-only medians/category maps, CIC deduplication, UNSW within-file deduplication, and removal of cross-partition duplicate feature vectors |
| At least five stochastic seeds | Complete | Predictive seeds 42, 52, 62, 72, and 82 |
| Simple predictive baselines | Complete | Majority and depth-5 tree in `outputs/extended/predictive_baselines.csv` |
| Macro-F1 and class-aware metrics | Complete | `predictive_metrics.csv`, `per_class_metrics.csv`, confusion matrices, FPR/FNR, PR-AUC |
| Calibration diagnosis | Complete | Brier score, ECE, ten-bin reliability data, and calibration plot in `outputs/extended/` |
| Same instances for SHAP and LIME | Complete | 40 matched flows per configuration with identical source indices |
| Correct/error-case explanation coverage | Complete | Available binary TP/FP/TN/FN cases explicitly included; error summaries saved |
| Top-k deletion fidelity | Complete | k={1,3,5,10}, per-instance values in `outputs/fidelity.csv` |
| Random and bottom controls | Complete | 16 corrected control tests in `fidelity_control_tests.csv` |
| Alternative perturbation references | Complete | Median/mode, zero, marginal row, nearest same prediction, nearest opposite prediction |
| Neighborhood stability | Complete | Cosine and top-five Jaccard values plus Q10/median/Q90 summaries |
| LIME repeated-seed stability | Complete | Five LIME seeds on six repeated flows per configuration |
| LIME sampling ablation | Complete | 100, 350, and 700 perturbation samples with fidelity/stability/latency |
| Explanation latency distributions | Complete | Median, P90, P95, P99, throughput, warm-up, fixed host |
| Paired inference | Complete | 2,000 paired bootstraps, one-sided Wilcoxon tests, effect sizes, Holm correction |
| Feature-count curve | Complete | Top 5, 10, 20, and full feature sets; every reduced model retrained |
| Explanation results by class/error | Complete | `class_specific_fidelity.csv` and `error_type_fidelity.csv` |
| Balancing ablation | Partial | Class weighting versus none is complete; SMOTE is intentionally omitted because interpolation of ordinal-encoded categories can generate invalid flows |
| Analyst-facing explanation | Complete as software | `explain_alert.py` produces Markdown/JSON prediction, attribution, local random test, reference sensitivity, warnings, and evidence tier |
| Human usefulness study | Future work | Requires participants, task design, ethics/consent, and analyst outcome measures |
| Causal explanation | Outside scope | The system explains fitted-model behavior under declared interventions, not causality in the network |

## Literature success criteria

### 1. Classifiers outperform simple baselines

Satisfied. The depth-5 tree macro-F1 ranges from 0.249 to 0.955. Every full Random Forest/XGBoost context exceeds its matching shallow baseline. Majority macro-F1 ranges from 0.022 to 0.445.

### 2. Top-ranked perturbation exceeds controls

Satisfied for TreeSHAP under the prespecified training median/mode reference. TreeSHAP passes all 8 corrected top-versus-random tests. LIME passes 7 of 8.

The alternative-reference audit adds an important qualification: pooled TreeSHAP deletion is 0.392 for median/mode, 0.418 for zero, 0.386 for a marginal training row, 0.310 for a nearest opposite prediction, and -0.004 for a nearest same prediction. Fidelity is therefore intervention-specific rather than an unconditional property.

### 3. Stability does not collapse

Partially contextual rather than a universal pass/fail. TreeSHAP mean neighborhood cosine ranges from 0.543 to 0.947; LIME ranges from 0.472 to 0.801. Increasing LIME samples from 100 to 700 improves pooled repeated-seed cosine from 0.610 to 0.855 while increasing latency.

### 4. Timing supports a declared use mode

Satisfied as explainer-call timing, not end-to-end service latency. XGBoost TreeSHAP has 1.6-23.5 ms median latency on the measured host. UNSW Random Forest TreeSHAP is slower and is recommended for lower-volume or analyst-on-demand use.

### 5. Results are reproducible

Satisfied locally. Configuration, seeds, source hashes, cleaning counts, package versions, hardware, preprocessing, frozen seed-42 models, raw metric tables, figures, tests, and the generated paper are stored in the project. Raw datasets and serialized models are excluded from the public release and regenerated locally.

## Black-box problem claim

The project does not claim to eliminate the universal black-box problem. Its validated contribution is:

> The framework converts opaque tree-based IDS predictions into auditable, quantitatively tested, and rejectable per-alert model evidence under declared interventions.

This claim is supported by:

- Matched TreeSHAP/LIME comparisons
- Global corrected random-ranking controls
- Alert-specific randomization tests
- Multiple masking-reference strategies
- Neighborhood and repeated-seed stability
- Probability calibration diagnosis
- Class/error analysis
- Explicit `SUPPORTED_FOR_THIS_ALERT`, `REVIEW_REQUIRED`, and `UNSUPPORTED` tiers
- Warnings that attribution is not causality and may not transfer under drift

## Verification

Run the complete checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe verify_outputs.py
```

The current project passes five unit tests and all output, extended-audit, manifest, and paper checks.
