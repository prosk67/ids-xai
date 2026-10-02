# Publication Readiness Report

**Project:** Explainable Machine-Learning Intrusion Detection on CIC-IDS2017 and UNSW-NB15  
**Audit date:** 17 August 2026  
**Status:** Technically ready for peer-review preparation; author, venue, and archival-repository details remain to be supplied before submission.

## Executive verdict

The project now answers its research question with a reproducible, leakage-audited comparison of Random Forest and XGBoost across binary and multiclass intrusion detection. It does not claim to make the models intrinsically transparent. Instead, it addresses the operational black-box problem by attaching local TreeSHAP and LIME explanations, testing whether those rankings affect model confidence when important features are removed, measuring stability and latency, comparing against random-feature controls, and rejecting weak alert-level evidence.

The strongest publication-critical defect found during this audit was data leakage in the accessible UNSW-NB15 parquet mirror. The original files contained 78,519 exact duplicate training rows, 32,361 exact duplicate testing rows, and 2,255 test feature vectors also present in training after within-partition deduplication. The pipeline now removes all three forms of duplication before fitting or evaluation. The cleaned predefined partition contains 96,822 training rows and 47,716 testing rows, with zero exact feature-vector overlap. All UNSW metrics, explanations, figures, audits, and paper claims were regenerated from this cleaned partition.

## What was corrected

- Added explicit within-partition deduplication and cross-partition feature-overlap removal for UNSW-NB15, with counts written to the run manifest.
- Replaced the unsupported description of the UNSW split as “temporal” with “predefined.”
- Re-ran 40 predictive experiments: 2 datasets × 2 tasks × 2 models × 5 seeds.
- Re-ran paired TreeSHAP/LIME deletion tests on the same 40 flows per configuration.
- Re-ran random-ranking controls, neighborhood and repeated-LIME stability checks, latency measurements, masking-baseline sensitivity, calibration, class-weighting, class/error breakdowns, and feature ablations.
- Removed a class-weighting confound: weighted and unweighted models now use the same seed (42).
- Changed the alert explanation policy so weak random-control evidence receives `REVIEW_REQUIRED` or `UNSUPPORTED`, rather than being presented as automatically trustworthy.
- Corrected stale metrics and overclaims throughout the paper, handbook, traceability document, and notebook.
- Anonymized the manuscript metadata for peer review and moved identity fields to `paper_metadata.yaml`.
- Clarified that latency covers the explainer call only, neighborhood stability uses the 40-flow audit subset, and feature ablation is exploratory rather than nested model selection.

## Reproduced headline results

Mean macro-F1 over five seeds:

| Dataset | Task | Random Forest | XGBoost |
|---|---:|---:|---:|
| CIC-IDS2017 | Binary | 0.9965 | 0.9974 |
| CIC-IDS2017 | Multiclass | 0.9254 | 0.9324 |
| UNSW-NB15 (cleaned) | Binary | 0.8013 | 0.8156 |
| UNSW-NB15 (cleaned) | Multiclass | 0.5140 | 0.5083 |

TreeSHAP produced higher paired top-5 deletion fidelity than LIME in all 8 configurations. Every bootstrap confidence interval for the paired difference was above zero; Holm-adjusted one-sided Wilcoxon p-values were at most 0.00319, and paired effect sizes ranged from 0.382 to 0.974. TreeSHAP beat the random-ranking control in 8/8 configurations. LIME did so in 7/8; the failed case was CIC-IDS2017 binary Random Forest (mean lift 0.01393, Holm-adjusted p = 0.33763).

These results support a bounded conclusion: in this experimental protocol, TreeSHAP rankings are more faithful and more consistently distinguishable from random rankings than LIME rankings. They do not prove that either explanation is causal, universally valid, or sufficient for autonomous security decisions.

## How the black-box problem is addressed

The project provides five layers of transparency:

1. **Local attribution:** each prediction receives signed feature contributions from TreeSHAP and LIME.
2. **Behavioral validation:** top-ranked features are masked and the predicted-class probability change is measured.
3. **Negative control:** the same test is performed with random feature rankings.
4. **Reliability checks:** explanation stability, masking assumptions, class/error behavior, and latency are reported.
5. **Conservative alert evidence:** alert reports expose the predicted class, confidence, top contributors, deletion effect, random-control lift, empirical p-value, and an evidence tier.

This reduces opacity and makes individual predictions auditable. It does not “solve” black-box inference in the absolute sense: post-hoc feature attribution remains dependent on the model, background/masking distribution, dataset, and explanation method. The paper now states that limitation explicitly.

## Reproducibility and integrity controls

- Fixed five-seed predictive protocol: 42, 52, 62, 72, and 82.
- Training-only preprocessing, including categorical mappings and numeric medians.
- Dataset-file SHA-256 hashes and software versions in `outputs/run_manifest.json`.
- Exact paired source-flow matching between TreeSHAP and LIME enforced by `verify_outputs.py`.
- Unit test for removal of duplicate and cross-partition UNSW feature vectors.
- Raw datasets and trained models excluded from the publication repository; acquisition paths and hashes remain documented.
- Generated tables, figures, alert reports, and the PDF derive from machine-readable outputs.

## Remaining submission actions

The research artifacts are internally consistent, but the manuscript should not be sent to a venue until these external details are completed:

1. Select the target IEEE journal or conference and apply its exact template, page limit, taxonomy, and anonymization rules.
2. For double-blind review, retain `Anonymous Author(s)`. For camera-ready submission, replace it with the real author names, affiliations, emails, and acknowledgments in `paper_metadata.yaml`.
3. Publish the code and non-restricted artifacts to a stable repository, ideally with a Zenodo DOI, then add the archival URL to the paper. Do not redistribute datasets unless their licenses permit it.
4. Confirm the target venue's disclosure policy for AI-assisted writing and code.
5. Run the venue's PDF compliance checker, plagiarism/similarity check, and reference validator after final template conversion.
6. Have the named authors read and approve every claim; authors remain responsible for the manuscript and experimental validity.

## Deliverable map

- Final manuscript: `output/pdf/xai_ids_ieee_paper.pdf`
- Reproducible learning notebook: `PROJECT_WALKTHROUGH.ipynb`
- Complete beginner-to-end explanation: `PROJECT_HANDBOOK.md`
- Literature-to-experiment traceability: `LITERATURE_REQUIREMENTS_TRACEABILITY.md`
- Machine-readable experiment outputs: `outputs/`
- Integrity verifier: `verify_outputs.py`
- Editable anonymous/camera-ready identity metadata: `paper_metadata.yaml`

## Final assessment

The project is substantially stronger and more defensible than the pre-audit version. The corrected UNSW scores are lower, but they are scientifically more credible because the leakage has been removed. The paper now presents a transparent, testable contribution with negative controls and appropriately bounded claims. Subject to the venue-specific and authorship actions above, it is suitable for peer-review submission preparation.
