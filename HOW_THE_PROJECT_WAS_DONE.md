# How the Project Was Done

**A Quantitative Framework for Evaluating Explainable AI in Network Intrusion Detection Systems** (CSE 498R)

This document explains the whole project from the research problem to the final paper: what was built, why each step exists, how every number was produced, what changed between versions, and how to run it yourself (locally or on Google Colab). Every result quoted here comes from the files in `outputs/`, produced by the current code and configuration.

---

## Contents

1. [The project in one paragraph](#1-the-project-in-one-paragraph)
2. [The problem and the research questions](#2-the-problem-and-the-research-questions)
3. [How the project evolved](#3-how-the-project-evolved)
4. [Pipeline overview](#4-pipeline-overview)
5. [Data: sources, cleaning and leakage control](#5-data-sources-cleaning-and-leakage-control)
6. [Preprocessing](#6-preprocessing)
7. [Models and validation-based selection](#7-models-and-validation-based-selection)
8. [Measuring detection performance](#8-measuring-detection-performance)
9. [Generating explanations](#9-generating-explanations)
10. [Measuring whether explanations are faithful](#10-measuring-whether-explanations-are-faithful)
11. [Stability and latency](#11-stability-and-latency)
12. [Statistics](#12-statistics)
13. [Robustness audits](#13-robustness-audits)
14. [Results](#14-results)
15. [Per-alert transparency reports](#15-per-alert-transparency-reports)
16. [How the paper is produced and checked](#16-how-the-paper-is-produced-and-checked)
17. [Running the project locally](#17-running-the-project-locally)
18. [Running the project on Google Colab](#18-running-the-project-on-google-colab)
19. [Limitations and what is not claimed](#19-limitations-and-what-is-not-claimed)
20. [File map](#20-file-map)
21. [Key references](#21-key-references)

---

## 1. The project in one paragraph

Machine-learning intrusion detection systems (IDSs) can label network flows accurately while giving analysts no defensible reason for an alert. Explanation methods such as SHAP and LIME produce attractive feature-importance charts, but a chart is not evidence that the highlighted features actually drive the model. This project builds an end-to-end pipeline that trains Random Forest and XGBoost detectors on CIC-IDS2017 and UNSW-NB15 (binary and multiclass), explains their decisions with TreeSHAP and LIME, and then **tests** those explanations quantitatively: does removing the features an explainer ranks highest lower the model's confidence more than removing random features, the least important features, or simply the features the model uses everywhere? Do attributions track the effect of masking? Do explanations change when the model learned nothing? Are they stable and fast enough to use? The answer, under a leakage-controlled protocol with 200 matched flows and five model seeds, is that TreeSHAP's explanations are consistently more faithful than LIME's, and that most of LIME's per-alert rankings add little beyond a fixed global importance list. The project ends with a per-alert report that accepts an explanation only when it passes these tests.

---

## 2. The problem and the research questions

### 2.1 Why accuracy is not enough

A network IDS classifies flows (summaries of connections: durations, byte and packet counts, flags, timing statistics) as normal or malicious. A model can reach 99% accuracy and still be hard to trust in a security operations center. An analyst needs to know *why* an alert fired, whether that reason is consistent for similar traffic, and whether it can be produced quickly. Without that, alerts are hard to triage and errors are hard to spot.

### 2.2 Why "use SHAP/LIME" is not a solution by itself

Post-hoc explainers always produce *some* ranking of features. Prior IDS papers mostly show these rankings as plots. Explanation research (Adebayo et al. 2018; Yeh et al. 2019; Bhatt et al. 2020) shows that visually convincing explanations can fail basic checks. So the question this project asks is not "what do the explanations look like?" but "can the explanations be measured and, when weak, rejected?"

### 2.3 Research questions

| RQ | Question |
|---|---|
| RQ1 | How accurately and consistently do RF and XGBoost detect normal and malicious traffic, in binary and multiclass settings? |
| RQ2 | When an explainer's top-ranked features are masked, does confidence drop more than for random, bottom-ranked, or globally important features, and does attribution mass track the effect of masking? |
| RQ3 | Do nearby flows with the same prediction get similar explanations? |
| RQ4 | What latency do the explainers add, and which operating mode (real-time, analyst-on-demand, offline) does that support? |
| RQ5 | Do the conclusions hold across datasets, tasks, models, feature counts and random seeds? |

These came from the proposal (`proposal.pdf`: fidelity, stability, speed) and were refined in the literature review (`Lit_Review_498R__IDS_.pdf`, §1.3), which also set the success criteria used throughout: beat simple baselines; top-ranked masking must beat random and bottom controls; stability must not collapse; timing must support a stated use mode; everything must be reproducible from a config file and recorded seeds.

---

## 3. How the project evolved

| Stage | What happened |
|---|---|
| Proposal | Problem, three metrics (fidelity, stability, speed), CIC-IDS2017 + UNSW-NB15, RF + gradient boosting, SHAP + LIME. |
| Literature review | Six papers reviewed; research questions RQ1-RQ5, implementation plan, ablation list and success criteria defined. |
| First implementation (7 Aug 2026) | Full pipeline and a five-page paper. It used the UNSW mirror as supplied, which later turned out to contain duplicated and overlapping rows. |
| Leakage correction (17 Aug 2026) | The UNSW mirror contained 78,519 duplicate training rows, 32,361 duplicate test rows and 2,255 test feature vectors also present in training. These were removed; all results were regenerated (UNSW scores dropped, as expected once leakage is gone). Per-alert reports and the extended audit were added. |
| Six-page draft (22 Aug 2026) | Paper expanded with ethics, artifact availability and more related work. The source code for this version was not kept, so the paper could not be regenerated. |
| Consolidation and rework (2 Oct 2026) | Duplicate project copies merged into one git repository; the paper made reproducible from source; then a second audit and protocol rework (below). |

### 3.1 What the October rework found and changed

1. **The paper could not be rebuilt.** The six-page PDF had no matching source. The paper generator was ported to the six-page text, and portability bugs were fixed. Fonts had been loaded from a Windows-only path, silently falling back to non-embedded fonts, which IEEE rejects. Every font is now embedded, and the figures are rebuilt from the saved CSVs.
2. **The verifier asserted results, not integrity.** `verify_outputs.py` required "TreeSHAP passes 8/8, LIME passes 7/8, every interval above zero". Those are research outcomes. It now checks structure and internal consistency only (sample sizes from `config.yaml`, identical flows across explainers and seeds, CI ordering, Holm ≥ raw p, one selected candidate per context, embedded fonts).
3. **The old result was fragile.** Rerunning the unchanged 40-flow, single-seed protocol on Linux changed one verdict: LIME passed 6/8 random controls instead of 7/8, because XGBoost's multithreaded floating point differs slightly across platforms. Random Forest reproduced exactly. The conclusion depended on 40 flows explained under a single model, which was too little evidence.
4. **Selection used no validation data.** Hyperparameters were fixed by hand, and the "best feature count" was read off the test set (the paper admitted this).
5. **Missing checks from the evaluation literature.** There was no control against a global importance ranking, no metric covering the whole attribution vector, no sanity check, and no audit of whether failing classes are even learnable.

Changes made, each as a separate tested commit (see `git log`):

- Stratified 15% validation split carved from the training data, used for every selection decision.
- 8-candidate hyperparameter grid per model, chosen by validation macro-F1 (prespecified settings first and winning ties).
- Feature count for the ablation chosen on validation, not test.
- **Global-importance control**, testing top-ranked masking against the model's own fixed importance ranking.
- **Faithfulness correlation** (Bhatt et al. 2020).
- 200 fixed flows explained under **all five** model seeds (previously 40 flows, one seed).
- **Data-randomization sanity check** (Adebayo et al. 2018).
- **Label-conflict audit.**
- Isotonic recalibration and an interventional-TreeSHAP background ablation.
- Robustness audits enlarged from 12 and 6 flows to 100 and 20.
- All masked variants of a flow scored in one model call. Results are unchanged (pinned by a test), and runs are much faster.
- Every number in the paper computed from output files; positioned against the closest prior work, E-XAI (Arreche et al. 2024).

---

## 4. Pipeline overview

```text
Raw parquet files (CIC-IDS2017 shards, UNSW-NB15 train/test)
   │  SHA-256 hashes recorded in outputs/run_manifest.json
   ▼
Load + clean  ── CIC: per-class cap 25,000, dedup, stratified 80/20 split
              ── UNSW: predefined split, within-partition dedup, remove test rows seen in training
   ▼
Training-only preprocessing (medians, category maps, masking baseline)
   ▼
Validation split (15% of training) ── hyperparameter grid ── feature-count choice
   ▼
Train RF and XGBoost × {binary, multiclass} × 5 seeds on all training rows
   ▼
Predictive metrics (test set)                    → predictive_metrics.csv, per_class_metrics.csv
   ▼
Choose 200 explanation flows once (seed-42 model)
   ▼
For each of 5 model seeds × {TreeSHAP, LIME}:
   attributions → deletion vs random/bottom/global controls,
   faithfulness correlation, neighbor stability, latency     → fidelity.csv.gz, explanation_flows.csv.gz, ...
   ▼
Paired statistics (bootstrap, Wilcoxon, Holm)               → paired_comparisons.csv, fidelity_control_tests.csv
   ▼
Extended audit: baselines, weighting, calibration + recalibration, masking references,
LIME budget, SHAP background, randomization sanity, label conflicts → outputs/extended/
   ▼
Paper generator (reads only output files)                  → output/pdf/xai_ids_ieee_paper.pdf
   ▼
Integrity verifier + unit tests
```

Entry points: `run_benchmark.py` → `src/xai_ids/benchmark.py`; `run_extended_audit.py` → `extended_audit.py`; `build_paper.py` → `paper.py` + `paper_figures.py`; `explain_alert.py` → `transparency.py`; `verify_outputs.py`.

---

## 5. Data: sources, cleaning and leakage control

The raw datasets are not redistributed. The four parquet files are declared in `config.yaml`, and their SHA-256 hashes are in `outputs/run_manifest.json` → `files`; a mismatch means a different input.

### 5.1 CIC-IDS2017

- The accessible representation has 2,830,743 flows, 28 selected features and 15 labels (BENIGN plus 14 attack types). Day/timestamp metadata is missing, so a temporal split is impossible.
- To keep computation manageable while keeping rare attacks, at most **25,000 rows per label** are sampled (`load_cic`). BENIGN appears in both shards, so a second cap is applied after concatenation.
- **15,103 exact duplicates** are removed after sampling, leaving **124,513 rows**.
- A **stratified 80/20 split** gives 99,610 training and 24,903 test rows.
- Binary task: BENIGN → Normal, everything else → Attack. Multiclass keeps all 15 labels.

### 5.2 UNSW-NB15

- The predefined partition has 175,341 training and 82,332 testing rows. In the accessible mirror, the file names are swapped relative to these counts, and `config.yaml` maps them correctly.
- **Leakage found and removed** (`clean_unsw_partitions`):
  - 78,519 exact duplicate rows inside the training file;
  - 32,361 exact duplicate rows inside the test file;
  - 2,255 test rows whose feature vector (ignoring labels) also appears in training.
- The clean split is **96,822 training / 47,716 test rows** with zero exact feature-vector overlap. A unit test pins this behaviour.
- 34 features; binary (Normal vs Attack) and 10-class tasks (Normal plus 9 attack families).

### 5.3 Label-conflict audit (new)

Even after cleaning, UNSW contains rows whose *exact* feature vector also appears with a *different* label. No model can separate such rows, because the inputs are identical. `label_conflict_rows` measures this per class and computes an accuracy ceiling (the majority label in each identical-feature group):

| UNSW multiclass class | Share of test rows in conflicting groups | Seed-42 recall (RF / XGB) |
|---|---:|---|
| Backdoor | 79% | 0.521 / 0.547 |
| Analysis | 77% | 0.436 / 0.396 |
| DoS | 18% | 0.335 / 0.404 |
| all other classes | ≤ 6% | 0.63-0.92 |

The ceiling is 0.980 accuracy on test and 0.951 on training (CIC: ≈ 1.000). The classes the models fail on are exactly the ones whose features cannot distinguish them, so this is a dataset limitation, not a model deficit. A third-party estimate of 17.4% conflicting training rows was checked against this project's cleaned data and found to be 6.3%; only the measured figure is used.

---

## 6. Preprocessing

`TabularPreprocessor` is fitted on training data only, then applied unchanged to test data:

- **Numeric features:** ±∞ → missing → filled with the *training* median.
- **Categorical features** (e.g. UNSW `proto`, `service`, `state`): ordinal codes from a *training* dictionary; categories unseen in training become −1. A unit test covers this.
- **Masking baseline:** the training median (numeric) or mode (categorical) of every feature. This is the default "removed" value for the deletion tests.
- Label and attack-category columns are excluded from the features.
- No resampling. SMOTE was deliberately not used, because interpolating ordinal-encoded protocol/service codes creates impossible flows. Class weighting is used instead and audited (§13).

---

## 7. Models and validation-based selection

### 7.1 Model families

- **Random Forest** (scikit-learn): `class_weight="balanced_subsample"`, all cores.
- **XGBoost**: histogram trees, balanced sample weights, row/column subsampling 0.85, log-loss objective.

Both are strong tabular models with exact, fast TreeSHAP support.

### 7.2 Validation split

`split_validation` takes a stratified 15% of the training rows. Classes with fewer than 3 rows stay entirely in the fit part, because XGBoost multiclass needs every class present. A unit test checks this using a 2-row class. **The test partition is never used to choose anything.**

### 7.3 Hyperparameter grid

`select_hyperparameters` fits each candidate on the 85% fit split and scores macro-F1 on the 15% validation split. The prespecified configuration is candidate 0 and wins ties, so tuning never changes a model without evidence. The chosen setting is refitted on **all** training rows for each of the five seeds.

| Model | Prespecified | Grid (8 candidates) |
|---|---|---|
| RF | 120 trees, depth 18, min leaf 2 | depth {18, none} × min leaf {1, 2} × max features {sqrt, 0.5} |
| XGBoost | 160 trees, depth 7, η 0.08 | trees {160, 400} × depth {7, 10} × η {0.08, 0.15} |

The selected settings (stored in the manifest) differed from the prespecified ones in all 8 contexts. RF always chose `max_features=0.5`. On test data, tuning raised UNSW binary macro-F1 by about 0.01 and UNSW multiclass by about 0.02, without ever looking at the test set.

### 7.4 Seeds

Every dataset × task × model is trained with seeds 42, 52, 62, 72 and 82: 8 contexts × 5 seeds = 40 predictive runs. The CIC split itself uses seed 42 throughout.

---

## 8. Measuring detection performance

`predictive_metrics` reports accuracy, balanced accuracy, macro-F1, weighted-F1, per-class precision/recall/F1/support and confusion matrices. Binary tasks also get FPR, FNR and PR-AUC, with Attack as the positive class.

**Macro-F1 is the headline metric**, because it weights every class equally. Accuracy hides rare-attack failures: UNSW multiclass reaches 0.68 accuracy but only 0.53 macro-F1.

Simple baselines (extended audit) put the numbers in context: a majority-class predictor reaches macro-F1 0.022-0.445, and a depth-5 decision tree 0.250-0.955.

---

## 9. Generating explanations

- **TreeSHAP** (`shap.TreeExplainer`, default path-dependent mode) gives exact Shapley-value attributions for tree ensembles. The vector for the *predicted* class is used, and signs are kept.
- **LIME** (`LimeTabularExplainer`) samples 350 perturbations around the flow (background: a 5,000-row stratified training sample, continuous features discretized, categorical features declared) and fits a weighted linear surrogate. Its weights for all features are mapped back to the original feature order.

### 9.1 Which flows are explained

`explanation_indices` picks **200 test flows per configuration**, stratified by label. For binary tasks it guarantees that every available TP/FP/TN/FN type is included. The flows are chosen **once**, from the seed-42 model, and then reused for both explainers and **all five model seeds**. This matching makes paired statistics valid; the verifier checks it.

Raw attribution vectors for the seed-42 models are saved in `attributions.csv.gz`.

---

## 10. Measuring whether explanations are faithful

### 10.1 Deletion fidelity

For flow *x*, predicted class *c*, ranking *r* and baseline *b*:

```text
D_k(x, r) = p(c | x) − p(c | x with features r[1..k] replaced by b)
```

The fidelity score is the mean of D_k over k = 1, 3, 5, 10. If an explanation is faithful, masking its top features should make the model much less confident.

### 10.2 Negative controls

The same score is computed for three alternative rankings of the same flow:

| Control | Ranking | What beating it shows |
|---|---|---|
| Random | a random permutation of the features | the explanation is better than chance |
| Bottom | the explainer's own ranking reversed | the head of the ranking matters more than the tail |
| **Global** (new) | the model's own `feature_importances_`, the same for every flow | the *per-alert* explanation adds information beyond a fixed importance list |

The global control is the strictest. An explainer that only ever highlights the features the model uses everywhere is not explaining individual alerts.

### 10.3 Faithfulness correlation (new)

Deletion only tests the top of the ranking. Faithfulness correlation (Bhatt et al. 2020) tests the whole vector. For each flow, 50 random subsets of 5 features are masked, and the Pearson correlation is computed between each subset's summed attribution and the confidence drop it causes. High correlation means the attribution magnitudes predict the model's behaviour. It uses a separate random stream, so the random-control results are unaffected.

### 10.4 Flip rate

This is the share of top-k maskings that change the predicted class.

### 10.5 Sanity check: data randomization (new)

`_randomization_rows` trains a second model on **permuted labels**, so it has learned nothing about attacks, and explains the same 100 audit flows with both models. If the explanations of the real and the meaningless model were similar, the explainer would be reflecting the input data rather than the learned decision function (Adebayo et al. 2018). Similarity is measured by signed cosine and top-5 Jaccard overlap.

### 10.6 Masking reference sensitivity

Masking creates synthetic inputs, which can be unrealistic (Hooker et al. 2019; Rong et al. 2022). The audit repeats deletion with five references: training median/mode, zero, a random training row, the nearest training flow with the *same* prediction, and the nearest flow with the *opposite* prediction. Fidelity is therefore reported as an intervention-specific quantity, not an absolute property.

---

## 11. Stability and latency

- **Neighborhood stability.** Within each 200-flow set, each flow is paired with its nearest flow (standardized Euclidean distance) that has the same prediction. The pair's explanations are compared by signed cosine and top-5 Jaccard overlap.
- **LIME self-stability.** The same 6 flows are explained with 5 different LIME seeds, and the pairwise cosine is computed.
- **Latency.** The explainer call alone is timed per flow after one warm-up and summarized by median, P90, P95, P99 and throughput. It excludes loading, preprocessing and report rendering.

Stability is not fidelity: an explanation can be stable and wrong. That is why both are measured.

---

## 12. Statistics

- **Paired design.** TreeSHAP and LIME explain the *same* flows. Each flow's score is averaged over k and the five model seeds, and the TreeSHAP − LIME difference is taken per flow.
- **Bootstrap.** The 200 per-flow differences are resampled 2,000 times, and the 2.5th/97.5th percentiles form a 95% confidence interval.
- **Wilcoxon signed-rank tests.** One-sided for the prespecified deletion hypothesis (TreeSHAP > LIME) and the control tests; two-sided for faithfulness correlation, which was added after the original protocol.
- **Holm correction** (`holm_adjust`) controls family-wise error within four families: 8 deletion comparisons, 8 faithfulness comparisons, 16 top-vs-random tests and 16 top-vs-global tests.
- **Pass rule for a control.** The Holm-adjusted p is below 0.05 *and* the bootstrap lower bound is above 0.
- **Effect size.** Paired d_z is the mean difference divided by the standard deviation of the differences.

These describe the selected flows under the declared intervention, not every future network. Flows from the same session may be correlated, so the intervals may be slightly optimistic.

---

## 13. Robustness audits

`run_extended_audit.py` (results in `outputs/extended/`):

| Audit | Result |
|---|---|
| Simple baselines | Depth-5 tree macro-F1 0.250-0.955, majority 0.022-0.445; both full model families beat them everywhere. |
| Class weighting vs none (same seed) | Weighting helps in 4/8 contexts, so it is not universally beneficial. |
| Calibration (binary) | ECE 0.0015-0.0017 on CIC, 0.109-0.111 on UNSW. |
| Isotonic recalibration (fitted on validation) | CIC mean ECE 0.0017 → 0.0006; UNSW 0.111 → 0.108. UNSW's predefined test partition is distributed differently from training: validation macro-F1 exceeds test macro-F1 by about 0.10, so calibration learned on training data transfers poorly. |
| Masking reference (100 flows) | TreeSHAP mean deletion: median/mode 0.346, zero 0.424, random row 0.402, nearest-opposite 0.396, nearest-same 0.013. |
| LIME budget (20 flows × 5 repeats) | 100 → 700 samples raises self-stability 0.591 → 0.843, deletion fidelity 0.212 → 0.248 (still below TreeSHAP) and median latency 17.8 → 24.0 ms. |
| SHAP background (XGBoost) | Interventional TreeSHAP (100-row background) 0.299 vs path-dependent 0.308 mean deletion, with the ordering varying by dataset. Interventional TreeSHAP still beats LIME in 4/4 configurations. |
| Randomization sanity check | Real vs permuted-label explanations: mean top-5 Jaccard 0.13 (TreeSHAP), 0.10 (LIME), ≤ 0.18 everywhere. Signed cosine is low except CIC multiclass RF (0.51). Both explainers pass. |
| Label conflicts | See §5.3. |
| Class / error-type fidelity | Saved per class and per TP/FP/TN/FN (`class_specific_fidelity.csv`, `error_type_fidelity.csv`). |

The masking audit has an important consequence. Replacing features with the nearest *same-prediction* flow barely changes confidence, while every other reference does. Deletion fidelity measures how the model responds to a stated intervention; it is not a causal statement about the network.

---

## 14. Results

### 14.1 Detection (RQ1): mean over 5 seeds

| Dataset | Task | RF macro-F1 | XGB macro-F1 | Earlier (Aug, untuned) RF / XGB |
|---|---|---:|---:|---|
| CIC-IDS2017 | Binary | 0.9975 | 0.9977 | 0.9965 / 0.9974 |
| CIC-IDS2017 | Multiclass | 0.9299 | 0.9334 | 0.9254 / 0.9324 |
| UNSW-NB15 (clean) | Binary | 0.8148 | 0.8201 | 0.8013 / 0.8156 |
| UNSW-NB15 (clean) | Multiclass | 0.5323 | 0.5292 | 0.5140 / 0.5083 |

Seed standard deviations are at most 0.0045. UNSW binary: FPR about 0.24-0.25, FNR about 0.055-0.058, PR-AUC 0.939-0.941. CIC multiclass errors concentrate in Web Attack classes (XSS recall 0.54-0.69, Brute Force 0.61-0.63), which have small support.

### 14.2 Explanation quality (RQ2, RQ3, RQ4)

Mean over 200 flows × 5 model seeds (`explanation_summary.csv`):

| Context | Deletion S / L | Faithfulness corr. S / L | Neighbor cosine S / L | Median ms S / L |
|---|---|---|---|---|
| CIC bin RF | 0.293 / 0.206 | 0.44 / 0.19 | 0.946 / 0.825 | 8.6 / 29.4 |
| CIC bin XGB | 0.243 / 0.068 | 0.55 / 0.20 | 0.926 / 0.809 | 0.8 / 6.9 |
| CIC multi RF | 0.501 / 0.391 | 0.56 / 0.29 | 0.973 / 0.825 | 20.5 / 29.4 |
| CIC multi XGB | 0.363 / 0.176 | 0.46 / 0.15 | 0.935 / 0.706 | 8.9 / 13.4 |
| UNSW bin RF | 0.342 / 0.245 | 0.56 / 0.25 | 0.857 / 0.502 | 69.0 / 29.5 |
| UNSW bin XGB | 0.290 / 0.164 | 0.60 / 0.27 | 0.747 / 0.547 | 1.7 / 7.0 |
| UNSW multi RF | 0.399 / 0.292 | 0.57 / 0.28 | 0.880 / 0.638 | 139.1 / 29.7 |
| UNSW multi XGB | 0.341 / 0.213 | 0.52 / 0.22 | 0.747 / 0.567 | 27.5 / 14.4 |

(S = TreeSHAP, L = LIME.)

**Paired comparisons.** TreeSHAP beats LIME in **8/8** configurations on deletion fidelity (differences 0.087-0.187, d_z 0.52-1.03, Holm-adjusted p ≤ 9 × 10⁻¹⁷) and in **8/8** on faithfulness correlation (differences 0.25-0.35).

**Controls:**

| Control | TreeSHAP passes | LIME passes |
|---|---:|---:|
| Random ranking | 8/8 | 7/8 (fails CIC binary XGB) |
| Global importance ranking | **8/8** (lift 0.063-0.283) | **3/8** (fails all four RF configurations and UNSW binary XGB) |

This is the main new finding. For five of eight configurations, an analyst would learn as much from the model's single global importance list as from LIME's per-alert explanation. TreeSHAP's per-flow rankings always add information.

**Latency.** XGBoost + TreeSHAP takes 0.8-27.5 ms median per explanation. RF + TreeSHAP is slow on UNSW multiclass (139 ms median, 145 ms P95), where LIME (about 30 ms) is faster but less faithful. Latency depends on the model-explainer pair, not the explainer name.

### 14.3 Feature count (RQ5)

Choosing the smallest count within 0.005 validation macro-F1 of the full model picks a reduced set in 7/8 contexts (10-20 features typically, 5 for UNSW multiclass RF). The chosen-minus-full *test* macro-F1 is −0.012 to +0.003. Feature reduction is viable, but the right count is context-specific.

### 14.4 Answers

- **RQ1:** Tuned RF and XGBoost beat baselines and are nearly tied. UNSW multiclass is limited by label conflicts in the data.
- **RQ2:** TreeSHAP is more faithful than LIME on every metric and passes every control. LIME often fails the global control.
- **RQ3:** TreeSHAP is more stable across neighboring flows. LIME's stability improves with more samples.
- **RQ4:** XGBoost + TreeSHAP suits near-real-time use. UNSW RF + TreeSHAP suits analyst-on-demand use.
- **RQ5:** The TreeSHAP advantage holds across both datasets, both tasks, both models and five seeds. Calibration, balancing, feature count and masking magnitude remain context-dependent.

---

## 15. Per-alert transparency reports

`explain_alert.py` turns the evaluation into an analyst-facing report (Markdown + JSON) for one flow:

- predicted class and every class probability;
- the top signed TreeSHAP features with observed values;
- a deletion test of this alert's ranking against its bottom ranking and 200 random rankings, giving an empirical one-sided p-value;
- benchmark context: whether TreeSHAP passed the global random control for that model, its stability, latency and masking-reference sensitivity;
- a fixed warning that attribution is not causality.

It assigns an **evidence tier**:

- `SUPPORTED_FOR_THIS_ALERT`: benchmark control passed and this alert's ranking beats random rankings at p ≤ 0.05.
- `REVIEW_REQUIRED`: benchmark control passed, but this alert did not clear the local test.
- `UNSUPPORTED`: the explainer failed benchmark-level validation.

The regenerated example (`outputs/alert_reports/unsw_binary_xgb_example.md`, the first UNSW test flow predicted as Attack) is labelled `REVIEW_REQUIRED`. Its top-feature deletion score (0.062) is barely above the mean of 200 random rankings (0.056, p = 0.42). This is the system working as intended: it declines to present a weak explanation as trustworthy.

---

## 16. How the paper is produced and checked

`build_paper.py` regenerates all figures from the CSVs and lays out a two-column IEEE-style PDF with ReportLab (`src/xai_ids/paper.py`). **No result in the paper is typed by hand.** Every number, table and many sentences (pass counts, failing configurations, conflict shares, top features) are computed from `outputs/` at build time, so a rerun updates the paper automatically. Text whose meaning depends on a result's direction (for example, whether recalibration helps) was rewritten after the October rerun and checked against the numbers.

Quality control:

1. `verify_outputs.py` checks every required file; table sizes derived from `config.yaml`; identical flows across explainers and seeds; CI ordering and Holm consistency; one selected tuning candidate and feature count per context; manifest hashes and UNSW cleaning counts; and that the paper has 6-8 US-Letter pages, all required sections, references and **only embedded fonts**.
2. Unit tests (`tests/test_benchmark.py`, 21 tests) cover cleaning, preprocessing, validation split, tuning ties, feature-count choice, batched fidelity equality, controls, faithfulness correlation, flow matching, summary aggregation, calibration error, label conflicts, the sanity check and the alert-report context.
3. Every rendered page is inspected for clipped tables, broken figures and orphaned headings.

The paper is anonymous for double-blind review. Author details go in `paper_metadata.yaml` for the camera-ready version.

---

## 17. Running the project locally

Requirements: Python 3.12, about 8 GB RAM, and the four parquet files.

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

# place the data (or symlink an existing data folder):
#   data/raw/cic_train_0000.parquet  data/raw/cic_train_0001.parquet
#   data/raw/unsw_test.parquet       data/raw/unsw_train.parquet

python -B -m unittest discover -s tests       # 21 tests
python -B run_benchmark.py --config config.yaml      # ~28 min on 16 cores
python -B run_extended_audit.py --config config.yaml # ~15 min on 16 cores
python -B build_paper.py                              # → output/pdf/xai_ids_ieee_paper.pdf
python -B verify_outputs.py
```

To explain one alert:

```bash
python explain_alert.py --model outputs/models/UNSW_NB15_binary_xgboost.joblib \
  --input path/to/flows.csv --row 0 --output outputs/alert_reports/alert_0
```

`PROJECT_WALKTHROUGH.ipynb` reads the saved results step by step without rerunning anything.

---

## 18. Running the project on Google Colab

Use **`COLAB_RUN.ipynb`**. It performs the whole run on a Colab runtime.

**1. Prepare two zip files** on the computer that has the project:

```bash
# in the project folder (packs exactly the committed files)
git archive -o ids-xai.zip HEAD
# in the folder that contains data/raw/
zip -r ids-xai-data.zip data/raw
```

**2. Upload both** to Google Drive in a folder `MyDrive/ids-xai/`. The data zip is about 220 MB.

**3. Open `COLAB_RUN.ipynb` in Colab:** File → Upload notebook, or open it from Drive with "Open with → Google Colaboratory".

**4. Choose a mode** in the first code cell:
- `QUICK = False` runs the full protocol. Expect about 2-4 hours on a free 2-vCPU runtime (28 + 15 min on 16 cores here). A Colab Pro high-CPU runtime is much faster.
- `QUICK = True` runs a 15-30 minute smoke test (one seed, 40 flows, no grid search). It exercises every step but does not reproduce the paper's numbers, so the paper and verifier are skipped.

**5. Run all cells:** Runtime → Run all. The notebook:
1. mounts Drive and unzips the project and data to `/content/ids-xai`;
2. installs the pinned packages (`%pip install -r requirements.txt`);
3. checks the parquet SHA-256 hashes against the recorded run;
4. runs the unit tests;
5. runs the benchmark and the extended audit;
6. builds the paper and runs the verifier (full mode);
7. copies results and the PDF to `MyDrive/ids-xai/results-<timestamp>/`, because the Colab disk is erased when the session ends.

**Tips:**
- If pip replaces numpy or pandas, Colab may ask you to restart. Use Runtime → Restart session, re-run the settings cell, then continue from the hash check.
- Free sessions disconnect when idle. Keep the tab active during a full run, or use quick mode first.
- To just *read* the results, unzip `ids-xai.zip` in Colab and open `PROJECT_WALKTHROUGH.ipynb`. It finds `/content/ids-xai` automatically, and no data is needed.

The quick-mode path of the notebook was exercised end to end on a clean `git archive` copy before release.

---

## 19. Limitations and what is not claimed

- **No causal claims.** Attributions explain the fitted model under a stated masking intervention, not the network. The nearest-same-prediction reference shows how much fidelity depends on the intervention.
- **Datasets are testbeds.** CIC has no timestamps, so its split is stratified, not temporal. The available CIC mirror predates the Engelen et al. (2021) relabelling, which corrected more than 20% of flows. UNSW's predefined split is one testbed, and its partitions differ in distribution. Neither represents a live network.
- **Statistics describe the selected flows.** Session-level dependence may make intervals optimistic. Five seeds capture training randomness, not alternative captures or pipelines.
- **Scope.** Two tree models and two post-hoc explainers were evaluated. There are no deep models, counterfactuals, drift or adversarial tests, and no analyst user study.
- **Platform sensitivity.** XGBoost results vary slightly across operating systems. The larger 200-flow, five-seed protocol was adopted because the earlier 40-flow verdicts were not robust to this.

The defensible claim is this: **the framework turns opaque tree-IDS predictions into auditable, quantitatively tested, and rejectable per-alert evidence within the tested data and models.**

---

## 20. File map

```text
config.yaml                      experiment contract (data paths, grids, seeds, sample sizes)
run_benchmark.py                 → src/xai_ids/benchmark.py   data, models, XAI, statistics
run_extended_audit.py            → src/xai_ids/extended_audit.py   robustness audits
explain_alert.py                 → src/xai_ids/transparency.py    per-alert reports
build_paper.py                   → src/xai_ids/paper.py, paper_figures.py
verify_outputs.py                integrity checks
tests/test_benchmark.py          unit tests
COLAB_RUN.ipynb                  run everything on Google Colab
PROJECT_WALKTHROUGH.ipynb        guided reading of the saved results
HOW_THE_PROJECT_WAS_DONE.md      this document
README.md                        short overview and headline results
PUBLICATION_READINESS_REPORT.md  audit history and submission checklist
LITERATURE_REQUIREMENTS_TRACEABILITY.md   literature-review requirements → evidence
docs/superpowers/plans/          the October rework plan
outputs/
  predictive_metrics.csv         40 predictive runs (8 contexts × 5 seeds)
  per_class_metrics.csv, confusion_matrices.csv
  tuning.csv                     validation grid search, one selected row per context
  feature_ablation.csv           feature-count curve; validation-chosen count marked
  attributions.csv.gz            per-flow attributions (seed-42 models)
  fidelity.csv.gz                deletion results for every flow, k, control and seed
  explanation_flows.csv.gz       per-flow latency and faithfulness correlation
  stability.csv                  neighbor-stability pairs
  explanation_summary.csv        16 rows: one per context × explainer
  paired_comparisons.csv         TreeSHAP − LIME (deletion and faithfulness families)
  fidelity_control_tests.csv     top vs random and top vs global, per explainer
  run_manifest.json              config, data hashes, cleaning counts, selected hyperparameters, environment
  extended/                      audit tables + black_box_audit_summary.json
  figures/                       generated plots
  alert_reports/                 example per-alert report
output/pdf/xai_ids_ieee_paper.pdf    generated manuscript
```

---

## 21. Key references

- Sharafaldin, Lashkari, Ghorbani, "Toward generating a new intrusion detection dataset…", ICISSP 2018 (CIC-IDS2017).
- Moustafa, Slay, "UNSW-NB15: a comprehensive data set…", MilCIS 2015.
- Lundberg, Lee, "A unified approach to interpreting model predictions", NeurIPS 2017 (SHAP).
- Ribeiro, Singh, Guestrin, "Why should I trust you?", KDD 2016 (LIME).
- Adebayo et al., "Sanity checks for saliency maps", NeurIPS 2018.
- Bhatt, Weller, Moura, "Evaluating and aggregating feature-based model explanations", IJCAI 2020, doi:10.24963/ijcai.2020/417.
- Chen, Janizek, Lundberg, Lee, "True to the model or true to the data?", arXiv:2006.16234, 2020.
- Arreche, Guntur, Roberts, Abdallah, "E-XAI: Evaluating black-box explainable AI frameworks for network intrusion detection", IEEE Access 12:23954-23988, 2024, doi:10.1109/ACCESS.2024.3365140.
- Engelen, Rimmer, Joosen, "Troubleshooting an intrusion detection dataset: the CICIDS2017 case study", IEEE SPW 2021, pp. 7-12, doi:10.1109/SPW53761.2021.00009.
- Hooker et al., "A benchmark for interpretability methods in deep neural networks" (ROAR), NeurIPS 2019; Rong et al., "A consistent and efficient evaluation strategy for attribution methods" (ROAD), ICML 2022.
- Neupane et al., "Explainable intrusion detection systems (X-IDS): a survey…", IEEE Access 2022.

The full reference list is in the paper.
