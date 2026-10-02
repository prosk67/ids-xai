# Complete Project Handbook

## A Quantitative Framework for Evaluating Explainable AI in Network Intrusion Detection Systems

This handbook explains the complete CSE 498R project from the basic idea to the finished IEEE-style paper. It is written so that a reader who is new to machine learning, intrusion detection, and explainable AI can understand what was built, why each decision was made, how the experiments were conducted, and how the numerical results became the claims in the paper.

---

## 1. The project in one paragraph

The project investigates two connected questions. First, how accurately can Random Forest and XGBoost detect network attacks in the CIC-IDS2017 and UNSW-NB15 datasets? Second, when TreeSHAP or LIME explains a model prediction, how can we tell whether the explanation is genuinely related to the model's behavior rather than merely looking convincing? The implementation trains the classifiers for binary and multiclass intrusion detection, evaluates their predictions, explains matched test flows using both explainers, and quantitatively tests explanation fidelity, stability, latency, and feature-count sensitivity. The final paper reports the resulting evidence rather than judging explanations only from attractive feature-importance plots.

## 2. What problem are we solving?

### 2.1 Network intrusion detection

A network flow is a summary of communication between computers. It may contain features such as duration, packet counts, byte counts, packet-length statistics, ports, protocol, service, and TCP-state information.

An intrusion detection system (IDS) attempts to determine whether a flow is normal or malicious. In this project, two forms of classification are considered:

- **Binary classification:** predict `Normal` or `Attack`.
- **Multiclass classification:** predict the particular normal/attack family supplied by the dataset.

Binary detection answers whether an alert should be raised. Multiclass detection attempts to say what kind of attack produced the alert and is generally more difficult, particularly when rare attack families contain few examples.

### 2.2 Why prediction accuracy is not enough

A classifier can predict an attack without telling an analyst why it made that decision. In a security operations center, an analyst may want to know which flow properties influenced the alert, whether the explanation remains similar for comparable traffic, and whether an explanation can be produced quickly enough for operational use.

Explainable AI (XAI) methods generate feature attributions: numerical values describing how features contributed to a prediction. However, a visually reasonable explanation is not automatically correct. This creates the central research problem:

> How can explanation quality in a network IDS be measured quantitatively and reproducibly?

### 2.3 The proposed solution

The project treats the detector and explainer as one combined system. It measures five aspects:

1. **Predictive performance:** does the classifier label flows correctly?
2. **Fidelity:** do the features ranked highly by an explanation actually affect model confidence?
3. **Stability:** do similar flows receive compatible explanations?
4. **Latency:** how long does one explanation take?
5. **Feature-count sensitivity:** how much predictive quality is retained with fewer features?

---

## 3. Research questions

The paper organizes the project around five research questions.

### RQ1 — Predictive performance

How do Random Forest and XGBoost perform on CIC-IDS2017 and UNSW-NB15 for binary and multiclass detection?

### RQ2 — Explanation fidelity

Do the features selected by TreeSHAP and LIME reduce predicted confidence more than random or bottom-ranked features when removed?

### RQ3 — Explanation stability

Do nearby, same-prediction flows receive similar explanations, and is LIME consistent when repeated with different random seeds?

### RQ4 — Operational latency

Are explanations fast enough for offline analysis, analyst-on-demand use, or near-real-time deployment?

### RQ5 — Robustness of the conclusion

Do the conclusions remain consistent across datasets, tasks, model families, predictive seeds, and feature counts?

These questions are important because an explainer can be accurate but slow, fast but unstable, or visually plausible but unfaithful. No single metric answers the entire problem.

---

## 4. End-to-end workflow

The complete experimental workflow is:

```text
Raw datasets
    |
    v
Label audit, cleaning, deduplication, and fixed train/test splits
    |
    v
Training-only preprocessing
    |
    v
Random Forest and XGBoost training for binary and multiclass tasks
    |
    v
Predictive evaluation over seeds 42, 52, 62, 72, and 82
    |
    v
Freeze the seed-42 models
    |
    v
Explain the same stratified test flows with TreeSHAP and LIME
    |
    v
Measure fidelity, controls, stability, latency, and feature ablation
    |
    v
Bootstrap intervals, paired Wilcoxon tests, and Holm correction
    |
    v
CSV results, figures, reproducibility manifest, and IEEE paper
```

The implementation is primarily in `src/xai_ids/benchmark.py`. The paper is generated by `src/xai_ids/paper.py`, and its publication figures are generated by `src/xai_ids/paper_figures.py`.

---

## 5. Datasets

### 5.1 CIC-IDS2017

CIC-IDS2017 represents benign traffic and multiple modern attack types. The local public preprocessing contains 2,830,743 flows and 28 selected flow features.

To keep computation manageable while retaining minority classes, the implementation samples at most 25,000 rows from each class. It then removes 15,103 sampled duplicate rows and creates a stratified 80/20 split:

- Training rows: 99,610
- Test rows: 24,903
- Features: 28
- Binary classes: 2
- Multiclass labels: 15

The available CIC preprocessing does not contain capture-day or timestamp metadata. Therefore, the paper correctly describes this as a deduplicated stratified benchmark, not as evidence of temporal generalization.

### 5.2 UNSW-NB15

UNSW-NB15 contains modern normal and attack traffic with predefined training/testing files. A publication audit found substantial exact duplication and feature-vector overlap, so the pipeline applies explicit leakage controls:

- Original training/test rows: 175,341 / 82,332
- Exact duplicates removed: 78,519 / 32,361
- Cross-partition duplicate test feature vectors removed: 2,255
- Clean training/test rows: 96,822 / 47,716
- Features: 34
- Binary classes: 2
- Multiclass labels: 10

The remaining partition preserves the predefined train/test direction while removing direct duplicate leakage. The dataset owner documents it as a configured partition, not a temporal holdout, so temporal generalization is not claimed.

### 5.3 Important filename detail

The local UNSW mirror filenames are reversed relative to the official row counts. `config.yaml` intentionally maps:

```yaml
train: data/raw/unsw_test.parquet
test: data/raw/unsw_train.parquet
```

This is not an accidental train/test swap. The original row counts were checked against the dataset owner's predefined files before the documented cleaning audit.

### 5.4 Why use two datasets?

Using one dataset could produce a conclusion that depends on its collection method, class balance, or artifacts. CIC and UNSW differ in features, labels, time structure, difficulty, and data distribution. Agreement across both gives stronger evidence; disagreement reveals where conclusions are context-dependent.

---

## 6. Preventing data leakage

Data leakage occurs when information from the test set influences model training or preprocessing. Leakage can make results appear unrealistically strong.

The `TabularPreprocessor` class prevents this by learning all preprocessing parameters from training data only:

- Infinite numeric values are changed to missing values.
- Missing numeric values are filled with medians calculated from training data.
- Categorical dictionaries are created from training categories only.
- A category not observed during training receives an explicit unseen-category code.
- Label columns and attack-category metadata are excluded from the feature space.

The fitted preprocessor is then applied unchanged to the held-out test set. This principle is essential: test data may be transformed, but it must not teach the pipeline how to transform itself.

For CIC, sampling is performed before deduplication to control memory use, and duplicates are removed before splitting. For UNSW, exact duplicates are removed within each file and every test feature vector that occurs in training is excluded. The clean partition contains no exact-row or exact-feature overlap.

---

## 7. Models

### 7.1 Random Forest

Random Forest trains many decision trees on perturbed versions of the data and combines their predictions. It is often accurate, robust, and naturally compatible with tree-based SHAP explanations.

Fixed settings:

- 120 trees
- Maximum depth 18
- Minimum leaf size 2
- Balanced bootstrap class weights

### 7.2 XGBoost

XGBoost builds decision trees sequentially. Each new tree attempts to correct errors made by the existing ensemble.

Fixed settings:

- 160 trees
- Maximum depth 7
- Learning rate 0.08
- Row subsampling 0.85
- Column subsampling 0.85
- Histogram-based training
- Balanced sample weights

### 7.3 Why these two models?

Both are strong tabular classifiers and are common in IDS research. Comparing a bagging model with a boosting model helps reveal whether an explanation result belongs to the explainer in general or to a particular model-explainer combination.

### 7.4 Why no hyperparameter search?

The project uses prespecified fixed settings instead of optimizing on the test set. This keeps the experiment reproducible and avoids overstating performance through extensive tuning. The goal is a controlled XAI comparison, not a leaderboard search.

### 7.5 Predictive seeds

Each dataset-task-model combination is trained with seeds 42, 52, 62, 72, and 82. This produces:

```text
2 datasets x 2 tasks x 2 models x 5 seeds = 40 predictive runs
```

The five seeds (42, 52, 62, 72, and 82) measure variation due to model randomness. They do not measure uncertainty caused by alternative dataset captures or preprocessing choices.

---

## 8. Predictive evaluation metrics

### 8.1 Accuracy

Accuracy is the proportion of all predictions that are correct:

```text
accuracy = correct predictions / all predictions
```

Accuracy can be misleading when one class is much more common than others.

### 8.2 Balanced accuracy

Balanced accuracy gives each class equal weight by averaging class recall. It is more informative for imbalanced security data.

### 8.3 Precision, recall, and F1

- **Precision:** among flows predicted as a class, how many were correct?
- **Recall:** among flows truly belonging to a class, how many were detected?
- **F1:** harmonic mean of precision and recall.

### 8.4 Macro-F1 versus weighted-F1

Macro-F1 calculates F1 independently for every class and averages them equally. A rare attack therefore matters as much as a frequent class.

Weighted-F1 weights each class by its number of test examples. It can remain high even when rare attacks are missed.

This distinction explains the cleaned UNSW multiclass result: weighted-F1 is around 0.71–0.72, while macro-F1 is only about 0.51. The system handles frequent classes much better than several rare attack families.

### 8.5 Binary security metrics

For binary detection, the project also reports:

- False-positive rate (FPR): fraction of normal flows incorrectly flagged.
- False-negative rate (FNR): fraction of attacks missed.
- Precision-recall area under the curve (PR-AUC).

FPR describes analyst alert burden, while FNR describes missed attacks.

---

## 9. Explainable AI methods

### 9.1 What is a feature attribution?

For one prediction, an explainer produces a vector with one value per input feature. A positive or negative sign indicates direction, and magnitude indicates importance according to that explainer. The project converts both methods to signed vectors in the original feature order so they can be evaluated consistently.

### 9.2 TreeSHAP

TreeSHAP uses the structure of the fitted tree ensemble to calculate SHAP values based on Shapley-value principles. It is optimized for tree models and generally deterministic for a fixed model and input.

Conceptually, it asks how the predicted output changes as features participate in coalitions. Its strengths here are direct use of tree structure, consistent attribution, and often strong fidelity.

### 9.3 LIME

LIME explains one prediction by generating perturbed samples near that input, querying the classifier, and fitting a sparse local surrogate model.

The project uses 350 perturbation samples per explanation. LIME is flexible and model-agnostic, but it is sensitive to sampling, feature discretization, and whether the synthetic neighborhood accurately represents the real data manifold.

### 9.4 Fair comparison

TreeSHAP and LIME explain the same 40 label-stratified test flows for every dataset-task-model configuration. The selector also explicitly includes available binary true-positive, false-positive, true-negative, and false-negative cases. This matched design is essential. If different instances were used, the explainer comparison could be confounded by instance difficulty.

There are eight configurations:

```text
2 datasets x 2 tasks x 2 models = 8 configurations
```

Both explainers are evaluated on the seed-42 frozen models. Predictive variation was already measured separately with five seeds. LIME self-stability uses five explainer seeds on six repeated flows per configuration.

---

## 10. How fidelity is measured

### 10.1 Basic idea

Suppose a model predicts an attack with probability 0.90. An explainer says that the top features caused this prediction. If replacing those features with neutral baseline values reduces confidence to 0.50, the confidence drop is 0.40. A large drop suggests that the ranking identified influential features.

For input `x`, predicted class `c`, ranking `r`, baseline `b`, and the top `k` features, the deletion score is:

```text
D_k(x, r) = p(c | x) - p(c | mask(x, r[1:k], b))
```

The baseline uses training medians for numeric features and training modes for categorical features. Test information is not used to create the baseline.

Scores are computed at `k = 1, 3, 5, 10` and averaged to obtain the primary fidelity score.

### 10.2 Why use several k values?

Testing only the top feature may be noisy. Removing too many features can produce unrealistic inputs. Averaging several small deletion budgets evaluates the ranking over a useful range without relying on one arbitrary threshold.

### 10.3 Control rankings

A high deletion score is not sufficient by itself. Some predictions may be sensitive to deleting almost any feature. Therefore, the same deletion operation is applied to:

- **Top-ranked features:** features the explainer considers most important.
- **Random features:** a fixed random ranking.
- **Bottom-ranked features:** features the explainer considers least important.

The key control quantity is:

```text
fidelity lift = top-ranked deletion score - random deletion score
```

A positive, statistically supported lift means the explainer's ranking is more informative than choosing features randomly.

### 10.4 Flip rate

The flip rate records how often feature masking changes the predicted class. It is a secondary, easily interpreted measure of decision sensitivity.

### 10.5 Important limitation

Median or mode masking can produce feature combinations that do not occur in real traffic because flow variables are correlated. The deletion test measures fitted-model faithfulness under the stated intervention; it does not prove causality.

---

## 11. How stability is measured

### 11.1 Neighborhood stability

For each explained test flow, the system finds nearby standardized test flows with the same predicted class. Explanations are compared using:

- **Signed cosine similarity:** agreement in attribution direction and relative magnitude.
- **Top-5 Jaccard overlap:** agreement between the sets of five highest-ranked features.

Cosine similarity of 1 means the vectors point in the same direction. Jaccard overlap of 1 means the top-five feature sets are identical.

The same-prediction restriction avoids comparing explanations of flows assigned to completely different decisions, although it does not eliminate all boundary effects.

### 11.2 LIME self-stability

Because LIME uses random perturbations, the same instance is explained repeatedly with different seeds. Six repeated flows are used to estimate how consistently LIME explains an identical input.

TreeSHAP is deterministic for the tested fixed tree model, so repeated-seed self-stability is principally relevant to LIME.

### 11.3 Stability is not fidelity

An explanation can be stable and consistently wrong. It can also be faithful but change appropriately near a decision boundary. Therefore, stability and fidelity must be reported separately.

---

## 12. Latency measurement

The first explanation is treated as a warm-up. Subsequent per-instance wall-clock times are recorded. The project reports:

- Median latency
- P90, P95, and P99 latency
- Throughput in explanations per second

Tail latency matters because an apparently fast average can hide occasional slow explanations. The measured values apply to the recorded Windows 11 CPU host; they are not universal hardware benchmarks.

---

## 13. Statistical analysis

### 13.1 Why matched analysis?

TreeSHAP and LIME explain identical flows. The correct comparison is therefore within-flow:

```text
difference_i = TreeSHAP fidelity_i - LIME fidelity_i
```

This removes much of the variation caused by some flows being easier to explain than others.

### 13.2 Bootstrap confidence intervals

The 40 paired differences are sampled with replacement 2,000 times. The mean is calculated for every resample. The 2.5th and 97.5th percentiles form a percentile 95% confidence interval.

If the complete interval is above zero, the observed advantage is directionally consistent with TreeSHAP having greater mean deletion fidelity.

### 13.3 Wilcoxon signed-rank test

A one-sided Wilcoxon signed-rank test evaluates the prespecified directional hypothesis that TreeSHAP fidelity is greater than LIME fidelity. This nonparametric paired test does not require the differences to be normally distributed.

The same type of one-sided test evaluates whether top-ranked fidelity is greater than random-ranking fidelity.

### 13.4 Holm correction

Running many tests increases the chance of false positives. Holm's step-down procedure adjusts p-values while controlling family-wise error:

- 8 TreeSHAP-versus-LIME tests are corrected as one family.
- 16 top-versus-random control tests are corrected as another family.

The paper treats an adjusted p-value below 0.05 as statistically significant.

### 13.5 Effect size

The paired standardized effect is calculated as:

```text
d_z = mean(paired differences) / standard deviation(paired differences)
```

The raw probability drop says how large the confidence change is in model-output units. `d_z` says how large the paired difference is relative to its across-flow variation. They answer different questions and should not be confused.

---

## 14. Feature-count ablation

Global feature rankings are obtained from the frozen full model. New models are then retrained with the top 5, 10, and 20 features, as well as the complete feature set.

This is a retraining experiment, not merely a deletion experiment. It asks whether a smaller input set preserves predictive quality and reduces explanation cost.

Examples:

- CIC Random Forest binary reaches about 0.997 macro-F1 with 10 features, essentially matching the 28-feature model.
- UNSW XGBoost binary reaches about 0.816 macro-F1 with 20 features versus about 0.815 with 34.
- UNSW Random Forest multiclass reaches about 0.537 macro-F1 with 10 features versus about 0.522 with 34.
- CIC XGBoost multiclass is different: all 28 features are needed for its best result.

Therefore, feature reduction is promising but must be chosen per dataset, task, and model. A universal feature-count rule would not be supported.

---

## 15. Main measured results

### 15.1 Predictive performance

Mean macro-F1 over the five seeds is approximately:

| Dataset | Task | Random Forest | XGBoost | Interpretation |
|---|---:|---:|---:|---|
| CIC-IDS2017 | Binary | 0.996 | 0.997 | Both are extremely strong on the sampled split. |
| CIC-IDS2017 | Multiclass | 0.925 | 0.932 | XGBoost is slightly stronger. |
| UNSW-NB15 | Binary | 0.801 | 0.816 | XGBoost is modestly stronger after leakage removal. |
| UNSW-NB15 | Multiclass | 0.514 | 0.508 | Both struggle with minority attack families. |

The CIC result should not be interpreted as proof that real deployment would achieve 99% performance. The split is stratified, and the source lacks time metadata. UNSW is a harder predefined partition, but it is not documented as a temporal or live-network holdout.

### 15.2 Minority-class behavior

For UNSW multiclass at seed 42, recall includes:

| Class | Random Forest | XGBoost |
|---|---:|---:|
| Analysis | 0.41 | 0.40 |
| DoS | 0.29 | 0.43 |
| Backdoor | 0.53 | 0.55 |

This is why macro-F1 and per-class recall are necessary. A weighted average alone would hide serious failures on Analysis, DoS, and Backdoor.

### 15.3 TreeSHAP versus LIME fidelity

TreeSHAP has higher paired top-k deletion fidelity in all eight dataset-task-model comparisons.

- Mean paired improvements range from approximately 0.062 to 0.223 probability-drop units.
- All eight bootstrap confidence intervals are above zero.
- All eight one-sided Wilcoxon tests remain significant after Holm correction.
- The largest adjusted p-value is approximately 0.0032.
- Paired standardized effects range from approximately 0.382 to 0.974.

The strongest raw improvement occurs for CIC multiclass XGBoost, where the mean TreeSHAP-minus-LIME fidelity difference is about 0.223.

### 15.4 Random-ranking controls

TreeSHAP's top-ranked features beat the random-ranking control after Holm correction in all eight configurations.

LIME passes the corrected control in seven configurations. It fails only in:

- CIC binary Random Forest

The failed CIC Random Forest lift is small but positive (approximately 0.014) with Holm-adjusted p=0.338. This does not prove that every failed explanation is useless. It means the tested ranking was not demonstrably better than the fixed random control under the selected perturbation and deletion protocol.

### 15.5 Stability

TreeSHAP neighborhood cosine similarity ranges from approximately 0.543 to 0.947. LIME ranges from approximately 0.472 to 0.801. These neighbors are selected only within each 40-flow evaluation subset, so the metric is contextual rather than a population-wide estimate.

LIME repeated-seed self-stability ranges from about 0.618 to 0.910. A separate sampling-budget audit shows that pooled stability improves from 0.610 with 100 perturbations to 0.855 with 700, confirming a measurable quality-latency trade-off.

### 15.6 Latency

Latency is determined by the model-explainer pair, not by the explainer name alone.

- XGBoost TreeSHAP median latency ranges from about 1.6 to 23.5 ms.
- LIME medians for the same XGBoost configurations range from about 15.4 to 24.2 ms.
- Random Forest TreeSHAP is faster than LIME on CIC.
- On UNSW Random Forest, TreeSHAP becomes slower; multiclass median latency is about 324 ms with a P95 around 357 ms.

Thus, “SHAP is always slower than LIME” and “SHAP is always faster than LIME” are both incorrect. Deployment decisions require the actual model-explainer configuration and its tail latency.

---

## 16. Answers to the research questions

### Answer to RQ1

Model choice matters less than evaluation protocol for the easiest tasks. Both models are excellent on sampled CIC and useful on UNSW binary detection. Neither solves UNSW multiclass minority-class detection.

### Answer to RQ2

TreeSHAP is the stronger fidelity method in this experiment. It has higher paired deletion fidelity in every configuration and is the only explainer whose top ranking beats the corrected random control in all eight configurations.

### Answer to RQ3

TreeSHAP is generally more stable for nearby same-prediction flows. LIME exhibits repeated-seed sensitivity, so reporting only one LIME run can hide important variation.

### Answer to RQ4

No explainer has a universal speed advantage. XGBoost plus TreeSHAP is the strongest near-real-time candidate. UNSW Random Forest plus TreeSHAP is more appropriate for lower-volume or offline explanation unless a roughly 0.3-second tail is acceptable.

### Answer to RQ5

The TreeSHAP fidelity advantage is robust across every tested dataset-task-model context. Feature-reduction and latency findings are context-dependent and should not be turned into universal rules.

---

## 17. Operational interpretation

The benchmark separates three questions that are often confused:

1. **Predictive quality:** does the IDS label traffic correctly?
2. **Model fidelity:** does an explanation reflect the fitted model?
3. **Human usefulness:** does the explanation help an analyst make a better decision?

This project measures the first two and the computational cost of explanation. It does not perform a human-subject study, so it does not claim that a faithful explanation is automatically understandable, causal, or actionable.

A practical analyst-facing explanation record should include:

- Predicted class and model probability, clearly marked as uncalibrated unless calibration is added
- Important signed features with observed values and units
- Masking/reference baseline used by the evaluator
- Similar reference flows
- A warning that attribution is not causality
- Model and explainer versions

Production governance should monitor feature drift, repeated-seed stability where relevant, false-positive explanations, class-specific fidelity, and changes in recurring top features.

### 17.1 What was added to address decision opacity directly

The benchmark originally proved that TreeSHAP is the stronger tested explainer at the group level. That evidence alone does not tell an analyst whether one particular alert has a strong explanation. The project now includes `explain_alert.py`, which turns the research protocol into a per-alert transparency layer.

For a selected flow, it produces Markdown for an analyst and JSON for another system. Each report contains:

- The predicted class and probability for every class
- The important signed TreeSHAP features
- The actual observed value of every reported feature
- The direction in which each feature influences the predicted class
- Top-ranked and bottom-ranked deletion scores
- A random-ranking distribution generated specifically for that alert
- Top-minus-random fidelity lift and an empirical one-sided p-value
- The corrected benchmark-level confidence interval and p-value
- Previously measured neighborhood stability and latency context
- A scope warning that model evidence is not real-world causality

The report assigns one of three evidence tiers:

- `SUPPORTED_FOR_THIS_ALERT`: TreeSHAP passed the corrected benchmark control, the alert's top-minus-random lift is positive, and its empirical local p-value is at most 0.05.
- `REVIEW_REQUIRED`: TreeSHAP passed the benchmark control, but this individual alert did not clear the strict local randomization test.
- `UNSUPPORTED`: the explainer did not pass the relevant benchmark-level validation.

This rejection rule is important. A transparency system should be able to say “the available explanation is not sufficiently supported” instead of generating a confident-looking story for every prediction.

Example command:

```powershell
.\.venv\Scripts\python.exe explain_alert.py `
  --model outputs/models/UNSW_NB15_binary_xgboost.joblib `
  --input data/raw/unsw_train.parquet `
  --row 0 `
  --output outputs/alert_reports/unsw_binary_xgb_example
```

The regenerated example predicts `Attack` with model probability 0.886009. Its top-feature deletion score is 0.156774, compared with a mean of 0.082928 over 200 random rankings. The top-minus-random lift is 0.073846 and its empirical local p-value is 0.189055, so the report correctly assigns `REVIEW_REQUIRED` rather than forcing a supported explanation. These values belong to that example flow and should not be generalized to all alerts.

### 17.2 Did this solve the black-box problem?

The precise answer is:

- **Yes, operational decision opacity is addressed for the tested saved IDS models.** An analyst can inspect what drove an alert, see the underlying values, test whether deleting those features changes the decision, compare the ranking against randomized alternatives, and reject weak explanations.
- **No, the universal theoretical black-box problem is not eliminated.** A post-hoc explanation cannot prove causality, guarantee human understanding, prevent dataset shift, or reveal every internal interaction of every possible model.

The scientifically defensible project claim is therefore:

> The framework converts opaque tree-based IDS predictions into auditable, quantitatively validated, and rejectable per-alert evidence within the tested data and model scope.

That is stronger and more useful than claiming that a feature plot “solves” interpretability, while remaining honest about the boundary between fitted-model evidence and causal truth.

### 17.3 Additional audits derived from the literature review

The supplied literature review proposed several checks beyond the minimum experiment. `run_extended_audit.py` now implements the highest-value checks that are possible without a human-subject study:

- **Simple baselines:** majority and depth-5 decision-tree classifiers. The shallow-tree macro-F1 range is 0.249-0.955, and every full Random Forest/XGBoost context exceeds its corresponding simple baseline.
- **Balancing sensitivity:** class-weighted versus unweighted training with identical seed 42. Weighting improves macro-F1 in five of eight contexts.
- **Probability calibration:** binary reliability bins, Brier scores, and expected calibration error (ECE). ECE is 0.003-0.005 on CIC and 0.111-0.113 on cleaned UNSW, warning that raw probabilities are less trustworthy under dataset shift.
- **Masking-reference sensitivity:** median/mode, zero, marginal training row, nearest same-prediction flow, and nearest opposite-prediction flow. Pooled TreeSHAP deletion is 0.392 for median/mode, 0.418 for zero, 0.386 for a marginal row, and 0.310 for a nearest opposite prediction. It is -0.004 for a nearest same prediction.
- **LIME sampling sensitivity:** 100, 350, and 700 perturbations with five repeated seeds. Pooled fidelity is 0.227, 0.257, and 0.266; self-stability increases from 0.610 to 0.798 and 0.855 while median explainer-call latency rises from 31.0 to about 37.8 ms.
- **Class/error analysis:** fidelity summaries are saved by actual class and by TP, FP, TN, FN, correct, or incorrect status.
- **Stability distributions:** Q10, median, and Q90 are reported for cosine similarity and top-five Jaccard overlap instead of only global means.

The masking audit is especially important for the black-box claim. A nearest flow with the same prediction is expected to preserve confidence, while a nearest opposite prediction provides a contrastive reference. Therefore, deletion fidelity is not a property of an explanation alone; it is a property of the explanation, model, input, masking rule, and reference distribution together.

SMOTE was not added. Interpolating ordinal-encoded protocol and service categories can create invalid synthetic traffic unless a categorical-aware, fold-local method is carefully validated. The project instead reports the safer class-weight/no-weight comparison and documents SMOTE as a future controlled experiment.

---

## 18. Threats to validity

### 18.1 Data and external validity

- CIC's local preprocessing lacks timestamps and may contain capture-specific artifacts.
- UNSW's predefined split represents one dataset construction process, not a documented temporal holdout or live network.
- Sampling CIC before deduplication may alter class distributions, although duplicates are removed before splitting.
- Results should not be generalized automatically to encrypted traffic, new attacks, different networks, or concept drift.

### 18.2 Construct validity

- The multi-reference audit shows that masking conclusions depend strongly on the intervention; no tested baseline proves causality.
- Nearest neighbors in standardized feature space are an approximation to semantic similarity.
- Fidelity to a model is not causality or human usefulness.
- SHAP and LIME attribution magnitudes are on different conceptual scales, so the project compares rankings and behavioral tests rather than raw values directly.

### 18.3 Statistical validity

- Forty matched explanation flows are used per configuration; this is stronger than the initial run but still describes a selected evaluation sample rather than repeated datasets.
- Bootstrap intervals describe these selected flows, not repeated datasets.
- Rare CIC classes have very small test support, so their individual recall values are uncertain.
- Five model seeds capture algorithmic randomness but not uncertainty from alternate data captures or pipelines.

### 18.4 Scope limitations

- Only two tree classifiers and two post-hoc explainers are evaluated.
- LIME budgets from 100 to 700 samples are audited, but other kernels and discretization settings might change its quality-latency trade-off.
- Counterfactual, rule-based, deep-learning, adversarial, drift, and analyst-study evaluations remain future work.

Good research does not hide these limitations. It states precisely what the evidence supports and where further work is required.

---

## 19. Repository structure

```text
498R/
|-- config.yaml                         Experimental configuration
|-- requirements.txt                    Python dependencies
|-- run_benchmark.py                    Small entry point for all experiments
|-- build_paper.py                      Small entry point for paper generation
|-- verify_outputs.py                   Artifact and paper integrity checks
|-- README.md                           Short reproduction instructions
|-- PROJECT_HANDBOOK.md                 This complete explanation
|-- data/raw/                           Local raw Parquet datasets (not committed)
|-- src/xai_ids/
|   |-- benchmark.py                    Data, models, XAI, statistics, and outputs
|   |-- paper.py                        IEEE-style PDF layout and manuscript
|   `-- paper_figures.py                Publication-specific figures
|-- outputs/
|   |-- predictive_metrics.csv          40 model-seed predictive runs
|   |-- per_class_metrics.csv           Per-class precision/recall/F1/support
|   |-- confusion_matrices.csv          Machine-readable confusion matrices
|   |-- attributions.csv                Per-instance feature attributions
|   |-- fidelity.csv                    Per-instance deletion/control results
|   |-- stability.csv                   Neighbor and self-stability observations
|   |-- explanation_summary.csv         Aggregated XAI quality and latency
|   |-- paired_comparisons.csv          TreeSHAP-minus-LIME paired inference
|   |-- fidelity_control_tests.csv      Top-ranking versus random controls
|   |-- feature_ablation.csv            Reduced/full feature-set experiments
|   |-- run_manifest.json               Hashes, versions, configuration, hardware
|   `-- figures/                         Generated experimental/paper plots
`-- output/pdf/
    `-- xai_ids_ieee_paper.pdf          Final five-page paper
```

---

## 20. How to reproduce the project

### 20.1 Requirements

- Windows, Linux, or macOS with Python 3.12 recommended
- Sufficient memory for the datasets and tree models
- The four Parquet files referenced in `config.yaml`

### 20.2 Set up the environment on Windows PowerShell

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, the virtual environment's Python executable can be used directly without activation.

### 20.3 Place the data

Create `data/raw/` and add the files named in `config.yaml`:

```text
data/raw/cic_train_0000.parquet
data/raw/cic_train_0001.parquet
data/raw/unsw_test.parquet
data/raw/unsw_train.parquet
```

Raw data is intentionally not stored in the repository because of size and redistribution concerns.

### 20.4 Run the benchmark

```powershell
.\.venv\Scripts\python.exe run_benchmark.py --config config.yaml
```

The run loads and audits the datasets, trains all predictive models, evaluates XAI, performs statistics and ablations, writes CSV outputs, generates figures, and creates `run_manifest.json`.

This is the expensive command. Runtime depends on CPU, memory, package versions, and whether the outputs already exist.

### 20.5 Build the paper

```powershell
.\.venv\Scripts\python.exe build_paper.py
```

The paper builder reads the measured CSV files. It does not invent experimental numbers. It generates the paper figures and writes:

```text
output/pdf/xai_ids_ieee_paper.pdf
```

### 20.6 Verify the outputs

```powershell
.\.venv\Scripts\python.exe verify_outputs.py
```

The verifier checks that required files exist, expected table sizes are correct, metric values are valid, paired confidence intervals are above zero, source dataset hashes are present, the paper contains five pages, and required paper sections are extractable.

### 20.7 Run unit tests

Using the standard library:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests check bootstrap behavior, stability metric bounds, and correct handling of an unseen test category by the training-only preprocessor.

---

## 21. How the code works

### 21.1 Entry point

`run_benchmark.py` calls `src.xai_ids.benchmark.main()`. The main function reads the `--config` path and calls `run()`.

### 21.2 Configuration

`config.yaml` stores dataset paths, labels, sampling rules, seeds, model hyperparameters, explanation sample counts, deletion budgets, and feature-ablation counts. Keeping these choices outside the source code makes the experiment auditable and easier to reproduce.

### 21.3 Data preparation functions

- `load_cic()` loads CIC shards, cleans labels, samples per class, and deduplicates.
- `load_unsw()` loads the official UNSW train and test tables.
- `split_cic()` creates the stratified CIC split.
- `prepare_dataset()` creates binary/multiclass targets and fitted preprocessing outputs.

### 21.4 Model and prediction functions

- `make_model()` constructs Random Forest or XGBoost with fixed settings.
- `predictive_metrics()` calculates aggregate and per-class measures.
- `save_confusion_plot()` visualizes class-prediction errors.

### 21.5 Explanation functions

- `make_lime_explainer()` initializes LIME from training data.
- `explain_lime()` maps LIME's sparse output back into the complete original feature vector.
- `extract_shap_vector()` extracts the predicted class's SHAP vector despite version-dependent SHAP output shapes.
- `evaluate_explainer()` selects matched flows, generates attributions, records latency, and calls the fidelity and stability routines.

### 21.6 Evaluation and inference functions

- `fidelity_rows()` performs top, random, and bottom deletion tests at all k values.
- `nearest_neighbor_stability()` calculates signed cosine and top-k Jaccard agreement.
- `bootstrap_mean_ci()` calculates percentile bootstrap intervals.
- `paired_bootstrap_comparisons()` performs matched TreeSHAP-minus-LIME analysis.
- `fidelity_control_tests()` tests top-ranked features against random controls.
- `feature_ablation()` ranks features, retrains reduced models, and measures performance/latency.

### 21.7 Output generation

`make_summary_plots()` creates experimental charts. The run concludes by writing the manifest, including data hashes, row counts, packages, configuration, model pins, and hardware information.

---

## 22. How the paper was written

### 22.1 Start with the evidence, not the prose

The benchmark was implemented and run before the final claims were written. Each important paper number comes from a machine-readable output:

- Predictive claims come from `predictive_metrics.csv` and `per_class_metrics.csv`.
- Fidelity and latency claims come from `explanation_summary.csv`.
- Paired superiority claims come from `paired_comparisons.csv`.
- Random-control claims come from `fidelity_control_tests.csv`.
- Feature-reduction claims come from `feature_ablation.csv`.

This evidence-first workflow reduces copy/paste mistakes and prevents the paper from making claims that the experiment did not support.

### 22.2 Paper structure

The five-page manuscript follows a compact IEEE-style research structure:

1. **Title and abstract:** problem, methods, central numerical findings, and conclusion.
2. **Introduction:** motivation, research gap, and contributions.
3. **Related work:** datasets, IDS XAI studies, and objective explanation evaluation.
4. **Methodology:** research questions, data controls, models, metrics, and statistical rules.
5. **Experimental results:** prediction, XAI fidelity/stability, latency, ablation, and error analysis.
6. **Discussion:** direct answers to the research questions and interpretation of why explainers differ.
7. **Deployment recommendations:** what the measurements imply for operational use.
8. **Threats to validity:** boundaries of the evidence.
9. **Reproducibility:** software, hardware, hashes, files, and seeds.
10. **Conclusion:** concise answer to the research problem and future work.
11. **References:** primary dataset, model, explainer, IDS, and evaluation sources.

### 22.3 Connecting claims to research questions

Every result is included because it answers a research question:

- Predictive tables answer RQ1.
- Deletion tables, confidence intervals, and random controls answer RQ2.
- Cosine, Jaccard, and repeated-seed measures answer RQ3.
- Configuration-level latency plots answer RQ4.
- Multi-dataset comparisons and feature ablations answer RQ5.

This prevents the paper from becoming a collection of unrelated charts.

### 22.4 Why the paper uses tables and figures

Tables report exact values. Figures reveal patterns quickly:

- Workflow diagram: shows the complete experimental logic.
- Predictive bar chart: compares tasks, datasets, and models.
- Paired fidelity forest plot: displays effect direction and uncertainty.
- Latency configuration plot: prevents an oversimplified explainer-level conclusion.
- Feature-ablation curve: shows how performance changes with feature count.
- UNSW recall heat map: makes minority-class failures visible.

### 22.5 Automated paper generation

`build_paper.py` calls `src.xai_ids.paper.main()`. The paper builder:

1. Regenerates publication figures from the current results.
2. Reads result CSV files and the manifest.
3. Constructs paragraph styles, equations, tables, captions, headers, and footers.
4. Lays out the manuscript in two columns.
5. Writes the final PDF.

This means updated benchmark outputs can flow into the paper without manually rebuilding every table.

### 22.6 Final quality assurance

After generation, the PDF was checked in two ways:

- Programmatically: page count, headings, output files, hashes, table lengths, and interval direction.
- Visually: every rendered page was inspected for clipping, overlap, unreadable text, broken tables, and reference overflow.

The final reference style was tightened so all references fit on page five instead of leaving two lines on an otherwise blank sixth page.

---

## 23. How to read the result files

### `predictive_metrics.csv`

One row represents one dataset-task-model-seed run. Filter by dataset and task, then compare macro-F1, balanced accuracy, FPR/FNR, training time, and prediction time.

### `per_class_metrics.csv`

Use this file when aggregate metrics look acceptable but you need to determine which attack families fail. Pay close attention to recall and support.

### `explanation_summary.csv`

One row represents one dataset-task-model-explainer configuration. Compare fidelity, confidence intervals, random/bottom controls, flip rate, stability, latency quantiles, and throughput.

### `paired_comparisons.csv`

Each row compares TreeSHAP and LIME on matched flows. The most important columns are:

- `mean_difference_shap_minus_lime`
- `ci_low` and `ci_high`
- `paired_effect_dz`
- `holm_adjusted_p`

A positive interval and adjusted p below 0.05 support the TreeSHAP superiority claim for that context.

### `fidelity_control_tests.csv`

Each row tests one explainer's top ranking against a random ranking. A positive confidence interval and Holm-adjusted p below 0.05 support the claim that the ranking is better than random.

### `feature_ablation.csv`

Compare macro-F1 and SHAP latency across feature counts. The `selected_features` column records the exact global ranking used to retrain each reduced model.

### `run_manifest.json`

Use this file to establish provenance. It records the source file paths, SHA-256 hashes, row counts, package versions, configuration, and hardware. If a dataset file changes, its hash will change even if its filename does not.

---

## 24. Common misunderstandings

### “The CIC model has 99% accuracy, so the IDS problem is solved.”

No. The sampled CIC split is easier than a future-time deployment and may contain capture-specific patterns. The harder UNSW multiclass result shows that realistic distribution shift and rare classes remain difficult.

### “TreeSHAP wins, so SHAP is always the best explainer.”

No. TreeSHAP is supported for the two tree models, two datasets, and stated evaluation protocol. Other model classes, SHAP variants, baselines, and operational goals may produce different conclusions.

### “A significant p-value means the effect is large.”

No. Statistical significance addresses evidence against a null hypothesis. Raw mean differences and effect sizes describe magnitude.

### “A stable explanation must be correct.”

No. Stability is consistency, while fidelity is behavioral alignment with the model.

### “Fidelity means causality.”

No. The deletion experiment intervenes on model inputs and observes fitted-model output. It does not establish causal effects in the real network.

### “The weighted-F1 is good, so every attack is detected.”

No. Frequent classes dominate weighted-F1. Macro-F1 and per-class recall expose minority-class failure.

### “The fastest explainer should be deployed.”

No. Speed cannot compensate for a failed fidelity control. Deployment must jointly consider predictive performance, fidelity, stability, and tail latency.

---

## 25. Suggested presentation or defense outline

If presenting this project, a clear 10-part story is:

1. IDS alerts need explanations, but attractive plots are not evidence of correctness.
2. The research goal is to evaluate the classifier and explainer as one system.
3. Two datasets, two tasks, two models, and two explainers provide controlled comparisons.
4. Training-only preprocessing and fixed splits prevent leakage.
5. Predictive metrics show strong CIC and binary UNSW results but weak UNSW multiclass minority detection.
6. Deletion fidelity tests whether ranked features really influence model confidence.
7. Random and bottom controls prevent misleading interpretation of raw deletion scores.
8. Paired inference shows TreeSHAP has higher fidelity in all eight contexts.
9. Stability and latency show that deployment decisions are model-explainer specific.
10. The conclusion is measured: TreeSHAP is stronger here, but human usefulness and broader deployment require additional study.

### Short defense answer: What is the main contribution?

The main contribution is not another accurate IDS classifier. It is a reproducible quantitative framework that evaluates whether local explanations are faithful, stable, and operationally feasible, using matched instances, explicit ranking controls, paired inference, multiplicity correction, and feature-count analysis.

### Short defense answer: Why TreeSHAP instead of ordinary feature importance?

Ordinary global importance summarizes a model across the dataset. The project needs a local explanation for one alert. TreeSHAP supplies per-instance signed feature contributions and is then independently tested through deletion behavior.

### Short defense answer: Why compare against random features?

Deleting any features may reduce confidence. The random control determines whether the explainer's ranking is actually more useful than an arbitrary ranking.

### Short defense answer: Why use macro-F1?

Network datasets are imbalanced. Macro-F1 gives rare attack classes equal importance and exposes failures that weighted averages can hide.

### Short defense answer: Why use the Wilcoxon test?

The explainers evaluate identical flows, so the samples are paired. With only 20 flows per configuration and no need to assume normal differences, a paired nonparametric Wilcoxon signed-rank test is appropriate.

### Short defense answer: Why apply Holm correction?

Multiple simultaneous comparisons increase false-positive risk. Holm correction controls family-wise error while being less conservative than a simple Bonferroni adjustment.

### Short defense answer: What is the most important limitation?

The benchmark measures faithfulness to fitted models, not causal truth or improvement in analyst decisions. A task-based human study would be required to establish operational human usefulness.

---

## 26. Possible future work

The most valuable extensions are:

- Add capture-day-aware CIC evaluation or another temporally split dataset.
- Increase explanation samples and repeat the full study across dataset resamples.
- Test conditional or neighborhood-preserving masking instead of marginal medians/modes.
- Compare KernelSHAP, permutation methods, counterfactual explanations, and rule explanations.
- Evaluate calibrated probabilities and decision thresholds.
- Add concept-drift experiments and recurring-feature monitoring.
- Test adversarial manipulation of both the classifier and explainer.
- Conduct an analyst study measuring decision accuracy, time, and trust calibration.
- Evaluate deep-learning IDS models while maintaining matched quantitative controls.

Each extension should preserve the central principle of this project: explanation claims must be tied to explicit, falsifiable measurements.

---

## 27. Final takeaway

This project demonstrates that building an explainable IDS involves more than training an accurate classifier and drawing a feature-importance chart. The complete process requires careful data splits, leakage-free preprocessing, class-aware predictive metrics, matched explanation instances, behavioral fidelity controls, stability analysis, operational latency, paired statistics, multiplicity correction, honest limitations, and reproducible artifacts.

The evidence shows that TreeSHAP is more faithful and generally more neighborhood-stable than LIME across the eight tested configurations. It also shows that speed depends strongly on the underlying model and dataset, that feature reduction can help in some contexts but not all, and that high aggregate performance can hide serious minority-class failures. Those nuanced conclusions—not a universal claim that one algorithm is always best—are the central result of the work.
