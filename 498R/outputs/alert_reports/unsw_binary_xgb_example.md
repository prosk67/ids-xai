# Evidence-Backed IDS Alert Explanation

## Decision transparency status

- **Evidence tier:** `REVIEW_REQUIRED`
- **Meaning:** The explainer passed globally, but this alert did not clear the strict local random-ranking test.
- **Black-box status:** mitigated with tested local evidence; not theoretically eliminated
- **Evidence scope:** primary training median/mode masking intervention
- **Required warning:** This report explains the fitted model under explicit masking tests. It is not a causal explanation and does not replace analyst review.

## Prediction

- Dataset/model context: UNSW-NB15 / binary / xgboost
- Predicted class: **Attack**
- Predicted probability: **0.886009**
- Model seed: 42

Class probabilities:

- Attack: 0.886009
- Normal: 0.113991

## Top signed TreeSHAP evidence

| Rank | Feature | Observed value | Attribution | Direction |
|---:|---|---:|---:|---|
| 1 | dload | 0.0 | 1.061371 | supports prediction |
| 2 | ct_src_dport_ltm | 1 | 0.344122 | supports prediction |
| 3 | ackdat | 0.0 | -0.335638 | opposes prediction |
| 4 | dbytes | 0 | 0.245044 | supports prediction |
| 5 | tcprtt | 0.0 | 0.192306 | supports prediction |
| 6 | synack | 0.0 | 0.167836 | supports prediction |
| 7 | smean | 248 | 0.166420 | supports prediction |
| 8 | dmean | 0 | 0.144980 | supports prediction |
| 9 | sloss | 0 | 0.139830 | supports prediction |
| 10 | djit | 0.0 | 0.081560 | supports prediction |

Positive attribution supports the reported class; negative attribution opposes it. Attribution is model evidence, not real-world causality.

## Alert-specific fidelity test

- Top-ranked deletion score: 0.156774
- Bottom-ranked deletion score: -0.000649
- Mean of 200 random rankings: 0.082928
- Top-minus-random lift: 0.073846
- Empirical one-sided local p-value: 0.189055

The deletion score is the mean predicted-probability drop after replacing the top 1, 3, 5, and 10 features with training-derived baseline values.

## Benchmark validation context

- TreeSHAP passed corrected random control: True
- Mean top-minus-random lift: 0.199435
- 95% bootstrap interval: [0.108212, 0.291204]
- Holm-adjusted p-value: 0.000877
- Mean neighbor cosine stability: 0.543362
- Mean top-5 neighbor overlap: 0.507837
- Median/P95 explanation latency: 2.781 / 2.937 ms

Mean TreeSHAP deletion scores from the separate reference-sensitivity audit:

- marginal_training_row: 0.189395
- nearest_opposite_prediction: 0.236665
- nearest_same_prediction: -0.005704
- training_median_mode: 0.232671
- zero: 0.277167

## Interpretation rule

- `SUPPORTED_FOR_THIS_ALERT`: benchmark control passed and this alert beats randomized rankings at empirical p <= 0.05.
- `REVIEW_REQUIRED`: benchmark control passed, but the alert-specific ranking did not clear the strict local test.
- `UNSUPPORTED`: the explainer failed benchmark-level validation.

This report makes the model's decision auditable and tests whether its stated evidence changes the model output under a declared intervention. It does not claim causal truth, intervention-invariant fidelity, complete human comprehension, or immunity to dataset shift.
