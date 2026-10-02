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
- Predicted probability: **0.887324**
- Model seed: 42

Class probabilities:

- Attack: 0.887324
- Normal: 0.112676

## Top signed TreeSHAP evidence

| Rank | Feature | Observed value | Attribution | Direction |
|---:|---|---:|---:|---|
| 1 | dload | 0.0 | 1.088695 | supports prediction |
| 2 | ackdat | 0.0 | -0.379357 | opposes prediction |
| 3 | ct_src_dport_ltm | 1 | 0.320968 | supports prediction |
| 4 | dmean | 0 | 0.297214 | supports prediction |
| 5 | dbytes | 0 | 0.293959 | supports prediction |
| 6 | synack | 0.0 | 0.178716 | supports prediction |
| 7 | sloss | 0 | 0.160218 | supports prediction |
| 8 | sjit | 0.0 | -0.153432 | opposes prediction |
| 9 | smean | 248 | 0.152121 | supports prediction |
| 10 | dinpkt | 0.0 | -0.114517 | opposes prediction |

Positive attribution supports the reported class; negative attribution opposes it. Attribution is model evidence, not real-world causality.

## Alert-specific fidelity test

- Top-ranked deletion score: 0.062053
- Bottom-ranked deletion score: -0.027533
- Mean of 200 random rankings: 0.055947
- Top-minus-random lift: 0.006106
- Empirical one-sided local p-value: 0.422886

The deletion score is the mean predicted-probability drop after replacing the top 1, 3, 5, and 10 features with training-derived baseline values.

## Benchmark validation context

- TreeSHAP passed corrected random control: True
- Mean top-minus-random lift: 0.243804
- 95% bootstrap interval: [0.205468, 0.281411]
- Holm-adjusted p-value: 0.000000
- Mean neighbor cosine stability: 0.747170
- Mean top-5 neighbor overlap: 0.629841
- Median/P95 explanation latency: 1.670 / 3.526 ms

Mean TreeSHAP deletion scores from the separate reference-sensitivity audit:

- marginal_training_row: 0.297949
- nearest_opposite_prediction: 0.266929
- nearest_same_prediction: -0.002803
- training_median_mode: 0.290387
- zero: 0.266992

## Interpretation rule

- `SUPPORTED_FOR_THIS_ALERT`: benchmark control passed and this alert beats randomized rankings at empirical p <= 0.05.
- `REVIEW_REQUIRED`: benchmark control passed, but the alert-specific ranking did not clear the strict local test.
- `UNSUPPORTED`: the explainer failed benchmark-level validation.

This report makes the model's decision auditable and tests whether its stated evidence changes the model output under a declared intervention. It does not claim causal truth, intervention-invariant fidelity, complete human comprehension, or immunity to dataset shift.
