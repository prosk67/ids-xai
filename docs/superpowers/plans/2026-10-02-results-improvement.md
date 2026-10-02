# Results Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen the benchmark's evidence so its headline results survive peer review: no test-set selection anywhere, tuned models, a stronger explanation sample, a harder negative control, and paper text derived from the regenerated results.

**Architecture:** Keep the existing pipeline (`benchmark.py` → `extended_audit.py` → `paper.py` → `verify_outputs.py`). Add a stratified validation split carved from training data and route every selection decision (hyperparameters, feature count, calibration) through it. Expand explanation evaluation from 40 flows on one model seed to 200 fixed flows on all five model seeds, add a model-native global-importance control, and batch model calls so the larger run stays fast. Finally regenerate all artifacts and rewrite result-dependent prose from the new CSVs.

**Tech Stack:** Python 3.12, numpy 2.2.6, pandas 2.2.3, scikit-learn 1.6.1, xgboost 3.0.2, shap 0.47.2, lime 0.2.0.1, reportlab 4.4.1, `unittest`.

**Spec:** No separate spec. The requirements come from the project review on 2026-10-02 (summarised under "Why each change" below) and from `Lit_Review_498R__IDS_.pdf` §6.2–6.3 and §8.

## Why each change

| Weakness a reviewer will raise | Change | Task |
|---|---|---|
| `verify_outputs.py` asserts specific outcomes (8/8, 7/8, CI > 0, 40 flows), so honest new results would "fail" verification | Verifier checks structure and internal consistency, not which hypothesis won | 1 |
| Fixed, untuned hyperparameters; no validation split exists | Stratified validation split from training data; small grid search chosen by validation macro-F1 | 2, 3 |
| Feature-count "best k" chosen on test results (paper admits this) | Choose k on validation, report test only for the chosen k | 4 |
| Only random/bottom controls: a local explanation might be no better than a fixed global ranking | Add a "global" control (model `feature_importances_` ranking, same for every flow) | 5 |
| 40 flows, one model seed (42) | 200 fixed flows explained on all 5 seed models; per-flow scores averaged over seeds before the bootstrap | 6 |
| Masking-reference audit on 12 flows; LIME-budget audit on 6 flows; ECE diagnosed but never fixed; SHAP background ablation missing (lit review §6.2) | Audits sized from config; validation-fitted isotonic recalibration; interventional TreeSHAP background ablation for XGBoost | 7 |
| Paper prose hard-codes numbers and conclusions | Compute every number from CSVs and rewrite conclusions after the rerun | 8 |
| Deletion fidelity only tests the top of the ranking; no explainer-agnostic faithfulness metric from the XAI-evaluation literature (Bhatt et al. 2020; used by Quantus and E-XAI) | Faithfulness correlation: Pearson correlation between attribution mass of random feature subsets and the probability drop when that subset is masked | 5 |
| No sanity check that explanations depend on what the model learned (Adebayo et al. 2018) | Data-randomization test: explain a model trained on permuted labels; real-model explanations must differ from it | 7 |
| UNSW minority-class failures are reported without asking whether they are learnable | Label-conflict audit: share of rows whose exact feature vector also appears with a different label, per class, and the resulting accuracy ceiling | 7 |

### Research basis (checked 2026-10-02)

- Arreche et al., "E-XAI: Evaluating black-box XAI frameworks for network intrusion detection," IEEE Access 2024: closest prior work (SHAP/LIME on IDS, six metrics). The paper must cite it and state what is new here (matched negative controls, paired inference, reject option).
- Bhatt, Weller, Moura, "Evaluating and aggregating feature-based model explanations," IJCAI 2020: faithfulness correlation.
- Adebayo et al., "Sanity checks for saliency maps," NeurIPS 2018: model/data randomization tests (already cited as [16]).
- Hooker et al., ROAR, NeurIPS 2019; Rong et al., ROAD, ICML 2022: masking creates out-of-distribution inputs, so deletion scores must be read alongside reference sensitivity. This motivates keeping the reference audit and reporting top vs. bottom (MoRF/LeRF) consistency.
- Chen et al., "True to the model or true to the data?", 2020: path-dependent vs. interventional TreeSHAP (Task 7 background ablation).
- Engelen et al., "Troubleshooting an intrusion detection dataset: the CICIDS2017 case study," 2021: about 20% of CIC-IDS2017 flows relabelled or reconstructed. Cite as a threat to validity; the corrected data is not available locally.
- UNSW-NB15 label conflicts, measured on this project's cleaned split: 6.3% of training rows (Analysis 66%, Backdoor 67%, DoS 28%) share an exact feature vector with a differently labelled row, and the feature-identical accuracy ceiling is 0.951 on training and 0.980 on test.

Not in scope: temporal or corrected CIC-IDS2017 (needs new data; the mirror lacks timestamps), new model families, and a human analyst study.

## Global Constraints

- Python 3.12 virtualenv at `.venv/`; run everything with `.venv/bin/python -B`.
- Raw data at `data/raw/` (symlink, git-ignored). SHA-256 hashes must still match `outputs/run_manifest.json` → `files`.
- Predictive seeds stay `[42, 52, 62, 72, 82]`; the data split and selection seed stays `seed: 42`.
- The test partition must never influence a selection decision (hyperparameters, feature count, calibration, explanation sample).
- Every number printed in the paper must be computed from an output file, never typed as a literal.
- Generated outputs over 10 MB are written gzip-compressed (`*.csv.gz`); pandas reads them transparently.
- Tests use `unittest` and live in `tests/`; run with `.venv/bin/python -B -m unittest discover -s tests -v`.
- Commit after each task on branch `fix/reproducibility` (or a child branch), with the Co-Authored-By trailer.

## Review Focus

1. **Rare classes in the validation split** (CIC Heartbleed and SQL Injection have about 10 training rows): a stratified split must still leave every class in the fit partition, or XGBoost multiclass raises on missing labels. Pinned by `test_validation_split_keeps_every_class_in_fit_partition` (Task 2).
2. **Ties in the hyperparameter grid**: when several candidates share the best validation score, the prespecified base configuration must win, so tuning never changes a model without evidence. Pinned by `test_hyperparameter_selection_prefers_base_on_ties` (Task 3).
3. **Batched fidelity must equal the old per-row computation**: the speed-up must not silently change the numbers. Pinned by `test_batched_fidelity_matches_row_by_row` (Task 5).
4. **Explanation flows identical across explainers and seeds**: paired tests are invalid otherwise. Pinned by `test_explanation_flows_shared_across_seeds_and_explainers` (Task 6) and by the verifier (Task 1).
5. **The alert report must keep reading the random-control row** once `fidelity_control_tests.csv` gains a `reference` column. Pinned by `test_benchmark_context_uses_random_reference` (Task 5).

---

### Task 1: Make the verifier check integrity, not outcomes

**Files:**
- Modify: `verify_outputs.py` (whole file; currently 130 lines)

**Interfaces:**
- Consumes: `config.yaml` keys `evaluation.explanation_samples`, `predictive_seeds`.
- Produces: `verify_outputs.py` keeps the same CLI (`python -B verify_outputs.py`) and prints `All output, manifest, and paper checks passed.` on success. Tasks 5–8 extend it.

- [ ] **Step 1: Replace the outcome assertions with config-derived structural checks**

In `verify_outputs.py`, load the config and replace these outcome-specific lines:

```python
assert (paired["ci_low"] > 0).all()
assert (paired["n_pairs"] == 40).all()
...
    assert len(source_sets["TreeSHAP"]) == 40
...
assert (tree_controls["holm_adjusted_p"] < 0.05).sum() == 8
assert (lime_controls["holm_adjusted_p"] < 0.05).sum() == 7
```

with:

```python
import yaml

config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
expected_flows = int(config["evaluation"]["explanation_samples"])

assert paired["n_pairs"].eq(expected_flows).all(), paired["n_pairs"].unique()
assert paired["ci_low"].le(paired["mean_difference_shap_minus_lime"]).all()
assert paired["mean_difference_shap_minus_lime"].le(paired["ci_high"]).all()
assert paired["holm_adjusted_p"].between(0, 1).all()
assert paired["holm_adjusted_p"].ge(paired["wilcoxon_one_sided_p"]).all()
```

and inside the source-set loop:

```python
    assert len(source_sets["TreeSHAP"]) == expected_flows
```

and for the controls:

```python
assert controls["holm_adjusted_p"].between(0, 1).all()
assert controls["ci_low"].le(controls["ci_high"]).all()
```

Keep `assert len(controls) == 16` for now (Task 5 changes it). Keep the UNSW cleaning-count assertions: they are deterministic data audits, not outcomes.

- [ ] **Step 2: Bound the page count instead of fixing it**

Replace `assert len(reader.pages) == 6` with:

```python
# Six pages is the current draft; allow growth up to a typical IEEE
# conference limit until the target venue is chosen.
assert 6 <= len(reader.pages) <= 8, len(reader.pages)
```

- [ ] **Step 3: Run the verifier against the committed outputs**

Run: `.venv/bin/python -B verify_outputs.py`
Expected: `All output, manifest, and paper checks passed.`

- [ ] **Step 4: Commit**

```bash
git add verify_outputs.py
git commit -m "Verify output integrity instead of specific outcomes"
```

---

### Task 2: Stratified validation split from training data

**Files:**
- Modify: `src/xai_ids/benchmark.py` (add a function after `split_cic`, around line 301)
- Modify: `config.yaml` (add a `tuning` block)
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Produces: `split_validation(labels: np.ndarray, size: float, seed: int) -> tuple[np.ndarray, np.ndarray]` returning sorted, disjoint `fit_idx, val_idx` integer arrays that index the training rows. Tasks 3, 4 and 7 use it.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_benchmark.py` (and add `split_validation` to the import list):

```python
    def test_validation_split_is_disjoint_stratified_and_deterministic(self):
        labels = np.array([0] * 80 + [1] * 20)
        fit_a, val_a = split_validation(labels, 0.15, seed=42)
        fit_b, val_b = split_validation(labels, 0.15, seed=42)
        np.testing.assert_array_equal(fit_a, fit_b)
        np.testing.assert_array_equal(val_a, val_b)
        self.assertEqual(len(np.intersect1d(fit_a, val_a)), 0)
        self.assertEqual(len(fit_a) + len(val_a), len(labels))
        self.assertEqual(int((labels[val_a] == 1).sum()), 3)

    def test_validation_split_keeps_every_class_in_fit_partition(self):
        labels = np.array([0] * 200 + [1] * 3 + [2] * 2)
        fit, _ = split_validation(labels, 0.15, seed=42)
        self.assertEqual(set(np.unique(labels[fit])), {0, 1, 2})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -B -m unittest tests.test_benchmark -v`
Expected: ImportError, `cannot import name 'split_validation'`.

- [ ] **Step 3: Implement**

Add after `split_cic` in `src/xai_ids/benchmark.py`:

```python
def split_validation(
    labels: np.ndarray, size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a stratified validation subset from the training rows.

    Every selection decision (hyperparameters, feature count, calibration)
    uses this subset so the test partition never influences a choice. Classes
    too small to stratify are kept entirely in the fit partition, which XGBoost
    multiclass training requires.
    """
    from sklearn.model_selection import train_test_split

    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    classes, counts = np.unique(labels, return_counts=True)
    rare = np.isin(labels, classes[counts < 3])
    fit_idx, val_idx = train_test_split(
        indices[~rare], test_size=size, random_state=seed, stratify=labels[~rare]
    )
    fit_idx = np.concatenate([fit_idx, indices[rare]])
    return np.sort(fit_idx), np.sort(val_idx)
```

Add to `config.yaml`, after the `models:` block:

```yaml
tuning:
  validation_size: 0.15
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -B -m unittest discover -s tests -v`
Expected: all 8 tests OK.

- [ ] **Step 5: Commit**

```bash
git add src/xai_ids/benchmark.py tests/test_benchmark.py config.yaml
git commit -m "Add stratified validation split carved from training data"
```

---

### Task 3: Train-only hyperparameter selection

**Files:**
- Modify: `src/xai_ids/benchmark.py`: `make_model` (lines 303-320), new functions after it, `run()` (lines 1039-1057, 1159-1205)
- Modify: `config.yaml` (`tuning` grids)
- Modify: `src/xai_ids/extended_audit.py` (`run()` lines 364-367: the unweighted model must use the selected parameters)
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `split_validation` (Task 2).
- Produces:
  - `make_model(name: str, params: dict[str, Any], seed: int, n_classes: int)`. **Signature change:** the second argument is now the model's own parameter dict, not the whole `models` config. All callers are updated in this task (`run`, `feature_ablation`, `extended_audit.run`).
  - `candidate_params(base: dict, grid: dict[str, list]) -> list[dict]`: base first, then the grid product with duplicates removed.
  - `select_hyperparameters(model_name, base, grid, x_fit, y_fit, x_val, y_val, seed, n_classes) -> tuple[dict, list[dict]]`.
  - New output `outputs/tuning.csv` (columns `dataset, task, model, candidate, validation_macro_f1, selected, param_*`).
  - Manifest key `selected_hyperparameters["<dataset>/<task>/<model>"] -> dict`. `extended_audit.py` reads it.

- [ ] **Step 1: Write the failing tests**

```python
    def test_candidate_params_puts_base_first_without_duplicates(self):
        base = {"max_depth": 7, "learning_rate": 0.08}
        grid = {"max_depth": [7, 10]}
        candidates = candidate_params(base, grid)
        self.assertEqual(candidates[0], base)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[1], {"max_depth": 10, "learning_rate": 0.08})

    def test_hyperparameter_selection_prefers_base_on_ties(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 3))
        y = (x[:, 0] > 0).astype(int)
        base = {"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 1}
        grid = {"max_depth": [4, 5, 6]}
        chosen, rows = select_hyperparameters(
            "random_forest", base, grid, x[:150], y[:150], x[150:], y[150:], 42, 2
        )
        best = max(row["validation_macro_f1"] for row in rows)
        winners = [row for row in rows if row["validation_macro_f1"] == best]
        self.assertEqual(chosen["max_depth"], winners[0]["param_max_depth"])
        self.assertEqual(sum(row["selected"] for row in rows), 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -B -m unittest tests.test_benchmark -v`
Expected: ImportError for `candidate_params`.

- [ ] **Step 3: Implement**

Replace `make_model` and add the selection helpers (add `import itertools` at the top of the file):

```python
def make_model(name: str, params: dict[str, Any], seed: int, n_classes: int):
    if name == "random_forest":
        return RandomForestClassifier(
            **params,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "xgboost":
        params = dict(params)
        params.update(
            random_state=seed,
            n_jobs=max(1, min(8, os.cpu_count() or 1)),
            tree_method="hist",
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
        )
        return XGBClassifier(**params)
    raise KeyError(name)


def candidate_params(base: dict[str, Any], grid: dict[str, list]) -> list[dict[str, Any]]:
    """Return the prespecified configuration first, then distinct grid points."""
    keys = sorted(grid)
    candidates = [dict(base)]
    for values in itertools.product(*(grid[key] for key in keys)):
        candidate = {**base, **dict(zip(keys, values))}
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def select_hyperparameters(
    model_name: str,
    base: dict[str, Any],
    grid: dict[str, list],
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    n_classes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose parameters by validation macro-F1; earlier candidates win ties."""
    rows = []
    best_score, best_params = -np.inf, dict(base)
    for number, params in enumerate(candidate_params(base, grid)):
        model = make_model(model_name, params, seed, n_classes)
        weight = compute_sample_weight("balanced", y_fit) if model_name == "xgboost" else None
        model.fit(x_fit, y_fit, sample_weight=weight)
        score = float(f1_score(y_val, model.predict(x_val), average="macro", zero_division=0))
        rows.append(
            {
                "candidate": number,
                "validation_macro_f1": score,
                **{f"param_{key}": value for key, value in params.items()},
            }
        )
        if score > best_score:
            best_score, best_params = score, params
    for row in rows:
        row["selected"] = int(
            all(row[f"param_{key}"] == value for key, value in best_params.items())
        )
    return best_params, rows
```

Update `feature_ablation` so it takes `model_params: dict` instead of `model_config: dict`, and calls `make_model(model_name, model_params, seed, len(np.unique(y_train)))`.

In `run()`, inside the `for model_name in (...)` loop and before the seed loop, add:

```python
                fit_idx, val_idx = split_validation(
                    y_train, float(config["tuning"]["validation_size"]), seed
                )
                print(f"[tune] {dataset} / {task} / {model_name}", flush=True)
                selected, tuning = select_hyperparameters(
                    model_name,
                    config["models"][model_name],
                    config["tuning"].get(model_name, {}),
                    x_train[fit_idx],
                    y_train[fit_idx],
                    x_train[val_idx],
                    y_train[val_idx],
                    seed,
                    len(classes),
                )
                tuning_rows.extend(
                    {"dataset": dataset, "task": task, "model": model_name, **row}
                    for row in tuning
                )
                selected_hyperparameters[f"{dataset}/{task}/{model_name}"] = selected
```

Replace `make_model(model_name, config["models"], int(model_seed), len(classes))` with `make_model(model_name, selected, int(model_seed), len(classes))`, and pass `selected` to `feature_ablation`. Final models still train on the full training partition with the selected parameters. Initialise `tuning_rows: list[dict] = []` and `selected_hyperparameters: dict[str, dict] = {}` next to the other accumulators. Write `pd.DataFrame(tuning_rows).to_csv(output_dir / "tuning.csv", index=False)` and add `"selected_hyperparameters": selected_hyperparameters,` to the manifest dict.

In `extended_audit.py`, load the manifest once at the top of `run()`:

```python
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
```

and replace `make_model(model_name, config["models"], seed, len(classes))` with:

```python
                params = manifest["selected_hyperparameters"][f"{dataset}/{task}/{model_name}"]
                unweighted = make_model(model_name, params, seed, len(classes))
```

Add the grids to `config.yaml` under `tuning:`:

```yaml
  random_forest:
    max_depth: [18, null]
    min_samples_leaf: [1, 2]
    max_features: [sqrt, 0.5]
  xgboost:
    n_estimators: [160, 400]
    max_depth: [7, 10]
    learning_rate: [0.08, 0.15]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -B -m unittest discover -s tests -v`
Expected: all 10 tests OK.

- [ ] **Step 5: Smoke-test the pipeline on a reduced config**

Create a scratch config with `predictive_seeds: [42]`, `explanation_samples: 10`, `lime_samples: 50` and an `output_dir` in the scratchpad, then run:
`.venv/bin/python -B run_benchmark.py --config <scratch>/smoke.yaml`
Expected: finishes; `<scratch>/smoke_outputs/tuning.csv` has exactly one `selected == 1` per dataset/task/model (8 groups).

- [ ] **Step 6: Commit**

```bash
git add src/xai_ids/benchmark.py src/xai_ids/extended_audit.py tests/test_benchmark.py config.yaml
git commit -m "Select hyperparameters on a training-only validation split"
```

---

### Task 4: Choose feature count on validation, not test

**Files:**
- Modify: `src/xai_ids/benchmark.py`: `feature_ablation` (lines 780-826), its call in `run()`, and `make_summary_plots` (feature-ablation panel, lines 906-923)
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `split_validation`, `make_model(name, params, ...)`.
- Produces: `choose_feature_count(validation_scores: dict[int, float], full_count: int, tolerance: float = 0.005) -> int`, which returns the smallest k whose validation macro-F1 is within `tolerance` of the full model's. `feature_ablation.csv` gains `validation_macro_f1` and `selected` columns, and the existing `macro_f1` is renamed `test_macro_f1`.

- [ ] **Step 1: Write the failing test**

```python
    def test_feature_count_choice_uses_smallest_noninferior_k(self):
        scores = {5: 0.90, 10: 0.948, 20: 0.951, 34: 0.952}
        self.assertEqual(choose_feature_count(scores, 34), 10)
        self.assertEqual(choose_feature_count({5: 0.5, 34: 0.9}, 34), 34)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -B -m unittest tests.test_benchmark -v`
Expected: ImportError for `choose_feature_count`.

- [ ] **Step 3: Implement**

```python
def choose_feature_count(
    validation_scores: dict[int, float], full_count: int, tolerance: float = 0.005
) -> int:
    """Smallest feature count whose validation macro-F1 is within tolerance of full."""
    threshold = validation_scores[full_count] - tolerance
    return min(k for k, score in validation_scores.items() if score >= threshold)
```

Rewrite `feature_ablation` with this signature:

```python
def feature_ablation(
    dataset, task, model_name, model_params,
    x_train, y_train, x_test, y_test, feature_names, ranking, k_values, seed,
    validation_size: float,
) -> list[dict[str, Any]]:
```

For each k it should:
1. fit on `x_train[fit_idx][:, selected]` (with `fit_idx, val_idx = split_validation(y_train, validation_size, seed)` computed once);
2. score `validation_macro_f1` on `x_train[val_idx][:, selected]`;
3. refit on the full `x_train[:, selected]` and score `test_macro_f1` and `weighted_f1` on `x_test`.

Keep the existing timing and SHAP-latency fields. After the loop, call `choose_feature_count({row["feature_count"]: row["validation_macro_f1"] for row in rows}, len(feature_names))` and set `row["selected"] = int(row["feature_count"] == chosen)`. The `ranking` still comes from the seed-42 model's `feature_importances_`, which is fitted on training data only.

In `make_summary_plots`, change the feature-ablation `y="macro_f1"` to `y="test_macro_f1"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -B -m unittest discover -s tests -v`
Expected: all 11 tests OK.

- [ ] **Step 5: Commit**

```bash
git add src/xai_ids/benchmark.py tests/test_benchmark.py
git commit -m "Choose ablation feature count on validation data"
```

---

### Task 5: Global-importance control and batched deletion scoring

**Files:**
- Modify: `src/xai_ids/benchmark.py`: `fidelity_rows` (lines 413-447), `fidelity_control_tests` (lines 738-777), `evaluate_explainer` signature and call
- Modify: `src/xai_ids/transparency.py`: `_ranking_score` (lines 36-62), `_benchmark_context` (lines 120-130)
- Modify: `src/xai_ids/extended_audit.py` (summary JSON at lines 432-438 filters `reference == "random"`)
- Modify: `verify_outputs.py` (controls count)
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Produces:
  - `fidelity_rows(model, row, attribution, predicted_class, baseline, k_values, rng, global_ranking: np.ndarray | None = None)`, which emits `control` values `top`, `bottom`, `random` and, when `global_ranking` is given, `global`. All masked variants go through **one** `predict_proba` call. The RNG is consumed exactly as before (one `permutation` per call).
  - `fidelity_control_tests(fidelity, seed)` returns rows with a new `reference` column (`random` or `global`). Holm correction is applied separately within each reference family (16 tests each), giving 32 rows in total.
  - `_ranking_score` makes one batched `predict_proba` call; same return values.

- [ ] **Step 1: Write the failing tests**

```python
    def test_batched_fidelity_matches_row_by_row(self):
        from sklearn.ensemble import RandomForestClassifier
        rng_data = np.random.default_rng(1)
        x = rng_data.normal(size=(300, 6))
        y = (x[:, 0] + x[:, 1] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(x, y)
        row, baseline = x[0], np.median(x, axis=0)
        attribution = np.array([3.0, -2.0, 0.5, 0.1, 0.0, 0.2])
        predicted = int(model.predict(row.reshape(1, -1))[0])
        rows = fidelity_rows(
            model, row, attribution, predicted, baseline, [1, 3], np.random.default_rng(7)
        )
        expected_rng = np.random.default_rng(7)
        random_rank = expected_rng.permutation(6)
        original = model.predict_proba(row.reshape(1, -1))[0, predicted]
        masked = row.copy()
        masked[random_rank[:3]] = baseline[random_rank[:3]]
        expected = original - model.predict_proba(masked.reshape(1, -1))[0, predicted]
        got = [r for r in rows if r["control"] == "random" and r["k"] == 3][0]
        self.assertAlmostEqual(got["probability_drop"], expected, places=12)
        self.assertEqual({r["control"] for r in rows}, {"top", "bottom", "random"})

    def test_global_control_masks_the_global_ranking(self):
        from sklearn.ensemble import RandomForestClassifier
        rng_data = np.random.default_rng(2)
        x = rng_data.normal(size=(300, 4))
        y = (x[:, 2] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(x, y)
        rows = fidelity_rows(
            model, x[0], np.ones(4), int(model.predict(x[:1])[0]), np.zeros(4),
            [1], np.random.default_rng(0), global_ranking=np.array([2, 0, 1, 3]),
        )
        self.assertIn("global", {r["control"] for r in rows})

    def test_control_tests_report_random_and_global_families(self):
        records = []
        for idx in range(12):
            for control, drop in (("top", 0.5 + idx / 100), ("random", 0.1), ("global", 0.4), ("bottom", 0.0)):
                records.append({"dataset": "D", "task": "binary", "model": "m",
                                "explainer": "TreeSHAP", "source_test_index": idx,
                                "control": control, "probability_drop": drop})
        result = fidelity_control_tests(pd.DataFrame(records), seed=0)
        self.assertEqual(set(result["reference"]), {"random", "global"})
        self.assertTrue((result["ci_low"] > 0).all())
```

and, in a new test class in the same file:

```python
class TransparencyContextTests(unittest.TestCase):
    def test_benchmark_context_uses_random_reference(self):
        import tempfile
        from pathlib import Path
        from src.xai_ids.transparency import _benchmark_context
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            base = {"dataset": "D", "task": "binary", "model": "m", "explainer": "TreeSHAP"}
            pd.DataFrame([
                {**base, "reference": "global", "mean_lift_top_minus_reference": -0.1,
                 "ci_low": -0.2, "ci_high": 0.0, "holm_adjusted_p": 0.9},
                {**base, "reference": "random", "mean_lift_top_minus_reference": 0.2,
                 "ci_low": 0.1, "ci_high": 0.3, "holm_adjusted_p": 0.001},
            ]).to_csv(results / "fidelity_control_tests.csv", index=False)
            pd.DataFrame([{**base, "stability_cosine_mean": 0.9, "stability_jaccard_mean": 0.7,
                           "latency_median_ms": 1.0, "latency_p95_ms": 2.0,
                           "n_explanations": 200}]).to_csv(results / "explanation_summary.csv", index=False)
            context = _benchmark_context(results, "D", "binary", "m")
        self.assertTrue(context["global_random_control_passed"])
        self.assertAlmostEqual(context["global_top_minus_random_lift"], 0.2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -B -m unittest discover -s tests -v`
Expected: the global-control, control-family and context tests FAIL (no `global` control, no `reference` column). The batched-equality test may already pass, which is fine: it guards the refactor.

- [ ] **Step 3: Implement batched `fidelity_rows` with the global control**

```python
def fidelity_rows(
    model,
    row: np.ndarray,
    attribution: np.ndarray,
    predicted_class: int,
    baseline: np.ndarray,
    k_values: list[int],
    rng: np.random.Generator,
    global_ranking: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    ranked = np.argsort(-np.abs(attribution))
    rankings = [("top", ranked), ("bottom", ranked[::-1]), ("random", rng.permutation(len(ranked)))]
    if global_ranking is not None:
        rankings.append(("global", np.asarray(global_ranking)))
    variants = [row]
    labels = []
    for k in k_values:
        k_eff = min(int(k), len(row))
        for control, order in rankings:
            masked = row.copy()
            masked[order[:k_eff]] = baseline[order[:k_eff]]
            variants.append(masked)
            labels.append((k_eff, control))
    # One model call per flow: single-row predict_proba on a parallel forest
    # costs milliseconds of thread start-up each time.
    probabilities = model.predict_proba(np.vstack(variants))
    original = float(probabilities[0, predicted_class])
    return [
        {
            "k": k_eff,
            "control": control,
            "probability_drop": original - float(p[predicted_class]),
            "flipped": int(int(np.argmax(p)) != predicted_class),
        }
        for (k_eff, control), p in zip(labels, probabilities[1:])
    ]
```

- [ ] **Step 4: Generalise `fidelity_control_tests`**

Factor the Holm block, which is duplicated in `paired_bootstrap_comparisons` and `fidelity_control_tests`, into:

```python
def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(np.argsort(p_values)):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted
```

Use it in `paired_bootstrap_comparisons`. Rewrite `fidelity_control_tests` to loop `for reference in ("random", "global"):` over each group. Skip a reference when its column is absent. Compute `top - reference` differences per source flow exactly as now, store `reference` and `mean_lift_top_minus_reference`, and apply `holm_adjust` within each reference family. Also keep the old column name `mean_lift_top_minus_random`, filled only for `reference == "random"`, so `paper.py` keeps working until Task 8.

- [ ] **Step 5: Wire the global ranking into `evaluate_explainer`**

Add a parameter `global_ranking: np.ndarray` to `evaluate_explainer` and pass it to `fidelity_rows(..., global_ranking=global_ranking)`. In `run()`, compute it once per frozen model, before the explainer loop:

```python
                global_ranking = np.argsort(-np.asarray(frozen_model.feature_importances_))
```

and reuse the same array for the existing `ranking` passed to `feature_ablation`.

- [ ] **Step 6: Batch `_ranking_score` and filter the alert context**

In `transparency.py`:

```python
def _ranking_score(model, row, predicted_class, baseline, ranking, k_values):
    variants = [row]
    k_list = []
    for k in k_values:
        k_eff = min(int(k), len(row))
        masked = row.copy()
        masked[ranking[:k_eff]] = baseline[ranking[:k_eff]]
        variants.append(masked)
        k_list.append(k_eff)
    probabilities = model.predict_proba(np.vstack(variants))
    original = float(probabilities[0, predicted_class])
    records = [
        {
            "k": k_eff,
            "probability_after_masking": float(p[predicted_class]),
            "probability_drop": original - float(p[predicted_class]),
            "prediction_flipped": bool(np.argmax(p) != predicted_class),
        }
        for k_eff, p in zip(k_list, probabilities[1:])
    ]
    return float(np.mean([r["probability_drop"] for r in records])), records
```

In `_benchmark_context`, add `& controls.get("reference", pd.Series("random", index=controls.index)).eq("random")` to the controls selector, and read the lift from `mean_lift_top_minus_reference` when that column exists, otherwise from `mean_lift_top_minus_random`.

In `extended_audit.py`, filter `controls = controls[controls.reference.eq("random")]` before computing `black_box_audit_summary.json`. Add `global_control_passed` counts for both explainers to that JSON.

In `verify_outputs.py`, change `assert len(controls) == 16` to:

```python
assert len(controls) == 32
assert controls.groupby("reference").size().to_dict() == {"global": 16, "random": 16}
```

- [ ] **Step 6b: Faithfulness correlation (explainer-agnostic metric)**

Test first:

```python
    def test_faithfulness_correlation_is_high_for_true_linear_weights(self):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(4)
        x = rng.normal(size=(500, 6))
        weights = np.array([3.0, -2.0, 1.0, 0.5, 0.0, 0.0])
        y = (x @ weights > 0).astype(int)
        model = LogisticRegression().fit(x, y)
        row = x[0]
        predicted = int(model.predict(row.reshape(1, -1))[0])
        sign = 1 if predicted == 1 else -1
        attribution = sign * model.coef_[0] * (row - 0.0)
        good = faithfulness_correlation(model, row, attribution, predicted, np.zeros(6), 3, 60, np.random.default_rng(0))
        bad = faithfulness_correlation(model, row, -attribution, predicted, np.zeros(6), 3, 60, np.random.default_rng(0))
        self.assertGreater(good, 0.5)
        self.assertLess(bad, 0.0)
```

Implementation (in `benchmark.py`, next to `fidelity_rows`):

```python
def faithfulness_correlation(
    model,
    row: np.ndarray,
    attribution: np.ndarray,
    predicted_class: int,
    baseline: np.ndarray,
    subset_size: int,
    n_subsets: int,
    rng: np.random.Generator,
) -> float:
    """Bhatt et al. (2020): correlate attribution mass with the masking effect.

    Random feature subsets are replaced by the baseline; a faithful explanation
    assigns larger summed attribution to subsets whose removal lowers the
    predicted-class probability more.
    """
    subsets = [rng.choice(len(row), size=min(subset_size, len(row)), replace=False) for _ in range(n_subsets)]
    variants = [row]
    for subset in subsets:
        masked = row.copy()
        masked[subset] = baseline[subset]
        variants.append(masked)
    probabilities = model.predict_proba(np.vstack(variants))[:, predicted_class]
    drops = probabilities[0] - probabilities[1:]
    mass = np.asarray([attribution[subset].sum() for subset in subsets])
    if np.std(drops) == 0 or np.std(mass) == 0:
        return 0.0
    return float(np.corrcoef(mass, drops)[0, 1])
```

In `evaluate_explainer`, compute it once per flow after `fidelity_rows`, with a dedicated generator `np.random.default_rng(seed + 1)` created once per call (so the existing `rng` stream, and therefore the random-control results, are unchanged), using `subset_size = config.get("faithfulness_subset_size", 5)` and `n_subsets = config.get("faithfulness_subsets", 50)`. Store it as a `"faithfulness_correlation"` field on that flow's latency row (one row per flow). In `summarize_explanations` (Task 6), add `faithfulness_correlation_mean` (mean over flows and seeds). Add a paired TreeSHAP − LIME comparison of per-flow faithfulness correlation to `paired_bootstrap_comparisons` as a second family by adding a `metric` column (`deletion` or `faithfulness_correlation`), with Holm applied within each metric family.

- [ ] **Step 7: Run tests and the smoke config**

Run: `.venv/bin/python -B -m unittest discover -s tests -v`. Expected: all tests OK.
Run the Task 3 smoke config again. Expected: finishes; `fidelity_control_tests.csv` has a `reference` column with both values.

- [ ] **Step 8: Commit**

```bash
git add src/xai_ids tests verify_outputs.py
git commit -m "Add global-importance fidelity control and batch masked predictions"
```

---

### Task 6: Larger explanation sample on every model seed

**Files:**
- Modify: `src/xai_ids/benchmark.py`: `evaluate_explainer` (lines 496-687), new `summarize_explanations`, `run()` explain loop and output writing
- Modify: `config.yaml` (`evaluation`)
- Modify: `src/xai_ids/extended_audit.py` (`_summarize_existing_outputs` reads `fidelity.csv.gz`)
- Modify: `verify_outputs.py` (file names; seed coverage)
- Modify: `PROJECT_WALKTHROUGH.ipynb` (two `fidelity.csv` reads become `fidelity.csv.gz`)
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `fidelity_rows(..., global_ranking)` (Task 5), `make_model(name, params, ...)` (Task 3).
- Produces:
  - `evaluate_explainer(dataset, task, model_name, explainer_name, model, model_seed, indices, x_train, x_test, y_test, preprocessor, classes, config, seed, global_ranking, measure_self_stability: bool)` → `(attribution_rows, fidelity_rows, stability_rows, latency_rows, self_stability_values)`. Every row carries `model_seed`. Flows are passed in as `indices` and no longer chosen inside.
  - `summarize_explanations(fidelity: pd.DataFrame, stability: pd.DataFrame, latency: pd.DataFrame, self_stability: dict, seed: int) -> pd.DataFrame` with the same 16-row schema as today's `explanation_summary.csv`. Fidelity is averaged per flow over seeds and k, then bootstrapped over flows.
  - Outputs: `fidelity.csv.gz` (all seeds), `stability.csv` (all seeds), new `latency.csv`, and `attributions.csv.gz` (seed 42 only, to bound size).

- [ ] **Step 1: Write the failing test**

```python
    def test_explanation_flows_shared_across_seeds_and_explainers(self):
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(3)
        x = rng.normal(size=(400, 5))
        y = (x[:, 0] > 0).astype(int)
        pre = TabularPreprocessor([f"f{i}" for i in range(5)]).fit(pd.DataFrame(x, columns=[f"f{i}" for i in range(5)]))
        indices = np.arange(300, 320)
        config = {"lime_samples": 50, "fidelity_k": [1, 3], "self_stability_samples": 2, "lime_repeat_seeds": 2}
        seen = {}
        for model_seed in (42, 52):
            model = RandomForestClassifier(n_estimators=10, random_state=model_seed).fit(x[:300], y[:300])
            for name in ("TreeSHAP", "LIME"):
                _, fidelity, _, latency, _ = evaluate_explainer(
                    "D", "binary", "random_forest", name, model, model_seed, indices,
                    x[:300], x, y, pre, ["Attack", "Normal"], config, 42,
                    np.arange(5), measure_self_stability=False,
                )
                seen[(model_seed, name)] = {r["source_test_index"] for r in fidelity}
                self.assertEqual({r["model_seed"] for r in latency}, {model_seed})
        self.assertEqual(len({frozenset(v) for v in seen.values()}), 1)
        self.assertEqual(seen[(42, "TreeSHAP")], set(indices.tolist()))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -B -m unittest tests.test_benchmark -v`
Expected: FAIL with TypeError (unexpected positional arguments).

- [ ] **Step 3: Refactor `evaluate_explainer`**

Change the signature as given under Interfaces. Delete the internal `explanation_indices(...)` call and use `indices` as passed. Add `"model_seed": int(model_seed)` to `common` and to every stability row. For each flow, emit a latency row `{"dataset", "task", "model", "explainer", "model_seed", "source_test_index", "latency_ms"}` instead of only accumulating `speed_ms`. Run the LIME self-stability block only when `measure_self_stability` is true, and return its pairwise cosines as a list. Remove the summary block, which moves to `summarize_explanations`. Pass `global_ranking` to `fidelity_rows`.

- [ ] **Step 4: Add `summarize_explanations`**

```python
def summarize_explanations(
    fidelity: pd.DataFrame,
    stability: pd.DataFrame,
    latency: pd.DataFrame,
    self_stability: dict[tuple[str, str, str, str], list[float]],
    seed: int,
) -> pd.DataFrame:
    keys = ["dataset", "task", "model", "explainer"]
    rows = []
    for group_key, group in fidelity.groupby(keys):
        per_flow = group.groupby(["source_test_index", "control"])["probability_drop"].mean().unstack("control")
        top_flips = group[group.control.eq("top")]["flipped"].mean()
        mean, low, high = bootstrap_mean_ci(per_flow["top"].to_numpy(), seed)
        stab = stability.set_index(keys).loc[group_key] if len(stability) else pd.DataFrame()
        lat = latency.set_index(keys).loc[group_key, "latency_ms"].to_numpy()
        lime_self = self_stability.get(group_key, [])
        rows.append({
            **dict(zip(keys, group_key)),
            "n_explanations": int(per_flow.shape[0]),
            "n_model_seeds": int(group["model_seed"].nunique()),
            "fidelity_mean": mean,
            "fidelity_ci_low": low,
            "fidelity_ci_high": high,
            "random_fidelity_mean": float(per_flow["random"].mean()),
            "bottom_fidelity_mean": float(per_flow["bottom"].mean()),
            "global_fidelity_mean": float(per_flow["global"].mean()) if "global" in per_flow else np.nan,
            "fidelity_lift_vs_random": mean - float(per_flow["random"].mean()),
            "flip_rate": float(top_flips),
            "stability_cosine_mean": float(stab["cosine_similarity"].mean()) if len(stab) else np.nan,
            "stability_jaccard_mean": float(stab["top5_jaccard"].mean()) if len(stab) else np.nan,
            "self_stability_cosine_mean": float(np.mean(lime_self)) if lime_self else np.nan,
            "latency_median_ms": float(np.median(lat)),
            "latency_p90_ms": float(np.percentile(lat, 90)),
            "latency_p95_ms": float(np.percentile(lat, 95)),
            "latency_p99_ms": float(np.percentile(lat, 99)),
            "throughput_per_second": float(1000.0 / max(np.mean(lat), 1e-12)),
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 5: Rewire `run()`**

Keep every seed's fitted model in `seed_models: dict[int, Any]` inside the seed loop. After training, choose the flows once from the seed-42 model:

```python
                indices = explanation_indices(
                    y_test, frozen_prediction,
                    min(int(config["evaluation"]["explanation_samples"]), len(y_test)), seed,
                )
                for model_seed in config["evaluation"].get("explanation_seeds", [seed]):
                    for explainer_name in ("TreeSHAP", "LIME"):
                        attrs, fidelity, stability, latency, self_values = evaluate_explainer(
                            dataset, task, model_name, explainer_name,
                            seed_models[int(model_seed)], int(model_seed), indices,
                            x_train, x_test, y_test, preprocessor, classes,
                            config["evaluation"], seed, global_ranking,
                            measure_self_stability=(int(model_seed) == seed),
                        )
                        if int(model_seed) == seed:
                            attribution_rows.extend(attrs)
                        fidelity_records.extend(fidelity)
                        stability_records.extend(stability)
                        latency_records.extend(latency)
                        if self_values:
                            self_stability[(dataset, task, model_name, explainer_name)] = self_values
```

Write `attributions.csv.gz` and `fidelity.csv.gz`, plus `latency.csv`, and build `explanation_summary.csv` from `summarize_explanations(...)`. `paired_bootstrap_comparisons` needs no change: it already averages each flow over all its rows (k values and now seeds) before differencing. Delete the old `attributions.csv` and `fidelity.csv` from `outputs/` in the same commit (`git rm`), and update `make_summary_plots` to read `attributions.csv.gz`.

Update `config.yaml` → `evaluation`:

```yaml
  explanation_samples: 200
  explanation_seeds: [42, 52, 62, 72, 82]
```

- [ ] **Step 6: Update readers**

- `extended_audit._summarize_existing_outputs`: read `fidelity.csv.gz`, average per flow over `model_seed`, and keep the existing `explanation_id` grouping by using `source_test_index` instead (`explanation_id` is per call).
- `verify_outputs.py`: `required` lists `fidelity.csv.gz`, `attributions.csv.gz` and `latency.csv`. Add:

```python
expected_seeds = set(config["evaluation"]["explanation_seeds"])
for key, group in fidelity.groupby(["dataset", "task", "model", "explainer"]):
    assert set(group["model_seed"]) == expected_seeds, key
```

- `PROJECT_WALKTHROUGH.ipynb`: replace `"fidelity.csv"` with `"fidelity.csv.gz"` and `"attributions.csv"` with `"attributions.csv.gz"` in its code cells, using a JSON-aware edit with NotebookEdit or `json` in Python, never sed on the raw file.

- [ ] **Step 7: Run tests and the smoke config**

Run: `.venv/bin/python -B -m unittest discover -s tests -v`. Expected: all tests OK.
Run the smoke config with `explanation_seeds: [42, 52]`. Expected: `explanation_summary.csv` has 16 rows with `n_model_seeds == 2`.

- [ ] **Step 8: Commit**

```bash
git add -A src/xai_ids tests verify_outputs.py config.yaml PROJECT_WALKTHROUGH.ipynb
git commit -m "Explain 200 fixed flows on all five model seeds"
```

---

### Task 7: Scale and extend the robustness audits

**Files:**
- Modify: `src/xai_ids/extended_audit.py`: `_masking_baseline_rows` (n from config), `_lime_budget_rows` (n from config), new `_recalibration_rows`, new `_shap_background_rows`, `run()`
- Modify: `config.yaml` (`evaluation.audit_samples`, `evaluation.lime_budget_samples`)
- Modify: `verify_outputs.py` (new extended files)
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `split_validation` (Task 2), `make_model(name, params, ...)` and `manifest["selected_hyperparameters"]` (Task 3).
- Produces:
  - `expected_calibration_error(probability: np.ndarray, target: np.ndarray, bins: int = 10) -> float` in `extended_audit.py`; `_calibration_rows` reuses it.
  - New outputs `extended/recalibration.csv` (`dataset, model, ece_raw, ece_isotonic, brier_raw, brier_isotonic`) and `extended/shap_background_ablation.csv` (`dataset, task, model, source_test_index, shap_mode, deletion_score`).

- [ ] **Step 1: Write the failing test**

```python
    def test_expected_calibration_error_is_zero_for_perfect_bins(self):
        from src.xai_ids.extended_audit import expected_calibration_error
        probability = np.array([0.2] * 5 + [0.8] * 5)
        target = np.array([1, 0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=float)
        self.assertAlmostEqual(expected_calibration_error(probability, target), 0.0)
        self.assertAlmostEqual(expected_calibration_error(np.ones(4), np.zeros(4)), 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Expected: ImportError for `expected_calibration_error`.

- [ ] **Step 3: Implement**

```python
def expected_calibration_error(probability: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    assigned = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    gap = 0.0
    for bin_index in range(bins):
        mask = assigned == bin_index
        if np.any(mask):
            gap += mask.sum() * abs(probability[mask].mean() - target[mask].mean())
    return float(gap / len(target))
```

`_calibration_rows` computes its ECE through this function and keeps its bin rows.

`_recalibration_rows(dataset, model_name, params, x_train, y_train, x_test, y_test, classes, seed, validation_size)` should:
1. split training data with `split_validation`;
2. fit `make_model(model_name, params, seed, 2)` on the fit part (balanced sample weights for XGBoost, as in `run()`);
3. fit `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")` on validation attack probabilities;
4. report raw and isotonic ECE and Brier score on test.

Calibration is fitted on validation only.

`_shap_background_rows(...)` should run XGBoost models only, because interventional TreeSHAP on 120 deep RF trees is too slow. On the audit flows, compute TreeSHAP with the default `tree_path_dependent` mode and with `shap.TreeExplainer(model, data=x_train[stratified_indices(y_train, 100, seed)], feature_perturbation="interventional")`, and score both with `_ranking_score` against the median/mode baseline at k = {1,3,5,10}.

Replace the hard-coded `n=12` and `n=6` with `config["evaluation"]["audit_samples"]` and `config["evaluation"]["lime_budget_samples"]`. Add to `config.yaml` → `evaluation`:

```yaml
  audit_samples: 100
  lime_budget_samples: 20
```

Add `"Probability recalibration"` and `"SHAP background ablation"` rows to the `literature_purpose_coverage` table. Add `recalibration.csv` and `shap_background_ablation.csv` to `extended_required` in `verify_outputs.py`, plus:

```python
recal = pd.read_csv(outputs / "extended" / "recalibration.csv")
assert recal[["ece_raw", "ece_isotonic"]].stack().between(0, 1).all()
```

- [ ] **Step 3b: Data-randomization sanity check and label-conflict audit**

Tests first:

```python
    def test_label_conflict_audit_counts_conflicting_feature_groups(self):
        from src.xai_ids.extended_audit import label_conflict_rows
        frame = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"],
                              "target": ["DoS", "Exploits", "Normal", "Normal"]})
        rows, ceiling = label_conflict_rows(frame, ["a", "b"], "target")
        by_class = {r["class"]: r["conflicting_share"] for r in rows}
        self.assertEqual(by_class["DoS"], 1.0)
        self.assertEqual(by_class["Normal"], 0.0)
        self.assertAlmostEqual(ceiling, 0.75)
```

Implementation in `extended_audit.py`:

```python
def label_conflict_rows(frame: pd.DataFrame, feature_columns: list[str], target: str):
    """Share of each class whose exact feature vector also carries another label.

    Such rows are not separable by any model using these features, so the
    majority label per identical-feature group bounds achievable accuracy.
    """
    keys = pd.util.hash_pandas_object(frame[feature_columns], index=False).to_numpy()
    groups = frame.assign(_key=keys).groupby("_key")[target]
    conflicting = groups.transform("nunique").to_numpy() > 1
    rows = [
        {"class": str(label), "rows": int(mask.sum()), "conflicting_share": float(conflicting[mask].mean())}
        for label in sorted(frame[target].unique())
        for mask in [frame[target].to_numpy() == label]
    ]
    ceiling = float(groups.agg(lambda s: s.value_counts().iloc[0]).sum() / len(frame))
    return rows, ceiling
```

In `run()`, for each dataset/task, call it on `train` and `test` (feature columns = `metadata["feature_names"]`, target = the label column for binary or the multiclass column/raw labels for multiclass; for CIC use `prepare_dataset`'s label column after binary mapping) and write `extended/label_conflicts.csv` with columns `dataset, task, partition, class, rows, conflicting_share, accuracy_ceiling`.

Data-randomization sanity check, `_randomization_rows(dataset, task, model_name, params, x_train, y_train, x_test, indices, seed)`:
1. Fit `make_model(model_name, params, seed, n_classes)` on `x_train` with `rng.permutation(y_train)` (seeded `np.random.default_rng(seed)`).
2. On the first `audit_samples` explanation flows, compute TreeSHAP vectors for the real seed-42 model and the label-permuted model, each for its own predicted class.
3. Emit per-flow `cosine_real_vs_permuted` (`safe_cosine`) and `top5_jaccard_real_vs_permuted`.

A sound explainer gives low similarity. Run it for TreeSHAP and for LIME (LIME at `lime_samples`, explainer seed `seed`) and write `extended/randomization_sanity.csv` with an `explainer` column.

Add both files to `extended_required` in `verify_outputs.py`, and add rows `"Explanation sanity check (data randomization)"` and `"Label-conflict separability audit"` to the coverage table.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -B -m unittest discover -s tests -v`. Expected: all OK.

- [ ] **Step 5: Smoke-run the audit on the smoke outputs**

Run: `.venv/bin/python -B run_extended_audit.py --config <scratch>/smoke.yaml`
Expected: finishes; both new CSVs exist and are non-empty.

- [ ] **Step 6: Commit**

```bash
git add src/xai_ids/extended_audit.py tests config.yaml verify_outputs.py
git commit -m "Scale robustness audits and add recalibration and SHAP background ablations"
```

---

### Task 8: Full rerun, results-driven paper and docs

**Files:**
- Regenerate: `outputs/**`, `output/pdf/xai_ids_ieee_paper.pdf`
- Modify: `src/xai_ids/paper.py`, `src/xai_ids/paper_figures.py`, `README.md`, `PUBLICATION_READINESS_REPORT.md`, `LITERATURE_REQUIREMENTS_TRACEABILITY.md`, `PROJECT_HANDBOOK.md`

**Interfaces:**
- Consumes: every output from Tasks 2–7.

- [ ] **Step 1: Run the full pipeline**

```bash
.venv/bin/python -B run_benchmark.py --config config.yaml
.venv/bin/python -B run_extended_audit.py --config config.yaml
```

Expected: both finish. Record wall-clock times for the reproducibility section.

- [ ] **Step 2: Replace hard-coded literals in `paper.py` with computed values**

Each of these is currently typed as a literal and must be derived from the outputs:
- "Seed standard deviations are at most 0.0036": use `predictive.groupby([...]).macro_f1.std().max()`.
- "15,103 sampled duplicates … 124,513 rows" and "2,830,743 flows": use `manifest["datasets"]["CIC-IDS2017/binary"]` fields (add the raw flow count to the manifest in `load_cic` if needed).
- Table II (RF and XGBoost settings): build it from `manifest["selected_hyperparameters"]` and show the tuning grid size.
- "On 12 flows per configuration" and "On six flows per configuration and five repeats": use `evaluation["audit_samples"]` and `evaluation["lime_budget_samples"]` / `lime_repeat_seeds`.
- "the seed-42 model explains the same 40 …": use `explanation_count` and `len(evaluation["explanation_seeds"])` model seeds.
- `ablation_table` and the ablation paragraph: show the validation-chosen k and its test macro-F1, and delete the sentence admitting test-set selection from Threats → Statistical Validity.
- The control analysis paragraph and Table V/VI: add the global-control pass counts (`reference == "global"`).
- The "Run used Python …" sentence: also print `manifest["environment"]["platform"]` instead of the literal "Windows 11 with … AMD64".

- [ ] **Step 3: Rewrite result-dependent prose from the new numbers**

Read `outputs/paired_comparisons.csv`, `fidelity_control_tests.csv`, `explanation_summary.csv`, `predictive_metrics.csv`, `feature_ablation.csv` and `extended/*.csv`. Then rewrite each claim in the abstract, §IV, §V-A (RQ answers), §VI and §IX so it states what the new numbers show. This includes the case where TreeSHAP no longer beats LIME everywhere or local rankings fail to beat the global control. Report negative results plainly; do not drop them.

- [ ] **Step 4: Rebuild and verify**

```bash
.venv/bin/python -B build_paper.py
.venv/bin/python -B verify_outputs.py
.venv/bin/python -B -m unittest discover -s tests -v
```

Expected: `All output, manifest, and paper checks passed.` and all tests OK. Inspect each PDF page visually (render with `pdftoppm -r 60`) for overflowing tables or orphaned headings.

- [ ] **Step 5: Update the documentation**

Update the headline results tables and claims in `README.md`, `PUBLICATION_READINESS_REPORT.md` and `LITERATURE_REQUIREMENTS_TRACEABILITY.md` to the new numbers. Mark "SHAP background ablation", "Probability recalibration" and "Validation-based feature selection" complete with their output files. In `PROJECT_HANDBOOK.md`, update the sections describing the sample size (40 → 200 × 5 seeds), the fixed hyperparameters, and the feature ablation.

- [ ] **Step 6: Commit**

```bash
git add -A outputs output src README.md PUBLICATION_READINESS_REPORT.md LITERATURE_REQUIREMENTS_TRACEABILITY.md PROJECT_HANDBOOK.md
git commit -m "Regenerate results and paper with validation-based selection and stronger explanation evidence"
```
