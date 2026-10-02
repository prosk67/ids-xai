import unittest

import numpy as np
import pandas as pd

from src.xai_ids.benchmark import (
    TabularPreprocessor,
    clean_unsw_partitions,
    predictive_metrics,
    bootstrap_mean_ci,
    candidate_params,
    choose_feature_count,
    evaluate_explainer,
    faithfulness_correlation,
    fidelity_control_tests,
    fidelity_rows,
    nearest_neighbor_stability,
    safe_cosine,
    select_hyperparameters,
    split_validation,
    summarize_explanations,
)
from src.xai_ids.transparency import evidence_decision


class BenchmarkUnitTests(unittest.TestCase):
    def test_unsw_cleaning_removes_duplicates_and_cross_partition_vectors(self):
        train = pd.DataFrame(
            {
                "feature_a": [1, 1, 2],
                "feature_b": ["x", "x", "y"],
                "label": [1, 1, 0],
                "attack_cat": ["Attack", "Attack", "Normal"],
            }
        )
        test = pd.DataFrame(
            {
                "feature_a": [1, 1, 3, 3],
                "feature_b": ["x", "x", "z", "z"],
                "label": [0, 0, 1, 1],
                "attack_cat": ["Normal", "Normal", "Attack", "Attack"],
            }
        )
        clean_train, clean_test, audit = clean_unsw_partitions(
            train, test, "label", "attack_cat"
        )
        self.assertEqual(len(clean_train), 2)
        self.assertEqual(len(clean_test), 1)
        self.assertEqual(audit["training_exact_duplicates_removed"], 1)
        self.assertEqual(audit["testing_exact_duplicates_removed"], 2)
        self.assertEqual(audit["cross_partition_feature_overlaps_removed_from_test"], 1)
        self.assertEqual(clean_test.iloc[0]["feature_a"], 3)

    def test_binary_security_metrics_use_named_attack_class(self):
        y_true = np.array([0, 0, 1, 1])  # classes are [Attack, Normal]
        y_pred = np.array([0, 1, 0, 1])
        probabilities = np.array([[0.9, 0.1], [0.4, 0.6], [0.7, 0.3], [0.1, 0.9]])
        metrics, _, _ = predictive_metrics(
            y_true, y_pred, probabilities, ["Attack", "Normal"]
        )
        self.assertAlmostEqual(metrics["false_positive_rate"], 0.5)
        self.assertAlmostEqual(metrics["false_negative_rate"], 0.5)
        self.assertGreater(metrics["pr_auc"], 0.8)

    def test_transparency_evidence_decision_is_conservative(self):
        self.assertEqual(
            evidence_decision(True, 0.2, 0.04)[0], "SUPPORTED_FOR_THIS_ALERT"
        )
        self.assertEqual(evidence_decision(True, 0.2, 0.06)[0], "REVIEW_REQUIRED")
        self.assertEqual(evidence_decision(False, 0.2, 0.01)[0], "UNSUPPORTED")

    def test_training_only_preprocessor_handles_unseen_category(self):
        train = pd.DataFrame({"numeric": [1.0, np.nan, 3.0], "category": ["a", "b", "a"]})
        test = pd.DataFrame({"numeric": [np.inf], "category": ["unseen"]})
        prep = TabularPreprocessor(["numeric", "category"]).fit(train)
        transformed = prep.transform(test)
        self.assertEqual(transformed.shape, (1, 2))
        self.assertEqual(transformed[0, 0], 2.0)
        self.assertEqual(transformed[0, 1], -1.0)

    def test_cosine_and_neighbor_stability_are_bounded(self):
        self.assertEqual(safe_cosine(np.zeros(2), np.zeros(2)), 1.0)
        x = np.array([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [3.1, 3.0]])
        predicted = np.array([0, 0, 1, 1])
        attributions = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        rows = nearest_neighbor_stability(x, predicted, attributions)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(-1.0 <= row["cosine_similarity"] <= 1.0 for row in rows))
        self.assertTrue(all(0.0 <= row["top5_jaccard"] <= 1.0 for row in rows))

    def test_bootstrap_interval_contains_constant_mean(self):
        mean, low, high = bootstrap_mean_ci(np.ones(12) * 0.25, seed=42, repetitions=100)
        self.assertEqual(mean, 0.25)
        self.assertEqual(low, 0.25)
        self.assertEqual(high, 0.25)

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


    def test_hyperparameter_selection_handles_grid_keys_absent_from_base(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(200, 3))
        y = (x[:, 0] > 0).astype(int)
        base = {"n_estimators": 10, "max_depth": 3}
        chosen, rows = select_hyperparameters(
            "random_forest", base, {"max_features": ["sqrt", None]},
            x[:150], y[:150], x[150:], y[150:], 42, 2,
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(row["selected"] for row in rows), 1)


    def test_feature_count_choice_uses_smallest_noninferior_k(self):
        scores = {5: 0.90, 10: 0.948, 20: 0.951, 34: 0.952}
        self.assertEqual(choose_feature_count(scores, 34), 10)
        self.assertEqual(choose_feature_count({5: 0.5, 34: 0.9}, 34), 34)


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
        random_rank = np.random.default_rng(7).permutation(6)
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
            for control, drop in (
                ("top", 0.5 + idx / 100), ("random", 0.1), ("global", 0.4), ("bottom", 0.0)
            ):
                records.append(
                    {"dataset": "D", "task": "binary", "model": "m", "explainer": "TreeSHAP",
                     "source_test_index": idx, "control": control, "probability_drop": drop}
                )
        result = fidelity_control_tests(pd.DataFrame(records), seed=0)
        self.assertEqual(set(result["reference"]), {"random", "global"})
        self.assertTrue((result["ci_low"] > 0).all())


    def test_faithfulness_correlation_is_high_for_true_linear_weights(self):
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(4)
        x = rng.normal(size=(500, 6))
        y = (x @ np.array([3.0, -2.0, 1.0, 0.5, 0.0, 0.0]) > 0).astype(int)
        model = LogisticRegression().fit(x, y)
        row = x[0]
        predicted = int(model.predict(row.reshape(1, -1))[0])
        sign = 1 if predicted == 1 else -1
        attribution = sign * model.coef_[0] * row
        good = faithfulness_correlation(
            model, row, attribution, predicted, np.zeros(6), 3, 60, np.random.default_rng(0)
        )
        bad = faithfulness_correlation(
            model, row, -attribution, predicted, np.zeros(6), 3, 60, np.random.default_rng(0)
        )
        self.assertGreater(good, 0.5)
        self.assertLess(bad, 0.0)

    def test_explanation_flows_shared_across_seeds_and_explainers(self):
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.default_rng(3)
        x = rng.normal(size=(400, 5))
        y = (x[:, 0] > 0).astype(int)
        names = [f"f{i}" for i in range(5)]
        pre = TabularPreprocessor(names).fit(pd.DataFrame(x, columns=names))
        indices = np.arange(300, 320)
        config = {"lime_samples": 50, "fidelity_k": [1, 3], "self_stability_samples": 2,
                  "lime_repeat_seeds": 2, "faithfulness_subsets": 10}
        seen = {}
        fidelity_all, stability_all, latency_all = [], [], []
        for model_seed in (42, 52):
            model = RandomForestClassifier(n_estimators=10, random_state=model_seed).fit(x[:300], y[:300])
            for name in ("TreeSHAP", "LIME"):
                _, fidelity, stability, latency, _ = evaluate_explainer(
                    "D", "binary", "random_forest", name, model, model_seed, indices,
                    x[:300], x, y, pre, ["Attack", "Normal"], config, 42,
                    np.arange(5), measure_self_stability=False,
                )
                seen[(model_seed, name)] = {r["source_test_index"] for r in fidelity}
                self.assertEqual({r["model_seed"] for r in latency}, {model_seed})
                fidelity_all += fidelity
                stability_all += stability
                latency_all += latency
        self.assertEqual(len({frozenset(v) for v in seen.values()}), 1)
        self.assertEqual(seen[(42, "TreeSHAP")], set(indices.tolist()))
        summary = summarize_explanations(
            pd.DataFrame(fidelity_all), pd.DataFrame(stability_all),
            pd.DataFrame(latency_all), {}, seed=42,
        )
        self.assertEqual(len(summary), 2)
        self.assertTrue((summary["n_explanations"] == 20).all())
        self.assertTrue((summary["n_model_seeds"] == 2).all())
        self.assertTrue(summary["faithfulness_correlation_mean"].between(-1, 1).all())


class TransparencyContextTests(unittest.TestCase):
    def test_benchmark_context_uses_random_reference(self):
        import tempfile
        from pathlib import Path

        from src.xai_ids.transparency import _benchmark_context

        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            base = {"dataset": "D", "task": "binary", "model": "m", "explainer": "TreeSHAP"}
            pd.DataFrame(
                [
                    {**base, "reference": "global", "mean_lift_top_minus_reference": -0.1,
                     "ci_low": -0.2, "ci_high": 0.0, "holm_adjusted_p": 0.9},
                    {**base, "reference": "random", "mean_lift_top_minus_reference": 0.2,
                     "ci_low": 0.1, "ci_high": 0.3, "holm_adjusted_p": 0.001},
                ]
            ).to_csv(results / "fidelity_control_tests.csv", index=False)
            pd.DataFrame(
                [{**base, "stability_cosine_mean": 0.9, "stability_jaccard_mean": 0.7,
                  "latency_median_ms": 1.0, "latency_p95_ms": 2.0, "n_explanations": 200}]
            ).to_csv(results / "explanation_summary.csv", index=False)
            context = _benchmark_context(results, "D", "binary", "m")
        self.assertTrue(context["global_random_control_passed"])
        self.assertAlmostEqual(context["global_top_minus_random_lift"], 0.2)


class ExtendedAuditTests(unittest.TestCase):
    def test_expected_calibration_error_is_zero_for_perfect_bins(self):
        from src.xai_ids.extended_audit import expected_calibration_error

        probability = np.array([0.2] * 5 + [0.8] * 5)
        target = np.array([1, 0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=float)
        self.assertAlmostEqual(expected_calibration_error(probability, target), 0.0)
        self.assertAlmostEqual(expected_calibration_error(np.ones(4), np.zeros(4)), 1.0)

    def test_label_conflict_audit_counts_conflicting_feature_groups(self):
        from src.xai_ids.extended_audit import label_conflict_rows

        frame = pd.DataFrame(
            {"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"],
             "target": ["DoS", "Exploits", "Normal", "Normal"]}
        )
        rows, ceiling = label_conflict_rows(frame, ["a", "b"], "target")
        by_class = {row["class"]: row["conflicting_share"] for row in rows}
        self.assertEqual(by_class["DoS"], 1.0)
        self.assertEqual(by_class["Normal"], 0.0)
        self.assertAlmostEqual(ceiling, 0.75)

    def test_randomization_sanity_separates_real_and_permuted_models(self):
        from sklearn.ensemble import RandomForestClassifier

        from src.xai_ids.extended_audit import _randomization_rows

        rng = np.random.default_rng(5)
        x = rng.normal(size=(600, 6))
        y = (x[:, 0] - x[:, 1] > 0).astype(int)
        names = [f"f{i}" for i in range(6)]
        pre = TabularPreprocessor(names).fit(pd.DataFrame(x[:500], columns=names))
        real = RandomForestClassifier(n_estimators=30, random_state=42).fit(x[:500], y[:500])
        rows = _randomization_rows(
            "D", "binary", "random_forest",
            {"n_estimators": 30, "max_depth": None},
            real, x[:500], y[:500], x[500:], np.arange(20), pre,
            ["Attack", "Normal"], 42, lime_samples=100,
        )
        frame = pd.DataFrame(rows)
        self.assertEqual(set(frame.explainer), {"TreeSHAP", "LIME"})
        tree = frame[frame.explainer.eq("TreeSHAP")]
        self.assertLess(tree.top5_jaccard_real_vs_permuted.mean(), 0.8)


if __name__ == "__main__":
    unittest.main()
