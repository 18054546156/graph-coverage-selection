import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step1 = load_module("step1_diagnostic", "scripts/diagnose_representation_drift.py")
step2 = load_module("step2_variance", "scripts/run_training_variance.py")


class SelectionArtifactTests(unittest.TestCase):
    def make_artifact(self, root):
        directory = root / "random" / "pathmnist" / "ratio_0.5" / "selection_seed_42"
        directory.mkdir(parents=True)
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        selected = np.asarray([0, 2], dtype=np.int64)
        np.save(directory / "selected_indices.npy", selected)
        selected_hash = step2.sha256_array(selected)
        config = {
            "dataset": "pathmnist",
            "method": "random",
            "ratio": 0.5,
            "selection_seed": 42,
            "budget_per_class": 1,
            "n_selected": 2,
            "selected_indices_sha256": selected_hash,
            "source_train_indices_sha256": step2.sha256_array(np.arange(4, dtype=np.int64)),
        }
        (directory / "selection_config.json").write_text(json.dumps(config), encoding="utf-8")
        (directory / "class_counts.json").write_text(json.dumps({"0": 1, "1": 1}), encoding="utf-8")
        (directory / "index_order_sha256.txt").write_text(selected_hash + "\n", encoding="utf-8")
        return labels, directory

    def test_strict_selection_accepts_matching_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            labels, _ = self.make_artifact(Path(temporary))
            artifact = step2.inspect_selection(Path(temporary), "pathmnist", 0.5, "random", 42, labels)
            np.testing.assert_array_equal(artifact["indices"], [0, 2])

    def test_strict_selection_rejects_wrong_recorded_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            labels, directory = self.make_artifact(Path(temporary))
            (directory / "class_counts.json").write_text(json.dumps({"0": 2, "1": 0}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                step2.inspect_selection(Path(temporary), "pathmnist", 0.5, "random", 42, labels)

    def test_strict_selection_rejects_omitted_class(self):
        with tempfile.TemporaryDirectory() as temporary:
            labels, directory = self.make_artifact(Path(temporary))
            np.save(directory / "selected_indices.npy", np.asarray([0], dtype=np.int64))
            selected_hash = step2.sha256_array(np.asarray([0], dtype=np.int64))
            config_path = directory / "selection_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config.update(n_selected=1, selected_indices_sha256=selected_hash)
            config_path.write_text(json.dumps(config), encoding="utf-8")
            (directory / "index_order_sha256.txt").write_text(selected_hash + "\n", encoding="utf-8")
            (directory / "class_counts.json").write_text(json.dumps({"0": 1}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                step2.inspect_selection(Path(temporary), "pathmnist", 0.5, "random", 42, labels)


class MetricAndVarianceTests(unittest.TestCase):
    def test_formal_metric_rejects_missing_class(self):
        labels = np.asarray([0, 0, 1, 1])
        logits = np.asarray([[2, 1, 0], [2, 1, 0], [0, 2, 1], [0, 2, 1]], dtype=float)
        with self.assertRaises(RuntimeError):
            step2.metric_row(labels, logits, 3, allow_missing=False)

    def test_factorial_variance_is_reported(self):
        import pandas as pd

        rows = []
        for selection_seed, selection_hash, selection_effect in [(1, "a", 0.1), (2, "b", -0.1)]:
            for training_seed, training_effect in [(10, 0.02), (11, -0.02)]:
                rows.append({
                    "dataset": "pathmnist", "method": "random", "ratio": 0.05,
                    "split": "clean", "corruption": "clean", "severity": 0,
                    "selection_seed": selection_seed, "training_seed": training_seed,
                    "selection_sha256": selection_hash,
                    "balanced_accuracy": 0.7 + selection_effect + training_effect,
                })
        result = step2.compute_variance_components(pd.DataFrame(rows))
        ba = result[result.metric == "balanced_accuracy"].iloc[0]
        self.assertEqual(ba.status, "complete")
        self.assertGreater(ba.variance_selection, 0)
        self.assertGreater(ba.variance_training, 0)


class KnnDefinitionTests(unittest.TestCase):
    def test_reciprocal_knn_is_distinct_from_cross_space_overlap(self):
        neighbors = np.asarray([[1], [0], [1]], dtype=np.int64)
        np.testing.assert_allclose(step1.reciprocal_knn_rate(neighbors), [1.0, 1.0, 0.0])


class DifficultyCacheTests(unittest.TestCase):
    def test_exact_dynamics_cache_does_not_fall_back_to_wrong_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(
                root / "pathmnist_train_dynamics_28_e2_s1.npz",
                all_l2_scores=np.ones((2, 4)),
                forgetting_scores=np.zeros(4),
            )
            scores, source = step1.load_difficulty(
                root, "pathmnist", 4, size=28, epochs=2, seed=42
            )
            self.assertEqual(scores, {})
            self.assertEqual(source["status"], "missing")
            self.assertTrue(source["path"].endswith("pathmnist_train_dynamics_28_e2_s42.npz"))

    def test_dynamics_selection_config_must_match_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels, directory = SelectionArtifactTests().make_artifact(root)
            target = root / "el2n_top" / "pathmnist" / "ratio_0.5" / "selection_seed_42"
            target.parent.mkdir(parents=True, exist_ok=True)
            directory.rename(target)
            config_path = target / "selection_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config.update(method="el2n_top", dynamic_image_size=28, dynamics_epochs=100)
            config_path.write_text(json.dumps(config), encoding="utf-8")
            step1.load_verified_selection(root, "pathmnist", 0.5, "el2n_top", 42, labels)
            with self.assertRaises(RuntimeError):
                step1.validate_selection_dynamics_config(
                    root, "pathmnist", 0.5, "el2n_top", 42, size=28, epochs=200
                )


if __name__ == "__main__":
    unittest.main()
