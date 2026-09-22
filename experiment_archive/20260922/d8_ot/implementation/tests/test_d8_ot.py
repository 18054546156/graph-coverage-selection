import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from d8_ot import select_equal_mass_ot
from d8_registry import register_d8_method


class EqualMassOTTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.labels = np.repeat(np.array([0, 1, 2]), 18)
        centers = np.repeat(np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]), 18, axis=0)
        self.embeddings = centers + rng.normal(0, 0.15, size=(len(self.labels), 2))

    def test_per_class_budget_and_unique_rows(self):
        selected, diagnostics = select_equal_mass_ot(
            self.embeddings, self.labels, budget_per_class=4, seed=3,
            outer_iterations=2, sinkhorn_iterations=30,
        )
        self.assertEqual(len(selected), 12)
        self.assertEqual(len(np.unique(selected)), 12)
        self.assertEqual(set(np.bincount(self.labels[selected])), {4})
        self.assertEqual(len(diagnostics), 3)
        self.assertTrue(all(np.isfinite(d["entropic_transport_cost"]) for d in diagnostics))

    def test_same_seed_is_reproducible(self):
        kwargs = dict(budget_per_class=3, seed=19, outer_iterations=2, sinkhorn_iterations=30)
        first, _ = select_equal_mass_ot(self.embeddings, self.labels, **kwargs)
        second, _ = select_equal_mass_ot(self.embeddings, self.labels, **kwargs)
        np.testing.assert_array_equal(first, second)

    def test_nonfinite_embeddings_are_rejected(self):
        bad = self.embeddings.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            select_equal_mass_ot(bad, self.labels, budget_per_class=2, seed=0)

    def test_oversized_quota_selects_whole_class(self):
        selected, _ = select_equal_mass_ot(
            self.embeddings, self.labels, budget_per_class=30, seed=0,
            outer_iterations=1, sinkhorn_iterations=10,
        )
        self.assertEqual(len(selected), len(self.labels))
        self.assertEqual(set(selected.tolist()), set(range(len(self.labels))))

    def test_cli_writes_indices_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "source_train.npz"
            output_path = root / "selected.npz"
            np.savez(input_path, embeddings=self.embeddings, labels=self.labels)
            subprocess.run([
                sys.executable,
                str(ROOT / "select_d8.py"),
                "--input", str(input_path),
                "--output", str(output_path),
                "--budget-per-class", "3",
                "--seed", "4",
                "--outer-iterations", "2",
                "--sinkhorn-iterations", "20",
            ], check=True, capture_output=True, text=True)
            with np.load(output_path, allow_pickle=False) as saved:
                selected = saved["selected_indices"]
            self.assertEqual(len(selected), 9)
            self.assertTrue(output_path.with_suffix(".json").is_file())

    def test_registry_adapter_matches_framework_contract(self):
        registry = {}

        def fake_register(name, **metadata):
            def decorate(function):
                registry[name] = (metadata, function)
                return function
            return decorate

        method = register_d8_method(fake_register)
        metadata, registered = registry["d8_ot"]
        self.assertIs(method, registered)
        self.assertEqual(metadata["needs"], {"embeddings"})
        selected = registered(
            self.embeddings,
            self.labels,
            budget_per_class=2,
            importance=np.ones(len(self.labels)),
            seed=7,
            epsilon=0.05,
            outer_iterations=1,
            sinkhorn_iterations=20,
        )
        self.assertEqual(len(selected), 6)
        self.assertEqual(len(set(selected)), 6)


if __name__ == "__main__":
    unittest.main()
