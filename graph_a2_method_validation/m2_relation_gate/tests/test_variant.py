import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from graphcov.run.method_variant import apply_relation_gate, estimate_class_pair_gate
from graphcov.run.selection import METHODS, _greedy_facility_global_sparse, select


class M2Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        dense = rng.random((24, 24), dtype=np.float32)
        dense[dense < 0.65] = 0.0
        self.kernel = sp.csr_matrix(dense)
        self.labels = np.repeat(np.arange(3), 8)
        self.importance = np.ones(24, dtype=np.float32)
        self.embeddings = rng.random((24, 5), dtype=np.float32)

    def test_registered(self):
        self.assertIn("graph_a2_m2", METHODS)

    def test_baseline_mode_preserves_kernel_and_indices(self):
        gated, gate = apply_relation_gate(self.kernel, self.labels, "baseline")
        np.testing.assert_array_equal(gated.toarray(), self.kernel.toarray())
        np.testing.assert_array_equal(gate, np.ones_like(gate))
        expected = _greedy_facility_global_sparse(
            self.kernel, self.labels, 3, self.importance
        )
        with patch("graphcov.run.method_variant.build_graph_a2_kernel", return_value=self.kernel):
            actual = METHODS["graph_a2_m2"].fn(
                self.embeddings, self.labels, 3, self.importance,
                relation_mode="baseline", global_selection=True, sparse_cpu=True,
            )
        self.assertEqual(actual, expected)

    def test_pair_gate_diagonal_and_disabled_strength(self):
        gate, _ = estimate_class_pair_gate(self.kernel, self.labels, strength=1.0)
        np.testing.assert_allclose(np.diag(gate), 1.0)
        disabled, _ = estimate_class_pair_gate(self.kernel, self.labels, strength=0.0)
        np.testing.assert_allclose(disabled, 1.0)

    def test_zero_scalar_is_same_endpoint(self):
        gated, _ = apply_relation_gate(
            self.kernel, self.labels, "scalar", cross_credit=0.0
        )
        coo = gated.tocoo()
        self.assertTrue(np.all(self.labels[coo.row] == self.labels[coo.col]))

    def test_disabled_variant_matches_official_end_to_end(self):
        expected = select(
            "graph_a2", self.labels, 2, embeddings=self.embeddings,
            global_selection=True, sparse_cpu=True, k_neighbors=5, k_hops=2,
        )
        actual = select(
            "graph_a2_m2", self.labels, 2, embeddings=self.embeddings,
            global_selection=True, sparse_cpu=True, k_neighbors=5, k_hops=2,
            relation_mode="baseline",
        )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
