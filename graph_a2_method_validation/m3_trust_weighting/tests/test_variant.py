import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from graphcov.run.method_variant import compute_purity_and_support, compute_trust_weights
from graphcov.run.selection import METHODS, _greedy_facility_global_sparse, select


class M3Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        dense = rng.random((24, 24), dtype=np.float32)
        dense[dense < 0.65] = 0.0
        self.kernel = sp.csr_matrix(dense)
        self.labels = np.repeat(np.arange(3), 8)
        self.importance = np.ones(24, dtype=np.float32)
        self.embeddings = rng.random((24, 5), dtype=np.float32)

    def test_registered(self):
        self.assertIn("graph_a2_m3", METHODS)

    def test_zero_exponent_preserves_weights_and_indices(self):
        trust, _ = compute_trust_weights(
            self.kernel, self.labels, 0.0, trust_mode="adaptive"
        )
        np.testing.assert_array_equal(trust, np.ones_like(trust))
        expected = _greedy_facility_global_sparse(
            self.kernel, self.labels, 3, self.importance
        )
        with patch("graphcov.run.method_variant.build_graph_a2_kernel", return_value=self.kernel):
            actual = METHODS["graph_a2_m3"].fn(
                self.embeddings, self.labels, 3, self.importance,
                purity_exponent=0.0, global_selection=True, sparse_cpu=True,
            )
        self.assertEqual(actual, expected)

    def test_purity_and_support_are_bounded(self):
        purity, support = compute_purity_and_support(self.kernel, self.labels)
        self.assertTrue(np.all((0 <= purity) & (purity <= 1)))
        self.assertTrue(np.all((0 <= support) & (support <= 1)))

    def test_disabled_variant_matches_official_end_to_end(self):
        expected = select(
            "graph_a2", self.labels, 2, embeddings=self.embeddings,
            global_selection=True, sparse_cpu=True, k_neighbors=5, k_hops=2,
        )
        actual = select(
            "graph_a2_m3", self.labels, 2, embeddings=self.embeddings,
            global_selection=True, sparse_cpu=True, k_neighbors=5, k_hops=2,
            purity_exponent=0.0,
        )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
