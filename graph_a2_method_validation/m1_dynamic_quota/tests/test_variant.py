import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from graphcov.run.method_variant import allocate_dynamic_quotas
from graphcov.run.selection import METHODS, _greedy_facility_global_sparse, select


class M1Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        dense = rng.random((24, 24), dtype=np.float32)
        dense[dense < 0.65] = 0.0
        self.kernel = sp.csr_matrix(dense)
        self.labels = np.repeat(np.arange(3), 8)
        self.importance = np.ones(24, dtype=np.float32)
        self.embeddings = rng.random((24, 5), dtype=np.float32)

    def test_registered(self):
        self.assertIn("graph_a2_m1", METHODS)

    def test_floor_one_exactly_matches_official_greedy(self):
        expected = _greedy_facility_global_sparse(
            self.kernel, self.labels, 3, self.importance
        )
        with patch("graphcov.run.method_variant.build_graph_a2_kernel", return_value=self.kernel):
            actual = METHODS["graph_a2_m1"].fn(
                self.embeddings, self.labels, 3, self.importance,
                quota_floor_ratio=1.0, global_selection=True, sparse_cpu=True,
            )
        self.assertEqual(actual, expected)

    def test_dynamic_quota_preserves_budget_and_floor(self):
        quotas, _ = allocate_dynamic_quotas(
            self.kernel, self.labels, self.importance, 3, floor_ratio=0.5
        )
        self.assertEqual(sum(quotas.values()), 9)
        self.assertTrue(all(value >= 1 for value in quotas.values()))

    def test_disabled_variant_matches_official_end_to_end(self):
        expected = select(
            "graph_a2", self.labels, 2, embeddings=self.embeddings,
            global_selection=True, sparse_cpu=True, k_neighbors=5, k_hops=2,
        )
        actual = select(
            "graph_a2_m1", self.labels, 2, embeddings=self.embeddings,
            global_selection=True, sparse_cpu=True, k_neighbors=5, k_hops=2,
            quota_floor_ratio=1.0,
        )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
