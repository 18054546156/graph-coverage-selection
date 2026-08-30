#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.sparse as sp

from graphcov.run.selection import (
    METHODS,
    _greedy_facility_global_sparse,
)
from graphcov.run.selection_opt import (
    _allocate_graph_marginal_quota,
    _allocate_quota_sqrt,
    _greedy_variable_quota,
    _rescale_cross_class,
)


VARIANTS = {
    "graph_a2_dec",
    "graph_a2_damp25",
    "graph_a2_damp50",
    "graph_a2_damp75",
    "graph_a2_sqrt",
    "graph_a2_dec_sqrt",
    "graph_a2_marginal",
    "graph_a2_dec_marginal",
}


class SelectionOptimizationTests(unittest.TestCase):
    def test_methods_are_registered(self):
        self.assertTrue(VARIANTS.issubset(METHODS))

    def test_sqrt_quota_preserves_feasible_budget(self):
        labels = np.array([0] * 100 + [1] * 25 + [2] * 16 + [3] * 9)
        quotas = _allocate_quota_sqrt(labels, budget_per_class=5)
        self.assertEqual(sum(quotas.values()), 20)
        self.assertEqual(set(quotas), {0, 1, 2, 3})
        for cls, count in zip(*np.unique(labels, return_counts=True)):
            self.assertGreaterEqual(quotas[int(cls)], 1)
            self.assertLessEqual(quotas[int(cls)], int(count))

    def test_sqrt_quota_rejects_infeasible_budget(self):
        labels = np.array([0] * 1000 + [1] * 10)
        with self.assertRaises(ValueError):
            _allocate_quota_sqrt(labels, budget_per_class=600)

    def test_cross_class_rescaling(self):
        kernel = sp.csr_matrix(np.ones((4, 4), dtype=np.float32))
        labels = np.array([0, 0, 1, 1])
        hard = _rescale_cross_class(kernel, labels, beta=0.0).toarray()
        soft = _rescale_cross_class(kernel, labels, beta=0.25).toarray()
        same = labels[:, None] == labels[None, :]
        np.testing.assert_allclose(hard[same], 1.0)
        np.testing.assert_allclose(hard[~same], 0.0)
        np.testing.assert_allclose(soft[same], 1.0)
        np.testing.assert_allclose(soft[~same], 0.25)

    def test_equal_quota_matches_official_sparse_greedy(self):
        rng = np.random.default_rng(42)
        dense = rng.random((30, 30), dtype=np.float32)
        dense[dense < 0.75] = 0.0
        kernel = sp.csr_matrix(dense)
        labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
        importance = rng.random(30, dtype=np.float32) + 0.1
        official = _greedy_facility_global_sparse(
            kernel, labels, 3, importance
        )
        variable = _greedy_variable_quota(
            kernel, labels, {0: 3, 1: 3, 2: 3}, importance
        )
        self.assertEqual(variable, official)

    def test_graph_marginal_quota_is_dynamic_and_budget_exact(self):
        class_zero = np.eye(4, dtype=np.float32)
        class_one = np.ones((4, 4), dtype=np.float32)
        kernel = sp.block_diag((class_zero, class_one), format="csr")
        labels = np.array([0] * 4 + [1] * 4)
        quotas, curves = _allocate_graph_marginal_quota(
            kernel,
            labels,
            np.ones(8, dtype=np.float32),
            budget_per_class=3,
        )
        self.assertEqual(sum(quotas.values()), 6)
        self.assertEqual(quotas, {0: 4, 1: 2})
        self.assertEqual(set(curves), {0, 1})

    def test_all_registered_variants_select_the_fixed_total_budget(self):
        rng = np.random.default_rng(7)
        dense = rng.random((12, 12), dtype=np.float32)
        dense[dense < 0.5] = 0.0
        kernel = sp.csr_matrix(dense)
        embeddings = rng.random((12, 4), dtype=np.float32)
        labels = np.array([0] * 7 + [1] * 5)
        importance = np.ones(12, dtype=np.float32)

        with patch(
            "graphcov.run.selection_opt._build_global_kernel",
            return_value=kernel,
        ):
            for method in sorted(VARIANTS):
                selected = METHODS[method].fn(
                    embeddings=embeddings,
                    labels=labels,
                    budget_per_class=2,
                    importance=importance,
                    global_selection=True,
                    sparse_cpu=True,
                )
                self.assertEqual(len(selected), 4, method)
                self.assertEqual(len(set(selected)), 4, method)


if __name__ == "__main__":
    unittest.main()
