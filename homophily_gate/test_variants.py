#!/usr/bin/env python3
"""三个快速变体的单元测试（不需GPU，快速验证逻辑）。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from graphcov.run.selection import METHODS
from homophily_gate.graph_a2_variants import (
    _compute_tiered_homophily,
    _compute_multiscale_homophily,
    _compute_local_homophily_soft,
)

class VariantsTests(unittest.TestCase):

    def test_tiered_homophily_classification(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        knn_indices = np.array([
            [1, 2, 3, 4, 5],
            [0, 2, 3, 4, 5],
            [0, 1, 3, 4, 5],
            [0, 1, 2, 4, 5],
            [0, 1, 2, 3, 5],
            [0, 1, 2, 3, 4],
        ])
        tier_damping = _compute_tiered_homophily(labels, knn_indices)
        np.testing.assert_allclose(tier_damping, [0.5] * 6)

    def test_multiscale_registered(self):
        assert "graph_a2_tiered_homophily" in METHODS
        assert "graph_a2_multiscale_homophily" in METHODS
        assert "graph_a2_soft_damping_homophily" in METHODS

if __name__ == "__main__":
    unittest.main()
