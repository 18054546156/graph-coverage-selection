from __future__ import annotations

import unittest

import numpy as np

from paired_core import canonicalize_indices, index_sha256, validate_equal_quota


class PairedCoreTests(unittest.TestCase):
    def test_canonicalization_sorts_and_hashes_stably(self) -> None:
        canonical = canonicalize_indices(np.array([5, 1, 3]), n_train=6, n_selected=3)
        np.testing.assert_array_equal(canonical, np.array([1, 3, 5]))
        self.assertEqual(index_sha256(canonical), index_sha256(np.array([1, 3, 5], dtype=np.int32)))

    def test_canonicalization_rejects_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates"):
            canonicalize_indices(np.array([1, 1]), n_train=3, n_selected=2)

    def test_equal_quota(self) -> None:
        labels = np.array([0, 0, 1, 1, 2, 2])
        counts = validate_equal_quota(labels, np.array([0, 2, 4]), n_classes=3, budget_per_class=1)
        self.assertEqual(counts, {"0": 1, "1": 1, "2": 1})


if __name__ == "__main__":
    unittest.main()
