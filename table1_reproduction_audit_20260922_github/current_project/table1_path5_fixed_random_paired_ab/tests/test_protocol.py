from __future__ import annotations

import json
from pathlib import Path
import unittest


class ProtocolTests(unittest.TestCase):
    def test_frozen_pairing_invariants(self) -> None:
        protocol = json.loads((Path(__file__).parents[1] / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["dataset"]["name"], "pathmnist")
        self.assertEqual(protocol["dataset"]["ratio"], 0.05)
        self.assertEqual(protocol["dataset"]["budget_per_class"], 499)
        self.assertEqual(protocol["dataset"]["n_selected"], 4491)
        self.assertEqual(protocol["training_seeds"], [42, 43, 44, 45, 46])
        self.assertEqual(protocol["arm_order"], ["random_seed42", "graph_a2"])
        self.assertEqual(protocol["training"]["epochs"], 1000)
        self.assertFalse(protocol["training"]["augmentation"])
        self.assertFalse(protocol["training"]["deterministic"])
        self.assertEqual(protocol["training"]["endpoint"], "final epoch official-test balanced accuracy")


if __name__ == "__main__":
    unittest.main()
