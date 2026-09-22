from __future__ import annotations

import json
from pathlib import Path
import unittest


class ProtocolTests(unittest.TestCase):
    def test_selection_seed_sweep_is_frozen(self) -> None:
        protocol = json.loads((Path(__file__).parents[1] / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["selection"]["selection_seeds"], [42, 43, 44, 45, 46])
        self.assertEqual(protocol["training"]["training_seed"], 42)
        self.assertEqual(protocol["dataset"]["n_selected"], 4491)
        self.assertEqual(protocol["training"]["epochs"], 1000)
        self.assertFalse(protocol["training"]["augmentation"])
        self.assertEqual(protocol["training"]["endpoint"], "final epoch official-test balanced accuracy")


if __name__ == "__main__":
    unittest.main()
