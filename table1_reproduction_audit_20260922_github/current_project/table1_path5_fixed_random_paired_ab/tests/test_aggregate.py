from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from aggregate import summarize


class AggregateTests(unittest.TestCase):
    def test_paired_sign_controls_frozen_decision(self) -> None:
        protocol = {
            "training_seeds": [42, 43, 44, 45, 46],
            "arm_order": ["random_seed42", "graph_a2"],
            "training": {"endpoint": "final BA"},
            "paper_path5_pct": {},
            "decision": {
                "interpretation_if_reproduced": "restored",
                "interpretation_if_not_reproduced": "not restored",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in protocol["training_seeds"]:
                for arm, ba in (("random_seed42", 0.80), ("graph_a2", 0.81)):
                    path = root / f"seed{seed}" / arm / "metrics.json"
                    path.parent.mkdir(parents=True)
                    path.write_text(
                        json.dumps(
                            {
                                "complete": True,
                                "training_seed": seed,
                                "balanced_accuracy": ba,
                                "canonical_index_sha256": arm,
                            }
                        ),
                        encoding="utf-8",
                    )
            result = summarize(protocol, root)
        self.assertTrue(result["decision"]["paper_ranking_reproduced"])
        self.assertAlmostEqual(
            result["aggregate"]["mean_paired_delta_graph_a2_minus_random_pp"], 1.0
        )
        self.assertEqual(result["aggregate"]["graph_a2_wins"], 5)


if __name__ == "__main__":
    unittest.main()
