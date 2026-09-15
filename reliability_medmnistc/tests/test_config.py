from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ConfigContractTests(unittest.TestCase):
    def test_configs_have_frozen_protocol_fields(self):
        for path in (ROOT / "configs").glob("*.yaml"):
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(config["selection_seed"], 42)
            self.assertEqual(config["image_size"], 224)
            self.assertEqual(config["dynamics_epochs"], 200)
            self.assertEqual(config["checkpoint_rule"], "final_epoch")
            self.assertTrue(0.01 in config["ratios"] or 0.02 in config["ratios"])

    def test_graph_a2_parameters_are_explicit(self):
        config = yaml.safe_load(
            (ROOT / "configs" / "full_5datasets_8methods.yaml").read_text(encoding="utf-8")
        )
        graph = config["method_config"]["graph_a2"]
        self.assertEqual(graph, {
            "global_selection": True,
            "k_neighbors": 50,
            "k_hops": 2,
            "kernel": "A_sym_plus_A_sym_squared",
        })

    def test_formal_config_is_single_seed_80_run_protocol(self):
        config = yaml.safe_load(
            (ROOT / "configs" / "full_5datasets_8methods.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["training_seeds"], [42])
        self.assertEqual(len(config["datasets"]), 5)
        self.assertEqual(len(config["methods"]), 8)
        self.assertEqual(config["ratios"], [0.02, 0.05])
        self.assertEqual(
            len(config["datasets"]) * len(config["methods"]) * len(config["ratios"])
            * len(config["training_seeds"]),
            80,
        )


if __name__ == "__main__":
    unittest.main()
