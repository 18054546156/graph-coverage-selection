from pathlib import Path
import unittest

import yaml

from reliability_medmnistc.scripts.prepare_medmnist_224 import archive_filename


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

    def test_facility_has_deterministic_memory_fallback(self):
        config = yaml.safe_load(
            (ROOT / "configs" / "full_5datasets_8methods.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["method_config"]["facility"], {
            "global_selection": False,
            "execution_device": "auto",
            "cpu_min_class_size": 40000,
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

    def test_clean_archive_names_cover_dynamics_and_downstream_sizes(self):
        self.assertEqual(archive_filename("pathmnist", 28), "pathmnist.npz")
        self.assertEqual(archive_filename("pathmnist", 224), "pathmnist_224.npz")

    def test_slurm_scripts_resolve_the_submission_directory(self):
        for path in (ROOT / "slurm").glob("*.slurm"):
            source = path.read_text(encoding="utf-8")
            self.assertIn("SLURM_SUBMIT_DIR", source, path.name)
            self.assertIn("configure_imagemagick", source, path.name)

    def test_training_tasks_exclusively_own_the_gpu_node(self):
        source = (ROOT / "slurm" / "03_train_eval_array.slurm").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --exclusive", source)

    def test_command_wrappers_do_not_shadow_standard_library_select(self):
        direct_entries = (
            "prepare_medmnist_224.py", "generate_medmnistc.py", "run_pipeline.py",
            "select.py", "validate_selections.py", "train.py", "evaluate.py",
            "summarize.py", "run_pipeline_full.py",
        )
        for filename in direct_entries:
            source = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
            self.assertIn("sys.path = [entry for entry in sys.path", source, filename)
        for filename in (
            "select.py", "validate_selections.py", "train.py", "evaluate.py",
            "summarize.py", "run_pipeline_full.py",
        ):
            source = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
            self.assertIn("from reliability_medmnistc.scripts.run_pipeline import main", source, filename)


if __name__ == "__main__":
    unittest.main()
