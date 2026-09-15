import ast
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np


PIPELINE = Path(__file__).resolve().parents[1] / "reliability" / "pipeline.py"


def load_functions(names, extra_globals=None):
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in set(names)
    ]
    namespace = {
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "np": np,
    }
    namespace.update(extra_globals or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(PIPELINE), "exec"), namespace)
    return namespace


class MetricContractTests(unittest.TestCase):
    def test_balanced_accuracy_rejects_missing_classes(self):
        functions = load_functions(
            ["log_softmax_np", "ece_top", "risk_coverage", "metric_row"],
            {"ALLOW_MISSING_CLASSES": False},
        )
        with self.assertRaisesRegex(RuntimeError, "missing required classes"):
            functions["metric_row"](np.array([0]), np.array([[1.0, 0.0]]), 2)


class SelectionContractTests(unittest.TestCase):
    def test_selection_loader_validates_hash_mapping_and_quota(self):
        with tempfile.TemporaryDirectory() as temporary:
            selection_root = Path(temporary)
            functions = load_functions(
                ["array_sha256", "file_sha256", "verify_source_file", "load_selection_artifact"],
                {
                    "SELECTION_OUT": selection_root,
                    "SELECTION_SEED": 42,
                    "EXPECTED_GRAPHCOV_COMMIT": "8cf757a",
                    "SIZE": 224,
                    "DYNAMICS_EPOCHS": 200,
                    "EMBEDDING_SOURCE": "uni",
                    "FACILITY_GLOBAL": False,
                    "GRAPH_GLOBAL": True,
                    "GRAPH_K": 50,
                    "GRAPH_HOPS": 2,
                    "VERIFIED_SOURCE_FILES": {},
                },
            )
            source_indices = np.arange(6, dtype=np.int64)
            labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
            local = np.array([0, 3], dtype=np.int64)
            original = source_indices[local]
            directory = selection_root / "random" / "toy" / "ratio_0.34" / "selection_seed_42"
            directory.mkdir(parents=True)
            np.save(directory / "selected_indices.npy", original)
            np.save(directory / "selected_local_indices.npy", local)
            (directory / "selection_seed.txt").write_text("42\n", encoding="utf-8")
            (directory / "index_order_sha256.txt").write_text(
                functions["array_sha256"](original) + "\n", encoding="utf-8"
            )
            (directory / "class_counts.json").write_text(
                json.dumps({"0": 1, "1": 1}), encoding="utf-8"
            )
            config = {
                "dataset": "toy",
                "method": "random",
                "ratio": 0.34,
                "selection_seed": 42,
                "budget_per_class": 1,
                "n_selected": 2,
                "source_train_indices_sha256": functions["array_sha256"](source_indices),
                "selected_indices_sha256": functions["array_sha256"](original),
                "graphcov_commit": "8cf757a",
                "embedding_source": "uni",
                "image_size": 224,
                "dynamic_image_size": 28,
                "dynamics_epochs": 200,
                "source_train_labels_sha256": functions["array_sha256"](labels),
                "selection_sources": {},
                "facility_global": False,
                "graph_global": None,
                "graph_k": None,
                "graph_hops": None,
            }
            (directory / "selection_config.json").write_text(json.dumps(config), encoding="utf-8")
            loaded = functions["load_selection_artifact"](
                "toy", "random", 0.34, source_indices, labels, {"label": {"0": "a", "1": "b"}}
            )
            np.testing.assert_array_equal(loaded[0], local)
            np.save(directory / "selected_indices.npy", np.array([1, 3], dtype=np.int64))
            with self.assertRaises(RuntimeError):
                functions["load_selection_artifact"](
                    "toy", "random", 0.34, source_indices, labels,
                    {"label": {"0": "a", "1": "b"}},
                )


class CompletionContractTests(unittest.TestCase):
    def test_completion_marker_binds_selection_and_all_conditions(self):
        function_names = [
            "json_default", "array_sha256", "file_sha256", "row_key", "load_run_rows",
            "validate_metric_rows", "validate_training_history", "validate_completion_marker",
        ]
        functions = load_functions(function_names, {"ALLOW_MISSING_CLASSES": False})
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            selected = np.array([2, 7], dtype=np.int64)
            np.save(run_dir / "selected_indices.npy", selected)
            np.save(run_dir / "selected_local_indices.npy", np.array([0, 1], dtype=np.int64))
            selection_hash = functions["array_sha256"](selected)
            config = {
                "dataset": "toy", "method": "random", "selection_seed": 42,
                "training_seed": 42, "selection_sha256": selection_hash,
                "selection_config_sha256": "a" * 64, "n_selected": 2,
                "num_classes": 2, "epochs": 1, "corruptions": ["blur"],
                "ratio": 0.02, "augment": 0, "budget_per_class": 1,
            }
            config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            (run_dir / "run_config.json").write_text(
                json.dumps({**config, "config_hash": config_hash}), encoding="utf-8"
            )
            (run_dir / "final.pt").write_bytes(b"checkpoint")
            (run_dir / "training_history.json").write_text(
                json.dumps([{"epoch": 1, "train_loss": 1.0, "train_accuracy": 0.5, "lr": 0.0}]),
                encoding="utf-8",
            )
            checkpoint_hash = functions["file_sha256"](run_dir / "final.pt")
            conditions = [("clean", 0)] + [("blur", severity) for severity in range(1, 6)]
            rows = []
            for corruption, severity in conditions:
                rows.append({
                    "dataset": "toy", "method": "random", "selection_seed": 42,
                    "training_seed": 42, "selection_sha256": selection_hash,
                    "checkpoint_sha256": checkpoint_hash, "corruption": corruption,
                    "severity": severity, "n_classes_observed": 2, "acc": 0.5,
                    "ba": 0.5, "worst_recall": 0.4, "nll": 1.0,
                    "brier": 0.5, "ece15": 0.1, "aurc": 0.2, "ratio": 0.02,
                    "augment": 0, "n_selected": 2, "budget_per_class": 1,
                })
            (run_dir / "metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            (run_dir / "predictions_clean.npz").write_bytes(b"clean")
            (run_dir / "predictions_blur.npz").write_bytes(b"blur")
            completion = {
                "status": "complete", "conditions": 6, "config_hash": config_hash,
                "selection_sha256": selection_hash, "selection_config_sha256": "a" * 64,
                "checkpoint_sha256": checkpoint_hash,
                "training_history_sha256": functions["file_sha256"](run_dir / "training_history.json"),
                "metrics_sha256": functions["file_sha256"](run_dir / "metrics.jsonl"),
                "prediction_sha256": {
                    "clean": functions["file_sha256"](run_dir / "predictions_clean.npz"),
                    "blur": functions["file_sha256"](run_dir / "predictions_blur.npz"),
                },
            }
            functions["validate_completion_marker"](run_dir, completion)
            np.save(run_dir / "selected_indices.npy", np.array([2, 8], dtype=np.int64))
            with self.assertRaisesRegex(RuntimeError, "selection hash mismatch"):
                functions["validate_completion_marker"](run_dir, completion)


class PhaseIsolationTests(unittest.TestCase):
    def test_selector_work_is_guarded_by_selecting_phase(self):
        source = PIPELINE.read_text(encoding="utf-8")
        self.assertIn('selecting = PHASE in {"select", "full"}', source)
        self.assertIn("if selecting:\n        train_28_full", source.replace("\r\n", "\n"))
        self.assertIn("else:\n                local_indices, original_indices", source.replace("\r\n", "\n"))


if __name__ == "__main__":
    unittest.main()
