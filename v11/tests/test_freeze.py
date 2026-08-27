from __future__ import annotations

import json
import sys

from v11.experiments.freeze_validation_winners import main


def test_freeze_selects_safe_validation_winner_and_writes_test_config(tmp_path, monkeypatch):
    validation_root = tmp_path / "validation"
    validation_root.mkdir()
    # Final BA deliberately disagrees with best BA. Winner selection must use
    # best_balanced_accuracy while the safety gate remains final-epoch recall.
    for variant, final_ba, best_ba, worst in (
        ("a0_original", 0.70, 0.50, 0.40),
        ("safe_candidate", 0.40, 0.52, 0.39),
        ("unsafe_candidate", 0.80, 0.60, 0.35),
    ):
        for seed in (42, 43, 44):
            run_dir = validation_root / variant / f"seed{seed}"
            run_dir.mkdir(parents=True)
            (run_dir / "val_result.json").write_text(
                json.dumps(
                    {
                        "dataset": "dermamnist",
                        "ratio": 0.05,
                        "variant": variant,
                        "training_seed": seed,
                        "evaluation_split": "val",
                        "test_read": False,
                        "balanced_accuracy": final_ba,
                        "best_balanced_accuracy": best_ba,
                        "best_epoch": 500,
                        "worst_class_recall": worst,
                        "class_cvar20": worst + 0.01,
                    }
                ),
                encoding="utf-8",
            )
    source = tmp_path / "validation_config.json"
    source.write_text(
        json.dumps(
            {
                "schema": "graphcov-v11/job2-config-v1",
                "training": {"size": 224, "epochs": 1000},
                "jobs": [
                    {
                        "dataset": "dermamnist",
                        "ratios": [0.05],
                        "variants": [
                            "a0_original",
                            "safe_candidate",
                            "unsafe_candidate",
                        ],
                        "seeds": [42, 43, 44],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "frozen_test.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_validation_winners.py",
            "--validation-root",
            str(validation_root),
            "--validation-config",
            str(source),
            "--selection-root",
            str(tmp_path / "selection"),
            "--output-config",
            str(output),
            "--test-output-root",
            str(tmp_path / "test_output"),
        ],
    )
    assert main() == 0
    config = json.loads(output.read_text(encoding="utf-8"))
    assert config["evaluation_split"] == "test"
    assert config["jobs"][0]["variants"] == ["a0_original", "safe_candidate"]
    manifest = json.loads(
        output.with_suffix(".freeze_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["selection_metric"] == "best_balanced_accuracy"
    assert manifest["safety_metric"] == "final_worst_class_recall"
    assert manifest["decisions"]["dermamnist"]["winner"] == "safe_candidate"
    decision = manifest["decisions"]["dermamnist"]
    assert abs(decision["winner_best_balanced_accuracy_gain"] - 0.02) < 1e-12
    assert abs(
        decision["reference_metrics"]["final_balanced_accuracy_mean"] - 0.70
    ) < 1e-12
