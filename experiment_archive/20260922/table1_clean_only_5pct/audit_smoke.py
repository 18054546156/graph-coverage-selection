#!/usr/bin/env python3
"""Audit all 5% smoke outputs before formal submission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASETS = ("organsmnist", "organamnist", "pathmnist", "tissuemnist", "bloodmnist")
METHODS = ("random", "el2n_top", "forgetting", "eva", "facility", "fps", "herding", "graph_a2")
RATIO = 0.05
SMOKE_N = 8192


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smoke_root", type=Path)
    args = parser.parse_args()
    root = args.smoke_root
    failures: list[str] = []
    checked = 0
    test_counts: dict[str, int] = {}

    for dataset in DATASETS:
        task = root / dataset / "seed_0"
        yaml_config = task / "config.yaml"
        if not yaml_config.is_file() or "ratios: [0.05]" not in yaml_config.read_text(encoding="utf-8"):
            failures.append(f"{dataset}: missing 5% task config")

        for method in METHODS:
            run_dir = (task / "results" / dataset / method / "ratio_0.05" /
                       "selection_seed_0" / "train_seed_0" / "aug_0")
            selection_config_path = (task / "selections" / method / dataset / "ratio_0.05" /
                                     "selection_seed_0" / "selection_config.json")
            marker_path = run_dir / "run_complete.json"
            metrics_path = run_dir / "metrics.jsonl"
            run_config_path = run_dir / "run_config.json"
            required = (selection_config_path, marker_path, metrics_path, run_config_path)
            if not all(path.is_file() for path in required):
                failures.append(f"{dataset}/{method}: missing selection/metric/completion artifact")
                continue

            selection = read_json(selection_config_path)
            marker = read_json(marker_path)
            run_config = read_json(run_config_path)
            if marker.get("status") != "complete":
                failures.append(f"{dataset}/{method}: completion marker status is not complete")
            if sha256(metrics_path) != marker.get("metrics_sha256"):
                failures.append(f"{dataset}/{method}: metrics hash mismatch")
            selection_config_hash = canonical_json_sha256(selection)
            if selection_config_hash != marker.get("selection_config_sha256"):
                failures.append(f"{dataset}/{method}: selection config hash mismatch")
            if selection_config_hash != run_config.get("selection_config_sha256"):
                failures.append(f"{dataset}/{method}: run config selection-config hash mismatch")
            if run_config.get("selection_sha256") != marker.get("selection_sha256"):
                failures.append(f"{dataset}/{method}: selection hash mismatch")
            if run_config.get("ratio") != RATIO or selection.get("ratio") != RATIO:
                failures.append(f"{dataset}/{method}: ratio is not 0.05")
            if run_config.get("clean_only") is not True:
                failures.append(f"{dataset}/{method}: clean_only is not true")
            if run_config.get("training_source") != "clean" or run_config.get("selection_train_source") != "clean":
                failures.append(f"{dataset}/{method}: selection/training source is not clean")
            if run_config.get("deterministic") is not True:
                failures.append(f"{dataset}/{method}: deterministic mode is not enabled")

            classes = int(run_config["num_classes"])
            expected_budget = int(SMOKE_N * RATIO) // classes
            expected_selected = expected_budget * classes
            actual_budget = int(run_config["budget_per_class"])
            actual_selected = int(run_config["n_selected"])
            if actual_budget != expected_budget or int(selection["budget_per_class"]) != expected_budget:
                failures.append(f"{dataset}/{method}: per-class budget {actual_budget}, expected {expected_budget}")
            if actual_selected != expected_selected or int(selection["n_selected"]) != expected_selected:
                failures.append(f"{dataset}/{method}: selected {actual_selected}, expected {expected_selected}")
            class_counts_path = task / "selections" / method / dataset / "ratio_0.05" / "selection_seed_0" / "class_counts.json"
            if not class_counts_path.is_file():
                failures.append(f"{dataset}/{method}: missing class_counts.json")
            else:
                class_counts = read_json(class_counts_path)
                counts = class_counts.get("counts", class_counts)
                if not isinstance(counts, dict) or len(counts) != classes or any(int(n) != expected_budget for n in counts.values()):
                    failures.append(f"{dataset}/{method}: per-class selection counts do not match {expected_budget}")

            rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
            if not rows:
                failures.append(f"{dataset}/{method}: empty clean-test metrics")
            else:
                sample_counts = {int(row.get("n_samples", 0)) for row in rows}
                if any(row.get("corruption") != "clean" or int(row.get("severity", -1)) != 0 for row in rows):
                    failures.append(f"{dataset}/{method}: smoke metrics include non-clean evaluation")
                if len(sample_counts) != 1 or not 0 < next(iter(sample_counts)) <= SMOKE_N:
                    failures.append(f"{dataset}/{method}: invalid smoke clean-test sample count {sample_counts}")
                else:
                    count = next(iter(sample_counts))
                    test_counts.setdefault(dataset, count)
                    if test_counts[dataset] != count:
                        failures.append(f"{dataset}/{method}: clean-test sample count differs across methods")
                if any(int(row.get("n_classes_observed", 0)) != classes for row in rows):
                    failures.append(f"{dataset}/{method}: clean smoke test did not contain every class")
            checked += 1
            print(f"{dataset}\t{method}\tbudget/class={actual_budget}\tselected={actual_selected}\tclean_test_n={test_counts.get(dataset, 0)}")

    if checked != len(DATASETS) * len(METHODS):
        failures.append(f"checked {checked}/40 method cells")
    if failures:
        raise SystemExit("SMOKE_AUDIT_FAILED\n" + "\n".join(failures))
    print(f"SMOKE_AUDIT_PASS cells={checked} ratio=0.05 train_smoke_n={SMOKE_N} test_counts={test_counts}")


if __name__ == "__main__":
    main()
