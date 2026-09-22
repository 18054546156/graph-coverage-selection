#!/usr/bin/env python3
"""Complete the twelve missing public Table 1 k=10 downstream runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from table1_reproduction.official_runtime import configure_official_runtime  # noqa: E402

configure_official_runtime()

from graphcov.run.data import load_dataset  # noqa: E402
from graphcov.run.evaluation import evaluate_selection  # noqa: E402


ALLOWED_METHODS = {"el2n", "eva", "fps"}
ALLOWED_SEEDS = {45, 46}


def ratio_key(ratio: float) -> str:
    return f"r{ratio:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(f"{values.dtype}|{values.shape}".encode("ascii"))
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def selection_seed(method: str, training_seed: int, config: dict[str, Any]) -> int:
    policy = config["selection_seed_policy"].get(method, "fixed")
    if policy == "training_seed":
        return training_seed
    if policy == "fixed":
        return int(config["fixed_selection_seed"])
    raise ValueError(f"unexpected selection seed policy: {policy}")


def validate_selection(selected: np.ndarray, labels: np.ndarray, budget: int) -> dict[int, int]:
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    classes = np.unique(labels)
    expected = budget * len(classes)
    if len(selected) != expected or len(np.unique(selected)) != expected:
        raise ValueError(f"invalid selected size: {len(selected)} != {expected}")
    if expected and (selected.min() < 0 or selected.max() >= len(labels)):
        raise ValueError("selected index is outside the training dataset")
    counts = {int(cls): int(np.sum(labels[selected] == cls)) for cls in classes}
    if any(value != budget for value in counts.values()):
        raise ValueError(f"class quota violation: {counts}")
    return counts


def run_one(
    config: dict[str, Any],
    project_root: Path,
    method: str,
    training_seed: int,
    ratio: float,
    force: bool,
) -> None:
    if method not in ALLOWED_METHODS or training_seed not in ALLOWED_SEEDS:
        raise ValueError("this completion runner is restricted to the twelve frozen gaps")
    dataset = "bloodmnist"
    config_dir = project_root / "table1_reproduction" / "configs"
    selection_root = project_root / "table1_reproduction" / "outputs" / "job1_selection"
    output_root = project_root / "table1_reproduction" / "outputs" / "job2_table1"
    selection_seed_value = selection_seed(method, training_seed, config)
    selection_dir = (
        selection_root / dataset / ratio_key(ratio) / method / f"seed{selection_seed_value}"
    )
    selection_path = selection_dir / "selected_indices.npy"
    metrics_path = selection_dir / "selection_metrics.json"
    if not selection_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"missing frozen selection: {selection_dir}")

    output_dir = output_root / dataset / ratio_key(ratio) / method / f"seed{training_seed}"
    result_path = output_dir / "test_result.json"
    if result_path.exists() and not force:
        print(f"[missing-k10] skip existing {result_path}", flush=True)
        return

    selection_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected = np.asarray(np.load(selection_path), dtype=np.int64).reshape(-1)
    if sha256_array(selected) != selection_metrics["index_sha256"]:
        raise ValueError(f"selection hash mismatch: {selection_path}")

    training = config["training"]
    train_raw, info = load_dataset(dataset, "train", size=int(training["size"]), verbose=True)
    evaluation_raw, _ = load_dataset(
        dataset, config["evaluation_split"], size=int(training["size"]), verbose=True
    )
    labels = np.asarray(train_raw.labels, dtype=np.int64).reshape(-1)
    budget = int(selection_metrics["budget_per_class"])
    class_counts = validate_selection(selected, labels, budget)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    print(
        f"[missing-k10] start {dataset} {ratio_key(ratio)} {method} "
        f"selection_seed={selection_seed_value} training_seed={training_seed}",
        flush=True,
    )
    accuracy, balanced_accuracy, history, best_metrics, per_class = evaluate_selection(
        train_dataset=train_raw,
        test_dataset=evaluation_raw,
        selected_indices=selected.tolist(),
        num_classes=len(np.unique(labels)),
        in_channels=int(info["n_channels"]),
        training_paradigm="epoch",
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        lr=float(training["learning_rate"]),
        momentum=float(training.get("momentum", 0.9)),
        nesterov=bool(training.get("nesterov", False)),
        weight_decay=float(training.get("weight_decay", 5e-4)),
        augment=bool(training.get("augmentation", False)),
        size=int(training["size"]),
        seed=training_seed,
        return_history=True,
        verbose=True,
        verbose_per_class=True,
        deterministic=bool(training.get("deterministic", False)),
        num_workers=int(training.get("num_workers", 4)),
        test_every_n_epochs=int(training.get("evaluate_every", 10)),
    )
    recalls = [float(value["accuracy"]) for value in per_class.values()]
    cvar_count = max(1, int(math.ceil(0.2 * len(recalls))))
    result = {
        "schema": "graphcov-table1/downstream-v1",
        "dataset": dataset,
        "ratio": ratio,
        "method": method,
        "selection_seed": selection_seed_value,
        "training_seed": training_seed,
        "evaluation_split": config["evaluation_split"],
        "test_read": True,
        "selection_path": str(selection_path),
        "selection_file_sha256": sha256_file(selection_path),
        "selection_index_sha256": sha256_array(selected),
        "n_selected": len(selected),
        "budget_per_class": budget,
        "class_counts": class_counts,
        "training": training,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "worst_class_recall": float(min(recalls)),
        "class_cvar20": float(np.mean(sorted(recalls)[:cvar_count])),
        "best_balanced_accuracy": float(best_metrics["best_bal_acc"]),
        "best_epoch": int(best_metrics.get("best_epoch", -1)),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (output_dir / "per_class.json").write_text(
        json.dumps(per_class, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[missing-k10] wrote {result_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(ALLOWED_METHODS), required=True)
    parser.add_argument("--training-seed", type=int, choices=sorted(ALLOWED_SEEDS), required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema") != "graphcov-table1/job2-config-v1":
        raise ValueError("unexpected public Table 1 config schema")
    for ratio in (0.02, 0.05):
        run_one(config, args.project_root.resolve(), args.method, args.training_seed, ratio, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
