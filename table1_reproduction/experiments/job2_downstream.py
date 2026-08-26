#!/usr/bin/env python3
"""Job 2: train ResNet-18 on frozen Job-1 indices and evaluate one split."""

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
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from table1_reproduction.official_runtime import configure_official_runtime  # noqa: E402

configure_official_runtime()

from graphcov.run.data import load_dataset  # noqa: E402
from graphcov.run.evaluation import evaluate_selection  # noqa: E402


def resolve_path(value: str, project_root: Path, config_dir: Path) -> Path:
    value = value.replace("{project_root}", str(project_root)).replace(
        "{config_dir}", str(config_dir)
    )
    path = Path(os.path.expandvars(value)).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


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


class LongLabelDataset(Dataset):
    """Ensure MedMNIST labels enter CrossEntropyLoss as torch.long."""

    def __init__(self, dataset: Any):
        self.dataset = dataset
        self.labels = dataset.labels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, torch.as_tensor(label, dtype=torch.long).reshape(-1)[0]


def selection_seed_for(method: str, training_seed: int, config: dict[str, Any]) -> int:
    policy_cfg = config.get("selection_seed_policy", {})
    policy = policy_cfg.get(method, policy_cfg.get("default", "fixed"))
    if policy == "training_seed":
        return int(training_seed)
    if policy == "fixed":
        return int(config.get("fixed_selection_seed", 42))
    raise ValueError(f"unknown selection policy {policy}")


def expand_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    for dataset in config["datasets"]:
        for ratio in config["ratios"]:
            for method in config["methods"]:
                for seed in config["training_seeds"]:
                    jobs.append(
                        {
                            "dataset": str(dataset),
                            "ratio": float(ratio),
                            "method": str(method),
                            "training_seed": int(seed),
                        }
                    )
    return jobs


def validate_selection(selected: np.ndarray, labels: np.ndarray, budget: int) -> dict[int, int]:
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    classes = np.unique(labels)
    expected = budget * len(classes)
    if len(selected) != expected or len(np.unique(selected)) != expected:
        raise ValueError(f"invalid selected size or duplicate indices: {len(selected)} != {expected}")
    if expected and (selected.min() < 0 or selected.max() >= len(labels)):
        raise ValueError("selected index is outside the training dataset")
    counts = {int(cls): int(np.sum(labels[selected] == cls)) for cls in classes}
    if any(value != budget for value in counts.values()):
        raise ValueError(f"class quota violation: {counts}")
    return counts


def run_one(
    job: dict[str, Any],
    selection_root: Path,
    output_root: Path,
    config: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    dataset = job["dataset"]
    ratio = float(job["ratio"])
    method = job["method"]
    training_seed = int(job["training_seed"])
    selection_seed = selection_seed_for(method, training_seed, config)
    selection_dir = selection_root / dataset / ratio_key(ratio) / method / f"seed{selection_seed}"
    selection_path = selection_dir / "selected_indices.npy"
    metrics_path = selection_dir / "selection_metrics.json"
    if not selection_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"missing frozen selection: {selection_dir}")
    selection_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected = np.asarray(np.load(selection_path), dtype=np.int64).reshape(-1)
    if sha256_array(selected) != selection_metrics["index_sha256"]:
        raise ValueError(f"selection hash mismatch: {selection_path}")

    output_dir = output_root / dataset / ratio_key(ratio) / method / f"seed{training_seed}"
    result_path = output_dir / f"{config['evaluation_split']}_result.json"
    if result_path.exists() and not force:
        print(f"[table1:job2] skip existing {result_path}", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))

    training = config["training"]
    train_raw, info = load_dataset(dataset, "train", size=int(training["size"]), verbose=True)
    evaluation_raw, _ = load_dataset(
        dataset, config["evaluation_split"], size=int(training["size"]), verbose=True
    )
    train_dataset = LongLabelDataset(train_raw)
    evaluation_dataset = LongLabelDataset(evaluation_raw)
    labels = np.asarray(train_raw.labels, dtype=np.int64).reshape(-1)
    classes = np.unique(labels)
    budget = int(selection_metrics["budget_per_class"])
    class_counts = validate_selection(selected, labels, budget)

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(
        f"[table1:job2] start {dataset} {ratio_key(ratio)} {method} "
        f"selection_seed={selection_seed} training_seed={training_seed}",
        flush=True,
    )
    accuracy, balanced_accuracy, history, best_metrics, per_class = evaluate_selection(
        train_dataset=train_dataset,
        test_dataset=evaluation_dataset,
        selected_indices=selected.tolist(),
        num_classes=len(classes),
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
        "selection_seed": selection_seed,
        "training_seed": training_seed,
        "evaluation_split": config["evaluation_split"],
        "test_read": config["evaluation_split"] == "test",
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
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--only-dataset")
    parser.add_argument("--only-method")
    parser.add_argument("--only-ratio", type=float)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard arguments")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "graphcov-table1/job2-config-v1":
        raise ValueError("unexpected Job 2 schema")
    jobs = expand_jobs(config)
    jobs = [
        job for job in jobs
        if not args.only_dataset or job["dataset"] == args.only_dataset
    ]
    jobs = [
        job for job in jobs
        if not args.only_method or job["method"] == args.only_method
    ]
    jobs = [
        job for job in jobs
        if args.only_ratio is None or abs(job["ratio"] - args.only_ratio) < 1e-12
    ]
    if args.dry_run:
        for job in jobs:
            job["selection_seed"] = selection_seed_for(job["method"], job["training_seed"], config)
        print(json.dumps(jobs, indent=2))
        return 0
    jobs = [job for pos, job in enumerate(jobs) if pos % args.num_shards == args.shard_index]
    config_dir = config_path.parent
    project_root = args.project_root.resolve()
    selection_root = resolve_path(config["selection_root"], project_root, config_dir)
    output_root = resolve_path(config["output_root"], project_root, config_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        run_one(job, selection_root, output_root, config, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
