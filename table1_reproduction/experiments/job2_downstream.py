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
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from table1_reproduction.official_runtime import configure_official_runtime  # noqa: E402

configure_official_runtime()

from graphcov.run.data import load_dataset, wrap_with_augmentation  # noqa: E402
from graphcov.run.embeddings import ResNet18WithFeatures  # noqa: E402
from graphcov.run.evaluation import (  # noqa: E402
    device,
    evaluate_model,
    set_seed,
    train_one_epoch,
)


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


def train_with_validation_checkpoint(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    selected_indices: list[int],
    num_classes: int,
    in_channels: int,
    training: dict[str, Any],
    seed: int,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Train on train, select the checkpoint on validation, and test once.

    The upstream evaluator calls its input ``test_dataset`` and evaluates it
    every N epochs. That is unsuitable for an unbiased final test estimate.
    This wrapper keeps the upstream model/training primitives but separates
    validation model selection from the one-time final test evaluation.
    """
    set_seed(seed, deterministic=bool(training.get("deterministic", False)))

    selected_subset: Dataset = Subset(train_dataset, selected_indices)
    if bool(training.get("augmentation", False)):
        selected_subset = wrap_with_augmentation(
            selected_subset, in_channels=in_channels, size=int(training["size"])
        )

    batch_size = int(training["batch_size"])
    num_workers = int(training.get("num_workers", 4))
    train_loader = DataLoader(
        selected_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    model = ResNet18WithFeatures(num_classes, in_channels, pretrained=False).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(training["learning_rate"]),
        momentum=float(training.get("momentum", 0.9)),
        weight_decay=float(training.get("weight_decay", 5e-4)),
        nesterov=bool(training.get("nesterov", False)),
    )
    epochs = int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = torch.nn.CrossEntropyLoss()
    evaluate_every = max(1, int(training.get("evaluate_every", 10)))
    history: list[dict[str, Any]] = []
    best_val_acc = -float("inf")
    best_val_balanced_accuracy = -float("inf")
    best_epoch = 0

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        scheduler.step()
        epoch_number = epoch + 1
        val_acc = None
        val_balanced_accuracy = None
        if epoch_number % evaluate_every == 0 or epoch_number == epochs:
            val_acc, val_balanced_accuracy = evaluate_model(
                model, validation_loader
            )
            if (
                val_balanced_accuracy > best_val_balanced_accuracy
                or best_epoch == 0
            ):
                best_val_acc = float(val_acc)
                best_val_balanced_accuracy = float(val_balanced_accuracy)
                best_epoch = epoch_number
                torch.save(
                    {
                        "schema": "graphcov-table1/best-checkpoint-v1",
                        "epoch": best_epoch,
                        "seed": seed,
                        "best_val_accuracy": best_val_acc,
                        "best_val_balanced_accuracy": best_val_balanced_accuracy,
                        "model_state_dict": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                    },
                    checkpoint_path,
                )

        entry: dict[str, Any] = {
            "epoch": epoch_number,
            "train_loss": round(float(train_loss), 6),
            "train_acc": round(float(train_acc), 6),
            "lr": round(float(scheduler.get_last_lr()[0]), 8),
        }
        if val_acc is not None:
            entry["val_acc"] = round(float(val_acc), 6)
            entry["val_bal_acc"] = round(float(val_balanced_accuracy), 6)
        history.append(entry)

    if best_epoch == 0 or not checkpoint_path.exists():
        raise RuntimeError("training ended without a validation checkpoint")

    return {
        "history": history,
        "best_val_accuracy": best_val_acc,
        "best_val_balanced_accuracy": best_val_balanced_accuracy,
        "best_epoch": best_epoch,
    }


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
    validation_split = str(config.get("validation_split", "val"))
    if validation_split == config["evaluation_split"]:
        raise ValueError("validation_split and evaluation_split must be different")
    train_raw, info = load_dataset(
        dataset, "train", size=int(training["size"]), verbose=True
    )
    validation_raw, _ = load_dataset(
        dataset, validation_split, size=int(training["size"]), verbose=True
    )
    train_dataset = LongLabelDataset(train_raw)
    validation_dataset = LongLabelDataset(validation_raw)
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
    checkpoint_path = output_dir / "best_val_checkpoint.pt"
    training_result = train_with_validation_checkpoint(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        selected_indices=selected.tolist(),
        num_classes=len(classes),
        in_channels=int(info["n_channels"]),
        training=training,
        seed=training_seed,
        checkpoint_path=checkpoint_path,
    )
    # Test data is intentionally loaded only after validation-selected
    # checkpoint creation. It is evaluated exactly once below.
    evaluation_raw, _ = load_dataset(
        dataset, config["evaluation_split"], size=int(training["size"]), verbose=True
    )
    evaluation_dataset = LongLabelDataset(evaluation_raw)
    test_loader = DataLoader(
        evaluation_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training.get("num_workers", 4)),
        pin_memory=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ResNet18WithFeatures(
        len(classes), int(info["n_channels"]), pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    accuracy, balanced_accuracy, per_class = evaluate_model(
        model, test_loader, verbose=True, verbose_per_class=True, return_per_class=True
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
        "evaluation_protocol": "validation_checkpoint_v1",
        "validation_split": validation_split,
        "evaluation_split": config["evaluation_split"],
        "test_read": config["evaluation_split"] == "test",
        "test_evaluations": 1,
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
        # Kept for the existing summarizer: this is test BA from the
        # validation-selected checkpoint, not the maximum test BA.
        "best_balanced_accuracy": float(balanced_accuracy),
        "best_val_accuracy": float(training_result["best_val_accuracy"]),
        "best_val_balanced_accuracy": float(
            training_result["best_val_balanced_accuracy"]
        ),
        "best_epoch": int(training_result["best_epoch"]),
        "checkpoint_path": str(checkpoint_path),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "history.json").write_text(
        json.dumps(training_result["history"]), encoding="utf-8"
    )
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
