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

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graphcov.run.data import load_dataset, wrap_with_augmentation  # noqa: E402
from graphcov.run.embeddings import ResNet18WithFeatures  # noqa: E402
from graphcov.run.evaluation import (  # noqa: E402
    device,
    evaluate_model,
    set_seed,
    train_one_epoch,
)


class LongLabelDataset(Dataset):
    """Ensure MedMNIST uint8 labels enter CrossEntropyLoss as torch.long."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = dataset.labels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        label = torch.as_tensor(label, dtype=torch.long).reshape(-1)[0]
        return image, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--only-dataset")
    parser.add_argument("--only-variant")
    parser.add_argument(
        "--evaluation-split",
        choices=("val", "test"),
        default=None,
        help="Evaluation split. Defaults to val; test must be requested explicitly.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_evaluation_split(config: dict, cli_value: str | None) -> str:
    """Require an explicit CLI opt-in before any test split can be read."""
    evaluation_split = cli_value or "val"
    declared = str(config.get("evaluation_split", "val"))
    if declared != evaluation_split:
        raise ValueError(
            f"config evaluation_split={declared!r} does not match CLI split "
            f"{evaluation_split!r}; pass --evaluation-split test explicitly "
            "to authorize test access"
        )
    return evaluation_split


def resolve_path(value: str, project_root: Path, config_dir: Path) -> Path:
    expanded = os.path.expandvars(value.replace("{project_root}", str(project_root)))
    path = Path(expanded).expanduser()
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
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(f"{contiguous.dtype}|{contiguous.shape}".encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def expand_jobs(config: dict) -> list[dict[str, object]]:
    expanded = []
    for group in config["jobs"]:
        for ratio in group["ratios"]:
            for variant in group["variants"]:
                for seed in group["seeds"]:
                    expanded.append(
                        {
                            "dataset": str(group["dataset"]),
                            "ratio": float(ratio),
                            "variant": str(variant),
                            "seed": int(seed),
                        }
                    )
    return expanded


def validate_selection(selected: np.ndarray, labels: np.ndarray, budget: int) -> dict[int, int]:
    classes = np.unique(labels)
    expected = int(budget * len(classes))
    if len(selected) != expected or len(np.unique(selected)) != expected:
        raise ValueError(f"invalid selected size: {len(selected)} != {expected}")
    if selected.min() < 0 or selected.max() >= len(labels):
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
    training: dict,
    seed: int,
    checkpoint_path: Path,
) -> dict:
    """Train on train, select by validation BA, and persist the best model.

    The upstream ``evaluate_selection`` function calls its evaluation input
    ``test_dataset`` and reads it throughout training. v11 needs a separate
    validation-selection phase so that the final test score is not used to
    choose a checkpoint.
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()
    evaluate_every = max(1, int(training.get("evaluate_every", 10)))
    history = []
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
            val_acc, val_balanced_accuracy = evaluate_model(model, validation_loader)
            if (
                val_balanced_accuracy > best_val_balanced_accuracy
                or best_epoch == 0
            ):
                best_val_acc = float(val_acc)
                best_val_balanced_accuracy = float(val_balanced_accuracy)
                best_epoch = epoch_number
                torch.save(
                    {
                        "schema": "graphcov-v11/best-checkpoint-v1",
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

        entry = {
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


def evaluate_checkpoint(
    checkpoint_path: Path,
    evaluation_dataset: Dataset,
    num_classes: int,
    in_channels: int,
    batch_size: int,
    num_workers: int,
) -> tuple[float, float, dict]:
    """Load one validation-selected checkpoint and evaluate one split once."""
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ResNet18WithFeatures(num_classes, in_channels, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return evaluate_model(
        model,
        evaluation_loader,
        verbose=True,
        verbose_per_class=True,
        return_per_class=True,
    )


def run_one(
    job: dict[str, object],
    *,
    selection_root: Path,
    output_root: Path,
    training: dict,
    validation_split: str,
    evaluation_split: str,
    force: bool,
) -> dict[str, object]:
    dataset_name = str(job["dataset"])
    ratio = float(job["ratio"])
    variant = str(job["variant"])
    seed = int(job["seed"])
    selection_dir = selection_root / dataset_name / ratio_key(ratio) / variant
    selection_path = selection_dir / "selected_indices.npy"
    metrics_path = selection_dir / "selection_metrics.json"
    if not selection_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"Job-1 output missing for {dataset_name}/{ratio}/{variant}")
    selection_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    budget = int(selection_metrics["budget_per_class"])

    run_dir = output_root / dataset_name / ratio_key(ratio) / variant / f"seed{seed}"
    result_path = run_dir / f"{evaluation_split}_result.json"
    if result_path.exists() and not force:
        print(f"[v11:job2] skip existing {result_path}", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))

    selected = np.asarray(np.load(selection_path), dtype=np.int64).reshape(-1)
    observed_index_hash = sha256_array(selected)
    expected_index_hash = str(selection_metrics["index_sha256"])
    if observed_index_hash != expected_index_hash:
        raise ValueError(
            f"selected-index hash mismatch for {dataset_name}/{ratio}/{variant}: "
            f"{observed_index_hash} != {expected_index_hash}"
        )
    train_raw, info = load_dataset(
        dataset_name, "train", size=int(training["size"]), verbose=True
    )
    validation_raw, _ = load_dataset(
        dataset_name, validation_split, size=int(training["size"]), verbose=True
    )
    train_dataset = LongLabelDataset(train_raw)
    validation_dataset = LongLabelDataset(validation_raw)
    labels = np.asarray(train_raw.labels, dtype=np.int64).reshape(-1)
    class_counts = validate_selection(selected, labels, budget)
    classes = np.unique(labels)

    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[v11:job2] start dataset={dataset_name} ratio={ratio} "
        f"variant={variant} seed={seed}",
        flush=True,
    )
    started = time.perf_counter()
    checkpoint_path = run_dir / "best_val_checkpoint.pt"
    training_result = train_with_validation_checkpoint(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        selected_indices=selected.tolist(),
        num_classes=len(classes),
        in_channels=int(info["n_channels"]),
        training=training,
        seed=seed,
        checkpoint_path=checkpoint_path,
    )
    # Validation calibration never reads test. Frozen confirmation reads test
    # only after training, from the checkpoint chosen by validation.
    if evaluation_split == "val":
        evaluation_dataset = validation_dataset
        test_evaluations = 0
    else:
        evaluation_raw, _ = load_dataset(
            dataset_name, "test", size=int(training["size"]), verbose=True
        )
        evaluation_dataset = LongLabelDataset(evaluation_raw)
        test_evaluations = 1
    accuracy, balanced_accuracy, per_class = evaluate_checkpoint(
        checkpoint_path,
        evaluation_dataset,
        len(classes),
        int(info["n_channels"]),
        int(training["batch_size"]),
        int(training.get("num_workers", 4)),
    )
    recalls = [float(value["accuracy"]) for value in per_class.values()]
    cvar_count = max(1, int(math.ceil(0.2 * len(recalls))))
    row = {
        "schema": "graphcov-v11/downstream-v1",
        "dataset": dataset_name,
        "ratio": ratio,
        "variant": variant,
        "training_seed": seed,
        "evaluation_protocol": "validation_checkpoint_v1",
        "validation_split": validation_split,
        "evaluation_split": evaluation_split,
        "test_read": evaluation_split == "test",
        "test_evaluations": test_evaluations,
        "selection_path": str(selection_path),
        "selection_file_sha256": sha256_file(selection_path),
        "selection_index_sha256": observed_index_hash,
        "selection_metrics_sha256": sha256_file(metrics_path),
        "n_selected": int(len(selected)),
        "budget_per_class": budget,
        "class_counts": class_counts,
        "training": training,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "worst_class_recall": float(min(recalls)),
        "class_cvar20": float(np.mean(sorted(recalls)[:cvar_count])),
        "best_balanced_accuracy": float(balanced_accuracy),
        "best_val_accuracy": float(training_result["best_val_accuracy"]),
        "best_val_balanced_accuracy": float(
            training_result["best_val_balanced_accuracy"]
        ),
        "best_epoch": int(training_result["best_epoch"]),
        "checkpoint_path": str(checkpoint_path),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    result_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "history.json").write_text(
        json.dumps(training_result["history"]), encoding="utf-8"
    )
    (run_dir / "per_class.json").write_text(
        json.dumps(per_class, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard arguments")
    project_root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "graphcov-v11/job2-config-v1":
        raise ValueError("unexpected job2 config schema")
    config_dir = config_path.parent
    selection_root = resolve_path(config["selection_root"], project_root, config_dir)
    output_root = resolve_path(config["output_root"], project_root, config_dir)
    evaluation_split = resolve_evaluation_split(config, args.evaluation_split)
    validation_split = str(config.get("validation_split", "val"))
    if validation_split not in {"val", "test"}:
        raise ValueError("validation_split must be val or test")
    if evaluation_split == "test" and validation_split != "val":
        raise ValueError("final test evaluation requires validation_split=val")
    jobs = expand_jobs(config)
    jobs = [
        job
        for job in jobs
        if (not args.only_dataset or job["dataset"] == args.only_dataset)
        and (not args.only_variant or job["variant"] == args.only_variant)
    ]
    jobs = [job for position, job in enumerate(jobs) if position % args.num_shards == args.shard_index]
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return 0
    output_root.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        run_one(
            job,
            selection_root=selection_root,
            output_root=output_root,
            training=config["training"],
            validation_split=validation_split,
            evaluation_split=evaluation_split,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
