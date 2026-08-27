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
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graphcov.run.data import load_dataset  # noqa: E402
from graphcov.run.evaluation import evaluate_selection  # noqa: E402


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


def run_one(
    job: dict[str, object],
    *,
    selection_root: Path,
    output_root: Path,
    training: dict,
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
    evaluation_raw, _ = load_dataset(
        dataset_name, evaluation_split, size=int(training["size"]), verbose=True
    )
    train_dataset = LongLabelDataset(train_raw)
    evaluation_dataset = LongLabelDataset(evaluation_raw)
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
        seed=seed,
        return_history=True,
        verbose=True,
        verbose_per_class=True,
        deterministic=bool(training.get("deterministic", False)),
        num_workers=int(training.get("num_workers", 4)),
        test_every_n_epochs=int(training.get("evaluate_every", 10)),
    )
    recalls = [float(value["accuracy"]) for value in per_class.values()]
    cvar_count = max(1, int(math.ceil(0.2 * len(recalls))))
    row = {
        "schema": "graphcov-v11/downstream-v1",
        "dataset": dataset_name,
        "ratio": ratio,
        "variant": variant,
        "training_seed": seed,
        "evaluation_split": evaluation_split,
        "test_read": evaluation_split == "test",
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
        "best_balanced_accuracy": float(best_metrics["best_bal_acc"]),
        "best_epoch": int(best_metrics.get("best_epoch", -1)),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    result_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
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
            evaluation_split=evaluation_split,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
