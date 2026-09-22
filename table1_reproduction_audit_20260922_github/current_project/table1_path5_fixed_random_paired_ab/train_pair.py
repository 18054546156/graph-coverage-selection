#!/usr/bin/env python3
"""Train the frozen Random and Graph-A2 subsets for one paired seed."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np

from official_runtime import configure_official_runtime
from paired_core import file_sha256, index_sha256


def atomic_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.move(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.seed not in protocol["training_seeds"]:
        raise ValueError(f"seed {args.seed} is not frozen in protocol")
    manifest = json.loads((args.artifacts / "manifest.json").read_text(encoding="utf-8"))
    configure_official_runtime()

    import torch
    from graphcov.run.data import load_dataset
    from graphcov.run.evaluation import evaluate_selection

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dataset_cfg = protocol["dataset"]
    training = protocol["training"]
    train_dataset, info = load_dataset(
        dataset_cfg["name"], "train", int(training["image_size"]), verbose=True
    )
    test_dataset, _ = load_dataset(
        dataset_cfg["name"], "test", int(training["image_size"]), verbose=False
    )
    if len(train_dataset) != int(dataset_cfg["n_train"]):
        raise ValueError(f"training-set length mismatch: {len(train_dataset)}")

    for arm_name in protocol["arm_order"]:
        output_dir = args.output / f"seed{args.seed}" / arm_name
        metrics_path = output_dir / "metrics.json"
        if metrics_path.is_file():
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            if existing.get("complete") is True:
                print(f"skip complete arm: seed={args.seed} arm={arm_name}", flush=True)
                continue

        index_path = args.artifacts / arm_name / "selected_indices.npy"
        indices = np.load(index_path, allow_pickle=False).astype(np.int64, copy=False)
        arm_manifest = manifest["arms"][arm_name]
        if index_sha256(indices) != arm_manifest["canonical_index_sha256"]:
            raise ValueError(f"{arm_name} canonical index SHA changed")
        if file_sha256(index_path) != arm_manifest["canonical_file_sha256"]:
            raise ValueError(f"{arm_name} canonical file SHA changed")
        if not np.all(indices[1:] > indices[:-1]):
            raise ValueError(f"{arm_name} indices are not strictly increasing")

        started = time.time()
        final_acc, final_ba, history, best, per_class = evaluate_selection(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            selected_indices=indices.tolist(),
            num_classes=int(dataset_cfg["n_classes"]),
            in_channels=int(info["n_channels"]),
            training_paradigm="epoch",
            epochs=int(training["epochs"]),
            test_every_n_epochs=int(training["test_every_n_epochs"]),
            batch_size=int(training["batch_size"]),
            lr=float(training["learning_rate"]),
            momentum=float(training["momentum"]),
            nesterov=bool(training["nesterov"]),
            weight_decay=float(training["weight_decay"]),
            augment=bool(training["augmentation"]),
            size=int(training["image_size"]),
            seed=args.seed,
            return_history=True,
            verbose=True,
            deterministic=bool(training["deterministic"]),
            num_workers=int(training["num_workers"]),
        )
        elapsed = time.time() - started
        atomic_json(output_dir / "history.json", history)
        atomic_json(
            metrics_path,
            {
                "schema": "graphcov-table1/path5-fixed-selection-metrics-v1",
                "complete": True,
                "dataset": dataset_cfg["name"],
                "ratio": dataset_cfg["ratio"],
                "arm": arm_name,
                "arm_label": protocol["arms"][arm_name]["label"],
                "selection_seed": protocol["arms"][arm_name]["selection_seed"],
                "training_seed": args.seed,
                "n_selected": int(len(indices)),
                "canonical_index_sha256": index_sha256(indices),
                "canonical_file_sha256": file_sha256(index_path),
                "canonical_order": "strictly increasing",
                "accuracy": float(final_acc),
                "balanced_accuracy": float(final_ba),
                "best_accuracy_audit_only": float(best["best_acc"]),
                "best_balanced_accuracy_audit_only": float(best["best_bal_acc"]),
                "best_epoch_audit_only": int(best["best_epoch"]),
                "per_class": per_class,
                "elapsed_seconds": elapsed,
                "gpu": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "endpoint": training["endpoint"],
                "test_read": True,
            },
        )
        del history, best, per_class
        gc.collect()
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
