#!/usr/bin/env python3
"""Freeze the ten Table 1 Graph-A2 selections without downstream training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from official_runtime import configure_official_runtime


def array_sha(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(f"{values.dtype}|{values.shape}".encode("ascii"))
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def ratio_key(value: float) -> str:
    return f"r{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.dataset not in protocol["datasets"]:
        raise ValueError(f"unknown dataset: {args.dataset}")
    configure_official_runtime()

    from graphcov.run.data import get_labels, load_dataset
    from graphcov.run.embeddings import load_or_compute_embeddings
    from graphcov.run.selection import select

    train_dataset, info = load_dataset(args.dataset, "train", size=224, verbose=True)
    labels = get_labels(train_dataset)
    embeddings = load_or_compute_embeddings(
        dataset_name=args.dataset,
        split="train",
        source="uni",
        dataset=train_dataset,
        num_classes=len(np.unique(labels)),
        in_channels=int(info["n_channels"]),
        size=224,
        seed=42,
        cache_dir=args.cache,
        force_recompute=False,
        verbose=True,
    )["embeddings"]

    records = []
    for ratio in protocol["ratios"]:
        budget = int(int(len(labels) * float(ratio)) // len(np.unique(labels)))
        selected = np.asarray(
            select(
                method="graph_a2",
                labels=labels,
                budget_per_class=budget,
                embeddings=embeddings,
                importance=np.ones(len(labels), dtype=np.float32),
                seed=42,
                k_neighbors=50,
                k_hops=2,
                global_selection=True,
                sparse_cpu=True,
                verbose=True,
                _verbose_level=1,
            ),
            dtype=np.int64,
        ).reshape(-1)
        destination = args.output / args.dataset / ratio_key(float(ratio))
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "selected_indices.npy", selected)
        np.save(destination / "selected_sorted.npy", np.sort(selected))
        record = {
            "dataset": args.dataset,
            "ratio": float(ratio),
            "selection_seed": 42,
            "n_train": int(len(labels)),
            "n_classes": int(len(np.unique(labels))),
            "budget_per_class": budget,
            "n_selected": int(selected.size),
            "ordered_index_sha256": array_sha(selected),
            "sorted_index_sha256": array_sha(np.sort(selected)),
            "local_path": str(destination / "selected_indices.npy"),
            "test_read": False,
            "selection": protocol["selection"],
        }
        (destination / "selection_metrics.json").write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
        records.append(record)

    (args.output / "selection_manifest.json").write_text(
        json.dumps({"schema": "graphcov-table1-ours-selection-manifest-v1", "records": records}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(records, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
