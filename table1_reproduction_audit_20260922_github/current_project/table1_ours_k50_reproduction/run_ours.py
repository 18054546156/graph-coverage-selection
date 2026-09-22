#!/usr/bin/env python3
"""Run one Table 1 dataset with the immutable author Graph-A2 implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from official_runtime import configure_official_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.dataset not in protocol["datasets"]:
        raise ValueError(f"dataset is not in frozen protocol: {args.dataset}")
    configure_official_runtime()

    from graphcov.run.experiment import run_experiment

    selection = protocol["selection"]
    training = protocol["training"]
    config = {
        "datasets": [args.dataset],
        "embeddings": [selection["embedding_source"]],
        "methods": ["graph_a2"],
        "ratios": [float(value) for value in protocol["ratios"]],
        "trials": len(protocol["training_seeds"]),
        "seed": int(protocol["selection_seed"]),
        "training_paradigm": "epoch",
        "epochs": int(training["epochs"]),
        "test_every_n_epochs": int(training["evaluate_every"]),
        "batch_size": int(training["batch_size"]),
        "lr": float(training["lr"]),
        "momentum": float(training["momentum"]),
        "nesterov": bool(training["nesterov"]),
        "weight_decay": float(training["weight_decay"]),
        "augment": bool(training["augmentation"]),
        "size": int(training["size"]),
        "num_workers": 4,
        "deterministic": False,
        "k_neighbors": int(selection["k_neighbors"]),
        "k_hops": int(selection["k_hops"]),
        "global_selection": True,
        "sparse_cpu": bool(selection["sparse_cpu"]),
        "cache_dir": str(args.cache.resolve()),
        "output_dir": str(args.output.resolve()),
        "verbose": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(config, indent=2, sort_keys=True), flush=True)
    run_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
