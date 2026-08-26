#!/usr/bin/env python3
"""Job 1: create frozen Table 1 subsets without reading validation or test data."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from graphcov.run.data import get_labels, load_dataset  # noqa: E402
from graphcov.run.embeddings import load_or_compute_embeddings, load_or_compute_raw_dynamics  # noqa: E402
from graphcov.run.eva import derive_eva_scores, get_optimal_windows  # noqa: E402
from graphcov.run.selection import select  # noqa: E402


TABLE1_METHODS = {
    "random",
    "el2n",
    "forgetting",
    "eva",
    "facility",
    "fps",
    "herding",
    "graph_a2",
}


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


def sha256_array(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(f"{values.dtype}|{values.shape}".encode("ascii"))
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def validate_selected(selected: np.ndarray, labels: np.ndarray, budget: int) -> dict[int, int]:
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    classes = np.unique(labels)
    expected = int(budget * len(classes))
    if len(selected) != expected:
        raise ValueError(f"selected size {len(selected)} != expected {expected}")
    if len(np.unique(selected)) != expected:
        raise ValueError("selection contains duplicate indices")
    if expected and (int(selected.min()) < 0 or int(selected.max()) >= len(labels)):
        raise ValueError("selection contains an out-of-range index")
    counts = {int(cls): int(np.sum(labels[selected] == cls)) for cls in classes}
    if any(count != budget for count in counts.values()):
        raise ValueError(f"class quota violation: {counts}")
    return counts


def selection_seeds(method_cfg: dict[str, Any], config: dict[str, Any]) -> list[int]:
    policy = method_cfg.get("selection_seed_policy", "fixed")
    if policy == "training_seed":
        return [int(seed) for seed in config["training_seeds"]]
    if policy == "fixed":
        return [int(config.get("fixed_selection_seed", 42))]
    raise ValueError(f"unknown selection_seed_policy: {policy}")


def load_dynamics(
    dataset_name: str,
    info: dict[str, Any],
    dynamics_cfg: dict[str, Any],
    dynamics_cache_root: Path,
) -> dict[str, np.ndarray]:
    size = int(dynamics_cfg.get("size", 28))
    dynamics_dataset, _ = load_dataset(dataset_name, "train", size=size, verbose=False)
    return load_or_compute_raw_dynamics(
        dataset_name=dataset_name,
        split="train",
        dataset=dynamics_dataset,
        num_classes=len(info["label"]),
        in_channels=int(info["n_channels"]),
        size=size,
        seed=int(dynamics_cfg.get("seed", 42)),
        eva_epochs=int(dynamics_cfg.get("epochs", 200)),
        window_size=int(dynamics_cfg.get("window_size", 10)),
        cache_dir=dynamics_cache_root,
        force_recompute=False,
        verbose=True,
    )


def scores_for_method(
    method_id: str,
    dataset_name: str,
    ratio: float,
    labels: np.ndarray,
    info: dict[str, Any],
    dynamics: dict[str, np.ndarray] | None,
    config: dict[str, Any],
) -> dict[str, np.ndarray | None]:
    result: dict[str, np.ndarray | None] = {
        "embeddings": None,
        "el2n_scores": None,
        "forgetting_scores": None,
        "eva_scores": None,
    }
    if dynamics is None:
        return result
    result["forgetting_scores"] = dynamics["forgetting_scores"]
    all_l2 = dynamics["all_l2_scores"]
    if method_id == "el2n":
        result["el2n_scores"] = all_l2[: min(20, all_l2.shape[0])].mean(axis=0)
    elif method_id == "eva":
        dynamics_cfg = config.get("dynamics", {})
        early_start, late_start = get_optimal_windows(
            dataset_name,
            ratio,
            eva_epochs=int(dynamics_cfg.get("epochs", 200)),
            verbose=False,
        )
        result["eva_scores"] = derive_eva_scores(
            all_l2,
            window_size=int(dynamics_cfg.get("window_size", 10)),
            early_window_start=early_start,
            late_window_start=late_start,
            verbose=False,
        )[0]
    return result


def run(config_path: Path, project_root: Path, args: argparse.Namespace) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "graphcov-table1/job1-config-v1":
        raise ValueError("unexpected Job 1 schema")
    config_dir = config_path.parent
    output_root = resolve_path(config["output_root"], project_root, config_dir)
    embedding_cache_root = resolve_path(
        config["embedding_cache_root"], project_root, config_dir
    )
    dynamics_cache_root = resolve_path(
        config.get("dynamics_cache_root", config["embedding_cache_root"]),
        project_root,
        config_dir,
    )
    methods_cfg = config["methods"]
    datasets = [args.only_dataset] if args.only_dataset else config["datasets"]
    methods = [args.only_method] if args.only_method else list(methods_cfg)
    for dataset in datasets:
        if dataset not in config["datasets"]:
            raise ValueError(f"dataset {dataset} is not present in config")
    for method in methods:
        if method not in methods_cfg:
            raise ValueError(f"method {method} is not present in config")

    if args.dry_run:
        jobs = []
        for dataset in datasets:
            for ratio in config["ratios"]:
                for method in methods:
                    for seed in selection_seeds(methods_cfg[method], config):
                        jobs.append({"dataset": dataset, "ratio": ratio, "method": method, "selection_seed": seed})
        print(json.dumps(jobs, indent=2))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    resolved_name = (
        f"resolved_config_{args.only_dataset}.json"
        if args.only_dataset
        else "resolved_config.json"
    )
    (output_root / resolved_name).write_text(
        json.dumps(json_ready(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    embedding_cache: dict[str, np.ndarray] = {}
    dynamics_cache: dict[str, dict[str, np.ndarray]] = {}
    summary_jobs = []
    selection_cfg = config.get("selection", {})
    geometry_methods = {"facility", "fps", "herding", "graph_a2", "heat_kernel", "graph_coverage"}
    dynamics_methods = {"el2n", "forgetting", "eva"}

    for dataset_name in datasets:
        train_dataset, info = load_dataset(dataset_name, "train", size=224, verbose=True)
        labels = get_labels(train_dataset)
        n_classes = len(np.unique(labels))
        if any(method in geometry_methods for method in methods):
            embedding_data = load_or_compute_embeddings(
                dataset_name=dataset_name,
                split="train",
                source=str(selection_cfg.get("embedding_source", "uni")),
                dataset=train_dataset,
                num_classes=n_classes,
                in_channels=int(info["n_channels"]),
                size=224,
                seed=42,
                cache_dir=embedding_cache_root,
                force_recompute=False,
                verbose=True,
            )
            embedding_cache[dataset_name] = np.asarray(embedding_data["embeddings"])
        if any(method in dynamics_methods for method in methods):
            dynamics_cache[dataset_name] = load_dynamics(
                dataset_name,
                info,
                config.get("dynamics", {}),
                dynamics_cache_root,
            )

        for ratio in [float(value) for value in config["ratios"]]:
            budget = int(int(len(labels) * ratio) // n_classes)
            for method_id in methods:
                method_cfg = methods_cfg[method_id]
                official_method = str(method_cfg["official_method"])
                for selection_seed in selection_seeds(method_cfg, config):
                    run_dir = (
                        output_root
                        / dataset_name
                        / ratio_key(ratio)
                        / method_id
                        / f"seed{selection_seed}"
                    )
                    metrics_path = run_dir / "selection_metrics.json"
                    if metrics_path.exists() and not args.force:
                        print(f"[table1:job1] skip existing {metrics_path}", flush=True)
                        summary_jobs.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                        continue
                    run_dir.mkdir(parents=True, exist_ok=True)
                    started = time.perf_counter()
                    scores = scores_for_method(
                        method_id,
                        dataset_name,
                        ratio,
                        labels,
                        info,
                        dynamics_cache.get(dataset_name),
                        config,
                    )
                    embeddings = embedding_cache.get(dataset_name)
                    overrides = {
                        "k_neighbors": int(method_cfg.get("k_neighbors", selection_cfg.get("k_neighbors", 10))),
                        "global_selection": bool(method_cfg.get("global_selection", selection_cfg.get("global_selection", True))),
                        "sparse_cpu": bool(selection_cfg.get("sparse_cpu", True)),
                    }
                    if "k_hops" in method_cfg or official_method == "graph_a2":
                        overrides["k_hops"] = int(method_cfg.get("k_hops", selection_cfg.get("k_hops", 2)))
                    selected = np.asarray(
                        select(
                            method=official_method,
                            labels=labels,
                            budget_per_class=budget,
                            embeddings=embeddings,
                            el2n_scores=scores["el2n_scores"],
                            eva_scores=scores["eva_scores"],
                            forgetting_scores=scores["forgetting_scores"],
                            importance=np.ones(len(labels), dtype=np.float32),
                            seed=selection_seed,
                            verbose=True,
                            _verbose_level=1,
                            **overrides,
                        ),
                        dtype=np.int64,
                    ).reshape(-1)
                    counts = validate_selected(selected, labels, budget)
                    np.save(run_dir / "selected_indices.npy", selected)
                    np.save(run_dir / "selection_order.npy", selected)
                    np.save(run_dir / "selected_sorted.npy", np.sort(selected))
                    metrics = {
                        "schema": "graphcov-table1/selection-v1",
                        "dataset": dataset_name,
                        "ratio": ratio,
                        "method": method_id,
                        "official_method": official_method,
                        "selection_seed": selection_seed,
                        "n_train": len(labels),
                        "n_classes": n_classes,
                        "budget_per_class": budget,
                        "n_selected": len(selected),
                        "class_counts": counts,
                        "index_sha256": sha256_array(selected),
                        "selection_seconds": time.perf_counter() - started,
                        "embedding_source": selection_cfg.get("embedding_source", "uni") if embeddings is not None else None,
                        "selection_config": {**selection_cfg, **method_cfg},
                        "test_read": False,
                    }
                    metrics_path.write_text(
                        json.dumps(json_ready(metrics), indent=2, sort_keys=True), encoding="utf-8"
                    )
                    (run_dir / "variant_config.json").write_text(
                        json.dumps(json_ready({"method": method_id, **method_cfg}), indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                    summary_jobs.append(metrics)
                    print(
                        f"[table1:job1] {dataset_name} {ratio_key(ratio)} {method_id} "
                        f"selection_seed={selection_seed} n={len(selected)}",
                        flush=True,
                    )
    manifest_name = (
        f"selection_manifest_{args.only_dataset}.json"
        if args.only_dataset
        else "selection_manifest.json"
    )
    (output_root / manifest_name).write_text(
        json.dumps(json_ready({"schema": "graphcov-table1/selection-manifest-v1", "jobs": summary_jobs}), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--only-dataset")
    parser.add_argument("--only-method")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run(args.config.resolve(), args.project_root.resolve(), args)


if __name__ == "__main__":
    raise SystemExit(main())
