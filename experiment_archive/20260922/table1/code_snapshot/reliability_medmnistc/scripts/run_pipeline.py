"""Run the reproducible pipeline from a YAML config and explicit phase."""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import os
import runpy

import yaml


PHASES = {"select", "validate", "train", "evaluate", "full", "summarize"}


def _as_bool(value: object) -> str:
    return "1" if bool(value) else "0"


def main(default_phase: str | None = None) -> int:
    package_root = Path(__file__).resolve().parents[1]
    repo_root = package_root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), default=default_phase or "full")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must contain a YAML mapping")

    methods = [str(x) for x in config["methods"]]
    datasets = [str(x) for x in config["datasets"]]
    ratios = [str(x) for x in config["ratios"]]
    env = {
        "PREP_ROOT": str(repo_root),
        "GRAPH_ROOT": str(repo_root),
        "MEDC_ROOT": str(repo_root / "third_party" / "medmnistc"),
        "METHODS": ",".join(methods),
        "DATASETS": ",".join(datasets),
        "RATIOS": ",".join(ratios),
        "SELECTION_SEED": str(config["selection_seed"]),
        "TRAINING_SEEDS": ",".join(str(x) for x in config["training_seeds"]),
        "EPOCHS": str(config["epochs"]),
        "DYNAMICS_EPOCHS": str(config["dynamics_epochs"]),
        "IMAGE_SIZE": str(config["image_size"]),
        "BATCH_SIZE": str(config["batch_size"]),
        "NUM_WORKERS": str(config["num_workers"]),
        "EMBEDDING_SOURCE": str(config["embedding_source"]),
        "AUGMENT": _as_bool(config["augment"]),
        "DOWNLOAD_MEDMNIST": _as_bool(config["download_medmnist"]),
        "CORR_HASH": _as_bool(config["corruption_hash"]),
        "RUN_FULL_TRAIN": "0",
        "PHASE": args.phase,
    }
    method_config = config.get("method_config", {})
    facility_config = method_config.get("facility", {})
    graph_config = method_config.get("graph_a2", {})
    env.update({
        "FACILITY_GLOBAL_SELECTION": _as_bool(facility_config.get("global_selection", False)),
        "FACILITY_EXECUTION_DEVICE": str(facility_config.get("execution_device", "auto")),
        "FACILITY_CPU_MIN_CLASS_SIZE": str(facility_config.get("cpu_min_class_size", 40000)),
        "GRAPH_GLOBAL_SELECTION": _as_bool(graph_config.get("global_selection", True)),
        "GRAPH_K_NEIGHBORS": str(graph_config.get("k_neighbors", 50)),
        "GRAPH_K_HOPS": str(graph_config.get("k_hops", 2)),
    })
    defaults = {
        "MEDMNIST_ROOT": repo_root / "data" / "medmnist",
        "MEDMNISTC_ROOT": repo_root / "data" / "medmnistc",
        "RELIABILITY_OUT": repo_root / "runs" / config["name"],
        "SELECTION_OUT": repo_root / "runs" / config["name"] / "selections",
        "GRAPH_CACHE": repo_root / "cache" / config["name"],
    }
    for key, value in defaults.items():
        env.setdefault(key, str(value))
    for key, value in env.items():
        os.environ.setdefault(key, value)
    pipeline = package_root / "reliability" / "pipeline.py"
    runpy.run_path(str(pipeline), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
