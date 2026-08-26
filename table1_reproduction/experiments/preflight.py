#!/usr/bin/env python3
"""Check Table 1 caches and archives before starting a long server run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from table1_reproduction.official_runtime import verify_official_vendor  # noqa: E402


def resolve_path(value: str, project_root: Path, config_dir: Path) -> Path:
    value = value.replace("{project_root}", str(project_root)).replace(
        "{config_dir}", str(config_dir)
    )
    path = Path(os.path.expandvars(value)).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def main() -> int:
    verify_official_vendor()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("table1_reproduction/configs/job1_table1.json"),
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    registry_path = resolve_path(
        config["dataset_registry"], args.project_root.resolve(), config_path.parent
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    cache_root = resolve_path(
        config["embedding_cache_root"], args.project_root.resolve(), config_path.parent
    )
    failures = []
    for dataset in config["datasets"]:
        spec = registry["datasets"][dataset]
        embedding_path = resolve_path(
            spec["embedding"], args.project_root.resolve(), config_path.parent
        )
        archive_path = Path(os.path.expandvars(spec["label_archive"])).expanduser()
        if not embedding_path.exists():
            failures.append(f"missing embedding: {embedding_path}")
            continue
        if not archive_path.exists():
            failures.append(f"missing MedMNIST archive: {archive_path}")
        try:
            with np.load(embedding_path, allow_pickle=False) as data:
                if "embeddings" not in data:
                    failures.append(f"{embedding_path} has no 'embeddings' array")
                    continue
                shape = tuple(data["embeddings"].shape)
                expected = (int(spec["n_train"]), int(spec["embedding_dim"]))
                if shape != expected:
                    failures.append(f"{embedding_path} shape {shape} != {expected}")
                else:
                    print(f"OK {dataset}: embeddings={shape}, archive={archive_path}")
        except Exception as error:
            failures.append(f"cannot read {embedding_path}: {type(error).__name__}: {error}")
    if failures:
        print("Preflight failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"Preflight passed for {len(config['datasets'])} datasets")
    print(f"Embedding cache root: {cache_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
