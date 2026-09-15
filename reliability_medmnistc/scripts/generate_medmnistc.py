"""Generate official MedMNIST-C files through the pinned upstream API.

This wrapper does not implement or modify corruptions.  It calls the clean
medmnistc-api DatasetManager for the five datasets used by this experiment.
"""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
import os
import platform
import time

import numpy as np


DATASETS = [
    "organsmnist",
    "organamnist",
    "pathmnist",
    "tissuemnist",
    "bloodmnist",
]


def array_sha256(values) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_dataset(directory: Path, expected: list[str], clean_labels: np.ndarray) -> list[dict]:
    present = sorted(path.stem for path in directory.glob("*.npz")) if directory.is_dir() else []
    if present != expected:
        raise RuntimeError(f"Unexpected files in {directory}: expected={expected}, present={present}")
    records = []
    expected_labels = np.tile(clean_labels, 5)
    for corruption in expected:
        path = directory / f"{corruption}.npz"
        with np.load(path, allow_pickle=False) as package:
            if not {"test_images", "test_labels"}.issubset(package.files):
                raise RuntimeError(f"Missing test arrays in {path}: {package.files}")
            images = package["test_images"]
            labels = np.asarray(package["test_labels"]).reshape(-1).astype(np.int64)
            if len(images) != len(expected_labels) or not np.array_equal(labels, expected_labels):
                raise RuntimeError(f"Label/severity alignment failure in {path}")
            if images.dtype != np.uint8:
                raise RuntimeError(f"Expected uint8 images in {path}, got {images.dtype}")
            records.append({
                "corruption": corruption,
                "path": str(path),
                "shape": list(images.shape),
                "dtype": str(images.dtype),
                "labels_sha256": array_sha256(labels),
                "bytes": path.stat().st_size,
            })
    return records


def main() -> int:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--medmnist-root",
        type=Path,
        default=package_root / "data" / "medmnist",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=package_root / "data" / "medmnistc",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=package_root.parent / "third_party" / "medmnistc",
        help="pinned upstream medmnistc-api checkout",
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    args.source_root = args.source_root.resolve()
    if not (args.source_root / "medmnistc" / "dataset_manager.py").is_file():
        raise FileNotFoundError(f"Missing upstream API checkout: {args.source_root}")
    sys.path.insert(0, str(args.source_root))

    try:
        from medmnist import INFO
        from medmnistc.corruptions.registry import CORRUPTIONS_DS
        from medmnistc.dataset_manager import DatasetManager
    except ImportError as exc:
        raise SystemExit(
            "Run this script with the pinned medmnistc-py311 environment"
        ) from exc

    args.medmnist_root = args.medmnist_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        clean_file = args.medmnist_root / f"{dataset}_224.npz"
        if not clean_file.is_file():
            raise FileNotFoundError(f"Missing verified clean file: {clean_file}")

    started = time.time()
    records = []
    for dataset in args.datasets:
        with np.load(args.medmnist_root / f"{dataset}_224.npz", allow_pickle=False) as clean_package:
            clean_labels = np.asarray(clean_package["test_labels"]).reshape(-1).astype(np.int64)
        dataset_dir = args.output_root / dataset
        expected = sorted(CORRUPTIONS_DS[dataset])
        try:
            corruption_records = validate_dataset(dataset_dir, expected, clean_labels)
            print(f"Verified existing atomic dataset: {dataset_dir}", flush=True)
        except (RuntimeError, OSError, ValueError, KeyError):
            staging_root = args.output_root / f".staging_{dataset}_{os.getpid()}"
            staging_root.mkdir(parents=True, exist_ok=False)
            print(f"Generating {dataset} through upstream DatasetManager into {staging_root}", flush=True)
            manager = DatasetManager(
                medmnist_path=str(args.medmnist_root),
                output_path=str(staging_root),
                random_seed=args.random_seed,
            )
            manager.create_dataset(dataset)
            staged_dataset = staging_root / dataset
            validate_dataset(staged_dataset, expected, clean_labels)
            if dataset_dir.exists():
                quarantine = args.output_root / f"{dataset}.incomplete.{int(time.time())}"
                os.replace(dataset_dir, quarantine)
                print(f"Moved incomplete output to {quarantine}", flush=True)
            os.replace(staged_dataset, dataset_dir)
            staging_root.rmdir()
            corruption_records = validate_dataset(dataset_dir, expected, clean_labels)
        records.append(
            {
                "dataset": dataset,
                "clean_file": str(args.medmnist_root / f"{dataset}_224.npz"),
                "output_dir": str(dataset_dir),
                "corruptions": expected,
                "files": corruption_records,
                "severity_count": 5,
                "clean_test_labels_sha256": array_sha256(clean_labels),
            }
        )

    manifest = {
        "kind": "official_medmnistc_generation",
        "api_source": "medmnistc-api",
        "api_commit": "8acfd2710c6e0e8b2745be8b1fa1c17b94b8f8a7",
        "random_seed": args.random_seed,
        "medmnist_root": str(args.medmnist_root),
        "output_root": str(args.output_root),
        "source_root": str(args.source_root),
        "datasets": records,
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = (args.manifest or (args.output_root / "generation_manifest.json")).resolve()
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
