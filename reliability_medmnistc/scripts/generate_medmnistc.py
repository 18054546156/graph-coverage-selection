"""Generate official MedMNIST-C files through the pinned upstream API.

This wrapper does not implement or modify corruptions.  It calls the clean
medmnistc-api DatasetManager for the five datasets used by this experiment.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path


DATASETS = [
    "organsmnist",
    "organamnist",
    "pathmnist",
    "tissuemnist",
    "bloodmnist",
]


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

    manager = DatasetManager(
        medmnist_path=str(args.medmnist_root),
        output_path=str(args.output_root),
        random_seed=args.random_seed,
    )

    started = time.time()
    records = []
    for dataset in args.datasets:
        print(f"Generating {dataset} with upstream DatasetManager", flush=True)
        manager.create_dataset(dataset)
        dataset_dir = args.output_root / dataset
        expected = sorted(CORRUPTIONS_DS[dataset])
        present = sorted(path.stem for path in dataset_dir.glob("*.npz"))
        if present != expected:
            raise RuntimeError(
                f"Unexpected generated files for {dataset}: "
                f"expected={expected}, present={present}"
            )
        records.append(
            {
                "dataset": dataset,
                "clean_file": str(args.medmnist_root / f"{dataset}_224.npz"),
                "output_dir": str(dataset_dir),
                "corruptions": expected,
                "severity_count": 5,
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
    manifest_path = package_root / "protocol" / "medmnistc_generation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
