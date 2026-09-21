"""Download and verify official MedMNIST 224-pixel archives.

This script only downloads clean MedMNIST files. It does not select data,
train a model, or create MedMNIST-C files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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


def md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, retries: int, timeout: int) -> None:
    part = target.with_name(target.name + ".part")
    command = [
        "curl",
        "-fL",
        "--retry",
        str(retries),
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--connect-timeout",
        str(timeout),
        "--max-time",
        "0",
        "-C",
        "-",
        "-o",
        str(part),
        url,
    ]
    print(f"Downloading {url}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"curl failed with exit code {result.returncode}; partial file kept at {part}"
        )
    if not part.exists() or part.stat().st_size == 0:
        raise RuntimeError(f"download produced no data: {part}")
    os.replace(part, target)


def verify_medmnist(root: Path, dataset: str) -> dict:
    from medmnist import INFO

    info = INFO[dataset]
    filename = f"{dataset}_224.npz"
    target = root / filename
    expected = info["MD5_224"]
    actual = md5(target)
    if actual != expected:
        raise RuntimeError(
            f"MD5 mismatch for {target}: expected {expected}, got {actual}"
        )

    dataset_class = getattr(__import__("medmnist", fromlist=[info["python_class"]]), info["python_class"])
    train = dataset_class(split="train", size=224, root=str(root), download=False)
    test = dataset_class(split="test", size=224, root=str(root), download=False)
    train_shape = list(train.imgs.shape)
    test_shape = list(test.imgs.shape)
    if train_shape[1:3] != [224, 224] or test_shape[1:3] != [224, 224]:
        raise RuntimeError(
            f"unexpected image shape for {dataset}: train={train_shape}, test={test_shape}"
        )
    return {
        "dataset": dataset,
        "filename": filename,
        "path": str(target),
        "md5": actual,
        "train_shape": train_shape,
        "test_shape": test_shape,
        "n_train": len(train),
        "n_test": len(test),
    }


def main() -> int:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=package_root / "data" / "medmnist",
        help="clean MedMNIST output directory",
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    try:
        from medmnist import INFO
    except ImportError as exc:
        raise SystemExit("Install medmnist before running this script") from exc

    args.root.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset in args.datasets:
        info = INFO[dataset]
        target = args.root / f"{dataset}_224.npz"
        expected = info["MD5_224"]
        if target.exists():
            actual = md5(target)
            if actual != expected:
                raise RuntimeError(
                    f"existing file has wrong MD5: {target}; expected {expected}, got {actual}"
                )
            print(f"Already verified: {target}")
        else:
            download(info["url_224"], target, args.retries, args.timeout)
        records.append(verify_medmnist(args.root, dataset))

    manifest = {
        "kind": "official_medmnist_224_download",
        "created_unix": time.time(),
        "root": str(args.root.resolve()),
        "datasets": records,
    }
    manifest_path = package_root / "protocol" / "medmnist_download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
