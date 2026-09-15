"""Download and verify official MedMNIST 28- and 224-pixel archives.

This script only downloads clean MedMNIST files. It does not select data,
train a model, or create MedMNIST-C files.
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
import subprocess
import time


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


def archive_fields(info: dict, size: int) -> tuple[str, str]:
    if size == 28:
        return info["url"], info["MD5"]
    if size == 224:
        return info["url_224"], info["MD5_224"]
    raise ValueError(f"unsupported MedMNIST size: {size}")


def archive_filename(dataset: str, size: int) -> str:
    return f"{dataset}.npz" if size == 28 else f"{dataset}_{size}.npz"


def verify_medmnist(root: Path, dataset: str, size: int) -> dict:
    from medmnist import INFO

    info = INFO[dataset]
    filename = archive_filename(dataset, size)
    target = root / filename
    _, expected = archive_fields(info, size)
    actual = md5(target)
    if actual != expected:
        raise RuntimeError(
            f"MD5 mismatch for {target}: expected {expected}, got {actual}"
        )

    dataset_class = getattr(__import__("medmnist", fromlist=[info["python_class"]]), info["python_class"])
    train = dataset_class(split="train", size=size, root=str(root), download=False)
    test = dataset_class(split="test", size=size, root=str(root), download=False)
    train_shape = list(train.imgs.shape)
    test_shape = list(test.imgs.shape)
    if train_shape[1:3] != [size, size] or test_shape[1:3] != [size, size]:
        raise RuntimeError(
            f"unexpected image shape for {dataset}: train={train_shape}, test={test_shape}"
        )
    return {
        "dataset": dataset,
        "size": size,
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
    parser.add_argument("--sizes", nargs="+", type=int, default=[28, 224], choices=[28, 224])
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    try:
        from medmnist import INFO
    except ImportError as exc:
        raise SystemExit("Install medmnist before running this script") from exc

    args.root.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset in args.datasets:
        info = INFO[dataset]
        for size in args.sizes:
            url, expected = archive_fields(info, size)
            target = args.root / archive_filename(dataset, size)
            if target.exists():
                actual = md5(target)
                if actual != expected:
                    raise RuntimeError(
                        f"existing file has wrong MD5: {target}; expected {expected}, got {actual}"
                    )
                print(f"Already verified: {target}")
            elif args.verify_only:
                raise FileNotFoundError(f"required official MedMNIST archive is missing: {target}")
            else:
                download(url, target, args.retries, args.timeout)
            records.append(verify_medmnist(args.root, dataset, size))

    manifest = {
        "kind": "official_medmnist_clean_download",
        "created_unix": time.time(),
        "root": str(args.root.resolve()),
        "datasets": records,
    }
    manifest_path = (args.manifest or (args.root / "download_manifest.json")).resolve()
    if not args.no_manifest:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_name(manifest_path.name + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)
    print(json.dumps(manifest, indent=2))
    if not args.no_manifest:
        print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
