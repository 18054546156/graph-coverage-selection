#!/usr/bin/env python3
"""Validate the 3090 server environment and v11 dataset registry."""

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

from v11.experiments.job1_select import resolve_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_ROOT / "v11" / "configs" / "datasets_server_3090.json",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--datasets", nargs="*")
    args = parser.parse_args()
    registry_path = args.registry.resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))["datasets"]
    names = args.datasets or list(registry)

    import scipy
    import sklearn
    import torch
    import torchvision
    import medmnist

    try:
        import faiss
        faiss_status = getattr(faiss, "__version__", "available")
    except ImportError as error:
        raise RuntimeError("FAISS is required by the server configs") from error

    print(
        json.dumps(
            {
                "python": sys.version,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "sklearn": sklearn.__version__,
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "medmnist": medmnist.__version__,
                "faiss": faiss_status,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            indent=2,
        )
    )
    failures = []
    for name in names:
        if name not in registry:
            failures.append(f"unknown dataset: {name}")
            continue
        spec = registry[name]
        embedding_path = resolve_path(spec["embedding"], args.project_root.resolve(), registry_path.parent)
        label_path = resolve_path(spec["label_archive"], args.project_root.resolve(), registry_path.parent)
        if not embedding_path.exists() or not label_path.exists():
            failures.append(f"{name}: missing {embedding_path if not embedding_path.exists() else label_path}")
            continue
        with np.load(embedding_path) as payload:
            embedding_shape = tuple(payload["embeddings"].shape)
        with np.load(label_path) as payload:
            label_shape = tuple(payload[spec.get("label_key", "train_labels")].shape)
        expected = (int(spec["n_train"]), int(spec["embedding_dim"]))
        if embedding_shape != expected or label_shape[0] != int(spec["n_train"]):
            failures.append(
                f"{name}: embedding={embedding_shape}, labels={label_shape}, expected={expected}"
            )
        else:
            print(f"OK {name}: embeddings={embedding_shape}, labels={label_shape}")
    if failures:
        raise RuntimeError("preflight failed:\n" + "\n".join(failures))
    print("v11 server preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
