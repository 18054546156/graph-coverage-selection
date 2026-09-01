#!/usr/bin/env python3
"""Print SHA256 and class counts for selected-index NumPy files."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("indices", type=Path, nargs="+")
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    labels = np.asarray(np.load(args.labels)).reshape(-1) if args.labels else None
    for path in args.indices:
        indices = np.ascontiguousarray(np.load(path), dtype=np.int64)
        digest = hashlib.sha256(indices.tobytes()).hexdigest()
        message = f"{path}: n={len(indices)} sha256={digest}"
        if labels is not None:
            classes, counts = np.unique(labels[indices], return_counts=True)
            message += f" class_counts={dict(zip(classes.tolist(), counts.tolist()))}"
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

