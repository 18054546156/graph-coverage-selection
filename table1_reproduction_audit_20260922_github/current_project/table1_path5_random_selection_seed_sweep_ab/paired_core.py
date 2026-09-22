"""Pure helpers shared by input freezing, training, and aggregation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np


def index_sha256(indices: np.ndarray) -> str:
    """Use the Table 1 artifact hash: dtype/shape header plus int64 bytes."""
    normalized = np.ascontiguousarray(np.asarray(indices, dtype=np.int64).reshape(-1))
    digest = hashlib.sha256()
    digest.update(f"{normalized.dtype}|{normalized.shape}".encode("ascii"))
    digest.update(memoryview(normalized).cast("B"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_indices(
    indices: Iterable[int] | np.ndarray,
    *,
    n_train: int,
    n_selected: int,
) -> np.ndarray:
    values = np.asarray(indices)
    if values.ndim != 1:
        raise ValueError(f"selected indices must be one-dimensional, got {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"selected indices must have integer dtype, got {values.dtype}")
    values = np.asarray(values, dtype=np.int64)
    if len(values) != n_selected:
        raise ValueError(f"expected {n_selected} selected indices, got {len(values)}")
    if len(np.unique(values)) != len(values):
        raise ValueError("selected indices contain duplicates")
    if values.min(initial=0) < 0 or values.max(initial=-1) >= n_train:
        raise ValueError(f"selected index outside [0, {n_train})")
    return np.sort(values)


def validate_equal_quota(
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    n_classes: int,
    budget_per_class: int,
) -> dict[str, int]:
    flat_labels = np.asarray(labels).reshape(-1).astype(np.int64, copy=False)
    selected_labels = flat_labels[np.asarray(indices, dtype=np.int64)]
    counts = np.bincount(selected_labels, minlength=n_classes)
    if len(counts) != n_classes or np.any(counts != budget_per_class):
        raise ValueError(
            f"equal quota failed: expected {budget_per_class} per class, got {counts.tolist()}"
        )
    return {str(class_id): int(count) for class_id, count in enumerate(counts)}
