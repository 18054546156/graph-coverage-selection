"""Atomic graph-kernel and constrained facility-location primitives.

The module is self-contained so v11 does not depend on any earlier experimental
folder.  Every kernel design choice is explicit in a JSON configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import diags
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class MarginResult:
    same_distance: np.ndarray
    different_distance: np.ndarray
    ratio: np.ndarray
    gap: np.ndarray
    unsafe: np.ndarray


@dataclass(frozen=True)
class SelectionResult:
    indices: np.ndarray
    order: np.ndarray
    class_counts: dict[int, int]
    unsafe_counts: dict[int, int]


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


def _faiss_index(vectors: np.ndarray, gpu: int | None):
    import faiss

    cpu_index = faiss.IndexFlatIP(vectors.shape[1])
    resources = None
    index = cpu_index
    if gpu is not None and hasattr(faiss, "StandardGpuResources"):
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, int(gpu), cpu_index)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index, resources


def _knn_faiss(
    normalized: np.ndarray,
    k: int,
    gpu: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    search_k = min(len(normalized), k + 8)
    index, resources = _faiss_index(normalized, gpu)
    similarities, candidates = index.search(normalized, search_k)
    del index, resources

    output_indices = np.empty((len(normalized), k), dtype=np.int64)
    output_similarities = np.empty((len(normalized), k), dtype=np.float32)
    for row in range(len(normalized)):
        keep = candidates[row] != row
        row_indices = candidates[row, keep][:k]
        row_similarities = similarities[row, keep][:k]
        if len(row_indices) != k:
            raise RuntimeError(f"could not find {k} non-self neighbors for row {row}")
        output_indices[row] = row_indices
        output_similarities[row] = row_similarities
    squared_distances = np.clip(2.0 * (1.0 - output_similarities), 0.0, 4.0)
    return output_indices, squared_distances.astype(np.float32)


def _knn_sklearn(normalized: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    model.fit(normalized)
    distances, indices = model.kneighbors(normalized)
    return indices[:, 1:].astype(np.int64), np.square(distances[:, 1:]).astype(np.float32)


def build_or_load_knn(
    embeddings: np.ndarray,
    k: int,
    cache_path: Path,
    backend: str = "auto",
    gpu: int | None = 0,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build an exact cosine kNN cache or load a compatible existing cache."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        with np.load(cache_path) as payload:
            indices = np.asarray(payload["indices"], dtype=np.int64)
            distances = np.asarray(payload["distances"], dtype=np.float32)
        if indices.shape[0] != len(embeddings) or indices.shape[1] < k:
            raise ValueError(f"incompatible kNN cache: {cache_path} has {indices.shape}")
        return indices[:, :k], distances[:, :k], {
            "backend": "cache",
            "path": str(cache_path),
            "k": int(k),
        }

    if k < 1 or k >= len(embeddings):
        raise ValueError(f"invalid k={k} for n={len(embeddings)}")
    normalized = l2_normalize(embeddings)
    used_backend = backend
    if backend in {"auto", "faiss"}:
        try:
            indices, distances = _knn_faiss(normalized, k, gpu)
            used_backend = "faiss_gpu" if gpu is not None else "faiss_cpu"
        except (ImportError, AttributeError, RuntimeError) as error:
            if backend == "faiss":
                raise
            indices, distances = _knn_sklearn(normalized, k)
            used_backend = f"sklearn_fallback:{type(error).__name__}"
    elif backend == "sklearn":
        indices, distances = _knn_sklearn(normalized, k)
    else:
        raise ValueError(f"unknown kNN backend: {backend}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, indices=indices, distances=distances)
    return indices, distances, {
        "backend": used_backend,
        "path": str(cache_path),
        "k": int(k),
    }


def build_weighted_adjacency(
    knn_indices: np.ndarray,
    knn_squared_distances: np.ndarray,
    n_samples: int,
) -> sparse.csr_matrix:
    indices = np.asarray(knn_indices, dtype=np.int64)
    distances = np.asarray(knn_squared_distances, dtype=np.float32)
    if indices.shape != distances.shape or indices.shape[0] != n_samples:
        raise ValueError("invalid kNN shapes")
    rows = np.repeat(np.arange(n_samples, dtype=np.int64), indices.shape[1])
    weights = np.clip(1.0 - distances.ravel() / 2.0, 0.0, 1.0).astype(np.float32)
    adjacency = sparse.csr_matrix(
        (weights, (rows, indices.ravel())),
        shape=(n_samples, n_samples),
        dtype=np.float32,
    )
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.eliminate_zeros()
    return adjacency


def symmetric_normalize(adjacency: sparse.spmatrix) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(adjacency, dtype=np.float32)
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    degree[degree <= 0] = 1.0
    inverse = diags(1.0 / np.sqrt(degree))
    result = (inverse @ matrix @ inverse).tocsr().astype(np.float32)
    result.eliminate_zeros()
    return result


def truncate_sparse_rows(matrix: sparse.spmatrix, max_degree: int) -> sparse.csr_matrix:
    if max_degree < 1:
        raise ValueError("max_degree must be positive")
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        row_columns = matrix.indices[start:end]
        row_values = matrix.data[start:end]
        if not len(row_values):
            continue
        order = np.lexsort((row_columns, -row_values))[:max_degree]
        row_parts.append(np.full(len(order), row, dtype=np.int64))
        col_parts.append(row_columns[order].astype(np.int64, copy=False))
        data_parts.append(row_values[order].astype(np.float32, copy=False))
    if not row_parts:
        return sparse.csr_matrix(matrix.shape, dtype=np.float32)
    result = sparse.csr_matrix(
        (
            np.concatenate(data_parts),
            (np.concatenate(row_parts), np.concatenate(col_parts)),
        ),
        shape=matrix.shape,
        dtype=np.float32,
    )
    result.eliminate_zeros()
    return result


def _kernel_weight(hop: int, mode: str, hop_decay: float) -> float:
    if mode == "uniform":
        return 1.0
    if mode == "geometric":
        if not 0.0 < hop_decay <= 1.0:
            raise ValueError("hop_decay must lie in (0, 1]")
        return float(hop_decay ** (hop - 1))
    raise ValueError(f"unknown weight mode: {mode}")


def build_diffusion_kernel(
    normalized: sparse.csr_matrix,
    *,
    hops: int,
    weight_mode: str = "uniform",
    hop_decay: float = 1.0,
    truncation: str = "none",
    max_degree: int | None = None,
    symmetrize_after: bool = False,
) -> tuple[sparse.csr_matrix, dict[str, object]]:
    """Build a polynomial graph kernel with independently controlled choices.

    ``per_hop`` reproduces the bounded v10 behavior: both the current power and
    the accumulated kernel are truncated after every hop. ``final`` truncates
    only the completed polynomial and therefore isolates final sparsification.
    """
    if hops < 1:
        raise ValueError("hops must be positive")
    if truncation not in {"none", "final", "per_hop"}:
        raise ValueError(f"unknown truncation mode: {truncation}")
    if truncation != "none" and max_degree is None:
        raise ValueError("a truncated kernel requires max_degree")

    transition = sparse.csr_matrix(normalized, dtype=np.float32)
    power = transition.copy()
    result = sparse.csr_matrix(transition.shape, dtype=np.float32)
    hop_diagnostics: list[dict[str, object]] = []
    for hop in range(1, hops + 1):
        if hop > 1:
            power = (power @ transition).tocsr().astype(np.float32)
        raw_power_nnz = int(power.nnz)
        if truncation == "per_hop":
            power = truncate_sparse_rows(power, int(max_degree))
        weight = _kernel_weight(hop, weight_mode, hop_decay)
        result = (result + weight * power).tocsr().astype(np.float32)
        if truncation == "per_hop":
            result = truncate_sparse_rows(result, int(max_degree))
        hop_diagnostics.append(
            {
                "hop": hop,
                "weight": weight,
                "raw_power_nnz": raw_power_nnz,
                "retained_power_nnz": int(power.nnz),
                "accumulated_nnz": int(result.nnz),
            }
        )
    if truncation == "final":
        result = truncate_sparse_rows(result, int(max_degree))
    if symmetrize_after:
        result = result.maximum(result.T).tocsr().astype(np.float32)
    result.eliminate_zeros()
    return result, {
        "hops": int(hops),
        "weight_mode": weight_mode,
        "hop_decay": float(hop_decay),
        "truncation": truncation,
        "max_degree": int(max_degree) if max_degree is not None else None,
        "symmetrize_after": bool(symmetrize_after),
        "hop_diagnostics": hop_diagnostics,
    }


def _exact_margins_numpy(normalized: np.ndarray, labels: np.ndarray, batch_size: int) -> MarginResult:
    n_samples = len(labels)
    same_distance = np.empty(n_samples, dtype=np.float32)
    different_distance = np.empty(n_samples, dtype=np.float32)
    all_indices = np.arange(n_samples)
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        similarities = normalized[start:stop] @ normalized.T
        rows = np.arange(stop - start)
        queries = all_indices[start:stop]
        similarities[rows, queries] = -np.inf
        same = labels[None, :] == labels[queries, None]
        same_distance[start:stop] = 1.0 - np.max(
            np.where(same, similarities, -np.inf), axis=1
        )
        different_distance[start:stop] = 1.0 - np.max(
            np.where(~same, similarities, -np.inf), axis=1
        )
    return _finish_margins(same_distance, different_distance)


def _exact_margins_faiss(
    normalized: np.ndarray,
    labels: np.ndarray,
    gpu: int | None,
) -> MarginResult:
    same_distance = np.empty(len(labels), dtype=np.float32)
    different_distance = np.empty(len(labels), dtype=np.float32)
    for cls in np.unique(labels):
        class_indices = np.flatnonzero(labels == cls)
        other_indices = np.flatnonzero(labels != cls)
        if len(class_indices) < 2:
            raise ValueError(f"class {cls} has fewer than two samples")

        same_vectors = normalized[class_indices]
        same_index, same_resources = _faiss_index(same_vectors, gpu)
        search_k = min(8, len(class_indices))
        same_similarities, same_neighbors = same_index.search(same_vectors, search_k)
        for local_row, global_row in enumerate(class_indices):
            valid = same_neighbors[local_row] != local_row
            if not np.any(valid):
                raise RuntimeError(f"no non-self same-class neighbor for sample {global_row}")
            same_distance[global_row] = 1.0 - same_similarities[local_row, valid][0]
        del same_index, same_resources

        different_vectors = normalized[other_indices]
        different_index, different_resources = _faiss_index(different_vectors, gpu)
        different_similarities, _ = different_index.search(same_vectors, 1)
        different_distance[class_indices] = 1.0 - different_similarities[:, 0]
        del different_index, different_resources
    return _finish_margins(same_distance, different_distance)


def _finish_margins(same_distance: np.ndarray, different_distance: np.ndarray) -> MarginResult:
    same_distance = np.maximum(same_distance, 0.0).astype(np.float32)
    different_distance = np.maximum(different_distance, 0.0).astype(np.float32)
    eps = np.finfo(np.float32).eps
    ratio = different_distance / np.maximum(same_distance, eps)
    gap = different_distance - same_distance
    unsafe = gap <= 0.0
    return MarginResult(same_distance, different_distance, ratio, gap, unsafe)


def compute_exact_margins(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    backend: str = "auto",
    gpu: int | None = 0,
    batch_size: int = 512,
) -> tuple[MarginResult, str]:
    normalized = l2_normalize(embeddings)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(normalized) != len(labels):
        raise ValueError("embedding and label counts differ")
    if backend in {"auto", "faiss"}:
        try:
            return _exact_margins_faiss(normalized, labels, gpu), "faiss_exact"
        except (ImportError, AttributeError, RuntimeError) as error:
            if backend == "faiss" or len(labels) > 20_000:
                raise RuntimeError("FAISS exact margins are required for this dataset") from error
            return _exact_margins_numpy(normalized, labels, batch_size), "numpy_exact_fallback"
    if backend == "numpy":
        return _exact_margins_numpy(normalized, labels, batch_size), "numpy_exact"
    raise ValueError(f"unknown margin backend: {backend}")


def _validate_caps(
    labels: np.ndarray,
    unsafe: np.ndarray,
    budget_per_class: int,
    unsafe_caps: Mapping[int, int] | None,
) -> dict[int, int] | None:
    if unsafe_caps is None:
        return None
    caps = {int(cls): int(unsafe_caps[int(cls)]) for cls in np.unique(labels)}
    for cls, cap in caps.items():
        if cap < 0 or cap > budget_per_class:
            raise ValueError(f"invalid unsafe cap for class {cls}: {cap}")
        safe_count = int(np.sum((labels == cls) & ~unsafe))
        if safe_count + cap < budget_per_class:
            raise ValueError(f"unsafe cap for class {cls} is infeasible")
    return caps


def greedy_facility(
    kernel: sparse.spmatrix,
    labels: np.ndarray,
    budget_per_class: int,
    *,
    unsafe: np.ndarray | None = None,
    unsafe_caps: Mapping[int, int] | None = None,
) -> SelectionResult:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    unsafe = np.zeros(len(labels), dtype=bool) if unsafe is None else np.asarray(unsafe, dtype=bool)
    caps = _validate_caps(labels, unsafe, budget_per_class, unsafe_caps)
    classes = np.unique(labels)
    total_budget = int(budget_per_class * len(classes))
    matrix = sparse.csc_matrix(kernel, dtype=np.float32)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("kernel and labels differ in size")

    rows = matrix.indices
    indptr = matrix.indptr
    values = np.maximum(matrix.data, 0.0)
    columns = np.repeat(np.arange(len(labels), dtype=np.int64), np.diff(indptr))
    initial = np.bincount(columns, weights=values, minlength=len(labels))
    heap = [(-float(initial[index]), int(index)) for index in range(len(labels))]
    heapq.heapify(heap)
    last_evaluated = np.zeros(len(labels), dtype=np.int64)
    eligible = np.ones(len(labels), dtype=bool)
    covered = np.zeros(len(labels), dtype=np.float32)
    class_counts = {int(cls): 0 for cls in classes}
    unsafe_counts = {int(cls): 0 for cls in classes}
    selected: list[int] = []

    for iteration in range(total_budget):
        chosen = -1
        while heap:
            _, candidate = heapq.heappop(heap)
            if not eligible[candidate]:
                continue
            cls = int(labels[candidate])
            if caps is not None and unsafe[candidate] and unsafe_counts[cls] >= caps[cls]:
                eligible[candidate] = False
                continue
            if last_evaluated[candidate] == iteration:
                chosen = candidate
                break
            start, end = indptr[candidate], indptr[candidate + 1]
            candidate_rows = rows[start:end]
            gain = float(np.maximum(values[start:end] - covered[candidate_rows], 0.0).sum())
            last_evaluated[candidate] = iteration
            heapq.heappush(heap, (-gain, candidate))
        if chosen < 0:
            raise RuntimeError(f"constraints exhausted candidates after {iteration} selections")
        selected.append(chosen)
        cls = int(labels[chosen])
        class_counts[cls] += 1
        unsafe_counts[cls] += int(unsafe[chosen])
        eligible[chosen] = False
        if class_counts[cls] == budget_per_class:
            eligible[labels == cls] = False
        start, end = indptr[chosen], indptr[chosen + 1]
        chosen_rows = rows[start:end]
        covered[chosen_rows] = np.maximum(covered[chosen_rows], values[start:end])

    order = np.asarray(selected, dtype=np.int64)
    if len(np.unique(order)) != total_budget:
        raise AssertionError("selection contains duplicate indices")
    if any(count != budget_per_class for count in class_counts.values()):
        raise AssertionError(f"class quota violation: {class_counts}")
    if caps is not None and any(unsafe_counts[c] > caps[c] for c in classes):
        raise AssertionError(f"unsafe cap violation: {unsafe_counts}")
    return SelectionResult(np.sort(order), order, class_counts, unsafe_counts)


def _coverage_vector(kernel: sparse.spmatrix, selected: Sequence[int]) -> np.ndarray:
    selected = np.asarray(selected, dtype=np.int64)
    if not len(selected):
        return np.zeros(kernel.shape[0], dtype=np.float32)
    values = sparse.csr_matrix(kernel)[:, selected].max(axis=1)
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values).reshape(-1).astype(np.float32)


def _candidate_gain(matrix: sparse.csc_matrix, candidate: int, covered: np.ndarray) -> float:
    start, end = matrix.indptr[candidate], matrix.indptr[candidate + 1]
    rows = matrix.indices[start:end]
    values = matrix.data[start:end]
    return float(np.maximum(values - covered[rows], 0.0).sum())


def posthoc_margin_repair(
    kernel: sparse.spmatrix,
    labels: np.ndarray,
    base: SelectionResult,
    unsafe: np.ndarray,
    unsafe_caps: Mapping[int, int],
) -> SelectionResult:
    """Meet unsafe caps with the fewest deterministic swaps.

    The latest-selected violating sample is removed, then the best safe
    same-class replacement is inserted. This isolates a minimal repair from the
    path-dependent in-greedy hard constraint.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    unsafe = np.asarray(unsafe, dtype=bool)
    caps = {int(key): int(value) for key, value in unsafe_caps.items()}
    selected = list(map(int, base.order))
    selected_set = set(selected)
    matrix = sparse.csc_matrix(kernel, dtype=np.float32)
    for cls in np.unique(labels):
        cls = int(cls)
        while sum(int(unsafe[index]) for index in selected if labels[index] == cls) > caps[cls]:
            removable = [index for index in selected if labels[index] == cls and unsafe[index]]
            remove = removable[-1]
            selected.remove(remove)
            selected_set.remove(remove)
            covered = _coverage_vector(matrix, selected)
            candidates = np.flatnonzero((labels == cls) & ~unsafe)
            candidates = [int(index) for index in candidates if int(index) not in selected_set]
            if not candidates:
                raise RuntimeError(f"no safe replacement available for class {cls}")
            add = max(candidates, key=lambda index: (_candidate_gain(matrix, index, covered), -index))
            selected.append(add)
            selected_set.add(add)
    order = np.asarray(selected, dtype=np.int64)
    class_counts = {int(cls): int(np.sum(labels[order] == cls)) for cls in np.unique(labels)}
    unsafe_counts = {
        int(cls): int(np.sum(unsafe[order[labels[order] == cls]])) for cls in np.unique(labels)
    }
    return SelectionResult(np.sort(order), order, class_counts, unsafe_counts)


def coverage_metrics(
    kernel: sparse.spmatrix,
    labels: np.ndarray,
    selected: Sequence[int],
) -> dict[str, object]:
    covered = _coverage_vector(kernel, selected)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    means = {int(cls): float(np.mean(covered[labels == cls])) for cls in np.unique(labels)}
    p10 = {int(cls): float(np.quantile(covered[labels == cls], 0.1)) for cls in np.unique(labels)}
    return {
        "mean": float(np.mean(covered)),
        "p10": float(np.quantile(covered, 0.1)),
        "macro_class_mean": float(np.mean(list(means.values()))),
        "worst_class_mean": float(min(means.values())),
        "worst_class_p10": float(min(p10.values())),
        "per_class_mean": means,
        "per_class_p10": p10,
    }


def selection_margin_metrics(
    margins: MarginResult,
    labels: np.ndarray,
    selected: Sequence[int],
) -> dict[str, object]:
    selected = np.asarray(selected, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    per_class = {}
    for cls in np.unique(labels):
        subset = selected[labels[selected] == cls]
        per_class[int(cls)] = {
            "unsafe_count": int(np.sum(margins.unsafe[subset])),
            "unsafe_ratio": float(np.mean(margins.unsafe[subset])),
            "ratio_mean": float(np.mean(margins.ratio[subset])),
            "gap_mean": float(np.mean(margins.gap[subset])),
        }
    return {
        "unsafe_count": int(np.sum(margins.unsafe[selected])),
        "unsafe_ratio": float(np.mean(margins.unsafe[selected])),
        "ratio_mean": float(np.mean(margins.ratio[selected])),
        "ratio_p10": float(np.quantile(margins.ratio[selected], 0.1)),
        "gap_mean": float(np.mean(margins.gap[selected])),
        "gap_p10": float(np.quantile(margins.gap[selected], 0.1)),
        "per_class": per_class,
    }


def knn_purity(knn_indices: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    purity = np.mean(labels[knn_indices] == labels[:, None], axis=1)
    per_class = {int(cls): float(np.mean(purity[labels == cls])) for cls in np.unique(labels)}
    return {
        "mean": float(np.mean(purity)),
        "macro": float(np.mean(list(per_class.values()))),
        "worst_class": float(min(per_class.values())),
        "p10": float(np.quantile(purity, 0.1)),
        "per_class": per_class,
    }


def kernel_metrics(
    kernel: sparse.spmatrix,
    labels: np.ndarray,
    *,
    sample_rows: int = 2048,
) -> dict[str, object]:
    matrix = sparse.csr_matrix(kernel, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    row_nnz = np.diff(matrix.indptr)
    cross_count = 0
    cross_mass = 0.0
    total_mass = float(np.sum(np.abs(matrix.data)))
    for start in range(0, len(labels), 4096):
        stop = min(start + 4096, len(labels))
        block = matrix[start:stop]
        block_rows = np.repeat(np.arange(start, stop), np.diff(block.indptr))
        cross = labels[block_rows] != labels[block.indices]
        cross_count += int(np.sum(cross))
        cross_mass += float(np.sum(np.abs(block.data[cross])))
    row_ids = np.linspace(0, len(labels) - 1, min(sample_rows, len(labels)), dtype=np.int64)
    left = matrix[row_ids]
    right = matrix.T.tocsr()[row_ids]
    difference_mass = float(np.sum(np.abs((left - right).data)))
    sampled_mass = float(np.sum(np.abs(left.data)) + np.sum(np.abs(right.data)))
    return {
        "nnz": int(matrix.nnz),
        "row_nnz_mean": float(np.mean(row_nnz)),
        "row_nnz_p50": float(np.quantile(row_nnz, 0.5)),
        "row_nnz_p90": float(np.quantile(row_nnz, 0.9)),
        "row_nnz_max": int(np.max(row_nnz)),
        "cross_class_edge_ratio": float(cross_count / max(1, matrix.nnz)),
        "cross_class_mass_ratio": float(cross_mass / max(np.finfo(float).eps, total_mass)),
        "sampled_relative_asymmetry": float(
            difference_mass / max(np.finfo(float).eps, sampled_mass)
        ),
    }


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
