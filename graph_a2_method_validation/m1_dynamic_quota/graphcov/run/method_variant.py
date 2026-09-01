"""M1: Graph-A2 with safe, normalized dynamic class quotas."""

from __future__ import annotations

import heapq
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags

from .graph import build_adjacency_matrix, build_knn_graph
from .selection import _greedy_facility_global, register_method


def build_graph_a2_kernel(
    embeddings: np.ndarray,
    k_neighbors: int,
    k_hops: int,
    verbose: bool = False,
) -> sp.csr_matrix:
    """Build the official global sum(A_sym^h) kernel."""
    if k_hops < 1:
        raise ValueError("k_hops must be at least 1")
    n = len(embeddings)
    knn_indices, knn_distances = build_knn_graph(
        embeddings, k_neighbors, verbose=verbose
    )
    adjacency = build_adjacency_matrix(knn_indices, knn_distances, n)
    row_sums = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    row_sums[row_sums == 0] = 1.0
    scale = diags(1.0 / np.sqrt(row_sums))
    normalized = (scale @ adjacency @ scale).tocsr()
    kernel = normalized.copy()
    power = normalized.copy()
    for _ in range(k_hops - 1):
        power = power @ normalized
        kernel = kernel + power
    return kernel.tocsr()


def _class_marginal_curve(
    kernel: sp.spmatrix,
    labels: np.ndarray,
    importance: np.ndarray,
    cls: int,
    max_quota: int,
) -> Tuple[List[float], List[float]]:
    """Compute cumulative and marginal class-local coverage curves."""
    indices = np.flatnonzero(labels == cls)
    local = kernel[indices][:, indices].tocsc().astype(np.float32)
    weights = np.asarray(importance[indices], dtype=np.float32)
    if float(weights.sum()) <= 0:
        weights = np.ones(len(indices), dtype=np.float32)

    data, rows, indptr = local.data, local.indices, local.indptr
    columns = np.repeat(np.arange(local.shape[1], dtype=np.intp), np.diff(indptr))
    covered = np.zeros(local.shape[0], dtype=np.float32)
    available = np.ones(local.shape[1], dtype=bool)
    initial = np.maximum(data, 0.0) * weights[rows]
    gains = np.bincount(columns, weights=initial, minlength=local.shape[1])
    heap = [(-float(gains[j]), int(j)) for j in range(local.shape[1])]
    heapq.heapify(heap)
    last_eval = np.zeros(local.shape[1], dtype=np.int64)

    marginals: List[float] = []
    cumulative: List[float] = []
    for step in range(min(int(max_quota), len(indices))):
        best = -1
        best_gain = 0.0
        while heap:
            neg_gain, candidate = heapq.heappop(heap)
            if not available[candidate]:
                continue
            if last_eval[candidate] == step:
                best, best_gain = candidate, -float(neg_gain)
                break
            start, end = indptr[candidate], indptr[candidate + 1]
            touched = rows[start:end]
            gain = float(
                np.dot(weights[touched], np.maximum(data[start:end] - covered[touched], 0.0))
            )
            last_eval[candidate] = step
            heapq.heappush(heap, (-gain, candidate))
        if best < 0:
            break
        available[best] = False
        start, end = indptr[best], indptr[best + 1]
        touched = rows[start:end]
        covered[touched] = np.maximum(covered[touched], data[start:end])
        marginals.append(best_gain)
        cumulative.append(float(np.dot(weights, covered)))
    return cumulative, marginals


def allocate_dynamic_quotas(
    kernel: sp.spmatrix,
    labels: np.ndarray,
    importance: np.ndarray,
    budget_per_class: int,
    floor_ratio: float = 0.75,
    normalization: str = "self_optimum",
) -> Tuple[Dict[int, int], Dict[int, Dict[str, List[float]]]]:
    """Allocate the official total budget with a guaranteed per-class floor."""
    if not 0.0 <= floor_ratio <= 1.0:
        raise ValueError("floor_ratio must be in [0, 1]")
    if normalization not in {"mass", "self_optimum"}:
        raise ValueError("normalization must be 'mass' or 'self_optimum'")

    classes, counts = np.unique(labels, return_counts=True)
    total = int(budget_per_class) * len(classes)
    floor = max(1, int(np.floor(float(floor_ratio) * budget_per_class)))
    if total > int(counts.sum()) or np.any(counts < floor):
        raise ValueError("requested budget or quota floor is infeasible")

    quotas = {int(cls): floor for cls in classes}
    maximum = {
        int(cls): min(int(count), total - floor * (len(classes) - 1))
        for cls, count in zip(classes, counts)
    }
    curves: Dict[int, Dict[str, List[float]]] = {}
    for cls in classes:
        cls = int(cls)
        cumulative, marginal = _class_marginal_curve(
            kernel, labels, importance, cls, maximum[cls]
        )
        class_mass = float(np.asarray(importance)[labels == cls].sum())
        if class_mass <= 0:
            class_mass = float(np.sum(labels == cls))
        if normalization == "self_optimum":
            ref_index = min(int(budget_per_class), len(cumulative)) - 1
            normalizer = cumulative[ref_index] if ref_index >= 0 else 0.0
        else:
            normalizer = class_mass
        normalizer = max(float(normalizer), np.finfo(np.float64).eps)
        curves[cls] = {
            "cumulative": cumulative,
            "marginal": marginal,
            "normalized_marginal": [float(value / normalizer) for value in marginal],
        }

    while sum(quotas.values()) < total:
        candidates = []
        for cls in classes:
            cls = int(cls)
            next_index = quotas[cls]
            curve = curves[cls]["normalized_marginal"]
            if quotas[cls] < maximum[cls] and next_index < len(curve):
                candidates.append((float(curve[next_index]), -cls, cls))
        if not candidates:
            raise RuntimeError("dynamic quota curves cannot satisfy the total budget")
        quotas[max(candidates)[2]] += 1
    return quotas, curves


def greedy_variable_quota(
    kernel: sp.spmatrix,
    labels: np.ndarray,
    quotas: Dict[int, int],
    importance: np.ndarray,
) -> List[int]:
    """Official sparse lazy greedy with a class-specific quota map."""
    matrix = sp.csc_matrix(kernel, dtype=np.float32)
    data, rows, indptr = matrix.data, matrix.indices, matrix.indptr
    columns = np.repeat(np.arange(matrix.shape[1], dtype=np.intp), np.diff(indptr))
    weights = np.asarray(importance, dtype=np.float32)
    eligible = np.ones(len(labels), dtype=bool)
    covered = np.zeros(len(labels), dtype=np.float32)
    selected: List[int] = []
    counts = {int(cls): 0 for cls in np.unique(labels)}
    for cls, quota in quotas.items():
        if quota == 0:
            eligible[labels == cls] = False

    initial = np.maximum(data, 0.0) * weights[rows]
    gains = np.bincount(columns, weights=initial, minlength=len(labels))
    heap = [(-float(gains[j]), int(j)) for j in range(len(labels))]
    heapq.heapify(heap)
    last_eval = np.zeros(len(labels), dtype=np.int64)

    for step in range(sum(quotas.values())):
        best = -1
        while heap:
            neg_gain, candidate = heapq.heappop(heap)
            if not eligible[candidate]:
                continue
            if last_eval[candidate] == step:
                best = candidate
                break
            start, end = indptr[candidate], indptr[candidate + 1]
            touched = rows[start:end]
            gain = float(
                np.dot(weights[touched], np.maximum(data[start:end] - covered[touched], 0.0))
            )
            last_eval[candidate] = step
            heapq.heappush(heap, (-gain, candidate))
        if best < 0:
            raise RuntimeError(f"selection stopped at {step}/{sum(quotas.values())}")
        selected.append(best)
        cls = int(labels[best])
        counts[cls] += 1
        eligible[best] = False
        if counts[cls] >= quotas[cls]:
            eligible[labels == cls] = False
        start, end = indptr[best], indptr[best + 1]
        touched = rows[start:end]
        covered[touched] = np.maximum(covered[touched], data[start:end])
    return selected


@register_method(
    "graph_a2_m1",
    needs={"embeddings"},
    importance="optional",
    kwargs={
        "k_neighbors": 10,
        "k_hops": 2,
        "quota_floor_ratio": 0.75,
        "quota_normalization": "self_optimum",
    },
    description="Graph-A2 with safe normalized dynamic class quotas",
)
def select_graph_a2_m1(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    k_neighbors: int = 10,
    k_hops: int = 2,
    quota_floor_ratio: float = 0.75,
    quota_normalization: str = "self_optimum",
    seed: int = 42,
    verbose: bool = False,
    _verbose_level: int = 1,
    global_selection: bool = False,
    **kwargs,
) -> List[int]:
    if not global_selection:
        raise ValueError("graph_a2_m1 requires --global")
    np.random.seed(seed)
    kernel = build_graph_a2_kernel(
        embeddings, k_neighbors, k_hops, verbose and _verbose_level >= 2
    )
    if float(quota_floor_ratio) == 1.0:
        return _greedy_facility_global(
            kernel, labels, budget_per_class, importance,
            verbose, _verbose_level, "GraphA2-M1-baseline",
            kwargs.get("sparse_cpu", False),
        )
    quotas, _ = allocate_dynamic_quotas(
        kernel, labels, importance, budget_per_class,
        floor_ratio=quota_floor_ratio,
        normalization=quota_normalization,
    )
    if verbose:
        print(f"  [GraphA2-M1] quotas={quotas}")
    return greedy_variable_quota(kernel, labels, quotas, importance)
