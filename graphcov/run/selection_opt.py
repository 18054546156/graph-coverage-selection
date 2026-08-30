"""Graph-A2 optimization variants for controlled SOTA screening.

The registered methods use the official global Graph-A2 kernel construction.
They only change cross-class coverage credit, class-budget allocation, or both.
Every method preserves the official total selection budget.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp

from .graph import build_adjacency_matrix, build_knn_graph
from .selection import _greedy_facility_global, register_method


def _build_global_kernel(
    embeddings: np.ndarray,
    k_neighbors: int,
    k_hops: int,
    seed: int,
    verbose: bool = False,
    _verbose_level: int = 1,
) -> sp.csr_matrix:
    """Build the same global sum(A_sym ** h) kernel as official Graph-A2."""
    from scipy.sparse import diags

    if k_hops < 1:
        raise ValueError(f"k_hops must be positive, got {k_hops}")

    np.random.seed(seed)
    n = len(embeddings)
    knn_indices, knn_distances = build_knn_graph(
        embeddings,
        k_neighbors,
        verbose=(verbose and _verbose_level >= 2),
    )
    adjacency = build_adjacency_matrix(knn_indices, knn_distances, n)
    row_sums = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    row_sums[row_sums == 0] = 1
    d_inv_sqrt = diags(1.0 / np.sqrt(row_sums))
    normalized = (d_inv_sqrt @ adjacency @ d_inv_sqrt).tocsr()

    kernel = normalized.copy()
    power = normalized.copy()
    for _ in range(k_hops - 1):
        power = power @ normalized
        kernel = kernel + power
    return kernel.tocsr()


def _rescale_cross_class(
    kernel: sp.spmatrix,
    labels: np.ndarray,
    beta: float,
) -> sp.csr_matrix:
    """Scale cross-class entries by beta while preserving same-class entries."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")

    coo = kernel.tocoo()
    same_class = labels[coo.row] == labels[coo.col]
    if beta == 0.0:
        result = sp.csr_matrix(
            (coo.data[same_class], (coo.row[same_class], coo.col[same_class])),
            shape=coo.shape,
        )
    else:
        data = np.where(same_class, coo.data, coo.data * beta)
        result = sp.csr_matrix((data, (coo.row, coo.col)), shape=coo.shape)
    result.eliminate_zeros()
    return result


def _allocate_power_quota(
    labels: np.ndarray,
    budget_per_class: int,
    power: float,
) -> Dict[int, int]:
    """Allocate the fixed total budget proportionally to class_count ** power."""
    classes, counts = np.unique(labels, return_counts=True)
    total = int(budget_per_class) * len(classes)
    if budget_per_class < 1 or total > int(counts.sum()):
        raise ValueError("requested budget is infeasible")

    weights = counts.astype(np.float64) ** float(power)
    raw = total * weights / weights.sum()
    quota = np.clip(np.floor(raw).astype(np.int64), 1, counts)

    while int(quota.sum()) < total:
        eligible = np.flatnonzero(quota < counts)
        if not len(eligible):
            raise RuntimeError("cannot distribute the remaining budget")
        score = raw[eligible] - quota[eligible]
        quota[eligible[int(np.argmax(score))]] += 1

    while int(quota.sum()) > total:
        eligible = np.flatnonzero(quota > 1)
        if not len(eligible):
            raise RuntimeError("cannot reduce quota to the requested budget")
        score = quota[eligible] - raw[eligible]
        quota[eligible[int(np.argmax(score))]] -= 1

    return {int(cls): int(value) for cls, value in zip(classes, quota)}


def _allocate_quota_sqrt(labels: np.ndarray, budget_per_class: int) -> Dict[int, int]:
    return _allocate_power_quota(labels, budget_per_class, power=0.5)


def _class_coverage_marginals(
    kernel: sp.spmatrix,
    labels: np.ndarray,
    importance: np.ndarray,
    cls: int,
    max_quota: int,
) -> List[float]:
    """Return normalized class-local facility-location marginal gains."""
    indices = np.flatnonzero(labels == cls)
    local = kernel[indices][:, indices].tocsc().astype(np.float32)
    local_importance = importance[indices].astype(np.float32)
    normalizer = float(local_importance.sum())
    if normalizer <= 0:
        local_importance = np.ones(len(indices), dtype=np.float32)
        normalizer = float(len(indices))

    max_quota = min(int(max_quota), len(indices))
    data = local.data
    row_indices = local.indices
    indptr = local.indptr
    col_indices = np.repeat(np.arange(local.shape[1], dtype=np.intp), np.diff(indptr))
    coverage = np.zeros(local.shape[0], dtype=np.float32)
    available = np.ones(local.shape[1], dtype=bool)

    initial = np.maximum(data, 0.0) * local_importance[row_indices]
    gains = np.bincount(col_indices, weights=initial, minlength=local.shape[1])
    heap = [(-float(gains[j]), int(j)) for j in range(local.shape[1])]
    heapq.heapify(heap)
    last_eval = np.zeros(local.shape[1], dtype=np.int64)

    marginals: List[float] = []
    for step in range(max_quota):
        best = -1
        best_gain = 0.0
        while heap:
            neg_gain, candidate = heapq.heappop(heap)
            if not available[candidate]:
                continue
            if last_eval[candidate] == step:
                best = candidate
                best_gain = -float(neg_gain)
                break
            start, end = indptr[candidate], indptr[candidate + 1]
            rows = row_indices[start:end]
            gain = float(
                np.dot(
                    local_importance[rows],
                    np.maximum(data[start:end] - coverage[rows], 0.0),
                )
            )
            last_eval[candidate] = step
            heapq.heappush(heap, (-gain, candidate))

        if best < 0:
            break
        available[best] = False
        start, end = indptr[best], indptr[best + 1]
        rows = row_indices[start:end]
        coverage[rows] = np.maximum(coverage[rows], data[start:end])
        marginals.append(best_gain / normalizer)

    return marginals


def _allocate_graph_marginal_quota(
    kernel: sp.spmatrix,
    labels: np.ndarray,
    importance: np.ndarray,
    budget_per_class: int,
) -> Tuple[Dict[int, int], Dict[int, List[float]]]:
    """Allocate each next unit to the class with the largest coverage marginal."""
    classes, counts = np.unique(labels, return_counts=True)
    total = int(budget_per_class) * len(classes)
    if budget_per_class < 1 or total > int(counts.sum()):
        raise ValueError("requested budget is infeasible")

    curve_limit = budget_per_class + max(20, budget_per_class // 2)
    curves = {
        int(cls): _class_coverage_marginals(
            kernel,
            labels,
            importance,
            int(cls),
            min(curve_limit, int(count)),
        )
        for cls, count in zip(classes, counts)
    }
    quotas = {int(cls): 1 for cls in classes}

    for _ in range(total - len(classes)):
        candidates = []
        for cls in classes:
            cls = int(cls)
            next_index = quotas[cls]
            curve = curves[cls]
            if next_index < len(curve):
                candidates.append((float(curve[next_index]), -cls, cls))
        if not candidates:
            raise RuntimeError("coverage curves cannot satisfy the total budget")
        _, _, chosen = max(candidates)
        quotas[chosen] += 1

    return quotas, curves


def _greedy_variable_quota(
    kernel: sp.spmatrix,
    labels: np.ndarray,
    quotas: Dict[int, int],
    importance: np.ndarray,
    verbose: bool = False,
    _verbose_level: int = 1,
    method_name: str = "GraphA2-VariableQuota",
) -> List[int]:
    """Official sparse lazy greedy with a per-class quota dictionary."""
    classes, counts = np.unique(labels, return_counts=True)
    expected_classes = {int(cls) for cls in classes}
    if set(quotas) != expected_classes:
        raise ValueError("quota keys must exactly match label classes")
    if any(quotas[int(cls)] < 0 or quotas[int(cls)] > int(count) for cls, count in zip(classes, counts)):
        raise ValueError("quota is outside class capacity")

    n = len(labels)
    total_budget = int(sum(quotas.values()))
    kernel_csc = sp.csc_matrix(kernel, dtype=np.float32)
    imp = importance.astype(np.float32)
    data = kernel_csc.data
    row_indices = kernel_csc.indices
    indptr = kernel_csc.indptr
    col_indices = np.repeat(np.arange(n, dtype=np.intp), np.diff(indptr))

    eligible = np.ones(n, dtype=bool)
    for cls in classes:
        if quotas[int(cls)] == 0:
            eligible[labels == cls] = False
    selected: List[int] = []
    class_counts = {int(cls): 0 for cls in classes}
    max_coverage = np.zeros(n, dtype=np.float32)

    initial = np.maximum(data, 0.0) * imp[row_indices]
    gains = np.bincount(col_indices, weights=initial, minlength=n)
    heap = [(-float(gains[j]), int(j)) for j in range(n)]
    heapq.heapify(heap)
    last_eval = np.zeros(n, dtype=np.int64)

    for step in range(total_budget):
        best = -1
        best_gain = 0.0
        while heap:
            neg_gain, candidate = heapq.heappop(heap)
            if not eligible[candidate]:
                continue
            if last_eval[candidate] == step:
                best = candidate
                best_gain = -float(neg_gain)
                break
            start, end = indptr[candidate], indptr[candidate + 1]
            rows = row_indices[start:end]
            gain = float(
                np.dot(
                    imp[rows],
                    np.maximum(data[start:end] - max_coverage[rows], 0.0),
                )
            )
            last_eval[candidate] = step
            heapq.heappush(heap, (-gain, candidate))

        if best < 0:
            raise RuntimeError(f"selection stopped at {step}/{total_budget}")

        selected.append(best)
        cls = int(labels[best])
        class_counts[cls] += 1
        eligible[best] = False
        if class_counts[cls] >= quotas[cls]:
            eligible[labels == cls] = False

        start, end = indptr[best], indptr[best + 1]
        rows = row_indices[start:end]
        max_coverage[rows] = np.maximum(max_coverage[rows], data[start:end])

        if verbose and _verbose_level >= 3 and (step < 10 or step % 50 == 0):
            print(f"    [{method_name}] step={step + 1}, gain={best_gain:.6f}, class={cls}")

    if verbose:
        print(f"  [{method_name}] selected={len(selected)}, quotas={quotas}")
    return selected


def _run_variant(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    k_neighbors: int,
    k_hops: int,
    seed: int,
    beta: float,
    allocation: str,
    verbose: bool,
    _verbose_level: int,
    method_name: str,
    global_selection: bool,
    sparse_cpu: bool,
) -> List[int]:
    if not global_selection:
        raise ValueError(f"{method_name} is defined only for --global screening")

    kernel = _build_global_kernel(
        embeddings,
        k_neighbors,
        k_hops,
        seed,
        verbose=verbose,
        _verbose_level=_verbose_level,
    )
    if beta != 1.0:
        kernel = _rescale_cross_class(kernel, labels, beta)

    if allocation == "equal":
        return _greedy_facility_global(
            kernel=kernel,
            labels=labels,
            budget_per_class=budget_per_class,
            importance=importance,
            verbose=verbose,
            _verbose_level=_verbose_level,
            method_name=method_name,
            sparse_cpu=sparse_cpu,
        )
    if allocation == "sqrt":
        quotas = _allocate_quota_sqrt(labels, budget_per_class)
    elif allocation == "graph_marginal":
        quotas, _ = _allocate_graph_marginal_quota(
            kernel,
            labels,
            importance,
            budget_per_class,
        )
    else:
        raise ValueError(f"unknown allocation: {allocation}")

    return _greedy_variable_quota(
        kernel,
        labels,
        quotas,
        importance,
        verbose=verbose,
        _verbose_level=_verbose_level,
        method_name=method_name,
    )


def _variant(beta: float, allocation: str, method_name: str):
    def run(
        embeddings,
        labels,
        budget_per_class,
        importance,
        k_neighbors=10,
        k_hops=2,
        seed=42,
        verbose=False,
        _verbose_level=1,
        global_selection=False,
        **kwargs,
    ):
        return _run_variant(
            embeddings=embeddings,
            labels=labels,
            budget_per_class=budget_per_class,
            importance=importance,
            k_neighbors=k_neighbors,
            k_hops=k_hops,
            seed=seed,
            beta=beta,
            allocation=allocation,
            verbose=verbose,
            _verbose_level=_verbose_level,
            method_name=method_name,
            global_selection=global_selection,
            sparse_cpu=kwargs.get("sparse_cpu", False),
        )

    return run


_REGISTERED_VARIANTS = (
    ("graph_a2_dec", 0.0, "equal", "GraphA2-Dec"),
    ("graph_a2_damp25", 0.25, "equal", "GraphA2-Damp25"),
    ("graph_a2_damp50", 0.50, "equal", "GraphA2-Damp50"),
    ("graph_a2_damp75", 0.75, "equal", "GraphA2-Damp75"),
    ("graph_a2_sqrt", 1.0, "sqrt", "GraphA2-Sqrt"),
    ("graph_a2_dec_sqrt", 0.0, "sqrt", "GraphA2-DecSqrt"),
    ("graph_a2_marginal", 1.0, "graph_marginal", "GraphA2-Marginal"),
    ("graph_a2_dec_marginal", 0.0, "graph_marginal", "GraphA2-DecMarginal"),
)

for _method, _beta, _allocation, _display_name in _REGISTERED_VARIANTS:
    register_method(
        _method,
        needs={"embeddings"},
        importance="optional",
        kwargs={"k_neighbors": 10, "k_hops": 2},
        description=(
            f"Global Graph-A2 with cross-class beta={_beta:g} "
            f"and {_allocation} class-budget allocation"
        ),
    )(_variant(_beta, _allocation, _display_name))
