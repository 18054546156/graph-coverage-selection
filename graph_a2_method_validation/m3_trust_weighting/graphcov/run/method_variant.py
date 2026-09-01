"""M3: Graph-A2 with purity and local-support trust weighting."""

from __future__ import annotations

from typing import List, Tuple

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


def compute_purity_and_support(
    kernel: sp.spmatrix,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return row purity and within-class percentile support for each client."""
    coo = kernel.tocoo()
    total = np.bincount(coo.row, weights=np.maximum(coo.data, 0.0), minlength=len(labels))
    same_mask = labels[coo.row] == labels[coo.col]
    same = np.bincount(
        coo.row[same_mask],
        weights=np.maximum(coo.data[same_mask], 0.0),
        minlength=len(labels),
    )
    purity = np.divide(same, total, out=np.ones_like(same), where=total > 0)
    purity = np.clip(purity, 0.0, 1.0)

    support = np.zeros(len(labels), dtype=np.float64)
    for cls in np.unique(labels):
        indices = np.flatnonzero(labels == cls)
        order = np.argsort(same[indices], kind="mergesort")
        ranks = np.empty(len(indices), dtype=np.float64)
        ranks[order] = np.arange(len(indices), dtype=np.float64)
        support[indices] = ranks / max(len(indices) - 1, 1)
    return purity.astype(np.float32), support.astype(np.float32)


def compute_trust_weights(
    kernel: sp.spmatrix,
    labels: np.ndarray,
    purity_exponent: float,
    trust_mode: str = "purity",
    weight_floor: float = 0.0,
) -> Tuple[np.ndarray, dict]:
    """Compute p^beta or p^(beta*(1-support)) client weights."""
    if purity_exponent < 0:
        raise ValueError("purity_exponent must be non-negative")
    if trust_mode not in {"purity", "adaptive"}:
        raise ValueError("trust_mode must be 'purity' or 'adaptive'")
    if not 0.0 <= weight_floor <= 1.0:
        raise ValueError("weight_floor must be in [0, 1]")
    if purity_exponent == 0:
        ones = np.ones(len(labels), dtype=np.float32)
        return ones, {"purity": ones.copy(), "support": ones.copy()}

    purity, support = compute_purity_and_support(kernel, labels)
    exponent = np.full(len(labels), float(purity_exponent), dtype=np.float32)
    if trust_mode == "adaptive":
        exponent *= 1.0 - support
    weights = np.power(np.maximum(purity, np.finfo(np.float32).eps), exponent)
    weights = np.maximum(weights, float(weight_floor)).astype(np.float32)
    return weights, {"purity": purity, "support": support, "exponent": exponent}


@register_method(
    "graph_a2_m3",
    needs={"embeddings"},
    importance="optional",
    kwargs={
        "k_neighbors": 10,
        "k_hops": 2,
        "purity_exponent": 1.0,
        "trust_mode": "purity",
        "trust_weight_floor": 0.0,
    },
    description="Graph-A2 with purity or adaptive local-support client weighting",
)
def select_graph_a2_m3(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    k_neighbors: int = 10,
    k_hops: int = 2,
    purity_exponent: float = 1.0,
    trust_mode: str = "purity",
    trust_weight_floor: float = 0.0,
    seed: int = 42,
    verbose: bool = False,
    _verbose_level: int = 1,
    global_selection: bool = False,
    **kwargs,
) -> List[int]:
    if not global_selection:
        raise ValueError("graph_a2_m3 requires --global")
    np.random.seed(seed)
    kernel = build_graph_a2_kernel(
        embeddings, k_neighbors, k_hops, verbose and _verbose_level >= 2
    )
    trust, diagnostics = compute_trust_weights(
        kernel,
        labels,
        purity_exponent=purity_exponent,
        trust_mode=trust_mode,
        weight_floor=trust_weight_floor,
    )
    effective_importance = np.asarray(importance, dtype=np.float32) * trust
    if verbose:
        purity = diagnostics["purity"]
        print(
            f"  [GraphA2-M3] mode={trust_mode}, exponent={purity_exponent:g}, "
            f"purity_mean={purity.mean():.4f}, weight_mean={trust.mean():.4f}"
        )
    return _greedy_facility_global(
        kernel, labels, budget_per_class, effective_importance,
        verbose, _verbose_level, "GraphA2-M3",
        kwargs.get("sparse_cpu", False),
    )
