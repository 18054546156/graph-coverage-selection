"""M2: Graph-A2 with training-free class-pair relation gates."""

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


def estimate_class_pair_gate(
    kernel: sp.spmatrix,
    labels: np.ndarray,
    strength: float = 1.0,
    min_credit: float = 0.0,
    smoothing: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a symmetric class-pair gate from graph mass relative to class priors."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("relation_strength must be in [0, 1]")
    if not 0.0 <= min_credit <= 1.0:
        raise ValueError("pair_min_credit must be in [0, 1]")
    if smoothing < 0:
        raise ValueError("pair_smoothing must be non-negative")

    classes, encoded = np.unique(labels, return_inverse=True)
    n_classes = len(classes)
    coo = kernel.tocoo()
    pair_mass = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(pair_mass, (encoded[coo.row], encoded[coo.col]), coo.data)

    counts = np.bincount(encoded, minlength=n_classes).astype(np.float64)
    prior = counts / counts.sum()
    row_total = pair_mass.sum(axis=1, keepdims=True)
    observed = (pair_mass + smoothing * prior[None, :]) / np.maximum(
        row_total + smoothing, np.finfo(np.float64).eps
    )
    lift = observed / np.maximum(prior[None, :], np.finfo(np.float64).eps)
    symmetric_lift = np.sqrt(np.maximum(lift * lift.T, 0.0))
    estimated = np.clip(symmetric_lift, min_credit, 1.0)
    np.fill_diagonal(estimated, 1.0)
    gate = (1.0 - strength) + strength * estimated
    np.fill_diagonal(gate, 1.0)
    return gate.astype(np.float32), classes


def apply_relation_gate(
    kernel: sp.spmatrix,
    labels: np.ndarray,
    mode: str,
    cross_credit: float = 0.0,
    relation_strength: float = 1.0,
    pair_min_credit: float = 0.0,
    pair_smoothing: float = 1.0,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Apply baseline, scalar, same-class, or estimated class-pair credit."""
    if mode not in {"baseline", "scalar", "same_class", "class_pair"}:
        raise ValueError("relation_mode must be baseline, scalar, same_class, or class_pair")
    if not 0.0 <= cross_credit <= 1.0:
        raise ValueError("cross_credit must be in [0, 1]")
    classes, encoded = np.unique(labels, return_inverse=True)
    if mode == "baseline":
        gate = np.ones((len(classes), len(classes)), dtype=np.float32)
        return kernel.tocsr(), gate
    if mode in {"scalar", "same_class"}:
        credit = 0.0 if mode == "same_class" else float(cross_credit)
        gate = np.full((len(classes), len(classes)), credit, dtype=np.float32)
        np.fill_diagonal(gate, 1.0)
    else:
        gate, classes = estimate_class_pair_gate(
            kernel,
            labels,
            strength=relation_strength,
            min_credit=pair_min_credit,
            smoothing=pair_smoothing,
        )
    coo = kernel.tocoo()
    data = coo.data * gate[encoded[coo.row], encoded[coo.col]]
    result = sp.csr_matrix((data, (coo.row, coo.col)), shape=coo.shape)
    result.eliminate_zeros()
    return result, gate


@register_method(
    "graph_a2_m2",
    needs={"embeddings"},
    importance="optional",
    kwargs={
        "k_neighbors": 10,
        "k_hops": 2,
        "relation_mode": "class_pair",
        "cross_credit": 0.25,
        "relation_strength": 1.0,
        "pair_min_credit": 0.0,
        "pair_smoothing": 1.0,
    },
    description="Graph-A2 with training-free class-pair relation gates",
)
def select_graph_a2_m2(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    k_neighbors: int = 10,
    k_hops: int = 2,
    relation_mode: str = "class_pair",
    cross_credit: float = 0.25,
    relation_strength: float = 1.0,
    pair_min_credit: float = 0.0,
    pair_smoothing: float = 1.0,
    seed: int = 42,
    verbose: bool = False,
    _verbose_level: int = 1,
    global_selection: bool = False,
    **kwargs,
) -> List[int]:
    if not global_selection:
        raise ValueError("graph_a2_m2 requires --global")
    np.random.seed(seed)
    kernel = build_graph_a2_kernel(
        embeddings, k_neighbors, k_hops, verbose and _verbose_level >= 2
    )
    gated, gate = apply_relation_gate(
        kernel,
        labels,
        mode=relation_mode,
        cross_credit=cross_credit,
        relation_strength=relation_strength,
        pair_min_credit=pair_min_credit,
        pair_smoothing=pair_smoothing,
    )
    if verbose:
        print(f"  [GraphA2-M2] mode={relation_mode}, gate=\n{gate}")
    return _greedy_facility_global(
        gated, labels, budget_per_class, importance,
        verbose, _verbose_level, "GraphA2-M2",
        kwargs.get("sparse_cpu", False),
    )
