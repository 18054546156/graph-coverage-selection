from __future__ import annotations

import numpy as np
from scipy import sparse

from graphcov.run.selection import _greedy_facility_global_sparse
from graphcov.run.graph import build_adjacency_matrix

from v11.methods.selection import (
    SelectionResult,
    build_diffusion_kernel,
    build_or_load_knn,
    build_weighted_adjacency,
    compute_exact_margins,
    greedy_facility,
    posthoc_margin_repair,
    symmetric_normalize,
)


def synthetic_embeddings() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    centers = np.eye(3, 8, dtype=np.float32)
    embeddings = np.vstack(
        [centers[cls] + 0.03 * rng.normal(size=(12, 8)) for cls in range(3)]
    ).astype(np.float32)
    labels = np.repeat(np.arange(3), 12).astype(np.int64)
    return embeddings, labels


def test_kernel_and_greedy_preserve_equal_class_quotas(tmp_path):
    embeddings, labels = synthetic_embeddings()
    indices, distances, _ = build_or_load_knn(
        embeddings, 4, tmp_path / "knn.npz", backend="sklearn", gpu=None
    )
    adjacency = build_weighted_adjacency(indices, distances, len(labels))
    official_adjacency = build_adjacency_matrix(
        indices,
        distances,
        len(labels),
        symmetric=True,
        similarity_weights=True,
    )
    assert (adjacency != official_adjacency).nnz == 0
    kernel, metadata = build_diffusion_kernel(
        symmetric_normalize(adjacency),
        hops=4,
        weight_mode="geometric",
        hop_decay=0.85,
        truncation="per_hop",
        max_degree=7,
    )
    assert metadata["hop_diagnostics"][-1]["accumulated_nnz"] <= len(labels) * 7
    selected = greedy_facility(kernel, labels, budget_per_class=3)
    assert len(selected.indices) == 9
    assert selected.class_counts == {0: 3, 1: 3, 2: 3}
    official = _greedy_facility_global_sparse(
        kernel,
        labels,
        budget_per_class=3,
        importance=np.ones(len(labels), dtype=np.float32),
    )
    assert selected.order.tolist() == official


def test_exact_margin_and_hard_cap(tmp_path):
    embeddings, labels = synthetic_embeddings()
    margins, backend = compute_exact_margins(
        embeddings, labels, backend="numpy", gpu=None, batch_size=8
    )
    assert backend == "numpy_exact"
    assert not np.any(margins.unsafe)

    indices, distances, _ = build_or_load_knn(
        embeddings, 3, tmp_path / "knn.npz", backend="sklearn", gpu=None
    )
    kernel, _ = build_diffusion_kernel(
        symmetric_normalize(build_weighted_adjacency(indices, distances, len(labels))),
        hops=2,
    )
    synthetic_unsafe = np.zeros(len(labels), dtype=bool)
    synthetic_unsafe[[0, 1, 12, 13, 24, 25]] = True
    result = greedy_facility(
        kernel,
        labels,
        budget_per_class=3,
        unsafe=synthetic_unsafe,
        unsafe_caps={0: 1, 1: 1, 2: 1},
    )
    assert all(value <= 1 for value in result.unsafe_counts.values())


def test_posthoc_repair_uses_minimum_number_of_swaps():
    labels = np.repeat(np.arange(2), 5).astype(np.int64)
    unsafe = np.asarray([True, True, False, False, False] * 2)
    base_order = np.asarray([0, 1, 5, 6], dtype=np.int64)
    base = SelectionResult(
        indices=np.sort(base_order),
        order=base_order,
        class_counts={0: 2, 1: 2},
        unsafe_counts={0: 2, 1: 2},
    )
    kernel = sparse.eye(len(labels), format="csr", dtype=np.float32)
    repaired = posthoc_margin_repair(
        kernel, labels, base, unsafe, unsafe_caps={0: 1, 1: 1}
    )
    assert repaired.class_counts == {0: 2, 1: 2}
    assert repaired.unsafe_counts == {0: 1, 1: 1}
    assert len(np.setdiff1d(repaired.indices, base.indices)) == 2
