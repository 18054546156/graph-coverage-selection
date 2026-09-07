#!/usr/bin/env python3
"""
graph_a2_homophily 的三个快速变体（Tier 2 试错）。

用来对标标准 homophily 方案，看哪个效果最好。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphcov.run.graph import build_knn_graph
from graphcov.run.selection import _greedy_facility_global, register_method
from graphcov.run.selection_opt import _build_global_kernel


# ============================================================================
# 【变体 A】分层异配率（Tiered Homophily）
# ============================================================================

def _compute_tiered_homophily(labels: np.ndarray, knn_indices: np.ndarray) -> np.ndarray:
    """
    根据本地同配率的值分层，返回不同等级的缩放系数。

    三层划分：
      - h(j) >= 0.8 (纯净区): tier_damping = 1.0   [完全保留跨类边]
      - 0.3 <= h(j) < 0.8 (边界区): tier_damping = 0.5   [软damping]
      - h(j) < 0.3 (混杂区): tier_damping = 0.0   [硬解耦]
    """
    k = knn_indices.shape[1]
    if k == 0:
        return np.ones(labels.shape[0], dtype=np.float64)

    neighbor_labels = labels[knn_indices]
    same_class = neighbor_labels == labels[:, None]
    h = same_class.sum(axis=1).astype(np.float64) / float(k)

    # 分层映射
    tier_damping = np.where(
        h >= 0.8, 1.0,
        np.where(h >= 0.3, 0.5, 0.0)
    )
    return tier_damping.astype(np.float64)


def _select_graph_a2_tiered_homophily(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    k_neighbors: int = 10,
    k_hops: int = 2,
    seed: int = 42,
    verbose: bool = False,
    _verbose_level: int = 1,
    global_selection: bool = False,
    **kwargs,
) -> List[int]:
    """分层异配率变体：纯净/边界/混杂 三层 damping。"""
    if not global_selection:
        raise ValueError("graph_a2_tiered_homophily is defined only for --global screening")

    kernel = _build_global_kernel(embeddings, k_neighbors, k_hops, seed,
                                   verbose=verbose, _verbose_level=_verbose_level)
    knn_indices, _ = build_knn_graph(embeddings, k_neighbors,
                                      verbose=(verbose and _verbose_level >= 2))

    tier_damping = _compute_tiered_homophily(labels, knn_indices)

    # 应用分层damping
    coo = kernel.tocoo()
    same_class = labels[coo.row] == labels[coo.col]
    gate = tier_damping[coo.col]
    data = np.where(same_class, coo.data, coo.data * gate)
    gated_kernel = sp.csr_matrix((data, (coo.row, coo.col)), shape=coo.shape)
    gated_kernel.eliminate_zeros()

    if verbose:
        print(f"  [GraphA2-TieredHomophily] tier distribution:")
        pure_count = (tier_damping == 1.0).sum()
        boundary_count = (tier_damping == 0.5).sum()
        mixed_count = (tier_damping == 0.0).sum()
        print(f"    pure (β=1.0): {pure_count}, boundary (β=0.5): {boundary_count}, "
              f"mixed (β=0.0): {mixed_count}")

    return _greedy_facility_global(
        kernel=gated_kernel, labels=labels, budget_per_class=budget_per_class,
        importance=importance, verbose=verbose, _verbose_level=_verbose_level,
        method_name="GraphA2-TieredHomophily", sparse_cpu=kwargs.get("sparse_cpu", False))


register_method(
    "graph_a2_tiered_homophily",
    needs={"embeddings"},
    importance="optional",
    kwargs={"k_neighbors": 10, "k_hops": 2},
    description=(
        "Tiered homophily-gating: pure-region (h>=0.8, β=1.0), "
        "boundary (0.3<=h<0.8, β=0.5), mixed (h<0.3, β=0.0)"
    ),
)(_select_graph_a2_tiered_homophily)


# ============================================================================
# 【变体 B】多尺度同配率（Multi-Scale Homophily）
# ============================================================================

def _compute_multiscale_homophily(labels: np.ndarray, knn_indices_list: List[np.ndarray],
                                   weights: List[float] = None) -> np.ndarray:
    """
    用不同 k 值的 kNN 同配率加权融合。

    输入：
      knn_indices_list: [(n, k1), (n, k2), (n, k3), ...] 不同k值的邻接表
      weights: [w1, w2, w3, ...] 每个尺度的权重（默认均匀）

    输出：
      multiscale_h: (n,) 融合后的同配率
    """
    n_scales = len(knn_indices_list)
    if weights is None:
        weights = [1.0 / n_scales] * n_scales

    multiscale_h = np.zeros(labels.shape[0], dtype=np.float64)

    for knn_indices, w in zip(knn_indices_list, weights):
        k = knn_indices.shape[1]
        if k == 0:
            continue
        neighbor_labels = labels[knn_indices]
        same_class = neighbor_labels == labels[:, None]
        h_k = same_class.sum(axis=1).astype(np.float64) / float(k)
        multiscale_h += w * h_k

    return np.clip(multiscale_h, 0.0, 1.0)


def _select_graph_a2_multiscale_homophily(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    k_neighbors: int = 10,
    k_hops: int = 2,
    seed: int = 42,
    verbose: bool = False,
    _verbose_level: int = 1,
    global_selection: bool = False,
    **kwargs,
) -> List[int]:
    """多尺度同配率变体：用 k=10, 20, 50 三个尺度的加权融合。"""
    if not global_selection:
        raise ValueError("graph_a2_multiscale_homophily is defined only for --global screening")

    kernel = _build_global_kernel(embeddings, k_neighbors, k_hops, seed,
                                   verbose=verbose, _verbose_level=_verbose_level)

    # 计算多个尺度的 kNN
    knn_list = []
    for k_val in [10, 20, 50]:
        knn_indices, _ = build_knn_graph(embeddings, k_val,
                                          verbose=False)  # 不输出日志
        knn_list.append(knn_indices)

    # 融合多尺度同配率（等权重）
    multiscale_h = _compute_multiscale_homophily(labels, knn_list, weights=[1/3, 1/3, 1/3])

    # 应用到 K
    coo = kernel.tocoo()
    same_class = labels[coo.row] == labels[coo.col]
    gate = np.clip(multiscale_h[coo.col], 0.0, 1.0)
    data = np.where(same_class, coo.data, coo.data * gate)
    gated_kernel = sp.csr_matrix((data, (coo.row, coo.col)), shape=coo.shape)
    gated_kernel.eliminate_zeros()

    if verbose:
        print(f"  [GraphA2-MultiScaleHomophily] multiscale h(j) stats (k=10,20,50):")
        print(f"    mean={multiscale_h.mean():.4f}, min={multiscale_h.min():.4f}, "
              f"max={multiscale_h.max():.4f}")

    return _greedy_facility_global(
        kernel=gated_kernel, labels=labels, budget_per_class=budget_per_class,
        importance=importance, verbose=verbose, _verbose_level=_verbose_level,
        method_name="GraphA2-MultiScaleHomophily", sparse_cpu=kwargs.get("sparse_cpu", False))


register_method(
    "graph_a2_multiscale_homophily",
    needs={"embeddings"},
    importance="optional",
    kwargs={"k_neighbors": 10, "k_hops": 2},
    description=(
        "Multi-scale homophily-gating: fuse local homophily from k=10,20,50 "
        "with equal weights"
    ),
)(_select_graph_a2_multiscale_homophily)


# ============================================================================
# 【变体 C】软damping + 置信度加权（Soft Damping with Min-Confidence Floor）
# ============================================================================

def _compute_local_homophily_soft(labels: np.ndarray, knn_indices: np.ndarray,
                                   floor: float = 0.1) -> np.ndarray:
    """
    计算本地同配率，但加一个 floor，避免过度damping（太激进的 h(j)=0）。

    floor: 最小 damping 系数，即使 h(j)=0 也至少保留 floor 的信号
    """
    k = knn_indices.shape[1]
    if k == 0:
        return np.ones(labels.shape[0], dtype=np.float64)

    neighbor_labels = labels[knn_indices]
    same_class = neighbor_labels == labels[:, None]
    h = same_class.sum(axis=1).astype(np.float64) / float(k)

    # 应用 floor：h' = max(h, floor)，这样最坏情况下也保留 floor 的权重
    h_floored = np.maximum(h, floor)
    return h_floored


def _select_graph_a2_soft_damping_homophily(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    importance: np.ndarray,
    k_neighbors: int = 10,
    k_hops: int = 2,
    seed: int = 42,
    verbose: bool = False,
    _verbose_level: int = 1,
    global_selection: bool = False,
    **kwargs,
) -> List[int]:
    """软damping变体：h(j) 最小floor=0.1，避免完全清零。"""
    if not global_selection:
        raise ValueError("graph_a2_soft_damping_homophily is defined only for --global screening")

    kernel = _build_global_kernel(embeddings, k_neighbors, k_hops, seed,
                                   verbose=verbose, _verbose_level=_verbose_level)
    knn_indices, _ = build_knn_graph(embeddings, k_neighbors,
                                      verbose=(verbose and _verbose_level >= 2))

    # 带floor的软damping
    soft_h = _compute_local_homophily_soft(labels, knn_indices, floor=0.1)

    # 应用到 K
    coo = kernel.tocoo()
    same_class = labels[coo.row] == labels[coo.col]
    gate = soft_h[coo.col]
    data = np.where(same_class, coo.data, coo.data * gate)
    gated_kernel = sp.csr_matrix((data, (coo.row, coo.col)), shape=coo.shape)
    gated_kernel.eliminate_zeros()

    if verbose:
        print(f"  [GraphA2-SoftDampingHomophily] soft h(j) with floor=0.1:")
        print(f"    mean={soft_h.mean():.4f}, min={soft_h.min():.4f}, "
              f"max={soft_h.max():.4f}")

    return _greedy_facility_global(
        kernel=gated_kernel, labels=labels, budget_per_class=budget_per_class,
        importance=importance, verbose=verbose, _verbose_level=_verbose_level,
        method_name="GraphA2-SoftDampingHomophily", sparse_cpu=kwargs.get("sparse_cpu", False))


register_method(
    "graph_a2_soft_damping_homophily",
    needs={"embeddings"},
    importance="optional",
    kwargs={"k_neighbors": 10, "k_hops": 2},
    description=(
        "Soft damping homophily-gating: cross-class edges scaled by max(h(j), 0.1), "
        "avoiding aggressive masking"
    ),
)(_select_graph_a2_soft_damping_homophily)
