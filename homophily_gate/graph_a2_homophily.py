#!/usr/bin/env python3
"""
graph_a2_homophily —— 用"每个候选点自己的局部同配率"给跨类覆盖信用打折。

============================================================================
背景（为什么要做这个东西）
============================================================================

官方 Graph-A2 的贪心目标函数长这样（selection.py::_greedy_facility_global）：

    gain(j) = sum_i importance[i] * max(0, K[i,j] - max_coverage[i])

也就是说，选中候选点 j 时，它能拿到的"分数"来自 K 矩阵里所有跟它相连的
样本 i —— 不管 i 和 j 是不是同一个类别。

问题在于：如果 j 恰好处在一个"类别混杂"的区域（它的近邻里一半是 A 类，
一半是 B 类），K[i,j] 里那些跨类的边际增益，很大概率只是"embedding 空间
里两个类别本来就分不太开"造成的噪声，而不是 j 真的蕴含了什么有价值的
决策边界信息。这样的 j 会被贪心算法高估，抢走本该分给"类内代表性样本"
的预算。

我们已经在 Tissue（跨类边占比 60.82%）上验证过：把所有跨类边的信用
一刀切砍掉（beta=0，见 graph_a2_dec），balanced accuracy 反而 +7.00pp。
但在 OrganA（跨类边占比 25.57%）上，同样的一刀切只带来 +0.32pp 的提升
——因为"一刀切用一个全局系数"没有区分"这条跨类边到底是不是噪声"，
只是笼统地按整个数据集的统计量做全局折扣。

本文件实现一个更细粒度的规则：不再用一个数据集级别的全局 beta，而是
给每一个候选点 j 单独算一个"这个点自己靠不靠谱"的分数，用这个分数
决定它的跨类边际增益该打几折。这个分数来自图神经网络文献里研究"异配图
（heterophily graph）"时常用的统计量：局部同配率（local homophily）。

============================================================================
局部同配率 h(j) 是什么、怎么算
============================================================================

对每个样本 j，看它在 k 近邻图里的 k 个最近邻居，数一数这 k 个邻居里有
多少个跟 j 是同一个类别，除以 k：

    h(j) = |{ i ∈ kNN(j) : label(i) == label(j) }| / k

直觉：
  - h(j) 接近 1  →  j 周围几乎全是同类样本  →  j 处在"类内纯净区"
  - h(j) 接近 0  →  j 周围大部分是别的类样本  →  j 处在"类别混杂/边界模糊区"

============================================================================
怎么用 h(j) 去改造覆盖矩阵 K
============================================================================

对 K 里的每一个非零条目 K[i, j]（含义：候选点 j 能给样本 i 提供多少
"覆盖"）：

  - 如果 label(i) == label(j)（同类边）：原样保留，不打折。
    —— 这部分信用是"j 真的代表了它自己这个类"，没有争议。

  - 如果 label(i) != label(j)（跨类边）：乘上 h(j)，即
        K'[i, j] = K[i, j] * h(j)
    —— 用"j 自己周围有多纯"来当作这条跨类边有多可信的代理指标。
    j 自己都处在类别混杂区（h(j) 低），它声称"能覆盖别的类的样本"这件事
    就该被大幅打折；反过来，如果 j 处在类内纯净区（h(j) 高，比如 j 是
    某个类的核心区域但恰好跟另一个类的边界样本挨得很近），它提供的
    跨类覆盖信用更值得信任，打折就轻一些。

这跟我们已经验证过的 graph_a2_dec / graph_a2_damp 系列（selection_opt.py）
的区别是：那些方法用同一个 beta 对整个数据集里所有跨类边一视同仁，
这个方法让"每个候选点自己的局部结构"决定折扣力度，理论上应该同时在
Tissue（大范围混杂）和 OrganA（局部混杂、局部纯净并存）上都工作，
不需要为每个数据集单独查表调 beta。

============================================================================
和官方 Graph-A2 的关系
============================================================================

k 近邻图、对称归一化、两项多项式核 K = A_sym + A_sym^2 的构造方式，
完全复用 selection_opt.py 里已经验证过、和官方逐字一致的
`_build_global_kernel`。本文件只改"贪心目标函数看到的 K"，不改图的
构造方式、不改贪心算法本身、不改每类预算（仍然是官方的 equal quota）。
这样做 A/B 对比时，涨跌只能来自这一个改动。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import scipy.sparse as sp

# 让本文件能以 "sota_race_v1 是包根目录" 的方式 import graphcov.run.*，
# 不需要把这个文件塞进 graphcov/run 目录里，也不用改动官方代码。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphcov.run.graph import build_knn_graph  # noqa: E402
from graphcov.run.selection import _greedy_facility_global, register_method  # noqa: E402
from graphcov.run.selection_opt import _build_global_kernel  # noqa: E402


def _compute_local_homophily(labels: np.ndarray, knn_indices: np.ndarray) -> np.ndarray:
    """
    计算每个样本 j 的局部同配率 h(j)。

    输入：
      labels：(n,) 每个样本的类别标签
      knn_indices：(n, k) 每个样本的 k 个最近邻居的下标
                   （来自 build_knn_graph，不含自身）

    输出：
      homophily：(n,) 每个样本的同配率，取值范围 [0, 1]

    计算方式：
      对第 j 行，取出它的 k 个邻居的标签 labels[knn_indices[j]]，
      跟 labels[j] 逐个比较，数相同的个数，除以 k。
    """
    if labels.shape[0] != knn_indices.shape[0]:
        raise ValueError("labels and knn_indices must describe the same n samples")
    k = knn_indices.shape[1]
    if k == 0:
        # 边缘情况：n 太小导致没有邻居，约定同配率为 1（不打折，退化成原始 K）。
        return np.ones(labels.shape[0], dtype=np.float64)

    neighbor_labels = labels[knn_indices]              # (n, k)
    same_class = neighbor_labels == labels[:, None]    # (n, k) 布尔矩阵
    homophily = same_class.sum(axis=1).astype(np.float64) / float(k)
    return homophily


def _apply_homophily_gate(
    kernel: sp.spmatrix,
    labels: np.ndarray,
    homophily: np.ndarray,
    floor: float = 0.0,
) -> sp.csr_matrix:
    """
    用逐候选点的同配率给跨类覆盖信用打折，同类信用原样保留。

    对 K 里每个非零条目 K[i, j]：
      - label[i] == label[j]（同类边）  -> 保留原值
      - label[i] != label[j]（跨类边）  -> 乘上 clip(homophily[j], floor, 1.0)

    参数 floor：折扣系数的下限。floor=0 时，如果某个候选点的同配率算出来
    正好是 0（它的 k 个邻居没有一个跟它同类），它所有跨类边际增益会被
    完全清零，等价于对这个点做"硬解耦"。设置 floor>0（比如 0.05）可以
    避免这种极端情况让某些孤立候选点完全失去被跨类覆盖计入的机会，
    默认保持 0，跟 graph_a2_dec 的"激进程度"处在同一量级，方便直接对比。
    """
    coo = kernel.tocoo()
    same_class = labels[coo.row] == labels[coo.col]
    gate = np.clip(homophily[coo.col], floor, 1.0)
    data = np.where(same_class, coo.data, coo.data * gate)
    result = sp.csr_matrix((data, (coo.row, coo.col)), shape=coo.shape)
    result.eliminate_zeros()
    return result


def _select_graph_a2_homophily(
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
    """graph_a2_homophily 的主入口，接口形状和 selection_opt.py 里的变体一致。"""
    if not global_selection:
        raise ValueError("graph_a2_homophily is defined only for --global screening")

    # 第一步：跟官方 graph_a2 一模一样地构造覆盖核 K
    #   K = A_sym + A_sym^2   (k_hops=2 时)
    kernel = _build_global_kernel(
        embeddings, k_neighbors, k_hops, seed,
        verbose=verbose, _verbose_level=_verbose_level,
    )

    # 第二步：单独跑一次 kNN，只是为了拿到每个点的"近邻下标"，用来算同配率。
    # 这一步和 _build_global_kernel 内部做的 kNN 查询用的是同一套参数
    # （相同 embeddings、相同 k_neighbors、相同后端），所以近邻表是一致的；
    # 之所以再跑一次而不是复用，是为了不改动 selection_opt.py 里已经跑过
    # 服务器验证的官方核构造代码，保证这个新方法完全不影响正在跑的实验。
    knn_indices, _ = build_knn_graph(
        embeddings, k_neighbors, verbose=(verbose and _verbose_level >= 2))

    # 第三步：算每个候选点的局部同配率 h(j)
    homophily = _compute_local_homophily(labels, knn_indices)

    # 第四步：用 h(j) 给跨类覆盖信用打折，同类信用不动
    gated_kernel = _apply_homophily_gate(kernel, labels, homophily)

    if verbose:
        print(
            f"  [GraphA2-Homophily] homophily: "
            f"mean={homophily.mean():.4f} min={homophily.min():.4f} "
            f"max={homophily.max():.4f}"
        )

    # 第五步：贪心 facility location，跟官方一样的 equal per-class quota
    return _greedy_facility_global(
        kernel=gated_kernel,
        labels=labels,
        budget_per_class=budget_per_class,
        importance=importance,
        verbose=verbose,
        _verbose_level=_verbose_level,
        method_name="GraphA2-Homophily",
        sparse_cpu=kwargs.get("sparse_cpu", False),
    )


register_method(
    "graph_a2_homophily",
    needs={"embeddings"},
    importance="optional",
    kwargs={"k_neighbors": 10, "k_hops": 2},
    description=(
        "Global Graph-A2 where each candidate's cross-class coverage credit "
        "is scaled by its own local kNN label-homophily rate (per-node gate, "
        "no dataset-level beta lookup)"
    ),
)(_select_graph_a2_homophily)
