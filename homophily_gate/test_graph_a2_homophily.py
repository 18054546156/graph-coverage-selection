#!/usr/bin/env python3
"""
graph_a2_homophily 的单元测试。

不需要 GPU、不需要下载数据集，几秒钟跑完。用来验证：
  1. 同配率 h(j) 算得对不对（用手算过的小例子核对）
  2. 打折规则对不对：同类边不动、跨类边按 h(j) 缩放
  3. 注册进 METHODS 之后能正常跑一遍完整流程，选出正确数量的样本

运行方式（在 sota_race_v1 目录下）：
    python -m pytest homophily_gate/test_graph_a2_homophily.py -v
或者：
    python homophily_gate/test_graph_a2_homophily.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.sparse as sp

from graphcov.run.selection import METHODS

from homophily_gate.graph_a2_homophily import (  # noqa: E402
    _apply_homophily_gate,
    _compute_local_homophily,
)


class HomophilyGateTests(unittest.TestCase):
    def test_method_is_registered(self):
        # 只要 import 了 homophily_gate.graph_a2_homophily 这个模块，
        # register_method 装饰器就会把它塞进全局 METHODS 字典。
        self.assertIn("graph_a2_homophily", METHODS)

    def test_local_homophily_hand_computed_example(self):
        # 手算一个 4 个点、2 近邻的小例子来核对公式。
        #
        #   点 0（类别 A）的 2 个近邻是 点1（A）、点2（B）  -> 1/2 同类 -> h(0) = 0.5
        #   点 1（类别 A）的 2 个近邻是 点0（A）、点2（B）  -> 1/2 同类 -> h(1) = 0.5
        #   点 2（类别 B）的 2 个近邻是 点3（B）、点0（A）  -> 1/2 同类 -> h(2) = 0.5
        #   点 3（类别 B）的 2 个近邻都是 点2（B）、点1（A）-> 1/2 同类 -> h(3) = 0.5
        labels = np.array([0, 0, 1, 1])
        knn_indices = np.array([
            [1, 2],
            [0, 2],
            [3, 0],
            [2, 1],
        ])
        homophily = _compute_local_homophily(labels, knn_indices)
        np.testing.assert_allclose(homophily, [0.5, 0.5, 0.5, 0.5])

    def test_local_homophily_pure_neighborhoods(self):
        # 点 0、1 都只跟同类点相连 -> h 应该是 1.0
        # 点 2、3 都只跟异类点相连 -> h 应该是 0.0
        labels = np.array([0, 0, 1, 1])
        knn_indices = np.array([
            [1],   # 点0 的唯一邻居是点1，同类
            [0],   # 点1 的唯一邻居是点0，同类
            [0],   # 点2 的唯一邻居是点0，异类
            [1],   # 点3 的唯一邻居是点1，异类
        ])
        homophily = _compute_local_homophily(labels, knn_indices)
        np.testing.assert_allclose(homophily, [1.0, 1.0, 0.0, 0.0])

    def test_gate_preserves_same_class_entries(self):
        # 4x4 全 1 核，前两个点是类别0，后两个点是类别1。
        # 同配率随便设一个不是 1 的值，验证同类块完全不受影响。
        kernel = sp.csr_matrix(np.ones((4, 4), dtype=np.float32))
        labels = np.array([0, 0, 1, 1])
        homophily = np.array([0.3, 0.7, 0.4, 0.6])

        gated = _apply_homophily_gate(kernel, labels, homophily).toarray()
        same_mask = labels[:, None] == labels[None, :]

        np.testing.assert_allclose(gated[same_mask], 1.0)

    def test_gate_scales_cross_class_entries_by_target_homophily(self):
        # 同样的 4x4 全 1 核。跨类条目 K[i, j] 应该被乘上 homophily[j]
        # (j 是"列"，也就是被选中的候选点)。
        kernel = sp.csr_matrix(np.ones((4, 4), dtype=np.float32))
        labels = np.array([0, 0, 1, 1])
        homophily = np.array([0.3, 0.7, 0.4, 0.6])

        gated = _apply_homophily_gate(kernel, labels, homophily).toarray()

        # K[0, 2]：点0(A) 到 点2(B)，跨类，j=2 -> 应该乘 homophily[2] = 0.4
        self.assertAlmostEqual(gated[0, 2], 1.0 * 0.4)
        # K[2, 0]：点2(B) 到 点0(A)，跨类，j=0 -> 应该乘 homophily[0] = 0.3
        self.assertAlmostEqual(gated[2, 0], 1.0 * 0.3)
        # K[1, 3]：点1(A) 到 点3(B)，跨类，j=3 -> 应该乘 homophily[3] = 0.6
        self.assertAlmostEqual(gated[1, 3], 1.0 * 0.6)

    def test_gate_zero_homophily_hard_masks_that_candidate(self):
        # 如果某个候选点的同配率是 0（它的近邻全是别的类），
        # 它提供的所有跨类覆盖信用应该被完全清零（floor 默认是 0）。
        kernel = sp.csr_matrix(np.ones((3, 3), dtype=np.float32))
        labels = np.array([0, 1, 1])
        homophily = np.array([0.0, 1.0, 1.0])  # 点0 同配率为0

        gated = _apply_homophily_gate(kernel, labels, homophily).toarray()
        # 所有指向点0作为"候选/列"的跨类边都应该是0：K[1,0], K[2,0]
        self.assertAlmostEqual(gated[1, 0], 0.0)
        self.assertAlmostEqual(gated[2, 0], 0.0)

    def test_end_to_end_selects_fixed_total_budget(self):
        # 跟 tests/test_selection_opt.py 里其它变体的写法一致：
        # mock 掉 _build_global_kernel，只验证选择流程本身能跑通、
        # 预算数量对得上。
        rng = np.random.default_rng(7)
        dense = rng.random((12, 12), dtype=np.float32)
        dense[dense < 0.5] = 0.0
        kernel = sp.csr_matrix(dense)
        embeddings = rng.random((12, 4), dtype=np.float32)
        labels = np.array([0] * 7 + [1] * 5)
        importance = np.ones(12, dtype=np.float32)

        with patch(
            "homophily_gate.graph_a2_homophily._build_global_kernel",
            return_value=kernel,
        ):
            selected = METHODS["graph_a2_homophily"].fn(
                embeddings=embeddings,
                labels=labels,
                budget_per_class=2,
                importance=importance,
                global_selection=True,
                sparse_cpu=True,
            )
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(set(selected)), 4)

    def test_raises_without_global_selection(self):
        # 这个方法只在 --global 模式下有意义（跟其它 graph_a2_* 变体一致），
        # 不加 global_selection=True 应该直接报错，而不是默默算错。
        rng = np.random.default_rng(1)
        embeddings = rng.random((6, 3), dtype=np.float32)
        labels = np.array([0, 0, 0, 1, 1, 1])
        importance = np.ones(6, dtype=np.float32)
        with self.assertRaises(ValueError):
            METHODS["graph_a2_homophily"].fn(
                embeddings=embeddings,
                labels=labels,
                budget_per_class=1,
                importance=importance,
                global_selection=False,
            )


if __name__ == "__main__":
    unittest.main()
