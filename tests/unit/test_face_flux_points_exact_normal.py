"""face_flux_points/exact_normal.py 的单元测试——core/fr_residual/
inviscid_kernel.py 的 `true_normal`/`true_area_weight` 此前每个面只有一个
（平面、三角化的）常数值，不论该面（棱柱的四边形侧面）是否真的共面，面上
所有通量点都复用它；这个模块是对它的真正修复。完整机制见模块文档。

这些测试在函数接入真实几何流程之前验证核心的数学论断：
1. 对**平面**（任意四面体面，或四个角点恰好共面的棱柱四边形面），新的逐
   通量点计算必须退化为所有通量点上的**同一个**常数法向，并与手算的平面
   法向/面积精确（不是近似）相等——批量精确 Jacobian 的做法不能在绝大多数
   旧平面近似本来就精确的面上悄悄改变结果。
2. 对真正**翘曲**（不共面）的棱柱四边形面，法向必须在各通量点之间有实质
   变化（这正是被修复的缺陷——旧代码在这里也只用一个常数值）。
3. 批量 Jacobian 函数必须与已有的、验证过的单点 `tet_exact_jacobian`/
   `prism_exact_jacobian`（curved_mapping_exact_jacobian.py）精确一致——
   批量版是同一组公式的向量化而不是重新实现，在重叠的输入上必须逐位一致
   （至多差浮点结合顺序）。
"""

import numpy as np
import pytest

from autoflowcfd.fr.face_flux_points.exact_normal import (
    compute_exact_face_normals_and_weights,
    compute_exact_adj_rows,
    _tet_exact_jacobian_batched,
    _prism_exact_jacobian_batched,
)
from autoflowcfd.grid.curved_mapping.curved_mapping_exact_jacobian import (
    tet_exact_jacobian, prism_exact_jacobian,
)
from autoflowcfd.fr.face_flux_points import face_ref_grid
from autoflowcfd.fr.quadrature_points import gauss_legendre


class TestBatchedJacobianMatchesSinglePointVersion:
    def test_tet_batched_matches_loop_of_single_point(self):
        rng = np.random.default_rng(0)
        n_faces, n_fp = 5, 4
        ref_pts = rng.uniform(-1, 1, size=(n_fp, 3))
        cell_nodes = rng.uniform(-1, 1, size=(n_faces, 4, 3))

        batched = _tet_exact_jacobian_batched(ref_pts, cell_nodes)
        for f in range(n_faces):
            expected = tet_exact_jacobian(ref_pts, cell_nodes[f])
            np.testing.assert_allclose(batched[f], expected, rtol=1e-13, atol=1e-13)

    def test_prism_batched_matches_loop_of_single_point(self):
        rng = np.random.default_rng(1)
        n_faces, n_fp = 5, 4
        ref_pts = rng.uniform(-1, 1, size=(n_fp, 3))
        cell_nodes = rng.uniform(-1, 1, size=(n_faces, 6, 3))

        batched = _prism_exact_jacobian_batched(ref_pts, cell_nodes)
        for f in range(n_faces):
            expected = prism_exact_jacobian(ref_pts, cell_nodes[f])
            np.testing.assert_allclose(batched[f], expected, rtol=1e-13, atol=1e-13)


def _n1d_and_grids(order):
    n1d = order + 1
    sps_1d, weights_1d = gauss_legendre(n1d)
    return n1d, sps_1d, weights_1d


class TestPlanarFacesReduceToConstantNormal:
    def test_tet_face_gives_constant_normal_matching_flat_geometry(self):
        """正四面体的 'a=-1' 面（节点 0,2,3，见 TET_CUBE_FACES）按构造严格共面——
        每个通量点必须得到同一个法向，等于标准叉积给出的平面法向。
        """
        order = 2
        n1d, sps_1d, weights_1d = _n1d_and_grids(order)
        n_fp = n1d * n1d

        cell_nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        owner_cell = np.array([0], dtype=np.int64)
        owner_axis = np.array([0], dtype=np.int64)   # a
        owner_side = np.array([-1.0])                 # a=-1 -> TET_CUBE_FACES nodes (0,2,3)

        true_normal, true_area_weight = compute_exact_face_normals_and_weights(
            n_faces=1, n1d=n1d, sps_1d=sps_1d, weights_1d=weights_1d, n_prism=0,
            owner_cell=owner_cell, owner_axis=owner_axis, owner_side=owner_side,
            prism_conn=np.empty((0, 6), dtype=np.int64),
            tet_conn=np.array([[0, 1, 2, 3]], dtype=np.int64),
            node_coords=cell_nodes,
        )

        # 这个平面上的每个通量点法向必须完全相同
        n = true_normal[0]
        np.testing.assert_allclose(n, np.tile(n[0], (n_fp, 1)), atol=1e-12)

        # 物理空间里的面 (0,2,3)：(0,0,0),(0,1,0),(0,0,1)——x=0 平面，
        # 外法向（背离 x=1 处的节点 1）必须是 -x 方向。
        np.testing.assert_allclose(n[0], [-1.0, 0.0, 0.0], atol=1e-10)

        # 总面积必须等于真实的平面三角形面积（这个直角三角形是 0.5）
        assert true_area_weight[0].sum() == pytest.approx(0.5, rel=1e-10)

    def test_planar_prism_quad_gives_constant_normal(self):
        """长方体棱柱的四边形侧面严格共面——所有通量点必须给出同一个法向/方向，
        与该矩形的叉积平面法向一致。
        """
        order = 2
        n1d, sps_1d, weights_1d = _n1d_and_grids(order)
        n_fp = n1d * n1d

        # v0,v1,v2 是底面三角形，w0,w1,w2 是顶面——一个轴对齐的盒子，
        # 侧面 "a=-1" -> PRISM_CUBE_FACES 的节点 (0,2,5,3) = v0,v2,w2,w0。
        cell_nodes = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1],
        ], dtype=float)
        owner_cell = np.array([0], dtype=np.int64)
        owner_axis = np.array([0], dtype=np.int64)
        owner_side = np.array([-1.0])

        true_normal, true_area_weight = compute_exact_face_normals_and_weights(
            n_faces=1, n1d=n1d, sps_1d=sps_1d, weights_1d=weights_1d, n_prism=1,
            owner_cell=owner_cell, owner_axis=owner_axis, owner_side=owner_side,
            prism_conn=np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64),
            tet_conn=np.empty((0, 4), dtype=np.int64),
            node_coords=cell_nodes,
        )

        n = true_normal[0]
        np.testing.assert_allclose(n, np.tile(n[0], (n_fp, 1)), atol=1e-10)
        # 四边形 v0(0,0,0)-v2(0,1,0)-w2(0,1,1)-w0(0,0,1) 在 x=0 平面上；
        # 外法向（背离 x=1 处的 v1）是 -x。
        np.testing.assert_allclose(n[0], [-1.0, 0.0, 0.0], atol=1e-10)
        # unit square face area = 1.0
        assert true_area_weight[0].sum() == pytest.approx(1.0, rel=1e-10)


class TestWarpedPrismQuadVariesAcrossFluxPoints:
    def test_non_planar_quad_normal_is_not_constant(self):
        """把四边形侧面的一个角点移出平面——旧代码的单个常数法向正是被修复的
        缺陷；新的逐通量点计算必须在各通量点之间表现出真实的变化（这就是这次
        改动的全部意义）。
        """
        order = 2
        n1d, sps_1d, weights_1d = _n1d_and_grids(order)

        cell_nodes = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1],
        ], dtype=float)
        # 把 w0（节点 3）推出 x=0 平面，使四边形侧面 "a=-1" = (v0,v2,w2,w0) 翘曲
        # ——v0,v2,w2 留在 x=0，w0 移到 x=0.4。
        cell_nodes[3] = [0.4, 0.0, 1.0]

        owner_cell = np.array([0], dtype=np.int64)
        owner_axis = np.array([0], dtype=np.int64)
        owner_side = np.array([-1.0])

        true_normal, _ = compute_exact_face_normals_and_weights(
            n_faces=1, n1d=n1d, sps_1d=sps_1d, weights_1d=weights_1d, n_prism=1,
            owner_cell=owner_cell, owner_axis=owner_axis, owner_side=owner_side,
            prism_conn=np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64),
            tet_conn=np.empty((0, 4), dtype=np.int64),
            node_coords=cell_nodes,
        )

        n = true_normal[0]
        max_pairwise_diff = np.max(np.linalg.norm(n[:, None, :] - n[None, :, :], axis=-1))
        assert max_pairwise_diff > 1e-3, (
            "expected meaningfully different normals across FPs on a warped "
            f"quad, got max pairwise difference {max_pairwise_diff:.3e}"
        )
        # 每个方向仍必须是单位向量
        np.testing.assert_allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-10)


class TestComputeExactAdjRowsValidMask:
    """`compute_exact_adj_rows` 是 `true_normal`（owner 侧，按 side 定向并归一化）
    与"自洽方向"修复（owner 与 neighbor 两侧，原始/未归一化）共用的原语——
    这里钉住它的 `valid_mask` 行为：neighbor 侧的调用方靠它跳过边界面（边界面
    的 neighbor_axis/neighbor_side 是 -1/0.0 哨兵值，不是真实值）。
    """

    def test_invalid_entries_stay_zero(self):
        order = 1
        n1d, sps_1d, weights_1d = _n1d_and_grids(order)
        cell_nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

        cell_arr = np.array([0, 0], dtype=np.int64)
        axis_arr = np.array([0, -1], dtype=np.int64)  # 面 1：哨兵轴（边界面，没有邻居）
        side_arr = np.array([-1.0, 0.0])
        valid_mask = np.array([True, False])

        adj_row = compute_exact_adj_rows(
            n_faces=2, n1d=n1d, sps_1d=sps_1d, n_prism=0,
            cell_arr=cell_arr, axis_arr=axis_arr, side_arr=side_arr,
            prism_conn=np.empty((0, 6), dtype=np.int64),
            tet_conn=np.array([[0, 1, 2, 3]], dtype=np.int64),
            node_coords=cell_nodes,
            valid_mask=valid_mask,
        )

        assert not np.allclose(adj_row[0], 0.0)  # 有效项：真实（非零）的 adj(J) 行
        np.testing.assert_array_equal(adj_row[1], 0.0)  # 无效项：保持为零不动

    def test_raw_row_is_unnormalized_and_not_side_oriented(self):
        """与 `compute_exact_face_normals_and_weights` 的 `true_normal` 不同，这是
        *原始*的 adj(J) 行——不是单位向量，也没有按 `side` 翻转——对应
        inviscid_kernel.py 的 `a0,a1,a2`（在它自己的 `*oside` 方向修正之前）目前
        由解点外插得到的量。
        """
        order = 1
        n1d, sps_1d, weights_1d = _n1d_and_grids(order)
        cell_nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

        adj_row = compute_exact_adj_rows(
            n_faces=1, n1d=n1d, sps_1d=sps_1d, n_prism=0,
            cell_arr=np.array([0], dtype=np.int64),
            axis_arr=np.array([0], dtype=np.int64),
            side_arr=np.array([-1.0]),
            prism_conn=np.empty((0, 6), dtype=np.int64),
            tet_conn=np.array([[0, 1, 2, 3]], dtype=np.int64),
            node_coords=cell_nodes,
        )
        # 这个四面体的面 (0,2,3) 是 x=0 平面；axis=0 的原始 adj(J) 行指向 +x
        # （从 x=1 处的节点 1 指向单元内部）——*外*方向（-x，与上面 true_normal 的
        # 测试一致）要等调用方自己乘 `*side` 修正之后才出现。
        row = adj_row[0, 0]
        assert row[0] > 0  # 尚未翻到外向（-x）——原始值
        assert not np.isclose(np.linalg.norm(row), 1.0)  # NOT unit-normalized
