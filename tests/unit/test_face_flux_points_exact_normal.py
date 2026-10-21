"""face_flux_points/exact_normal.py：逐 Flux Point 精确法向 / 面积权重（原生面编码）。

1. **平面**（任意四面体面，或四个角点恰好共面的棱柱四边形面）上，逐通量点计算必须退化为所有通量点上的**同一个**
   常数法向，并与手算的平面法向 / 面积精确相等。
2. 真正**翘曲**（不共面）的棱柱四边形面，法向必须在各通量点之间有实质变化（每个面只取一个常数法向的做法在这里
   是错的）。
3. `valid_mask` 排除的项保持为零（neighbor 侧对边界面无意义）。
"""

import numpy as np
import pytest

from autoflowcfd.fr.face_flux_points.exact_normal import (
    compute_exact_adj_rows,
    compute_exact_face_normals_and_weights,
)
from autoflowcfd.fr.quadrature_points import gauss_legendre

TET_V1_EXCLUDED = 6 + 1     # 原生四面体面编码 = 6 + 被排除的局部顶点
PRISM_F3 = 10 + 3           # 原生棱柱 f3：侧四边形 (v0, v2, w2, w0)，排除三角形顶点 V1

UNIT_TET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
UNIT_PRISM = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=float)


def _grids(order):
    n1d = order + 1
    sps_1d, weights_1d = gauss_legendre(n1d)
    return n1d, sps_1d, weights_1d


def _tet_normals(order, nodes):
    n1d, sps_1d, weights_1d = _grids(order)
    return compute_exact_face_normals_and_weights(
        n_faces=1, n1d=n1d, sps_1d=sps_1d, weights_1d=weights_1d, n_prism=0,
        owner_cell=np.array([0], dtype=np.int64), owner_code=np.array([TET_V1_EXCLUDED], dtype=np.int64),
        prism_conn=np.empty((0, 6), dtype=np.int64), tet_conn=np.array([[0, 1, 2, 3]], dtype=np.int64),
        node_coords=nodes)


def _prism_normals(order, nodes):
    n1d, sps_1d, weights_1d = _grids(order)
    return compute_exact_face_normals_and_weights(
        n_faces=1, n1d=n1d, sps_1d=sps_1d, weights_1d=weights_1d, n_prism=1,
        owner_cell=np.array([0], dtype=np.int64), owner_code=np.array([PRISM_F3], dtype=np.int64),
        prism_conn=np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64), tet_conn=np.empty((0, 4), dtype=np.int64),
        node_coords=nodes)


class TestPlanarFacesReduceToConstantNormal:
    def test_tet_face_gives_constant_normal_matching_flat_geometry(self):
        """排除顶点 1 的面 (0,2,3) 是 x=0 平面，外法向（背离 x=1 处的节点 1）是 -x，面积 0.5。"""
        true_normal, true_area_weight = _tet_normals(2, UNIT_TET)
        n = true_normal[0]
        np.testing.assert_allclose(n, np.tile(n[0], (n.shape[0], 1)), atol=1e-12)
        np.testing.assert_allclose(n[0], [-1.0, 0.0, 0.0], atol=1e-10)
        assert true_area_weight[0].sum() == pytest.approx(0.5, rel=1e-10)

    def test_planar_prism_quad_gives_constant_normal(self):
        """轴对齐棱柱的侧面 (v0,v2,w2,w0) 在 x=0 平面上，外法向（背离 x=1 处的 v1）是 -x，面积 1。"""
        true_normal, true_area_weight = _prism_normals(2, UNIT_PRISM)
        n = true_normal[0]
        np.testing.assert_allclose(n, np.tile(n[0], (n.shape[0], 1)), atol=1e-10)
        np.testing.assert_allclose(n[0], [-1.0, 0.0, 0.0], atol=1e-10)
        assert true_area_weight[0].sum() == pytest.approx(1.0, rel=1e-10)


class TestWarpedPrismQuadVariesAcrossFluxPoints:
    def test_non_planar_quad_normal_is_not_constant(self):
        """把 w0（节点 3）推出 x=0 平面，侧面 (v0,v2,w2,w0) 翘曲：法向必须在各通量点之间有实质变化。"""
        nodes = UNIT_PRISM.copy()
        nodes[3] = [0.4, 0.0, 1.0]
        true_normal, _ = _prism_normals(2, nodes)
        n = true_normal[0]
        max_pairwise_diff = np.max(np.linalg.norm(n[:, None, :] - n[None, :, :], axis=-1))
        assert max_pairwise_diff > 1e-3, f"翘曲四边形上各通量点法向几乎相同：{max_pairwise_diff:.3e}"
        np.testing.assert_allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-10)


class TestComputeExactAdjRowsValidMask:
    def test_invalid_entries_stay_zero(self):
        n1d, sps_1d, _ = _grids(1)
        adj_row = compute_exact_adj_rows(
            n_faces=2, n1d=n1d, sps_1d=sps_1d, n_prism=0,
            cell_arr=np.array([0, 0], dtype=np.int64),
            code_arr=np.array([TET_V1_EXCLUDED, -1], dtype=np.int64),   # 面 1：无效项的编码不被读取
            prism_conn=np.empty((0, 6), dtype=np.int64), tet_conn=np.array([[0, 1, 2, 3]], dtype=np.int64),
            node_coords=UNIT_TET, valid_mask=np.array([True, False]))
        assert not np.allclose(adj_row[0], 0.0)
        np.testing.assert_array_equal(adj_row[1], 0.0)
