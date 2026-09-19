"""AutoFlowCFD V2.0 - `build_native_tet_boundary_extrap` 单元测试
（Part7 阶段2实现：native 四面体体积->自身面外插矩阵，可预计算一次、
全网格同阶数单元共享）。
"""

import numpy as np

from autoflowcfd.fr.native_tet.basis import (
    build_native_tet_operators,
    build_native_tet_boundary_extrap,
    map_native_tet_to_physical,
)
from autoflowcfd.fr.face_flux_points import native_tet_face_points_physical
from autoflowcfd.fr.quadrature_points import gauss_legendre

TET_NODES = np.array([[0.1, -0.3, 0.2], [1.2, 0.0, -0.1], [-0.1, 1.1, 0.0], [0.0, 0.1, 1.3]], dtype=float)


def _field(phys):
    x, y, z = phys[:, 0], phys[:, 1], phys[:, 2]
    return 1.0 + 2.0 * x - 3.0 * y + 0.5 * z


def test_extrap_matrix_shape():
    for order in [1, 2, 3]:
        n1d = order + 1
        ref_rst, _ = build_native_tet_operators(order)
        n_native = ref_rst.shape[0]
        for excluded_vertex in range(4):
            E = build_native_tet_boundary_extrap(order, excluded_vertex)
            assert E.shape == (n1d * n1d, n_native)


def test_extrap_matrix_reproduces_affine_field_at_face_points():
    """仿射场在体积节点上取值，经外插矩阵作用后，必须与直接在面物理点
    上求值的解析结果精确一致（机器精度）——最基本的相容性检验。"""
    for order in [1, 2, 3]:
        n1d = order + 1
        sps_1d, _ = gauss_legendre(n1d)
        ref_rst, _ = build_native_tet_operators(order)
        phys_vol = map_native_tet_to_physical(ref_rst, TET_NODES)
        field_vol = _field(phys_vol)

        for excluded_vertex in range(4):
            E = build_native_tet_boundary_extrap(order, excluded_vertex)
            field_at_fp_via_extrap = E @ field_vol

            phys_fp = native_tet_face_points_physical(n1d, excluded_vertex, TET_NODES, sps_1d)
            field_at_fp_exact = _field(phys_fp)

            np.testing.assert_allclose(field_at_fp_via_extrap, field_at_fp_exact, atol=1e-9)


def test_extrap_matrix_is_shared_across_differently_shaped_cells():
    """核心性质：矩阵只依赖 (order, excluded_vertex)，不依赖具体单元
    形状——用另一个形状差异很大的四面体重新验证仍然精确重现仿射场。"""
    other_nodes = np.array([[0.0, 0.0, 0.0], [3.0, 0.1, -0.2], [-0.3, 2.5, 0.1], [0.05, -0.1, 4.0]], dtype=float)
    order = 2
    n1d = order + 1
    sps_1d, _ = gauss_legendre(n1d)
    ref_rst, _ = build_native_tet_operators(order)
    phys_vol = map_native_tet_to_physical(ref_rst, other_nodes)
    field_vol = _field(phys_vol)

    for excluded_vertex in range(4):
        E = build_native_tet_boundary_extrap(order, excluded_vertex)
        field_at_fp_via_extrap = E @ field_vol
        phys_fp = native_tet_face_points_physical(n1d, excluded_vertex, other_nodes, sps_1d)
        field_at_fp_exact = _field(phys_fp)
        np.testing.assert_allclose(field_at_fp_via_extrap, field_at_fp_exact, atol=1e-9)
