"""AutoFlowCFD V2.0 - 四面体(native)-棱柱(collapsed) 跨基插值决定性
验证（Part6 阶段2 最后一块核心数学不确定性，尚未接入生产管线）。

与四面体-四面体情形（`test_native_tet_face_interface.py`，节点物理坐标
精确重合、不需要插值）不同，四面体(native)与棱柱(collapsed)节点分布
本质不同，共享面上的物理点一般不会精确重合，genuinely 需要跨基插值：
用四面体侧的原生基拟合体积节点上的场值，再在棱柱侧共享面的目标物理点
上求值。

判据：仿射场（任意阶数应精确表示，机器精度）+ 二次场（阶数相关的
标准多项式截断误差，P1 应该有真实但合理的误差，P2/P3 应精确）——
这是普通多项式插值理论预期的行为，用来确认跨基插值这一步本身没有
引入任何退化坐标相关的异常（不是"处处精确"，是"跟普通多项式插值
理论预期完全一致，没有额外的病态"）。
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from autoflowcfd.fr.native_simplex_basis import (
    build_native_tet_operators,
    simplex3d_value,
    restricted_tet_modes,
    rst_to_abc,
)
from autoflowcfd.fr.quadrature_points import gauss_legendre
from autoflowcfd.grid.curved_mapping.curved_mapping import cube_to_tri_rs, tri_barycentric, tet_barycentric

# 共享三角形 (0,0,0),(1,0,0),(0,1,0) 位于 z=0：棱柱占 z∈[-1,0]，
# 四面体占 z∈[0,1]，两者真实共享这个三角形面。
PRISM_NODES = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0],
    [0, 0, -1], [1, 0, -1], [0, 1, -1],
], dtype=float)
TET_NODES = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
], dtype=float)


def _field_affine(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return 1.0 + 2.0 * x + 3.0 * y + 4.0 * z


def _field_quadratic(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return z ** 2 + x * y


def _tet_inverse_barycentric(phys, cell_nodes):
    p0, p1, p2, p3 = cell_nodes
    J = np.column_stack([p1 - p0, p2 - p0, p3 - p0])
    Jinv = np.linalg.inv(J)
    L234 = (phys - p0) @ Jinv.T
    L2, L3, L4 = L234[:, 0], L234[:, 1], L234[:, 2]
    return 2 * L2 - 1, 2 * L3 - 1, 2 * L4 - 1


def _shared_face_target_points(order):
    n1d = order + 1
    sps_1d, _ = gauss_legendre(n1d)
    aa, bb = np.meshgrid(sps_1d, sps_1d, indexing="ij")
    r_tri, s_tri = cube_to_tri_rs(aa.ravel(), bb.ravel())
    l1, l2, l3 = tri_barycentric(r_tri, s_tri)
    v0, v1, v2 = PRISM_NODES[0], PRISM_NODES[1], PRISM_NODES[2]
    return l1[:, None] * v0 + l2[:, None] * v1 + l3[:, None] * v2


def _tet_native_interpolate_to_targets(order, target_phys, field_fn):
    ref_rst, _ = build_native_tet_operators(order)
    a, b, c = rst_to_abc(ref_rst[:, 0], ref_rst[:, 1], ref_rst[:, 2])
    modes = restricted_tet_modes(order)
    V_train = np.column_stack([simplex3d_value(a, b, c, i, j, k) for (i, j, k) in modes])
    lu = lu_factor(V_train)

    L1, L2, L3, L4 = tet_barycentric(ref_rst[:, 0], ref_rst[:, 1], ref_rst[:, 2])
    p0, p1, p2, p3 = TET_NODES
    phys_train = L1[:, None] * p0 + L2[:, None] * p1 + L3[:, None] * p2 + L4[:, None] * p3
    coeffs = lu_solve(lu, field_fn(phys_train))

    r_t, s_t, t_t = _tet_inverse_barycentric(target_phys, TET_NODES)
    a_t, b_t, c_t = rst_to_abc(r_t, s_t, t_t)
    V_target = np.column_stack([simplex3d_value(a_t, b_t, c_t, i, j, k) for (i, j, k) in modes])
    return V_target @ coeffs


def test_shared_face_target_points_lie_exactly_on_z_zero_plane():
    for order in [1, 2, 3]:
        target_phys = _shared_face_target_points(order)
        np.testing.assert_allclose(target_phys[:, 2], 0.0, atol=1e-12)


def test_native_tet_cross_basis_interpolation_exact_for_affine_field():
    """仿射场任意阶数都应该被精确表示（机器精度）——最基本的相容性。"""
    for order in [1, 2, 3]:
        target_phys = _shared_face_target_points(order)
        interp = _tet_native_interpolate_to_targets(order, target_phys, _field_affine)
        exact = _field_affine(target_phys)
        np.testing.assert_allclose(interp, exact, atol=1e-10)


def test_native_tet_cross_basis_interpolation_quadratic_field_matches_polynomial_theory():
    """二次场：P1（只有4个模态，总阶数<=1）应该有真实但合理的截断误差
    （不是机器精度，也不应该异常放大）；P2/P3（总阶数>=2，足够覆盖
    二次场）应该精确——这是普通多项式插值理论预期的标准行为，用来
    确认跨基插值本身没有引入任何退化坐标相关的额外病态。
    """
    target_phys = _shared_face_target_points(1)
    interp1 = _tet_native_interpolate_to_targets(1, target_phys, _field_quadratic)
    exact1 = _field_quadratic(target_phys)
    err1 = np.abs(interp1 - exact1)
    assert 1e-3 < err1.max() < 1.0, f"P1 二次场截断误差应该是真实但适度的量级，实测 {err1.max():.3e}"

    for order in [2, 3]:
        target_phys = _shared_face_target_points(order)
        interp = _tet_native_interpolate_to_targets(order, target_phys, _field_quadratic)
        exact = _field_quadratic(target_phys)
        np.testing.assert_allclose(interp, exact, atol=1e-9)
