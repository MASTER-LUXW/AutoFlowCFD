"""AutoFlowCFD V2.0 - native 四面体（路径C）DG 提升算子
`native_tet/basis.py::build_native_tet_lift` 决定性验证。

背景：坍缩坐标 FR 方案用 1D Radau/VCJH 修正函数 + `_distribute_point`
把面通量跳跃"修正"回体积自由度——这个机制建立在张量积/坍缩坐标结构上，
对 native（非张量积）单纯形基不适用，`build_native_tet_lift` 是标准
DG 提升算子（Hesthaven-Warburton《Nodal DG》第6章 `Lift3D.m`）针对
"面 Flux Points 不是体积节点子集"这一情形的推广（见该函数文档完整推导）。

本文件的核心判据（`test_lift_matches_independent_fine_quadrature`）是
**从头独立**验证提升算子满足它自己的弱形式定义方程
`M_ref @ (Lift_ref @ (w_phys*jump)) = ∮_face Ψ(x)*jump(x) dA`：
右端用与生产代码完全无关的参数化（直接对物理三角形用 (u,v) 重心坐标 +
`scipy.integrate.dblquad` 自适应积分，不经过 `cube_to_tri_rs`/坍缩三角形
采样这条生产管线路径）独立算出，如果 `build_native_tet_lift` 的实现有
笔误（矩阵转置错、漏乘权重、模态归一化用错），这个独立右端不会跟着错
一起被掩盖。
"""

import numpy as np
import pytest
from scipy import integrate
from scipy.linalg import lu_factor, lu_solve

from autoflowcfd.fr.native_tet.basis import (
    build_native_tet_operators,
    build_native_tet_lift,
    restricted_tet_modes,
    simplex3d_value,
    rst_to_abc,
    _native_mode_norm_squared,
)
from autoflowcfd.fr.quadrature_points import gauss_legendre
from autoflowcfd.grid.curved_mapping.curved_mapping import cube_to_tri_rs, tri_barycentric

TET_NODES = np.array([
    [0.1, -0.3, 0.2], [1.2, 0.0, -0.1], [-0.1, 1.1, 0.0], [0.0, 0.1, 1.3],
], dtype=float)


def _reference_mass_matrix(order):
    modes = restricted_tet_modes(order)
    ref_rst_sps, _ = build_native_tet_operators(order)
    a, b, c = rst_to_abc(ref_rst_sps[:, 0], ref_rst_sps[:, 1], ref_rst_sps[:, 2])
    V_sps = np.column_stack([simplex3d_value(a, b, c, i, j, k) for (i, j, k) in modes])
    N = np.array([_native_mode_norm_squared(i, j, k) for (i, j, k) in modes])
    V_inv = np.linalg.inv(V_sps)
    return V_inv.T @ np.diag(N) @ V_inv, V_sps, modes


def _face_physical_weight_and_points(order, excluded_vertex, cell_nodes):
    """与 `face_flux_points/exact_normal.py::_native_tet_adj_row_batched`
    同一套坍缩三角形采样 + 真实物理面积微元公式（生产路径），供
    LHS（Lift_ref 消费方）使用——与下面独立的 (u,v) dblquad 参数化
    (RHS) 刻意不同，避免"用同一套错误互相印证"。
    """
    n1d = order + 1
    sps_1d, w1d = gauss_legendre(n1d)
    face_verts = [v for v in range(4) if v != excluded_vertex]
    p0, p1, p2 = cell_nodes[face_verts[0]], cell_nodes[face_verts[1]], cell_nodes[face_verts[2]]

    g1, g2 = np.meshgrid(sps_1d, sps_1d, indexing="ij")
    r_tri, s_tri = cube_to_tri_rs(g1.ravel(), g2.ravel())
    l1, l2, l3 = tri_barycentric(r_tri, s_tri)
    phys_fp = l1[:, None] * p0 + l2[:, None] * p1 + l3[:, None] * p2

    a_arr, b_arr = g1.ravel(), g2.ravel()
    e1, e2 = p1 - p0, p2 - p0
    dpda = ((1.0 - b_arr) / 4.0)[:, None] * e1[None, :]
    dpdb = (-(1.0 + a_arr) / 4.0)[:, None] * e1[None, :] + 0.5 * e2[None, :]
    mag = np.linalg.norm(np.cross(dpda, dpdb), axis=1)
    wxw, wyw = np.meshgrid(w1d, w1d, indexing="ij")
    w_phys = mag * (wxw * wyw).ravel()
    return phys_fp, w_phys


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("excluded_vertex", [0, 1, 2, 3])
def test_lift_matches_independent_fine_quadrature(order, excluded_vertex):
    Lift_ref = build_native_tet_lift(order, excluded_vertex)
    M_ref, V_sps, modes = _reference_mass_matrix(order)
    n_native = len(modes)
    lu_piv = lu_factor(V_sps.T)  # 求解 Psi(x) = V_sps^{-T} @ P(x)

    face_verts = [v for v in range(4) if v != excluded_vertex]
    p0, p1, p2 = TET_NODES[face_verts[0]], TET_NODES[face_verts[1]], TET_NODES[face_verts[2]]
    jac = np.linalg.norm(np.cross(p1 - p0, p2 - p0))  # 独立 (u,v) 参数化的面积 Jacobian

    P0, P1, P2, P3 = TET_NODES
    Jt_inv = np.linalg.inv(np.column_stack([P1 - P0, P2 - P0, P3 - P0]))

    def jump_fn(phys):
        x, y, z = phys
        return 1.0 + 2.0 * x - 1.5 * y + 0.7 * z

    def psi_all(phys):
        L2, L3, L4 = Jt_inv @ (phys - P0)
        r, s, t = 2 * L2 - 1, 2 * L3 - 1, 2 * L4 - 1
        a, b, c = rst_to_abc(np.array([r]), np.array([s]), np.array([t]))
        p_vec = np.array([simplex3d_value(a, b, c, i, j, k)[0] for (i, j, k) in modes])
        return lu_solve(lu_piv, p_vec)

    def integrand(v, u, s_idx):
        phys = p0 + u * (p1 - p0) + v * (p2 - p0)
        return psi_all(phys)[s_idx] * jump_fn(phys) * jac

    rhs_fine = np.zeros(n_native)
    for s_idx in range(n_native):
        val, _ = integrate.dblquad(
            integrand, 0, 1, lambda u: 0, lambda u: 1 - u,
            args=(s_idx,), epsabs=1e-10, epsrel=1e-10,
        )
        rhs_fine[s_idx] = val

    phys_fp, w_phys = _face_physical_weight_and_points(order, excluded_vertex, TET_NODES)
    jump_fp = np.array([jump_fn(p) for p in phys_fp])

    lhs = M_ref @ (Lift_ref @ (w_phys * jump_fp))
    np.testing.assert_allclose(lhs, rhs_fine, atol=1e-9)


def test_lift_reference_matrix_is_shared_across_differently_shaped_cells():
    """`Lift_ref` 只依赖 (order, excluded_vertex)，与 `build_native_tet_
    boundary_extrap` 同一个"参考量、跟单元形状无关"的性质——用两个形状
    差异很大的物理面重新验证同一个 Lift_ref 仍满足弱形式定义方程。"""
    order = 2
    excluded_vertex = 2
    other_nodes = np.array([
        [0.0, 0.0, 0.0], [3.0, 0.1, -0.2], [-0.3, 2.5, 0.1], [0.05, -0.1, 4.0],
    ], dtype=float)

    Lift_ref = build_native_tet_lift(order, excluded_vertex)
    M_ref, V_sps, modes = _reference_mass_matrix(order)
    n_native = len(modes)
    lu_piv = lu_factor(V_sps.T)

    face_verts = [v for v in range(4) if v != excluded_vertex]
    p0, p1, p2 = other_nodes[face_verts[0]], other_nodes[face_verts[1]], other_nodes[face_verts[2]]
    jac = np.linalg.norm(np.cross(p1 - p0, p2 - p0))
    P0, P1, P2, P3 = other_nodes
    Jt_inv = np.linalg.inv(np.column_stack([P1 - P0, P2 - P0, P3 - P0]))

    def jump_fn(phys):
        x, y, z = phys
        return 3.0 - x + 0.5 * y - 2.0 * z

    def psi_all(phys):
        L2, L3, L4 = Jt_inv @ (phys - P0)
        r, s, t = 2 * L2 - 1, 2 * L3 - 1, 2 * L4 - 1
        a, b, c = rst_to_abc(np.array([r]), np.array([s]), np.array([t]))
        p_vec = np.array([simplex3d_value(a, b, c, i, j, k)[0] for (i, j, k) in modes])
        return lu_solve(lu_piv, p_vec)

    def integrand(v, u, s_idx):
        phys = p0 + u * (p1 - p0) + v * (p2 - p0)
        return psi_all(phys)[s_idx] * jump_fn(phys) * jac

    rhs_fine = np.zeros(n_native)
    for s_idx in range(n_native):
        val, _ = integrate.dblquad(
            integrand, 0, 1, lambda u: 0, lambda u: 1 - u,
            args=(s_idx,), epsabs=1e-10, epsrel=1e-10,
        )
        rhs_fine[s_idx] = val

    phys_fp, w_phys = _face_physical_weight_and_points(order, excluded_vertex, other_nodes)
    jump_fp = np.array([jump_fn(p) for p in phys_fp])
    lhs = M_ref @ (Lift_ref @ (w_phys * jump_fp))
    np.testing.assert_allclose(lhs, rhs_fine, atol=1e-9)


def test_lift_shape():
    for order in [1, 2, 3]:
        n1d = order + 1
        ref_rst, _ = build_native_tet_operators(order)
        n_native = ref_rst.shape[0]
        for ev in range(4):
            Lift_ref = build_native_tet_lift(order, ev)
            assert Lift_ref.shape == (n_native, n1d * n1d)
