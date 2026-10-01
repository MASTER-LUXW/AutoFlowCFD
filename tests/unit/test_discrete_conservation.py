"""无粘残差的离散守恒（体积 L2 投影 + 修正项本侧通量取细层通量多项式的迹）。

1. 迹算子的约定：随机直边四面体/棱柱、常数通量下，`Σ_{q,m} Tn[i,q,m] F~[q,m]` 等于生产
   几何的精确度量行投影 `a(fp_i)·F`（Nanson 公式；直边单元的 adj(J) 落在细层空间内，
   插值精确），且通量点顺序与面外插矩阵一致；
2. 构造期护栏：面记录必须满足"每个单元每个面恰一条 primary 侧"，重复或缺失都报错；
3. 封闭对称盒子（光滑非均匀场、全部 SYMMETRY）上全域积分 `Σ w_s det_s R_s`（对称面上
   质量与能量的法向通量为零，物理上守恒）：
   * 无粘、棱柱网格：质量与能量到机器精度（修复前 P1 质量 -3.0e-6、能量 2.0e-8）；
   * 无粘、四面体网格：质量到机器精度。能量还剩公共通量的面求积失配——三角形面的通量
     点是坍缩采样、取决于单元局部顶点编号，共享面两侧的点在约 1/5 的面上不重合，两侧
     各自对非线性 `F*` 求积就对不上（P1 1.1e-7、P2 1.5e-9、P3 4.6e-11，谱收敛）；
   * 粘性能量：两类网格都到机器精度（修复前 P1 7.0e-5）。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.face_kernels import _check_one_primary_side_per_cell_face
from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr, primitive_to_conserved
from autoflowcfd.core.fr_residual.viscous_flux import compute_viscous_residual_fr
from autoflowcfd.fr.face_flux_points.exact_normal import _native_tet_adj_row_batched
from autoflowcfd.fr.face_flux_trace import build_prism_face_flux_trace, build_tet_face_flux_trace
from autoflowcfd.fr.native_padding import real_sps_per_cell
from autoflowcfd.fr.native_prism.basis import native_prism_exact_jacobian
from autoflowcfd.fr.native_prism.face import native_prism_face_adj_rows
from autoflowcfd.fr.native_prism.overintegration import build_native_prism_overintegration_operators
from autoflowcfd.fr.native_prism.quadrature import build_native_prism_sp_weights
from autoflowcfd.fr.native_tet.basis import compute_native_tet_jacobian
from autoflowcfd.fr.native_tet.quadrature import build_native_tet_sp_weights
from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.fr.quadrature_points import gauss_legendre
from tests.validation._channel_mesh import (
    build_channel_mesh, build_channel_mesh_prism, build_face_exact_ghost_provider,
)

LX, H, LZ = 0.4, 0.1, 0.08
F0 = np.random.default_rng(0).standard_normal((3, 5))     # 常数物理通量


@pytest.mark.parametrize("order", [1, 2, 3])
def test_tet_trace_equals_exact_adj_projection(order):
    rng = np.random.default_rng(order)
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float) + 0.2 * rng.standard_normal((4, 3))
    det, adj = compute_native_tet_jacobian(nodes)
    assert det > 0
    T = build_tet_face_flux_trace(order, 2 * order)
    Ft = adj @ F0                                           # 常数逆变通量 (3,5)
    sps_1d, _ = gauss_legendre(order + 1)
    for v in range(4):
        trace = np.einsum("iqm,mv->iv", T[v], Ft)          # 细层 Lagrange 单位分解：Σ_q L_q = 1
        a = _native_tet_adj_row_batched(v, nodes[None], order + 1, sps_1d)[0]
        np.testing.assert_allclose(trace, a @ F0, rtol=1e-12, atol=1e-12 * np.abs(a @ F0).max())


@pytest.mark.parametrize("order", [1, 2, 3])
def test_prism_trace_equals_exact_adj_projection(order):
    rng = np.random.default_rng(10 + order)
    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]], float)
    nodes = base + 0.15 * rng.standard_normal((6, 3))      # 一般直边棱柱：侧面非平面、adj 随点变化
    oo = 2 * order
    ref_fine = build_native_prism_overintegration_operators(order, oo)[0]
    jac = native_prism_exact_jacobian(ref_fine, nodes)
    adj = np.linalg.det(jac)[:, None, None] * np.linalg.inv(jac)
    Ft = np.einsum("qmd,dv->qmv", adj, F0)                 # 细点上的逆变通量
    T = build_prism_face_flux_trace(order, oo)
    for fid in range(5):
        trace = np.einsum("iqm,qmv->iv", T[fid], Ft)
        a = native_prism_face_adj_rows(order, fid, nodes)
        np.testing.assert_allclose(trace, a @ F0, rtol=1e-11, atol=1e-11 * np.abs(a @ F0).max())


def test_guard_rejects_duplicate_or_missing_primary_sides():
    # 两个四面体共享一个面：0 的面码 6..9、1 的面码 6..9，内部面 (0 码 9 / 1 码 6)
    owner = np.array([0, 0, 0, 0, 1, 1, 1])
    neigh = np.array([-1, -1, -1, 1, -1, -1, -1])
    is_bnd = neigh < 0
    o_code = np.array([6, 7, 8, 9, 7, 8, 9])
    n_code = np.array([0, 0, 0, 6, 0, 0, 0])
    ok = dict(owner=owner, neighbor=neigh, is_boundary=is_bnd, owner_primary=np.ones(7, bool),
              neighbor_primary=np.ones(7, bool), owner_code=o_code, neighbor_code=n_code, n_cells=2, n_prism=0)
    _check_one_primary_side_per_cell_face(**ok)
    missing = dict(ok, neighbor_primary=np.zeros(7, bool))
    with pytest.raises(ValueError):
        _check_one_primary_side_per_cell_face(**missing)
    dup = dict(ok, owner=np.append(owner, 0), neighbor=np.append(neigh, -1), is_boundary=np.append(is_bnd, True),
               owner_primary=np.ones(8, bool), neighbor_primary=np.ones(8, bool),
               owner_code=np.append(o_code, 6), neighbor_code=np.append(n_code, 0))
    with pytest.raises(ValueError):
        _check_one_primary_side_per_cell_face(**dup)


def _closed_box(kind, order):
    """封闭对称盒子上的光滑非均匀状态；返回 `(mesh, ops, U, provider, 积分函数)`。"""
    build = build_channel_mesh_prism if kind == "prism" else build_channel_mesh
    mesh = build(order, 3, 3, 2, LX, H, LZ)
    ops = generate_fr_operators(order)
    nc, ns = mesh.n_cells, mesh.n_sps_per_cell
    n_real = real_sps_per_cell(order)[0 if kind == "prism" else 1]
    w = (build_native_prism_sp_weights if kind == "prism" else build_native_tet_sp_weights)(order)
    dj = np.asarray(mesh.jacobians["det_jacs"]).reshape(nc, ns)[:, :n_real]
    X = np.asarray(mesh.sps_coords)
    Q = np.zeros(X.shape[:2] + (5,))
    Q[..., 0] = 1.2 * (1 + 0.05 * np.sin(20 * X[..., 0]) * np.cos(30 * X[..., 1]))
    Q[..., 1] = 30 * np.sin(np.pi * X[..., 1] / H) + 5 * np.cos(15 * X[..., 2])
    Q[..., 2] = 4 * np.sin(np.pi * X[..., 0] / LX) * np.sin(np.pi * X[..., 1] / H)
    Q[..., 3] = 2 * np.sin(np.pi * X[..., 2] / LZ)
    Q[..., 4] = 101325 * (1 + 0.01 * np.cos(10 * X[..., 0]))
    bc = {n: {"type": "SYMMETRY"} for n in ("z_min", "z_max", "wall_bottom", "wall_top", "x_min", "x_max")}
    prov = build_face_exact_ghost_provider(mesh, LX, H, LZ, bc)

    def relative_integral(R):
        weighted = R[:, :n_real] * dj[..., None] * w[None, :, None]
        return weighted.sum((0, 1)) / np.maximum(np.abs(weighted).sum((0, 1)), 1e-300)

    return mesh, ops, primitive_to_conserved(Q), prov, relative_integral


@pytest.mark.parametrize("kind,variables", [("prism", (0, 4)), ("tet", (0,))])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_closed_box_inviscid_residual_is_conservative(kind, variables, order):
    mesh, ops, U, prov, rel = _closed_box(kind, order)
    r = rel(compute_inviscid_residual_fr(U, mesh, ops, boundary_ghost_provider=prov))
    for v in variables:
        assert abs(r[v]) < 1e-11, f"{kind} P{order} 分量 {v}: 相对 {r[v]:.2e}"


@pytest.mark.parametrize("kind", ["prism", "tet"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_closed_box_viscous_energy_is_conservative(kind, order):
    mesh, ops, U, prov, rel = _closed_box(kind, order)
    r = rel(compute_viscous_residual_fr(U, mesh, ops, 1.8e-5, 0.72, boundary_ghost_provider=prov))
    assert abs(r[4]) < 1e-11, f"{kind} P{order} 粘性能量: 相对 {r[4]:.2e}"
