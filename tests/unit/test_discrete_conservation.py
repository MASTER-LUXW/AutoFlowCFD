"""无粘残差的离散守恒（体积 L2 投影 + 本侧通量迹并进体积算子（按槽位组合各一份）
+ 三角形面通量点按全局节点号规范朝向）。

1. 迹算子的约定：随机直边四面体/棱柱、常数通量下，每个面、每个坍缩顶点槽位的
   `Σ_{q,m} Tn[i,q,m] F~[q,m]` 等于生产几何的精确度量行投影 `a(fp_i)·F`（Nanson 公式；
   直边单元的 adj(J) 落在细层空间内，插值精确），且通量点顺序与面外插矩阵一致；每个
   槽位组合的体积算子 `K` 都逐单元守恒；
2. 构造期护栏：面记录必须满足"每个单元每个面恰一条 primary 侧"，重复或缺失都报错；
3. 规范朝向（`fr/triangle_apex.py`）：共享三角形面两侧各自外插出的通量点物理坐标是
   同一组点（修复前约 1/5 的面不重合）；
4. 封闭对称盒子（光滑非均匀场、全部 SYMMETRY）上全域积分 `Σ w_s det_s R_s`（对称面上
   质量与能量的法向通量为零，物理上守恒）：无粘与粘性、棱柱与四面体网格，质量与能量
   都到机器精度。修复前四面体无粘能量只到面求积量级（P1 1.1e-7、P2 1.5e-9、
   P3 4.6e-11）：两侧通量点不重合，各自对非线性 `F*` 求积对不上。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.face_kernels import (
    _check_one_primary_side_per_cell_face, get_flat_face_geometry,
)
from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr, primitive_to_conserved
from autoflowcfd.core.fr_residual.viscous_flux import compute_viscous_residual_fr
from autoflowcfd.fr.face_flux_points.exact_normal import _native_tet_adj_row_batched
from autoflowcfd.fr.face_flux_trace import build_prism_face_flux_trace, build_tet_face_flux_trace
from autoflowcfd.fr.native_padding import real_sps_per_cell
from autoflowcfd.fr.native_prism.basis import build_native_prism_operators, native_prism_exact_jacobian
from autoflowcfd.fr.native_prism.face import native_prism_face_adj_rows
from autoflowcfd.fr.native_prism.quadrature import build_native_prism_sp_weights
from autoflowcfd.fr.native_tet.basis import compute_native_tet_jacobian
from autoflowcfd.fr.native_tet.quadrature import build_native_tet_sp_weights
from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.fr.quadrature_points import gauss_legendre
from autoflowcfd.fr.triangle_apex import N_TRI_SLOTS
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
        a = _native_tet_adj_row_batched(v, nodes[None], order + 1, sps_1d)[0]   # 与槽位无关（见 triangle_apex）
        for slot in range(N_TRI_SLOTS):
            trace = np.einsum("iqm,mv->iv", T[v, slot], Ft)     # 细层 Lagrange 单位分解：Σ_q L_q = 1
            np.testing.assert_allclose(trace, a @ F0, rtol=1e-12, atol=1e-12 * np.abs(a @ F0).max())


@pytest.mark.parametrize("order", [1, 2, 3])
def test_prism_trace_equals_exact_adj_projection(order):
    rng = np.random.default_rng(10 + order)
    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]], float)
    nodes = base + 0.15 * rng.standard_normal((6, 3))      # 一般直边棱柱：侧面非平面、adj 随点变化
    oo = 2 * order
    jac = native_prism_exact_jacobian(build_native_prism_operators(oo)[0], nodes)
    adj = np.linalg.det(jac)[:, None, None] * np.linalg.inv(jac)
    Ft = np.einsum("qmd,dv->qmv", adj, F0)                 # 细点上的逆变通量（adj 在细层空间内）
    T = build_prism_face_flux_trace(order, oo)
    for fid in range(5):
        for slot in range(N_TRI_SLOTS):
            if fid >= 2 and slot > 0:
                assert np.isnan(T[fid, slot]).all()        # 侧四边形只有槽位 0
                continue
            trace = np.einsum("iqm,qmv->iv", T[fid, slot], Ft)
            a = native_prism_face_adj_rows(order, fid, nodes, slot)
            np.testing.assert_allclose(trace, a @ F0, rtol=1e-11, atol=1e-11 * np.abs(a @ F0).max())


@pytest.mark.parametrize("order", [1, 2, 3])
def test_every_k_combo_is_cell_conservative(order):
    """每个槽位组合的 `K` 都满足 `Σ_s w_s K[s] = 0`（单元内体积散度与本侧迹的面求积精确
    相消），否则相应槽位组合的单元会有守恒误差。"""
    ops = generate_fr_operators(order)
    for K, w in ((ops.overint_lifted_div_tet, build_native_tet_sp_weights(order)),
                 (ops.overint_lifted_div_prism, build_native_prism_sp_weights(order))):
        n = len(w)
        assert np.abs(K[:, n:]).max() == 0.0
        col = np.einsum("s,csqm->cqm", w, K[:, :n])
        assert np.abs(col).max() < 1e-12 * np.abs(K).max()
        assert np.abs(K - K[:1]).max() > 1e-6 * np.abs(K).max()      # 组合之间确实不同


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
    if kind == "prism":
        mesh = build_channel_mesh_prism(order, 3, 3, 2, LX, H, LZ)
    else:   # "tet_rotated"：两侧通量点顺序互为镜像的面也出现（见 build_channel_mesh 文档）
        mesh = build_channel_mesh(order, 3, 3, 2, LX, H, LZ, rotate_odd_tets=(kind == "tet_rotated"))
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


@pytest.mark.parametrize("kind", ["prism", "tet", "tet_rotated"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_closed_box_inviscid_residual_is_conservative(kind, order):
    mesh, ops, U, prov, rel = _closed_box(kind, order)
    r = rel(compute_inviscid_residual_fr(U, mesh, ops, boundary_ghost_provider=prov))
    for v in (0, 4):
        assert abs(r[v]) < 1e-11, f"{kind} P{order} 分量 {v}: 相对 {r[v]:.2e}"


@pytest.mark.parametrize("kind", ["prism", "tet", "tet_rotated"])
@pytest.mark.parametrize("order", [1, 2])
def test_shared_triangle_faces_use_coincident_flux_points(kind, order):
    """两侧各自"外插解点坐标"得到的通量点物理坐标是同一组点（外插对坐标线性、直边
    单元精确）。同时确认网格上确实出现了非零槽位；四面体网格上槽位全取 0（修复前的
    约定）时有一批面不重合（33/174），测试不是平凡通过。结构化挤出的棱柱网格两侧
    局部编号本来一致，槽位 0 也重合。"""
    mesh, ops, _, _, _ = _closed_box(kind, order)
    flat = get_flat_face_geometry(mesh, ops)
    X = np.asarray(mesh.sps_coords)
    interior = np.nonzero(~flat.is_boundary & flat.owner_is_primary & flat.neighbor_is_primary)[0]
    tri = interior[np.asarray(flat.owner_cube_face[interior] < 12)]
    assert tri.size > 0
    ffp = mesh.face_flux_points
    assert np.any(ffp.owner_tri_slot[tri] != 0) or np.any(ffp.neighbor_tri_slot[tri] != 0)

    def points(cell, op):
        return np.einsum("ps,sd->pd", flat.boundary_extrap_native[op], X[cell])

    scale = np.ptp(X.reshape(-1, 3), axis=0).max()
    n_bad_slot0 = 0
    for f in tri:
        po = points(flat.owner_cell[f], flat.owner_face_op[f])
        pn = points(flat.neighbor_cell[f], flat.neighbor_face_op[f])
        d = np.linalg.norm(po[:, None, :] - pn[None, :, :], axis=-1).min(axis=1)
        assert d.max() < 1e-12 * scale, f"面 {f} 两侧通量点不重合：{d.max():.3e}"
        po0 = points(flat.owner_cell[f], flat.owner_cube_face[f] - 6)
        pn0 = points(flat.neighbor_cell[f], flat.neighbor_cube_face[f] - 6)
        n_bad_slot0 += np.linalg.norm(po0[:, None, :] - pn0[None, :, :], axis=-1).min(axis=1).max() > 1e-9 * scale
    assert (n_bad_slot0 > 0) == (kind != "prism")


@pytest.mark.parametrize("kind", ["prism", "tet"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_closed_box_viscous_energy_is_conservative(kind, order):
    mesh, ops, U, prov, rel = _closed_box(kind, order)
    r = rel(compute_viscous_residual_fr(U, mesh, ops, 1.8e-5, 0.72, boundary_ghost_provider=prov))
    assert abs(r[4]) < 1e-11, f"{kind} P{order} 粘性能量: 相对 {r[4]:.2e}"
