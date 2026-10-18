"""P0 混合拆分面（棱柱四边形侧面一半挨内部单元、一半在边界上，`fr/face_flux_points/grouping.py`）。

修复前（I-7）：P0 无粘核在内部子面记录上按面积占比混合两份通量——边界半区的黎曼问题用的是内部记录上从未计算过
的全零"幽灵态"（均匀来流残差 5.6e7），邻居一侧还多加了 (1-b) 倍的边界半区通量（不守恒）；P0 粘性核按唯一的
通量点落在哪个半区把整张面二选一，另一侧单元照样施加内部贡献。现在：无粘把两半当作两个独立的面（各用真实
三角形几何），粘性在棱柱一侧按面积占比加权两半的跳量。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved
from autoflowcfd.core.fr_residual.inviscid_p0 import compute_inviscid_residual_fv_p0
from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh
from tests.unit.test_fr_residual_inviscid import _MockCells, _MockNodes

FREESTREAM = {"rho_inf": 1.2, "vel_inf": 30.0, "p_inf": 1.0e5}


def _mixed_mesh(order, tri=(0, 1, 4)):
    """一个棱柱 + 一个贴在棱柱 y=0 四边形侧面一个三角形上的四面体：该侧面另一半在边界上。"""
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0.5, -1, 0.5]], float)
    mv = SimpleNamespace(cell_count=2, nodes=_MockNodes(nodes), cells=_MockCells(np.array([list(tri) + [6]], np.int32)),
                         prism_cells=_MockCells(np.array([[0, 1, 2, 3, 4, 5]], np.int32)))
    m = HighOrderMesh(order=order)
    m.load_from_volume_mesh(mv)
    fc = m.face_connectivity
    m.boundary_groups = {"farfield": np.unique(fc.owner_cell[fc.is_boundary]).astype(np.int64)}
    m.boundary_bc_types = {"farfield": "FARFIELD"}
    return m


def _provider(m):
    return build_boundary_ghost_provider(SimpleNamespace(mesh=m, freestream=dict(FREESTREAM), turb_model_name="NONE",
                                                         wmles_model=None), bc_overrides={})


@pytest.mark.parametrize("tri", [(0, 1, 4), (0, 4, 3)])
def test_mesh_really_has_a_mixed_face(tri):
    flat = get_flat_face_geometry(_mixed_mesh(0, tri), _mixed_mesh(0, tri).operators)
    assert int((flat.mixed_nb_partner >= 0).sum()) == 1


@pytest.mark.parametrize("tri", [(0, 1, 4), (0, 4, 3)])
def test_p0_inviscid_preserves_freestream_on_mixed_faces(tri):
    m = _mixed_mesh(0, tri)
    U = np.tile(primitive_to_conserved(np.array([1.2, 30.0, 0.0, 0.0, 1.0e5])), (m.n_cells, 1, 1))
    R = compute_inviscid_residual_fv_p0(U, m, _provider(m), mach_ref=0.1)
    flux_scale = 1.0e5 * 30.0                   # p u 量级
    assert np.abs(R).max() < 1e-12 * flux_scale


def test_p0_inviscid_interior_exchange_is_conservative():
    """非均匀状态：两单元的 V R 之和只剩各自边界面的通量（内部半区的通量两侧等值反号）。"""
    from autoflowcfd.core.fr_operators.kernels import compute_ausm_up_flux, resolve_ausm_precond_mode
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    from autoflowcfd.core.fr_residual.inviscid_p0 import _extract_p0_face_geometry, _precompute_ghost_states

    m = _mixed_mesh(0)
    prov = _provider(m)
    Qp = np.array([[1.25, 28.0, 3.0, -2.0, 1.02e5], [1.15, 33.0, -4.0, 1.0, 0.98e5]])
    U = np.stack([primitive_to_conserved(q) for q in Qp])[:, None, :]
    R = compute_inviscid_residual_fv_p0(U, m, prov, mach_ref=0.1)
    fc = m.face_connectivity
    normals, areas = _extract_p0_face_geometry(m.face_flux_points, fc, fc.n_faces)
    Q_all = conserved_to_primitive(U[:, 0, :5])
    Qg = _precompute_ghost_states(m.face_flux_points, fc, prov, Q_all, fc.n_faces, normals)
    mode = int(resolve_ausm_precond_mode())
    boundary_total = np.zeros(5)
    for f in np.flatnonzero(fc.is_boundary):
        if areas[f] > 0.0:
            boundary_total -= compute_ausm_up_flux(Q_all[fc.owner_cell[f]], Qg[f], normals[f], 0.1, mode) * areas[f]
    total = (R[:, 0, :] * m.cell_volumes[:, None]).sum(axis=0)
    np.testing.assert_allclose(total, boundary_total, rtol=1e-12, atol=1e-9 * np.abs(boundary_total).max())


@pytest.mark.parametrize("tri", [(0, 1, 4), (0, 4, 3)])
def test_p0_viscous_weights_the_two_halves_by_area(tri):
    """P0 粘性：棱柱一侧在混合面上的修正 = (1-b) x 只用内部邻居 + b x 只用边界幽灵态（提升算子线性）。

    两种拆法：唯一的通量点分别落在内部半区与边界半区。后者此前内部半区的对侧取了源矩阵插值，而那个通量点上
    没有内部来源（权重为 0），对侧成了全零态，残差为 nan（I-181 复核时查出）。"""
    from autoflowcfd.core.fr_residual.viscous_flux import compute_viscous_residual_fr

    m = _mixed_mesh(0, tri)
    flat = get_flat_face_geometry(m, m.operators)
    f_int = int(np.flatnonzero(flat.mixed_nb_partner >= 0)[0])
    b = float(flat.p0_second_half_frac[f_int])
    assert 0.0 < b < 1.0
    prism = int(flat.owner_cell[f_int])
    Qp = np.array([[1.25, 28.0, 3.0, -2.0, 1.02e5], [1.15, 33.0, -4.0, 1.0, 0.98e5]])
    U = np.stack([primitive_to_conserved(q) for q in Qp])[:, None, :]
    prov = _provider(m)

    def prism_row(frac):
        saved = flat.p0_second_half_frac.copy()
        flat.p0_second_half_frac[f_int] = frac
        try:
            return compute_viscous_residual_fr(U, m, m.operators, 1.8e-5, 0.72, boundary_ghost_provider=prov)[prism, 0]
        finally:
            flat.p0_second_half_frac[:] = saved

    r_int, r_bnd, r_b = prism_row(0.0), prism_row(1.0), prism_row(b)
    assert np.all(np.isfinite(r_int)) and np.all(np.isfinite(r_bnd))
    np.testing.assert_allclose(r_b, (1.0 - b) * r_int + b * r_bnd, rtol=1e-10, atol=1e-12 * np.abs(r_int).max())
    assert np.abs(r_int - r_bnd).max() > 1e-6 * np.abs(r_int).max()       # 两半确实不同，加权有判别力


def _two_tet_mesh(order):
    """一个棱柱 + 两个四面体分别贴在其 y=0 四边形侧面的两个三角形上（真多源面：两半各挨一个单元）。"""
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0.5, -1, 0.5]], float)
    mv = SimpleNamespace(cell_count=3, nodes=_MockNodes(nodes),
                         cells=_MockCells(np.array([[0, 1, 4, 6], [0, 4, 3, 6]], np.int32)),
                         prism_cells=_MockCells(np.array([[0, 1, 2, 3, 4, 5]], np.int32)))
    m = HighOrderMesh(order=order)
    m.load_from_volume_mesh(mv)
    fc = m.face_connectivity
    m.boundary_groups = {"farfield": np.unique(fc.owner_cell[fc.is_boundary]).astype(np.int64)}
    m.boundary_bc_types = {"farfield": "FARFIELD"}
    return m


def _p0_state(n_cells, overrides):
    Qp = np.tile([1.2, 30.0, 0.0, 0.0, 1.0e5], (n_cells, 1))
    for c, q in overrides.items():
        Qp[c] = q
    return np.stack([primitive_to_conserved(q) for q in Qp])[:, None, :]


def test_p0_viscous_prism_sees_both_tets_on_a_two_source_face():
    """真多源面：棱柱一侧的 P0 粘性修正同时依赖两个四面体，且等于两半各自跳量按面积占比的加权（I-182）。

    修复前棱柱一侧只取源矩阵在唯一通量点上的那个四面体：扰动另一个四面体，棱柱残差恰好不变（0.0）。"""
    from autoflowcfd.core.fr_residual.viscous_flux import compute_viscous_residual_fr

    m = _two_tet_mesh(0)
    flat = get_flat_face_geometry(m, m.operators)
    f = int(np.flatnonzero(flat.neighbor_src1_idx >= 0)[0])
    b = float(flat.p0_second_half_frac[f])
    assert flat.owner_cell[f] == 0 and 0.0 < b < 1.0
    prov = _provider(m)

    def prism_row(U, frac=None):
        saved = flat.p0_second_half_frac.copy()
        if frac is not None:
            flat.p0_second_half_frac[f] = frac
        try:
            return compute_viscous_residual_fr(U, m, m.operators, 1.8e-5, 0.72, boundary_ghost_provider=prov)[0, 0]
        finally:
            flat.p0_second_half_frac[:] = saved
    for c in (1, 2):
        lo = prism_row(_p0_state(3, {c: [1.2, 30.0, 4.0, 0.0, 1.0e5]}))
        hi = prism_row(_p0_state(3, {c: [1.2, 30.0, -4.0, 0.0, 1.05e5]}))
        assert np.abs(lo - hi).max() > 1e-6 * np.abs(lo).max(), f"棱柱 P0 粘性残差不依赖四面体 {c}"
    U = _p0_state(3, {1: [1.25, 28.0, 3.0, -2.0, 1.02e5], 2: [1.15, 33.0, -4.0, 1.0, 0.98e5]})
    r0, r1, rb = prism_row(U, 0.0), prism_row(U, 1.0), prism_row(U)
    np.testing.assert_allclose(rb, (1.0 - b) * r0 + b * r1, rtol=1e-10, atol=1e-12 * np.abs(r0).max())
    assert np.abs(r0 - r1).max() > 1e-6 * np.abs(r0).max()
