"""湍流标量对流的来流条件覆盖混合拆分面的边界半区（I-66）。

混合拆分面（棱柱四边形侧面一半挨内部单元、一半在边界上，`fr/face_flux_points/grouping.py`）的边界半区按配对
边界面取边界条件。此前来流覆盖只作用于 owner 坐标系的真边界面：配对面是开放边界时，边界半区上流入的通量点取
零梯度外插，平均流取的却是配对面的来流幽灵态。现在两侧坐标系的边界点共用 `transport/faces.py::inflow_points`
（CPU 残差、GPU 残差与湍流解析 Jacobian 的 ghost 规则同一份）。
"""

from types import SimpleNamespace

import numpy as np
import pytest

import autoflowcfd.core.gpu.residual.gpu_gradients as gpu_gradients_mod
import autoflowcfd.core.gpu.residual.gpu_volume_contract as gpu_volume_contract_mod
import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.turbulence.jacobian.scalar_blocks import convection_ghost_affine
from autoflowcfd.core.turbulence.transport import compute_scalar_convection_residual, open_boundary_code_mask
from autoflowcfd.core.turbulence.transport.faces import inflow_points, precompute_scalar_convection_geometry
from tests.unit._gpu_cupy_shim import patch_module_get_cupy
from tests.unit._numpy_as_cupy import NumpyAsCupy
from tests.unit.test_p0_mixed_split_faces import _mixed_mesh, _provider

U = 30.0


def _case():
    """P1 混合网格（全部边界为 FARFIELD）、均匀密度、`+y` 方向均匀速度：棱柱只有 y=0 四边形侧面是流入面（x=0
    侧面与两个三角形面上质量通量恰为零，斜面流出），而该侧面一半挨四面体、一半是边界半区。"""
    m = _mixed_mesh(1)
    ops = m.operators
    flat = get_flat_face_geometry(m, ops)
    rho = np.ones((m.n_cells, m.n_sps_per_cell))
    vel = np.zeros((m.n_cells, m.n_sps_per_cell, 3))
    vel[..., 1] = U
    open_code = open_boundary_code_mask(_provider(m), flat.n_faces)
    return m, ops, flat, rho, vel, open_code


def _prism_response(residual_fn):
    """均匀标量 `phi = 1`：来流值取 1 与 2 时残差之差。只有流入的边界点感受到来流值。"""
    return residual_fn(2.0) - residual_fn(1.0)


def test_mixed_boundary_half_takes_freestream_value_on_inflow():
    m, ops, flat, rho, vel, open_code = _case()
    assert int((flat.mixed_nb_partner >= 0).sum()) == 1
    phi = np.ones((m.n_cells, m.n_sps_per_cell))

    def residual(phi_inf):
        return compute_scalar_convection_residual(phi, rho, vel, m, ops, open_boundary_face=open_code,
                                                  freestream_value=phi_inf)

    d = _prism_response(residual)
    # 棱柱（单元 0）唯一的流入边界点在混合面的边界半区：修复前这里逐位为零（零梯度外插与来流值无关）
    assert np.abs(d[0]).max() > 1e-2 * U


def test_inflow_points_neighbor_frame_uses_partner_open_flag():
    """neighbor 坐标系：混合面边界半区的点按配对边界面的开放标志、按 neighbor 侧质量通量判流入。"""
    n_fp = 4
    flat = SimpleNamespace(
        neighbor_src0_cell=np.array([3, 3, -1]), neighbor_src1_idx=np.array([-1, -1, -1]),
        mixed_nb_partner=np.full(3, -1), mixed_nb_mask=np.zeros((3, n_fp), bool),
        mixed_ow_partner=np.array([2, -1, -1]),
        mixed_ow_mask=np.array([[True, True, False, False]] + [[False] * n_fp] * 2),
    )
    m_side = np.array([[-1.0, 1.0, -1.0, -1.0]] * 3)
    open_code = np.array([False, True, True])   # 面 1 是内部面（缺省配置可为开放），不得被当成边界点
    got = inflow_points(np, flat, "neighbor", open_code, m_side)
    np.testing.assert_array_equal(got, [[True, False, False, False], [False] * n_fp, [False] * n_fp])
    # 配对面不是开放边界：不取来流
    assert not inflow_points(np, flat, "neighbor", np.array([False, True, False]), m_side).any()


def test_jacobian_ghost_rule_matches_residual_on_mixed_inflow():
    """湍流解析 Jacobian 的 ghost 仿射系数：流入的边界点（含混合面边界半区）上外部态是常数，系数为 0。"""
    m, ops, flat, rho, vel, open_code = _case()
    geom = precompute_scalar_convection_geometry(rho, vel, m, ops, flat)
    no_wall = np.zeros(flat.n_faces, dtype=bool)
    ghost, a = convection_ghost_affine(flat, "owner", geom.mass_flux, (no_wall,), open_code)
    f = int(np.flatnonzero(flat.mixed_nb_partner >= 0)[0])
    half = flat.mixed_nb_mask[f]
    inflow = half & (geom.mass_flux[f] < 0)
    assert inflow.any() and ghost[f, half].all()
    np.testing.assert_array_equal(a[0, f, inflow], 0.0)


def test_gpu_residual_matches_cpu_on_mixed_inflow(monkeypatch):
    patch_module_get_cupy(monkeypatch, [gst, gpu_gradients_mod, gpu_volume_contract_mod], NumpyAsCupy())
    from tests.unit.test_gpu_scalar_transport import _prepare_mesh_ops_data

    m, ops, flat, rho, vel, open_code = _case()
    mesh_data, ops_data = _prepare_mesh_ops_data(m, ops)
    n_cells, n_prism, n_sps = m.n_cells, m.n_prism_cells, m.n_sps_per_cell
    phi = np.random.default_rng(5).uniform(0.5, 1.5, size=(n_cells, n_sps))
    mass_div = gst.scalar_convection_volume_divergence_gpu(np, np.ones((n_cells, n_sps)), rho, vel, mesh_data,
                                                           ops_data, n_cells, n_prism, n_sps)
    expected = compute_scalar_convection_residual(phi, rho, vel, m, ops, open_boundary_face=open_code,
                                                  freestream_value=2.0)
    actual = gst.compute_scalar_convection_residual_gpu(phi, rho, vel, mesh_data, ops_data, flat, n_cells, n_prism,
                                                        n_sps, mass_div, open_boundary_face=open_code,
                                                        freestream_value=2.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.parametrize("frame", ["owner", "neighbor"])
def test_boundary_row_subset_matches_whole_face_rule(frame):
    """分类只在含边界点的面上求值（`frame_boundary_rows`）：与整面求值同一规则逐位一致，掩码在内部面也为真时
    其余面仍全为假（`target` 只在 `is_dirichlet` 处有意义）。"""
    from autoflowcfd.core.turbulence.transport.faces import (
        boundary_diffusion_targets, frame_boundary_points, frame_boundary_rows,
    )

    m, ops, flat, rho, vel, open_code = _case()
    rng = np.random.default_rng(3)
    nf, nfp = flat.owner_adj_row_exact.shape[:2]
    wz = rng.random(nf) < 0.4
    hv = (rng.random(nf) < 0.4) & ~wz
    wv = rng.random((nf, nfp))
    m_side = rng.standard_normal((nf, nfp))
    rows = frame_boundary_rows(np, flat, frame)
    assert frame_boundary_rows(np, flat, frame) is rows          # 按面几何缓存
    assert (rows.size > 0) == (frame == "owner")                # 本网格的混合面只在 owner 坐标系有配对边界面

    true_bnd, mp, in_mixed = frame_boundary_points(np, flat, frame)
    is_bnd = true_bnd[:, None] | in_mixed
    zero = np.where(in_mixed, wz[mp][:, None], wz[:, None])
    value = np.where(in_mixed, hv[mp][:, None], hv[:, None]) & ~zero
    want_dir = is_bnd & (zero | value)
    want_target = np.where(value, np.where(in_mixed, wv[mp], wv), 0.0)
    want_inflow = np.where(in_mixed, open_code[mp][:, None], (true_bnd & open_code)[:, None]) & (m_side < 0)

    b, d, t = boundary_diffusion_targets(np, flat, frame, wz, wv, hv)
    np.testing.assert_array_equal(b, is_bnd)
    np.testing.assert_array_equal(d, want_dir)
    np.testing.assert_array_equal(np.where(d, t, 0.0), np.where(want_dir, want_target, 0.0))
    np.testing.assert_array_equal(inflow_points(np, flat, frame, open_code, m_side), want_inflow)
