"""壁面力积分（`postprocess/wall_forces.py`、`postprocess/fr_coefficients.py`）。

判据：
1. 压力扣参考压力：静止流体、压力处处等于参考压力时，任何（含不封闭的）壁面组合力为零；压力高出 Δp 时，
   合力 = Δp x 该组面的向量面积（此前用绝对压力，不封闭的物体面多出 p_ref x 开口投影面积）。
2. 摩擦用求解器的粘性公共通量：Couette 流 u = U y/H 在 P1/P2 精确，下壁受力 = mu U/H x 面积（+x）；
   上壁反向。P0 时单元内梯度为零，摩擦完全来自罚项，必须非零且与正确值同号（此前恒为零）。
3. 选组：只积分指定组；指定非 WALL 组报错。
4. 四面体拥有的壁面（原生面编码 >= 6）能正常积分。
"""

import numpy as np
import pytest

from autoflowcfd.postprocess.fr_coefficients import compute_aerodynamic_coefficients_fr, pressure_force_coefficients
from autoflowcfd.postprocess.wall_forces import sample_wall, wall_group_codes
from autoflowcfd.core.time_integration import TimeIntegrationScheme

MU, RHO, U, P = 1.8e-5, 1.225, 30.0, 101325.0
LX, H, LZ = 0.4, 0.1, 0.08


def _couette(order):
    from tests.unit.test_corrected_velocity_gradient import _couette_solver

    solver, Q = _couette_solver(order)
    solver.state.Q = Q
    return solver


def _codes(solver, *names):
    return list(wall_group_codes(solver.boundary_ghost_provider, list(names)))


@pytest.mark.parametrize("order", [1, 2])
def test_couette_wall_shear_is_exact(order):
    solver = _couette(order)
    tau_area = MU * U / H * (LX * LZ)
    bottom = sample_wall(solver, _codes(solver, "wall_bottom")).force_viscous.sum(axis=0)
    top = sample_wall(solver, _codes(solver, "wall_top")).force_viscous.sum(axis=0)
    np.testing.assert_allclose(bottom, [tau_area, 0.0, 0.0], rtol=1e-9, atol=1e-12 * tau_area)
    np.testing.assert_allclose(top, [-tau_area, 0.0, 0.0], rtol=1e-9, atol=1e-12 * tau_area)


def test_p0_wall_friction_comes_from_the_penalty():
    solver = _couette(0)
    tau_area = MU * U / H * (LX * LZ)
    bottom = sample_wall(solver, _codes(solver, "wall_bottom")).force_viscous.sum(axis=0)
    assert bottom[0] > 0.1 * tau_area


@pytest.mark.parametrize("order", [0, 1, 2])
def test_reference_pressure_is_subtracted_on_open_surfaces(order):
    solver = _couette(order)
    Q = np.asarray(solver.state.Q)
    Q[..., 1:4] = 0.0
    Q[..., 4] = P
    solver.state.Q = Q
    f = sample_wall(solver, _codes(solver, "wall_bottom"), viscous=False).force_inviscid.sum(axis=0)
    np.testing.assert_allclose(f, 0.0, atol=1e-9 * P * LX * LZ)

    dp = 250.0
    Q[..., 4] = P + dp
    solver.state.Q = Q
    f = sample_wall(solver, _codes(solver, "wall_bottom"), viscous=False).force_inviscid.sum(axis=0)
    # 下壁外法向（流体 -> 物体）为 -y：作用在下壁上的压力合力 = dp * A * (-y)
    np.testing.assert_allclose(f, [0.0, -dp * LX * LZ, 0.0], rtol=1e-9, atol=1e-9 * dp * LX * LZ)


def test_group_selection():
    solver = _couette(1)
    with pytest.raises(ValueError, match="WALL"):
        wall_group_codes(solver.boundary_ghost_provider, ["x_min"])
    both = compute_aerodynamic_coefficients_fr(solver, reference_area=LX * LZ)
    bottom = compute_aerodynamic_coefficients_fr(solver, reference_area=LX * LZ, groups=["wall_bottom"])
    q = 0.5 * RHO * (0.5 * U) ** 2 * LX * LZ          # 来流速度取 0.5 U（_couette_solver 的设置）
    assert bottom.Cd == pytest.approx(MU * U / H * LX * LZ / q, rel=1e-6)
    assert abs(both.Cd) < 1e-6 * abs(bottom.Cd)          # 上下壁摩擦相互抵消


def test_pressure_monitor_matches_inviscid_part():
    solver = _couette(1)
    mon = pressure_force_coefficients(solver, reference_area=LX * LZ, groups=["wall_bottom"])
    full = compute_aerodynamic_coefficients_fr(solver, reference_area=LX * LZ, groups=["wall_bottom"],
                                               include_viscous=False)
    assert mon["Cl"] == pytest.approx(full.Cl, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("order", [1, 2])
def test_tet_owned_wall_faces_integrate(order):
    """四面体拥有的壁面（原生面编码 >= 6）：静止流体、全部边界为壁面的封闭体，压力 p_ref + dp 下合力为零。"""
    from autoflowcfd.core.fr_solver import FRSolver
    from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

    mesh = _build_synthetic_mixed_mesh(order)
    fc = mesh.face_connectivity
    assert np.any(fc.is_boundary & (fc.owner_cell >= mesh.n_prism_cells))
    mesh.boundary_bc_types = {"farfield": "WALL"}
    solver = FRSolver(mesh=mesh, order=order, rho_inf=RHO, vel_inf=U, p_inf=P, mu_molecular=MU, turb_model_name="SST",
                      time_scheme=TimeIntegrationScheme.SSP_RK3)
    Q = np.asarray(solver.state.Q).copy()
    Q[..., 1:4] = 0.0
    Q[..., 4] = P + 500.0
    solver.state.Q = Q
    samples = sample_wall(solver, list(wall_group_codes(solver.boundary_ghost_provider)), viscous=False)
    area = np.abs(samples.force_inviscid).sum()
    assert area > 0.0
    np.testing.assert_allclose(samples.force_inviscid.sum(axis=0), 0.0, atol=1e-10 * area)


def test_compute_inlet_fp_positions_on_native_tet_face():
    """SEM 合成湍流入口的通量点坐标在四面体拥有的边界面（原生面编码 >= 6）上能正常外插。"""
    from types import SimpleNamespace

    from autoflowcfd.core.fr_solver.boundary import _compute_inlet_fp_positions
    from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

    mesh = _build_synthetic_mixed_mesh(2)
    fc = mesh.face_connectivity
    tet_boundary_faces = np.flatnonzero(fc.is_boundary & (fc.owner_cell >= mesh.n_prism_cells))
    assert tet_boundary_faces.size > 0
    positions = _compute_inlet_fp_positions(SimpleNamespace(mesh=mesh, ops=mesh.operators), fc.is_boundary.copy())
    assert set(positions) == set(np.flatnonzero(fc.is_boundary).tolist())
    assert all(np.all(np.isfinite(positions[f])) for f in tet_boundary_faces)
