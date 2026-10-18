"""WMLES 壁面模型（`core/turbulence/wall_model/`）。

修复前：壁面模型的输入（速度与壁距）外插到壁面通量点本身——壁距在壁面上约为 0、被 1e-8 下限托住，
y+ 极小，Spalding 律退化为 u+ = y+，tau_w = mu U / 1e-8，等于刚性罚项的伪无滑移；且对 SLIP_WALL（隧道壁、
远场滑移面）同样施加壁面剪应力。现在：匹配点取在 owner 单元形心高度（Kawai & Larsson 2012），只作用于物理
无滑移壁面，不需要壁距场。
"""

import numpy as np
import pytest

from autoflowcfd.core.turbulence.wall_model.law import KAPPA, B_LOG, solve_u_plus, spalding_y_plus, wall_shear_stress

LX, H, LZ = 0.4, 0.1, 0.08
NY = 3
U0, RHO, MU = 30.0, 1.2, 1.8e-5


# ---------------------------------------------------------------- 壁面律

def test_spalding_solve_accuracy():
    re = np.concatenate([[0.0], np.logspace(-8, 13, 300)])
    u, n_bad = solve_u_plus(np, re)
    assert n_bad == 0
    y, _ = spalding_y_plus(np, u)
    np.testing.assert_allclose((u * y)[1:], re[1:], rtol=1e-12)
    assert u[0] == 0.0


def test_spalding_limits():
    """粘性底层 u+ = y+；对数层趋于 u+ = ln(y+)/kappa + B。Spalding 式与对数律之差是 u+ 与指数展开前四项，
    相对量约 (kappa u+)^3 / 6 / e^(kappa u+)：y+ = 1e4 处 1e-3、1e6 处 1e-4。"""
    u = np.array([1e-3, 1e-2])
    y, _ = spalding_y_plus(np, u)
    np.testing.assert_allclose(y, u, rtol=1e-6)
    for y_plus, rel in ((1e4, 2e-3), (1e6, 2e-4)):
        u_log = np.log(y_plus) / KAPPA + B_LOG
        y_spalding, _ = spalding_y_plus(np, np.array([u_log]))
        assert y_spalding[0] == pytest.approx(y_plus, rel=rel)


def test_traction_opposes_tangential_velocity_and_recovers_u_tau():
    u_t = np.array([[3.0, 4.0, 0.0]])
    tau, y_plus = wall_shear_stress(np, u_t, np.array([2e-3]), np.array([1.2]), np.array([1.5e-5]))
    assert np.dot(tau[0], u_t[0]) < 0.0
    assert np.allclose(np.cross(tau[0], u_t[0]), 0.0)
    u_plus = 5.0 / np.sqrt(np.linalg.norm(tau[0]) / 1.2)
    y_check, _ = spalding_y_plus(np, np.array([u_plus]))
    assert y_check[0] == pytest.approx(y_plus[0], rel=1e-10)


def test_buffer_layer_point_recovers_true_u_tau():
    """由已知 u_tau 在缓冲层（y+ = 12）构造匹配点速度，壁面律反解回同一个 u_tau。"""
    nu, u_tau, y_plus = 1.5e-5, 0.8, 12.0
    y = y_plus * nu / u_tau
    from scipy.optimize import brentq

    u_plus = brentq(lambda up: spalding_y_plus(np, np.array([up]))[0][0] - y_plus, 1e-6, 50.0, xtol=1e-14)
    U = u_plus * u_tau
    tau, _ = wall_shear_stress(np, np.array([[U, 0.0, 0.0]]), np.array([y]), np.array([1.0]), np.array([nu]))
    assert np.sqrt(-tau[0, 0]) == pytest.approx(u_tau, rel=1e-10)


# ---------------------------------------------------------------- 匹配点与残差

def _channel(order, top_is_slip=True):
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from tests.validation._channel_mesh import build_channel_mesh_prism, build_face_exact_ghost_provider

    mesh = build_channel_mesh_prism(order, nx=3, ny=NY, nz=2, Lx=LX, H=H, Lz=LZ)
    bc = {n: {"type": "SYMMETRY"} for n in ("z_min", "z_max", "x_min", "x_max")}
    bc["wall_bottom"] = {"type": "WALL", "is_no_slip": False, "physical_no_slip": True}
    bc["wall_top"] = {"type": "WALL", "is_no_slip": False, "physical_no_slip": not top_is_slip}
    provider = build_face_exact_ghost_provider(mesh, LX, H, LZ, bc)
    return mesh, get_flat_face_geometry(mesh, mesh.operators), provider


def _linear_profile_U(mesh):
    """u = U0 y / H、rho 常数的守恒变量（能量列与本模型无关，取正值即可）。"""
    X = np.asarray(mesh.sps_coords)
    U = np.zeros(X.shape[:2] + (5,))
    U[..., 0] = RHO
    U[..., 1] = RHO * U0 * X[..., 1] / H
    U[..., 4] = 101325.0 / 0.4
    return U


@pytest.mark.parametrize("order", [0, 1, 2])
def test_matching_points_sit_at_owner_centroid_height(order):
    from autoflowcfd.core.turbulence.wall_model.model import EquilibriumWallModel

    mesh, flat, provider = _channel(order)
    m = EquilibriumWallModel().matching(np, mesh, flat, provider)
    assert m is not None and m.face.size > 0
    # 只有下壁（物理无滑移）；上壁是滑移壁，不施加壁面模型
    centers = np.asarray(mesh.face_connectivity.center)[m.face]
    assert np.allclose(centers[:, 1], 0.0)
    X = np.asarray(mesh.sps_coords)
    if order >= 1:
        x_m = np.einsum('ks,ksd->kd', m.E_m, X[m.owner])
        # 下壁法向 -y：匹配点 = 通量点 + h_wm * y_hat，y 坐标即 h_wm
        np.testing.assert_allclose(x_m[:, 1], m.h_wm, rtol=1e-10, atol=1e-14)
    # 均匀分层的棱柱：首层单元形心高度 = 首层厚度的一半
    np.testing.assert_allclose(m.h_wm, 0.5 * H / NY, rtol=1e-9)


@pytest.mark.parametrize("order", [1, 2])
def test_linear_profile_traction_uses_matching_velocity(order):
    """匹配点速度 = U0 h_wm / H，牵引等于壁面律在 (U(h_wm), h_wm) 处的值。"""
    from autoflowcfd.core.turbulence.wall_model.model import EquilibriumWallModel

    mesh, flat, provider = _channel(order)
    m, tau = EquilibriumWallModel().traction(np, mesh, flat, provider, _linear_profile_U(mesh), MU)
    u_m = U0 * m.h_wm / H
    ref, _ = wall_shear_stress(np, np.stack([u_m, 0 * u_m, 0 * u_m], axis=-1), m.h_wm, RHO + 0 * u_m, MU / RHO)
    np.testing.assert_allclose(tau, ref, rtol=1e-10)
    assert np.all(tau[:, 0] < 0.0)


def test_slip_walls_get_no_wall_model():
    from autoflowcfd.core.turbulence.wall_model.model import EquilibriumWallModel

    both = EquilibriumWallModel().matching(np, *_channel(1, top_is_slip=False))
    bottom_only = EquilibriumWallModel().matching(np, *_channel(1, top_is_slip=True))
    assert both.face.size == 2 * bottom_only.face.size


@pytest.mark.parametrize("order", [0, 1, 2])
def test_residual_correction_is_conservative_momentum_only(order):
    """修正只在动量分量；按解点积分权重对全域积分等于 sum A tau_w（壁面传给流体的动量，逐位守恒）。"""
    from autoflowcfd.core.turbulence.wall_model.model import EquilibriumWallModel

    mesh, flat, provider = _channel(order)
    U = _linear_profile_U(mesh)
    model = EquilibriumWallModel()
    corr = model.residual_correction(np, mesh, flat, provider, U, MU)
    m, tau = model.traction(np, mesh, flat, provider, U, MU)
    assert not corr[..., 0].any() and not corr[..., 4].any()
    from autoflowcfd.fr.native_padding import sp_integration_weights

    n_cells, n_sps = U.shape[:2]
    det = np.asarray(mesh.jacobians["det_jacs"]).reshape(n_cells, n_sps)
    W = sp_integration_weights(det, np.arange(n_cells) < mesh.n_prism_cells, order)
    total = np.einsum('cs,csd->d', W, corr[..., 1:4])
    expected = np.sum(m.area[:, None] * tau, axis=0)
    np.testing.assert_allclose(total, expected, rtol=1e-12, atol=1e-14 * np.abs(expected).max())


def test_native_tet_wall_face():
    """四面体 owner 的壁面通量点：匹配点反解走原生四面体节点基，修正有限、只落在该单元。"""
    import types

    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved
    from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
    from autoflowcfd.core.turbulence.wall_model.model import EquilibriumWallModel
    from autoflowcfd.fr.operators import generate_fr_operators
    from tests.unit._boundary_groups import tag_wall_cells
    from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

    order = 2
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    fc = mesh.face_connectivity
    tet_faces = np.nonzero(fc.is_boundary & (fc.owner_cell >= mesh.n_prism_cells))[0]
    wall_cell = int(fc.owner_cell[tet_faces[0]])
    tag_wall_cells(mesh, [wall_cell])
    stub = types.SimpleNamespace(mesh=mesh, freestream={"rho_inf": RHO, "vel_inf": U0, "p_inf": 101325.0},
                                 turb_model_name="WMLES", wall_model=EquilibriumWallModel())
    provider = build_boundary_ghost_provider(stub, bc_overrides={})
    U = np.tile(primitive_to_conserved(np.array([RHO, U0, 5.0, -3.0, 101325.0])),
                (mesh.n_cells, mesh.n_sps_per_cell, 1))
    model = EquilibriumWallModel()
    corr = model.residual_correction(np, mesh, get_flat_face_geometry(mesh, ops), provider, U, MU)
    assert corr is not None and np.all(np.isfinite(corr))
    others = np.arange(mesh.n_cells) != wall_cell
    assert not corr[others].any()
    assert np.any(corr[wall_cell, :, 1:4] != 0.0)
    m = model.matching(np, mesh, get_flat_face_geometry(mesh, ops), provider)
    assert np.all(m.h_wm > 0.0) and np.all(m.owner == wall_cell)


def _host(mesh, provider, U, wmles):
    import types

    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

    return types.SimpleNamespace(
        mesh=mesh, ops=mesh.operators, boundary_ghost_provider=provider, wall_model=wmles, mu_molecular=MU,
        freestream={"mach_ref": 0.1, "p_inf": 101325.0},
        state=types.SimpleNamespace(U=U, Q=conserved_to_primitive(U)),
        _get_turbulent_viscosity_field=lambda: None)


def test_wall_forces_use_wall_model_traction():
    """壁面力积分：物理无滑移壁面的摩擦来自壁面模型牵引（物体受力 -sum A tau_w），与残差同一个；滑移镜像的
    粘性公共通量在壁面上没有切向分量。"""
    from autoflowcfd.core.turbulence.wall_model.model import EquilibriumWallModel
    from autoflowcfd.postprocess.wall_forces import sample_wall, wall_group_codes

    mesh, flat, provider = _channel(1)
    U = _linear_profile_U(mesh)
    codes = list(wall_group_codes(provider))
    with_model = sample_wall(_host(mesh, provider, U, EquilibriumWallModel()), codes)
    without = sample_wall(_host(mesh, provider, U, None), codes)
    m, tau = EquilibriumWallModel().traction(np, mesh, flat, provider, U, MU)
    friction = with_model.force_viscous.sum(axis=0) - without.force_viscous.sum(axis=0)
    np.testing.assert_allclose(friction, -np.sum(m.area[:, None] * tau, axis=0), rtol=1e-12)
    assert abs(without.force_viscous[:, 0].sum()) < 1e-12 * abs(friction[0])
    assert friction[0] > 0.0                       # 流向 +x 的剪切把物体往 +x 拖


def test_default_force_groups_exclude_slip_walls():
    from autoflowcfd.postprocess.wall_forces import wall_group_codes

    _, _, provider = _channel(1, top_is_slip=True)
    assert sorted(wall_group_codes(provider).values()) == ["wall_bottom"]
    assert sorted(wall_group_codes(provider, ["wall_top"]).values()) == ["wall_top"]
