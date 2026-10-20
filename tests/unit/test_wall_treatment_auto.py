"""RANS 自动壁面处理（`wall_treatment="auto"`，`core/turbulence/wall_model/treatment.py`，I-137）的接入。

auto：平衡壁面模型给壁面剪应力、物理无滑移壁面取滑移镜像幽灵态；SST 的 omega 壁面值取 Menter 混合式、k 壁面
零通量；SA-neg 只换动量壁面条件（auto 为默认，依据见 `treatment.py`）。resolved 保持低雷诺数壁面处理。四个后端由同一组函数构造。
"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration import TimeIntegrationScheme
from autoflowcfd.core.turbulence.wall_model.treatment import build_wall_model, needs_wall_model
from tests.unit.test_gpu_solver_order_continuation import _patch_gpu_modules  # noqa: F401（自动夹具）
from tests.validation._channel_mesh import build_channel_mesh_prism, channel_wall_source
from autoflowcfd.core.freestream_defaults import MU_MOLECULAR, P_INF, RHO_INF

LX, H, LZ = 0.4, 0.1, 0.08
RHO_Y = 1.2
_BC = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL", "wall_top": "WALL",
       "z_min": "SYMMETRY", "z_max": "SYMMETRY"}


@pytest.mark.parametrize("model,treatment,expected", [
    ("SST", "auto", True), ("SA", "auto", True), ("DDES", "auto", True), ("IDDES", "auto", True),
    ("SST", "resolved", False), ("NONE", "auto", False), ("LES", "auto", False),
    ("WMLES", "resolved", True), ("WMLES", "auto", True),
])
def test_wall_model_rule(model, treatment, expected):
    assert needs_wall_model(model, treatment) is expected
    assert (build_wall_model(model, treatment) is not None) is expected


def test_unknown_treatment_is_rejected():
    with pytest.raises(ValueError):
        needs_wall_model("SST", "wallfunction")


def _mesh():
    m = build_channel_mesh_prism(1, 4, 3, 2, LX, H, LZ)
    m.boundary_bc_types = dict(_BC)
    return m


def _cpu(model, treatment):
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.fr_solver.turbulence.wall_distance import apply_wall_distance_source

    s = FRSolver(mesh=_mesh(), order=1, turb_model_name=model, time_scheme=TimeIntegrationScheme.SSP_RK3,
                 vel_inf=30.0, wall_treatment=treatment, rho_inf=RHO_INF, p_inf=P_INF, mu_molecular=MU_MOLECULAR)
    s.order_continuation_enabled = False
    apply_wall_distance_source(s, channel_wall_source(LX, H, LZ))
    return s


def _gpu(model, treatment):
    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver

    s = GPUFRSolver(mesh=_mesh(), order=1, turb_model_name=model, time_scheme=TimeIntegrationScheme.SSP_RK3,
                    vel_inf=30.0, wall_treatment=treatment, wall_distance_source=channel_wall_source(LX, H, LZ), rho_inf=RHO_INF, p_inf=P_INF, mu_molecular=MU_MOLECULAR)
    s.order_continuation_enabled = False
    return s


def _dist(model, treatment):
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

    mesh = _mesh()
    return DistributedFRSolver(mesh=mesh, ops=mesh.operators, face_connectivity=mesh.face_connectivity, n_ranks=1,
                               order=1, turb_model_name=model, time_scheme=TimeIntegrationScheme.SSP_RK3,
                               vel_inf=30.0, wall_treatment=treatment,
                               wall_distance_source=channel_wall_source(LX, H, LZ), rho_inf=RHO_INF, p_inf=P_INF, mu_molecular=MU_MOLECULAR)


def _wall_configs(solver):
    """WALL 组的幽灵态配置（CPU 分布式的幽灵态提供者在其内部的单机求解器上，按全局面编号）。"""
    provider = getattr(solver, "boundary_ghost_provider", None)
    if provider is None:
        provider = solver.local_solver.boundary_ghost_provider
    return [cfg for cfg in provider.code_to_config.values() if cfg["type"] == "WALL"]


@pytest.mark.parametrize("build", [_cpu, _gpu, _dist])
@pytest.mark.parametrize("model", ["SST", "SA"])
def test_auto_builds_the_wall_model_and_slip_ghosts_on_every_backend(build, model):
    s = build(model, "auto")
    assert s.wall_treatment == "auto" and s.wall_model is not None
    walls = _wall_configs(s)
    assert walls and all(not c["is_no_slip"] and c["physical_no_slip"] for c in walls)
    turb = getattr(s, "turb_model", None) or getattr(s, "turb_model_gpu", None)
    if model == "SST":
        assert turb.wall_treatment == "auto"


@pytest.mark.parametrize("build", [_cpu, _gpu, _dist])
def test_resolved_keeps_the_low_reynolds_wall(build):
    s = build("SST", "resolved")
    assert s.wall_model is None
    assert all(c["is_no_slip"] for c in _wall_configs(s))
    turb = getattr(s, "turb_model", None) or getattr(s, "turb_model_gpu", None)
    assert turb.wall_treatment == "resolved"


def test_auto_k_wall_condition_and_omega_target():
    """SST auto：k 不在壁面钉 0（零通量）；omega 壁面目标值 = Menter 混合式（d1 取均值口径）。"""
    from autoflowcfd.core.turbulence.sst.bounds import k_wall_zero_mask
    from autoflowcfd.core.turbulence.transport.omega_wall import (
        _compute_omega_wall_target, _compute_wall_dirichlet_face_mask)

    res, auto = _cpu("SST", "resolved"), _cpu("SST", "auto")
    for s in (res, auto):
        s.turb_model.k_field = np.full_like(s.turb_model.k_field, 0.5)
    wall = _compute_wall_dirichlet_face_mask(auto)
    assert wall.any() and np.array_equal(wall, _compute_wall_dirichlet_face_mask(res))
    assert not k_wall_zero_mask(wall, "auto", np).any() and np.array_equal(k_wall_zero_mask(wall, "resolved", np), wall)
    rho = auto.state.Q[..., 0]
    w_res, _ = _compute_omega_wall_target(res, wall, res.mu_molecular, rho)
    w_auto, _ = _compute_omega_wall_target(auto, wall, auto.mu_molecular, rho)
    assert not np.allclose(w_res[wall], w_auto[wall])


def test_auto_adds_the_wall_stress_to_the_residual():
    """auto 的粘性残差含壁面模型剪应力：剪切流（壁面附近有切向速度）下贴壁单元的动量残差与 resolved 不同，
    远离壁面的单元相同。"""
    res, auto = _cpu("SA", "resolved"), _cpu("SA", "auto")
    y = res.mesh.sps_coords[..., 1]
    for s in (res, auto):
        Q = s.state.Q.copy()
        Q[..., 1] = 30.0 * np.sin(np.pi * y / H)                  # 两壁处速度为 0、中间最大
        from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved
        s.state.U = np.stack([primitive_to_conserved(q) for q in Q.reshape(-1, 5)]).reshape(Q.shape)
        s.state._update_primitives()
    r_res, r_auto = res.compute_viscous_residual(), auto.compute_viscous_residual()
    cell_wall = np.zeros(res.mesh.n_cells, bool)
    fc = res.mesh.face_connectivity
    walls = np.flatnonzero(fc.is_boundary & ((np.abs(fc.center[:, 1]) < 1e-12) | (np.abs(fc.center[:, 1] - H) < 1e-12)))
    cell_wall[fc.owner_cell[walls]] = True
    diff = np.abs(r_auto - r_res)[..., 1:4].max(axis=(1, 2))
    assert diff[cell_wall].min() > 0.0                                  # 贴壁单元：壁面剪应力来自壁面模型
    assert diff[~cell_wall].max() <= 1e-9 * np.abs(r_res).max()         # 其余单元不变


def _invoke(tmp_path, command):
    from click.testing import CliRunner

    from autoflowcfd.cli.main import cli

    mesh_file = tmp_path / "mesh.pkl"
    mesh_file.write_bytes(b"")
    return CliRunner().invoke(cli, ["solve", command, str(mesh_file), "--turbulence-model", "sst",
                                    "--wall-treatment", "auto", "--max-iter", "1"])


def test_cli_steady_forwards_the_wall_treatment(tmp_path):
    from unittest.mock import patch

    with patch("autoflowcfd.cli.solve.run.run_solve") as run:
        result = _invoke(tmp_path, "steady")
    assert result.exit_code == 0, result.output
    assert run.call_args.args[0].physics["wall_treatment"] == "auto"


def test_cli_transient_forwards_the_wall_treatment(tmp_path):
    from unittest.mock import patch

    with patch("autoflowcfd.cli.solve.run.run_solve") as run:
        result = _invoke(tmp_path, "transient")
    assert result.exit_code == 0, result.output
    assert run.call_args.args[0].physics["wall_treatment"] == "auto"


def test_checkpoint_records_the_wall_treatment():
    from autoflowcfd.core.utils.checkpoint_physics import physics_metadata

    assert physics_metadata(_cpu("SST", "auto"))["wall_treatment"] == "auto"



def _set_shear(s, u_of_y, rho=RHO_Y):
    """守恒变量取给定流向速度剖面 u(y)、均匀密度与压力。"""
    from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved

    y = s.mesh.sps_coords[..., 1]
    Q = np.zeros(y.shape + (5,))
    Q[..., 0] = rho
    Q[..., 1] = u_of_y(y)
    Q[..., 4] = 101325.0
    s.state.U = np.stack([primitive_to_conserved(q) for q in Q.reshape(-1, 5)]).reshape(Q.shape)
    s.state._update_primitives()


def test_wall_y_plus_resolved_matches_the_exact_couette_value():
    """resolved：线性剪切 u = S y（P1 精确表示、壁面处为 0）下壁面剪应力恰为 mu S，贴壁单元形心高度为第一层
    单元高度的一半（棱柱通道 ny=3），y+ 与解析值一致；上壁同理（u 在 y=H 处不为 0，按 |du/dy| 计）。"""
    from autoflowcfd.postprocess.wall_yplus import wall_y_plus

    s = _cpu("SST", "resolved")
    S = 200.0
    _set_shear(s, lambda y: S * y)
    yp = wall_y_plus(s)
    bottom = np.abs(yp.position[:, 1]) < 1e-12
    mu = s.mu_molecular
    exact = np.sqrt(mu * S / RHO_Y) * (H / 6.0) / (mu / RHO_Y)
    np.testing.assert_allclose(yp.y_plus[bottom], exact, rtol=1e-9)


def test_wall_y_plus_auto_matches_the_wall_model():
    """auto：均匀密度下壁面通量点处的 nu 与匹配点相同，y+ 与壁面模型在匹配点（形心高度）算出的一致。"""
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.postprocess.wall_yplus import wall_y_plus

    s = _cpu("SST", "auto")
    _set_shear(s, lambda y: 30.0 * np.sin(np.pi * y / H))
    yp = wall_y_plus(s)
    m, _ = s.wall_model.traction(np, s.mesh, get_flat_face_geometry(s.mesh, s.ops), s.boundary_ghost_provider,
                                 s.state.U, s.mu_molecular)
    np.testing.assert_allclose(np.sort(yp.y_plus), np.sort(s.wall_model.y_plus), rtol=1e-9)
    assert yp.y_plus.size == m.owner.size


@pytest.mark.parametrize("treatment,expect_warning", [("resolved", True), ("auto", False)])
def test_y_plus_report_checks_the_treatment_range(monkeypatch, treatment, expect_warning):
    """贴壁单元 y+ 远大于 5、小于 300：resolved 报警（越出粘性底层），auto 不报警。"""
    from autoflowcfd.postprocess import wall_yplus

    s = _cpu("SST", treatment)
    _set_shear(s, lambda y: 3.0 * np.sin(np.pi * y / H))
    warnings = []
    monkeypatch.setattr(wall_yplus.logger, "warning", lambda msg: warnings.append(msg))
    yp = wall_yplus.report_wall_y_plus(s, "SST", treatment)
    assert wall_yplus.VISCOUS_SUBLAYER_Y_PLUS < yp.quantile(0.95) < wall_yplus.LOG_LAYER_Y_PLUS
    assert bool(warnings) is expect_warning


def test_wall_model_rejects_a_no_slip_ghost_configuration():
    """壁面模型下物理无滑移壁面的幽灵态若被覆盖成无滑移（is_no_slip=True），粘性公共通量已带壁面剪应力，再加壁面
    模型牵引就是重复计入：直接报错（此前静默双计）。"""
    s = _cpu("SA", "auto")
    for cfg in _wall_configs(s):
        cfg["is_no_slip"] = True
    s.wall_model._cache_flat = None
    with pytest.raises(ValueError, match="计入两次"):
        s.compute_viscous_residual()


def _moving_bottom_cpu(model, treatment, u_wall):
    """通道下壁以 u_wall 移动（组配置的 wall_velocity，例如移动地面），上壁静止。"""
    s = _cpu(model, treatment)
    provider = s.boundary_ghost_provider
    names = provider.code_to_name
    for code, cfg in provider.code_to_config.items():
        if names.get(code) == "wall_bottom":
            cfg["wall_velocity"] = list(u_wall)
    s.wall_model._cache_flat = None
    return s


def test_moving_wall_uses_the_relative_velocity_and_does_work():
    """移动壁面：壁面律取相对壁面的切向速度（流体随壁面运动时下壁剪应力为 0），剪应力对流体做功
    `tau . u_wall` 进能量方程；静止上壁能量分量为 0。"""
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.turbulence.wall_model.stress import wall_flux

    u_w = np.array([10.0, 0.0, 0.0])
    s = _moving_bottom_cpu("SA", "auto", u_w)
    flat = get_flat_face_geometry(s.mesh, s.ops)
    _set_shear(s, lambda y: np.full_like(y, u_w[0]))                    # 均匀流 = 下壁速度
    m, tau = s.wall_model.traction(np, s.mesh, flat, s.boundary_ghost_provider, s.state.U, s.mu_molecular)
    bottom = np.abs(m.normal[:, 1] + 1.0) < 1e-12                       # 外法向 -y
    # 插值后 (rho u)/rho 与 u_wall 差一个舍入单位：相对速度 ~1e-15 量级，剪应力相应只剩舍入量级
    assert np.abs(tau[bottom]).max() < 1e-12 * np.abs(tau[~bottom]).max()
    np.testing.assert_array_equal(m.wall_velocity[bottom], np.broadcast_to(u_w, (int(bottom.sum()), 3)))

    _set_shear(s, lambda y: 3.0 * np.sin(np.pi * y / H))
    m, tau = s.wall_model.traction(np, s.mesh, flat, s.boundary_ghost_provider, s.state.U, s.mu_molecular)
    F = wall_flux(np, m, tau)
    np.testing.assert_allclose(F[:, 3], m.area * (tau @ u_w) * bottom, rtol=1e-14, atol=0.0)
    assert np.abs(F[bottom, 3]).min() > 0.0


@pytest.mark.parametrize("treatment", ["resolved", "auto"])
def test_gpu_omega_wall_target_matches_cpu(treatment):
    """GPU 的 omega 壁面目标值与 CPU 一致（P1 棱柱 8 个槽位里 2 个是填充槽位）。2026-10-18 以前 GPU 面几何
    （`GPUFlatFaceGeometry`）没有上传 `n_prism`，GPU 恒按整行平均、把填充槽位的壁距/密度/k 混进去：auto 档
    （d1 取均值）目标值差 17%，DUAL_TIME 三个物理步后 omega 场差 7e-5。"""
    from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import compute_omega_wall_target_gpu
    from autoflowcfd.core.turbulence.transport.omega_wall import (
        _compute_omega_wall_target, _compute_wall_dirichlet_face_mask)

    cpu, gpu = _cpu("SST", treatment), _gpu("SST", treatment)
    rng = np.random.default_rng(3)
    k = 0.5 * (1.0 + 0.3 * rng.standard_normal(cpu.turb_model.k_field.shape)) ** 2
    cpu.turb_model.k_field = k.copy()
    gpu.turb_model_gpu.k_field = k.copy()
    assert gpu.flat_face_gpu.n_prism == cpu.mesh.n_prism_cells
    wall = _compute_wall_dirichlet_face_mask(cpu)
    ref, _ = _compute_omega_wall_target(cpu, wall, cpu.mu_molecular, cpu.state.Q[..., 0])
    got, _ = compute_omega_wall_target_gpu(
        np, gpu.flat_face_gpu, gpu._wall_mask_gpu, gpu.wall_distance_gpu, gpu.Q_gpu, gpu.mu_molecular,
        gpu.turb_model_gpu.beta1, gpu.turb_model_gpu.omega_max, gpu.turb_model_gpu.k_field, treatment)
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-12, atol=0.0)
