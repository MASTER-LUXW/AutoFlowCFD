"""RANS 自动壁面处理（`wall_treatment="auto"`，`core/turbulence/wall_model/treatment.py`，I-137）的接入。

auto：平衡壁面模型给壁面剪应力、物理无滑移壁面取滑移镜像幽灵态；SST 的 omega 壁面值取 Menter 混合式、k 壁面
零通量；SA-neg 只换动量壁面条件。resolved（默认）保持既有低雷诺数壁面处理。四个后端由同一组函数构造。
"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration import TimeIntegrationScheme
from autoflowcfd.core.turbulence.wall_model.treatment import build_wall_model, needs_wall_model
from tests.unit.test_gpu_solver_order_continuation import _patch_gpu_modules  # noqa: F401（自动夹具）
from tests.validation._channel_mesh import build_channel_mesh_prism, channel_wall_source

LX, H, LZ = 0.4, 0.1, 0.08
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
                 vel_inf=30.0, wall_treatment=treatment)
    s.order_continuation_enabled = False
    apply_wall_distance_source(s, channel_wall_source(LX, H, LZ))
    return s


def _gpu(model, treatment):
    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver

    s = GPUFRSolver(mesh=_mesh(), order=1, turb_model_name=model, time_scheme=TimeIntegrationScheme.SSP_RK3,
                    vel_inf=30.0, wall_treatment=treatment, wall_distance_source=channel_wall_source(LX, H, LZ))
    s.order_continuation_enabled = False
    return s


def _dist(model, treatment):
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

    mesh = _mesh()
    return DistributedFRSolver(mesh=mesh, ops=mesh.operators, face_connectivity=mesh.face_connectivity, n_ranks=1,
                               order=1, turb_model_name=model, time_scheme=TimeIntegrationScheme.SSP_RK3,
                               vel_inf=30.0, wall_treatment=treatment,
                               wall_distance_source=channel_wall_source(LX, H, LZ))


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

    with patch("autoflowcfd.cli.solve.steady.command._run_single_node") as run:
        result = _invoke(tmp_path, "steady")
    assert result.exit_code == 0, result.output
    assert run.call_args.kwargs["wall_treatment"] == "auto"


def test_cli_transient_forwards_the_wall_treatment(tmp_path):
    from unittest.mock import patch

    class Stop(Exception):
        pass

    seen = {}

    def build(*a, **kw):
        seen.update(kw)
        raise Stop
    with patch("autoflowcfd.cli.solve.transient.load_mesh_for_solver", return_value=(object(), object())),             patch("autoflowcfd.cli.solve.transient.build_single_node_solver", side_effect=build):
        _invoke(tmp_path, "transient")
    assert seen["wall_treatment"] == "auto"


def test_checkpoint_records_the_wall_treatment():
    from autoflowcfd.core.utils.checkpoint_physics import physics_metadata

    assert physics_metadata(_cpu("SST", "auto"))["wall_treatment"] == "auto"

