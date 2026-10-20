# -*- coding: utf-8 -*-
"""合成湍流入口（SEM）在求解器里的接入：确定性种子、按来流向量推进、四个后端每物理步推进、参数传递。

2026-10-04 查出并修复：
1. 种子为 None：同一算例两次运行的入口脉动不同；CPU 分布式传统模式各 rank 各自构造、涡核互不相同。
2. CPU 单机推进 SEM 时平动速度写死 `(vel_inf, 0, 0)`，与按真实来流方向配置的入口盒不一致。
3. 单 GPU、多 GPU、CPU 分布式从不推进 SEM（入口涡结构在时间上冻结）。
4. `sem_num_eddies` 只有 CPU 单机在传：单 GPU 静默用默认值、多 GPU 写死 200、`solve transient` 的分布式路径
   不接收；完全分布式加载的根桩缺 `_sem_num_eddies/_turbulence_intensity`，边界构造的兜底让 CLI 设置被忽略。
"""

import inspect
import types
import zlib

import numpy as np
import pytest

from autoflowcfd.boundary.synthetic_inlet import advance_synthetic_inlets, synthetic_inlet_seed, synthetic_inlets
from tests.unit._module_source import module_source
from tests.validation._channel_mesh import build_channel_mesh_prism

FREESTREAM = {"rho_inf": 1.225, "vel_inf": 30.0, "p_inf": 101325.0, "aoa_deg": 10.0, "aos_deg": 0.0}


def _stub(**overrides):
    from autoflowcfd.fr.operators import generate_fr_operators

    mesh = build_channel_mesh_prism(1, 2, 2, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL",
                              "wall_top": "WALL", "z_min": "SYMMETRY", "z_max": "SYMMETRY"}
    attrs = dict(mesh=mesh, ops=generate_fr_operators(1), freestream=dict(FREESTREAM), turb_model_name="LES",
                 wall_model=None, _sem_num_eddies=40, _turbulence_intensity=0.05, _viscosity_ratio=5.0,
                 mu_molecular=1.8e-5)
    attrs.update(overrides)
    return types.SimpleNamespace(**attrs)


def _provider(stub):
    from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider

    return build_boundary_ghost_provider(stub, bc_overrides={})


def test_inlet_uses_the_users_eddy_count_and_a_deterministic_seed():
    a, b = synthetic_inlets(_provider(_stub())), synthetic_inlets(_provider(_stub()))
    assert len(a) == 1 and a[0].num_eddies == 40
    np.testing.assert_array_equal(a[0].eddy_centers, b[0].eddy_centers)
    np.testing.assert_array_equal(a[0].eddy_signs, b[0].eddy_signs)
    assert synthetic_inlet_seed("x_min") == zlib.crc32(b"x_min") != synthetic_inlet_seed("x_max")


@pytest.mark.parametrize("model", ["LES", "WMLES", "DDES", "IDDES"])
def test_every_scale_resolving_model_gets_a_synthetic_inlet(model):
    """解析湍流脉动的模型都在速度入口注入合成湍流（I-75：此前 WMLES 被排除，入口是层流）；RANS 不注入。"""
    assert len(synthetic_inlets(_provider(_stub(turb_model_name=model)))) == 1
    assert not synthetic_inlets(_provider(_stub(turb_model_name="SST")))


@pytest.mark.parametrize("missing", ["_sem_num_eddies", "_turbulence_intensity", "_viscosity_ratio", "mu_molecular"])
def test_missing_solver_settings_are_an_error_not_a_default(missing):
    stub = _stub()
    delattr(stub, missing)
    with pytest.raises(AttributeError):
        _provider(stub)


def test_advance_follows_the_freestream_vector():
    """平动速度 = vel_inf x 来流方向（攻角 10 度）：涡核（入口局部坐标）位移恰为 E v dt，且 E 的第一行即来流方向。"""
    from autoflowcfd.core.utils.flow_direction import direction_from_freestream

    provider = _provider(_stub())
    sem = synthetic_inlets(provider)[0]
    before = sem.eddy_centers.copy()
    dt = 1e-6
    advance_synthetic_inlets(provider, FREESTREAM, dt)
    v = FREESTREAM["vel_inf"] * direction_from_freestream(FREESTREAM)
    assert np.abs(v[1:]).max() > 1.0, "测试需要非零的横向分量（攻角在 x-z 平面内）"
    np.testing.assert_allclose(sem.frame[0], v / np.linalg.norm(v), rtol=1e-14)
    np.testing.assert_allclose(sem.eddy_centers - before, np.broadcast_to(sem.frame @ v * dt, before.shape), rtol=1e-9,
                               atol=1e-15)


@pytest.mark.parametrize("module", [
    "autoflowcfd.core.fr_solver.step",
    "autoflowcfd.core.mpi.distributed_solver.step",
    "autoflowcfd.core.gpu.solver.gpu_solver.step",
    "autoflowcfd.core.gpu.distributed.gpu_distributed.stepping",
])
def test_every_backend_advances_the_inlets_each_physical_step(module):
    import importlib

    assert "advance_synthetic_inlets(" in module_source(importlib.import_module(module))


@pytest.mark.parametrize("cls_path", [
    ("autoflowcfd.core.gpu.solver.gpu_solver", "GPUFRSolver"),
    ("autoflowcfd.core.gpu.distributed.gpu_distributed", "MultiGPUDistributedSolver"),
])
def test_gpu_solvers_accept_the_eddy_count(cls_path):
    import importlib

    from autoflowcfd.core.fr_solver.boundary.constants import _SEM_DEFAULT_NUM_EDDIES

    cls = getattr(importlib.import_module(cls_path[0]), cls_path[1])
    assert inspect.signature(cls.__init__).parameters["sem_num_eddies"].default == _SEM_DEFAULT_NUM_EDDIES


def test_fully_distributed_root_stub_carries_the_users_settings():
    from autoflowcfd.core.mpi.distributed_mesh_loader import fully_distributed

    src = module_source(fully_distributed)
    for attr in ("_turbulence_intensity=", "_viscosity_ratio=", "mu_molecular=", "_sem_num_eddies="):
        assert src.count(attr) >= 2, attr
    assert "sem_num_eddies" in inspect.signature(fully_distributed.distributed_mesh_load_v2).parameters


def test_transient_distributed_cli_forwards_the_eddy_count(tmp_path):
    from unittest.mock import patch

    from click.testing import CliRunner

    from autoflowcfd.cli.main import cli

    mesh_file = tmp_path / "mesh.pkl"
    mesh_file.write_bytes(b"")
    with patch("autoflowcfd.cli.solve.run.run_solve") as run:
        result = CliRunner().invoke(cli, ["solve", "transient", str(mesh_file), "--n-ranks", "2",
                                          "--sem-num-eddies", "77", "--max-iter", "1"])
    assert result.exit_code == 0, result.output
    assert run.call_args.args[0].sem_num_eddies == 77


@pytest.mark.parametrize("tu,vr", [(0.05, 5.0), (0.002, 200.0)])
def test_eddy_scale_and_reynolds_stress_come_from_the_freestream_turbulence(tu, vr):
    """sigma = max(k^1.5/epsilon, Delta_owner)，epsilon = beta* k omega；R = (2/3) k I（k、omega 与 RANS 来流同一函数）。

    两组来流：大涡粘比时耗散尺度大于网格尺度（取前者），小涡粘比时被网格尺度截住（取后者）。修复前恒为
    "入口包围盒最大跨度 / 10"。"""
    from autoflowcfd.core.fr_solver.turbulence.init import _set_freestream_turbulence
    from autoflowcfd.core.turbulence.sgs import les_filter_width

    stub = _stub(_turbulence_intensity=tu, _viscosity_ratio=vr)
    provider = _provider(stub)
    ghost = next(cfg["sem"] for cfg in provider.code_to_config.values() if cfg.get("sem") is not None)
    k, omega = _set_freestream_turbulence(stub)
    mesh = stub.mesh
    delta = les_filter_width(mesh.cell_volumes, np.arange(mesh.n_cells) < mesh.n_prism_cells, 1)
    ell = np.sqrt(k) / (0.09 * omega)
    for f, sig in ghost.face_sigma.items():
        np.testing.assert_allclose(sig, max(ell, delta[mesh.face_connectivity.owner_cell[f]]), rtol=1e-14)
    np.testing.assert_allclose(ghost.reynolds_stress, (2.0 / 3.0) * k * np.eye(3), rtol=1e-14)
    sig_all = np.concatenate(list(ghost.face_sigma.values()))
    assert bool(np.isclose(sig_all.min(), ell, rtol=1e-14)) == (vr > 100.0)      # 哪一项起作用


def test_cli_default_eddy_count_is_the_shared_constant():
    from autoflowcfd.cli.solve.steady.command import solve_steady
    from autoflowcfd.cli.solve.transient import transient as solve_transient
    from autoflowcfd.core.fr_solver.boundary.constants import _SEM_DEFAULT_NUM_EDDIES

    for cmd in (solve_steady, solve_transient):
        opt = next(p for p in cmd.params if p.name == "sem_num_eddies")
        assert opt.default is _SEM_DEFAULT_NUM_EDDIES is None
