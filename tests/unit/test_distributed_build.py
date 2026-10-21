"""分布式求解器的唯一构造入口（I-175，`cli/solve/distributed_build.py`）：四种组合都收到同一组参数，值原样到达。

此前这段构造在 `solve steady` 两个分支、`solve transient` 三个分支、续算重建四个分支里各写一份：先后出过完全分布式
路径缺攻角（必现 NameError）、四处漏传 CFL 边界（静默用默认值）、多 GPU 构造参数名不同（`turb_model=` 与大写名）、
多 GPU 的 dual-time 内迭代上限要在构造后改属性。这里不再按源码文本核对转发，而是让四个构造点真正被调用，
逐项比对收到的值。
"""

from unittest.mock import MagicMock, patch

import pytest

from autoflowcfd.core.time_integration.base import TimeIntegrationScheme

_PHYSICS = dict(rho_inf=1.1, vel_inf=21.0, p_inf=99000.0, aoa_deg=3.5, aos_deg=-1.25, mu_molecular=1.9e-5,
                turbulence_intensity=0.02, viscosity_ratio=7.0, artificial_viscosity_enabled=True,
                artificial_viscosity_alpha=0.7, wall_treatment="auto")
_NUMERICS = dict(cfl_start=0.02, cfl_max=0.05, cfl_min=0.005, low_mach_precond=False, time_accurate=True,
                 sem_num_eddies=321)


def _build(multi_gpu, fully_distributed):
    from autoflowcfd.cli.solve.distributed_build import build_distributed_solver

    calls = {}

    def record(name, ret):
        def f(*args, **kwargs):
            calls[name] = (args, kwargs)
            return ret
        return f

    mesh = MagicMock()
    p = "autoflowcfd.cli.solve.distributed_build"
    with patch(f"{p}.require_distributed_runtime"), \
         patch("autoflowcfd.cli.solve.mesh_loader.load_mesh_for_solver", side_effect=record("load", (mesh, "vd"))), \
         patch("autoflowcfd.cli.solve.wall_distance.wall_distance_source_if_needed", return_value="wd"), \
         patch("autoflowcfd.fr.operators.generate_fr_operators", return_value="ops"), \
         patch("autoflowcfd.core.mpi.distributed_mesh_loader.distributed_mesh_load_v2",
               side_effect=record("load_v2", ("pkg", "ctx"))), \
         patch("autoflowcfd.core.mpi.distributed_solver.DistributedFRSolver",
               side_effect=record("cpu", "solver")) as cpu, \
         patch("autoflowcfd.core.gpu.distributed.gpu_distributed.MultiGPUDistributedSolver",
               side_effect=record("gpu", "solver")) as gpu:
        cpu.from_fully_distributed_package.side_effect = record("cpu_fd", "solver")
        gpu.from_fully_distributed_package.side_effect = record("gpu_fd", "solver")
        solver, volume_data = build_distributed_solver(
            multi_gpu=multi_gpu, fully_distributed=fully_distributed, input_file="m.pkl", order=2,
            surface_mesh=None, skip_quality_check=True, n_ranks=4, gpu_device=1, threads=6,
            turbulence_model="ddes", time_scheme=TimeIntegrationScheme.DUAL_TIME, physics=dict(_PHYSICS),
            dual_time_inner_iter=9, **_NUMERICS)
    assert solver == "solver"
    return calls, volume_data


@pytest.mark.parametrize("multi_gpu", [False, True])
def test_traditional_mode_forwards_every_parameter(multi_gpu):
    calls, volume_data = _build(multi_gpu, False)
    assert volume_data == "vd" and calls["load"][1]["skip_quality_check"] is True
    kw = calls["gpu" if multi_gpu else "cpu"][1]
    for key, value in {**_PHYSICS, **_NUMERICS}.items():
        assert kw[key] == value, key
    assert kw["turb_model_name"] == "ddes" and kw["time_scheme"] is TimeIntegrationScheme.DUAL_TIME
    assert kw["dual_time_inner_iter"] == 9 and kw["wall_distance_source"] == "wd" and kw["n_ranks"] == 4
    if multi_gpu:
        assert kw["device_id"] == 1
    else:
        assert kw["n_threads"] == 6 and kw["order"] == 2


@pytest.mark.parametrize("multi_gpu", [False, True])
def test_fully_distributed_mode_forwards_every_parameter(multi_gpu):
    calls, volume_data = _build(multi_gpu, True)
    assert volume_data is None
    args, kw = calls["load_v2"]
    assert args[1:] == (2, 4) and "load" not in calls        # root 网格加载函数只在 root 的加载流程里调用
    args[0]()
    assert calls["load"] == (("m.pkl", 2), {"surface_mesh": None, "skip_quality_check": True})
    assert kw["freestream"] == {k: _PHYSICS[k] for k in ("rho_inf", "vel_inf", "p_inf", "aoa_deg", "aos_deg")}
    for key in ("mu_molecular", "turbulence_intensity", "viscosity_ratio", "artificial_viscosity_enabled",
                "artificial_viscosity_alpha", "wall_treatment"):
        assert kw[key] == _PHYSICS[key], key
    for key, value in _NUMERICS.items():
        assert kw[key] == value, key
    assert kw["turb_model_name"] == "DDES" and kw["dual_time_inner_iter"] == 9
    fd = calls["gpu_fd" if multi_gpu else "cpu_fd"][1]
    assert fd["n_ranks"] == 4 and fd["root_context"] == "ctx"
    assert (fd["device_id"] == 1) if multi_gpu else (fd["n_threads"] == 6)


def test_multi_gpu_constructor_takes_the_same_parameter_names():
    """多 GPU 构造参数与单机 / CPU 分布式同名：湍流模型名、dual-time 内迭代上限。"""
    import inspect

    from autoflowcfd.core.gpu.distributed.gpu_distributed import MultiGPUDistributedSolver
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

    params = inspect.signature(MultiGPUDistributedSolver.__init__).parameters
    assert "turb_model_name" in params and "turb_model" not in params and "dual_time_inner_iter" in params
    assert "n_threads" in inspect.signature(DistributedFRSolver.from_fully_distributed_package).parameters
