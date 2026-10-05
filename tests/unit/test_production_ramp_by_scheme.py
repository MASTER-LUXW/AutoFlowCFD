"""湍流产生项渐变按时间格式取步数，五个求解器构造点同一个初始化函数（2026-10-05）。

* 隐式稳态（Newton-Krylov）不渐变：按步数计的 50 步渐变在隐式上几乎覆盖整个 P0，期间每步要解的方程都在变，
  Newton 追着移动目标走。湍流平板 P0->P3 A/B：SA P0 55->30 步（P1~P3 相同、cf 逐位相同），SST P0 119->55 步
  （见 `fr_solver/turbulence/init.py::production_ramp_steps`）。
* 构造时不推进计数器：此前 CPU 单机与 CPU MPI 传统模式在构造时推进了一次（渐变期间产生项因子比单 GPU / CPU MPI
  完全分布式 / 多 GPU 多 1/N），多 GPU 靠 `advance_production_ramp` 里的懒默认值。
"""

import inspect

import pytest

from autoflowcfd.core.fr_solver.turbulence.init import (
    TURB_PRODUCTION_RAMP_STEPS, advance_production_ramp, production_ramp_steps,
)
from autoflowcfd.core.time_integration import TimeIntegrationScheme
from tests.unit.test_gpu_solver_order_continuation import _patch_gpu_modules  # noqa: F401（自动夹具）
from tests.unit.test_sa_neg_solver import LX, H, LZ, _sa_channel
from tests.validation._channel_mesh import build_channel_mesh_prism, channel_wall_source

NK = TimeIntegrationScheme.NEWTON_KRYLOV
RK3 = TimeIntegrationScheme.SSP_RK3


@pytest.mark.parametrize("scheme,expected", [
    (NK, 0), ("newton-krylov", 0),
    (RK3, TURB_PRODUCTION_RAMP_STEPS), (TimeIntegrationScheme.DUAL_TIME, TURB_PRODUCTION_RAMP_STEPS),
    (TimeIntegrationScheme.IMEX_EULER, TURB_PRODUCTION_RAMP_STEPS),
])
def test_ramp_steps_follow_the_time_scheme(scheme, expected):
    assert production_ramp_steps(scheme) == expected


def _assert_fresh_counters(solver, scheme):
    assert solver._turb_production_ramp_steps == production_ramp_steps(scheme)
    assert solver._turb_ramp_step == 0, "构造时不推进计数器（五个构造点一致）"
    assert solver._turb_production_ramp_complete is False


def _assert_first_step_factor(solver, model, scheme):
    advance_production_ramp(solver, model)
    if scheme == NK:
        assert float(model.production_factor) == 1.0 and solver._turb_production_ramp_complete
    else:
        assert float(model.production_factor) == 0.0 and not solver._turb_production_ramp_complete


@pytest.mark.parametrize("scheme", [NK, RK3])
def test_cpu_single_machine(scheme):
    s = _sa_channel("prism", 1, scheme)
    _assert_fresh_counters(s, scheme)
    _assert_first_step_factor(s, s.turb_model, scheme)


@pytest.mark.parametrize("scheme", [NK, RK3])
def test_cpu_distributed_traditional(scheme):
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
    from autoflowcfd.fr.operators import generate_fr_operators

    mesh = build_channel_mesh_prism(1, 3, 4, 2, LX, H, LZ)
    s = DistributedFRSolver(
        mesh=mesh, ops=generate_fr_operators(1), face_connectivity=mesh.face_connectivity,
        n_ranks=1, order=1, turb_model_name="SA", time_scheme=scheme,
        wall_distance_source=channel_wall_source(LX, H, LZ))
    _assert_fresh_counters(s, scheme)
    _assert_first_step_factor(s, s.turb_model, scheme)


@pytest.mark.parametrize("scheme", [NK, RK3])
def test_single_gpu(scheme):
    from tests.unit.test_single_gpu_unified import _gpu_solver

    s = _gpu_solver("SA", time_scheme=scheme)
    _assert_fresh_counters(s, scheme)
    _assert_first_step_factor(s, s.turb_model_gpu, scheme)


def test_package_and_multi_gpu_constructors_use_the_shared_initializer():
    """CPU MPI 完全分布式与多 GPU 的构造要真实分区包 / CuPy，这里钉住它们走同一个初始化函数；
    `advance_production_ramp` 不再有懒默认值，漏掉初始化会直接报错而不是静默取 50。"""
    from autoflowcfd.core.gpu.distributed.gpu_distributed import setup as mgpu_setup
    from autoflowcfd.core.mpi.distributed_solver import from_package

    assert "init_production_ramp(self, time_scheme)" in inspect.getsource(from_package)
    assert "init_production_ramp(self, time_scheme)" in inspect.getsource(mgpu_setup)
    assert "getattr" not in inspect.getsource(advance_production_ramp)


def test_cpu_distributed_explicit_ramp_completion_reaches_the_solver():
    """CPU MPI 显式路径在每步重建的湍流适配器上推进渐变；2026-10-05 以前适配器持有计数器副本，计数靠返回值写回、
    完成标记却留在适配器上，求解器上的 `_turb_production_ramp_complete` 永远是 False（Order Continuation 的
    渐变完成基准重置从不触发、PhaseGate 读不到完成）。现在适配器把三个属性转发到求解器。"""
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
    from autoflowcfd.fr.operators import generate_fr_operators

    mesh = build_channel_mesh_prism(1, 3, 4, 2, LX, H, LZ)
    s = DistributedFRSolver(
        mesh=mesh, ops=generate_fr_operators(1), face_connectivity=mesh.face_connectivity,
        n_ranks=1, order=1, turb_model_name="SA", time_scheme=RK3,
        wall_distance_source=channel_wall_source(LX, H, LZ))
    s._turb_production_ramp_steps = 2
    for _ in range(3):
        s.step(1e-6)
    assert s._turb_ramp_step == 3
    assert s._turb_production_ramp_complete is True
