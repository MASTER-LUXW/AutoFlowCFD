"""分布式湍流求值的渐变计数器、湍流模型名与幽灵态提供者是必需参数（I-101）。

此前默认值（None -> 渐变步数为 0 的假计数器、"SST"、None）：2026-09-25 的真实缺陷正是适配器缺幽灵态提供者，
CPU 分布式上壁面 k=0、omega 壁面值与来流条件全部静默退化；默认值让同类缺陷可以再次无声出现。
"""

import inspect

import pytest

from autoflowcfd.core.mpi import distributed_turbulence as dt


@pytest.mark.parametrize("fn, names", [
    (dt.DistributedTurbulenceSolverAdapter.__init__,
     ("ramp_owner", "turb_model_name", "boundary_ghost_provider", "halo_refresh")),
    (dt.distributed_compute_turbulence_source_and_viscosity,
     ("ramp_owner", "turb_model_name", "boundary_ghost_provider")),
])
def test_required_keyword_arguments(fn, names):
    params = inspect.signature(fn).parameters
    for name in names:
        assert params[name].default is inspect.Parameter.empty, name
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name


def test_adapter_has_no_dead_gradient_method():
    assert not hasattr(dt.DistributedTurbulenceSolverAdapter, "_compute_gradients")
