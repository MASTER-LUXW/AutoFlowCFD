"""默认来流与分子粘度只在 `core/freestream_defaults.py` 定义（I-46，2026-10-19），且只在配置层作为默认值（I-78）。

此前这组值在配置类（两份）、四个求解器构造函数、分布式求解器的 `solver_kwargs.get(...)`、完全分布式包读取与 CLI
各写一份；完全分布式包读取的 `.get(..., 33.33/1.225/101325)` 还会在包缺键时静默按缺省来流运行。
"""

import ast
import inspect
from pathlib import Path

import pytest

import autoflowcfd
from autoflowcfd.core import freestream_defaults as fd

DEFAULTS = {"rho_inf": fd.RHO_INF, "vel_inf": fd.VEL_INF, "p_inf": fd.P_INF, "mu_molecular": fd.MU_MOLECULAR}


def _constructors():
    from autoflowcfd.core.fr_solver.solver import FRSolver
    from autoflowcfd.core.gpu.distributed.gpu_distributed import MultiGPUDistributedSolver
    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver

    return (FRSolver, GPUFRSolver, MultiGPUDistributedSolver)


_REQUIRED = {"FRSolver": ("order", "turb_model_name", "time_scheme"),
             "GPUFRSolver": ("order", "turb_model_name", "time_scheme"),
             "MultiGPUDistributedSolver": ("turb_model_name", "time_scheme")}


@pytest.mark.parametrize("cls", _constructors(), ids=lambda c: c.__name__)
def test_solver_physical_inputs_are_required(cls):
    """求解器构造的物理输入没有默认值（I-78）：用户可见的默认值只在配置层（下一条用例）。此前给了默认值、且各后端
    的湍流模型默认值不一致（CPU SST、单 GPU 与多 GPU 层流），漏传即静默按另一个算例计算。"""
    params = inspect.signature(cls.__init__).parameters
    for name in tuple(DEFAULTS) + _REQUIRED[cls.__name__]:
        assert params[name].default is inspect.Parameter.empty, (cls.__name__, name)
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, (cls.__name__, name)


def test_cpu_distributed_solver_requires_the_physical_inputs():
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
    from autoflowcfd.core.mpi.distributed_solver.core import _REQUIRED_KWARGS

    assert set(_REQUIRED_KWARGS) == set(DEFAULTS) | {"order", "turb_model_name", "time_scheme"}
    with pytest.raises(TypeError, match="rho_inf"):
        DistributedFRSolver(mesh=None, ops=None, face_connectivity=None, n_ranks=1, order=1, turb_model_name="none",
                            time_scheme="ssp_rk3", vel_inf=1.0, p_inf=1.0, mu_molecular=1.0)


def test_config_defaults_are_the_shared_constants():
    from autoflowcfd.config.solver_config import SteadyConfig, TransientConfig

    for cfg in (SteadyConfig(), TransientConfig()):
        for name in ("rho_inf", "vel_inf", "p_inf"):
            assert getattr(cfg, name) == DEFAULTS[name]
    assert SteadyConfig().mu_molecular == fd.MU_MOLECULAR


def test_no_other_module_writes_the_default_values():
    """源码里不再有这组默认值的字面量（旧 checkpoint 的历史恢复值与文档字符串除外）。"""
    src = Path(autoflowcfd.__file__).parent
    allowed = {"core/freestream_defaults.py", "cli/solve/checkpoint_io/rebuild.py", "core/utils/checkpoint_physics.py"}
    values = {33.33, 1.225, 101325.0, 1.8e-5}
    offenders = []
    for path in src.rglob("*.py"):
        rel = path.relative_to(src).as_posix()
        if rel in allowed:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, float) and node.value in values:
                offenders.append(f"{rel}:{node.lineno}={node.value}")
    assert offenders == []


def test_fully_distributed_package_without_freestream_key_is_an_error():
    """完全分布式包读取直接取键：缺键报错，不按缺省来流静默运行（来流经 `_settings_from_package` 原样进入
    `solver.freestream`，初场在共用装配 `assembly.py` 里直接取键）。"""
    from tests.unit._module_source import module_source

    src = module_source("autoflowcfd.core.mpi.distributed_solver")
    assert "{**package['freestream'], \"mach_ref\": package['mach_ref']}" in src
    for key in ("vel_inf", "rho_inf", "p_inf"):
        assert f"freestream'].get('{key}'" not in src and f'freestream.get("{key}"' not in src
        assert f'self.freestream["{key}"]' in src
