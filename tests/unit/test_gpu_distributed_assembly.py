"""多 GPU 两条构造路径共用装配阶段（I-110，`core/gpu/distributed/gpu_distributed/setup.py`）：完全分布式构造初始化
产生项渐变计数器；传统模式与完全分布式的属性集合只在几何来源相关的少数属性上不同。夹具与替身见
`test_gpu_distributed_fully_distributed.py`（方法论同那份模块文档）。"""

import numpy as np

from tests.unit._wall_source import synthetic_wall_source
from tests.unit.test_gpu_distributed_fully_distributed import (  # noqa: F401  （夹具）
    _build_sst_package, _patch_sst_standins, gpu_shim, mesh_and_ops,
)


def test_fully_distributed_sst_initializes_the_production_ramp(mesh_and_ops, gpu_shim, monkeypatch):
    """完全分布式构造经两条路径共用的 `_setup_time_integration`（I-110）：产生项渐变计数器已初始化。2026-10-22
    以前这条路径从不调用 `init_production_ramp`，输运模型第一步 `advance_production_ramp` 即 AttributeError。"""
    from autoflowcfd.core.fr_solver.turbulence.init import production_ramp_steps
    from autoflowcfd.core.gpu.distributed.gpu_distributed import MultiGPUDistributedSolver
    from autoflowcfd.core.gpu.distributed.gpu_distributed_fully_distributed import (
        build_multi_gpu_solver_from_fully_distributed_package,
    )

    _patch_sst_standins(monkeypatch)
    mesh, ops = mesh_and_ops
    package = _build_sst_package(mesh, ops)
    solver = build_multi_gpu_solver_from_fully_distributed_package(
        MultiGPUDistributedSolver, package, n_ranks=1, device_id=0, rank=0, root_context=None)
    assert solver._turb_ramp_step == 0 and solver._turb_production_ramp_complete is False
    assert solver._turb_production_ramp_steps == production_ramp_steps(package["time_scheme"])


def test_both_multi_gpu_constructors_assemble_the_same_attributes(mesh_and_ops, gpu_shim, monkeypatch):
    """多 GPU 传统模式与完全分布式构造共用装配阶段：属性集合只在几何来源相关的少数属性上不同（I-110）。"""
    from autoflowcfd.core.gpu.distributed.gpu_distributed import MultiGPUDistributedSolver
    from autoflowcfd.core.gpu.distributed.gpu_distributed_fully_distributed import (
        build_multi_gpu_solver_from_fully_distributed_package,
    )
    from autoflowcfd.core.time_integration import TimeIntegrationScheme

    _patch_sst_standins(monkeypatch)
    mesh, ops = mesh_and_ops
    monkeypatch.setattr(MultiGPUDistributedSolver, "_init_wall_distance_distributed",
                        lambda self: setattr(self, "wall_distance_gpu", np.zeros(1)))
    traditional = MultiGPUDistributedSolver(
        mesh, ops, 1, rank=0, device_id=0, time_scheme=TimeIntegrationScheme.SSP_RK3, mu_molecular=1.8e-5,
        rho_inf=1.225, vel_inf=33.33, p_inf=101325.0, turb_model_name="SST", turbulence_intensity=0.02,
        viscosity_ratio=8.0, wall_distance_source=synthetic_wall_source(mesh))
    fully = build_multi_gpu_solver_from_fully_distributed_package(
        MultiGPUDistributedSolver, _build_sst_package(mesh, ops), n_ranks=1, device_id=0, rank=0, root_context=None)
    a, b = set(vars(traditional)), set(vars(fully))
    assert a - b <= {"_wall_distance_source"}, a - b
    assert b - a <= {"_root_context", "_block_jacobi_colors_local", "_coupling_colors_local"}, b - a
    for name in ("turb_model_name", "mu_molecular", "_turbulence_intensity", "_viscosity_ratio", "wall_treatment",
                 "low_mach_precond_enabled", "_turb_production_ramp_steps", "order"):
        assert getattr(traditional, name) == getattr(fully, name), name
