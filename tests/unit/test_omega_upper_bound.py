# -*- coding: utf-8 -*-
"""SST 的 omega 上界随最近壁面解点给定（`turbulence/sst/bounds.py` 模块文档，2026-10-03）。

此前 `omega_max` 写死 1e6。壁面解析网格（y+~1）上贴壁解点的物理壁面 omega
`60 nu/(beta1 d^2)` 远超这个数：湍流平板 `d_min = 4.2e-6` 时为 2.7e8，壁面条件被
压低两个量级，钳位处残差不可微，P1 停在 400 步不收敛（162 个解点顶到钳位）；改成按
最小壁距给定后 187 步收敛到 1e-8，没有解点被钳住。

判据：
1. 上界公式：粗网格保持 1e6 的下限，细网格取 `10 * 60 nu/(beta1 d_min^2)`，非有限
   或非正的壁距退回下限；
2. 四个后端在壁距设定（含换阶重查）后都按当前壁距更新上界：CPU 单机
   `apply_wall_distance_source`、单 GPU `_init_wall_distance_gpu`、CPU 分布式与
   多 GPU 共用的 `apply_distributed_omega_bound`（按**全局**最小壁距，各 rank 一致）。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.core.turbulence.sst.bounds import (
    OMEGA_MAX_FLOOR,
    OMEGA_MAX_WALL_FACTOR,
    OMEGA_WALL_AMPLIFICATION,
    OMEGA_WALL_VISCOUS_COEFF,
    omega_upper_bound,
)

RHO, MU, BETA1 = 1.225, 7.35e-6, 0.075
NU = MU / RHO


def _expected(d_min):
    return OMEGA_MAX_WALL_FACTOR * OMEGA_WALL_AMPLIFICATION * OMEGA_WALL_VISCOUS_COEFF * NU / (BETA1 * d_min ** 2)


def _model():
    return SimpleNamespace(omega_max=OMEGA_MAX_FLOOR, beta1=BETA1)


def test_upper_bound_formula():
    assert omega_upper_bound(1e-2, NU, BETA1) == OMEGA_MAX_FLOOR
    assert omega_upper_bound(4.23e-6, NU, BETA1) == pytest.approx(_expected(4.23e-6), rel=1e-14)
    assert omega_upper_bound(4.23e-6, NU, BETA1) > 1e9
    for bad in (0.0, -1.0, np.inf, np.nan):
        assert omega_upper_bound(bad, NU, BETA1) == OMEGA_MAX_FLOOR


def test_cpu_single_wall_distance_sets_bound():
    """生产求解器：壁距经 `apply_wall_distance_source` 设定（平板算例的构造入口）后上界随
    最小壁距；换阶重查（`recompute_wall_distance_for_current_order`）后按新解点更新；
    粗壁面网格保持下限。"""
    from autoflowcfd.core.fr_solver.turbulence import recompute_wall_distance_for_current_order
    from tests.validation._flat_plate_case import build_flat_plate_solver

    solver, _ = build_flat_plate_solver(1, nx_up=2, nx_plate=3, ny=6, dy_wall=2e-5)
    d = solver.wall_distance
    d_min = float(d[d > 0].min())
    assert d_min < 1e-5
    assert solver.turb_model.omega_max == pytest.approx(_expected(d_min), rel=1e-12)

    solver.mesh.set_order(2)
    solver.state.U = np.zeros((solver.mesh.n_cells, solver.mesh.n_sps_per_cell, 5))
    assert recompute_wall_distance_for_current_order(solver)
    d2 = solver.wall_distance
    d2_min = float(d2[d2 > 0].min())
    assert d2_min < d_min
    assert solver.turb_model.omega_max == pytest.approx(_expected(d2_min), rel=1e-12)

    coarse, _ = build_flat_plate_solver(1, nx_up=2, nx_plate=3, ny=6, dy_wall=0.1)
    assert coarse.turb_model.omega_max == OMEGA_MAX_FLOOR


def test_single_gpu_wall_distance_sets_bound(monkeypatch):
    import autoflowcfd.core.gpu.solver.gpu_solver_init as init_mod
    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver
    from tests.unit._gpu_cupy_shim import patch_module_get_cupy
    from tests.validation._flat_plate_case import AnalyticWallDistance

    patch_module_get_cupy(monkeypatch, init_mod, SimpleNamespace(asarray=np.asarray))
    y = np.array([[1e-6, 5e-3], [2e-2, 3e-1]])
    sps = np.zeros(y.shape + (3,))
    sps[..., 0] = 0.5
    sps[..., 1] = y
    fake = SimpleNamespace(_wall_distance_source=AnalyticWallDistance(), turb_model_name="SST",
                           mesh=SimpleNamespace(sps_coords=sps, n_cells=2, n_sps_per_cell=2),
                           turb_model_gpu=_model(), mu_molecular=MU, freestream={"rho_inf": RHO})
    GPUFRSolver._init_wall_distance_gpu(fake)
    assert fake.turb_model_gpu.omega_max == pytest.approx(_expected(1e-6), rel=1e-12)


def test_distributed_bound_uses_global_min(monkeypatch):
    """两个 rank 本地最小壁距不同：经全局最小归约后上界一致，等于单机值。"""
    import autoflowcfd.core.mpi.comm as comm_mod
    from autoflowcfd.core.mpi.distributed_turbulence import apply_distributed_omega_bound

    local = {0: np.array([[3e-6, 1e-2]]), 1: np.array([[4e-4, 2e-1]])}
    global_min = min(float(v.min()) for v in local.values())
    monkeypatch.setattr(comm_mod, "allreduce_min", lambda v: min(v, global_min))
    solver = SimpleNamespace(mu_molecular=MU, freestream={"rho_inf": RHO})
    bounds = []
    for rank in (0, 1):
        model = _model()
        apply_distributed_omega_bound(model, local[rank], solver)
        bounds.append(model.omega_max)
    assert bounds[0] == bounds[1] == pytest.approx(_expected(global_min), rel=1e-12)
    # 没有 SST 类模型或没有壁距时不做事
    apply_distributed_omega_bound(None, local[0], solver)
    untouched = _model()
    apply_distributed_omega_bound(untouched, None, solver)
    assert untouched.omega_max == OMEGA_MAX_FLOOR
