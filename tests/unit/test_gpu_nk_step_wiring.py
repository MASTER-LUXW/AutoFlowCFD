# -*- coding: utf-8 -*-
"""单机 GPU 的 NEWTON_KRYLOV 分支接线（`gpu/solver/gpu_solver/step.py`）。

隐式步本身（`implicit/mean_flow_step.py`）CPU 与 GPU 共用；本文件验的是
GPU 这一侧的接线：残差拼装（含 GPU 版低马赫预处理从试探态自己推原始
变量）、只解 5 个平均流变量、块 Jacobi 的面相邻关系来源、跨步状态、SER
用 Newton 信息更新。

做法：真实的 `_GPUSolverStepMixin.step` 跑在替身上——`get_cupy` 换成
numpy，GPU 的残差与局部步长入口委托给同一个 CPU 求解器——与 CPU 的
`step()` 在同一状态、同一组步长上各走 3 步，状态、Newton 诊断与 CFL
序列必须一致。
"""

import types

import numpy as np
import pytest

from tests.unit._gpu_cupy_shim import patch_module_get_cupy
from tests.validation._channel_mesh import (
    build_channel_mesh_prism,
    build_face_exact_ghost_provider,
)

RHO, U_INF, P, GAMMA = 1.225, 30.0, 101325.0, 1.4
LX, H, LZ = 0.4, 0.1, 0.08


class _NumpyAsCupy:
    def __getattr__(self, name):
        return getattr(np, name)

    def asnumpy(self, x):
        return np.asarray(x)


def _laminar_channel():
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.time_integration import TimeIntegrationScheme

    mesh = build_channel_mesh_prism(1, nx=3, ny=4, nz=2, Lx=LX, H=H, Lz=LZ)
    bc = {n: {"type": "SYMMETRY"} for n in ("z_min", "z_max")}
    for n in ("wall_bottom", "wall_top"):
        bc[n] = {"type": "WALL", "is_no_slip": True, "wall_velocity": [0.0, 0.0, 0.0]}
    for n in ("x_min", "x_max"):
        bc[n] = {"type": "FARFIELD", "Q_free": [RHO, U_INF, 0.0, 0.0, P]}
    s = FRSolver(mesh=mesh, order=1, turb_model_name="NONE",
                 time_scheme=TimeIntegrationScheme.NEWTON_KRYLOV,
                 rho_inf=RHO, vel_inf=U_INF, p_inf=P, mu_molecular=1.8e-5, bc_overrides=bc)
    s.order_continuation_enabled = False
    s.boundary_ghost_provider = build_face_exact_ghost_provider(mesh, LX, H, LZ, bc)
    Ua = np.asarray(s.state.U).copy()
    Ua[..., 0] = RHO
    Ua[..., 1] = RHO * U_INF
    Ua[..., 2:4] = 0.0
    Ua[..., 4] = P / (GAMMA - 1.0) + 0.5 * RHO * U_INF ** 2
    s.state.U = np.ascontiguousarray(Ua)
    s.state._update_primitives()
    return s


@pytest.fixture
def patched(monkeypatch):
    import autoflowcfd.core.gpu.gpu_preconditioning as gpre
    import autoflowcfd.core.gpu.solver.gpu_solver as gsol
    import autoflowcfd.core.gpu.solver.gpu_solver_io as gio

    patch_module_get_cupy(monkeypatch, [gsol, gio, gpre], _NumpyAsCupy())


def _frozen_dt(cpu):
    """两侧共用的逐单元步长：CPU 自己算一次，取每单元第一个解点，之后固定。"""
    dt_l, dt_p = cpu._compute_local_time_step(return_physical_too=True)
    n_cells = cpu.state.U.shape[0]
    return (np.asarray(dt_l).reshape(n_cells, -1)[:, 0].copy(),
            np.asarray(dt_p).reshape(n_cells, -1)[:, 0].copy())


def _gpu_standin(cpu, dt_cell, dt_phys_cell):
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    from autoflowcfd.core.gpu.solver.gpu_solver_io import _GPUSolverIOMixin
    from autoflowcfd.core.time_integration.adaptive_cfl.policy import build_cfl_policy
    from autoflowcfd.core.time_integration.base import TimeIntegrationScheme as S

    def _with_state(fn):
        def run(U_trial=None, **kw):
            saved = cpu.state.U
            cpu.state.U = np.ascontiguousarray(U_trial)
            try:
                return np.asarray(fn(**kw))
            finally:
                cpu.state.U = saved
        return run

    g = types.SimpleNamespace(
        mesh=cpu.mesh, n_vars=cpu.state.n_vars, U_gpu=cpu.state.U.copy(),
        time_integrator=types.SimpleNamespace(scheme=S.NEWTON_KRYLOV),
        filter_func_gpu=None, low_mach_precond_enabled=cpu.low_mach_precond_enabled,
        freestream=cpu.freestream, flat_face_gpu=get_flat_face_geometry(cpu.mesh, cpu.ops),
        order=1, current_order=None, turb_model_gpu=None, sgs_model_gpu=None,
        turb_model_name="NONE", residual_history=[], iteration=0,
        _newton_forcing=None, _newton_last_info=None, _newton_dtau_scale=1.0,
        _newton_block_precond=None, _newton_turb_state=None,
    )
    g._cfl_controller, g.fixed_cfl_number = build_cfl_policy(S.NEWTON_KRYLOV)
    g._update_primitives_gpu = lambda: setattr(g, "Q_gpu", conserved_to_primitive(g.U_gpu[..., :5]))
    g.compute_inviscid_residual_gpu = lambda U_trial: _with_state(cpu.compute_inviscid_residual)(U_trial)
    g.compute_viscous_residual_gpu = (
        lambda U_trial, mu_t_field=None: _with_state(cpu.compute_viscous_residual)(U_trial))
    g._compute_local_time_step_gpu = lambda return_physical_too=False: (
        (dt_cell, dt_phys_cell) if return_physical_too else dt_cell)
    for name in ("compute_turbulence_source_gpu", "_apply_turbulence_corrections_gpu"):
        setattr(g, name, types.MethodType(getattr(_GPUSolverIOMixin, name), g))
    return g


def test_gpu_nk_step_matches_cpu_nk_step(patched):
    from autoflowcfd.core.gpu.solver.gpu_solver.step import _GPUSolverStepMixin

    cpu = _laminar_channel()
    dt_cell, dt_phys_cell = _frozen_dt(cpu)
    n_cells, n_sps = cpu.state.U.shape[:2]
    full = lambda a: np.broadcast_to(a[:, None], (n_cells, n_sps)).copy()  # noqa: E731
    cpu._compute_local_time_step = lambda return_physical_too=False: (
        (full(dt_cell), full(dt_phys_cell)) if return_physical_too else full(dt_cell))
    gpu = _gpu_standin(cpu, dt_cell, dt_phys_cell)

    for _ in range(3):
        cpu.step(1e-3)
        _GPUSolverStepMixin.step(gpu, 1e-3)
        ic, ig = cpu._newton_last_info, gpu._newton_last_info
        assert ig["gmres_iters"] == ic["gmres_iters"]
        assert ig["res_norm"] == pytest.approx(ic["res_norm"], rel=1e-10)
        assert gpu._cfl_controller.cfl_number == pytest.approx(cpu._cfl_controller.cfl_number, rel=1e-10)
        np.testing.assert_allclose(gpu.U_gpu, cpu.state.U, rtol=1e-10, atol=1e-8)
    assert gpu._newton_block_precond is not None and gpu._newton_block_precond.n_builds == 1
