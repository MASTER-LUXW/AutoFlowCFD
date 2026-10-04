# -*- coding: utf-8 -*-
"""单机 GPU 的平均流 + 湍流紧耦合 Newton 步接线（`gpu/turbulence/gpu_implicit_turbulence.py::
GpuCoupledBackend`、`gpu/solver/gpu_solver/step.py`）；SST（棱柱槽道）与 SA-neg（四面体槽道，含
壁面解点的强 Dirichlet）各验一遍。

紧耦合步本身（`time_integration/implicit/coupled_step.py`）CPU 与 GPU 共用；这里验 GPU 这一侧：
试探状态写进 `U_gpu` 并更新原始变量、湍流求值走 GPU 求值件（`GPUTurbulenceSST` / 数组模块为替身的
`SAModel`，与 `GPUFRSolver` 的 `_prepare/_evaluate/_finalize_turbulence_*_gpu`）、解析块 Jacobian 的
GPU 部件（`gpu_linearization_parts`）、快照还原、步后收尾。

做法：真实的 `_GPUSolverStepMixin.step` 跑在替身上——`get_cupy` 换成 numpy，平均流残差与
局部步长委托给同一个 CPU 求解器，湍流全部走 GPU 求值件——与 CPU 的 `step()` 在同一状态、
同一组步长上：

1. 耦合残差（平均流 5 列 + 湍流未知量列）两侧一致；
2. 各走 3 个耦合 Newton 步，GMRES 次数相同、平均流与湍流状态一致。
"""

import types

import numpy as np
import pytest

from tests.unit._gpu_cupy_shim import patch_module_get_cupy
from tests.unit.test_gpu_solver_turbulence_source import _NumpyAsCupy, _prepare_mesh_ops_data
from tests.unit.test_implicit_sst_nk import _channel_solver

import autoflowcfd.core.gpu.gpu_preconditioning as gpu_pre_mod
import autoflowcfd.core.gpu.residual.gpu_corrected_gradient as gpu_corrected_gradient_mod
import autoflowcfd.core.gpu.residual.gpu_gradients as gpu_gradients_mod
import autoflowcfd.core.gpu.residual.gpu_inviscid as gpu_inviscid_mod
import autoflowcfd.core.gpu.residual.gpu_volume_contract as gpu_volume_contract_mod
import autoflowcfd.core.gpu.solver.gpu_solver as gpu_solver_mod
import autoflowcfd.core.gpu.solver.gpu_solver_io as gpu_solver_io_mod
import autoflowcfd.core.gpu.turbulence.gpu_implicit_turbulence as gpu_implicit_turb_mod
import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst_mod
import autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst as gpu_turbulence_sst_mod


@pytest.fixture(autouse=True)
def _patch_get_cupy(monkeypatch):
    patch_module_get_cupy(monkeypatch, [
        gpu_solver_io_mod, gpu_solver_mod, gpu_gradients_mod, gpu_volume_contract_mod, gst_mod,
        gpu_turbulence_sst_mod, gpu_implicit_turb_mod, gpu_pre_mod, gpu_inviscid_mod,
        gpu_corrected_gradient_mod], _NumpyAsCupy())
    monkeypatch.setattr(gpu_turbulence_sst_mod, "gpu_available", True)


def _frozen_dt(cpu):
    dt_l, dt_p = cpu._compute_local_time_step(return_physical_too=True)
    n_cells = cpu.state.U.shape[0]
    return (np.asarray(dt_l).reshape(n_cells, -1)[:, 0].copy(),
            np.asarray(dt_p).reshape(n_cells, -1)[:, 0].copy())


def _gpu_turbulence_model(cpu):
    """与 CPU 模型同状态的 GPU 侧模型（SST 为 `GPUTurbulenceSST`，SA 为数组模块是替身的 `SAModel`）。"""
    from autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst import GPUTurbulenceSST
    from autoflowcfd.core.turbulence.sa import SAModel

    tc = cpu.turb_model
    n_cells, n_sps = tc.transported_fields()[0].shape
    if isinstance(tc, SAModel):
        tg = SAModel(n_cells, n_sps, tc.nu_ref, tc.viscosity_ratio, xp=_NumpyAsCupy())
        attrs = ("nu_tilde_field", "nu_t", "wall_points", "production_factor")
    else:
        tg = GPUTurbulenceSST(n_cells, n_sps, device_id=0)
        attrs = ("k_field", "omega_field", "nu_t", "k_inf", "omega_inf", "k_max", "omega_max", "production_factor")
    for a in attrs:
        v = getattr(tc, a)
        setattr(tg, a, np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
    return tg


def _gpu_standin(cpu, dt_cell, dt_phys_cell):
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    from autoflowcfd.core.gpu.solver.gpu_solver_io import _GPUSolverIOMixin
    from autoflowcfd.core.time_integration.adaptive_cfl.policy import build_cfl_policy
    from autoflowcfd.core.time_integration.base import TimeIntegrationScheme as S
    from autoflowcfd.core.turbulence.transport.omega_wall import (
        _compute_open_boundary_face_mask, _compute_wall_dirichlet_face_mask,
    )

    tg = _gpu_turbulence_model(cpu)

    def _with_state(fn, U_trial, **kw):
        saved = cpu.state.U
        cpu.state.U = np.ascontiguousarray(U_trial)
        cpu.state._update_primitives()
        try:
            return np.asarray(fn(**kw)).copy()
        finally:
            cpu.state.U = saved
            cpu.state._update_primitives()

    flat = get_flat_face_geometry(cpu.mesh, cpu.ops)
    md = _prepare_mesh_ops_data(cpu.mesh, cpu.ops)
    g = types.SimpleNamespace(
        mesh=cpu.mesh, ops=cpu.ops, mesh_data=md, ops_data=md, n_vars=cpu.state.n_vars, U_gpu=cpu.state.U.copy(),
        device_id=0,
        time_integrator=types.SimpleNamespace(scheme=S.NEWTON_KRYLOV),
        filter_func_gpu=None, low_mach_precond_enabled=cpu.low_mach_precond_enabled,
        freestream=cpu.freestream, flat_face_gpu=flat, order=1, current_order=None,
        turb_model_gpu=tg, sgs_model_gpu=None, ddes_model_gpu=None, turb_model_name=cpu.turb_model_name,
        residual_history=[], iteration=0,
        _newton_forcing=None, _newton_last_info=None, _newton_dtau_scale=1.0, _newton_local_dtau=None,
        _newton_block_precond=None, _newton_turb_state=None,
        boundary_ghost_provider=cpu.boundary_ghost_provider, mu_molecular=cpu.mu_molecular,
        wmles_model=None, artificial_viscosity_enabled=False,
        wall_distance_gpu=np.asarray(cpu.wall_distance), _iddes_h_max_gpu=None, _iddes_h_wn_gpu=None,
        _wall_mask_k_gpu=_compute_wall_dirichlet_face_mask(cpu),
        _open_mask_gpu=_compute_open_boundary_face_mask(cpu, flat),
        _turb_ramp_step=cpu._turb_ramp_step, _turb_production_ramp_steps=cpu._turb_production_ramp_steps,
        _turb_production_ramp_complete=cpu._turb_production_ramp_complete,
    )
    g._cfl_controller, g.fixed_cfl_number = build_cfl_policy(S.NEWTON_KRYLOV)
    g._update_primitives_gpu = lambda: setattr(g, "Q_gpu", conserved_to_primitive(g.U_gpu[..., :5]))
    g._update_primitives_gpu()
    g.compute_inviscid_residual_gpu = lambda U_trial: _with_state(cpu.compute_inviscid_residual, U_trial)
    g.compute_viscous_residual_gpu = lambda U_trial, mu_t_field=None, nu_av=None: _with_state(
        cpu.compute_viscous_residual, U_trial, mu_t_turb=mu_t_field, nu_av=nu_av)
    g.compute_artificial_diffusivity_field_gpu = lambda: None   # 人工粘性未启用
    g._compute_local_time_step_gpu = lambda return_physical_too=False, nu_av=None: (
        (dt_cell, dt_phys_cell) if return_physical_too else dt_cell)
    for name in ("compute_turbulence_source_gpu", "_apply_turbulence_corrections_gpu", "_update_production_ramp_gpu",
                 "_prepare_turbulence_inputs_gpu", "_turbulence_velocity_gradient_gpu", "_evaluate_turbulence_rates_gpu",
                 "_finalize_turbulence_update_gpu", "_turbulent_mu_t_gpu"):
        setattr(g, name, types.MethodType(getattr(_GPUSolverIOMixin, name), g))
    return g


@pytest.fixture(params=["SST", "SA"])
def pair(request):
    from autoflowcfd.core.time_integration import TimeIntegrationScheme
    from tests.unit.test_sa_neg_solver import _sa_channel

    cpu = (_channel_solver(TimeIntegrationScheme.NEWTON_KRYLOV) if request.param == "SST"
           else _sa_channel("tet", 1))
    dt_cell, dt_phys_cell = _frozen_dt(cpu)
    n_cells, n_sps = cpu.state.U.shape[:2]
    full = lambda a: np.broadcast_to(a[:, None], (n_cells, n_sps)).copy()  # noqa: E731
    cpu._compute_local_time_step = lambda return_physical_too=False: (
        (full(dt_cell), full(dt_phys_cell)) if return_physical_too else full(dt_cell))
    return cpu, _gpu_standin(cpu, dt_cell, dt_phys_cell)


def test_coupled_residual_matches_cpu(pair):
    from autoflowcfd.core.fr_solver.turbulence.implicit import CpuCoupledBackend
    from autoflowcfd.core.gpu.turbulence.gpu_implicit_turbulence import GpuCoupledBackend
    from autoflowcfd.core.time_integration.implicit.coupled_step import CoupledResidual
    from autoflowcfd.fr.native_padding import real_row_mask

    cpu, gpu = pair
    bc, bg = CpuCoupledBackend(cpu), GpuCoupledBackend(gpu)
    real = real_row_mask(bc.cell_is_prism, bc.n_sps, bc.order)
    xc, xg = bc.state(), bg.state()
    np.testing.assert_array_equal(xc, xg)
    rng = np.random.default_rng(1)
    x = xc.copy()
    x[:, 0] *= 1.0 + 1e-3 * rng.standard_normal(x.shape[0])
    x[:, 5] *= 1.0 + 0.1 * rng.standard_normal(x.shape[0])
    if x.shape[1] > 6:
        x[:, 6] += 0.1 * rng.standard_normal(x.shape[0])
    rc = CoupledResidual(bc, real, xc)(x)
    rg = CoupledResidual(bg, real, xg)(x)
    scale = np.abs(rc).max(axis=0)
    assert np.all(np.abs(rg - rc).max(axis=0) <= 1e-9 * scale), np.abs(rg - rc).max(axis=0) / scale
    np.testing.assert_array_equal(gpu.U_gpu, cpu.state.U)               # 试探求值后两侧都还原
    for fg, fc in zip(gpu.turb_model_gpu.transported_fields(), cpu.turb_model.transported_fields()):
        np.testing.assert_array_equal(fg, fc)


def test_gpu_coupled_step_matches_cpu(pair):
    from autoflowcfd.core.gpu.solver.gpu_solver.step import _GPUSolverStepMixin

    cpu, gpu = pair
    for _ in range(3):
        cpu.step(1e-3)
        _GPUSolverStepMixin.step(gpu, 1e-3)
        ic, ig = cpu._newton_last_info, gpu._newton_last_info
        assert ig["gmres_iters"] == ic["gmres_iters"]
        # 范数与湍流场同一个舍入敏感度下限（见下面湍流场比较处的注释）
        assert ig["res_norm"] == pytest.approx(ic["res_norm"], rel=1e-6)
        assert ig["res_norm_turbulence"] == pytest.approx(ic["res_norm_turbulence"], rel=1e-6)
        assert gpu._cfl_controller.cfl_number == pytest.approx(cpu._cfl_controller.cfl_number, rel=1e-9)
        np.testing.assert_allclose(gpu.U_gpu, cpu.state.U, rtol=1e-8, atol=1e-7)
        # 湍流场按场最大值归一化比较：块预处理以 float32 存储、每步 GMRES 只迭代 1 次，更新量带着
        # float32 舍入，两侧输入差 1e-13 就会在个别解点上翻转。CPU 孪生对照（同一算例、初值加 1e-13
        # 相对扰动，SA 四面体槽道）实测第 1 步起差 7.6e-8、3 步内在 1e-7~3e-7 之间波动、不增长——
        # 这是算法对舍入的敏感度下限，不是两侧实现的差异（第 0 步两侧一致到 3.8e-13）。
        for fg, fc in zip(gpu.turb_model_gpu.transported_fields(), cpu.turb_model.transported_fields()):
            assert np.abs(fg - fc).max() <= 1e-6 * np.abs(fc).max(), np.abs(fg - fc).max() / np.abs(fc).max()
    assert gpu._newton_turb_state is not None and gpu._newton_block_precond is not None


def test_explicit_turbulence_update_matches_cpu(pair):
    """显式路径的湍流段（产生项斜坡 -> 速率 -> 模型点隐式更新 -> 未知量空间滤波收尾 -> 涡粘）：
    GPU `compute_turbulence_source_gpu` 与 CPU `compute_turbulence_source` 在同一状态、同一逐单元步长
    上逐步一致（没有 Newton/GMRES，差异只有两侧残差核的求和顺序）。"""
    cpu, gpu = pair
    dt_phys = np.asarray(cpu._compute_local_time_step(return_physical_too=True)[1])
    for _ in range(3):
        cpu.compute_turbulence_source(dt_phys)
        gpu.compute_turbulence_source_gpu(dt_phys[:, :1])
        pairs = list(zip(gpu.turb_model_gpu.transported_fields(), cpu.turb_model.transported_fields()))
        pairs.append((gpu.turb_model_gpu.nu_t, cpu.turb_model.nu_t))
        for fg, fc in pairs:
            assert np.abs(fg - fc).max() <= 1e-10 * np.abs(fc).max(), np.abs(fg - fc).max() / np.abs(fc).max()
