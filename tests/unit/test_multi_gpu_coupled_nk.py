# -*- coding: utf-8 -*-
"""多 GPU 分布式的平均流 + k-omega 紧耦合 Newton 接线（`gpu/distributed/gpu_distributed_implicit.py::
MultiGpuCoupledBackend` 与 `MultiGpuTurbulenceBackend`）。

紧耦合步本身（`time_integration/implicit/coupled_step.py`）各后端共用；这里验多 GPU 这一侧：试探
状态写进 `U_gpu`、湍流求值走真实的 compact 视图件（`gpu_distributed_init/turb_source.py` 的
`_prepare_turbulence_view_distributed` / `_sync_turbulence_view` / `_evaluate_turbulence_rates_distributed`
/ `_finalize_turbulence_update_distributed` / `_write_back_turbulence_distributed`）、湍流解析块在
compact 视图上装配并按 `inv_perm` 取行、快照还原、步后收尾。

做法：单 rank、全棱柱槽道——compact 排列恰为恒等（第一条断言钉住这个前提），分区、`dist_flat_face`、
halo 交换与 compact 面空间的边界提供者借自同一网格上的 CPU 分布式求解器；`get_cupy` 换成 numpy；平均流
残差委托给单机 CPU 求解器。与单机 CPU 的紧耦合步对照：

1. 耦合残差（7 列）一致；
2. 各走 3 个耦合 Newton 步，GMRES 次数相同、平均流与湍流状态一致。

此前多 GPU + k-omega 的隐式路径没有任何测试：`MultiGpuTurbulenceBackend.block_assembler` 用到
`np.asarray` 而模块从未导入 numpy（2026-10-02 本文件首次覆盖时一并修复）。
"""

import types

import numpy as np
import pytest

from tests.unit._gpu_cupy_shim import patch_module_get_cupy
from tests.unit.test_distributed_turbulence_bc_parity import _pair
from tests.unit.test_gpu_solver_turbulence_source import _NumpyAsCupy, _prepare_mesh_ops_data

import autoflowcfd.core.gpu.distributed.gpu_distributed_implicit as mgi_mod
import autoflowcfd.core.gpu.gpu_preconditioning as gpu_pre_mod
import autoflowcfd.core.gpu.distributed.gpu_distributed_init.turb_source as ts_mod
import autoflowcfd.core.gpu.residual.gpu_flux as gpu_flux_mod
import autoflowcfd.core.gpu.residual.gpu_gradients as gpu_gradients_mod
import autoflowcfd.core.gpu.residual.gpu_volume_contract as gpu_volume_contract_mod
import autoflowcfd.core.gpu.turbulence.gpu_implicit_turbulence as gpu_implicit_turb_mod
import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst_mod
import autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst as gpu_turbulence_sst_mod


@pytest.fixture(autouse=True)
def _patch_get_cupy(monkeypatch):
    patch_module_get_cupy(monkeypatch, [
        mgi_mod, ts_mod, gpu_flux_mod, gpu_gradients_mod, gpu_volume_contract_mod, gst_mod,
        gpu_implicit_turb_mod, gpu_turbulence_sst_mod, gpu_pre_mod], _NumpyAsCupy())
    monkeypatch.setattr(gpu_turbulence_sst_mod, "gpu_available", True)


def _multi_gpu_standin(single, dist):
    from autoflowcfd.core.gpu.distributed.gpu_distributed.residual import _MultiGPUResidualMixin
    from autoflowcfd.core.gpu.distributed.gpu_distributed_init.turb_source import _GPUDistributedTurbSourceMixin
    from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import compute_turbulence_face_masks_gpu
    from autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst import GPUTurbulenceSST
    from autoflowcfd.core.time_integration.implicit.mean_flow_step import reset_newton_state
    from autoflowcfd.core.time_integration.positivity import get_positivity_limiter

    n, n_sps = single.turb_model.k_field.shape
    fc = dist.dist_flat_face
    assert np.array_equal(np.asarray(fc.perm)[:n], np.arange(n)), "compact 排列不是恒等，本测试的替身前提不成立"
    tc = single.turb_model
    tg = GPUTurbulenceSST(n, n_sps, device_id=0, k_inf=tc.k_inf, omega_inf=tc.omega_inf)
    for a in ("k_field", "omega_field", "nu_t"):
        setattr(tg, a, np.array(getattr(tc, a), copy=True))
    for a in ("k_max", "omega_max", "production_factor"):
        setattr(tg, a, getattr(tc, a))
    provider = dist.local_solver.boundary_ghost_provider
    wall, open_ = compute_turbulence_face_masks_gpu(
        types.SimpleNamespace(face_connectivity=types.SimpleNamespace(n_faces=fc.base_flat.n_faces)), provider)
    md = _prepare_mesh_ops_data(single.mesh, single.ops)
    identity = types.SimpleNamespace(exchange=lambda a: a)
    g = types.SimpleNamespace(
        partition=dist.partition, dist_flat_face=fc, mesh=single.mesh, ops=single.ops, mesh_data=md, ops_data=md,
        n_compact=n, device_id=0, rank=0, order=1, current_order=1,
        _perm_gpu=np.asarray(fc.perm), _inv_perm_gpu=np.asarray(fc.inv_perm),
        gpu_halo=types.SimpleNamespace(exchange=dist.halo_exchange.exchange), turb_halo_gpu=identity,
        des_length_scale_halo_gpu=identity, flat_face_gpu=fc.base_flat,
        wall_distance_gpu=np.asarray(dist.wall_distance_compact), _wall_mask_k_gpu=wall, _open_mask_gpu=open_,
        turb_model_gpu=tg, ddes_model_gpu=None, iddes_h_max_compact=None, iddes_h_wn_compact=None,
        turb_model_name="SST", mu_molecular=single.mu_molecular, freestream=dist.local_solver.freestream,
        boundary_ghost_provider=provider, low_mach_precond_enabled=single.low_mach_precond_enabled,
        wmles_model=None, U_gpu=np.array(single.state.U[..., :5], copy=True),
        _turb_ramp_step=single._turb_ramp_step, _turb_production_ramp_steps=single._turb_production_ramp_steps,
        _turb_production_ramp_complete=single._turb_production_ramp_complete,
        _block_jacobi_colors_local=None,
    )
    for name in ("_prepare_turbulence_view_distributed", "_sync_turbulence_view",
                 "_evaluate_turbulence_rates_distributed", "_finalize_turbulence_update_distributed",
                 "_write_back_turbulence_distributed"):
        setattr(g, name, types.MethodType(getattr(_GPUDistributedTurbSourceMixin, name), g))
    for name in ("_permute_to_compact", "_unpermute_from_compact"):
        setattr(g, name, types.MethodType(getattr(_MultiGPUResidualMixin, name), g))
    g._get_positivity_limiter_gpu = lambda: get_positivity_limiter(single, xp=np)

    def _compute_total_residual_gpu(mu_t_field=None, inviscid=True, viscous=True, nu_av_compact=None):
        saved = single.state.U
        U = np.array(saved, copy=True)
        U[..., :5] = g.U_gpu
        single.state.U = U
        single.state._update_primitives()
        try:
            res = single.compute_inviscid_residual() + single.compute_viscous_residual(mu_t_turb=mu_t_field)
        finally:
            single.state.U = saved
            single.state._update_primitives()
        return res[..., :5]
    g._compute_total_residual_gpu = _compute_total_residual_gpu
    reset_newton_state(g)
    return g


@pytest.fixture
def pair():
    from autoflowcfd.core.time_integration.base import TimeIntegrationScheme

    single, dist = _pair(TimeIntegrationScheme.NEWTON_KRYLOV, 1)
    # 非平凡湍流场（均匀来流下 k/omega 也均匀，耦合项与湍流输运都退化）
    rng = np.random.default_rng(5)
    single.turb_model.k_field = single.turb_model.k_field * (1 + 0.2 * rng.standard_normal(single.turb_model.k_field.shape))
    single.turb_model.omega_field = single.turb_model.omega_field * (
        1 + 0.2 * rng.standard_normal(single.turb_model.omega_field.shape))
    single.step(1e-6)                       # 一个耦合步，让平均流也非均匀
    # 两侧从同样干净的跨步状态出发（块缓存、forcing 历史、dtau 缩放）：预热步留下的状态会让
    # 单机复用过时的块，两侧的 Newton 步因此不可比
    from autoflowcfd.core.time_integration.implicit.mean_flow_step import reset_newton_state

    reset_newton_state(single)
    g = _multi_gpu_standin(single, dist)
    g._block_jacobi_colors_local = np.asarray(
        __import__("autoflowcfd.core.fr_solver.turbulence.implicit", fromlist=["x"]).single_machine_cell_colors(single))
    return single, g


def _cell_is_prism(single):
    return np.arange(single.mesh.n_cells) < int(single.mesh.n_prism_cells)


def test_coupled_residual_matches_single_machine(pair):
    from autoflowcfd.core.fr_solver.turbulence.implicit import CpuCoupledBackend
    from autoflowcfd.core.time_integration.implicit.coupled_step import CoupledResidual
    from autoflowcfd.fr.native_padding import real_row_mask

    single, g = pair
    bc = CpuCoupledBackend(single)
    bg = mgi_mod.MultiGpuCoupledBackend(g, _cell_is_prism(single), 1, None, None)
    real = real_row_mask(bc.cell_is_prism, bc.n_sps, bc.order)
    xc, xg = bc.state(), bg.state()
    np.testing.assert_array_equal(xc, xg)
    rng = np.random.default_rng(1)
    x = xc.copy()
    x[:, 0] *= 1.0 + 1e-3 * rng.standard_normal(x.shape[0])
    x[:, 5] *= 1.0 + 0.1 * rng.standard_normal(x.shape[0])
    x[:, 6] += 0.1 * rng.standard_normal(x.shape[0])
    rc = CoupledResidual(bc, real, xc)(x)
    rg = CoupledResidual(bg, real, xg)(x)
    scale = np.abs(rc).max(axis=0)
    assert np.all(np.abs(rg - rc).max(axis=0) <= 1e-9 * scale), np.abs(rg - rc).max(axis=0) / scale
    np.testing.assert_array_equal(g.U_gpu, single.state.U[..., :5])     # 试探求值后还原
    np.testing.assert_array_equal(g.turb_model_gpu.k_field, single.turb_model.k_field)


def test_multi_gpu_coupled_steps_match_single_machine(pair):
    """两侧残差只差浮点重结合（GPU 件与 CPU 件的求和顺序不同，实测 1e-14）。实测各步相对差：
    第 1 步平均流 1e-11、k/omega/nu_t 5e-10（接线正确的直接证据，下面单独按 1e-8 断言）；第 2 步
    放大到平均流 2e-8、湍流 7e-7，其后逐步衰减（第 5 步 3e-7）——每步 GMRES 只迭代 1 次，更新量
    直接继承右端项的差，是有界的浮点传播而不是累积的不一致。容差平均流 1e-7、湍流 2e-6；接线
    错误给出的是 O(1) 的差。"""
    from autoflowcfd.core.fr_solver.turbulence.implicit import CpuCoupledBackend
    from autoflowcfd.core.time_integration.implicit.coupled_step import step_coupled_newton

    single, g = pair
    dtau = np.asarray(single._compute_local_time_step()).reshape(-1)
    for k in range(3):
        ic = step_coupled_newton(single, CpuCoupledBackend(single), dtau)
        ig = step_coupled_newton(g, mgi_mod.MultiGpuCoupledBackend(g, _cell_is_prism(single), 1, None, None), dtau)
        assert ig["gmres_iters"] == ic["gmres_iters"], f"第 {k + 1} 步 GMRES 次数不同：{ig} vs {ic}"
        assert ig["res_norm"] == pytest.approx(ic["res_norm"], rel=1e-6)
        exp, got = single.state.U[..., :5], g.U_gpu
        scale = np.abs(exp).max(axis=(0, 1))
        scale[1:4] = np.abs(exp[..., 1:4]).max()          # 动量三分量共用动量模的尺度
        rel = (np.abs(got - exp) / scale).max()
        tol_mean, tol_turb = (1e-8, 1e-8) if k == 0 else (1e-7, 2e-6)
        assert rel <= tol_mean, f"第 {k + 1} 步平均流相对差 {rel:.3e}"
        for name in ("k_field", "omega_field", "nu_t"):
            a, b = np.asarray(getattr(g.turb_model_gpu, name)), np.asarray(getattr(single.turb_model, name))
            r = np.abs(a - b).max() / np.abs(b).max()
            assert r <= tol_turb, f"第 {k + 1} 步 {name} 相对差 {r:.2e}"
