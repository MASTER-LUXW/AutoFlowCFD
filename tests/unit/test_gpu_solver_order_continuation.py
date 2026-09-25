"""单 GPU（`GPUFRSolver`）Order Continuation 验证（2026-09-02）。

背景：三条分布式路径（CPU MPI"传统模式"/CPU"完全分布式加载"/多GPU
分布式）之外，单 GPU（`--backend gpu` 不加 `--multi-gpu`）此前是唯一
仍然完全没有 Order Continuation 机制的后端——`GPUFRSolver` 是单机
（非分布式）求解器，架构上比三条分布式路径都更接近单机 CPU
`FRSolver`（`self.mesh` 全程是完整真实网格，没有 partition/halo/
compact 索引空间这层复杂度），本次补齐，见 core/gpu/solver/
gpu_solver_order_continuation.py 模块文档。

验证方式（用 numpy-as-cupy 替身完整构造真实 `GPUFRSolver` 并调用
`_interpolate_to_new_order`，比此前任何单 GPU 测试都走得更远）：
过程中真实发现并修复了 `gpu_inviscid.py::_native_self_extrap`/
`_native_or_collapsed_contrib` 的一个独立真实 bug——`tet_basis_mode=
"collapsed"`（CLI 默认值）时 `boundary_extrap_native`/`lift_native`
是空数组，`cp.clip(x, 0, -1)` 病态区间 + `cp.where` 两个分支被提前
无条件求值，导致对空数组的越界 gather，任何单 GPU + collapsed 模式 +
含四面体网格 + 阶数>=1 的组合在第一次残差求值时都会崩溃，与 Order
Continuation 本身无关。已修复（短路：完全没有 native 单元时直接返回
collapsed 分支）。

同时如实记录一个发现但本次未修复的独立真实 bug：修复上面这处之后，
本文件测试网格（含多源/混合拆分面的真实合成网格）在真正调用
`solver.step()`（不是 `_interpolate_to_new_order` 本身）时，会在
`_ausm_up_flux_batch_gpu`/`distribute_face_correction_to_sps` 里遇到
另一处法向量/修正函数导数的形状广播不匹配（`(4,9) vs (4,9,1)`/
`(4,8,1) vs (4,8,4,5)` 这类），在 P1 和 P2 直接构造（不经过 Order
Continuation）时同样复现——这是单 GPU 无粘残差 kernel 对这类真实
多源混合面网格拓扑的一个更深、独立的既有缺口（此前从未有任何测试
用完整构造+真正 step() 走到这一步，因为既有 native/collapsed
crosscheck 测试恒用 `tet_basis_mode="native"`），修复它需要审计
`gpu_inviscid.py`/`gpu_inviscid_volume.py` 全部涉及法向量/修正函数
形状的调用约定，是一个独立于本次 Order Continuation 任务、工作量
相当的单独课题，本次不在范围内解决。因此本文件的决定性测试止步于
`_interpolate_to_new_order` 本身（阶数切换的几何/数组重建逻辑，即
本次改动实际新增的代码），不进一步调用 `solver.step()`。
"""

import numpy as np
from tests.unit._wall_source import synthetic_wall_source
import pytest

from tests.unit._patch_pkg import patch_pkg_attr
from tests.unit._gpu_cupy_shim import patch_module_get_cupy


class _NumpyAsCupy:
    def __getattr__(self, name):
        return getattr(np, name)

    def scatter_add(self, a, indices, b):
        np.add.at(a, indices, b)

    def asnumpy(self, x):
        return np.asarray(x)

    class cuda:
        class Device:
            def __init__(self, device_id=0):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class runtime:
            @staticmethod
            def getDeviceCount():
                return 1

            @staticmethod
            def getDeviceProperties(device_id):
                return {'name': b'FakeGPU', 'totalGlobalMem': 8 * 1024 ** 3}

        class Stream:
            def __init__(self, non_blocking=True):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def synchronize(self):
                pass

        @staticmethod
        def get_default_memory_pool():
            class _Pool:
                def used_bytes(self):
                    return 0

                def free_all_blocks(self):
                    pass
            return _Pool()


@pytest.fixture(autouse=True)
def _patch_gpu_modules(monkeypatch):
    shim = _NumpyAsCupy()

    import autoflowcfd.core.gpu as core_gpu_mod
    import autoflowcfd.core.gpu.solver.gpu_solver as gs_mod
    import autoflowcfd.core.gpu.solver.gpu_solver_init as gsi_mod
    import autoflowcfd.core.gpu.solver.gpu_solver_io as gsio_mod
    import autoflowcfd.core.gpu.solver.gpu_solver_order_continuation as gsoc_mod
    import autoflowcfd.core.gpu.residual.gpu_gradients as gpu_gradients_mod
    import autoflowcfd.core.gpu.residual.gpu_volume_contract as gpu_volume_contract_mod
    import autoflowcfd.core.gpu.residual.gpu_flux as gpu_flux_mod
    import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst_mod
    import autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst as gpu_turbulence_sst_mod
    import autoflowcfd.core.gpu.array_manager as array_mgr_mod
    import autoflowcfd.core.gpu.gpu_time_integration as gti_mod
    import autoflowcfd.core.gpu.gpu_face_geometry as gfg_mod
    import autoflowcfd.core.gpu.residual.gpu_inviscid as gpu_inviscid_mod
    import autoflowcfd.core.gpu.residual.gpu_viscous as gpu_viscous_mod
    import autoflowcfd.core.gpu.gpu_modal_filter as gmf_mod

    mods = [
        gs_mod, gsi_mod, gsio_mod, gsoc_mod, gpu_gradients_mod, gpu_volume_contract_mod,
        gpu_flux_mod, gst_mod, gpu_turbulence_sst_mod, array_mgr_mod, gti_mod, gfg_mod,
        gpu_inviscid_mod, gpu_viscous_mod, gmf_mod,
    ]
    # get_cupy 一次性整批替换（含各包的子模块）；断言是聚合的，所以
    # `mods` 里含 gpu_inviscid_volume 这类本就没有 get_cupy 的模块无妨。
    patch_module_get_cupy(monkeypatch, [core_gpu_mod] + mods, shim)
    # `gpu_available` 是各类 __init__ 单独检查的模块级标志，与 get_cupy
    # 无关，仍需逐模块 patch。
    patch_pkg_attr(monkeypatch, core_gpu_mod, "gpu_available", True)
    for m in mods:
        if hasattr(m, "gpu_available"):
            monkeypatch.setattr(m, "gpu_available", True)


def _make_solver(order, turb_model="none", turbulence_intensity=0.01, viscosity_ratio=5.0):
    from autoflowcfd.fr.operators import generate_fr_operators
    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver
    from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    return GPUFRSolver(
        mesh=mesh, ops=ops, order=order, device_id=0,
        mu_molecular=1.8e-5, rho_inf=1.225, vel_inf=33.33, p_inf=101325.0,
        turb_model=turb_model,
        turbulence_intensity=turbulence_intensity, viscosity_ratio=viscosity_ratio,
        wall_distance_source=synthetic_wall_source(mesh),
    )


class TestGpuSolverOrderContinuationDispatch:
    def test_construction_sets_current_order_and_flag(self):
        solver = _make_solver(2)
        assert solver.order == 2
        assert solver.current_order == 2
        assert solver.order_continuation_enabled is True

    def test_order_1_target_uses_order_continuation(self):
        """目标 P1 也从 P0 起步（2026-09-25，A/B 数据见
        core/utils/order_continuation/policy.py）；显式关闭时直接迭代。判据只有
        一份，GPU `solve()` 调用的就是它。"""
        from autoflowcfd.core.utils.order_continuation.policy import uses_order_continuation

        solver = _make_solver(1)
        assert uses_order_continuation(solver)
        solver.order_continuation_enabled = False
        assert not uses_order_continuation(solver)


class TestGpuSolverInterpolateToNewOrder:
    """决定性验证 `_interpolate_to_new_order` 本身（阶数切换的几何/
    GPU 常驻数组重建逻辑，即本次改动实际新增的代码）——不调用
    `solver.step()`（既有的、独立于本次改动的 GPU 无粘 kernel 形状
    缺口会在那一步崩溃，见模块文档"验证方式"一节）。"""

    def test_p1_to_p2_upgrade_interpolates_uniform_field_exactly(self):
        solver = _make_solver(1, turb_model="sst")
        assert solver.current_order == 1
        assert solver.mesh.n_sps_per_cell == 8

        rho_inf, vel_inf, p_inf = 1.225, 33.33, 101325.0
        gamma = 1.4
        e = p_inf / ((gamma - 1.0) * rho_inf) + 0.5 * vel_inf ** 2

        solver._interpolate_to_new_order(2)

        assert solver.current_order == 2
        assert solver.mesh.n_sps_per_cell == 27
        assert solver.U_gpu.shape == (solver.mesh.n_cells, 27, 5)
        # 均匀自由流场：升阶插值必须精确保持（不是近似）。
        np.testing.assert_allclose(solver.U_gpu[:, :, 0], rho_inf)
        np.testing.assert_allclose(solver.U_gpu[:, :, 1], rho_inf * vel_inf)
        np.testing.assert_allclose(solver.U_gpu[:, :, 4], rho_inf * e)
        assert np.all(np.isfinite(solver.U_gpu))

        # 湍流场同样精确保持均匀初值，形状正确随阶数切换。
        assert solver.turb_model_gpu.k_field.shape == (solver.mesh.n_cells, 27)
        assert solver.turb_model_gpu.omega_field.shape == (solver.mesh.n_cells, 27)
        assert np.all(np.isfinite(solver.turb_model_gpu.k_field))
        assert np.all(np.isfinite(solver.turb_model_gpu.omega_field))

        # boundary_ghost_provider/mesh_data/ops 都必须随阶数重建
        # （新阶数 SEM/FP 几何形状），不是残留旧阶数的对象。
        assert solver.ops.D_3d.shape[0] == 27
        assert solver.mesh_data['det_jacs'].shape[1] == 27

    def test_reset_to_p0_when_downgrading(self):
        """降阶（reset 场景，供 run_distributed_order_continuation 在
        非 resume 场景下从目标阶数重置回 P0 时使用）：不做插值，直接
        重置为均匀自由流场。"""
        solver = _make_solver(2, turb_model="none")
        assert solver.current_order == 2

        rho_inf, vel_inf, p_inf = 1.225, 33.33, 101325.0

        solver._interpolate_to_new_order(0)

        assert solver.current_order == 0
        assert solver.mesh.n_sps_per_cell == 1
        assert solver.U_gpu.shape == (solver.mesh.n_cells, 1, 5)
        np.testing.assert_allclose(solver.U_gpu[:, :, 0], rho_inf)
        np.testing.assert_allclose(solver.U_gpu[:, :, 1], rho_inf * vel_inf)

    def test_same_order_is_noop(self):
        solver = _make_solver(2)
        mesh_data_before = solver.mesh_data
        solver._interpolate_to_new_order(2)
        assert solver.current_order == 2
        # 阶数相同时应该直接返回，不重建（对象引用不变）。
        assert solver.mesh_data is mesh_data_before


class TestGpuSolverStoresTurbulenceIntensityAndViscosityRatio:
    """真实 bug 回归测试（2026-09-05，代码复审发现）：`GPUFRSolver.
    __init__` 此前接收 `turbulence_intensity`/`viscosity_ratio` 构造
    参数、只用它们算一次初始 k_inf/omega_inf，却从不存成
    `self._turbulence_intensity`/`self._viscosity_ratio`——CPU 版
    `DistributedFRSolver`/`MultiGPUDistributedSolver` 都会存这两个
    属性（见 `core/fr_solver/turbulence.py::_set_freestream_turbulence`
    文档：`getattr(solver, '_turbulence_intensity', 0.01)`）。任何后续
    需要重新推导来流湍流值的调用点（`_interpolate_to_new_order` 降阶
    重置分支、`distributed_order_continuation.py::_reset_turbulence_
    if_resumed_field_exploded` resume 安全重置）在此前的代码上都会
    因为取不到真实属性而静默退回默认值 Tu=0.01/VR=5.0，用非默认湍流
    强度/粘性比构造的算例会被静默重置成错误的来流值——本类验证属性
    真的被存了下来、且 `_set_freestream_turbulence` 用的确实是这份
    真实值而不是默认值。"""

    def test_non_default_values_are_stored_and_used(self):
        from autoflowcfd.core.fr_solver.turbulence import _set_freestream_turbulence

        solver = _make_solver(1, turb_model="sst", turbulence_intensity=0.05, viscosity_ratio=8.0)

        assert solver._turbulence_intensity == 0.05
        assert solver._viscosity_ratio == 8.0

        k_inf, omega_inf = _set_freestream_turbulence(solver)
        vel_inf, rho_inf, mu = 33.33, 1.225, 1.8e-5
        nu = mu / rho_inf
        k_inf_expected = 1.5 * (vel_inf * 0.05) ** 2
        omega_inf_expected = k_inf_expected / (8.0 * nu)
        np.testing.assert_allclose(k_inf, k_inf_expected, rtol=1e-10)
        np.testing.assert_allclose(omega_inf, omega_inf_expected, rtol=1e-10)
        # 用默认值算出的结果必须明显不同——否则这个测试对"是否真的用了
        # 非默认配置"没有区分度。
        k_inf_default = 1.5 * (vel_inf * 0.01) ** 2
        assert not np.isclose(k_inf, k_inf_default)
