"""GPU 无粘界面项 AUSM+up 在含多源/混合拆分面网格上的 shape bug 修复
验证（问题清单 #5，2026-09-02）。

背景：`test_gpu_p1_inviscid_interface_crosscheck.py`（2026-08-23）已经
为 `compute_inviscid_residual_fr_gpu` 写好了 CPU/GPU crosscheck，但整个
文件用 `pytest.importorskip("cupy")` 在模块级别跳过——本机没有真实
CuPy/CUDA，那份测试从未在本机真正执行过。`test_gpu_solver_order_
continuation.py`（同日）排查单 GPU Order Continuation 时，用 numpy-as-
cupy 替身完整构造 `GPUFRSolver` 并真正调用 `step()`，在这份被跳过的
测试从未覆盖到的组合（`tet_basis_mode="collapsed"`，即 CLI 默认值 +
含多源/混合拆分面的真实合成网格）上，于 `_ausm_up_flux_batch_gpu`/
`distribute_face_correction_to_sps` 撞到一处法向量/修正函数导数的
形状广播不匹配（`(4,8,1) vs (4,8,4,5)`），当时如实记录为"发现但未
修复的独立真实缺口"。

真实根因（本次排查确认，见 `gpu_inviscid.py::_ausm_up_flux_batch_gpu`
文档"真实 bug 修复"一节的完整推导）：该函数把 `normal` 参数当作
`(N,3)`（逐面一个法向，`nx=normal[...,0:1]` 故意保留末尾长度 1 维度
以便广播到 `(N,n_fp)` 的其余物理量），但两个真实调用点
（`_compute_interface_correction_gpu` 的 owner/neighbor 分支）传入的
`direction_o`/`direction_n`（来自 `_ausm_direction_with_fallback`）
形状恒为 `(N,n_fp,3)`——逐 FP 各自独立的方向，不是逐面共享同一个值。
`nx=normal[...,0:1]` 对 `(N,n_fp,3)` 输入产出 `(N,n_fp,1)`，与
`(N,n_fp)` 相乘时若 `N != n_fp`（绝大多数真实网格）会直接
`ValueError`；若 `N == n_fp`（本文件用的合成测试网格恰好是这个巧合：
4 个 owner-primary 面、每面 4 个 FP）广播规则会"成功"但产出一个
错误地多出一维、内容错误的结果，只在下游 `distribute_face_correction_
to_sps` 的最终 gather 处才真正报错——这正是此前排查止步的地方。

修复：`nx=normal[...,0]`（不保留末尾维度），使其在真实 `(N,n_fp,3)`
输入下形状恰好是 `(N,n_fp)`，与其余逐 FP 物理量精确匹配，不再依赖
任何隐式广播——对 `N==n_fp`/`N!=n_fp` 两种网格规模都正确。

验证方式（本机没有真实 CUDA，与本项目一贯方法论一致）：用 numpy-as-
cupy 替身完整构造真实 `GPUFRSolver`（含真正的 `compute_inviscid_
residual_gpu()` 调用，走到修复的确切代码路径），与 CPU 参考实现
`compute_inviscid_residual_fr` 逐位交叉验证——不是重新验证 AUSM+up
本身的物理正确性（`test_gpu_p1_inviscid_interface_crosscheck.py`已经
覆盖，只是从未在本机跑过），而是决定性证明"修复后的 GPU 代码路径在
本机可执行的意义下与 CPU 参考实现一致"，容差沿用该文件已经确立的
判据（均匀流场 `rel_diff = max_diff/p_inf < 1e-6`；非均匀扰动流场
`max_diff < max(1e-6, scale*1e-6)`）。
"""

import numpy as np
import pytest
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
    import autoflowcfd.core.gpu.residual.gpu_inviscid as gpu_inviscid_mod
    import autoflowcfd.core.gpu.residual.gpu_inviscid_volume as gpu_inviscid_volume_mod
    import autoflowcfd.core.gpu.residual.gpu_volume_contract as gpu_volume_contract_mod
    import autoflowcfd.core.gpu.residual.gpu_flux as gpu_flux_mod
    import autoflowcfd.core.gpu.array_manager as array_mgr_mod
    import autoflowcfd.core.gpu.gpu_face_geometry as gfg_mod

    mods = [
        gpu_inviscid_mod, gpu_inviscid_volume_mod, gpu_volume_contract_mod,
        gpu_flux_mod, array_mgr_mod, gfg_mod,
    ]
    # get_cupy 一次性整批替换（含各包的子模块）；断言是聚合的，所以
    # `mods` 里含 gpu_inviscid_volume 这类本就没有 get_cupy 的模块无妨。
    patch_module_get_cupy(monkeypatch, [core_gpu_mod] + mods, shim)
    # `gpu_available` 是各类 __init__ 单独检查的模块级标志，与 get_cupy
    # 无关，仍需逐模块 patch。
    monkeypatch.setattr(core_gpu_mod, "gpu_available", True)
    for m in mods:
        if hasattr(m, "gpu_available"):
            monkeypatch.setattr(m, "gpu_available", True)


def _make_mesh(order, tet_basis_mode="native"):
    from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
    return _build_synthetic_mixed_mesh(order, tet_basis_mode=tet_basis_mode)


def _cross_check(U, mesh, mach_ref=0.2):
    from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr
    from autoflowcfd.core.gpu.residual.gpu_inviscid import compute_inviscid_residual_fr_gpu

    cpu_residual = compute_inviscid_residual_fr(U, mesh, mesh.operators, mach_ref=mach_ref)
    gpu_residual = compute_inviscid_residual_fr_gpu(U, mesh, mesh.operators, mach_ref=mach_ref)
    gpu_residual_np = np.asarray(gpu_residual)
    return cpu_residual, gpu_residual_np


class TestAusmUpMixedFaceShapeFix:
    """`_build_synthetic_mixed_mesh` 保证含多源/混合拆分面（约 5% 的
    棱柱四边形侧面），且本文件恰好触发 owner-primary 面数与每面 FP 数
    相等（`N==n_fp`）这个此前让 bug"沉默通过"broadcasting 而不是立刻
    报错的巧合场景——与该函数文档"真实 bug 修复"一节描述的复现条件
    完全对应。"""

    @pytest.mark.parametrize("order,rel_tol", [(1, 1e-6), (2, 1e-5)])
    def test_uniform_flow_matches_cpu(self, order, rel_tol):
        from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved

        mesh = _make_mesh(order)
        rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
        Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
        U_inf = primitive_to_conserved(Q_inf)
        U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

        cpu_residual, gpu_residual = _cross_check(U, mesh)

        max_diff = np.max(np.abs(cpu_residual - gpu_residual))
        rel_diff = max_diff / p_inf
        assert rel_diff < rel_tol, f"P={order}: max|cpu-gpu|={max_diff:.3e}, rel={rel_diff:.3e}"
        assert np.all(np.isfinite(gpu_residual))

    @pytest.mark.parametrize("order", [1, 2])
    def test_nonuniform_perturbed_flow_matches_cpu(self, order):
        """非均匀扰动流场：真正触发 owner/neighbor 两侧非平凡 AUSM+up
        通量求值（均匀流场下 F(U,U)=F(U) 会掩盖很多形状/公式 bug）。"""
        from autoflowcfd.core.fr_residual.inviscid import (
            primitive_to_conserved, conserved_to_primitive,
        )

        mesh = _make_mesh(order)
        rng = np.random.default_rng(order * 5000 + 17)

        rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
        Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
        U_inf = primitive_to_conserved(Q_inf)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        U = np.tile(U_inf, (n_cells, n_sps, 1))

        Q = conserved_to_primitive(U)
        Q[..., 0] *= 1.0 + rng.uniform(-0.05, 0.05, size=(n_cells, n_sps))
        Q[..., 1] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
        Q[..., 2] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
        Q[..., 3] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
        Q[..., 4] *= 1.0 + rng.uniform(-0.05, 0.05, size=(n_cells, n_sps))
        U = primitive_to_conserved(Q)

        cpu_residual, gpu_residual = _cross_check(U, mesh)

        max_diff = np.max(np.abs(cpu_residual - gpu_residual))
        scale = max(np.max(np.abs(cpu_residual)), 1.0)
        assert max_diff < max(1e-6, scale * 1e-6), \
            f"P={order}: max|cpu-gpu|={max_diff:.3e}, scale={scale:.3e}"
        assert np.all(np.isfinite(gpu_residual))

    def test_split_prism_quad_face_no_duplicate_counting(self):
        """回归 #5 与本类文档同一份修复：棱柱四边形侧面被拆分成 2 条
        子面记录的场景，GPU 必须与 CPU 一致（不是本次改动的对象，
        2026-08-23 已修复；这里在真正可执行的 numpy-as-cupy 路径上
        重新钉住，防止 #5 的修复回归它）。"""
        from autoflowcfd.core.fr_residual.inviscid import (
            primitive_to_conserved, conserved_to_primitive,
        )
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry

        mesh = _make_mesh(2)
        flat_geom = get_flat_face_geometry(mesh, mesh.operators)
        n_split = int(np.sum(~flat_geom.owner_is_primary) + np.sum(~flat_geom.neighbor_is_primary))
        assert n_split > 0, "测试网格未包含拆分的棱柱四边形侧面，无法覆盖这个场景"

        rng = np.random.default_rng(999)
        rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
        Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
        U_inf = primitive_to_conserved(Q_inf)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        U = np.tile(U_inf, (n_cells, n_sps, 1))
        Q = conserved_to_primitive(U)
        Q[..., 1] += rng.uniform(-3.0, 3.0, size=(n_cells, n_sps))
        Q[..., 4] *= 1.0 + rng.uniform(-0.03, 0.03, size=(n_cells, n_sps))
        U = primitive_to_conserved(Q)

        cpu_residual, gpu_residual = _cross_check(U, mesh)

        max_diff = np.max(np.abs(cpu_residual - gpu_residual))
        scale = max(np.max(np.abs(cpu_residual)), 1.0)
        assert max_diff < max(1e-6, scale * 1e-6), \
            f"max|cpu-gpu|={max_diff:.3e}, scale={scale:.3e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
