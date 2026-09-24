"""GPU 无粘残差 `tet_basis_mode="native"` 组合的 axis 越界 bug 修复
验证（2026-09-03，排查"是否应删除 collapsed、补全 native"这个问题时
用真实 numpy-as-cupy 替身首次真正端到端跑通 `tet_basis_mode="native"`
+ GPU 才发现——此前唯一覆盖这个组合的 crosscheck 测试整个文件
`pytest.importorskip("cupy")` 跳过，从未在本机（或任何已知环境）真正
执行过）。

真实根因：`owner_axis`/`neighbor_axis` 对 native 四面体面存的是复用的
`excluded_vertex`（取值 0~3，见 `fr/face_flux_points/merge.py` 模块
文档"owner_axis 对 native 面存的是复用的 excluded_vertex"一节），不是
坍缩坐标的真实轴（0~2）——`gpu_inviscid.py::_compute_interface_
correction_gpu` 里两处：(1) `E_o_collapsed = ff.boundary_extrap[
celltype_o, oax, oside_idx]`（自身面外插矩阵）、(2)
`distribute_face_correction_to_sps(cp, jump_owner, oax, ...)`（面
校正分配）都无条件用原始 `oax`/`nax` 去 gather 只有 3 个轴的表
（`ff.boundary_extrap`/`ff.dist_fp_of_sp`/`ff.dist_axis_coord_of_sp`），
`excluded_vertex==3` 时直接 `IndexError`——即便下游
`_native_or_collapsed_contrib`/`_native_self_extrap` 最终会用
`is_native` 掩码丢弃这个 collapsed 分支的结果，NumPy/CuPy 的花式索引
在"丢弃"发生前就已经越界崩溃了（与 `_native_self_extrap` 自己那处
"`boundary_extrap_native` 空数组 + `cp.clip(x,0,-1)`"是同一类"提前
无条件求值导致越界"问题，只是这里是另外两张表）。`gpu_viscous.py::
_self_extrap_side`（新增于同日排查单GPU AUSM+up shape bug 时）从
`gpu_inviscid.py` 抄来的同一套自身外插逻辑，继承了同一个 bug。

修复：owner/neighbor 两侧都在使用 `oax`/`nax` 索引 `boundary_extrap`/
`dist_fp_of_sp`/`dist_axis_coord_of_sp` 之前，先用 `is_native`（由
`owner_cube_face`/`neighbor_cube_face >= 6` 判定）把 native 面的
`oax`/`nax` clip 到一个恒安全的哑值 0（结果本来就会被下游的 `is_
native` 掩码丢弃，不影响 collapsed 面的真实行为）。四处：
`gpu_inviscid.py` 的 owner/neighbor 自身外插各一处 + 面校正分配各
一处；`gpu_viscous.py::_self_extrap_side` 一处（owner/neighbor 共用）。

修复后用真实含 native 四面体的合成网格，`compute_inviscid_residual_
fr_gpu` 与 CPU 参考实现 `compute_inviscid_residual_fr` 交叉验证到
~1e-15 相对精度（`tet_basis_mode="native"` 组合此前从未在任何环境
被真正执行验证过，这是第一次）。

**续接（同日）：粘性残差在这个组合下另外两个独立真实 bug，逐一定位并
修复**：axis-clip 崩溃解决后，粘性残差与 CPU 交叉验证仍有真实数值
偏差（P1 ~9%，P2 更大）。逐项定位：
1. `gpu_gradients.py::compute_physical_gradient_gpu` 四面体分支此前
   无条件用 `ops_data['D_3d_tet']`（坍缩坐标微分算子），从未像
   `compute_volume_term_gpu` 那样按是否存在 `D_native_tet_padded`
   分派——`tet_basis_mode="native"` 时四面体的物理梯度因此用了完全
   错误的参考空间微分算子（棱柱不受影响，两种模式下都用同一个
   `D_3d_prism`）。修复：`'D_native_tet_padded' in ops_data` 自描述
   判断，不需要改任何调用点签名。
2. GPU 粘性残差路径当年缺少 CPU 侧那一步"残差量级离群清零"（旧称
   "机制3"），P2 随机扰动流场上 CPU 真实触发、GPU 未触发，两者从
   那一步起分道扬镳。当年的修法是给 GPU 补上同款调用。
   **2026-09-19 起这一条不再存在**：机制3 已整体删除（真实网格消融
   对照证明它触发了但只把残差轨迹改变 ~1e-10 相对量、不改变发散
   结局，见 `core/fr_residual/inviscid.py`），CPU/GPU 两侧现在都不做
   这一步，对称性由"都没有"保证。本文件的交叉验证因此只剩第 1 条
   （梯度算子分派）这一个真实回归点，判据不变。

两处修复后，`tet_basis_mode="native"` 下 `compute_viscous_residual_
fr_gpu` 与 CPU 参考实现交叉验证到 ~1e-16/1e-17 相对精度（P1/P2 均
验证）。
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


@pytest.fixture(autouse=True)
def _patch_gpu_modules(monkeypatch):
    shim = _NumpyAsCupy()

    import autoflowcfd.core.gpu as core_gpu_mod
    import autoflowcfd.core.gpu.residual.gpu_inviscid as gpu_inviscid_mod
    import autoflowcfd.core.gpu.residual.gpu_inviscid_volume as gpu_inviscid_volume_mod
    import autoflowcfd.core.gpu.residual.gpu_volume_contract as gpu_volume_contract_mod
    import autoflowcfd.core.gpu.residual.gpu_flux as gpu_flux_mod
    import autoflowcfd.core.gpu.residual.gpu_gradients as gpu_gradients_mod
    import autoflowcfd.core.gpu.residual.gpu_viscous as gpu_viscous_mod
    import autoflowcfd.core.gpu.array_manager as array_mgr_mod
    import autoflowcfd.core.gpu.gpu_face_geometry as gfg_mod

    mods = [
        gpu_inviscid_mod, gpu_inviscid_volume_mod, gpu_volume_contract_mod,
        gpu_flux_mod, gpu_gradients_mod, gpu_viscous_mod, array_mgr_mod, gfg_mod,
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


class TestNativeTetAxisClipFix:
    @pytest.mark.parametrize("order", [1, 2])
    def test_native_inviscid_residual_no_longer_crashes_and_matches_cpu(self, order):
        from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
        from autoflowcfd.fr.operators import generate_fr_operators
        from autoflowcfd.core.fr_residual.inviscid import (
            compute_inviscid_residual_fr, primitive_to_conserved, conserved_to_primitive,
        )
        from autoflowcfd.core.gpu.residual.gpu_inviscid import compute_inviscid_residual_fr_gpu

        mesh = _build_synthetic_mixed_mesh(order, tet_basis_mode="native")
        ops = generate_fr_operators(order)
        rng = np.random.default_rng(order * 4000 + 3)

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

        r_cpu = compute_inviscid_residual_fr(U, mesh, ops, mach_ref=0.2)
        r_gpu = np.asarray(compute_inviscid_residual_fr_gpu(U, mesh, ops, mach_ref=0.2))

        max_diff = np.max(np.abs(r_gpu - r_cpu))
        scale = max(np.max(np.abs(r_cpu)), 1.0)
        rel = max_diff / scale
        assert rel < 1e-9, f"P={order} native: max|cpu-gpu|={max_diff:.3e}, scale={scale:.3e}, rel={rel:.3e}"
        assert np.all(np.isfinite(r_gpu))

    @pytest.mark.parametrize("order", [1, 2])
    def test_native_viscous_residual_matches_cpu(self, order):
        """锁定同日续接发现的另外两个独立 bug（`compute_physical_
        gradient_gpu` 四面体梯度分派缺失 + `compute_viscous_residual_
        fr_gpu` 缺失机制3异常抑制）——P1 残差量级小，两个 bug 都不会
        单独暴露出可观测差异；P2（真实随机扰动流场）才能同时触发。"""
        from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
        from autoflowcfd.fr.operators import generate_fr_operators
        from autoflowcfd.core.fr_residual.inviscid import (
            primitive_to_conserved, conserved_to_primitive,
        )
        from autoflowcfd.core.fr_residual.viscous import compute_viscous_residual
        from autoflowcfd.core.gpu.residual.gpu_viscous import compute_viscous_residual_fr_gpu

        mesh = _build_synthetic_mixed_mesh(order, tet_basis_mode="native")
        ops = generate_fr_operators(order)
        rng = np.random.default_rng(order * 4000 + 3)

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

        from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr
        from autoflowcfd.core.gpu.residual.gpu_inviscid import compute_inviscid_residual_fr_gpu
        compute_inviscid_residual_fr(U, mesh, ops, mach_ref=0.2)
        compute_inviscid_residual_fr_gpu(U, mesh, ops, mach_ref=0.2)

        mu_t_field = rng.uniform(0.0, 5e-5, size=(n_cells, n_sps))
        r_cpu = compute_viscous_residual(U, conserved_to_primitive(U), ops, mesh, mu=1.8e-5, mu_t_field=mu_t_field)
        r_gpu = np.asarray(compute_viscous_residual_fr_gpu(U, mesh, ops, mu=1.8e-5, mu_t_field=mu_t_field))

        max_diff = np.max(np.abs(r_gpu - r_cpu))
        scale = max(np.max(np.abs(r_cpu)), 1.0)
        rel = max_diff / scale
        assert rel < 1e-9, f"P={order} native viscous: max|cpu-gpu|={max_diff:.3e}, scale={scale:.3e}, rel={rel:.3e}"
        assert np.all(np.isfinite(r_gpu))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
