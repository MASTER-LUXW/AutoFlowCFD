"""GPU 粘性残差（`compute_viscous_residual_fr_gpu`）与 CPU 参考实现
（`compute_viscous_residual`）的完整 crosscheck（2026-09-03）。

背景：既有的 `test_gpu_viscous_interface_crosscheck.py` 整个文件
`pytest.importorskip("cupy")`，本机没有真实 CuPy，从未在本机实际
执行过。用户报告"均匀自由流场下 GPU 粘性残差本应精确为零，实测非零
（~0.43，CPU 给出~2e-12）"后排查，用 numpy-as-cupy 替身 + 真实含
多源/混合拆分面合成网格，决定性定位到**三个独立的真实 bug**（均与
问题清单 #5 的 AUSM+up shape bug/`viscous_physical_flux_gpu` 批量维
shape bug 相互独立，是同一轮排查中在粘性残差路径上连续发现的）：

1. **自身外插机制用错**（`_compute_viscous_interface_correction_gpu`）：
   owner-primary 块的 `Q_o`/`gv_o`/`gT_o`/`mut_o`（本单元自身在 FP 上
   的外插值）此前错误地复用了 `_extrap_side`（`owner_src0_cell`/
   `owner_src0_mat`——这组数组的真实语义是"给*对侧*记录查询本单元值
   用的交叉引用表"，不是"本单元查自己"），对 owner-primary 面而言这
   组数组恒为零/未设置，导致 `Q_o` 恒为零而不是真实自身状态。已改用
   新增的 `_self_extrap_side`（`boundary_extrap`/native 自身外插表，
   与 CPU 版 `E_o=boundary_extrap[celltype_o,oax,oside_idx]` 逐字
   对应），neighbor-primary 块的 `Q_n_native` 同一处修复。见
   `gpu_viscous.py::_self_extrap_side` 文档完整推导。

2. **热传导项符号搞反**（`gpu_flux.py::viscous_physical_flux_gpu`）：
   Fourier 定律 `q=-k*grad(T)`（热流方向与温度梯度相反），此前这里
   写成 `+k_eff*dTdx`（及 dTdy/dTdz），与 CPU 版 `flux_kernels.py::
   viscous_physical_flux_point` 的 `qx=-k_cond*grad_T[0]` 方向相反。
   均匀流场下 `grad_T≡0`，这个符号错误不会被"零梯度"类测试捕捉到。

3. **速度梯度用了守恒变量的梯度，不是原始变量的梯度**
   （`compute_viscous_residual_fr_gpu`）：此前对**守恒变量** `U`
   （rho, rho*u, rho*v, rho*w, rho*E）求梯度，直接把 `grad_U[...,
   1:4,:]` 当"速度梯度"用——但 `grad(rho*u)=rho*grad(u)+u*grad(rho)`，
   不是 `grad(u)`，只有密度处处均匀时两者才相等。CPU 版
   `viscous_flux.py::compute_viscous_residual_fr` 一直是对**原始
   变量** `Q`（`grad_Q=compute_physical_gradient(Q,...)`,
   `grad_vel=grad_Q[:,:,1:4,:]`）求梯度。已改为对 `Q` 求梯度，与
   CPU 一致。密度均匀/近似均匀的流场（本项目大量既有测试用的正是
   这类流场）这个 bug 造成的误差很小，容易被当成浮点噪声忽略，真实
   非均匀密度场（本文件用的合成网格扰动流场）会明显暴露。

续接（2026-09-03，排查"是否应删除 collapsed、补全 native"时用真实
native 四面体合成网格 + P2 阶数发现，见 `test_gpu_native_tet_axis_
clip_crosscheck.py` 完整推导）又发现两个独立真实 bug，其中第二个
**不是 native 专属**，任意网格（含本文件的 collapsed 模式）残差量级
足够大时都会分歧：

4. `gpu_gradients.py::compute_physical_gradient_gpu` 四面体分支此前
   无条件用 `ops_data['D_3d_tet']`（坍缩坐标微分算子），从未按是否
   存在 `D_native_tet_padded` 分派——只影响 `tet_basis_mode="native"`，
   本文件不覆盖（见 native 专属测试文件）。
5. GPU 粘性残差路径当年缺少 CPU 侧那一步"残差量级离群清零"（旧称
   "机制3"，判据是同单元中位数的 1e4 倍），残差量级足够大时 CPU 清零、
   GPU 不清零，两者分道扬镳；当年补上了同款调用。
   **2026-09-19 起这一条不再存在**：机制3 已整体删除（依据见
   `core/fr_residual/inviscid.py` —— 真实网格消融对照里它触发 15 次却
   只把残差轨迹改变 ~1e-10 相对量、不改变结局），CPU/GPU 现在都不做
   这一步。顺带说明：本文件（collapsed 模式）当年尝试构造的多组随机／
   人工单点异常场景**从未真正触发**过那个判据，这本身就是"它在健康
   与半健康场上都是惰性的"这一结论的早期证据之一。

验证方式（本机没有真实 CUDA，用 numpy-as-cupy 替身实际执行，与
`test_gpu_inviscid_ausm_mixed_face_crosscheck.py` 同一方法论）：
真实含多源/混合拆分面合成网格，均匀自由流场（残差应精确为零）+
非均匀密度/速度/温度扰动流场（真正触发全部三处 bug 的路径）两组
判据，与 CPU `compute_viscous_residual` 交叉验证。
"""

import numpy as np
import pytest
from tests.unit._gpu_cupy_shim import patch_module_get_cupy


class _NumpyAsCupy:
    def __getattr__(self, name):
        return getattr(np, name)

    def asnumpy(self, x):
        return np.asarray(x)

    def scatter_add(self, a, indices, b):
        np.add.at(a, indices, b)

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


@pytest.fixture(autouse=True)
def _patch_gpu_modules(monkeypatch):
    shim = _NumpyAsCupy()

    import autoflowcfd.core.gpu as core_gpu_mod
    import autoflowcfd.core.gpu.residual.gpu_viscous as gpu_viscous_mod
    import autoflowcfd.core.gpu.residual.gpu_inviscid as gpu_inviscid_mod
    import autoflowcfd.core.gpu.residual.gpu_inviscid_volume as gpu_inviscid_volume_mod
    import autoflowcfd.core.gpu.residual.gpu_volume_contract as gpu_volume_contract_mod
    import autoflowcfd.core.gpu.residual.gpu_flux as gpu_flux_mod
    import autoflowcfd.core.gpu.residual.gpu_gradients as gpu_gradients_mod
    import autoflowcfd.core.gpu.array_manager as array_mgr_mod
    import autoflowcfd.core.gpu.gpu_face_geometry as gfg_mod

    mods = [
        gpu_viscous_mod, gpu_inviscid_mod, gpu_inviscid_volume_mod,
        gpu_volume_contract_mod, gpu_flux_mod, gpu_gradients_mod,
        array_mgr_mod, gfg_mod,
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


def _make_mesh(order):
    from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
    return _build_synthetic_mixed_mesh(order)


def _cross_check(U, mesh, ops, mu_t_field=None):
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    from autoflowcfd.core.fr_residual.viscous import compute_viscous_residual
    from autoflowcfd.core.gpu.residual.gpu_viscous import compute_viscous_residual_fr_gpu

    Q = conserved_to_primitive(U)
    r_cpu = compute_viscous_residual(U, Q, ops, mesh, mu=1.8e-5, mu_t_field=mu_t_field)
    r_gpu = compute_viscous_residual_fr_gpu(U, mesh, ops, mu=1.8e-5, mu_t_field=mu_t_field)
    return r_cpu, np.asarray(r_gpu)


class TestGpuViscousResidualMatchesCpu:
    @pytest.mark.parametrize("order", [1, 2])
    def test_uniform_flow_residual_is_exactly_zero(self, order):
        """均匀自由流场：CPU 给出 ~机器精度零，GPU 修复前给出物理
        量级的虚假非零残差（~0.43，用户报告的原始症状），是本次三处
        bug 里问题一（自身外插机制用错）的决定性判据——三处 bug 里
        只有它在均匀流场下也会触发（另外两处都依赖非零梯度/非均匀
        密度才会暴露）。"""
        from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved

        mesh = _make_mesh(order)
        ops = mesh.operators
        rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 33.33, 0.0, 0.0, 101325.0
        Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
        U_inf = primitive_to_conserved(Q_inf)
        U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

        r_cpu, r_gpu = _cross_check(U, mesh, ops)

        assert np.max(np.abs(r_cpu)) < 1e-6, "CPU 参考实现本身必须先满足这个判据"
        assert np.max(np.abs(r_gpu)) < 1e-6, \
            f"GPU 均匀流场残差应接近零，实测 {np.max(np.abs(r_gpu)):.3e}（修复前的真实症状是 ~0.43）"

    @pytest.mark.parametrize("order", [1, 2])
    def test_nonuniform_perturbed_flow_matches_cpu(self, order):
        """非均匀密度/速度/温度扰动流场——真正触发全部三处 bug 的路径
        （问题二的 grad_T 非零、问题三的非均匀密度）。容差与既有
        `test_gpu_p1_inviscid_interface_crosscheck.py` 同一判据风格
        （相对 CPU 残差幅值的量级，不是绝对机器精度——numpy-as-cupy
        替身路径本身就是逐位相同的浮点运算，实测可以到 ~1e-16 相对
        误差，这里用更宽松的 1e-6 留出余量，判据本身不因为"恰好是同
        一套 numpy 运算"而失去意义：真正的 bug 会在这个层级产生 >>1
        的相对误差，已用人工回退验证过，见修复提交历史）。"""
        from autoflowcfd.core.fr_residual.inviscid import (
            primitive_to_conserved, conserved_to_primitive,
        )

        mesh = _make_mesh(order)
        ops = mesh.operators
        rng = np.random.default_rng(order * 3000 + 7)

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

        mu_t_field = rng.uniform(0.0, 5e-5, size=(n_cells, n_sps))

        r_cpu, r_gpu = _cross_check(U, mesh, ops, mu_t_field=mu_t_field)

        max_diff = np.max(np.abs(r_gpu - r_cpu))
        scale = max(np.max(np.abs(r_cpu)), 1.0)
        rel = max_diff / scale
        assert rel < 1e-6, f"P={order}: max|cpu-gpu|={max_diff:.3e}, scale={scale:.3e}, rel={rel:.3e}"
        assert np.all(np.isfinite(r_gpu))

    def test_laminar_no_turbulent_viscosity_matches(self):
        """mu_t_field=None（纯层流，问题三 bug 影响最直接的场景——没有
        湍流涡粘度分散注意力）。"""
        from autoflowcfd.core.fr_residual.inviscid import (
            primitive_to_conserved, conserved_to_primitive,
        )

        mesh = _make_mesh(1)
        ops = mesh.operators
        rng = np.random.default_rng(555)
        rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
        Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
        U_inf = primitive_to_conserved(Q_inf)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        U = np.tile(U_inf, (n_cells, n_sps, 1))
        Q = conserved_to_primitive(U)
        Q[..., 0] *= 1.0 + rng.uniform(-0.05, 0.05, size=(n_cells, n_sps))
        Q[..., 1] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
        Q[..., 4] *= 1.0 + rng.uniform(-0.03, 0.03, size=(n_cells, n_sps))
        U = primitive_to_conserved(Q)

        r_cpu, r_gpu = _cross_check(U, mesh, ops, mu_t_field=None)

        max_diff = np.max(np.abs(r_gpu - r_cpu))
        scale = max(np.max(np.abs(r_cpu)), 1.0)
        assert max_diff / scale < 1e-6

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
