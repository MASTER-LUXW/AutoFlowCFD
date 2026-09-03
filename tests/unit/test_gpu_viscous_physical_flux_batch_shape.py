"""`gpu_flux.py::viscous_physical_flux_gpu` 多前导批量维形状 bug 修复验证
（问题清单 #5 排查附带发现，2026-09-02）。

背景：修复单 GPU AUSM+up 多源混合面 shape bug（问题清单 #5）时，用
numpy-as-cupy 替身完整构造 `GPUFRSolver` 并真正调用 `step()`（此前
从未有任何测试走到这一步——`test_gpu_modules.py`/`test_gpu_p1_
inviscid_interface_crosscheck.py` 等既有 crosscheck 测试全部
`pytest.importorskip("cupy")`，本机没有真实 CuPy，整体跳过），在 AUSM+up
之后紧接着又撞到一处独立的真实 bug：`compute_viscous_residual_fr_gpu`
的体积项分支直接把 `Q=(n_cells,n_sps,5)`/`grad_vel=(n_cells,n_sps,3,3)`
（两个前导批量维）传给 `viscous_physical_flux_gpu`，但该函数内部
`grad_vel`/`G` 的索引/构造用 `[:, i, j]`（只认定*恰好一个*前导批量维，
把 `Q`/`grad_T` 的 `[...,k]` 省略号写法混着用），`grad_vel[:,0,0]` 会
把 n_sps 维误当"速度分量 i"维去索引，产出形状错误的中间量，最终在
`G[:,0,1]=tau_xx` 直接 `ValueError`。

严重性：这不是"多源混合面"专属的边缘情况——体积项调用点对**任意**
网格、任意阶数都是无条件按 `(n_cells,n_sps,5)` 传参（不像两处界面
校正调用点会先显式 `.reshape` 压平），意味着单 GPU（`GPUFRSolver`）/
多 GPU 分布式（`gpu_distributed.py`）后端的粘性残差体积项，此前只要
真正被调用就必然崩溃——从未被任何测试捕捉到，是因为所有既有 GPU
测试要么只测 2D 展平后的输入（凑巧绕开），要么整个文件被
`pytest.importorskip("cupy")` 跳过，从未有测试真正走到"完整构造+真正
调用 step()"这一步。

修复：`grad_vel`/`grad_T`/`G` 统一改用 `[..., i, j]`（省略号），与
`Q`（本来就已经用 `[...,k]`）保持一致的、对前导批量维数量无感知的
约定；`G` 的分配形状从硬编码的 `(N,3,5)`（`N=Q.shape[0]`）改为
`Q.shape[:-1] + (3,5)`。对已经压平成 2D 的两处界面校正调用点，
`[...,i,j]` 与原来的 `[:,i,j]` 结果完全相同（只有一个前导维时二者
等价），不改变其行为。

验证方式（本机没有真实 CUDA，用 numpy-as-cupy 替身实际执行）：
1. 2D 输入（`(N,5)`/`(N,3,3)`）：零梯度 → 零粘性通量（钉住既有
   `test_gpu_modules.py::TestGPUViscousFlux::test_zero_gradient_zero_
   viscous_flux` 同一个物理判据，那份测试本身在本机被整体跳过，这里
   用可执行的方式重新验证一遍，防止回归）。
2. 3D 批量输入（`(n_cells,n_sps,5)`/`(n_cells,n_sps,3,3)`，即真实体积
   项调用点的形状）：此前必现崩溃，现在必须不崩溃且给出正确结果。
3. 不变量：把同一份数据分别以"3D 批量"和"手动展平成 2D 再 reshape
   回来"两种方式调用，两者必须逐位相等——这是修复正确性（不只是
   "不崩溃"）的决定性判据，不依赖任何外部 CPU 参考实现。
"""

import numpy as np
import pytest


class _NumpyAsCupy:
    def __getattr__(self, name):
        return getattr(np, name)

    def asnumpy(self, x):
        return np.asarray(x)


@pytest.fixture(autouse=True)
def _patch_get_cupy(monkeypatch):
    shim = _NumpyAsCupy()
    import autoflowcfd.core.gpu as core_gpu_mod
    import autoflowcfd.core.gpu.residual.gpu_flux as gpu_flux_mod
    monkeypatch.setattr(core_gpu_mod, "get_cupy", lambda: shim)
    monkeypatch.setattr(gpu_flux_mod, "get_cupy", lambda: shim)


def _make_batch(shape_extra, rng):
    """构造一个物理量级合理的原始变量场 + 非零速度/温度梯度。"""
    n = int(np.prod(shape_extra))
    rho = rng.uniform(1.1, 1.3, size=n).reshape(shape_extra)
    u = rng.uniform(20.0, 40.0, size=n).reshape(shape_extra)
    v = rng.uniform(-5.0, 5.0, size=n).reshape(shape_extra)
    w = rng.uniform(-5.0, 5.0, size=n).reshape(shape_extra)
    p = rng.uniform(9.5e4, 1.05e5, size=n).reshape(shape_extra)
    Q = np.stack([rho, u, v, w, p], axis=-1)

    grad_vel = rng.uniform(-50.0, 50.0, size=n * 9).reshape(shape_extra + (3, 3))
    grad_T = rng.uniform(-100.0, 100.0, size=n * 3).reshape(shape_extra + (3,))
    return Q, grad_vel, grad_T


class TestViscousPhysicalFluxZeroGradient:
    """零梯度 → 零粘性通量，2D/3D+ 批量输入都必须成立（钉住既有
    `test_gpu_modules.py` 同一判据，但用本机可实际执行的路径）。"""

    def test_2d_batch_zero_gradient(self):
        from autoflowcfd.core.gpu.residual.gpu_flux import viscous_physical_flux_gpu
        Q = np.asarray([[1.225, 30.0, 0.0, 0.0, 101325.0]])
        grad_vel = np.zeros((1, 3, 3))
        grad_T = np.zeros((1, 3))
        G = viscous_physical_flux_gpu(Q, grad_vel, grad_T, mu=1.8e-5, Pr=0.72)
        assert G.shape == (1, 3, 5)
        np.testing.assert_allclose(G, 0.0, atol=1e-14)

    def test_3d_batch_zero_gradient_does_not_crash_and_is_zero(self):
        """真实体积项调用点的形状（`(n_cells,n_sps,5)`）——此前必现
        `ValueError`，修复后必须不崩溃且给出精确零结果。"""
        from autoflowcfd.core.gpu.residual.gpu_flux import viscous_physical_flux_gpu
        n_cells, n_sps = 4, 8
        Q = np.tile(
            np.asarray([1.225, 30.0, 0.0, 0.0, 101325.0]), (n_cells, n_sps, 1)
        )
        grad_vel = np.zeros((n_cells, n_sps, 3, 3))
        grad_T = np.zeros((n_cells, n_sps, 3))
        G = viscous_physical_flux_gpu(Q, grad_vel, grad_T, mu=1.8e-5, Pr=0.72)
        assert G.shape == (n_cells, n_sps, 3, 5)
        np.testing.assert_allclose(G, 0.0, atol=1e-14)


class TestViscousPhysicalFluxBatchShapeInvariance:
    """决定性正确性判据：3D 批量调用与"手动展平成 2D 再 reshape 回来"
    调用必须逐位相等——不依赖外部 CPU 参考实现，纯粹是本函数自身
    对批量维数量应该无感知这个不变量。"""

    @pytest.mark.parametrize("n_cells,n_sps", [(4, 8), (3, 5), (7, 1)])
    def test_3d_matches_manually_flattened_2d(self, n_cells, n_sps):
        from autoflowcfd.core.gpu.residual.gpu_flux import viscous_physical_flux_gpu

        rng = np.random.default_rng(n_cells * 1000 + n_sps)
        Q, grad_vel, grad_T = _make_batch((n_cells, n_sps), rng)
        mu, Pr = 1.8e-5, 0.72
        mu_t_field = rng.uniform(0.0, 1e-4, size=(n_cells, n_sps))

        G_3d = viscous_physical_flux_gpu(Q, grad_vel, grad_T, mu, Pr, mu_t=mu_t_field, Pr_t=0.9)
        assert G_3d.shape == (n_cells, n_sps, 3, 5)

        N = n_cells * n_sps
        G_2d = viscous_physical_flux_gpu(
            Q.reshape(N, 5), grad_vel.reshape(N, 3, 3), grad_T.reshape(N, 3),
            mu, Pr, mu_t=mu_t_field.reshape(N), Pr_t=0.9,
        ).reshape(n_cells, n_sps, 3, 5)

        np.testing.assert_allclose(G_3d, G_2d, rtol=1e-12, atol=1e-12)
        assert np.all(np.isfinite(G_3d))

    def test_laminar_no_turbulent_viscosity_matches(self):
        """`mu_t=None`（层流，体积项调用点的默认情形）同样满足这个
        不变量。"""
        from autoflowcfd.core.gpu.residual.gpu_flux import viscous_physical_flux_gpu

        n_cells, n_sps = 4, 8
        rng = np.random.default_rng(42)
        Q, grad_vel, grad_T = _make_batch((n_cells, n_sps), rng)
        mu, Pr = 1.8e-5, 0.72

        G_3d = viscous_physical_flux_gpu(Q, grad_vel, grad_T, mu, Pr, mu_t=None, Pr_t=0.9)
        N = n_cells * n_sps
        G_2d = viscous_physical_flux_gpu(
            Q.reshape(N, 5), grad_vel.reshape(N, 3, 3), grad_T.reshape(N, 3),
            mu, Pr, mu_t=None, Pr_t=0.9,
        ).reshape(n_cells, n_sps, 3, 5)

        np.testing.assert_allclose(G_3d, G_2d, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
