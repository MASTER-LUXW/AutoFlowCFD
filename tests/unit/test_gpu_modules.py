"""
AutoFlowCFD V2.0 - GPU 模块单元测试

测试 GPU 模块的正确性（需要 CuPy + CUDA GPU）。
未安装 CuPy 时自动跳过。

测试项：
1. GPUArrayManager 基础功能
2. GPU 通量计算与 CPU 一致性
3. GPU 张量收缩与 CPU 一致性
4. GPU 时间积分正定性强制
5. GPU 版守恒/原始变量转换一致性
6. GPU 版粘性通量与 CPU 一致性
7. GPU 版梯度计算与 CPU 一致性
"""

import numpy as np
import pytest

# CuPy 是可选依赖
cupy = pytest.importorskip("cupy")

GAMMA = 1.4


class TestGPUArrayManager:
    """GPUArrayManager 基础功能测试。"""

    def test_init_and_cleanup(self):
        from autoflowcfd.core.gpu.array_manager import GPUArrayManager
        mgr = GPUArrayManager(device_id=0)
        assert mgr.device_id == 0
        mem = mgr.get_memory_usage()
        assert mem['total_mb'] > 0
        mgr.cleanup()

    def test_to_gpu_and_back(self):
        from autoflowcfd.core.gpu.array_manager import GPUArrayManager
        mgr = GPUArrayManager(device_id=0)
        arr_np = np.random.rand(100, 5)
        arr_gpu = mgr.to_gpu(arr_np)
        arr_back = mgr.to_cpu(arr_gpu)
        np.testing.assert_allclose(arr_np, arr_back)
        mgr.cleanup()

    def test_gpu_zeros(self):
        from autoflowcfd.core.gpu.array_manager import GPUArrayManager
        mgr = GPUArrayManager(device_id=0)
        z = mgr.gpu_zeros((10, 3))
        assert z.shape == (10, 3)
        np.testing.assert_allclose(mgr.to_cpu(z), 0.0)
        mgr.cleanup()


class TestGPUFlux:
    """GPU 通量计算与 CPU 一致性。"""

    def test_conserved_to_primitive(self):
        from autoflowcfd.core.gpu.gpu_flux import conserved_to_primitive_gpu
        # 构造守恒变量
        rho = 1.225
        u, v, w = 30.0, 0.0, 0.0
        p = 101325.0
        ke = 0.5 * (u**2 + v**2 + w**2)
        E = p / ((GAMMA - 1.0) * rho) + ke
        U = np.array([[[rho, rho*u, rho*v, rho*w, rho*E]]])
        U_gpu = cupy.asarray(U)
        Q_gpu = conserved_to_primitive_gpu(U_gpu)
        Q = cupy.asnumpy(Q_gpu)
        np.testing.assert_allclose(Q[0, 0, 0], rho, rtol=1e-10)
        np.testing.assert_allclose(Q[0, 0, 1], u, rtol=1e-10)
        np.testing.assert_allclose(Q[0, 0, 4], p, rtol=1e-6)

    def test_euler_flux_symmetry(self):
        from autoflowcfd.core.gpu.gpu_flux import euler_physical_flux_gpu
        # 静止流体：通量应只有压力项
        Q = cupy.asarray([[1.225, 0.0, 0.0, 0.0, 101325.0]])
        F = euler_physical_flux_gpu(Q)
        F_np = cupy.asnumpy(F)
        # 质量通量为零
        np.testing.assert_allclose(F_np[0, :, 0], 0.0, atol=1e-10)
        # 动量通量只有对角压力项
        np.testing.assert_allclose(F_np[0, 0, 1], 101325.0, rtol=1e-6)
        np.testing.assert_allclose(F_np[0, 1, 2], 101325.0, rtol=1e-6)
        np.testing.assert_allclose(F_np[0, 2, 3], 101325.0, rtol=1e-6)

    def test_primitive_to_conserved_roundtrip(self):
        from autoflowcfd.core.gpu.gpu_flux import (
            conserved_to_primitive_gpu, primitive_to_conserved_gpu
        )
        Q_orig = cupy.asarray([[1.225, 30.0, 5.0, 0.0, 101325.0]])
        U = primitive_to_conserved_gpu(Q_orig)
        Q_back = conserved_to_primitive_gpu(U)
        np.testing.assert_allclose(cupy.asnumpy(Q_back), cupy.asnumpy(Q_orig), rtol=1e-10)


class TestGPUVolumeContract:
    """GPU 张量收缩与 CPU 一致性。"""

    def test_contract_1axis(self):
        from autoflowcfd.core.gpu.gpu_volume_contract import gpu_contract_shared_operator_1axis
        D = np.random.rand(4, 6)
        X = np.random.rand(10, 6, 5)
        # CPU 参考
        ref = np.einsum("fs,csv->cfv", D, X)
        # GPU
        D_gpu = cupy.asarray(D)
        X_gpu = cupy.asarray(X)
        result_gpu = gpu_contract_shared_operator_1axis(D_gpu, X_gpu)
        result = cupy.asnumpy(result_gpu)
        np.testing.assert_allclose(result, ref, rtol=1e-10)

    def test_contract_2axis(self):
        from autoflowcfd.core.gpu.gpu_volume_contract import gpu_contract_shared_operator_2axis
        D = np.random.rand(4, 3, 2)
        X = np.random.rand(10, 3, 2, 5)
        ref = np.einsum("fjm,cjmv->cfv", D, X)
        D_gpu = cupy.asarray(D)
        X_gpu = cupy.asarray(X)
        result_gpu = gpu_contract_shared_operator_2axis(D_gpu, X_gpu)
        result = cupy.asnumpy(result_gpu)
        np.testing.assert_allclose(result, ref, rtol=1e-10)


class TestGPUTimeIntegration:
    """GPU 时间积分测试。"""

    def test_enforce_positivity(self):
        from autoflowcfd.core.gpu.gpu_time_integration import enforce_positivity_gpu
        # 负密度和压力
        U = cupy.array([[
            -0.1,  # 负密度 → 应被修正
            1.0, 0.0, 0.0,
            -100.0,  # 负压力 → 应被修正
        ]])
        enforce_positivity_gpu(U, p_floor=1.0)
        U_np = cupy.asnumpy(U)
        assert U_np[0, 0] >= 1e-6  # 密度被修正
        # 压力应 >= p_floor
        rho = U_np[0, 0]
        ke = 0.5 * (U_np[0, 1]**2) / rho
        p = (GAMMA - 1.0) * (U_np[0, 4] - ke)
        assert p >= 1.0

    def test_time_integrator_euler(self):
        from autoflowcfd.core.gpu.gpu_time_integration import GPUTimeIntegrator
        integrator = GPUTimeIntegrator(scheme="forward_euler", cfl=1.0)
        U = cupy.array([[1.0, 0.0, 0.0, 0.0, 100.0]])
        dt_local = cupy.array([0.01])
        # 残差函数：常数残差
        def residual(U):
            return cupy.ones_like(U) * 10.0
        U_new = integrator.step(U, residual, dt_local)
        U_expected = U - dt_local[:, None] * 10.0
        np.testing.assert_allclose(cupy.asnumpy(U_new), cupy.asnumpy(U_expected), rtol=1e-10)


class TestGPUGradients:
    """GPU 梯度计算测试。"""

    def test_constant_field_zero_gradient(self):
        """常数场的梯度应为零。"""
        from autoflowcfd.core.gpu.gpu_gradients import compute_physical_gradient_gpu
        n_cells = 5
        n_sps = 4
        n_vars = 3
        # 常数场
        field = cupy.ones((n_cells, n_sps, n_vars)) * 42.0
        # 简单的度量数据
        inv_jacs = cupy.zeros((n_cells, n_sps, 3, 3))
        for c in range(n_cells):
            for s in range(n_sps):
                inv_jacs[c, s] = cupy.eye(3)
        mesh_data = {'inv_jacs': inv_jacs, 'n_prism': 0}
        # 简单的微分矩阵（全零 → 参考空间梯度为零）
        D_3d_tet = cupy.zeros((n_sps, n_sps, 3))
        ops_data = {'D_3d_tet': D_3d_tet}
        grad = compute_physical_gradient_gpu(field, mesh_data, ops_data)
        np.testing.assert_allclose(cupy.asnumpy(grad), 0.0, atol=1e-14)


class TestGPUViscousFlux:
    """GPU 粘性通量测试。"""

    def test_zero_gradient_zero_viscous_flux(self):
        """零梯度时粘性通量应为零（除了能量方程中的热传导项）。"""
        from autoflowcfd.core.gpu.gpu_flux import viscous_physical_flux_gpu
        Q = cupy.asarray([[1.225, 30.0, 0.0, 0.0, 101325.0]])
        grad_vel = cupy.zeros((1, 3, 3))
        grad_T = cupy.zeros((1, 3))
        G = viscous_physical_flux_gpu(Q, grad_vel, grad_T, mu=1.8e-5, Pr=0.72)
        G_np = cupy.asnumpy(G)
        # 零梯度 → 零应力 → 零粘性通量
        np.testing.assert_allclose(G_np, 0.0, atol=1e-14)


class TestGPUTurbulenceSST:
    """GPU SST 湍流模型测试。"""

    def test_strain_rate_magnitude(self):
        """应变率模计算正确性。"""
        from autoflowcfd.core.gpu.gpu_turbulence_sst import GPUTurbulenceSST
        sst = GPUTurbulenceSST(n_cells=2, n_sps=4, device_id=0)
        # 纯剪切流：du/dy = 1, 其他梯度为零
        grad_u = cupy.zeros((2, 4, 3, 3))
        grad_u[:, :, 0, 1] = 1.0  # du/dy = 1
        S_mag = sst.compute_strain_rate_magnitude_gpu(grad_u)
        S_np = cupy.asnumpy(S_mag)
        # |S| = sqrt(2 * S_ij * S_ij) = sqrt(2 * 0.5^2 * 2) = 1
        np.testing.assert_allclose(S_np, 1.0, rtol=1e-10)
        sst.cleanup()

    def test_blending_functions_range(self):
        """Blending functions F1, F2 应在 [0, 1] 范围内。"""
        from autoflowcfd.core.gpu.gpu_turbulence_sst import GPUTurbulenceSST
        sst = GPUTurbulenceSST(n_cells=10, n_sps=4, device_id=0)
        k = cupy.ones((10, 4)) * 1e-3
        omega = cupy.ones((10, 4)) * 100.0
        d = cupy.ones((10, 4)) * 0.01
        nu = cupy.ones((10, 4)) * 1.5e-5
        rho = cupy.ones((10, 4)) * 1.225
        CD_kw = cupy.ones((10, 4)) * 1e-5

        F1 = sst.compute_blending_F1_gpu(k, omega, d, nu, rho, CD_kw)
        F2 = sst.compute_blending_F2_gpu(k, omega, d, nu)

        F1_np = cupy.asnumpy(F1)
        F2_np = cupy.asnumpy(F2)
        assert np.all(F1_np >= 0) and np.all(F1_np <= 1)
        assert np.all(F2_np >= 0) and np.all(F2_np <= 1)
        sst.cleanup()

    def test_eddy_viscosity_positive(self):
        """涡粘系数应为正值。"""
        from autoflowcfd.core.gpu.gpu_turbulence_sst import GPUTurbulenceSST
        sst = GPUTurbulenceSST(n_cells=5, n_sps=4, device_id=0)
        k = cupy.ones((5, 4)) * 1e-3
        omega = cupy.ones((5, 4)) * 100.0
        rho = cupy.ones((5, 4)) * 1.225
        S_mag = cupy.ones((5, 4)) * 10.0
        F2 = cupy.ones((5, 4)) * 0.5

        nu_t = sst.compute_eddy_viscosity_gpu(k, omega, rho, S_mag, F2, mu=1.8e-5)
        nu_t_np = cupy.asnumpy(nu_t)
        assert np.all(nu_t_np > 0)
        sst.cleanup()

    def test_source_terms_shape(self):
        """源项输出形状应与输入一致。"""
        from autoflowcfd.core.gpu.gpu_turbulence_sst import GPUTurbulenceSST
        n_cells, n_sps = 8, 4
        sst = GPUTurbulenceSST(n_cells, n_sps, device_id=0)

        Q = cupy.zeros((n_cells, n_sps, 5))
        Q[:, :, 0] = 1.225
        Q[:, :, 4] = 101325.0
        grad_U = cupy.zeros((n_cells, n_sps, 3, 3))
        d_wall = cupy.ones((n_cells, n_sps)) * 0.01
        grad_k = cupy.zeros((n_cells, n_sps, 3))
        grad_omega = cupy.zeros((n_cells, n_sps, 3))

        Sk, S_omega = sst.compute_source_terms_gpu(
            Q, grad_U, d_wall, 1.8e-5, grad_k, grad_omega
        )
        assert Sk.shape == (n_cells, n_sps)
        assert S_omega.shape == (n_cells, n_sps)
        sst.cleanup()


class TestGPUHaloExchange:
    """GPU Halo 交换测试（需要 MPI 环境）。"""

    def test_cuda_aware_mpi_detection(self):
        """CUDA-aware MPI 检测应返回布尔值。"""
        from autoflowcfd.core.gpu.gpu_halo_exchange import is_cuda_aware_mpi
        result = is_cuda_aware_mpi()
        assert isinstance(result, bool)
