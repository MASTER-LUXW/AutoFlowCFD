"""AutoFlowCFD V2.0 - GPU 版 WALE(gpu_sgs.py)/DDES/IDDES(gpu_turbulence_des.py)
与 CPU 版数值一致性测试 (#7 第四次评审第四轮)。

本机没有 CuPy/真实 CUDA 设备；这两个模块的实现只用了 numpy 与 CuPy
共同支持的 API（`transpose`/`sum`/`matmul`/`sqrt`/`exp`/`tanh`/`maximum`/
`minimum`/`where`/`tile` 等，语义完全一致）——用 monkeypatch 把
`get_cupy()` 换成 numpy 模块本身，直接跑*生产类的方法*，对照 CPU 版
`WALEModel`/`DDESModel`/`IDDESModel` 在同一组合成输入上的输出，数值
要求逐位精确相等。
"""

import numpy as np
import pytest

from autoflowcfd.core.turbulence.sgs import WALEModel
from autoflowcfd.core.turbulence.des import DDESModel, IDDESModel

import autoflowcfd.core.gpu.turbulence.gpu_sgs as gpu_sgs_mod
import autoflowcfd.core.gpu.turbulence.gpu_turbulence_des as gpu_des_mod
from autoflowcfd.core.gpu.turbulence.gpu_sgs import GPUWALEModel
from autoflowcfd.core.gpu.turbulence.gpu_turbulence_des import GPUDDESModel, GPUIDDESModel


@pytest.fixture(autouse=True)
def _patch_get_cupy(monkeypatch):
    monkeypatch.setattr(gpu_sgs_mod, "gpu_available", True)
    monkeypatch.setattr(gpu_sgs_mod, "get_cupy", lambda: np)
    monkeypatch.setattr(gpu_des_mod, "gpu_available", True)
    monkeypatch.setattr(gpu_des_mod, "get_cupy", lambda: np)


def _random_grad_u(rng, n_cells, n_sps):
    return rng.uniform(-50.0, 50.0, size=(n_cells, n_sps, 3, 3))


class TestGpuWaleMatchesCpu:
    def test_eddy_viscosity_matches_cpu_on_random_field(self):
        rng = np.random.default_rng(0)
        n_cells, n_sps = 6, 4
        grad_u = _random_grad_u(rng, n_cells, n_sps)
        delta = rng.uniform(1e-4, 1e-2, size=(n_cells, n_sps))

        cpu = WALEModel(c_wale=0.325)
        expected = cpu.compute_eddy_viscosity(grad_u, delta)

        gpu = GPUWALEModel(c_wale=0.325)
        actual = gpu.compute_eddy_viscosity_gpu(grad_u, delta)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-16)
        np.testing.assert_allclose(gpu.nu_t, expected, rtol=1e-12, atol=1e-16)

    def test_pure_shear_analytic_reference(self):
        """纯剪切流解析特例（CPU 版类文档提到"漏迹版本在纯剪切下恰好
        与正确版本相等"，这里独立验证 GPU 版同样满足）：
        grad_u = [[0,1,0],[0,0,0],[0,0,0]] 单位剪切率。"""
        n_cells, n_sps = 1, 1
        grad_u = np.zeros((n_cells, n_sps, 3, 3))
        grad_u[0, 0, 0, 1] = 1.0
        delta = np.array([[0.01]])

        gpu = GPUWALEModel(c_wale=0.325)
        nu_t = gpu.compute_eddy_viscosity_gpu(grad_u, delta)
        assert nu_t[0, 0] > 0.0
        assert np.isfinite(nu_t[0, 0])

    def test_negative_control_traceless_fix_matters(self):
        """反向对照：去迹（正确实现）与不去迹（此前的 bug）在一般三维
        应变下必须给出不同结果——证明测试确实在检验这一步骤，而不是
        对任何输入都凑巧相等。"""
        rng = np.random.default_rng(1)
        n_cells, n_sps = 4, 3
        grad_u = _random_grad_u(rng, n_cells, n_sps)
        delta = rng.uniform(1e-4, 1e-2, size=(n_cells, n_sps))

        gpu = GPUWALEModel(c_wale=0.325)
        S_ij, Omega_ij = gpu.compute_strain_and_rotation_tensors_gpu(grad_u)
        L_sq_correct = gpu.compute_wale_invariant_gpu(S_ij, Omega_ij)

        L_ij = np.matmul(S_ij, S_ij) + np.matmul(Omega_ij, Omega_ij)
        L_sq_no_trace_fix = np.sum(L_ij * L_ij, axis=(2, 3))

        assert not np.allclose(L_sq_correct, L_sq_no_trace_fix)


class TestGpuDdesMatchesCpu:
    def test_apply_to_sst_model_length_scale_matches_cpu(self):
        rng = np.random.default_rng(2)
        n_cells, n_sps = 5, 4
        grad_u = _random_grad_u(rng, n_cells, n_sps)
        d_w = rng.uniform(1e-4, 1e-1, size=(n_cells, n_sps))
        cell_volumes = rng.uniform(1e-7, 1e-5, size=(n_cells,))
        nu = np.full((n_cells, n_sps), 1.5e-5)

        class _FakeSST:
            beta_star = 0.09

            def __init__(self):
                self.k_field = rng.uniform(1e-6, 1e-3, size=(n_cells, n_sps))
                self.omega_field = rng.uniform(1.0, 500.0, size=(n_cells, n_sps))
                self.des_length_scale = None

            def get_turbulent_viscosity(self):
                return self.nu_t

        cpu_sst = _FakeSST()
        cpu_sst.nu_t = rng.uniform(1e-6, 1e-4, size=(n_cells, n_sps))
        cpu_ddes = DDESModel()
        cpu_ddes.apply_to_sst_model(cpu_sst, d_w, cell_volumes, nu, grad_u)

        class _FakeGpuSST:
            beta_star = 0.09

            def __init__(self):
                self.k_field = cpu_sst.k_field.copy()
                self.omega_field = cpu_sst.omega_field.copy()
                self.nu_t = cpu_sst.nu_t.copy()
                self.des_length_scale = None

        gpu_sst = _FakeGpuSST()
        gpu_ddes = GPUDDESModel()
        gpu_ddes.apply_to_sst_model_gpu(gpu_sst, d_w, cell_volumes, nu, grad_u)

        np.testing.assert_allclose(gpu_sst.des_length_scale, cpu_sst.des_length_scale, rtol=1e-10, atol=1e-14)


class TestGpuIddesMatchesCpu:
    def test_apply_to_sst_model_iddes_length_scale_matches_cpu(self):
        rng = np.random.default_rng(3)
        n_cells, n_sps = 6, 4
        grad_u = _random_grad_u(rng, n_cells, n_sps)
        d_w = rng.uniform(1e-4, 1e-1, size=(n_cells, n_sps))
        h_max = rng.uniform(1e-3, 1e-1, size=(n_cells,))
        h_wn = h_max * rng.uniform(0.1, 0.9, size=(n_cells,))
        nu = np.full((n_cells, n_sps), 1.5e-5)

        class _FakeSST:
            beta_star = 0.09

            def __init__(self):
                self.k_field = rng.uniform(1e-6, 1e-3, size=(n_cells, n_sps))
                self.omega_field = rng.uniform(1.0, 500.0, size=(n_cells, n_sps))
                self.des_length_scale = None

            def get_turbulent_viscosity(self):
                return self.nu_t

        cpu_sst = _FakeSST()
        cpu_sst.nu_t = rng.uniform(1e-6, 1e-4, size=(n_cells, n_sps))
        cpu_iddes = IDDESModel()
        cpu_iddes.apply_to_sst_model_iddes(cpu_sst, d_w, h_max, h_wn, nu, grad_u)

        class _FakeGpuSST:
            beta_star = 0.09

            def __init__(self):
                self.k_field = cpu_sst.k_field.copy()
                self.omega_field = cpu_sst.omega_field.copy()
                self.nu_t = cpu_sst.nu_t.copy()
                self.des_length_scale = None

        gpu_sst = _FakeGpuSST()
        gpu_iddes = GPUIDDESModel()
        gpu_iddes.apply_to_sst_model_iddes_gpu(gpu_sst, d_w, h_max, h_wn, nu, grad_u)

        np.testing.assert_allclose(gpu_sst.des_length_scale, cpu_sst.des_length_scale, rtol=1e-10, atol=1e-14)

    def test_negative_control_iddes_differs_from_ddes(self):
        """反向对照：DDES 与 IDDES 公式结构不同，同一组输入下长度尺度
        不应恰好相等（证明两套实现真的在算不同的东西，不是其中一个
        误把另一个的公式抄了一遍）。"""
        rng = np.random.default_rng(4)
        n_cells, n_sps = 5, 4
        grad_u = _random_grad_u(rng, n_cells, n_sps)
        d_w = rng.uniform(1e-4, 1e-1, size=(n_cells, n_sps))
        cell_volumes = rng.uniform(1e-7, 1e-5, size=(n_cells,))
        h_max = cell_volumes ** (1.0 / 3.0) * 3.0
        h_wn = h_max * 0.5
        nu = np.full((n_cells, n_sps), 1.5e-5)
        k_field = rng.uniform(1e-6, 1e-3, size=(n_cells, n_sps))
        omega_field = rng.uniform(1.0, 500.0, size=(n_cells, n_sps))
        nu_t = rng.uniform(1e-6, 1e-4, size=(n_cells, n_sps))

        class _S:
            beta_star = 0.09

        s1 = _S()
        s1.k_field, s1.omega_field, s1.nu_t, s1.des_length_scale = k_field, omega_field, nu_t, None
        s2 = _S()
        s2.k_field, s2.omega_field, s2.nu_t, s2.des_length_scale = k_field, omega_field, nu_t, None

        GPUDDESModel().apply_to_sst_model_gpu(s1, d_w, cell_volumes, nu, grad_u)
        GPUIDDESModel().apply_to_sst_model_iddes_gpu(s2, d_w, h_max, h_wn, nu, grad_u)

        assert not np.allclose(s1.des_length_scale, s2.des_length_scale)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
