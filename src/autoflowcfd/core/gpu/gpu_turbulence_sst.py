"""
AutoFlowCFD V2.0 - SST k-ω 湍流模型源项 GPU 化

将 SSTModelFR 的源项计算完全迁移到 GPU（CuPy），包括：
- 应变率模 |S| 计算
- Blending functions F1, F2
- 涡粘系数 ν_t 计算
- k 方程源项 Sk（产生项 - 耗散项）
- ω 方程源项 S_omega（产生项 - 耗散项 + 交叉扩散项）
- 正性保持限制器

设计：
- 与 CPU 版 SSTModelFR 保持相同接口语义
- 所有数组为 CuPy ndarray，数据常驻 GPU
- 支持 DES 长度尺度替换（与 CPU 版一致）

使用:
    from autoflowcfd.core.gpu.gpu_turbulence_sst import GPUTurbulenceSST
    gpu_sst = GPUTurbulenceSST(n_cells, n_sps, device_id=0)
    Sk, S_omega = gpu_sst.compute_source_terms(Q_gpu, grad_U_gpu, d_wall_gpu, mu, grad_k_gpu, grad_omega_gpu)
"""

import numpy as np
from typing import Optional, Tuple

from autoflowcfd.core.gpu import gpu_available, get_cupy


class GPUTurbulenceSST:
    """GPU 版 SST k-ω 湍流模型。

    所有场变量存储在 GPU 上，源项计算全程在 GPU 完成。

    Attributes:
        n_cells: 单元数
        n_sps: 每单元解点数
        device_id: GPU 设备 ID
        k_field: 湍动能场 (GPU)
        omega_field: 比耗散率场 (GPU)
        nu_t: 涡粘系数场 (GPU)
    """

    def __init__(self, n_cells: int, n_sps: int, device_id: int = 0):
        """初始化 GPU SST 模型。

        Args:
            n_cells: 单元数量
            n_sps: 每单元解点数
            device_id: GPU 设备 ID
        """
        if not gpu_available:
            raise RuntimeError("CuPy required for GPU turbulence model")

        cp = get_cupy()
        self.n_cells = n_cells
        self.n_sps = n_sps
        self.device_id = device_id

        with cp.cuda.Device(device_id):
            # 初始化湍流场（小正值避免除零）
            self.k_field = cp.ones((n_cells, n_sps), dtype=cp.float64) * 1e-6
            self.omega_field = cp.ones((n_cells, n_sps), dtype=cp.float64) * 1.0
            self.nu_t = cp.zeros((n_cells, n_sps), dtype=cp.float64)

        # SST 模型常数（与 CPU 版一致）
        self.sigma_k1 = 0.85
        self.sigma_k2 = 1.0
        self.sigma_w1 = 0.5
        self.sigma_w2 = 0.856
        self.beta1 = 0.075
        self.beta2 = 0.0828
        self.a1 = 0.31
        self.kappa = 0.41
        self.beta_star = 0.09
        self.CD_epsilon = 1e-10

        # DES 长度尺度（可选）
        self.des_length_scale: Optional['cp.ndarray'] = None

    def compute_strain_rate_magnitude_gpu(self, grad_u: 'cp.ndarray') -> 'cp.ndarray':
        """GPU 计算应变率张量模 |S|。

        Args:
            grad_u: 速度梯度 (n_cells, n_sps, 3, 3) CuPy 数组

        Returns:
            S_mag: 应变率模 (n_cells, n_sps)
        """
        cp = get_cupy()
        # S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        S_ij = 0.5 * (grad_u + cp.transpose(grad_u, (0, 1, 3, 2)))
        # |S| = sqrt(2 * S_ij * S_ij)
        S_mag = cp.sqrt(2.0 * cp.sum(S_ij * S_ij, axis=(2, 3)))
        return S_mag

    def compute_blending_F1_gpu(
        self, k: 'cp.ndarray', omega: 'cp.ndarray',
        d: 'cp.ndarray', nu: 'cp.ndarray',
        rho: 'cp.ndarray', CD_kw: 'cp.ndarray'
    ) -> 'cp.ndarray':
        """GPU 计算 SST blending function F1。

        arg1 = min[ max(sqrt(k)/(β*·ω·d), 500·ν/(d²·ω)), 4·ρ·σ_w2·k/(CD_kw·d²) ]
        F1 = tanh(arg1^4)
        """
        cp = get_cupy()
        omega = cp.maximum(omega, 1e-10)
        k = cp.maximum(k, 1e-10)
        d = cp.maximum(d, 1e-10)

        sqrt_k = cp.sqrt(k)
        term1 = sqrt_k / (self.beta_star * omega * d)
        term2 = 500.0 * nu / (d**2 * omega)
        term3 = 4.0 * rho * self.sigma_w2 * k / (cp.maximum(CD_kw, 1e-10) * d**2)

        arg1 = cp.minimum(cp.maximum(term1, term2), term3)
        F1 = cp.tanh(arg1**4)
        return F1

    def compute_blending_F2_gpu(
        self, k: 'cp.ndarray', omega: 'cp.ndarray',
        d: 'cp.ndarray', nu: 'cp.ndarray'
    ) -> 'cp.ndarray':
        """GPU 计算 SST blending function F2。

        arg2 = max(2·sqrt(k)/(β*·ω·d), 500·ν/(d²·ω))
        F2 = tanh(arg2^2)
        """
        cp = get_cupy()
        omega = cp.maximum(omega, 1e-10)
        k = cp.maximum(k, 1e-10)
        d = cp.maximum(d, 1e-10)

        sqrt_k = cp.sqrt(k)
        term1 = 2.0 * sqrt_k / (self.beta_star * omega * d)
        term2 = 500.0 * nu / (d**2 * omega)
        arg2 = cp.maximum(term1, term2)

        F2 = cp.tanh(arg2**2)
        return F2

    def compute_eddy_viscosity_gpu(
        self, k: 'cp.ndarray', omega: 'cp.ndarray',
        rho: 'cp.ndarray', S_mag: 'cp.ndarray', F2: 'cp.ndarray'
    ) -> 'cp.ndarray':
        """GPU 计算涡粘系数 ν_t。

        ν_t = a1 * k / max(a1*ω, F2*S)
        """
        cp = get_cupy()
        k = cp.maximum(k, 1e-10)
        omega = cp.maximum(omega, 1e-10)
        S_mag = cp.maximum(S_mag, 1e-10)

        nu_t = self.a1 * k / cp.maximum(self.a1 * omega, F2 * S_mag)
        nu_t = cp.minimum(nu_t, 1e6)
        return nu_t

    def compute_source_terms_gpu(
        self,
        Q: 'cp.ndarray',
        grad_U: 'cp.ndarray',
        d_wall: 'cp.ndarray',
        mu: float,
        grad_k: 'cp.ndarray',
        grad_omega: 'cp.ndarray',
    ) -> Tuple['cp.ndarray', 'cp.ndarray']:
        """GPU 计算 SST 源项。

        Args:
            Q: 原始变量场 (rho, u, v, w, p) (n_cells, n_sps, 5)
            grad_U: 速度梯度张量 (n_cells, n_sps, 3, 3)
            d_wall: 壁面距离 (n_cells, n_sps)
            mu: 动力粘度
            grad_k: k 梯度 (n_cells, n_sps, 3)
            grad_omega: omega 梯度 (n_cells, n_sps, 3)

        Returns:
            Sk: k 方程源项 (n_cells, n_sps)
            S_omega: omega 方程源项 (n_cells, n_sps)
        """
        cp = get_cupy()

        rho = Q[:, :, 0]
        nu = mu / cp.maximum(rho, 1e-10)

        # 应变率模
        S_mag = self.compute_strain_rate_magnitude_gpu(grad_U)

        # 交叉扩散项
        grad_dot = cp.sum(grad_k * grad_omega, axis=2)
        omega_safe = cp.maximum(self.omega_field, 1e-10)
        CD_kw = cp.maximum(
            2.0 * rho * self.sigma_w2 / omega_safe * grad_dot, 1e-10
        )

        # Blending functions
        F1 = self.compute_blending_F1_gpu(
            self.k_field, self.omega_field, d_wall, nu, rho, CD_kw
        )
        F2 = self.compute_blending_F2_gpu(
            self.k_field, self.omega_field, d_wall, nu
        )

        # Blending 常数
        sigma_k = F1 * self.sigma_k1 + (1.0 - F1) * self.sigma_k2
        sigma_w = F1 * self.sigma_w1 + (1.0 - F1) * self.sigma_w2
        beta = F1 * self.beta1 + (1.0 - F1) * self.beta2

        # 涡粘系数
        self.nu_t = self.compute_eddy_viscosity_gpu(
            self.k_field, self.omega_field, rho, S_mag, F2
        )

        # === k 方程源项 ===
        P_k = self.nu_t * rho * S_mag**2
        P_k = cp.minimum(P_k, 10.0 * self.beta_star * rho * self.k_field * omega_safe)

        if self.des_length_scale is not None:
            D_k = rho * self.k_field**1.5 / cp.maximum(self.des_length_scale, 1e-10)
        else:
            D_k = rho * self.beta_star * self.k_field * self.omega_field

        Sk = P_k - D_k

        # === ω 方程源项 ===
        gamma1 = self.beta1 / self.beta_star - self.sigma_w1 * self.kappa**2 / cp.sqrt(self.beta_star)
        gamma2 = self.beta2 / self.beta_star - self.sigma_w2 * self.kappa**2 / cp.sqrt(self.beta_star)
        gamma = F1 * gamma1 + (1.0 - F1) * gamma2

        P_omega = rho * gamma * S_mag**2
        D_omega = rho * beta * self.omega_field**2
        CD_omega = 2.0 * rho * (1.0 - F1) * self.sigma_w2 / omega_safe * grad_dot

        S_omega = P_omega - D_omega + CD_omega

        return Sk, S_omega

    def apply_positivity_limiter_gpu(
        self, min_k: float = 1e-12, min_omega: float = 1e-12
    ):
        """GPU 正性保持限制器。"""
        cp = get_cupy()
        self.k_field = cp.maximum(self.k_field, min_k)
        self.omega_field = cp.maximum(self.omega_field, min_omega)

    def update_fields_gpu(
        self,
        dt: float,
        Sk: 'cp.ndarray',
        S_omega: 'cp.ndarray',
        transport_k: Optional['cp.ndarray'] = None,
        transport_omega: Optional['cp.ndarray'] = None,
    ):
        """GPU 湍流场时间更新。

        Args:
            dt: 时间步长
            Sk: k 方程源项
            S_omega: omega 方程源项
            transport_k: k 输运残差（可选）
            transport_omega: omega 输运残差（可选）
        """
        cp = get_cupy()
        dk_total = Sk.copy()
        domega_total = S_omega.copy()

        if transport_k is not None:
            dk_total += transport_k
        if transport_omega is not None:
            domega_total += transport_omega

        self.k_field += dt * dk_total
        self.omega_field += dt * domega_total

        self.apply_positivity_limiter_gpu()

    def get_nu_t_cpu(self) -> np.ndarray:
        """获取涡粘系数（下载到 CPU）。"""
        cp = get_cupy()
        return cp.asnumpy(self.nu_t)

    def get_fields_cpu(self) -> dict:
        """获取湍流场（下载到 CPU）。"""
        cp = get_cupy()
        return {
            'k': cp.asnumpy(self.k_field),
            'omega': cp.asnumpy(self.omega_field),
            'nu_t': cp.asnumpy(self.nu_t),
        }

    def set_fields_from_cpu(self, k: np.ndarray, omega: np.ndarray):
        """从 CPU 设置湍流场（上传到 GPU）。"""
        cp = get_cupy()
        with cp.cuda.Device(self.device_id):
            self.k_field = cp.asarray(k)
            self.omega_field = cp.asarray(omega)

    def cleanup(self):
        """释放 GPU 资源。"""
        del self.k_field
        del self.omega_field
        del self.nu_t
        if self.des_length_scale is not None:
            del self.des_length_scale
