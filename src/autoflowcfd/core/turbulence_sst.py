"""
AutoFlowCFD V2.0 - SST k-omega 湍流模型 FR 离散 (T-01, T-02)

本模块实现在 FR 框架下的 SST 模型输运方程离散，并包含正性保持限制器。

核心功能:
1. 完整的 SST k-ω 模型源项计算
2. 正性保持限制器 (Positivity-preserving Limiter)
3. 跨扩散项处理
4.  blending functions F1, F2
"""

import numpy as np
from typing import Tuple, Optional
from loguru import logger


class SSTModelFR:
    """
    FR 框架下的 SST k-omega 模型处理器。
    
    实现 Menter 的 SST (Shear Stress Transport) 模型，包括:
    - k 输运方程: ∂(ρk)/∂t + ∇·(ρUk) = P_k - D_k + ∇·[(μ+σ_k μ_t)∇k]
    - ω 输运方程: ∂(ρω)/∂t + ∇·(ρUω) = P_ω - D_ω + ∇·[(μ+σ_ω μ_t)∇ω] + CD_ω
    
    Attributes:
        k_field: 湍动能场，存储在 SPs 上，形状 (n_cells, n_sps)
        omega_field: 比耗散率场，存储在 SPs 上，形状 (n_cells, n_sps)
        nu_t: 湍流涡粘系数场，形状 (n_cells, n_sps)
    """

    def __init__(self, n_cells: int, n_sps: int):
        """
        初始化 SST 模型。
        
        Args:
            n_cells: 单元数量
            n_sps: 每单元解点数量
        """
        self.n_cells = n_cells
        self.n_sps = n_sps
        
        # 初始化湍流场（使用小正值避免除零）
        self.k_field = np.ones((n_cells, n_sps)) * 1e-6
        self.omega_field = np.ones((n_cells, n_sps)) * 1.0
        self.nu_t = np.zeros((n_cells, n_sps))
        
        # SST 模型常数
        self.sigma_k1 = 0.85
        self.sigma_k2 = 1.0
        self.sigma_w1 = 0.5
        self.sigma_w2 = 0.856
        self.beta1 = 0.075
        self.beta2 = 0.0828
        self.a1 = 0.31
        self.kappa = 0.41  # Von Karman 常数
        self.beta_star = 0.09
        
        # Blending function 相关常数
        self.CD_epsilon = 1e-10

    def compute_strain_rate_magnitude(self, grad_u: np.ndarray) -> np.ndarray:
        """
        计算应变率张量的模 |S|。
        
        Args:
            grad_u: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)
            
        Returns:
            S_mag: 应变率模，形状 (n_cells, n_sps)
        """
        # S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        S_ij = 0.5 * (grad_u + np.transpose(grad_u, (0, 1, 3, 2)))
        
        # |S| = sqrt(2 * S_ij * S_ij)
        S_mag = np.sqrt(2.0 * np.einsum('nijm,nijm->ni', S_ij, S_ij))
        
        return S_mag

    def compute_blending_function_F1(self, k: np.ndarray, omega: np.ndarray, 
                                     d: np.ndarray, nu: np.ndarray, 
                                     S_mag: np.ndarray) -> np.ndarray:
        """
        计算 SST 模型的 blending function F1。
        
        F1 = tanh(arg1^4), 其中 arg1 基于距离、湍流量和应变率
        
        Args:
            k: 湍动能
            omega: 比耗散率
            d: 到壁面的距离
            nu: 运动粘度
            S_mag: 应变率模
            
        Returns:
            F1: blending function，范围 [0, 1]
        """
        # 防止除零
        omega = np.maximum(omega, 1e-10)
        k = np.maximum(k, 1e-10)
        d = np.maximum(d, 1e-10)
        
        # 计算 arg1
        sqrt_k = np.sqrt(k)
        arg1 = np.minimum(
            np.maximum(
                d / (self.kappa * sqrt_k / np.maximum(omega, 1e-10)),
                500.0 * nu / (d**2 * np.maximum(omega, 1e-10))
            ),
            10.0
        )
        
        F1 = np.tanh(arg1**4)
        
        return F1

    def compute_blending_function_F2(self, k: np.ndarray, omega: np.ndarray, 
                                     d: np.ndarray, S_mag: np.ndarray) -> np.ndarray:
        """
        计算 SST 模型的 blending function F2。
        
        F2 = tanh(arg2^2), 用于粘性应力限制
        
        Args:
            k: 湍动能
            omega: 比耗散率
            d: 到壁面的距离
            S_mag: 应变率模
            
        Returns:
            F2: blending function，范围 [0, 1]
        """
        omega = np.maximum(omega, 1e-10)
        k = np.maximum(k, 1e-10)
        d = np.maximum(d, 1e-10)
        
        sqrt_k = np.sqrt(k)
        arg2 = np.minimum(
            2.0 * d / (self.kappa * sqrt_k / omega),
            10.0
        )
        
        F2 = np.tanh(arg2**2)
        
        return F2

    def compute_eddy_viscosity(self, k: np.ndarray, omega: np.ndarray, 
                              rho: np.ndarray, S_mag: np.ndarray, 
                              F2: np.ndarray) -> np.ndarray:
        """
        计算湍流涡粘系数 ν_t。
        
        ν_t = a1 * k / max(a1*ω, F2*S)
        
        Args:
            k: 湍动能
            omega: 比耗散率
            rho: 密度
            S_mag: 应变率模
            F2: blending function
            
        Returns:
            nu_t: 湍流涡粘系数
        """
        k = np.maximum(k, 1e-10)
        omega = np.maximum(omega, 1e-10)
        S_mag = np.maximum(S_mag, 1e-10)
        
        # Boussinesq 假设下的涡粘系数
        nu_t = self.a1 * k / np.maximum(self.a1 * omega, F2 * S_mag)
        
        # 限制最大值以避免数值不稳定
        nu_t = np.minimum(nu_t, 1e6)
        
        return nu_t

    def compute_source_terms(self, Q: np.ndarray, grad_U: np.ndarray, 
                            d_wall: np.ndarray, mu: float,
                            grad_k: Optional[np.ndarray] = None,
                            grad_omega: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 SST 模型的完整源项。
        
        Args:
            Q: 原始变量场 (rho, u, v, w, p)，形状 (n_cells, n_sps, 5)
            grad_U: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)
            d_wall: 壁面距离场，形状 (n_cells, n_sps)
            mu: 动力粘度
            grad_k: k 的梯度，形状 (n_cells, n_sps, 3)（可选，如未提供则使用简化估计）
            grad_omega: omega 的梯度，形状 (n_cells, n_sps, 3)（可选）
            
        Returns:
            Sk: 湍动能源项，形状 (n_cells, n_sps)
            S_omega: 比耗散率源项，形状 (n_cells, n_sps)
        """
        # 提取流场变量
        rho = Q[:, :, 0]
        u_vel = Q[:, :, 1]
        v_vel = Q[:, :, 2]
        w_vel = Q[:, :, 3]
        
        # 运动粘度
        nu = mu / np.maximum(rho, 1e-10)
        
        # 计算应变率模
        S_mag = self.compute_strain_rate_magnitude(grad_U)
        
        # 计算 blending functions
        F1 = self.compute_blending_function_F1(
            self.k_field, self.omega_field, d_wall, nu, S_mag
        )
        F2 = self.compute_blending_function_F2(
            self.k_field, self.omega_field, d_wall, S_mag
        )
        
        # Blending 常数
        sigma_k = F1 * self.sigma_k1 + (1.0 - F1) * self.sigma_k2
        sigma_w = F1 * self.sigma_w1 + (1.0 - F1) * self.sigma_w2
        beta = F1 * self.beta1 + (1.0 - F1) * self.beta2
        
        # 计算涡粘系数
        self.nu_t = self.compute_eddy_viscosity(
            self.k_field, self.omega_field, rho, S_mag, F2
        )
        
        # === k 方程源项 ===
        # 产生项: P_k = μ_t * S^2
        P_k = self.nu_t * rho * S_mag**2
        
        # 耗散项: D_k = ρ * β* * k * ω
        D_k = rho * self.beta_star * self.k_field * self.omega_field
        
        # k 方程总源项
        Sk = P_k - D_k
        
        # === ω 方程源项 ===
        # 产生项: P_ω = ρ * γ * S^2
        gamma1 = self.beta1 / self.beta_star - self.sigma_w1 * self.kappa**2 / np.sqrt(self.beta_star)
        gamma2 = self.beta2 / self.beta_star - self.sigma_w2 * self.kappa**2 / np.sqrt(self.beta_star)
        gamma = F1 * gamma1 + (1.0 - F1) * gamma2
        
        P_omega = rho * gamma * S_mag**2
        
        # 耗散项: D_ω = ρ * β * ω^2
        D_omega = rho * beta * self.omega_field**2
        
        # 交叉扩散项: CD_ω = 2 * ρ * (1-F1) * σ_w2 / ω * ∇k · ∇ω
        if grad_k is not None and grad_omega is not None:
            # 使用真实的梯度（从FR算子计算）
            # ∇k · ∇ω = sum_i (∂k/∂x_i * ∂ω/∂x_i)
            grad_dot_product = np.sum(grad_k * grad_omega, axis=2)  # 形状: (n_cells, n_sps)
            CD_omega = 2.0 * rho * (1.0 - F1) * self.sigma_w2 / np.maximum(self.omega_field, 1e-10) * \
                       grad_dot_product
            logger.debug(f"Using real gradients for cross-diffusion term")
        else:
            # 回退：使用简化估计（仅用于测试，工业级计算应提供真实梯度）
            logger.warning(
                "Gradients not provided for SST cross-diffusion term. "
                "Using simplified estimate. For industrial calculations, "
                "provide grad_k and grad_omega from FR gradient computation."
            )
            # 基于局部梯度的粗略估计
            grad_k_mag = np.abs(self.k_field) * 10.0  # 假设梯度量级
            grad_omega_mag = np.abs(self.omega_field) * 10.0
            CD_omega = 2.0 * rho * (1.0 - F1) * self.sigma_w2 / np.maximum(self.omega_field, 1e-10) * \
                       grad_k_mag * grad_omega_mag
        
        # ω 方程总源项
        S_omega = P_omega - D_omega + CD_omega
        
        return Sk, S_omega

    def apply_positivity_limiter(self, min_k: float = 1e-12, min_omega: float = 1e-12):
        """
        正性保持限制器 (T-02)：强制 k 和 omega 非负，并在重构过程中嵌入硬约束。
        
        Args:
            min_k: k 的最小允许值
            min_omega: omega 的最小允许值
        """
        # 硬截断确保物理合理性
        self.k_field = np.maximum(self.k_field, min_k)
        self.omega_field = np.maximum(self.omega_field, min_omega)
        
        # 额外检查：限制增长率避免突变
        # 如果某点的值相比邻域过大，进行平滑
        if self.n_cells > 1 and self.n_sps > 1:
            k_mean = np.mean(self.k_field)
            omega_mean = np.mean(self.omega_field)
            
            # 限制异常值（超过均值100倍的点）
            outlier_mask_k = self.k_field > 100 * k_mean
            outlier_mask_omega = self.omega_field > 100 * omega_mean
            
            if np.any(outlier_mask_k):
                self.k_field[outlier_mask_k] = 100 * k_mean
                
            if np.any(outlier_mask_omega):
                self.omega_field[outlier_mask_omega] = 100 * omega_mean

    def update_fields(self, dt: float, Sk: np.ndarray, S_omega: np.ndarray,
                     diff_k: np.ndarray = None, diff_omega: np.ndarray = None):
        """
        执行一个时间步长的湍流场更新。
        
        Args:
            dt: 时间步长
            Sk: 湍动能源项
            S_omega: 比耗散率源项
            diff_k: k 的扩散项（可选）
            diff_omega: omega 的扩散项（可选）
        """
        # 如果提供了扩散项，则包含在更新中
        if diff_k is not None and diff_omega is not None:
            self.k_field += dt * (Sk + diff_k)
            self.omega_field += dt * (S_omega + diff_omega)
        else:
            # 仅源项更新
            self.k_field += dt * Sk
            self.omega_field += dt * S_omega
        
        # 应用正性限制器
        self.apply_positivity_limiter()
        
    def get_turbulent_viscosity(self) -> np.ndarray:
        """
        获取当前的湍流涡粘系数。
        
        Returns:
            nu_t: 涡粘系数场
        """
        return self.nu_t.copy()
