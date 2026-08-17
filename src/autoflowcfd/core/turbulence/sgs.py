"""
AutoFlowCFD V2.0 - WALE 亚格子应力模型 (T-06)

本模块实现 Wall-Adapting Local Eddy-viscosity (WALE) 模型，
用于纯 LES 模式下的耗散补偿。同时提供 Smagorinsky-Lilly 备选方案。

核心功能:
1. WALE 模型：自动适应壁面，近壁处涡粘系数趋于零
2. Smagorinsky-Lilly 模型：经典SGS模型
3. 动态模型支持（可选，DynamicSmagorinskyModel 已搬到
   turbulence_sgs_dynamic.py——本文件原有 406 行，超过 400 行硬性拆分
   阈值，该模型是 SmagorinskyModel 的可选变体，与本文件里主要在用的
   WALE/Smagorinsky 模型没有其它耦合，独立成文件最清晰）
4. 亚格子应力张量计算
"""

import numpy as np
from typing import Optional, Tuple


class WALEModel:
    """
    WALE (Wall-Adapting Local Eddy-viscosity) 亚格子模型。
    
    WALE 模型的优势：
    - 在近壁区域自动衰减，无需阻尼函数
    - 对旋转和剪切流动有更好的适应性
    - 基于速度梯度张量的二阶不变量
    
    Attributes:
        c_wale: WALE 模型常数（通常取 0.325-0.5）
        nu_t: 亚格子涡粘系数场
    """

    def __init__(self, c_wale: float = 0.325):
        """
        初始化 WALE 模型。
        
        Args:
            c_wale: WALE 模型常数
        """
        self.c_wale = c_wale
        self.nu_t = None  # 亚格子涡粘系数
        
    def compute_velocity_gradients(self, u_field: np.ndarray, 
                                  v_field: np.ndarray,
                                  w_field: np.ndarray,
                                  grad_operator: np.ndarray) -> np.ndarray:
        """
        计算速度梯度张量 ∂u_i/∂x_j。
        
        Args:
            u_field, v_field, w_field: 速度分量场，形状 (n_cells, n_sps)
            grad_operator: FR 微分算子，形状 (n_sps, n_sps, 3)
            
        Returns:
            grad_u: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)
                   最后两维对应 (i, j)，即 ∂u_i/∂x_j
        """
        n_cells, n_sps = u_field.shape
        
        # 初始化梯度张量
        grad_u = np.zeros((n_cells, n_sps, 3, 3))
        
        # 对每个单元计算梯度
        for cell in range(n_cells):
            # x方向导数
            grad_u[cell, :, 0, 0] = np.dot(grad_operator[:, :, 0], u_field[cell, :])
            grad_u[cell, :, 0, 1] = np.dot(grad_operator[:, :, 1], u_field[cell, :])
            grad_u[cell, :, 0, 2] = np.dot(grad_operator[:, :, 2], u_field[cell, :])
            
            grad_u[cell, :, 1, 0] = np.dot(grad_operator[:, :, 0], v_field[cell, :])
            grad_u[cell, :, 1, 1] = np.dot(grad_operator[:, :, 1], v_field[cell, :])
            grad_u[cell, :, 1, 2] = np.dot(grad_operator[:, :, 2], v_field[cell, :])
            
            grad_u[cell, :, 2, 0] = np.dot(grad_operator[:, :, 0], w_field[cell, :])
            grad_u[cell, :, 2, 1] = np.dot(grad_operator[:, :, 1], w_field[cell, :])
            grad_u[cell, :, 2, 2] = np.dot(grad_operator[:, :, 2], w_field[cell, :])
        
        return grad_u
    
    def compute_strain_and_rotation_tensors(self, grad_u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算应变率张量 S_ij 和旋转率张量 Ω_ij。
        
        Args:
            grad_u: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)
            
        Returns:
            S_ij: 应变率张量，形状同 grad_u
            Omega_ij: 旋转率张量，形状同 grad_u
        """
        # S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        S_ij = 0.5 * (grad_u + np.transpose(grad_u, (0, 1, 3, 2)))
        
        # Ω_ij = 0.5 * (∂u_i/∂x_j - ∂u_j/∂x_i)
        Omega_ij = 0.5 * (grad_u - np.transpose(grad_u, (0, 1, 3, 2)))
        
        return S_ij, Omega_ij
    
    def compute_second_invariants(self, S_ij: np.ndarray, 
                                 Omega_ij: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算应变率和旋转率张量的二阶不变量。
        
        Args:
            S_ij: 应变率张量
            Omega_ij: 旋转率张量
            
        Returns:
            S_sq: S_ij*S_ij，形状 (n_cells, n_sps)
            Omega_sq: Ω_ij*Ω_ij，形状 (n_cells, n_sps)
        """
        # S^2 = S_ij * S_ij (Einstein summation)
        S_sq = np.einsum('nijm,nijm->ni', S_ij, S_ij)
        Omega_sq = np.einsum('nijm,nijm->ni', Omega_ij, Omega_ij)
        
        return S_sq, Omega_sq
    
    def compute_wale_invariant(self, S_ij: np.ndarray, Omega_ij: np.ndarray) -> np.ndarray:
        """
        计算 WALE 模型的核心不变量。
        
        Q_wale = (S_ik * S_kj + Ω_ik * Ω_kj) 的二阶项
        
        更精确的形式：
        L_ij = S_ik * S_kj + Ω_ik * Ω_kj
        L_ij^2 = L_ij * L_ij
        
        Args:
            S_ij: 应变率张量，形状 (n_cells, n_sps, 3, 3)
            Omega_ij: 旋转率张量，形状 (n_cells, n_sps, 3, 3)
            
        Returns:
            L_sq: WALE 不变量，形状 (n_cells, n_sps)
        """
        n_cells, n_sps = S_ij.shape[:2]
        
        # 计算 L_ij = S_ik * S_kj + Ω_ik * Ω_kj
        L_ij = np.zeros_like(S_ij)
        
        for cell in range(n_cells):
            for sp in range(n_sps):
                S_local = S_ij[cell, sp, :, :]
                O_local = Omega_ij[cell, sp, :, :]
                
                # 矩阵乘法
                L_ij[cell, sp, :, :] = np.dot(S_local, S_local) + np.dot(O_local, O_local)
        
        # L^2 = L_ij * L_ij
        L_sq = np.einsum('nijm,nijm->ni', L_ij, L_ij)
        
        return L_sq
    
    def compute_eddy_viscosity(self, grad_u: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """
        计算 WALE 亚格子涡粘系数 ν_t。
        
        核心公式：
        ν_t = (C_wale * Δ)^2 * (L_ij^2)^(3/2) / ((S_ij^2)^(5/2) + (L_ij^2)^(5/4))
        
        其中：
        - S_ij 是应变率张量
        - L_ij = S_ik*S_kj + Ω_ik*Ω_kj
        - Δ 是网格尺度
        
        Args:
            grad_u: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)
            delta: 网格尺度，形状 (n_cells, n_sps)
            
        Returns:
            nu_t: 亚格子涡粘系数，形状 (n_cells, n_sps)
        """
        # 计算应变率和旋转率张量
        S_ij, Omega_ij = self.compute_strain_and_rotation_tensors(grad_u)
        
        # 计算二阶不变量
        S_sq, Omega_sq = self.compute_second_invariants(S_ij, Omega_ij)
        
        # 计算 WALE 不变量 L^2
        L_sq = self.compute_wale_invariant(S_ij, Omega_ij)
        
        # 防止除以零
        S_sq = np.maximum(S_sq, 1e-10)
        L_sq = np.maximum(L_sq, 1e-10)
        delta = np.maximum(delta, 1e-10)
        
        # WALE 核心公式
        numerator = L_sq**(3.0/2.0)
        denominator = S_sq**(5.0/2.0) + L_sq**(5.0/4.0)
        
        nu_t = (self.c_wale * delta)**2 * numerator / np.maximum(denominator, 1e-10)
        
        # 限制最大值以避免数值不稳定
        nu_t = np.minimum(nu_t, 1e-3)
        
        # 存储结果
        self.nu_t = nu_t
        
        return nu_t
    
    def compute_subgrid_stress(self, nu_t: np.ndarray, S_ij: np.ndarray,
                              rho: np.ndarray) -> np.ndarray:
        """
        计算亚格子应力张量 τ_ij。
        
        τ_ij = 2 * ρ * ν_t * S_ij - (2/3) * ρ * k_sgs * δ_ij
        
        简化：忽略各向同性部分
        
        Args:
            nu_t: 亚格子涡粘系数
            S_ij: 应变率张量
            rho: 密度
            
        Returns:
            tau_ij: 亚格子应力张量，形状 (n_cells, n_sps, 3, 3)
        """
        # Boussinesq 假设
        tau_ij = 2.0 * rho[:, :, np.newaxis, np.newaxis] * nu_t[:, :, np.newaxis, np.newaxis] * S_ij
        
        return tau_ij


class SmagorinskyModel:
    """
    Smagorinsky-Lilly 亚格子模型。
    
    经典的 SGS 模型，形式简单但在近壁区域需要阻尼函数。
    
    Attributes:
        c_s: Smagorinsky 常数（通常取 0.1-0.2）
        nu_t: 亚格子涡粘系数
    """
    
    def __init__(self, c_s: float = 0.1):
        """
        初始化 Smagorinsky 模型。
        
        Args:
            c_s: Smagorinsky 常数
        """
        self.c_s = c_s
        self.nu_t = None
    
    def compute_eddy_viscosity(self, grad_u: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """
        计算 Smagorinsky 涡粘系数。
        
        ν_t = (C_s * Δ)^2 * |S|
        
        其中 |S| = sqrt(2 * S_ij * S_ij)
        
        Args:
            grad_u: 速度梯度张量
            delta: 网格尺度
            
        Returns:
            nu_t: 亚格子涡粘系数
        """
        # 计算应变率张量
        S_ij = 0.5 * (grad_u + np.transpose(grad_u, (0, 1, 3, 2)))
        
        # 计算 |S|
        S_sq = np.einsum('nijm,nijm->ni', S_ij, S_ij)
        S_mag = np.sqrt(2.0 * S_sq)
        
        # Smagorinsky 公式
        nu_t = (self.c_s * delta)**2 * S_mag
        
        # 限制
        nu_t = np.minimum(nu_t, 1e-3)
        
        self.nu_t = nu_t
        
        return nu_t
    
    def apply_van_driest_damping(self, nu_t: np.ndarray, y_dist: np.ndarray,
                                nu: float, u_tau: Optional[np.ndarray] = None) -> np.ndarray:
        """
        应用 Van Driest 阻尼函数（近壁修正）。
        
        f_d = 1 - exp(-y+/A+)
        
        Args:
            nu_t: 原始涡粘系数
            y_dist: 到壁面的距离
            nu: 运动粘度
            u_tau: 摩擦速度（可选）。提供时精确计算 y+；
                   未提供时用涡粘系数估计。
            
        Returns:
            nu_t_damped: 阻尼后的涡粘系数
        """
        A_plus = 25.0  # Van Driest 常数
        
        if u_tau is not None:
            # 精确 y+ = y * u_tau / nu
            y_plus = y_dist * np.abs(u_tau) / nu
        else:
            # 无摩擦速度时，用涡粘系数估计特征速度
            nu_t_max = np.max(nu_t)
            u_tau_est = np.sqrt(nu_t_max) if nu_t_max > 0 else 0.1
            y_plus = y_dist * u_tau_est / nu
        
        # 阻尼函数
        f_d = 1.0 - np.exp(-y_plus / A_plus)
        
        nu_t_damped = nu_t * f_d
        
        return nu_t_damped


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    
    n_cells = 50
    n_sps = 8
    
    # 创建测试速度场
    u = np.random.rand(n_cells, n_sps) * 10.0
    v = np.random.rand(n_cells, n_sps) * 10.0
    w = np.random.rand(n_cells, n_sps) * 10.0
    
    # 模拟速度梯度张量
    grad_u = np.random.rand(n_cells, n_sps, 3, 3) * 100.0
    
    # 网格尺度
    delta = np.ones((n_cells, n_sps)) * 0.01
    
    # 测试 WALE 模型
    print("Testing WALE model...")
    wale = WALEModel(c_wale=0.325)
    nu_t_wale = wale.compute_eddy_viscosity(grad_u, delta)
    print(f"WALE nu_t: min={nu_t_wale.min():.6e}, max={nu_t_wale.max():.6e}")
    
    # 测试 Smagorinsky 模型
    print("\nTesting Smagorinsky model...")
    smago = SmagorinskyModel(c_s=0.1)
    nu_t_smago = smago.compute_eddy_viscosity(grad_u, delta)
    print(f"Smagorinsky nu_t: min={nu_t_smago.min():.6e}, max={nu_t_smago.max():.6e}")
    
    print("\nSGS models test completed.")
