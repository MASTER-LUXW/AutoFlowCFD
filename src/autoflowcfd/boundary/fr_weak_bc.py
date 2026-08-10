"""
AutoFlowCFD V2.0 - FR 弱边界条件处理器 (BD-01)

本模块实现基于惩罚项 (Penalty Term) 的弱边界条件，用于处理 WALL, INLET, OUTLET, FARFIELD。

核心功能：
1. 无滑移壁面边界 (No-slip Wall)
2. 远场边界 (Farfield)
3. 速度/压力入口边界 (Inlet)
4. 压力出口边界 (Outlet)
5. 对称边界 (Symmetry)
"""

import numpy as np


class FRWeakBC:
    """
    FR 弱边界条件计算器。
    
    通过在界面通量中引入惩罚项 τ(U_int - U_bc) 来耦合边界信息。
    惩罚系数τ通常取较大的值以确保边界条件的强施加。
    
    Attributes:
        penalty_coeff: 惩罚系数，控制边界条件施加的强度
    """

    def __init__(self, penalty_coeff: float = 10.0):
        """
        初始化弱边界条件处理器。
        
        Args:
            penalty_coeff: 惩罚系数（默认10.0）
        """
        self.penalty_coeff = penalty_coeff

    def compute_wall_bc_flux(self, u_int: np.ndarray, normal: np.ndarray,
                            is_no_slip: bool = True) -> np.ndarray:
        """
        计算壁面 (Wall) 的通量贡献。
        
        Args:
            u_int: 内部解点的状态，形状 (n_sps, n_vars)
            normal: 壁面法向量，形状 (3,)
            is_no_slip: 是否为无滑移壁面（默认True）
            
        Returns:
            flux_bc: 边界惩罚通量，形状同u_int
        """
        u_bc = u_int.copy()
        
        if is_no_slip:
            # 无滑移壁面：u=v=w=0, dp/dn=0
            u_bc[:, 1:4] = 0.0  # 速度置零
        else:
            # 滑移壁面：法向速度为零
            vel = u_int[:, 1:4] / u_int[:, 0:1]  # 转换为速度
            vel_normal = np.sum(vel * normal[np.newaxis, :], axis=1, keepdims=True)
            vel_tangent = vel - vel_normal * normal[np.newaxis, :]
            u_bc[:, 1:4] = u_int[:, 0:1] * vel_tangent
        
        # 惩罚项
        delta_u = u_int - u_bc
        flux_bc = self.penalty_coeff * delta_u
        
        return flux_bc

    def compute_farfield_bc_flux(self, u_int: np.ndarray, u_free: np.ndarray) -> np.ndarray:
        """
        计算远场 (Farfield) 边界通量。
        
        Args:
            u_int: 内部解点状态
            u_free: 自由来流状态
            
        Returns:
            flux_bc: 边界惩罚通量
        """
        delta_u = u_int - u_free
        return self.penalty_coeff * delta_u

    def compute_inlet_bc_flux(self, u_int: np.ndarray, u_inlet: np.ndarray,
                             normal: np.ndarray) -> np.ndarray:
        """
        计算入口 (Inlet) 边界通量。
        
        Args:
            u_int: 内部解点状态
            u_inlet: 入口指定状态
            normal: 入口法向量（指向计算域外）
            
        Returns:
            flux_bc: 边界惩罚通量
        """
        # 检查流动方向
        vel_int = u_int[:, 1:4] / u_int[:, 0:1]
        un_int = np.sum(vel_int * normal[np.newaxis, :], axis=1)
        
        # 如果流出，使用内部状态；如果流入，使用入口状态
        u_bc = np.where(un_int[:, np.newaxis] > 0, u_int, u_inlet)
        
        delta_u = u_int - u_bc
        return self.penalty_coeff * delta_u

    def compute_outlet_bc_flux(self, u_int: np.ndarray, p_outlet: float,
                              normal: np.newaxis) -> np.ndarray:
        """
        计算出口 (Outlet) 边界通量。
        
        Args:
            u_int: 内部解点状态
            p_outlet: 出口指定压力
            normal: 出口法向量
            
        Returns:
            flux_bc: 边界惩罚通量
        """
        u_bc = u_int.copy()
        
        # 从内部状态提取原始变量
        rho = u_int[:, 0]
        vel = u_int[:, 1:4] / rho[:, np.newaxis]
        ke = 0.5 * np.sum(vel**2, axis=1)
        
        # 设置出口压力，保持其他变量不变
        gamma = 1.4
        e_outlet = p_outlet / ((gamma - 1.0) * rho) + ke
        u_bc[:, 4] = rho * e_outlet
        
        # 检查流动方向
        un_int = np.sum(vel * normal[np.newaxis, :], axis=1)
        
        # 如果流入，使用内部状态；如果流出，使用出口状态
        u_bc_final = np.where(un_int[:, np.newaxis] < 0, u_int, u_bc)
        
        delta_u = u_int - u_bc_final
        return self.penalty_coeff * delta_u

    def compute_symmetry_bc_flux(self, u_int: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """
        计算对称 (Symmetry) 边界通量。
        
        Args:
            u_int: 内部解点状态
            normal: 对称面法向量
            
        Returns:
            flux_bc: 边界惩罚通量
        """
        u_bc = u_int.copy()
        
        # 反射速度分量
        vel = u_int[:, 1:4] / u_int[:, 0:1]
        vel_normal = np.sum(vel * normal[np.newaxis, :], axis=1, keepdims=True)
        vel_reflected = vel - 2.0 * vel_normal * normal[np.newaxis, :]
        
        u_bc[:, 1:4] = u_int[:, 0:1] * vel_reflected
        
        delta_u = u_int - u_bc
        return self.penalty_coeff * delta_u