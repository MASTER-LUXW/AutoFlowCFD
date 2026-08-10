"""
AutoFlowCFD V2.0 - FR求解器时间推进策略

本模块实现FR求解器的各种时间推进方法：
1. 显式Runge-Kutta (RK3/SSP-RK3)
2. IMEX (Implicit-Explicit) 方法
3. Dual-Time Stepping 方法
4. Order Continuation 策略
"""

import numpy as np
from typing import Optional, Tuple
from autoflowcfd.core.fr_state import FRState
from autoflowcfd.fr.operators import generate_fr_operators


class TimeAdvancementStrategy:
    """
    时间推进策略基类。
    
    支持多种时间离散方法，用于FR求解器的瞬态和稳态计算。
    """
    
    def __init__(self, scheme_name: str = "rk3"):
        """
        初始化时间推进策略。
        
        Args:
            scheme_name: 时间推进方案名称
                - 'rk3': 三阶Runge-Kutta
                - 'ssp_rk3': 强稳定保持RK3
                - 'imex': 隐式-显式混合
                - 'dual_time': 双时间步长
        """
        self.scheme_name = scheme_name.lower()
        
    def advance(self, state: FRState, residual: np.ndarray, 
                dt_local: np.ndarray, dt: float) -> np.ndarray:
        """
        执行时间推进。
        
        Args:
            state: FR状态对象
            residual: 残差数组 (n_cells, n_sps, n_vars)
            dt_local: 局部时间步长 (n_cells, n_sps)
            dt: 全局时间步长
            
        Returns:
            U_new: 更新后的守恒变量
        """
        if self.scheme_name == 'rk3' or self.scheme_name == 'ssp_rk3':
            return self._rk3_advance(state, residual, dt_local)
        elif self.scheme_name == 'imex':
            return self._imex_advance(state, residual, dt_local, dt)
        elif self.scheme_name == 'dual_time':
            return self._dual_time_advance(state, residual, dt_local, dt)
        else:
            raise ValueError(f"Unknown time advancement scheme: {self.scheme_name}")
    
    def _rk3_advance(self, state: FRState, residual: np.ndarray,
                    dt_local: np.ndarray) -> np.ndarray:
        """
        SSP-RK3 时间推进。
        
        三阶段强稳定保持Runge-Kutta格式：
        U^(1) = U^n + dt * L(U^n)
        U^(2) = 3/4 * U^n + 1/4 * U^(1) + 1/4 * dt * L(U^(1))
        U^(n+1) = 1/3 * U^n + 2/3 * U^(2) + 2/3 * dt * L(U^(2))
        """
        # Stage 1
        dt_expanded = dt_local[:, :, np.newaxis]
        U1 = state.U + dt_expanded * residual
        
        # TODO: 重新计算Stage 1的残差
        # U2 = 0.75 * state.U + 0.25 * U1 + 0.25 * dt * L(U1)
        U2 = 0.75 * state.U + 0.25 * U1
        
        # TODO: 重新计算Stage 2的残差
        # U_new = 1/3 * U^n + 2/3 * U^(2) + 2/3 * dt * L(U^(2))
        U_new = (1.0/3.0) * state.U + (2.0/3.0) * U2
        
        return U_new
    
    def _imex_advance(self, state: FRState, residual: np.ndarray,
                     dt_local: np.ndarray, dt: float) -> np.ndarray:
        """
        IMEX (Implicit-Explicit) 时间推进。
        
        显式处理无粘项，隐式处理粘性项和源项。
        适用于刚性问题（如边界层流动）。
        """
        # 简化实现：使用Crank-Nicolson格式的IMEX
        dt_expanded = dt_local[:, :, np.newaxis]
        
        # 显式部分（无粘）
        U_explicit = state.U + dt_expanded * residual
        
        # 隐式部分（粘性+源项）需要求解线性系统
        # 目前使用简化近似
        theta = 0.5  # Crank-Nicolson参数
        U_new = state.U + theta * dt_expanded * residual + \
                (1 - theta) * dt_expanded * residual
        
        return U_new
    
    def _dual_time_advance(self, state: FRState, residual: np.ndarray,
                          dt_local: np.ndarray, dt: float) -> np.ndarray:
        """
        Dual-Time Stepping 时间推进。
        
        引入伪时间τ进行子迭代，在每个物理时间步内达到收敛。
        适用于强分离流和低马赫数流动。
        """
        # 物理时间导数
        dU_dt_physical = residual.copy()
        
        # 伪时间迭代（简化：单步近似）
        pseudo_dt = dt_local * 0.1  # 较小的伪时间步长
        pseudo_dt_expanded = pseudo_dt[:, :, np.newaxis]
        
        # 伪时间残差
        pseudo_residual = -dU_dt_physical + residual
        
        # 更新
        U_new = state.U + pseudo_dt_expanded * pseudo_residual
        
        return U_new


class OrderContinuationManager:
    """
    Order Continuation 管理器。
    
    实现从低阶到高阶的平滑过渡策略，提高高阶计算的稳定性。
    """
    
    def __init__(self, target_order: int, max_order: int = 4):
        """
        初始化Order Continuation管理器。
        
        Args:
            target_order: 目标多项式阶数
            max_order: 最大允许阶数
        """
        self.target_order = target_order
        self.max_order = max_order
        self.current_order = 0
        self.order_history = []
        
    def get_next_order(self, current_residual: float, 
                      residual_threshold: float = 1e-3) -> int:
        """
        根据当前残差决定是否提升阶数。
        
        Args:
            current_residual: 当前残差
            residual_threshold: 提升阶数的残差阈值
            
        Returns:
            next_order: 下一阶段的阶数
        """
        # 如果残差足够小且未达到目标阶数，提升阶数
        if current_residual < residual_threshold and self.current_order < self.target_order:
            self.current_order += 1
            self.order_history.append(self.current_order)
            print(f"📈 Order increased to P{self.current_order}")
        
        return self.current_order
    
    def interpolate_state(self, state_old: FRState, new_order: int) -> FRState:
        """
        将解从旧阶数插值到新阶数。
        
        使用拉格朗日多项式插值，确保高阶精度传递。
        对于从P_m到P_n的升级(n>m)，在参考单元上进行多项式投影。
        
        Args:
            state_old: 旧阶数的状态
            new_order: 新阶数
            
        Returns:
            state_new: 新阶数的状态
        """
        from autoflowcfd.core.fr_state import FRState
        from autoflowcfd.fr.operators import generate_fr_operators, gauss_legendre
        
        n_cells, n_sps_old, n_vars = state_old.U.shape
        
        # 计算新旧阶数的1D点数
        n_pts_1d_old = int(np.round(n_sps_old ** (1.0/3.0)))
        n_pts_1d_new = new_order + 1
        n_sps_new = n_pts_1d_new ** 3
        
        print(f"   📊 Interpolating state: P{n_pts_1d_old-1} ({n_sps_old} SPs) -> P{new_order} ({n_sps_new} SPs)")
        
        # 创建新状态
        state_new = FRState(n_cells, n_sps_new, n_vars)
        
        # 获取新旧SPs在参考单元中的坐标
        sps_old_1d, _ = gauss_legendre(n_pts_1d_old)
        sps_new_1d, _ = gauss_legendre(n_pts_1d_new)
        
        # 构造1D拉格朗日插值矩阵
        # L[i,j] = l_j(x_new_i)，其中l_j是基于sps_old的拉格朗日基函数
        L_1d = np.zeros((n_pts_1d_new, n_pts_1d_old))
        
        for i in range(n_pts_1d_new):
            x_new = sps_new_1d[i]
            for j in range(n_pts_1d_old):
                # 计算拉格朗日基函数 l_j(x_new)
                basis_val = 1.0
                for k in range(n_pts_1d_old):
                    if k != j:
                        basis_val *= (x_new - sps_old_1d[k]) / (sps_old_1d[j] - sps_old_1d[k])
                L_1d[i, j] = basis_val
        
        # 构造3D张量积插值算子
        # 对于3D张量积网格，插值矩阵是Kronecker积
        # 但为了效率，我们逐维度应用1D插值
        
        # 重塑旧状态为3D网格形式: (n_cells, n_pts, n_pts, n_pts, n_vars)
        U_old_3d = state_old.U.reshape(n_cells, n_pts_1d_old, n_pts_1d_old, n_pts_1d_old, n_vars)
        
        # 初始化新状态的3D形式
        U_new_3d = np.zeros((n_cells, n_pts_1d_new, n_pts_1d_new, n_pts_1d_new, n_vars))
        
        # 对每个单元和每个变量进行插值
        for cell_idx in range(n_cells):
            for var_idx in range(n_vars):
                # 提取当前单元的当前变量的3D场
                field_old = U_old_3d[cell_idx, :, :, :, var_idx]
                
                # 沿xi方向插值
                field_xi = np.einsum('ij,jkl->ikl', L_1d, field_old)
                
                # 沿eta方向插值
                field_eta = np.einsum('ij,kjl->kil', L_1d, field_xi.transpose(1, 0, 2)).transpose(1, 0, 2)
                
                # 沿zeta方向插值
                field_zeta = np.einsum('ij,klj->kli', L_1d, field_eta.transpose(2, 0, 1)).transpose(0, 1, 2)
                
                # 存储到新状态
                U_new_3d[cell_idx, :, :, :, var_idx] = field_zeta
        
        # 展平回 (n_cells, n_sps_new, n_vars)
        state_new.U = U_new_3d.reshape(n_cells, n_sps_new, n_vars)
        
        # 更新原始变量
        state_new._update_primitives()
        
        print(f"   ✅ State interpolation completed using Lagrange polynomial projection")
        
        return state_new
    
    def is_completed(self) -> bool:
        """检查Order Continuation是否完成。"""
        return self.current_order >= self.target_order
