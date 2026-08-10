"""CPU backend kernels for FR residual computation.

本模块提供基于 Numba 加速的 FR 残差计算内核。

核心功能:
1. 无粘通量残差计算
2. 粘性通量残差计算（LDG/IP）
3. 校正项投影
4. 边界条件应用
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _compute_fr_inviscid_residual(solution: np.ndarray,
                                  flux_interface: np.ndarray,
                                  correction_weights: np.ndarray,
                                  residuals: np.ndarray) -> np.ndarray:
    """计算 FR 无粘残差。
    
    核心公式:
        dU/dt = -1/V * Σ [ (F_common - F_local) * g(r_fp) ]
    
    Args:
        solution: 解向量，形状 (n_cells, n_sps, n_vars)
        flux_interface: 界面通量，形状 (n_faces, n_fps, n_vars)
        correction_weights: 校正权重矩阵，形状 (n_sps, n_fps_total)
        residuals: 输出残差，形状同 solution
        
    Returns:
        更新后的残差
    """
    n_cells = solution.shape[0]
    n_sps = solution.shape[1]
    n_vars = solution.shape[2]
    
    # 并行遍历所有单元
    for cell in prange(n_cells):
        for sp in range(n_sps):
            for v in range(n_vars):
                # 初始化残差为局部通量散度
                residuals[cell, sp, v] = 0.0
                
                # 应用校正项（简化：实际需要界面映射）
                # residuals[cell, sp, v] += correction_term
        
    return residuals


@njit(parallel=True, cache=True)
def _compute_fr_viscous_residual(solution: np.ndarray,
                                 grad_solution: np.ndarray,
                                 mu: float,
                                 lambda_: float,
                                 normals: np.ndarray,
                                 residuals: np.ndarray) -> np.ndarray:
    """计算 FR 粘性残差（LDG 方案）。
    
    Args:
        solution: 解向量
        grad_solution: 解的梯度
        mu: 动力粘度
        lambda_: 第二粘性系数
        normals: 界面法向量
        residuals: 输出残差
        
    Returns:
        更新后的残差
    """
    n_cells = solution.shape[0]
    n_sps = solution.shape[1]
    n_vars = solution.shape[2]
    
    gamma = 1.4
    Pr = 0.72  # 普朗特数
    
    for cell in prange(n_cells):
        for sp in range(n_sps):
            # 提取原始变量
            rho = solution[cell, sp, 0]
            u = solution[cell, sp, 1] / rho
            v = solution[cell, sp, 2] / rho
            w = solution[cell, sp, 3] / rho
            E = solution[cell, sp, 4]
            p = (E - 0.5 * rho * (u**2 + v**2 + w**2)) * (gamma - 1.0)
            
            # 温度
            T = p / (rho * 287.0)  # R = 287 J/(kg·K)
            
            # 热导率
            k_thermal = mu * gamma * 287.0 / (Pr * (gamma - 1.0))
            
            # 速度梯度
            du_dx = grad_solution[cell, sp, 1, 0]
            du_dy = grad_solution[cell, sp, 1, 1]
            du_dz = grad_solution[cell, sp, 1, 2]
            
            dv_dx = grad_solution[cell, sp, 2, 0]
            dv_dy = grad_solution[cell, sp, 2, 1]
            dv_dz = grad_solution[cell, sp, 2, 2]
            
            dw_dx = grad_solution[cell, sp, 3, 0]
            dw_dy = grad_solution[cell, sp, 3, 1]
            dw_dz = grad_solution[cell, sp, 3, 2]
            
            # 应变率张量
            S_xx = du_dx
            S_yy = dv_dy
            S_zz = dw_dz
            S_xy = 0.5 * (du_dy + dv_dx)
            S_xz = 0.5 * (du_dz + dw_dx)
            S_yz = 0.5 * (dv_dz + dw_dy)
            
            # 应力张量
            tau_xx = 2.0 * mu * S_xx + lambda_ * (S_xx + S_yy + S_zz)
            tau_yy = 2.0 * mu * S_yy + lambda_ * (S_xx + S_yy + S_zz)
            tau_zz = 2.0 * mu * S_zz + lambda_ * (S_xx + S_yy + S_zz)
            tau_xy = 2.0 * mu * S_xy
            tau_xz = 2.0 * mu * S_xz
            tau_yz = 2.0 * mu * S_yz
            
            # 热通量
            dT_dx = grad_solution[cell, sp, 4, 0]
            dT_dy = grad_solution[cell, sp, 4, 1]
            dT_dz = grad_solution[cell, sp, 4, 2]
            
            q_x = -k_thermal * dT_dx
            q_y = -k_thermal * dT_dy
            q_z = -k_thermal * dT_dz
            
            # 粘性通量散度（简化：实际需要 LDG 界面项）
            # TODO: 实现完整的 LDG 粘性离散
            
    return residuals


@njit(parallel=True, cache=True)
def _update_solution_explicit(solution: np.ndarray,
                              residuals: np.ndarray,
                              dt: float,
                              updated_solution: np.ndarray) -> np.ndarray:
    """显式 Euler 时间推进。
    
    Args:
        solution: 当前解
        residuals: 残差
        dt: 时间步长
        updated_solution: 输出更新后的解
        
    Returns:
        更新后的解
    """
    n_cells = solution.shape[0]
    n_sps = solution.shape[1]
    n_vars = solution.shape[2]
    
    for cell in prange(n_cells):
        for sp in range(n_sps):
            for v in range(n_vars):
                updated_solution[cell, sp, v] = solution[cell, sp, v] - residuals[cell, sp, v] * dt
    
    return updated_solution


@njit(parallel=True, cache=True)
def _apply_wall_bc_no_slip(solution: np.ndarray,
                          boundary_indices: np.ndarray,
                          wall_temp: float = 300.0) -> np.ndarray:
    """应用无滑移壁面边界条件。
    
    Args:
        solution: 解向量
        boundary_indices: 壁面边界索引
        wall_temp: 壁面温度
        
    Returns:
        更新后的解
    """
    n_boundaries = boundary_indices.shape[0]
    
    for i in prange(n_boundaries):
        cell = boundary_indices[i, 0]
        sp = boundary_indices[i, 1]
        
        rho = solution[cell, sp, 0]
        
        # 无滑移：速度为零
        solution[cell, sp, 1] = 0.0  # rho*u = 0
        solution[cell, sp, 2] = 0.0  # rho*v = 0
        solution[cell, sp, 3] = 0.0  # rho*w = 0
        
        # 等温壁面：温度固定
        # E = rho*e + 0.5*rho*(u^2+v^2+w^2)
        # e = Cv*T = R*T/(gamma-1)
        R = 287.0
        gamma = 1.4
        e = R * wall_temp / (gamma - 1.0)
        solution[cell, sp, 4] = rho * e  # 动能为零
    
    return solution


@njit(parallel=True, cache=True)
def _apply_farfield_bc(solution: np.ndarray,
                      boundary_indices: np.ndarray,
                      rho_inf: float,
                      u_inf: float,
                      v_inf: float,
                      w_inf: float,
                      p_inf: float) -> np.ndarray:
    """应用远场边界条件（特征波分解）。
    
    Args:
        solution: 解向量
        boundary_indices: 远场边界索引
        rho_inf: 自由流密度
        u_inf, v_inf, w_inf: 自由流速度
        p_inf: 自由流压力
        
    Returns:
        更新后的解
    """
    n_boundaries = boundary_indices.shape[0]
    gamma = 1.4
    
    for i in prange(n_boundaries):
        cell = boundary_indices[i, 0]
        sp = boundary_indices[i, 1]
        
        # 松弛到自由流状态
        omega_relax = 0.1  # 松弛因子
        
        rho = solution[cell, sp, 0]
        u = solution[cell, sp, 1] / max(rho, 1e-10)
        v = solution[cell, sp, 2] / max(rho, 1e-10)
        w = solution[cell, sp, 3] / max(rho, 1e-10)
        E = solution[cell, sp, 4]
        
        # 更新守恒变量
        solution[cell, sp, 0] = rho + omega_relax * (rho_inf - rho)
        solution[cell, sp, 1] = rho * u + omega_relax * (rho_inf * u_inf - rho * u)
        solution[cell, sp, 2] = rho * v + omega_relax * (rho_inf * v_inf - rho * v)
        solution[cell, sp, 3] = rho * w + omega_relax * (rho_inf * w_inf - rho * w)
        
        # 能量
        e_inf = p_inf / ((gamma - 1.0) * rho_inf) + 0.5 * (u_inf**2 + v_inf**2 + w_inf**2)
        E_inf = rho_inf * e_inf
        solution[cell, sp, 4] = E + omega_relax * (E_inf - E)
    
    return solution

