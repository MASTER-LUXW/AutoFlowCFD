"""
AutoFlowCFD V2.0 - 涡识别准则 (Q-criterion, Lambda2)

本模块实现常见的涡识别方法，用于 LES/DDES 后处理中可视化湍流结构。

核心功能:
1. Q 准则 (Q-criterion): 基于速度梯度张量的第二不变量
2. Lambda2 准则: 基于对称张量 S^2 + Omega^2 的第二大特征值
3. 等值面提取支持
"""

import numpy as np
from typing import Dict, Tuple, Optional
from loguru import logger


def compute_q_criterion(velocity_gradients: np.ndarray) -> np.ndarray:
    """
    计算 Q 准则（Q-criterion）。
    
    Q 准则定义为速度梯度张量的第二不变量：
    Q = 0.5 * (||Omega||^2 - ||S||^2)
    
    其中：
    - S = 0.5 * (grad_U + grad_U^T) 是应变率张量
    - Omega = 0.5 * (grad_U - grad_U^T) 是旋转张量
    - ||.|| 表示 Frobenius 范数
    
    物理意义：
    - Q > 0: 旋转占主导（涡核区域）
    - Q < 0: 应变占主导（剪切层）
    
    Args:
        velocity_gradients: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)
                           grad_U[i, s, j, k] = d(u_j)/d(x_k) at cell i, sp s
    
    Returns:
        q_values: Q 准则值，形状 (n_cells, n_sps)
    """
    n_cells, n_sps = velocity_gradients.shape[:2]
    q_values = np.zeros((n_cells, n_sps))
    
    for i in range(n_cells):
        for s in range(n_sps):
            # 提取该 SP 的速度梯度张量 (3x3)
            grad_u = velocity_gradients[i, s]  # [du/dx, du/dy, du/dz; dv/dx, ...]
            
            # 计算应变率张量 S = 0.5 * (grad_U + grad_U^T)
            S = 0.5 * (grad_u + grad_u.T)
            
            # 计算旋转张量 Omega = 0.5 * (grad_U - grad_U^T)
            Omega = 0.5 * (grad_u - grad_u.T)
            
            # 计算 Frobenius 范数的平方: ||A||^2 = sum(A_ij^2)
            S_norm_sq = np.sum(S**2)
            Omega_norm_sq = np.sum(Omega**2)
            
            # Q = 0.5 * (||Omega||^2 - ||S||^2)
            q_values[i, s] = 0.5 * (Omega_norm_sq - S_norm_sq)
    
    return q_values


def compute_lambda2_criterion(velocity_gradients: np.ndarray) -> np.ndarray:
    """
    计算 Lambda2 准则。
    
    Lambda2 准则基于对称张量 M = S^2 + Omega^2 的特征值。
    将特征值排序 lambda1 <= lambda2 <= lambda3，Lambda2 准则取 lambda2。
    
    物理意义：
    - Lambda2 < 0: 涡核区域（压力极小值）
    - Lambda2 > 0: 非涡区域
    
    Args:
        velocity_gradients: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)
    
    Returns:
        lambda2_values: Lambda2 准则值，形状 (n_cells, n_sps)
    """
    n_cells, n_sps = velocity_gradients.shape[:2]
    lambda2_values = np.zeros((n_cells, n_sps))
    
    for i in range(n_cells):
        for s in range(n_sps):
            grad_u = velocity_gradients[i, s]
            
            # 计算应变率张量和旋转张量
            S = 0.5 * (grad_u + grad_u.T)
            Omega = 0.5 * (grad_u - grad_u.T)
            
            # 计算 M = S^2 + Omega^2
            M = np.dot(S, S) + np.dot(Omega, Omega)
            
            # 计算特征值
            eigenvalues = np.linalg.eigvalsh(M)  # 返回已排序的特征值
            
            # Lambda2 是第二小的特征值
            lambda2_values[i, s] = eigenvalues[1]
    
    return lambda2_values


def extract_vortex_core(q_values: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    提取涡核区域（基于 Q 准则阈值）。
    
    Args:
        q_values: Q 准则值，形状 (n_cells, n_sps)
        threshold: Q 准则阈值，默认 0.0（Q > 0 为涡核）
    
    Returns:
        vortex_mask: 布尔掩码，True 表示涡核区域，形状 (n_cells, n_sps)
    """
    return q_values > threshold


def compute_vorticity_magnitude(velocity_gradients: np.ndarray) -> np.ndarray:
    """
    计算涡量幅值 |omega| = sqrt(2 * ||Omega||^2)。
    
    Args:
        velocity_gradients: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)
    
    Returns:
        vorticity_mag: 涡量幅值，形状 (n_cells, n_sps)
    """
    n_cells, n_sps = velocity_gradients.shape[:2]
    vorticity_mag = np.zeros((n_cells, n_sps))
    
    for i in range(n_cells):
        for s in range(n_sps):
            grad_u = velocity_gradients[i, s]
            Omega = 0.5 * (grad_u - grad_u.T)
            vorticity_mag[i, s] = np.sqrt(2.0 * np.sum(Omega**2))
    
    return vorticity_mag


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    
    # 创建测试数据：Taylor-Green Vortex 简化版
    n_cells = 10
    n_sps = 8
    
    # 模拟速度梯度场（简化的涡旋场）
    velocity_gradients = np.zeros((n_cells, n_sps, 3, 3))
    
    # 在中心区域添加旋转
    for i in range(n_cells):
        for s in range(n_sps):
            # 简化的二维旋转：du/dy = -omega, dv/dx = omega
            omega = 10.0 * np.exp(-((i-5)**2 + (s-4)**2) / 10.0)
            velocity_gradients[i, s, 0, 1] = -omega  # du/dy
            velocity_gradients[i, s, 1, 0] = omega   # dv/dx
    
    # 计算 Q 准则
    q_vals = compute_q_criterion(velocity_gradients)
    print(f"Q 准则统计:")
    print(f"  最小值: {q_vals.min():.6e}")
    print(f"  最大值: {q_vals.max():.6e}")
    print(f"  平均值: {q_vals.mean():.6e}")
    
    # 提取涡核
    vortex_mask = extract_vortex_core(q_vals, threshold=0.0)
    n_vortex_points = np.sum(vortex_mask)
    print(f"\n涡核区域:")
    print(f"  涡核点数: {n_vortex_points} / {n_cells * n_sps}")
    
    # 计算 Lambda2
    lambda2_vals = compute_lambda2_criterion(velocity_gradients)
    print(f"\nLambda2 准则统计:")
    print(f"  最小值: {lambda2_vals.min():.6e}")
    print(f"  最大值: {lambda2_vals.max():.6e}")
