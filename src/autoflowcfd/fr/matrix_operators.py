"""
AutoFlowCFD V2.0 - FR 矩阵算子生成器

本模块负责生成 Flux Reconstruction 方法所需的各种算子矩阵。

核心功能：
1. Vandermonde 矩阵构造与求逆
2. 一维/三维微分矩阵计算
3. 插值矩阵计算（SPs -> FPs）

**FR 校正函数族（VCJH η_p）已于 2026-09-24 整体移除**：那是一维张量积 FR
形式特有的自由度，原生单纯形/棱柱基的界面项是 DG 提升算子、本身就是 nodal
DG，没有这个参数可选。完整论证、两种修正函数的闭式解/线性系统公式与文献
引用保留在 `ProjectFiles/V2.0/27_FR修正函数族在原生基下不适用-flux-type
移除.md`，需要时可从那份文档重新实现。
"""

import numpy as np
from typing import Tuple



def compute_vandermonde(x: np.ndarray, n: int) -> np.ndarray:
    """
    构造 Vandermonde 矩阵。
    
    Args:
        x: 点集坐标，形状 (m,)
        n: 多项式阶数
        
    Returns:
        V: Vandermonde 矩阵，形状 (m, n)
    """
    V = np.vander(x, N=n, increasing=True)
    return V


def compute_diff_matrix_1d(points: np.ndarray) -> np.ndarray:
    """
    计算一维微分矩阵 D = V' * V^-1。
    
    Args:
        points: 求积点坐标，形状 (n,)
        
    Returns:
        D: 微分矩阵，形状 (n, n)
    """
    n = len(points)
    
    # 构造 Vandermonde 矩阵及其逆
    V = compute_vandermonde(points, n)
    V_inv = np.linalg.inv(V)
    
    # 计算导数 Vandermonde 矩阵
    dV = np.zeros_like(V)
    for i in range(n):
        for j in range(1, n):
            dV[i, j] = j * points[i]**(j-1)
    
    # 微分矩阵
    D = np.dot(dV, V_inv)
    
    return D


def compute_diff_matrix_3d(D_1d: np.ndarray) -> np.ndarray:
    """
    通过张量积构造三维微分算子。
    
    Args:
        D_1d: 一维微分矩阵，形状 (n, n)
        
    Returns:
        D_3d: 三维微分算子，形状 (n^3, n^3, 3)
              最后一个维度对应 ξ, η, ζ 方向
    """
    n = D_1d.shape[0]
    I = np.eye(n)
    
    # Kronecker 积构造三维算子
    D_xi = np.kron(np.kron(D_1d, I), I)
    D_eta = np.kron(np.kron(I, D_1d), I)
    D_zeta = np.kron(np.kron(I, I), D_1d)
    
    # 堆叠为 (n^3, n^3, 3)
    D_3d = np.stack([D_xi, D_eta, D_zeta], axis=-1)
    
    return D_3d


def compute_lagrange_weights_batch(sps: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """计算一批任意目标点（不要求落在预设的 FPs 网格上）处的 1D Lagrange
    基函数取值，用于在非 SP-网格对齐的位置（如面-面精确匹配点）求值。

    这是 compute_interpolation_matrix 的推广：后者只能算固定的一组 fps，
    这里 targets 可以是任意实数（包括 SPs 网格之外、非结构化的一批点）。

    Args:
        sps: Solution Points 坐标，形状 (n_sps,)
        targets: 任意目标点坐标，形状 (n_targets,)

    Returns:
        L: 形状 (n_targets, n_sps)，使得 value_at_targets = L @ values_at_sps
    """
    n_sps = len(sps)
    V_sps = compute_vandermonde(sps, n_sps)
    V_sps_inv = np.linalg.inv(V_sps)
    V_targets = compute_vandermonde(targets, n_sps)
    return V_targets @ V_sps_inv


def compute_interpolation_matrix(sps: np.ndarray, fps: np.ndarray) -> np.ndarray:
    """
    计算从 Solution Points 到 Flux Points 的插值矩阵。
    
    Args:
        sps: Solution Points 坐标，形状 (n_sps,)
        fps: Flux Points 坐标，形状 (n_fps,)
        
    Returns:
        L: 插值矩阵，形状 (n_fps, n_sps)
           使得 u_fps = L @ u_sps
    """
    n_sps = len(sps)
    n_fps = len(fps)
    
    # 构造 Vandermonde 矩阵
    V_sps = compute_vandermonde(sps, n_sps)
    V_sps_inv = np.linalg.inv(V_sps)
    
    # 在 FPS 位置评估拉格朗日基函数
    V_fps = compute_vandermonde(fps, n_sps)
    
    # 插值矩阵
    L = np.dot(V_fps, V_sps_inv)
    
    return L
