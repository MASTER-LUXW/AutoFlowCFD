"""
AutoFlowCFD V2.0 - FR 矩阵算子生成器

本模块负责生成 Flux Reconstruction 方法所需的各种算子矩阵。

核心功能：
1. Vandermonde 矩阵构造与求逆
2. 一维/三维微分矩阵计算
3. 插值矩阵计算（SPs -> FPs）
4. 校正函数权重矩阵
"""

import numpy as np
from typing import Tuple
from .quadrature_points import gauss_legendre, gauss_lobatto


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


def compute_correction_weights(n: int, flux_point_type: str = 'lobatto') -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 FR 校正函数的左右权重向量。
    
    对于Radau多项式，校正函数 g_L 和 g_R 满足：
    - g_L 在左边界 (-1) 为 1，在所有 SPs 处为 0
    - g_R 在右边界 (+1) 为 1，在所有 SPs 处为 0
    
    校正函数用于将界面通量跳跃投影回单元内部：
    δF(x) = δF_L * g_L(x) + δF_R * g_R(x)
    
    Args:
        n: SPs 数量（对应多项式阶数 P = n-1）
        flux_point_type: FP 类型 ('lobatto' 或 'radau')
        
    Returns:
        g_left, g_right: 左右校正权重向量，形状均为 (n,)
                        表示在每个SP处的校正值
    """
    # 获取 SPs 坐标
    from .quadrature_points import gauss_legendre
    sps, _ = gauss_legendre(n)
    
    # 初始化校正权重
    g_left = np.zeros(n)
    g_right = np.zeros(n)
    
    # 使用拉格朗日插值构造校正函数
    # g_L(x) = Π_{j=1}^{n} (x - x_j) / (-1 - x_j)
    # g_R(x) = Π_{j=1}^{n} (x - x_j) / (1 - x_j)
    
    for i in range(n):
        # 计算 g_L 在第 i 个 SP 处的值
        # g_L(x_i) = Π_{j≠i} (x_i - x_j) / (-1 - x_j) * (-1 - x_i) / (-1 - x_i)
        # 但我们需要的是 g_L 作为基函数的系数
        
        # 更准确的方法：构造通过以下点的多项式
        # g_L(-1) = 1, g_L(x_j) = 0 for all j
        # g_R(1) = 1, g_R(x_j) = 0 for all j
        
        # 使用拉格朗日基函数的思想
        # l_j(x) = Π_{k≠j} (x - x_k) / (x_j - x_k)
        
        # g_L 在 SP i 处的值：需要计算从边界-1到SP i的插值
        basis_left = 1.0
        basis_right = 1.0
        
        for j in range(n):
            if i != j:
                # 对 g_L：在 x=-1 处值为1，在所有SPs处值为0
                basis_left *= (-1 - sps[j]) / (sps[i] - sps[j])
                
                # 对 g_R：在 x=+1 处值为1，在所有SPs处值为0
                basis_right *= (1 - sps[j]) / (sps[i] - sps[j])
        
        # 还需要考虑边界点的影响
        # g_L(-1) = 1，所以需要除以 (-1 - sps[i]) 的连乘积
        denom_left = 1.0
        denom_right = 1.0
        for j in range(n):
            denom_left *= (-1 - sps[j])
            denom_right *= (1 - sps[j])
        
        # 最终值
        g_left[i] = basis_left / denom_left if abs(denom_left) > 1e-10 else 0.0
        g_right[i] = basis_right / denom_right if abs(denom_right) > 1e-10 else 0.0
    
    # 归一化：确保在校正过程中保持守恒
    # sum(g_L) 和 sum(g_R) 应该与数值积分权重相关
    from .quadrature_points import gauss_legendre
    _, weights = gauss_legendre(n)
    
    # 加权归一化
    sum_gL = np.sum(g_left * weights)
    sum_gR = np.sum(g_right * weights)
    
    if abs(sum_gL) > 1e-10:
        g_left /= sum_gL
    if abs(sum_gR) > 1e-10:
        g_right /= sum_gR
    
    return g_left, g_right
