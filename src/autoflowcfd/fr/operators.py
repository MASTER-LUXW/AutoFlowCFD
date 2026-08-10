"""
AutoFlowCFD V2.0 - 矩阵算子生成器

本模块负责生成 Flux Reconstruction 方法所需的各类矩阵算子。
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
        basis_left = 1.0
        basis_right = 1.0
        
        for j in range(n):
            if i != j:
                # 对 g_L：在 x=-1 处值为1，在所有SPs处值为0
                basis_left *= (-1 - sps[j]) / (sps[i] - sps[j])
                
                # 对 g_R：在 x=+1 处值为1，在所有SPs处值为0
                basis_right *= (1 - sps[j]) / (sps[i] - sps[j])
        
        # 还需要考虑边界点的影响
        denom_left = 1.0
        denom_right = 1.0
        for j in range(n):
            denom_left *= (-1 - sps[j])
            denom_right *= (1 - sps[j])
        
        # 最终值
        g_left[i] = basis_left / denom_left if abs(denom_left) > 1e-10 else 0.0
        g_right[i] = basis_right / denom_right if abs(denom_right) > 1e-10 else 0.0
    
    # 归一化：确保在校正过程中保持守恒
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

"""
AutoFlowCFD V2.0 - 求积点生成器

本模块负责生成 Flux Reconstruction 方法所需的各种求积点集。
"""

import numpy as np
from typing import Tuple


def gauss_legendre(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 1D Gauss-Legendre 求积点和权重。
    
    Args:
        n: 点数（对应多项式阶数 P = n-1）
        
    Returns:
        points: 求积点坐标 (-1, 1)
        weights: 求积权重
    """
    points, weights = np.polynomial.legendre.leggauss(n)
    return points, weights


def gauss_lobatto(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 1D Gauss-Lobatto 求积点和权重（包含端点 -1 和 1）。
    
    Args:
        n: 点数
        
    Returns:
        points: 求积点坐标 [-1, 1]
        weights: 求积权重
    """
    if n < 2:
        raise ValueError("Gauss-Lobatto requires at least 2 points")
    
    # 使用 Chebyshev 节点作为初值进行 Newton-Raphson 迭代
    points = np.cos(np.pi * np.arange(n) / (n - 1))
    
    for _ in range(20):
        p_val = np.polynomial.legendre.Legendre.basis(n - 1)(points)
        dp_val = np.polynomial.legendre.Legendre.basis(n - 1).deriv()(points)
        
        delta = p_val / dp_val
        points -= delta
        
        if np.max(np.abs(delta)) < 1e-14:
            break
            
    points.sort()
    
    # 确保端点精确为 -1 和 1
    points[0] = -1.0
    points[-1] = 1.0
    
    # 计算 Lobatto 权重
    weights = np.zeros(n)
    P_n_minus_1 = np.polynomial.legendre.Legendre.basis(n - 1)(points)
    for i in range(n):
        weights[i] = 2.0 / (n * (n - 1) * P_n_minus_1[i]**2)
        
    return points, weights


def generate_tensor_product_points(n: int, point_type: str = 'legendre') -> np.ndarray:
    """
    生成3D张量积点集。
    
    Args:
        n: 每方向点数
        point_type: 点集类型 ('legendre' 或 'lobatto')
        
    Returns:
        points_3d: 3D点集坐标，形状 (n^3, 3)
    """
    # 生成1D点集
    if point_type == 'legendre':
        points_1d, _ = gauss_legendre(n)
    elif point_type == 'lobatto':
        points_1d, _ = gauss_lobatto(n)
    else:
        raise ValueError(f"Unknown point type: {point_type}")
    
    # 构造3D张量积网格
    xx, yy, zz = np.meshgrid(points_1d, points_1d, points_1d, indexing='ij')
    
    # 展平为 (n^3, 3) 数组
    points_3d = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    
    return points_3d
"""
AutoFlowCFD V2.0 - FR 算子生成器主接口

本模块是 Flux Reconstruction 方法算子生成的统一入口，
整合了点集生成、矩阵算子计算等功能。

核心功能：
1. 统一的FR算子生成接口
2. 算子数据容器定义
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
from .quadrature_points import gauss_legendre, gauss_lobatto
from .matrix_operators import (
    compute_vandermonde,
    compute_diff_matrix_1d,
    compute_diff_matrix_3d,
    compute_interpolation_matrix,
    compute_correction_weights
)


@dataclass
class FROperators:
    """
    FR 算子容器，存储预计算的所有矩阵。
    
    Attributes:
        D_1d: 一维微分矩阵，形状 (n_pts, n_pts)
        D_3d: 三维微分算子，形状 (n_pts^3, n_pts^3, 3)
        L_interp: 插值矩阵 (SPs -> FPs)，形状 (n_fps, n_sps)
        g_left, g_right: 左右校正权重向量，形状 (n_sps,)
    """
    D_1d: np.ndarray
    D_3d: np.ndarray
    L_interp: np.ndarray = None
    g_left: np.ndarray = None
    g_right: np.ndarray = None
    
    def get_operators(self) -> Dict[str, np.ndarray]:
        """返回算子字典，兼容旧接口。"""
        return {
            'diff_matrix': self.D_1d,
            'interp_matrix': self.L_interp,
            'g_left': self.g_left,
            'g_right': self.g_right
        }


def generate_fr_operators(order: int, flux_point_type: str = 'lobatto') -> FROperators:
    """
    生成完整的 FR 算子集合。
    
    Args:
        order: 多项式阶数 P
        flux_point_type: FP 类型 ('lobatto' 或 'radau')
        
    Returns:
        operators: 包含所有预计算算子的 FROperators 对象
    """
    n = order + 1  # SPs 数量
    
    # 1. 生成点集
    sps, _ = gauss_legendre(n)
    
    # 2. 计算一维微分矩阵
    D_1d = compute_diff_matrix_1d(sps)
    
    # 3. 计算三维微分算子
    D_3d = compute_diff_matrix_3d(D_1d)
    
    # 4. 计算插值矩阵
    if flux_point_type == 'lobatto':
        fps, _ = gauss_lobatto(n + 1)
    else:
        fps = np.concatenate([[-1.0], sps, [1.0]])
    
    L_interp = compute_interpolation_matrix(sps, fps)
    
    # 5. 计算校正权重
    g_left, g_right = compute_correction_weights(n, flux_point_type)
    
    return FROperators(
        D_1d=D_1d,
        D_3d=D_3d,
        L_interp=L_interp,
        g_left=g_left,
        g_right=g_right
    )


if __name__ == "__main__":
    # 测试算子生成
    order = 2
    ops = generate_fr_operators(order)
    
    print(f"FR Operators for P={order}:")
    print(f"  D_1d shape: {ops.D_1d.shape}")
    print(f"  D_3d shape: {ops.D_3d.shape}")
    print(f"  L_interp shape: {ops.L_interp.shape}")
    print(f"  g_left shape: {ops.g_left.shape}")
    print(f"  g_right shape: {ops.g_right.shape}")
