"""
AutoFlowCFD V2.0 - FR 点集生成器

本模块负责生成 Flux Reconstruction 方法所需的各种求积点集。

核心功能：
1. 一维 Gauss-Legendre 点集生成
2. 一维 Gauss-Lobatto 点集生成
3. 三维张量积点集生成
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
