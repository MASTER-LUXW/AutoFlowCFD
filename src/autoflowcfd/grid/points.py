"""
AutoFlowCFD - 高阶求积点集生成器 (V2.0 Foundation)

本模块负责生成标准单元内的 Solution Points (SPs) 和 Flux Points (FPs)。
主要采用 Gauss-Legendre 和 Gauss-Lobatto 求积点，支撑 FR 方法的高阶离散。
"""

import numpy as np
from typing import Tuple


def gauss_legendre(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 1D Gauss-Legendre 求积点和权重。
    
    Args:
        n: 点数 (对应多项式阶数 P = n-1)
        
    Returns:
        points: 求积点坐标 (-1, 1)
        weights: 求积权重
    """
    points, weights = np.polynomial.legendre.leggauss(n)
    return points, weights


def gauss_lobatto(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 1D Gauss-Lobatto 求积点和权重（包含端点 -1 和 1）。
    通常用于 Flux Points (FPs) 以方便界面通量处理。
    """
    if n < 2:
        raise ValueError("Gauss-Lobatto requires at least 2 points")
    
    # 使用 Chebyshev 节点作为初值进行 Newton-Raphson 迭代
    points = np.cos(np.pi * np.arange(n) / (n - 1))
    
    for _ in range(20):
        p_val = np.polynomial.legendre.Legendre.basis(n - 1)(points)
        dp_val = np.polynomial.legendre.Legendre.basis(n - 1).deriv()(points)
        
        # Lobatto 多项式根求解: (1-x^2)P'_{n-1}(x) = 0
        # 内部点满足 P'_{n-1}(x) = 0
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
    生成 3D 张量积点集（用于六面体单元）。
    
    Args:
        n: 每个方向的点数
        point_type: 'legendre' (SPs) 或 'lobatto' (FPs)
        
    Returns:
        points: 形状为 (n^3, 3) 的坐标数组
    """
    if point_type == 'legendre':
        pts_1d, _ = gauss_legendre(n)
    elif point_type == 'lobatto':
        pts_1d, _ = gauss_lobatto(n)
    else:
        raise ValueError(f"Unsupported point type: {point_type}")
    
    xx, yy, zz = np.meshgrid(pts_1d, pts_1d, pts_1d, indexing='ij')
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    
    return points