"""
AutoFlowCFD V2.0 - FR 点集生成器

本模块负责生成 Flux Reconstruction 方法所需的各种求积点集。

核心功能：
1. 一维 Gauss-Legendre 点集生成
2. 一维 Gauss-Lobatto 点集生成
3. 三维张量积点集生成
"""

from functools import lru_cache
from typing import Tuple

import numpy as np


@lru_cache(maxsize=None)
def _leggauss(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """按点数缓存的 `leggauss`（只读数组）。逐面的几何例程（`native_prism/face.py` 等）对每个面都
    要同一组求积点：cube_demo（72.8 万个面）网格初始化时 `leggauss` 被调用 145 万次、127 s（每次都重解
    一遍伴随矩阵的特征值问题），占网格初始化 398 s 的三分之一。"""
    points, weights = np.polynomial.legendre.leggauss(n)
    points.setflags(write=False)
    weights.setflags(write=False)
    return points, weights


def gauss_legendre(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 1D Gauss-Legendre 求积点和权重。

    Args:
        n: 点数（对应多项式阶数 P = n-1）

    Returns:
        points: 求积点坐标 (-1, 1)
        weights: 求积权重（两者都是只读数组，按点数缓存；要改就先 copy）
    """
    return _leggauss(int(n))


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
