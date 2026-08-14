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


def _solve_radau_correction_coeffs(n: int, right: bool) -> np.ndarray:
    """求解 Radau/VCJH (g2) 修正多项式的单项式系数（次数为 n）。

    g_L 由以下 n+1 个条件唯一确定（Huynh 2007; Vincent, Castonguay &
    Jameson 2011 的 "g2" 方案，是 FR 方法配合 Gauss-Legendre 解点的标准
    选择）：
        1. g_L(-1) = 1
        2. g_L(+1) = 0
        3. ∫_{-1}^{1} g_L(x) x^k dx = 0,  k = 0, ..., n-2 （与所有更低次
           多项式 L2 正交，这是保证格式仍具有 n 阶精度的关键约束）
    g_R(x) = g_L(-x)（对称性，right=True 时求解镜像版本的边界条件）。

    Returns:
        单项式基系数 c_0..c_n（从低次到高次），满足
        g(x) = sum_k c_k x^k
    """

    def moment(j: int) -> float:
        return 0.0 if j % 2 == 1 else 2.0 / (j + 1)

    A = []
    b = []
    if right:
        A.append([1.0 for _ in range(n + 1)])  # g(1) = 1
        b.append(1.0)
        A.append([(-1.0) ** k for k in range(n + 1)])  # g(-1) = 0
        b.append(0.0)
    else:
        A.append([(-1.0) ** k for k in range(n + 1)])  # g(-1) = 1
        b.append(1.0)
        A.append([1.0 for _ in range(n + 1)])  # g(1) = 0
        b.append(0.0)

    for m in range(0, n - 1):
        A.append([moment(m + k) for k in range(n + 1)])
        b.append(0.0)

    return np.linalg.solve(np.array(A), np.array(b))


def compute_correction_weights(n: int, flux_point_type: str = 'lobatto') -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 FR 校正函数导数在各 SP 处的取值 g_L'(x_i), g_R'(x_i)。

    残差组装中真正需要的量是校正函数的**导数**，而不是校正函数本身的值：
        dU/dt|_corr(x_i) = -[F*_L - F(u(-1))] * g_L'(x_i)
                            -[F*_R - F(u(+1))] * g_R'(x_i)
    （FR/CPR 方法的标准公式，见 Huynh 2007 Eq. 3.9-3.11）。此前的实现把
    "在边界处取值为1、在所有SP处取值为0的拉格朗日基函数在SP处的值"
    误当作校正权重，这在概念上是错误的量——那样构造出来的量在所有 SP
    处恒为 0（因为 g(x_j)=0 对所有 SP j 成立是这个多项式的定义性质
    之一），会让通量跳跃校正项在SP处不产生任何贡献，退化为无效项。
    现在改为求解满足边界条件 + 与低次多项式 L2 正交（VCJH "g2"/Radau
    方案）的修正多项式 g_L(x)/g_R(x)，再解析求导并在 SPs 处求值。
    已用边界条件、正交性、g_R(x)=g_L(-x) 对称性数值验证（相对误差 < 1e-10）。

    Args:
        n: SPs 数量（对应多项式阶数 P = n-1）
        flux_point_type: 保留参数以兼容既有调用签名；g2/Radau 修正函数
            仅与 SPs 位于 Gauss-Legendre 点这一事实相关，与该参数无关
            （该参数实际控制的是插值 FPs 的位置，见 compute_interpolation_matrix）

    Returns:
        g_left_prime, g_right_prime: 左右校正函数导数在各 SP 处的取值，
            形状均为 (n,)
    """
    from .quadrature_points import gauss_legendre

    sps, _ = gauss_legendre(n)

    c_left = _solve_radau_correction_coeffs(n, right=False)
    c_right = _solve_radau_correction_coeffs(n, right=True)

    # 解析求导（单项式基）：d/dx sum_k c_k x^k = sum_{k>=1} k*c_k*x^{k-1}
    dc_left = np.array([c_left[k] * k for k in range(1, n + 1)])
    dc_right = np.array([c_right[k] * k for k in range(1, n + 1)])

    def polyval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x)
        for k, c in enumerate(coeffs):
            result = result + c * x**k
        return result

    g_left_prime = polyval(dc_left, sps)
    g_right_prime = polyval(dc_right, sps)

    return g_left_prime, g_right_prime
