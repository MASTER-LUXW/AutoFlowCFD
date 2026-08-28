"""
AutoFlowCFD V2.0 - GPU 版物理通量计算（欧拉 + 粘性）

与 core/fr_flux_kernels_pointwise.py 对应的 CuPy 向量化版本。
用 CuPy 的逐元素运算替代 numba @njit 的逐点循环，底层自动走 cuBLAS/cuDA。

包含：
- euler_physical_flux_gpu: 欧拉物理通量张量 F_i(Q)
- viscous_physical_flux_gpu: 粘性物理通量张量 G_i(Q, grad_vel, grad_T)
- conserved_to_primitive_gpu: 守恒变量 → 原始变量
- primitive_to_conserved_gpu: 原始变量 → 守恒变量
"""

import numpy as np
from autoflowcfd.core.gpu import get_cupy

GAMMA = 1.4
R_AIR = 287.0


def conserved_to_primitive_gpu(U):
    """GPU 版守恒变量→原始变量。U=(rho,rho*u,rho*v,rho*w,rho*E) -> Q=(rho,u,v,w,p)。

    Args:
        U: CuPy 数组 (..., 5+)

    Returns:
        Q: CuPy 数组 (..., 5)
    """
    cp = get_cupy()
    rho = cp.maximum(U[..., 0], 1e-10)
    u = U[..., 1] / rho
    v = U[..., 2] / rho
    w = U[..., 3] / rho
    E = U[..., 4] / rho
    ke = 0.5 * (u**2 + v**2 + w**2)
    p = (GAMMA - 1.0) * rho * (E - ke)
    p = cp.maximum(p, 10.0)
    return cp.stack([rho, u, v, w, p], axis=-1)


def primitive_to_conserved_gpu(Q):
    """GPU 版原始变量→守恒变量。Q=(rho,u,v,w,p) -> U=(rho,rho*u,rho*v,rho*w,rho*E)。

    Args:
        Q: CuPy 数组 (..., 5)

    Returns:
        U: CuPy 数组 (..., 5)
    """
    cp = get_cupy()
    rho, u, v, w, p = Q[..., 0], Q[..., 1], Q[..., 2], Q[..., 3], Q[..., 4]
    rho_safe = cp.maximum(rho, 1e-10)
    ke = 0.5 * (u**2 + v**2 + w**2)
    e_internal = p / ((GAMMA - 1.0) * rho_safe)
    E = e_internal + ke
    return cp.stack([rho, rho * u, rho * v, rho * w, rho * E], axis=-1)


def euler_physical_flux_gpu(Q):
    """GPU 版欧拉物理通量张量 F_i(Q)。

    与 core/fr_flux_kernels_pointwise.py::euler_physical_flux_batch 公式一致。

    Args:
        Q: CuPy 数组 (N, 5)，(rho, u, v, w, p)

    Returns:
        F: CuPy 数组 (N, 3, 5)
    """
    cp = get_cupy()
    rho = Q[..., 0]
    u = Q[..., 1]
    v = Q[..., 2]
    w = Q[..., 3]
    p = Q[..., 4]
    rho_safe = cp.maximum(rho, 1e-10)

    ke = 0.5 * (u**2 + v**2 + w**2)
    e_internal = p / ((GAMMA - 1.0) * rho_safe)
    rhoE = rho * (e_internal + ke)
    H = (rhoE + p) / rho_safe

    # 质量通量
    mf0 = rho * u
    mf1 = rho * v
    mf2 = rho * w

    N = Q.shape[0]
    F = cp.zeros((N, 3, 5), dtype=cp.float64)

    # x-direction
    F[:, 0, 0] = mf0
    F[:, 0, 1] = mf0 * u + p
    F[:, 0, 2] = mf0 * v
    F[:, 0, 3] = mf0 * w
    F[:, 0, 4] = rho * H * u

    # y-direction
    F[:, 1, 0] = mf1
    F[:, 1, 1] = mf1 * u
    F[:, 1, 2] = mf1 * v + p
    F[:, 1, 3] = mf1 * w
    F[:, 1, 4] = rho * H * v

    # z-direction
    F[:, 2, 0] = mf2
    F[:, 2, 1] = mf2 * u
    F[:, 2, 2] = mf2 * v
    F[:, 2, 3] = mf2 * w + p
    F[:, 2, 4] = rho * H * w

    return F


def viscous_physical_flux_gpu(Q, grad_vel, grad_T, mu, Pr, mu_t=None, Pr_t=0.9):
    """GPU 版粘性物理通量张量 G_i(Q, grad_vel, grad_T)。

    与 core/fr_flux_kernels_pointwise.py::viscous_physical_flux_batch 公式一致。
    Boussinesq 假设：mu_total = mu + mu_t。

    Args:
        Q: CuPy 数组 (N, 5)，(rho, u, v, w, p)
        grad_vel: CuPy 数组 (N, 3, 3)，速度梯度 grad_vel[:,i,j] = du_i/dx_j
        grad_T: CuPy 数组 (N, 3)，温度梯度
        mu: 分子动力粘度（标量）
        Pr: 分子普朗特数（标量）
        mu_t: CuPy 数组 (N,) 或标量，湍流涡粘度（默认 0）
        Pr_t: 湍流普朗特数（默认 0.9）

    Returns:
        G: CuPy 数组 (N, 3, 5)
    """
    cp = get_cupy()
    N = Q.shape[0]

    rho = Q[..., 0]
    u = Q[..., 1]
    v = Q[..., 2]
    w = Q[..., 3]

    # 有效粘度
    if mu_t is None:
        mu_total = mu
    elif np.isscalar(mu_t):
        mu_total = mu + mu_t
    else:
        mu_total = mu + mu_t

    # 热导率：k = mu * Cp / Pr（分子）+ mu_t * Cp / Pr_t（湍流）
    Cp = GAMMA * R_AIR / (GAMMA - 1.0)
    if np.isscalar(mu_total):
        k_eff = mu_total * Cp / Pr if np.isscalar(Pr) else mu_total * Cp / Pr
    else:
        # mu_total 是数组时，mu 部分用 Pr，mu_t 部分用 Pr_t
        k_molecular = mu * Cp / Pr
        k_turbulent = mu_t * Cp / Pr_t
        k_eff = k_molecular + k_turbulent

    # 速度梯度分量
    dudx = grad_vel[:, 0, 0]
    dudy = grad_vel[:, 0, 1]
    dudz = grad_vel[:, 0, 2]
    dvdx = grad_vel[:, 1, 0]
    dvdy = grad_vel[:, 1, 1]
    dvdz = grad_vel[:, 1, 2]
    dwdx = grad_vel[:, 2, 0]
    dwdy = grad_vel[:, 2, 1]
    dwdz = grad_vel[:, 2, 2]

    # 散度
    div_vel = dudx + dvdy + dwdz

    # 应力张量（Boussinesq 假设）
    # tau_ij = mu_total * (du_i/dx_j + du_j/dx_i) - 2/3 * mu_total * div(V) * delta_ij
    tau_xx = mu_total * (2.0 * dudx - (2.0/3.0) * div_vel)
    tau_yy = mu_total * (2.0 * dvdy - (2.0/3.0) * div_vel)
    tau_zz = mu_total * (2.0 * dwdz - (2.0/3.0) * div_vel)
    tau_xy = mu_total * (dudy + dvdx)
    tau_xz = mu_total * (dudz + dwdx)
    tau_yz = mu_total * (dvdz + dwdy)

    # 温度梯度分量
    dTdx = grad_T[:, 0]
    dTdy = grad_T[:, 1]
    dTdz = grad_T[:, 2]

    G = cp.zeros((N, 3, 5), dtype=cp.float64)

    # x-direction: G_0
    G[:, 0, 0] = 0.0
    G[:, 0, 1] = tau_xx
    G[:, 0, 2] = tau_xy
    G[:, 0, 3] = tau_xz
    G[:, 0, 4] = (u * tau_xx + v * tau_xy + w * tau_xz) + k_eff * dTdx

    # y-direction: G_1
    G[:, 1, 0] = 0.0
    G[:, 1, 1] = tau_xy
    G[:, 1, 2] = tau_yy
    G[:, 1, 3] = tau_yz
    G[:, 1, 4] = (u * tau_xy + v * tau_yy + w * tau_yz) + k_eff * dTdy

    # z-direction: G_2
    G[:, 2, 0] = 0.0
    G[:, 2, 1] = tau_xz
    G[:, 2, 2] = tau_yz
    G[:, 2, 3] = tau_zz
    G[:, 2, 4] = (u * tau_xz + v * tau_yz + w * tau_zz) + k_eff * dTdz

    return G
