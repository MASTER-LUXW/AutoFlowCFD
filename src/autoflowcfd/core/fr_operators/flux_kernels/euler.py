"""AutoFlowCFD V2.0 - 欧拉物理通量（逐点 / 批量 numba 核；GPU 版 `core/gpu/residual/gpu_flux.py` 公式相同）。

按原始变量 Q = (rho, u, v, w, p) 写成不含除法的多项式：

    F_i = (rho u_i,  rho u_i u_j + p delta_ij,  u_i (gamma/(gamma-1) p + rho |V|^2 / 2))

能量分量即 rho H u_i（rho H = rho E + p）。2026-10-22（I-25）以前经 H = (rho E + p) / max(rho, 1e-10) 求得：代数上
相同，只多一次除法和一个密度下限（正性限制器保证解点密度为正，下限不起作用；解失去正性时它只会掩盖问题）。
"""

import numpy as np

from numba import njit, prange
from autoflowcfd.core.thermo import GAMMA


@njit(cache=True)
def euler_physical_flux_point(Q: np.ndarray) -> np.ndarray:
    """欧拉物理通量，单点版。Q=(rho,u,v,w,p) -> F，形状 (3,5)（公式见模块文档）。"""
    rho = Q[0]
    u = Q[1]
    v = Q[2]
    w = Q[3]
    p = Q[4]
    rho_H = GAMMA / (GAMMA - 1.0) * p + 0.5 * rho * (u * u + v * v + w * w)     # rho * 总焓

    mf0 = rho * u
    mf1 = rho * v
    mf2 = rho * w

    F = np.zeros((3, 5))
    F[0, 0] = mf0
    F[0, 1] = mf0 * u + p
    F[0, 2] = mf0 * v
    F[0, 3] = mf0 * w
    F[0, 4] = rho_H * u

    F[1, 0] = mf1
    F[1, 1] = mf1 * u
    F[1, 2] = mf1 * v + p
    F[1, 3] = mf1 * w
    F[1, 4] = rho_H * v

    F[2, 0] = mf2
    F[2, 1] = mf2 * u
    F[2, 2] = mf2 * v
    F[2, 3] = mf2 * w + p
    F[2, 4] = rho_H * w
    return F


@njit(cache=True, parallel=True)
def euler_physical_flux_batch(Q: np.ndarray) -> np.ndarray:
    """`euler_physical_flux_point` 的批量版：Q (N,5) -> F (N,3,5)。

    调用方把 `(...,5)` 展平成 `(N,5)`、输出 reshape 回原前导维度。逐点 `prange`：每次迭代只写自己的输出行，
    无写冲突；线程数由 numba 运行时（求解器启动时 `numba.set_num_threads`）决定。
    """
    n = Q.shape[0]
    F = np.zeros((n, 3, 5))
    for i in prange(n):
        F[i] = euler_physical_flux_point(Q[i])
    return F
