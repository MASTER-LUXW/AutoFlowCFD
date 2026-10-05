"""AutoFlowCFD V2.0 - 欧拉物理通量与熵稳定体积散度

从 `src/autoflowcfd/core/fr_operators/flux_kernels.py`(原 615 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import numpy as np

from numba import njit, prange
from .constants import GAMMA


@njit(cache=True)
def euler_physical_flux_point(Q: np.ndarray) -> np.ndarray:
    """欧拉物理通量，单点版。Q=(rho,u,v,w,p) -> F，形状 (3,5)。

    与 fr_residual_inviscid.py::euler_physical_flux 的公式逐一对应。
    """
    rho = Q[0]
    u = Q[1]
    v = Q[2]
    w = Q[3]
    p = Q[4]
    rho_safe = max(rho, 1e-10)

    ke = 0.5 * (u * u + v * v + w * w)
    e_internal = p / ((GAMMA - 1.0) * rho_safe)
    rhoE = rho * (e_internal + ke)
    H = (rhoE + p) / rho_safe

    mf0 = rho * u
    mf1 = rho * v
    mf2 = rho * w

    F = np.zeros((3, 5))
    F[0, 0] = mf0
    F[0, 1] = mf0 * u + p
    F[0, 2] = mf0 * v
    F[0, 3] = mf0 * w
    F[0, 4] = rho * H * u

    F[1, 0] = mf1
    F[1, 1] = mf1 * u
    F[1, 2] = mf1 * v + p
    F[1, 3] = mf1 * w
    F[1, 4] = rho * H * v

    F[2, 0] = mf2
    F[2, 1] = mf2 * u
    F[2, 2] = mf2 * v
    F[2, 3] = mf2 * w + p
    F[2, 4] = rho * H * w
    return F


@njit(cache=True, parallel=True)
def euler_physical_flux_batch(Q: np.ndarray) -> np.ndarray:
    """`euler_physical_flux_point` 的批量版：Q (N,5) -> F (N,3,5)。

    体积项性能优化配套（原体积项调用的是 `fr_residual_inviscid.py::
    euler_physical_flux` 的向量化 numpy 实现，逐点重复分配 `np.zeros`
    大数组+`np.stack`，是新的性能瓶颈来源之一，见 py-spy 对生产网格的
    实测采样）。直接复用已经逐位验证过的 `euler_physical_flux_point`，
    不是新公式，只是换一种循环方式；调用方负责把任意形状的
    `(...,5)` 输入展平成 `(N,5)` 再调用，输出展平成 `(N,3,5)` 后自行
    reshape 回原始前导维度。

    多核并行（阶段二）：这是纯 gather——每次迭代 i 只写自己的输出行
    `F[i]`，不同 i 之间零索引冲突，`prange` 直接安全，不需要像两个
    界面 kernel（fr_residual_inviscid_kernel.py/fr_viscous_flux_kernel.py）
    那样用私有缓冲区+归约处理 scatter-add。线程数由 numba 运行时环境
    （`numba.set_num_threads`，求解器启动时设置一次）决定，这里不接收
    也不查询线程数参数。
    """
    n = Q.shape[0]
    F = np.zeros((n, 3, 5))
    for i in prange(n):
        F[i] = euler_physical_flux_point(Q[i])
    return F


