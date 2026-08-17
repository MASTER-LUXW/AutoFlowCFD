"""欧拉/粘性物理通量的逐点标量 numba 版本 (性能优化配套)。

`core/fr_residual_inviscid.py::euler_physical_flux` 和
`core/fr_viscous_flux.py::viscous_physical_flux` 是向量化 numpy 实现
（`np.stack`/`np.zeros`/`np.swapaxes`/`np.eye`/`einsum`），在
`fr_residual_inviscid_kernel.py`/`fr_viscous_flux_kernel.py` 的逐点
numba `@njit` 主循环里会被反复调用（每个 Flux Point 调一次）——numba
nopython 模式不支持 `einsum`/`swapaxes`，所以不能直接复用，必须重新写
逐点标量版。这是整个性能优化里除了 AUSM+up 之外风险最高的新代码
（尤其 `viscous_physical_flux_point` 涉及真实物理：Boussinesq 假设下
`mu_total=mu+mu_t` 统一处理应力张量、`k_cond` 混合分子/湍流普朗特数、
`work=vel·tau` 粘性功）——因此单独在这里、用随机输入与现有向量化实现
逐位对比验证（见 `tests/unit/test_fr_flux_kernels_pointwise.py`），不
把这一步的验证并入端到端残差对比，出问题能立刻定位到这里而不是别处。

两个函数的公式必须与 `fr_residual_inviscid.py::euler_physical_flux`/
`fr_viscous_flux.py::viscous_physical_flux` 严格一致，改动前者时必须
同步检查后者是否也要改。
"""

import numpy as np
from numba import njit, prange

GAMMA = 1.4
R_AIR = 287.0  # 空气比气体常数 J/(kg*K)，须与 fr_viscous_flux.py 保持一致


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


@njit(cache=True)
def viscous_physical_flux_point(
    Q: np.ndarray, grad_vel: np.ndarray, grad_T: np.ndarray,
    mu: float, Pr: float, mu_t: float, Pr_t: float,
) -> np.ndarray:
    """粘性物理通量，单点版。

    Args:
        Q: (5,) (rho,u,v,w,p)
        grad_vel: (3,3)，grad_vel[i,j]=d(u_i)/d(x_j)
        grad_T: (3,)
        mu, Pr: 分子动力粘度/普朗特数（标量）
        mu_t, Pr_t: 湍流涡粘度/湍流普朗特数（标量，层流传 0.0/任意值）

    Returns:
        G: (3,5)，与 fr_viscous_flux.py::viscous_physical_flux 的公式
        逐一对应（质量分量恒为0；G[i,1+j]=tau[i,j]（对称）；
        G[i,4]=work[i]+q[i]）。
    """
    mu_total = mu + mu_t

    S00 = grad_vel[0, 0]
    S11 = grad_vel[1, 1]
    S22 = grad_vel[2, 2]
    S01 = 0.5 * (grad_vel[0, 1] + grad_vel[1, 0])
    S02 = 0.5 * (grad_vel[0, 2] + grad_vel[2, 0])
    S12 = 0.5 * (grad_vel[1, 2] + grad_vel[2, 1])
    div_u = grad_vel[0, 0] + grad_vel[1, 1] + grad_vel[2, 2]
    lam = -2.0 / 3.0 * mu_total

    tau00 = 2.0 * mu_total * S00 + lam * div_u
    tau11 = 2.0 * mu_total * S11 + lam * div_u
    tau22 = 2.0 * mu_total * S22 + lam * div_u
    tau01 = 2.0 * mu_total * S01
    tau02 = 2.0 * mu_total * S02
    tau12 = 2.0 * mu_total * S12

    cp = GAMMA * R_AIR / (GAMMA - 1.0)
    k_cond = mu * cp / Pr + mu_t * cp / Pr_t
    qx = -k_cond * grad_T[0]
    qy = -k_cond * grad_T[1]
    qz = -k_cond * grad_T[2]

    u = Q[1]
    v = Q[2]
    w = Q[3]
    # work[j] = sum_i u_i * tau[i,j]，tau 对称
    work_x = u * tau00 + v * tau01 + w * tau02
    work_y = u * tau01 + v * tau11 + w * tau12
    work_z = u * tau02 + v * tau12 + w * tau22

    G = np.zeros((3, 5))
    G[0, 1] = tau00
    G[0, 2] = tau01
    G[0, 3] = tau02
    G[0, 4] = work_x + qx

    G[1, 1] = tau01
    G[1, 2] = tau11
    G[1, 3] = tau12
    G[1, 4] = work_y + qy

    G[2, 1] = tau02
    G[2, 2] = tau12
    G[2, 3] = tau22
    G[2, 4] = work_z + qz
    return G


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


@njit(cache=True, parallel=True)
def viscous_physical_flux_batch(
    Q: np.ndarray, grad_vel: np.ndarray, grad_T: np.ndarray,
    mu: float, Pr: float, mu_t: np.ndarray, Pr_t: float,
) -> np.ndarray:
    """`viscous_physical_flux_point` 的批量版：Q (N,5), grad_vel (N,3,3),
    grad_T (N,3), mu_t (N,) -> G (N,3,5)。理由同 `euler_physical_flux_batch`
    （含多核并行说明——同样是纯 gather，无需私有缓冲区）。
    """
    n = Q.shape[0]
    G = np.zeros((n, 3, 5))
    for i in prange(n):
        G[i] = viscous_physical_flux_point(Q[i], grad_vel[i], grad_T[i], mu, Pr, mu_t[i], Pr_t)
    return G
