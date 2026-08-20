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


@njit(cache=True, inline='always')
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


@njit(cache=True, inline='always')
def viscous_boundary_penalty_tilde(
    Q_o: np.ndarray, Q_ghost: np.ndarray, mu_total: float,
    vol: float, adj_mag: float, oside: float, c_ip: float,
) -> np.ndarray:
    """边界面粘性 Interior Penalty (IP) 罚项，动量分量，已转成 tilde（逆变）单位。

    根因：`viscous_physical_flux_point` 算出的应力张量 tau 只依赖速度梯度
    `grad_vel`，不依赖状态 `Q` 本身；而边界面的梯度按本代码库既定策略镜像
    内部值（`gv_ghost=gv_owner`，见 viscous_flux_kernel.py 模块文档"边界面
    梯度处理"一节），于是 `G_common` 与 `G_own` 的动量分量在任意边界条件
    下逐位相等——`jump_owner` 恒为零，等价于固壁上无滑移剪应力不存在
    （数值验证：WALL/SLIP_WALL/FARFIELD 给出逐位相同的动量残差，V2.0
    专家组评审新发现的阻塞级问题）。

    标准 DG/FR 文献（S-03 允许的 "IP" 方案，与 LDG 并列）对此的解法是在
    共同数值粘性通量里补一个正比于状态跳跃 [[u]]=u_owner-u_ghost 的耗散
    罚项（Interior Penalty / SIPG，Arnold et al. 2002 统一分析框架；系数
    形式取自 Shahbazi (2005) 的标准 penalty parameter η=C·μ/h）：

        G_num·n = {G(∇u)}·n - η·[[u]]，η = c_ip·μ_eff/h

    h 用本项目自己在粘性 CFL（core/fr_solver/cfl.py::_compute_local_time_step
    的 `dt_visc ∝ V^(2/3)/mu_eff` 隐含的长度尺度约定）里已经采用的
    "cell volume^(1/3)" 做局部特征长度，不引入新的长度尺度定义，量纲上
    `mu_total*(Δu)/h` 与 tau 同为 Pa；再乘以 `adj_mag*oside`
    （与本文件其余通量把物理量转成 tilde/逆变量的方式完全一致，
    见 fr_residual_inviscid_kernel.py 里 `F_tilde_common = F_common_n *
    adj_mag * oside` 的同一套惯例）得到可以直接叠加进 `G_tilde_common`
    的量。

    只在边界面调用（`is_boundary[f]==True` 分支）：内部面两侧的梯度本就
    是各自独立算出的真实局部梯度（不是镜像），已有非零、物理有意义的
    耦合，不属于本次修复范围，不额外加罚项，避免改动已通过验证的内部
    粘性通量路径。

    Args:
        Q_o: (5,) 面上 owner 侧原始变量外插值 (rho,u,v,w,p)
        Q_ghost: (5,) 边界幽灵态原始变量（含真实 BC，如 WALL 无滑移镜像）
        mu_total: 分子+湍流动力粘度之和（该 FP 处）
        vol: owner 单元的体积尺度（det_jacs 均值或 P0 的 det_jacs 本身）
        adj_mag: 该 FP 处 owner 侧逆变行范数（与本文件其余处一致的度量量）
        oside: owner 侧参考坐标方向（±1）
        c_ip: 罚项常数（标准 DG 惯例取 O(1)~O(10)，本实现固定用 4.0，
            未做多项式阶数相关的最优 trace-inequality 常数标定——这是
            稳定性调优参数，不影响"罚项存在与否/符号是否耗散"这一
            正确性核心，若未来观测到边界层数值振荡可调大）

    Returns:
        pen: (5,)，仅 [1:4] 非零（动量分量的罚项贡献），可直接
        `G_tilde_common[v] += pen[v]` for v in range(5)
    """
    pen = np.zeros(5)
    h = vol ** (1.0 / 3.0)
    if h < 1e-300:
        h = 1e-300
    eta = c_ip * mu_total / h
    scale = eta * adj_mag * oside
    for v in range(1, 4):
        pen[v] = -scale * (Q_o[v] - Q_ghost[v])
    return pen


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
