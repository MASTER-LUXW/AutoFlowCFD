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


@njit(cache=True, inline='always')
def _log_mean_point(a: float, b: float) -> float:
    """Ismail & Roe (2009) 数值稳定对数平均：ln_mean(a,b) = (a-b)/ln(a/b)，
    a≈b 时用泰勒展开避免 0/0（与
    `8_算法重构-Entropy-Stable_Split-Form通量重构-Part1/2.md` 决定性
    验证脚本 `log_mean` 同一公式，这里是单点标量版供 numba 逐点核使用）。
    """
    xi = a / b
    f = (xi - 1.0) / (xi + 1.0)
    u = f * f
    if u < 1e-4:
        F = 1.0 + u / 3.0 + u * u / 5.0 + u * u * u / 7.0
    else:
        F = np.log(xi) / (2.0 * f) if abs(f) > 1e-300 else np.log(xi)
    return (a + b) / (2.0 * F)


@njit(cache=True, inline='always')
def chandrashekar_flux_point(QL: np.ndarray, QR: np.ndarray) -> np.ndarray:
    """Chandrashekar (2013) 熵守恒 + 动能守恒两点数值通量，单点对版。

    QL, QR: (5,) 原始变量 (rho,u,v,w,p)。
    Returns: (3,5)，三个物理方向的两点通量向量（与
    `euler_physical_flux_point` 输出形状一致，QL==QR 时代数上精确退化
    为 `euler_physical_flux_point(QL)`——两点通量的标准一致性要求）。

    公式来源与验证：`8_算法重构-Entropy-Stable_Split-Form通量重构-
    Part1.md` 二、4 节（对照 arXiv:1209.4994 核实），已在
    `8_算法重构-Entropy-Stable_Split-Form通量重构-Part2.md` 用真实
    生产 `D_3d_tet` 矩阵决定性验证（180 个随机四面体样本，P1/P2/P3，
    与对称平均度量项 `{{adj(J)}}_ij=0.5*(adj_i+adj_j)` 配合使用，见
    `core/fr_residual/inviscid.py::compute_entropy_stable_volume_divergence`
    调用处）。
    """
    rhoL, uL, vL, wL, pL = QL[0], QL[1], QL[2], QL[3], QL[4]
    rhoR, uR, vR, wR, pR = QR[0], QR[1], QR[2], QR[3], QR[4]

    betaL = rhoL / (2.0 * pL)
    betaR = rhoR / (2.0 * pR)
    rho_ln = _log_mean_point(rhoL, rhoR)
    beta_ln = _log_mean_point(betaL, betaR)
    rho_bar = 0.5 * (rhoL + rhoR)
    beta_bar = 0.5 * (betaL + betaR)
    u_bar = 0.5 * (uL + uR)
    v_bar = 0.5 * (vL + vR)
    w_bar = 0.5 * (wL + wR)
    p_tilde = rho_bar / (2.0 * beta_bar)
    ke_bar = 0.5 * (u_bar * u_bar + v_bar * v_bar + w_bar * w_bar)
    e_term = 1.0 / (2.0 * (GAMMA - 1.0) * beta_ln) - ke_bar

    F = np.zeros((3, 5))
    # x 方向
    f_rho = rho_ln * u_bar
    fx1 = u_bar * f_rho + p_tilde
    fx2 = v_bar * f_rho
    fx3 = w_bar * f_rho
    F[0, 0] = f_rho
    F[0, 1] = fx1
    F[0, 2] = fx2
    F[0, 3] = fx3
    F[0, 4] = e_term * f_rho + u_bar * fx1 + v_bar * fx2 + w_bar * fx3
    # y 方向
    f_rho = rho_ln * v_bar
    fy1 = u_bar * f_rho
    fy2 = v_bar * f_rho + p_tilde
    fy3 = w_bar * f_rho
    F[1, 0] = f_rho
    F[1, 1] = fy1
    F[1, 2] = fy2
    F[1, 3] = fy3
    F[1, 4] = e_term * f_rho + u_bar * fy1 + v_bar * fy2 + w_bar * fy3
    # z 方向
    f_rho = rho_ln * w_bar
    fz1 = u_bar * f_rho
    fz2 = v_bar * f_rho
    fz3 = w_bar * f_rho + p_tilde
    F[2, 0] = f_rho
    F[2, 1] = fz1
    F[2, 2] = fz2
    F[2, 3] = fz3
    F[2, 4] = e_term * f_rho + u_bar * fz1 + v_bar * fz2 + w_bar * fz3
    return F


@njit(cache=True, parallel=True)
def entropy_stable_volume_divergence_batch(
    Q: np.ndarray, adj_j: np.ndarray, D_fine: np.ndarray
) -> np.ndarray:
    """体积项散度，entropy-stable/split-form 版本（逐单元、逐 SP 对
    two-point flux + 对称平均度量项，见
    `8_算法重构-Entropy-Stable_Split-Form通量重构-Part1/2.md` 完整推导，
    张量收缩约定与生产强形式 `np.matmul(adj_j, F_phys)` 逐一对应）。

    Args:
        Q: (n_cells, n_fine, 5) 原始变量，过积分 FINE 点集上
        adj_j: (n_cells, n_fine, 3, 3) 几何度量项（det(J)*inv(J)），FINE 点
        D_fine: (n_fine, n_fine, 3) FINE 点集自身的微分矩阵，单元间共享

    Returns:
        div_comp: (n_cells, n_fine, 5)，与强形式 `contract_shared_
        operator_2axis(D_fine, np.matmul(adj_j, F_phys))` 同一个量，
        供调用方按同样方式 `restrict_f2c @ div_comp` 后除以 coarse
        det(J)。

    复杂度是 O(n_cells * n_fine^2)（两点通量需要遍历 SP 对），比强形式的
    O(n_cells * n_fine) 更贵——这是 entropy-stable 方案的固有代价（两点
    通量结构性要求，不是实现效率问题），只作为可选项（默认关闭）供用户
    在能接受这个额外开销时启用。用 `prange` 按单元并行、内层 (i,j) 双重
    循环逐对累加而不是先构造 (n_fine,n_fine,...) 的密集张量，把每个单元
    的峰值内存控制在 O(n_fine)（几十~一百多个 FP 量级），避免 O(n_fine^2)
    的中间数组在生产网格规模（数十万单元）下引发内存问题（这类问题此前
    在 P2 SST 输运项上真实出现过，见项目记忆 `p2_sst_performance_and_
    oom_fixes`）。
    """
    n_cells, n_fine, _ = Q.shape
    div_comp = np.zeros((n_cells, n_fine, 5))
    for c in prange(n_cells):
        for i in range(n_fine):
            Qi = Q[c, i]
            adj_i = adj_j[c, i]
            acc = np.zeros(5)
            for j in range(n_fine):
                Qj = Q[c, j]
                adj_jj = adj_j[c, j]
                F_pair = chandrashekar_flux_point(Qi, Qj)  # (3,5)
                for m in range(3):
                    coeff = D_fine[i, j, m]
                    if coeff == 0.0:
                        continue
                    for v in range(5):
                        s = 0.0
                        for cc in range(3):
                            adj_sym = 0.5 * (adj_i[m, cc] + adj_jj[m, cc])
                            s += adj_sym * F_pair[cc, v]
                        acc[v] += coeff * s
            div_comp[c, i] = 2.0 * acc
    return div_comp
