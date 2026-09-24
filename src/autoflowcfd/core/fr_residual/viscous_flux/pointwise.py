"""AutoFlowCFD V2.0 - 逐点量：温度、粘性物理通量

从 `src/autoflowcfd/core/fr_residual/viscous_flux.py`(原 560 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""


import numpy as np




from .constants import GAMMA, R_AIR


def compute_temperature(Q: np.ndarray) -> np.ndarray:
    """T = p/(rho*R)。"""
    rho = np.maximum(Q[..., 0], 1e-10)
    return Q[..., 4] / (rho * R_AIR)


def viscous_physical_flux(
    Q: np.ndarray,
    grad_vel: np.ndarray,
    grad_T: np.ndarray,
    mu: float,
    Pr: float,
    mu_t=0.0,
    Pr_t: float = 0.9,
) -> np.ndarray:
    """计算粘性物理通量张量 G_i，与 euler_physical_flux 同样的 (...,3,5) 约定。

    Args:
        Q: (...,5) 原始变量 (rho,u,v,w,p)
        grad_vel: (...,3,3) 速度梯度，grad_vel[...,i,j] = d(u_i)/d(x_j)
        grad_T: (...,3) 温度梯度
        mu: 分子动力粘度（标量）
        Pr: 分子普朗特数
        mu_t: 湍流涡粘度（标量或可广播到 Q.shape[:-1] 的数组），默认0
            （层流/未提供湍流模型时）。应力张量按 Boussinesq 假设用
            mu_total=mu+mu_t 统一处理；热传导的湍流贡献用湍流普朗特数
            Pr_t（标准值0.9，非分子普朗特数）单独换算，两者不能共用同一
            个 Pr——这是本次修复把湍流涡粘度真正耦合进粘性应力张量
            （T-01/T-04/T-06）的核心：此前调用方从不传湍流粘度，
            粘性通量永远只用分子粘度。
        Pr_t: 湍流普朗特数

    Returns:
        G: (...,3,5)，G[...,i,:] 是方向 i 的粘性通量向量
           （质量分量恒为0；动量分量 G[...,i,1+j]=tau_ij；能量分量含粘性功+热传导）
    """
    mu_total = mu + mu_t
    mu_total = mu_total * np.ones(Q.shape[:-1]) if np.isscalar(mu_total) else mu_total

    S = 0.5 * (grad_vel + np.swapaxes(grad_vel, -1, -2))  # (...,3,3)
    div_u = grad_vel[..., 0, 0] + grad_vel[..., 1, 1] + grad_vel[..., 2, 2]
    lam = -2.0 / 3.0 * mu_total

    eye3 = np.eye(3)
    tau = 2.0 * mu_total[..., None, None] * S + lam[..., None, None] * div_u[..., None, None] * eye3  # (...,3,3)

    cp = GAMMA * R_AIR / (GAMMA - 1.0)
    k_cond = mu * cp / Pr + mu_t * cp / Pr_t
    q = -k_cond * grad_T if np.isscalar(k_cond) else -k_cond[..., None] * grad_T  # (...,3)

    vel = Q[..., 1:4]  # (...,3)
    work = np.einsum("...i,...ij->...j", vel, tau)  # work[...,j] = sum_i u_i*tau_ij

    shape = Q.shape[:-1]
    G = np.zeros(shape + (3, 5))
    G[..., :, 1:4] = np.swapaxes(tau, -1, -2)  # G[...,i,1+j] = tau[...,j,i] = tau[...,i,j] (对称)
    G[..., :, 4] = work + q
    return G
