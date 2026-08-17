"""物理正定性约束。

从 base.py 拆出，控制单文件行数。对守恒变量施加密度/压力正性约束
和速度限幅，防止时间推进后出现非物理状态。
"""

import numpy as np

# 比热比
GAMMA = 1.4


def enforce_positivity(U: np.ndarray, p_floor: float = 1.0) -> np.ndarray:
    """在一次时间步更新后，对守恒变量施加物理上的边界约束。

    把密度和压力投影到正的下限，同时保持速度不变。另外还会限幅速度
    大小，防止动能爆炸。
    """
    MAX_VELOCITY = 1e4  # 10 km/s 上界

    rho = np.maximum(U[:, 0], 1e-6)
    U[:, 0] = rho

    vel = U[:, 1:4] / rho[:, None]

    # 限幅速度大小，并把限幅后的动量写回 U——下面的 ke 必须从这个
    # 最终写回 U 的同一份 vel 推导，否则压力下限会针对一个与实际写回
    # 的动量不匹配的动能来计算。
    vel_mag = np.sqrt(np.sum(vel**2, axis=1))
    clip_mask = vel_mag > MAX_VELOCITY
    if np.any(clip_mask):
        clip_factor = MAX_VELOCITY / vel_mag[clip_mask]
        vel[clip_mask] *= clip_factor[:, None]
        U[clip_mask, 1:4] = (rho[clip_mask, None] * vel[clip_mask])

    ke = 0.5 * rho * np.sum(vel**2, axis=1)
    p = (GAMMA - 1.0) * (U[:, 4] - ke)
    low = p < p_floor
    if np.any(low):
        U[low, 4] = p_floor / (GAMMA - 1.0) + ke[low]
    if U.shape[1] > 5:
        U[:, 5] = np.maximum(U[:, 5], 0.0)      # rho*k >= 0
    if U.shape[1] > 6:
        U[:, 6] = np.maximum(U[:, 6], 1e-8)     # rho*omega > 0
    return U
