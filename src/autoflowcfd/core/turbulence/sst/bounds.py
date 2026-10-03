"""AutoFlowCFD V2.0 - SST 湍流量 k/omega 的可容许性（唯一定义）。

## realizability 只作用于模型项求值，不裁剪被输运的量（2026-09-26）

高阶 DG 下 k、omega 在欠分辨区（锐边分离剪切层、尾迹）里会出现单元内欠冲；
离散稳态在那些解点上**需要**一个低于下限（甚至为负）的值。此前的做法是每步把
被输运的场夹回 `[下限, 上限]`：Newton 往下推、限制器往上夹，离散稳态在那些点上
无解，逐单元松弛把整个单元冻住（plate_demo P1：被松弛单元 5 步内从 0.25% 涨到
0.8%，全部在锐边剪切层）。

高阶 DG k-omega 的标准做法（Bassi, Crivellini, Rebay & Savini 2005, Comput. Fluids
34；Ilinca & Pelletier 1998 的有限元版本同一思路）：被输运的 k、omega 不裁剪，
realizability 只在**模型项求值**时施加——

    k_bar     = max(k, 0)                         涡粘、F1/F2、产生项上限、DES 长度尺度
    omega_eff = max(omega, omega_r)               同上，以及交叉扩散里的 1/omega
    omega_r   = max(0.1 S, 0.1 omega_inf)         逐点时间尺度 realizability + 暂态安全网

耗散项写成"本变量的原值 × 其余量的有效值"（`D_k = rho beta* k omega_eff`、
`D_omega = rho beta omega_eff omega`）：k 或 omega 越过下限时耗散随原值变号，源项
把它推回来，不需要裁剪。

`clip_to_bounds` 因此只剩两件事：非有限值恢复，以及上界（`k_max`/`omega_max`，
防止 1e260 量级的正反馈爆炸，见 `update.py`）。

## omega 上界随最近壁面解点给定，不能写死（2026-10-03）

omega 的最大物理值在壁面：Wilcox/Menter 壁面值 `60 nu / (beta1 d1^2)`（`d1` 是贴壁第一层
解点到壁面的距离，`transport/omega_wall.py`）。此前上界写死 1e6，理由是"工程壁面 omega
~1e4"——那只对粗壁面网格成立：壁面解析网格（第一层 y+ ~ 1）上 `omega_wall ~ 800 u_tau^2 /
nu`，工程雷诺数下是 5e7 ~ 2e8。湍流平板（`tests/validation/_flat_plate_case.py`，第一层解点
4.2e-6 m）实测：壁面目标值 2.7e8 被夹到 1e6（壁面条件小 270 倍），前缘角点单元 omega 顶到
上界、钳位处残差不可微，P1 隐式 400 步停在降 3e4 倍（残差 99.9% 在前缘单元）；上界改为
`max(OMEGA_MAX_FLOOR, OMEGA_MAX_WALL_FACTOR * 60 nu / (beta1 d_min^2))` 后 187 步收敛到 1e-8、
无一解点顶到上界。粗壁面网格（plate_demo 壁面目标值最大 7.6e4、槽道 ~1e3）上界仍是 1e6，
行为不变。`apply_omega_upper_bound` 在每个后端设定/重算壁距处调用（换阶时贴壁解点更靠近
壁面），分布式取全局最小壁距，各 rank 与单机同一个上界。
"""

import numpy as np

#: 防止 0/负值进入 sqrt 与除法的绝对下限。
ABS_FLOOR = 1e-12

#: k 的尺度下限相对来流值的比例（隐式物理性限幅的尺度，见 `turbulence_row_limits`）。
K_FLOOR_FRACTION = 1e-3

#: omega 的环境安全网相对来流值的比例（`omega_r` 的第二项）。
OMEGA_FLOOR_FRACTION = 0.1

#: omega 上界的下限：没有壁面距离时的初值；粗壁面网格上壁面解析值远小于它（模块文档）。
OMEGA_MAX_FLOOR = 1.0e6

#: omega 上界相对最近壁面解点处壁面解析值的倍数（模块文档）。
OMEGA_MAX_WALL_FACTOR = 10.0

#: 壁面 omega 粘性底层解析式 `omega_vis = OMEGA_WALL_VISCOUS_COEFF * nu / (beta1 d1^2)`
#: 的系数（Wilcox）；CPU/GPU 壁面目标值（`transport/omega_wall.py`、
#: `gpu_scalar_transport/omega_wall.py`）与上界共用这一份。
OMEGA_WALL_VISCOUS_COEFF = 6.0

#: Menter 放大式 `omega_wall = OMEGA_WALL_AMPLIFICATION * omega_vis`（壁面目标值默认档）。
OMEGA_WALL_AMPLIFICATION = 10.0

#: k/omega 梯度模长上限。退化单元上理论为常数的场求梯度，度量比值 adj(J)/det(J)
#: 把浮点噪声放大到 >1e150（2026-08-22 真实网格），模长超过上限的点等比缩到上限。
MAX_GRADIENT_MAGNITUDE = 1e6


def omega_realizability_floor(model, S_mag, xp):
    """逐点 `omega_r = max(0.1 S, 0.1 omega_inf)`（两项的理由见 `source.py` 该处注释）。"""
    return xp.maximum(0.1 * S_mag, OMEGA_FLOOR_FRACTION * float(model.omega_inf))


def model_evaluation_fields(k, omega, omega_r, xp):
    """模型项求值用的 `(k_bar, omega_eff)`（见模块文档）。"""
    return xp.maximum(k, 0.0), xp.maximum(omega, xp.maximum(omega_r, ABS_FLOOR))


def clip_gradient_magnitude(grad, xp):
    """`grad (..., 3)` 模长超过 `MAX_GRADIENT_MAGNITUDE` 的点等比缩到上限，返回新数组。

    源项与输运（CPU、单机 GPU、多 GPU）共用这一份。分量平方溢出时模长为 inf、缩放为
    0（该点梯度置零，不是 NaN；有限输入不会产生 NaN）。
    """
    if xp is np:
        with np.errstate(over="ignore", invalid="ignore"):
            mag = np.linalg.norm(grad, axis=-1)
            return grad * np.clip(MAX_GRADIENT_MAGNITUDE / np.maximum(mag, 1e-10), 0.0, 1.0)[..., None]
    mag = xp.linalg.norm(grad, axis=-1)
    return grad * xp.clip(MAX_GRADIENT_MAGNITUDE / xp.maximum(mag, 1e-10), 0.0, 1.0)[..., None]


def turbulence_scales(model):
    """隐式 k-omega 未知量 `(k, ln omega)` 的逐列尺度：k 为尺度下限（物理性限幅与差分
    步长），`ln omega` 是 O(1) 的对数量、取 1（它的限幅是绝对的，见
    `time_integration/implicit/physicality.py::ScaledFieldRowLimits` 的 `log_columns`）。"""
    return max(ABS_FLOOR, K_FLOOR_FRACTION * float(model.k_inf)), 1.0


def omega_upper_bound(d_min: float, nu: float, beta1: float) -> float:
    """omega 上界：最近壁面解点处壁面解析值的 `OMEGA_MAX_WALL_FACTOR` 倍，不低于
    `OMEGA_MAX_FLOOR`；没有正的壁距（无壁面）时取下限。"""
    if not (np.isfinite(d_min) and d_min > 0.0):
        return OMEGA_MAX_FLOOR
    omega_wall = OMEGA_WALL_AMPLIFICATION * OMEGA_WALL_VISCOUS_COEFF * nu / (beta1 * d_min ** 2)
    return max(OMEGA_MAX_FLOOR, OMEGA_MAX_WALL_FACTOR * omega_wall)


def apply_omega_upper_bound(model, wall_distance, nu: float, global_min=None) -> None:
    """按壁距场（numpy/cupy）设定 `model.omega_max`（模块文档）。`global_min`：分布式下把
    本 rank 的最小正壁距归约成全局最小（`mpi/comm.py::allreduce_min`）；本 rank 没有正壁距
    时贡献 inf。"""
    d = wall_distance[wall_distance > 0.0]
    d_min = float(d.min()) if int(d.size) > 0 else float("inf")
    if global_min is not None:
        d_min = float(global_min(d_min))
    model.omega_max = omega_upper_bound(d_min, nu, float(model.beta1))


def clip_to_bounds(model, xp) -> None:
    """非有限值恢复（`xp.maximum(NaN, x)` 仍是 NaN，不先处理会永久污染场）与上界。
    **不裁剪下界**，理由见模块文档。"""
    k = xp.where(xp.isfinite(model.k_field), model.k_field, 0.0)
    w = xp.where(xp.isfinite(model.omega_field), model.omega_field, float(model.omega_inf))
    model.k_field = xp.minimum(k, model.k_max)
    model.omega_field = xp.minimum(w, model.omega_max)
