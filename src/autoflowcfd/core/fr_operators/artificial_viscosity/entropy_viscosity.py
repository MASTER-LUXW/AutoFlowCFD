"""AutoFlowCFD V2.0 - 问题单元人工粘性：熵残差判据（单元局部）+ 守恒变量拉普拉斯。

## 判据

    r = (h / p) * max_sp |u_hat . grad s|,    s = ln(p / rho^gamma)

即"沿流线方向、一个解点间距上的熵增量"（s 以 c_v 为单位，无量纲；不参照来流）。
光滑无粘流里熵沿流线守恒，所以 r 只在**数值熵产生**处（以及激波）显著；边界层
里的物理熵梯度主要沿壁面法向，投影到流向后天然被排除。判据只用单元自身的解
（单元内的破碎梯度），不读邻居 —— 分布式与 GPU 后端不需要任何 halo 或归约，
结果与分区无关。这是 Guermond 熵粘性（entropy viscosity）"用熵残差定位需要
耗散的地方"的单元局部形式。

## 为什么不是 BJ 越界比 / Persson-Peraire（2026-10-01 在 plate_demo 实测）

plate_demo P1 后缘锐边会长出等压熵尖峰（T 升到 2000 K 以上、压力不变）。在长程
运行的 P1 检查点（179,237 单元）上比较三个判据对热斑单元（T>320 K）的区分度：

    判据                     热斑中位   非热斑 99% 分位   覆盖全部热斑需选中
    h|u_hat.grad s|（本判据）  0.33~0.35  0.0014~0.0019     199~667 单元（0.1~0.4%）
    BJ 越界比（5 守恒变量）    8~60 分散  与热斑同量级       >3500 单元仍漏 4/37
    熵的 BJ 邻域越界           ——         ——                 r>1 只覆盖 22/37

BJ 驱动的人工粘性实测（alpha=1）：被标记单元 9 步内从 702 涨到 7867（4.4%），
残差停在 1e5 不降（基线同期降到 2.3e3）——等于在边界层里加一阶耗散。Persson-
Peraire 在 order=1 上原理性退化（`s0 = -4 log10(order) = 0`）。

## 系数与施加形式

    nu = ramp(r) * alpha * |u|_mean * h / p                       [m^2/s]
    dU_k/dt += div(nu grad U_k)，k = 0..4（全部守恒变量、同一个 nu）

`|u|_mean` 取单元真实解点平均（零填充槽位冻结在初值，见 `fr/native_padding.py`），
`nu` 单元内为常数。`ramp` 是 [R_LOW, R_FULL] 上的 C¹ 余弦斜坡：两端导数为零，
系数在伪时间步之间连续变化，Newton 不会在门限两侧来回跳。

**为什么是全部守恒变量的拉普拉斯**（Persson-Peraire 原文形式；离散对应一阶图
粘性，保持不变域与熵不等式）：此前的施加是"质量扩散 div(eps grad rho) + 把 eps
叠进 mu_t（动量走速度梯度应力、能量走温度传导）"。质量扩散搬运了质量却不带走
它的动量与焓，被扩散的单元温度被凭空改变 —— plate_demo P1 实测：触发后 6 步内
出现 237 K 冷点、触发区从 45 个单元扩散到 250 个、残差从 2.5e3 涨到 1.4e6。
"""

from typing import Callable

import numpy as np

from autoflowcfd.core.fr_operators.flux_kernels.constants import GAMMA
from autoflowcfd.core.utils.array_module import array_module as _array_module
from autoflowcfd.fr.native_padding import reduce_rows_over_real_sps

#: 斜坡起点：r <= R_LOW 的单元不加人工粘性。实测非热斑单元的 99% 分位是
#: 0.0014~0.0019、99.9% 分位 0.009~0.044，热斑单元最小 0.015~0.034。0.01 的
#: 物理含义：一个解点间距上沿流线的熵增 1%（等压下即温度变化 1%）。
ENTROPY_SENSOR_LOW = 0.01

#: 斜坡终点：r >= R_FULL 的单元拿满强度。热斑单元中位 0.33~0.35，在它之上。
ENTROPY_SENSOR_FULL = 0.1


def entropy_sensor_ramp(r, r_low: float = ENTROPY_SENSOR_LOW,
                        r_full: float = ENTROPY_SENSOR_FULL):
    """r 到 [0,1] 的 C¹ 余弦斜坡（r<=r_low 为 0、r>=r_full 为 1、两端导数为零）。"""
    if not r_full > r_low >= 0.0:
        raise ValueError(f"需要 r_full > r_low >= 0，当前 r_low={r_low!r} r_full={r_full!r}")
    xp = _array_module(r)
    t = xp.clip((r - r_low) / (r_full - r_low), 0.0, 1.0)
    return 0.5 * (1.0 - xp.cos(np.pi * t))


def compute_entropy_sensor(U, order: int, cell_volumes, cell_is_prism,
                           scalar_gradient: Callable):
    """逐单元熵残差判据 r = (h/p) max_sp |u_hat . grad s|（只在真实解点上统计）。

    Args:
        U: (n_cells, n_sps, >=5) 守恒变量
        order: 当前阶数（>= 1）
        cell_volumes: (n_cells,)
        cell_is_prism: (n_cells,) 布尔
        scalar_gradient: `f(phi) -> (n_cells, n_sps, 3)`，单元内物理梯度（各后端
            自己的实现，例如 `core/fr_residual/viscous.compute_scalar_gradient`）

    Returns:
        (n_cells,)
    """
    xp = _array_module(U)
    rho = U[:, :, 0]
    mom = U[:, :, 1:4]
    m2 = xp.sum(mom * mom, axis=-1)
    p = (GAMMA - 1.0) * (U[:, :, 4] - 0.5 * m2 / rho)
    s = xp.log(p) - GAMMA * xp.log(rho)
    g = scalar_gradient(xp.ascontiguousarray(s))
    # |u_hat . grad s| = |m . grad s| / |m|（rho 约掉）；静止点上方向无定义，取 0 ——
    # 那里系数本来就正比于 |u|。
    mag = xp.sqrt(m2)
    proj = xp.abs(xp.sum(mom * g, axis=-1)) / xp.where(mag > 0.0, mag, 1.0)
    proj = xp.where(mag > 0.0, proj, 0.0)

    cip = xp.asarray(cell_is_prism).astype(bool)
    pmax = reduce_rows_over_real_sps(proj, cip, order, 'max', xp=xp)
    h = xp.cbrt(xp.maximum(xp.asarray(cell_volumes), 1e-300))
    return pmax * h / order


def compute_artificial_diffusivity(U, order: int, cell_volumes, cell_is_prism,
                                   scalar_gradient: Callable, *, alpha_av: float = 1.0):
    """问题单元人工扩散系数 nu (n_cells, n_sps)，运动粘度量纲 [m^2/s]（单元内常数，
    已广播到每个解点）。order == 0 时没有亚单元内容可耗散，返回全零。"""
    xp = _array_module(U)
    n_cells, n_sps = U.shape[0], U.shape[1]
    if order == 0 or n_cells == 0:
        return xp.zeros((n_cells, n_sps))
    cip = xp.asarray(cell_is_prism).astype(bool)
    ramp = entropy_sensor_ramp(
        compute_entropy_sensor(U, order, cell_volumes, cip, scalar_gradient))
    speed = xp.sqrt(xp.sum(U[:, :, 1:4] ** 2, axis=-1)) / U[:, :, 0]
    speed_mean = reduce_rows_over_real_sps(speed, cip, order, 'mean', xp=xp)
    h = xp.cbrt(xp.maximum(xp.asarray(cell_volumes), 1e-300))
    nu = ramp * (alpha_av * speed_mean * h / order)
    return xp.repeat(nu[:, None], n_sps, axis=1)


def artificial_diffusion_residual(U, nu, scalar_diffusion: Callable):
    """`div(nu grad U_k)`，k = 0..4（dU/dt 约定），湍流分量（若有）为零。

    Args:
        scalar_diffusion: `f(phi, gamma) -> +div(gamma grad phi)`（各后端的标量扩散
            装配，例如 `turbulence/transport.compute_scalar_diffusion_residual`；
            边界上为齐次 Neumann，所以五个守恒量都严格守恒）
    """
    xp = _array_module(U)
    out = xp.zeros_like(U)
    nu_c = xp.ascontiguousarray(nu)
    for k in range(5):
        out[..., k] = scalar_diffusion(xp.ascontiguousarray(U[..., k]), nu_c)
    return out
