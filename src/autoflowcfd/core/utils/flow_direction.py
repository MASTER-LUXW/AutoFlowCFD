"""AutoFlowCFD V2.0 - 自由来流方向与风轴系（攻角 / 侧滑角）。

## 这个模块填的洞

在此之前，自由来流方向在**全代码库被硬编码成 +x**：

    core/fr_solver/boundary.py:87   Q_free = [rho_inf, vel_inf, 0.0, 0.0, p_inf]
    core/fr_solver/solver.py:464    initialize_uniform(u=vel_inf, v=0.0, w=0.0, ...)
    core/fr_solver/boundary.py      SEM 入口 flow_direction = [vel_inf, 0, 0]
    postprocess/fr_coefficients.py  Cd = F[0], Cl = F[2], Cs = F[1]（直接取分量）
    cli/solve_aero_coefficients.py  参考面积按 n_x < 0 做迎风投影

也就是说**没有任何攻角/侧滑角选项**。而攻角扫掠是最常见的 CFD 研究，
升阻比随攻角的变化基本上是外流气动计算的第一产出——这是一项真实的
能力缺口，不是"用户可以自己旋转网格"能替代的（旋转网格会让边界层
挤出方向、壁面距离、周期面配对全部跟着变）。

## 约定（与本代码库既有的 x=流向 / y=展向 / z=法向右手系一致）

攻角 alpha（aoa，绕 y 轴、抬头为正）、侧滑角 beta（aos，绕 z 轴）：

    来流单位向量  d = ( cos(a)cos(b),  sin(b),  sin(a)cos(b) )

标准风轴系正交三元组（aircraft wind axes，z 轴向上）：

    阻力方向  d_hat = ( cos(a)cos(b),  sin(b),  sin(a)cos(b) )
    侧力方向  s_hat = (-cos(a)sin(b),  cos(b), -sin(a)sin(b) )
    升力方向  l_hat = (-sin(a),        0.0,     cos(a)       )

三者严格正交且单位长（`test_flow_direction.py` 里对随机角度组合逐项
验证，不是"看起来对"）：

    d.l = -cos(a)cos(b)sin(a) + sin(a)cos(b)cos(a) = 0
    d.s = -cos^2(a)sin(b)cos(b) + sin(b)cos(b) - sin^2(a)sin(b)cos(b) = 0
    l.s =  sin(a)cos(a)sin(b) - sin(a)sin(b)cos(a) = 0
    |d|^2 = cos^2(a)cos^2(b) + sin^2(b) + sin^2(a)cos^2(b) = cos^2(b)+sin^2(b) = 1
    |s|^2 = cos^2(a)sin^2(b) + cos^2(b) + sin^2(a)sin^2(b) = 1
    |l|^2 = sin^2(a) + cos^2(a) = 1

`alpha = beta = 0` 时退化为 `d=(1,0,0)`、`s=(0,1,0)`、`l=(0,0,1)`，与
此前硬编码的行为**逐位相同**——所以默认路径的数值结果不变。

## 力矩留在体轴系

Cm（俯仰，绕 y）/ Cy（偏航，绕 z）/ Cr（滚转，绕 x）按惯例报在**体轴
系**，不随攻角旋转。力（Cd/Cl/Cs）报在风轴系。这是气动数据的标准呈现
方式；混用会让同一份数据在不同攻角下不可比。
"""

from typing import Tuple

import numpy as np

#: 攻角/侧滑角的合法范围（度）。超出这个范围不是"用户想要的大攻角"，
#: 而几乎一定是单位搞错了（弧度当成度传）——直接报错而不是静默接受一个
#: 让来流反向的角度。
_ANGLE_LIMIT_DEG = 90.0


def _validate(aoa_deg: float, aos_deg: float) -> Tuple[float, float]:
    a = float(aoa_deg)
    b = float(aos_deg)
    for name, v in (("aoa_deg", a), ("aos_deg", b)):
        if not np.isfinite(v):
            raise ValueError(f"{name} 必须是有限值，收到 {v!r}")
        if abs(v) > _ANGLE_LIMIT_DEG:
            raise ValueError(
                f"{name}={v} 超出 [-{_ANGLE_LIMIT_DEG}, {_ANGLE_LIMIT_DEG}] 度。"
                f"这个量是**角度**不是弧度；超出这个范围通常意味着单位传错了。"
                f"若确实要算 |角度| > {_ANGLE_LIMIT_DEG} 度的工况，请先确认"
                f"边界条件的进出口指派仍然自洽（入口面的外法向与来流方向"
                f"夹角超过 90 度时，那个面实际上已经变成出口）。"
            )
    return a, b


def freestream_direction(aoa_deg: float = 0.0, aos_deg: float = 0.0) -> np.ndarray:
    """自由来流的**单位**方向向量 (3,)。

    Args:
        aoa_deg: 攻角（度，绕 y 轴，抬头为正）
        aos_deg: 侧滑角（度，绕 z 轴）

    Returns:
        (3,) float64 单位向量。`aoa=aos=0` 时严格等于 `[1, 0, 0]`。
    """
    a, b = _validate(aoa_deg, aos_deg)
    ar, br = np.deg2rad(a), np.deg2rad(b)
    return np.array([
        np.cos(ar) * np.cos(br),
        np.sin(br),
        np.sin(ar) * np.cos(br),
    ], dtype=np.float64)


def wind_axes(aoa_deg: float = 0.0,
              aos_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """风轴系正交三元组 `(d_hat, s_hat, l_hat)` = (阻力, 侧力, 升力) 方向。

    公式与正交性推导见模块文档。`aoa=aos=0` 时严格等于单位基
    `(1,0,0) / (0,1,0) / (0,0,1)`，故默认路径下的 Cd/Cl/Cs 与此前
    "直接取 F[0]/F[2]/F[1]"逐位相同。
    """
    a, b = _validate(aoa_deg, aos_deg)
    ar, br = np.deg2rad(a), np.deg2rad(b)
    ca, sa, cb, sb = np.cos(ar), np.sin(ar), np.cos(br), np.sin(br)
    d_hat = np.array([ca * cb, sb, sa * cb], dtype=np.float64)
    s_hat = np.array([-ca * sb, cb, -sa * sb], dtype=np.float64)
    l_hat = np.array([-sa, 0.0, ca], dtype=np.float64)
    return d_hat, s_hat, l_hat


def freestream_velocity(vel_inf: float, aoa_deg: float = 0.0,
                        aos_deg: float = 0.0) -> np.ndarray:
    """自由来流速度矢量 (3,) = `vel_inf * freestream_direction(...)`。"""
    return float(vel_inf) * freestream_direction(aoa_deg, aos_deg)


def direction_from_freestream(freestream: dict) -> np.ndarray:
    """从 `solver.freestream` 字典取来流方向，缺字段时退化为 +x。

    为什么要容忍缺字段：`freestream` 字典由多条路径各自构造（单机 /
    CPU MPI 两种模式 / 多 GPU 两种模式 / checkpoint 重建），旧
    checkpoint 的 metadata 里也没有 aoa/aos。退化为 +x 与此前的硬编码
    行为**完全一致**，所以对旧数据是向后兼容的、不是静默吞掉配置
    ——真正需要报错的是"用户显式传了攻角但某条路径没转发"，那由
    `tests/unit/test_flow_direction_wiring.py` 的跨入口一致性检查覆盖。
    """
    return freestream_direction(
        float(freestream.get("aoa_deg", 0.0) or 0.0),
        float(freestream.get("aos_deg", 0.0) or 0.0),
    )
