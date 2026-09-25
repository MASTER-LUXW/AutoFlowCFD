"""AutoFlowCFD V2.0 - 自由来流方向与风轴系（攻角 / 侧滑角）。

## 这个模块填的洞

在此之前，自由来流方向在**全代码库被硬编码成 +x**：

    core/fr_solver/boundary.py:87   Q_free = [rho_inf, vel_inf, 0.0, 0.0, p_inf]
    core/fr_solver/solver.py:464    initialize_uniform(u=vel_inf, v=0.0, w=0.0, ...)
    core/fr_solver/boundary.py      SEM 入口 flow_direction = [vel_inf, 0, 0]
    postprocess/fr_coefficients.py  Cd = F[0], Cl = F[2], Cs = F[1]（直接取分量）
    cli/solve/aero_coefficients.py  参考面积按 n_x < 0 做迎风投影

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


def freestream_conservative_state(freestream: dict, n_vars: int = 5) -> np.ndarray:
    """均匀自由来流的守恒量向量 (n_vars,)，**速度方向取自 aoa/aos**。

    前 5 个分量是 `[rho, rho*u, rho*v, rho*w, rho*E]`（公式见
    `core/fr_solver/state.py::uniform_conservative`），其余分量为零
    —— 与此前各处 `np.zeros(...)` 后只填前 5 列的行为一致（湍流量由
    各湍流模型自己持有，不在这里给初值）。

    ## 这个函数填的洞（2026-09-24）

    "用自由来流填一个均匀守恒场"此前在 9 处各写一遍，**全部把速度写死
    成 `(vel_inf, 0, 0)`**：

        core/utils/order_continuation.py                  单机 CPU 降到 P0 重建
        core/mpi/distributed_order_continuation/rebuild.py      CPU MPI 阶数切换
        core/gpu/solver/gpu_solver/core.py                单机 GPU **初场**
        core/gpu/solver/gpu_solver_order_continuation.py   单机 GPU 阶数切换
        core/gpu/distributed/gpu_distributed.py            多 GPU 传统模式**初场**
        core/gpu/distributed/gpu_distributed_order_continuation.py   多 GPU 阶数切换
        core/gpu/distributed/gpu_distributed_fully_distributed/build.py      多 GPU 完全分布式**初场**
        core/gpu/distributed/gpu_distributed_fully_distributed/redistribute.py  同上，阶数切换
        core/mpi/distributed_mesh_loader/fully_distributed.py   CPU 完全分布式阶数切换重分发

    而边界条件（`Q_free`，经 `direction_from_freestream`）一直用的是正确
    方向。于是 `--aoa` 非零时初场与边界不一致 —— `FRSolver.__init__` 里
    那段注释早就写明这会让第一步吸收一个量级为 `vel_inf*sin(aoa)` 的
    速度跳跃。**每个目标阶数 >= 2 的全新算例都要经过 Order Continuation
    的降 P0 重建**（`FRSolver.solve` 里 `self.order >= 2` 才进入），所以
    单机 CPU 也躲不开：它自己的初场是对的，但随即被那次
    重建覆盖成零攻角。

    另外其中 4 处带着 `get('vel_inf', 33.33)` 这类魔法兜底值（与 CLI
    默认值是两份事实来源，缺字段时静默用一个可能与求解器实际来流不同
    的值）。本函数**不给兜底**：缺 `rho_inf/vel_inf/p_inf` 直接 KeyError。
    攻角缺省为 0（旧 checkpoint 没有这两个字段，理由见
    `direction_from_freestream`）。

    Args:
        freestream: 求解器的 `freestream` 字典（至少含 rho_inf/vel_inf/p_inf）。
        n_vars: 守恒量个数（>=5）。

    Returns:
        (n_vars,) float64。`aoa=aos=0` 时与此前的 `(vel_inf, 0, 0)` 写法
        在动量分量上逐位相同。
    """
    from autoflowcfd.core.fr_solver.state import uniform_conservative

    if n_vars < 5:
        raise ValueError(f"n_vars 至少为 5，收到 {n_vars}")
    vel = float(freestream["vel_inf"]) * direction_from_freestream(freestream)
    out = np.zeros(n_vars, dtype=np.float64)
    out[:5] = uniform_conservative(
        float(freestream["rho_inf"]), float(vel[0]), float(vel[1]),
        float(vel[2]), float(freestream["p_inf"]))
    return out


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
