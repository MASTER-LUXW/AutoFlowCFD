"""AutoFlowCFD V2.0 - omega 以 ln(omega) 求解与输运（k-ln(omega)，唯一定义）。

## 为什么（2026-09-27）

近壁 omega ~ 60 nu / (beta1 d^2)，第一层棱柱内跨一到两个数量级；锐边与尾迹边缘
同样有数量级的变化。直接用低阶多项式表示 omega，单元内与面上必然严重下冲：
plate_demo P1+SST（17.9 万单元）8% 的单元在面通量点或过积分细点上 omega 为负，
4.65% 的单元内 omega 最大/最小比超过 10；隐式 k-omega 步里被物理性限幅松弛的
单元中这两类分别占 51% 与 25%（富集 6 倍、5 倍），Newton 在那里要把 k 改 -29、
omega 改 1.9e7，冻住约 1% 的解点，P1 收敛卡死。

高阶 DG RANS 的标准做法是以 `w = ln(omega)` 为未知量（Bassi, Crivellini, Rebay &
Savini 2005, Comput. Fluids 34；DLR PADGE 的 k-ln(omega)，Landmann / Hartmann）：
`ln(omega)` 在近壁几乎线性，低阶多项式就能表示；任意点上 `omega = exp(w) > 0`。

## 变换

    omega = exp(w),   grad(omega) = omega * grad(w)
    rho Dw/Dt = S_omega / omega + div(Gamma_w grad(w)) + Gamma_w |grad(w)|^2

模型项（涡粘、F1/F2、产生/耗散、交叉扩散、DES 长度尺度）在解点上仍按物理
`omega` 与 `grad(omega) = omega grad(w)` 求值，公式逐字不变；只有被输运、被求解
的量换成 `w`。`turb_model.omega_field` 始终存解点上的物理 omega（checkpoint 格式
不变），`w` 在需要时由它取对数得到。

## 边界值

壁面 Dirichlet `ln(omega_w)`、来流 `ln(omega_inf)`（对流的幽灵态与扩散的面值）。

## 限幅

单步 `|dw| <= ln(1/(1-c))`（`c = PHYSICALITY_MAX_RELATIVE_CHANGE`，即 omega 单步
最多减半/加倍，与此前对 omega 的对数对称规则等价）；上界 `omega <= omega_max`。
"""

import math

import numpy as np

from .bounds import ABS_FLOOR, OMEGA_FLOOR_FRACTION


def log_omega(omega, xp=np):
    """`w = ln(omega)`。omega 由 `omega_from_log` 产生、恒为正；`ABS_FLOOR` 只防 0。"""
    return xp.log(xp.maximum(omega, ABS_FLOOR))


def omega_from_log(w, omega_max: float, xp=np):
    """`omega = exp(min(w, ln(omega_max)))`（上界在对数空间施加，避免 exp 溢出）。"""
    return xp.exp(xp.minimum(w, math.log(omega_max)))


def log_omega_gradient_source(gamma_w, grad_w, xp=np):
    """变换带出的逐点项 `Gamma_w |grad(w)|^2`（乘 1/rho 之前）。"""
    return gamma_w * xp.sum(grad_w * grad_w, axis=-1)


def admissible_omega(omega, omega_inf: float, xp=np, source: str = "checkpoint"):
    """载入一个可能来自旧格式（直接输运 omega、允许越过零）的状态时的可容许性投影：
    非有限或非正的解点置为环境安全网 `OMEGA_FLOOR_FRACTION * omega_inf`，并如实告警
    替换了多少个解点。k-ln(omega) 自身产生的状态恒为正，不会触发。"""
    bad = ~(xp.isfinite(omega) & (omega > 0.0))
    n_bad = int(bad.sum())
    if n_bad:
        from loguru import logger

        logger.warning(f"{source} 里有 {n_bad} 个解点的 omega 非正或非有限（旧格式直接输运 omega "
                       f"时允许越过零），k-ln(omega) 下不可容许，已置为 "
                       f"{OMEGA_FLOOR_FRACTION} * omega_inf")
        omega = xp.where(bad, OMEGA_FLOOR_FRACTION * float(omega_inf), omega)
    return omega


def lift_log_omega(omega, lift, omega_max: float):
    """升阶延拓 omega：多项式表示的是 `w = ln(omega)`，所以在对数空间延拓再取指数
    （`lift` 是作用在主机数组上的线性延拓算子，全部后端的切阶路径共用本函数）。"""
    return omega_from_log(lift(log_omega(np.asarray(omega), np)), omega_max, np)
