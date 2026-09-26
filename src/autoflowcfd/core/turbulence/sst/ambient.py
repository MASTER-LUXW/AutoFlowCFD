"""AutoFlowCFD V2.0 - SST 的环境维持项（Spalart & Rumsey 2007，"SST-sust"），唯一定义。

## 为什么需要

标准 SST 在无剪切的来流里只有耗散：`dk/dt = -beta* k omega`、
`domega/dt = -beta omega^2`，来流湍流沿流向按 `omega = omega_inf/(1 + beta
omega_inf t)` 衰减。外流算例的来流 omega 通常很大（按粘性比给定），衰减
时间 `1/(beta omega_inf)` 只有毫秒量级，而流体穿过计算域要上百毫秒——到达
物体时来流湍流已经所剩无几，给定的湍流度形同虚设（Spalart & Rumsey,
"Effective Inflow Conditions for Turbulence Models in Aerodynamic
Calculations", AIAA J. 45(10), 2007；NASA TMR 的 SST-sust 变体）。

本项目的真实数据（plate_demo P1+SST，隐式稳态）：域内 k 的中位数只剩
`0.10 k_inf`，omega 的物理稳态解低于正性限制器的下限 `0.1 omega_inf`
（`source.py`，暂态安全网，见下文）——贴住下限的单元（全域 10% 以上）里离散稳态无解，Newton 往下推、限制器往上夹，
逐单元松弛因子一路掉到 1e-7、被冻结的单元 5 步内从 0.37% 涨到 6.7%。

## 做法

k、omega 方程各加一个常数源项，大小等于**耗散项在环境状态上的取值**：

    S_k^amb     = D_k(k_amb, omega_amb)       RANS：rho beta* k_amb omega_amb
                                              DES ：rho k_amb^1.5 / l_eff
    S_omega^amb = rho beta omega_amb^2        （beta 为 F1 混合后的值）

于是 `(k, omega) = (k_amb, omega_amb)` 在无剪切区是方程的精确不动点：来流
不再衰减，给定的湍流度原样到达物体。环境值取来流值 `(k_inf, omega_inf)`。
DES 分支用同一个长度尺度 `l_eff` 求环境耗散，是为了让 DDES/IDDES 下同一个
不动点同样精确成立（Spalart–Rumsey 原文只写了 RANS 形式）。

与 `0.1 omega_inf` 下限的关系：那条下限是暂态安全网（P0 无产生项时
`omega -> 0` 是不动点），它成立的前提是低于物理稳态解。没有本项时这个前提在
外流算例上不成立；有了本项，来流区物理 omega 不低于 ~omega_amb。

棱柱通道 + SST（NK，220 步）四路消融，在湍流标量输运的两处离散缺陷修掉之后
（`turbulence/transport/face_frames.py`）：四种组合都收敛到 1e-7 量级、全程不
触发逐单元松弛；**有无下限逐位相同**（下限在整个暂态里从未激活）；核心区 k 有
本项时保持在 0.93 k_inf，没有时衰减到 0.27 k_inf。修复前同一消融里"无下限两档
停滞在 1e4~1e5"的结论来自输运缺陷造成的非物理模态，已不成立。
"""


def ambient_sustaining_terms(model, rho, beta, xp):
    """返回 `(S_k^amb, S_omega^amb)`，与 `compute_source_terms` 的 `Sk`/`S_omega`
    同量纲、可直接相加（形状随 `rho`/`beta` 广播）。"""
    k_amb = float(model.k_inf)
    omega_amb = float(model.omega_inf)
    if model.des_length_scale is not None:
        s_k = rho * k_amb ** 1.5 / xp.maximum(model.des_length_scale, 1e-10)
    else:
        s_k = rho * (model.beta_star * k_amb * omega_amb)
    s_omega = rho * beta * omega_amb ** 2
    return s_k, s_omega
