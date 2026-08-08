"""基于密度求解的可压缩稳态求解器所用的 Weiss-Smith 式低马赫数预处理。

在低马赫数下，一个未做预处理的可压缩格式必须解析声波（速度
~ sqrt(gamma*p/rho)，标况空气中约 340 m/s），即便真正关心的物理流动
速度慢得多（例如典型汽车外流场约 30 m/s，M~0.09）。这会让一个人为
极小的伪时间步长（CFL 被声波而非慢得多的对流波限制）与 HLLC 迎风
格式带来的过量数值耗散（其数值粘性同样按声速量级缩放）耦合在一起，
同时拖慢收敛速度并削弱解的稳健性——在驻点附近以及回流/分离区域尤为
严重，因为即便整体流场 M~0.1，局部马赫数也可能接近零（曾直接观测到：
一个钝体的分离尾迹在密度上塌缩到接近真空，真正的根因——出口回流不
稳定——是后来单独发现并修复的）。

预处理把物理声学特征值 (un +/- a) 替换成一对经过重新缩放的值，其分布
范围由一个局部截断的马赫数 beta 控制，从而把伪时间推进（以及 HLLC
通量的数值耗散）与真实声速解耦。这在数学上不改变收敛后的稳态残差
（R=0）：无论 SL/SR 波速估计怎么算，HLLC 的通量始终保持严格一致性
（对任意 U 都有 F(U,U)=F(U)），所以收敛时得到的答案完全相同——只是
经过了一条条件数好得多的伪时间路径。

Reference: Weiss & Smith (1995), "Preconditioning Applied to Variable and
Constant Density Flows", AIAA Journal 33(11):2050-2057.
"""

from __future__ import annotations

import numpy as np


def preconditioned_acoustic_eigs(
    un: np.ndarray,
    a: np.ndarray,
    mach_ref: float,
    k: float = 1.1,
):
    """把原始声学特征值 (un+a, un-a) 替换成对应的 Weiss-Smith 预处理值。

    Args:
        un: 局部法向（或特征）速度分量，带符号
        a: 局部物理声速（必须 > 0）
        mach_ref: 参考（自由来流）马赫数——beta^2 被截断为不会低于
            k*mach_ref^2，因此即便正好在局部马赫数为零的驻点上，它也能
            保持良好的数值行为（不会被过度松弛到接近不可压缩极限的
            刚性状态）
        k: mach_ref^2 上的安全裕度倍数（Weiss-Smith 建议约 1.1-1.2；
            防止 beta^2 正好卡在下限上，那样会让预处理器退化）

    Returns:
        (lambda_plus, lambda_minus, c_precond)：两个预处理后的声学特征值，
        以及有效（缩减后）声速 sqrt(beta2)*a——可直接替代基于谱半径的
        CFL 时间步估计里的 `a`。当 beta2=1 时（即 M >= mach_ref，跨/
        超声速区域附近），这会精确退化为 (un+a, un-a, a)——也就是未做
        预处理的原始值——因此预处理只在流动确实缓慢的地方放松刚性。
    """
    a_safe = np.maximum(a, 1e-30)
    mach_local2 = (un / a_safe) ** 2
    beta2 = np.clip(np.maximum(mach_local2, k * mach_ref ** 2), 1e-10, 1.0)

    lam_center = un * (1.0 + beta2) / 2.0
    radius = np.sqrt(((1.0 - beta2) * un / 2.0) ** 2 + beta2 * a_safe ** 2)

    lam_plus = lam_center + radius
    lam_minus = lam_center - radius
    c_precond = np.sqrt(beta2) * a_safe
    return lam_plus, lam_minus, c_precond
