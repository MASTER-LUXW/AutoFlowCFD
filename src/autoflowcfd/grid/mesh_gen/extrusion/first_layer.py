"""边界层第一层高度：由目标 y+ 与参考流动推出（湍流平板关联式）。

    Re_L = rho U L / mu,   C_f = 0.026 Re_L^(-1/7)（Schlichting/White 湍流平板 1/7 次方律的局部摩擦系数）
    tau_w = C_f rho U^2 / 2,   u_tau = sqrt(tau_w / rho),   y_1 = y+ mu / (rho u_tau)

这是网格生成阶段的**估计**（真实 y+ 随当地壁面剪应力变化，求解后应检查），工业前处理器（Pointwise/ANSA
的 y+ 计算器）用的是同一类关联式。SA-neg/SST 的低雷诺数壁面处理要求 y+ ~ 1；更粗的近壁网格由 RANS 壁面函数
接管（求解器按当地 y+ 自动切换）。
"""

import math


def first_layer_height_for_yplus(y_plus: float, velocity: float, length: float, rho: float, mu: float) -> float:
    """目标 y+ 对应的第一层高度（米）。参数都必须为正。"""
    for name, value in (("y_plus", y_plus), ("velocity", velocity), ("length", length), ("rho", rho), ("mu", mu)):
        if not value > 0.0:
            raise ValueError(f"第一层高度估计：{name} 必须为正（{value}）")
    reynolds = rho * velocity * length / mu
    cf = 0.026 * reynolds ** (-1.0 / 7.0)
    u_tau = math.sqrt(0.5 * cf) * velocity
    return y_plus * mu / (rho * u_tau)


def resolve_first_layer_height(first_layer_height, target_yplus, ref_velocity, ref_length, rho, mu):
    """第一层高度：直接给出（`first_layer_height`）或由目标 y+ 推出（二选一）；都不给时 None（生成器取缺省）。
    CLI `grid generate-volume` 与 Python API `generate_volume_mesh` 共用。

    Raises:
        ValueError: 两者都给了；给了目标 y+ 却缺参考速度或参考长度；没给目标 y+ 却给了参考速度 / 长度
    """
    if target_yplus is None:
        if ref_velocity is not None or ref_length is not None:
            raise ValueError("参考速度 / 参考长度只用于由目标 y+ 推出第一层高度（target_yplus）")
        return first_layer_height
    if first_layer_height is not None:
        raise ValueError("第一层高度（first_layer_height）与目标 y+（target_yplus）只能给一个")
    if ref_velocity is None or ref_length is None:
        raise ValueError("由目标 y+ 推出第一层高度需要参考速度（ref_velocity）与参考长度（ref_length）")
    return first_layer_height_for_yplus(target_yplus, ref_velocity, ref_length, rho, mu)
