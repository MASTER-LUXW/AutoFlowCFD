"""AutoFlowCFD V2.0 - 参考马赫数 mach_ref 的唯一来源。

AUSM+up Weiss-Smith 低马赫数预处理（`fr_operators/kernels.py::compute_ausm_up_flux`）
与 CFL 步长估计（`cfl.py`）共用同一个参考马赫数。

**为什么单独成模块（2026-09-25）**：此前 `FRSolver.__init__` 按 AUSM+up 预处理档
分派下限（physical 档 0.05、legacy 档 0.1），而 CLI 里构造"完全分布式加载"
（CPU 与多 GPU）求解包的 6 处各自手算 `max(M, _MACH_REF_FLOOR)`，用的是
legacy 档的 0.1 这个向后兼容别名 —— 于是默认档下，同一个算例在单机与完全
分布式路径上拿到不同的 mach_ref（plate_demo：单机 0.0882、完全分布式 0.1），
AUSM+up 预处理与 CFL 都不同，实际解的是两个不同的离散问题。现在全部从
`resolve_mach_ref` 取，别名已删除。
"""

import numpy as np
from loguru import logger

# AUSM+up Weiss-Smith 预处理参考马赫数的物理下限（完整推导与实证标定见
# `resolve_mach_ref` 文档）。低于此值时预处理通量的压差放大系数
# ~1/mach_ref² 会让显式时间推进在任何声学 CFL 步长下失稳。
#
# **按 AUSM+up 预处理档分派（2026-09-17）。** 原来是与档位无关的 0.1，而
# 那个值是在 `legacy` 档上标定的——`PRECOND_PHYSICAL`（2026-09-16 起的
# 默认）把压力分裂改用物理声速之后，`1/mach_ref²` 那条放大路径本身就变了。
# 用直接谱测量重新标定（组装预处理后算子 Γ⁻¹R 的稠密 Jacobian 求特征值，
# 与 SSP-RK3 稳定域 |Im|<=1.732 比；**扫的是自洽的 mach_ref**，即随 vel_inf
# 一起变，而不是在固定来流上人为改 mach_ref——后者是物理上不存在的组合）：
#
#   vel_inf  M_true    用 0.1 的裕度   用真实值 dt      用真实值裕度
#      30    0.0882      6.50x        1.04e-5 (+14%)     5.33x  稳定
#      17    0.0500      —            1.78e-5            3.55x  稳定
#     13.6   0.0400      —            2.22e-5            2.96x  稳定
#     10.2   0.0300      —            2.96e-5            2.32x  稳定
#      3     0.0088      5.66x        1.01e-4            0.74x  越界
#      1     0.0029      5.59x        3.02e-4            0.25x  越界
#
# 下限的最坏情形恰好是 `M_true == FLOOR` 那一点（那时 mach_ref 最小、dt
# 最大），所以上表第 2~4 行就是各候选下限的裕度。取 **0.05**（3.55 倍裕度）。
#
# `legacy` 档保持 0.1：同一套测量下它在 mach_ref=0.0088 就已越界 4.5 倍
# （max|dt*λ|=7.81）、在 0.00088 是 768——正是原注释记录的那条
# `1/mach_ref²` 放大，它的标定仍然有效。（legacy 只作回归对照用。）
#
# 收益：真实马赫数落在 [0.05, 0.1) 的算例不再被抬到 0.1。plate_demo
# （M=0.0882）dt +14%；M=0.05 的算例 dt +50%。
_MACH_REF_FLOOR_LEGACY = 0.1
_MACH_REF_FLOOR_PHYSICAL = 0.05


def _mach_ref_floor_for_mode(precond_mode: int) -> float:
    """按 AUSM+up 预处理档给出 mach_ref 下限（见上方常量的完整标定）。"""
    from autoflowcfd.core.fr_operators.kernels import PRECOND_LEGACY

    return (_MACH_REF_FLOOR_LEGACY if precond_mode == PRECOND_LEGACY
            else _MACH_REF_FLOOR_PHYSICAL)




def resolve_mach_ref(rho_inf: float, vel_inf: float, p_inf: float, precond_mode=None) -> float:
    """真实来流马赫数，按 AUSM+up 预处理档的下限钳制。

    mach_ref：AUSM+up Weiss-Smith 低马赫数预处理（kernels.py::
    compute_ausm_up_flux）和 CFL 步长估计（cfl.py）共用的同一个
    参考马赫数，从真实自由来流条件算一次，不再各处各用一套（2026-
    08-14 那次失稳正是因为 CFL 和通量各自假设了不一致的参考值，
    见 cfl.py 模块文档"已撤销"一节）。
    
    物理下限钳制（2026-08-26，P2 发散专项修复）：真实来流马赫数低于
    _MACH_REF_FLOOR 时钳制到下限。AUSM+up 的 Mp 压力扩散项正比于
    (pR-pL)/(fa*a_half_p²)，其中 fa≈2*mach_ref（滞止面）、
    a_half_p²=beta2*a²、beta2 下限=1.1*mach_ref²——两者同时随
    mach_ref 塌缩，压差的有效放大系数 ~1/mach_ref²：Couette 验证
    算例（vel_inf=U_wall=0.01 m/s，mach_ref≈2.9e-5）实测残差泛函
    对能量扰动的增益高达 ~3.4e13/Pa（mach_ref 扫描证实增益严格正比于
    1/mach_ref²），显式 SSP-RK3 在任何声学 CFL 步长下都必然发散。
    物理上这个下限对应低马赫数渐近展开的适用边界：真实压力扰动按
    ρ·U² ~ mach_ref² 缩小才与预处理通量的 1/mach_ref² 放大相互抵消，
    低于下限后离散舍入/边界瞬态扰动不再随 M² 缩小，方案转为舍入驱动失稳。
    下限值经真实算例实证扫描标定（分两档）：
    （1）Couette 棱柱算例：0.02 仍发散（iter 6）、0.05 稳定；
    （2）TGV 三向周期坍缩坐标四面体算例（真实 mach_ref≈0.0874，
    网格条件数更差）：0.0874 仍发散（step 5 KE 暴涨至 1e93、
    step 6 溢出）、0.1 稳定且动能衰减曲线与历史实测一致。
    因此下限取 0.1——恰好等于 kernels.py 历史注释记载的遗留硬编码值，
    那次把硬编码改成传入真实值的重构正是这两个算例的共同回归点；
    0.1 以上真实马赫数的算例不受影响。
    下限按 AUSM+up 预处理档分派（见 `_MACH_REF_FLOOR_PHYSICAL` 上方
    的重标定记录）：physical/pressure_physical 档 0.05、legacy 档 0.1。

    Args:
        precond_mode: AUSM+up 预处理档；None 时按环境变量解析（与通量核同一个
            解析器，保证同一次运行里各处取到同一档）。
    """
    from autoflowcfd.core.fr_operators.kernels import resolve_ausm_precond_mode

    mode = resolve_ausm_precond_mode() if precond_mode is None else precond_mode
    mach_ref = vel_inf / np.sqrt(max(1.4 * p_inf / max(rho_inf, 1e-10), 1e-10))
    floor = _mach_ref_floor_for_mode(mode)
    if mach_ref < floor:
        logger.info(
            f"[mach_ref] 真实来流马赫数 {mach_ref:.4g} 低于当前 AUSM+up "
            f"预处理档的下限 {floor:g}，钳到下限。低于它时预处理通量的"
            f"压差放大系数 ~1/mach_ref^2 会让显式推进在任何声学 CFL 步长"
            f"下失稳（实测标定见 mach_ref.py 里 _MACH_REF_FLOOR_PHYSICAL "
            f"上方的表）。代价是低马赫预处理在这个算例上只能部分生效。")
    return float(max(mach_ref, floor))
