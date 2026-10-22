"""基于密度求解的可压缩稳态求解器所用的 Weiss-Smith 式低马赫数预处理。

在低马赫数下，一个未做预处理的可压缩格式必须解析声波（速度
~ sqrt(gamma*p/rho)，标况空气中约 340 m/s），即便真正关心的物理流动
速度慢得多（例如典型汽车外流场约 30 m/s，M~0.09）。这会让一个人为
极小的伪时间步长（CFL 被声波而非慢得多的对流波限制）与迎风
格式带来的过量数值耗散（其数值粘性同样按声速量级缩放）耦合在一起，
同时拖慢收敛速度并削弱解的稳健性——在驻点附近以及回流/分离区域尤为
严重，因为即便整体流场 M~0.1，局部马赫数也可能接近零（曾直接观测到：
一个钝体的分离尾迹在密度上塌缩到接近真空，真正的根因——出口回流不
稳定——是后来单独发现并修复的）。

预处理把物理声学特征值 (un +/- a) 替换成一对经过重新缩放的值，其分布
范围由一个局部截断的马赫数 beta 控制，从而把伪时间推进（以及迎风
通量的数值耗散）与真实声速解耦。这在数学上不改变收敛后的稳态残差
（R=0）：无论波速估计怎么算，AUSM+up 的通量始终保持严格一致性
（对任意 U 都有 F(U,U)=F(U)），所以收敛时得到的答案完全相同——只是
经过了一条条件数好得多的伪时间路径。

Reference: Weiss & Smith (1995), "Preconditioning Applied to Variable and
Constant Density Flows", AIAA Journal 33(11):2050-2057.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numba import njit, prange

from autoflowcfd.core.utils.array_module import array_module

from autoflowcfd.core.thermo import GAMMA


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


def preconditioned_sound_speed(vel_mag, a, mach_ref: float, k: float = 1.1):
    """伪时间预处理下的**有效声速** sqrt(beta^2)*a，beta^2 按**速度模**取。

    与 `preconditioned_acoustic_eigs` 的区别是实质性的，不是风格问题：
    那个函数从它收到的第一个参数（通常是面法向速度 `un`）自己算 beta^2，
    因为它服务的是逐面通量的特征值；而伪时间预处理矩阵 Gamma 的 beta^2
    用的是当地**速度模**（见 `_precond_beta2` 文档与本模块末尾的推导）。

    为什么必须用这个函数来定伪时间步长：显式推进的是 dU/dtau = -Gamma R，
    其稳定性由 Gamma @ A_n 的谱半径决定。该谱半径的闭式上界是
        |un| + sqrt(beta^2)*a
    （由 max|lambda±| = |un|(1+b)/2 + sqrt(((1-b)|un|/2)^2 + b*a^2)
      <= |un|(1+b)/2 + (1-b)|un|/2 + sqrt(b)*a = |un| + sqrt(b)*a 得到），
    其中 b **必须**是 Gamma 里那一个 beta^2、也就是按速度模取的那一个。
    若误用 beta^2(un)：因为 |un| <= |u|，有 beta^2(un) <= beta^2(|u|)，
    有效声速偏小、dt 被系统性高估——流动与面法向越斜越严重（极端情形
    un=0、|u|=0.5a 时 dt 高估约 4.8 倍），本质上与 2026-08-24 那次
    "CFL 用了预处理波速、被积分的方程却没预处理"的事故是同一类错误：
    步长与算子不成对。这两个 beta^2 只在双方都落到下限 k*mach_ref^2 时
    才恰好相等（即处处 M < sqrt(k)*mach_ref 的严格低速区），所以问题在
    钝体绕流的加速区（局部 M 超过 mach_ref）才暴露出来。

    Args:
        vel_mag: 当地速度模 |u|（非负）
        a: 当地物理声速（必须 > 0）
        mach_ref: 参考马赫数，给 beta^2 提供下限 k*mach_ref^2
        k: 下限的安全裕度倍数（与 Gamma 保持一致，默认 1.1）

    Returns:
        c_precond = sqrt(beta^2) * a；M >= mach_ref 的区域精确退化为 a。数组模块无关（CPU 与 GPU 后端共用）。
    """
    xp = array_module(vel_mag, a)
    a_safe = xp.maximum(a, 1e-30)
    beta2 = _precond_beta2(xp.asarray(vel_mag) ** 2, a_safe ** 2, mach_ref, k)
    return xp.sqrt(beta2) * a_safe


# =============================================================================
# 伪时间预处理矩阵 Γ 本身（2026-09-14 新增，用户提出"从开始计算到收敛的
# 迭代步数太多（数万步）"后的根本性优化）
#
# 背景与这里补的到底是什么：
#   上面的 `preconditioned_acoustic_eigs` 只提供**特征值**（供 AUSM+up 通量
#   的低马赫数耗散修正使用，以及给 CFL 估计一个"预处理后的谱半径"）。
#   2026-08-24 曾经有人把 `c_precond` 直接塞进 CFL 步长估计、**但没有同时
#   改变被积分的方程**——那次必然失稳（真实复现：CFL=0.1 第 2 步就发散），
#   `core/fr_solver/cfl.py` 的文档把这次事故记录为"预处理不能用于 CFL"。
#   这个结论只对"只改步长"这半个改动成立：显式格式积分的是
#   `dU/dtau = -R(U)`，其稳定步长由 R 的 Jacobian 谱半径 (|un|+a) 决定，
#   单方面放大 dt 当然会炸。
#
#   真正的 Weiss-Smith 预处理要求把**方程本身**换成
#       dU/dtau = -Gamma(U) * R(U)
#   此时 Jacobian 变成 Gamma*A_n，其谱半径正是 (|un| + c_precond)，dt 才
#   可以按预处理后的波速取——两者必须成对出现。本节补的就是缺失的 Gamma。
#
# 为什么这不会改变计算结果（工业级要求"结果准确"的关键）：
#   下面推导给出 det(Gamma) = beta^2 > 0，即 Gamma 处处可逆，于是
#       Gamma(U) * R(U) = 0  <=>  R(U) = 0
#   稳态不动点**逐点完全相同**——预处理只改变逼近这个不动点的伪时间路径
#   （条件数从 O(1/M) 降到 O(1)），不改变收敛到的解。这与
#   `preconditioned_acoustic_eigs` 文档里"在数学上不改变收敛后的稳态残差"
#   是同一个论证，只是那里针对通量、这里针对时间导数项。
#
# 推导（秩一形式，实现只需 ~15 次浮点运算/点、不存任何矩阵）：
#   取等熵原始变量 (p,u,v,w,s)。Weiss-Smith 的全部改动就是把密度对压力的
#   等熵导数 d = drho/dp|_s = 1/a^2 换成 d~ = 1/(beta^2 a^2)（人为降低
#   "压缩性"，从而降低声波速度）。于是
#       M~ = M + (d~ - d) * phi * e_1^T ,   phi = dU/dp|_s * a^2 = (1,u,v,w,H)
#   （H = a^2/(gamma-1) + q^2/2 为总焓；phi 的最后一项来自
#    d(rho E)/dp|_s = 1/(gamma-1) + q^2/(2a^2)）
#   预处理系统 M~ dQ/dtau = -R_U 等价于 dU/dtau = -Gamma R_U，
#   Gamma = M M~^{-1}。因为 M e_1 = d*phi，用 Sherman-Morrison 直接得到
#       Gamma = I - ((1-beta^2)/a^2) * phi * psi^T ,
#       psi^T = dp/dU = (gamma-1) * (q^2/2, -u, -v, -w, 1)
#   作用到任意向量 r 上就是"把 r 里对应的压力扰动分量按 (1-beta^2) 削掉"：
#       Gamma r = r - ((1-beta^2)/a^2) * dp(r) * (1,u,v,w,H),
#       dp(r) = (gamma-1)*(q^2/2*r0 - u*r1 - v*r2 - w*r3 + r4)
#   两个可以直接验证的性质（见 tests/unit/test_low_mach_preconditioner.py）：
#     * psi . phi = a^2  =>  det(Gamma) = 1 - (1-beta^2) = beta^2 > 0（可逆）
#     * beta^2 = 1（M >= mach_ref，无需预处理的区域）时 Gamma = I，
#       精确退化回原格式，不引入任何改变
#     * eig(Gamma * A_n) = (un, un, un, lam_plus, lam_minus)，其中后两个
#       正是上面 `preconditioned_acoustic_eigs` 给出的闭式值——这是对整段
#       推导最强的数值校验（测试里用数值 Jacobian 直接比对）
# =============================================================================


def _precond_beta2(q2, a2, mach_ref: float, k: float):
    """预处理参数 beta^2 = clip(max(|u|^2/a^2, k*mach_ref^2), 1e-10, 1)。

    与 `preconditioned_acoustic_eigs` 的定义**刻意有一处区别**：那里用的是
    面法向速度 `un`（因为它算的是逐面通量的特征值），这里用速度模 |u|
    （因为时间导数项的预处理是单元/解点局部的、与任何面法向无关）。
    Weiss-Smith 原文对伪时间预处理用的就是当地速度模。数组模块无关。
    """
    xp = array_module(q2, a2)
    m2 = q2 / xp.maximum(a2, 1e-30)
    return xp.clip(xp.maximum(m2, k * mach_ref ** 2), 1e-10, 1.0)


@njit(cache=True, parallel=True)
def _apply_low_mach_precond_kernel(R, Q, mach_ref, k, out) -> None:
    """逐 (cell, SP) 施加 `out = Gamma(U) @ R`，见本模块末尾推导。

    Args:
        R: (n_cells, n_sps, 5) 平均流残差（守恒变量顺序 rho,rho_u,rho_v,rho_w,rho_E）
        Q: (n_cells, n_sps, >=5) 原始变量 (rho, u, v, w, p)
        mach_ref: 参考马赫数（beta^2 的下限来源，避免驻点退化）
        k: mach_ref^2 的安全裕度倍数
        out: (n_cells, n_sps, 5) 输出，可以与 R 是同一个数组（就地）
    """
    n_cells, n_sps = R.shape[0], R.shape[1]
    gm1 = GAMMA - 1.0
    floor2 = k * mach_ref * mach_ref
    for c in prange(n_cells):
        for s in range(n_sps):
            rho = Q[c, s, 0]
            u = Q[c, s, 1]
            v = Q[c, s, 2]
            w = Q[c, s, 3]
            p = Q[c, s, 4]
            if rho < 1e-30 or p <= 0.0:
                # 非物理点（正性限制器尚未介入的瞬态）：不预处理，原样透传，
                # 避免在这里产生 NaN 掩盖真正的问题
                for m in range(5):
                    out[c, s, m] = R[c, s, m]
                continue
            a2 = GAMMA * p / rho
            q2 = u * u + v * v + w * w
            m2 = q2 / a2
            beta2 = m2 if m2 > floor2 else floor2
            if beta2 > 1.0:
                beta2 = 1.0
            elif beta2 < 1e-10:
                beta2 = 1e-10
            r0 = R[c, s, 0]
            r1 = R[c, s, 1]
            r2 = R[c, s, 2]
            r3 = R[c, s, 3]
            r4 = R[c, s, 4]
            # dp(r) = dp/dU . r
            dp = gm1 * (0.5 * q2 * r0 - u * r1 - v * r2 - w * r3 + r4)
            coef = (1.0 - beta2) / a2 * dp
            H = a2 / gm1 + 0.5 * q2
            out[c, s, 0] = r0 - coef
            out[c, s, 1] = r1 - coef * u
            out[c, s, 2] = r2 - coef * v
            out[c, s, 3] = r3 - coef * w
            out[c, s, 4] = r4 - coef * H


@njit(cache=True, parallel=True)
def _precond_state_jacobian_kernel(R, Q, mach_ref, k, out) -> None:
    """`out[c, s, m, v] = d(Gamma(Q) R)_m / dQ_v`（`R` 固定），逐分支与 `_apply_low_mach_precond_kernel` 对应。

    `Gamma R = R - coef (1, u, v, w, H)`，`coef = (1 - beta^2)/a^2 * dp(R)`；`a^2 = gamma p/rho`，`H = a^2/(gamma-1)
    + q^2/2`，`beta^2 = q^2/a^2`（未被钳制时；被下限 `k mach_ref^2` / 1e-10 或上限 1 钳制时对状态导数为零），
    `dp(R)` 只经速度依赖状态。非物理点原样透传，导数为零。
    """
    n_cells, n_sps = R.shape[0], R.shape[1]
    gm1 = GAMMA - 1.0
    floor2 = k * mach_ref * mach_ref
    for c in prange(n_cells):
        for s in range(n_sps):
            for m in range(5):
                for v in range(5):
                    out[c, s, m, v] = 0.0
            rho = Q[c, s, 0]
            p = Q[c, s, 4]
            if rho < 1e-30 or p <= 0.0:
                continue
            vel = Q[c, s, 1:4]
            a2 = GAMMA * p / rho
            q2 = vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]
            m2 = q2 / a2
            beta2 = m2 if m2 > floor2 else floor2
            free = m2 > floor2
            if beta2 > 1.0:
                beta2 = 1.0
                free = False
            elif beta2 < 1e-10:
                beta2 = 1e-10
                free = False
            dp = gm1 * (0.5 * q2 * R[c, s, 0] - vel[0] * R[c, s, 1] - vel[1] * R[c, s, 2] - vel[2] * R[c, s, 3]
                        + R[c, s, 4])
            sc = (1.0 - beta2) / a2
            coef = sc * dp
            H = a2 / gm1 + 0.5 * q2
            # 对 (rho, u, v, w, p) 的导数：a^2、beta^2、s、dp(R)
            da2 = np.zeros(5)
            da2[0] = -a2 / rho
            da2[4] = GAMMA / rho
            dbeta2 = np.zeros(5)
            if free:
                for v in range(5):
                    dbeta2[v] = -q2 / (a2 * a2) * da2[v]
                for j in range(3):
                    dbeta2[1 + j] = 2.0 * vel[j] / a2
            ddp = np.zeros(5)
            for j in range(3):
                ddp[1 + j] = gm1 * (vel[j] * R[c, s, 0] - R[c, s, 1 + j])
            for v in range(5):
                dcoef = (-dbeta2[v] / a2 - (1.0 - beta2) / (a2 * a2) * da2[v]) * dp + sc * ddp[v]
                out[c, s, 0, v] = -dcoef
                for j in range(3):
                    out[c, s, 1 + j, v] = -dcoef * vel[j]
                out[c, s, 4, v] = -dcoef * H - coef * da2[v] / gm1
            for j in range(3):
                out[c, s, 1 + j, 1 + j] -= coef
                out[c, s, 4, 1 + j] -= coef * vel[j]


def low_mach_preconditioner_state_jacobian(residual: np.ndarray, Q: np.ndarray, mach_ref: float,
                                           k: float = 1.1) -> np.ndarray:
    """`d(Gamma(Q) R)/dQ`（`R` 固定）：`residual` (n_cells, n_sps, 5) 与原始变量 `Q` -> (n_cells, n_sps, 5, 5)，
    `[..., 守恒分量, 原始变量]`。解析式（`_precond_state_jacobian_kernel`），供平均流解析 Jacobian
    （`core/fr_residual/jacobian/assemble.py`）使用；`k` 与 `apply_low_mach_preconditioner` 同义同默认值。"""
    R5 = np.ascontiguousarray(residual[..., :5], dtype=np.float64)
    out = np.empty(R5.shape + (5,))
    _precond_state_jacobian_kernel(R5, np.ascontiguousarray(Q[..., :5], dtype=np.float64), float(mach_ref), float(k),
                                   out)
    return out


def apply_low_mach_preconditioner(residual: np.ndarray, Q: np.ndarray,
                                  mach_ref: float, k: float = 1.1,
                                  out: Optional[np.ndarray] = None) -> np.ndarray:
    """把 Weiss-Smith 伪时间预处理矩阵 Gamma 作用到平均流残差上。

    `dU/dtau = -R` 变成 `dU/dtau = -Gamma R`。稳态不动点不变
    （det(Gamma)=beta^2>0），但 Jacobian 谱半径从 (|un|+a) 降到
    (|un|+c_precond)，因此调用方**必须同时**把 CFL 步长估计里的物理声速
    换成 `preconditioned_acoustic_eigs` 给出的 `c_precond`——两者成对出现
    才既稳定又有收益（只改其中一个的后果见本模块末尾"背景"一节记录的
    2026-08-24 真实事故）。

    数组模块无关（CPU 与 GPU 后端共用这一份）：numpy 数组走逐点 numba 核，其余数组模块（CuPy）走同一算式的
    向量化表达式（`_apply_low_mach_precond_vectorized`）。2026-10-22（I-115）以前 GPU 有一份从守恒变量自己推导
    原始变量的拷贝（已删除的 `core/gpu/gpu_preconditioning.py`），beta^2 与有效声速也各写一份。

    Args:
        residual: (n_cells, n_sps, n_vars>=5) 残差；只有前 5 个守恒变量
            分量被预处理，n_vars>5 时（如平均流与湍流紧耦合的未知量）
            其余分量原样保留——湍流标量是被动输运量，不含声学模态，
            没有需要预处理的刚性。
        Q: (n_cells, n_sps, >=5) 原始变量 (rho,u,v,w,p)
        mach_ref: 参考马赫数
        k: beta^2 下限的安全裕度倍数（与 AUSM+up 的
            `_WEISS_SMITH_K`/本模块 `preconditioned_acoustic_eigs` 默认值一致）
        out: 可选输出数组；传 `residual` 本身即为就地运算

    Returns:
        预处理后的残差（形状与输入相同）
    """
    xp = array_module(residual, Q)
    if xp is not np:
        return _apply_low_mach_precond_vectorized(xp, residual, Q, float(mach_ref), float(k), out)
    n_vars = residual.shape[2]
    R5 = np.ascontiguousarray(residual[:, :, :5])
    Q5 = np.ascontiguousarray(Q[:, :, :5])
    tgt = np.empty_like(R5)
    _apply_low_mach_precond_kernel(R5, Q5, float(mach_ref), float(k), tgt)
    if out is None:
        if n_vars == 5:
            return tgt
        out = residual.copy()
    out[:, :, :5] = tgt
    return out


def resolve_low_mach_precond(requested, time_scheme) -> bool:
    """低马赫数伪时间预处理是否启用 —— 全部后端的唯一判据（2026-09-25 收拢）。

    此前 CPU 单机、CPU-MPI 两种加载模式、单机 GPU、多 GPU 各写一份（枚举比较 /
    字符串比较 / 两者兼容三种写法），多 GPU "完全分布式加载"那条路径**根本没有
    设置**这个属性，而它的 `step()` 直接读取 —— 那条路径的每个 RK 步都会
    `AttributeError`。

    低马赫数伪时间预处理（2026-09-14 新增，用户提出"收敛需要数万步"
    后的根本性优化）。完整推导/正确性论证见
    `core/utils/preconditioning.py` 模块末尾"伪时间预处理矩阵 Gamma"
    一节；接入点见 `time_integration/step_orchestration.py::advance_step` 的 `precondition`（残差侧）与
    `cfl.py::compute_local_time_step`（步长侧）——两者**必须成对启用**。

    为什么默认开：本项目的目标工况是汽车外流场，M~0.09（33m/s vs
    声速 340m/s）。不做预处理时显式格式的 dt 被声速限制，比对流
    时间尺度小约 11 倍，收敛步数因此白付约一个数量级——这正是
    用户观察到"需要数万步"的主因之一。预处理后 dt 由预处理波速
    (|un|+c_precond) 决定，M=0.1 下放大约 5 倍。
    不动点不变（det(Gamma)=beta^2>0），收敛解与关闭时是同一个解。

    对 SSP-RK2/RK3 与 NEWTON_KRYLOV 这三个"纯稳态伪时间推进"方案启用：这里的推进只是到达 R=0 的路径，
    Gamma 只改路径、不改不动点。DUAL_TIME 不启用（请求了也只打一条日志说明），显式格式的时间精确推进同样
    不启用（`time_integration/time_accurate.py::configure_time_stepping`）：
    * DUAL_TIME 的内迭代是带 BDF 物理时间项的 Newton 步（`time_integration/implicit/newton/physical_time.py`），
      对角上的 `a0/dt` 已经让系统良态，伪时间项只是步被拒时的回退手段，不需要 Gamma；"整体残差乘 Gamma"
      会把物理时间导数项一起乘掉，解的就不是 BDF 方程了；

    NEWTON_KRYLOV（2026-09-25 加入）：Newton 解的是 F(U)=Gamma(U)R(U)=0，Gamma
    可逆，不动点不变；伪瞬态延拓项随之被预处理（标准的"预处理隐式"做法），
    声学模态不再刚性。不加时的真实表现（plate_demo P0、层流、CFL 20）：残差
    4.7e8 降到 ~3e6 后停滞，驻点区压力在 Cp=+15 与 −11 之间周期振荡（封闭风洞里
    被入口/出口/洞壁反射的弱阻尼声学模态）；加上后同样 120 步残差降到 9.5e5 且
    仍在下降，驻点区 Cp≈0.92（物理值≈1）。其余方案不启用的原因：
    开关来自构造参数（CLI `solve steady --low-mach-precond/--no-low-mach-precond`、配置字段
    `SteadyConfig.low_mach_precond`），随 checkpoint 记录、续算默认沿用（`solve resume` 同名选项可覆盖）。
    2026-10-18 以前另有环境变量 `AFCFD_LOW_MACH_PRECOND` 且优先于构造参数：checkpoint 记不下它，
    关掉预处理跑出的结果续算时会静默换回开启。

    Args:
        requested: 构造参数/分布式包里请求的值。
        time_scheme: 时间积分方案（枚举或用户侧字符串）。
    """
    from autoflowcfd.core.time_integration.base import TimeIntegrationScheme, scheme_from_name

    req = bool(requested)
    scheme = scheme_from_name(time_scheme)
    applicable = scheme in (
        TimeIntegrationScheme.SSP_RK2, TimeIntegrationScheme.SSP_RK3,
        TimeIntegrationScheme.NEWTON_KRYLOV)
    if req and not applicable:
        from loguru import logger
        logger.info(
            f"低马赫预处理不用于时间格式 {scheme.name}（只用于稳态伪时间推进 SSP-RK2/RK3 与 NEWTON_KRYLOV；"
            f"物理时间精度不受影响，低马赫下内迭代/子迭代收敛较慢）")
    return req and applicable


def _apply_low_mach_precond_vectorized(xp, residual, Q, mach_ref: float, k: float, out=None):
    """`_apply_low_mach_precond_kernel` 的向量化表达式（逐点算式与运算顺序逐项相同，非物理点同样原样透传），
    `out` 语义与 `apply_low_mach_preconditioner` 相同（None 返回新数组；可以就是 residual，就地写）。

    就地写安全：`coef`、`u/v/w`、`H` 在任何写回之前都已物化成独立数组，第 i 个分量的新值只依赖 `r_i` 自己；
    逐分量写回不需要先攒 5 份临时数组（那会多占 5 个全场数组的显存，正好抵消 out= 想省的那一份）。
    """
    gm1 = GAMMA - 1.0
    rho, u, v, w, p = (Q[..., m] for m in range(5))
    physical = (rho >= 1e-30) & (p > 0.0)
    a2 = GAMMA * xp.where(physical, p, 1.0) / xp.where(physical, rho, 1.0)
    q2 = u * u + v * v + w * w
    beta2 = _precond_beta2(q2, a2, mach_ref, k)
    r0, r1, r2, r3, r4 = (residual[..., m] for m in range(5))
    dp = gm1 * (0.5 * q2 * r0 - u * r1 - v * r2 - w * r3 + r4)
    coef = (1.0 - beta2) / a2 * dp
    H = a2 / gm1 + 0.5 * q2
    if out is None:
        out = residual.copy()
    elif out is not residual:
        out[...] = residual
    out[..., 0] = xp.where(physical, r0 - coef, r0)
    out[..., 1] = xp.where(physical, r1 - coef * u, r1)
    out[..., 2] = xp.where(physical, r2 - coef * v, r2)
    out[..., 3] = xp.where(physical, r3 - coef * w, r3)
    out[..., 4] = xp.where(physical, r4 - coef * H, r4)
    return out
