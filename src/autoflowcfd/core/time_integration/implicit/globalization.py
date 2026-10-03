"""AutoFlowCFD V2.0 - Newton-Krylov 稳态的全局化：残差范数与步接受判据。

`jfnk.py::step_newton_krylov` 的每个 Newton 步用同一个残差范数做三件事：残差接受判据
（`accept_step`）、SER 自适应 CFL 的输入（`adaptive_cfl/ser.py`），以及线性求解的最小化
范数（`jfnk.py::_solve_direction`）。三者必须一致，见 `ResidualNorm` 文档。
"""

from typing import Callable, Tuple

import numpy as np

from .reductions import LocalReductions

#: 残差接受判据：允许一步之后 `||R||` 相对恶化的上限。
#:
#: 伪瞬态延拓不是对某个 merit function 做线搜索，所以**不能**要求残差
#: 严格单调下降 —— 真实瞬态本身会让它先升后降（Blasius 从均匀初场起步
#: 就是这样）。但也不能什么都接受：那正是"CFL 10 跑到 6.5e19"的来源。
#:
#: 取 1.5 的含义：一步之内残差恶化超过 50% 就认为这一步的方向不可信，
#: 回溯 `theta`。它足够松以容纳真实瞬态的上升（实测显式路径上单步的
#: 残差增幅远小于此），又足够紧以立刻拦住发散。
RESIDUAL_ACCEPT_GROWTH = 1.5

#: 回溯次数上限。每次回溯把 `theta` 减半并重算一次残差，所以代价是
#: 最多这么多次额外残差求值。4 次即 `theta` 最小到 1/16。
MAX_BACKTRACK = 4


class ResidualNorm:
    """Newton 全局化用的残差范数：逐行权重 `w`（单元体积在真实解点上均分，补零槽位
    为 0）下的体积加权 RMS `sqrt(sum w r^2 / (sum w * n_var))`；`w` 为 None
    时是逐点 RMS。

    **为什么要体积加权**（2026-09-26）：残差按 `dU/dt = 残差/体积` 存，逐点 RMS 把
    体积相差 1e6 倍的单元等权相加，被最小的单元主导。plate_demo P1：平板锐边一带
    6.57% 的单元占 `||Gamma R||^2` 的 78.6%，接受判据与 SER 自适应 CFL 被这几百个
    欠分辨角点单元的不规则行为牵着走，CFL 卡在 20~60。体积加权的积分范数就是
    有限体积代码的通量不平衡量（SU2/FUN3D 的残差），是物理上有意义的全域度量。

    权重取单元体积在真实解点上均分（`positivity/limiter.py::PositivityLimiter.norm_weights`，2026-10-03），
    不取解点插值型求积权重本身：原生基 P2/P3 的那套权重有零与负值，加权平方和只是
    半范数（该处文档）。线性求解在同一个加权范数里最小化（`jfnk.py::_solve_direction`）。
    """

    __slots__ = ("weights", "red")

    def __init__(self, weights, red: LocalReductions):
        self.red = red
        self.weights = None if weights is None else red.xp.asarray(weights, dtype=red.xp.float64).ravel()

    def __call__(self, r) -> float:
        if self.weights is None:
            return self.red.rms(r)
        s = self.red.sum(self.weights)
        if s <= 0.0:
            return 0.0
        return float(np.sqrt(self.red.sum(self.weights[:, None] * r * r) / (s * r.shape[1])))


def accept_step(residual: Callable, u0_flat, du_flat, theta0: float, res_norm0: float,
                 red: LocalReductions, norm: "ResidualNorm") -> Tuple[object, float, float, int]:
    """按残差接受判据回溯 `theta`，返回
    `(U_new, theta, res_norm_new, n_extra_residual_eval)`。

    从 `theta0`（物理性限幅给出的上界）开始，每次不被接受就减半，最多
    `MAX_BACKTRACK` 次。接受条件是

        ||R(U0 + theta*dU)||  <=  RESIDUAL_ACCEPT_GROWTH * ||R(U0)||

    全部回溯都不被接受时返回 `theta = 0`（**这一步不前进**）。调用方
    `step_newton_krylov` 据此**当场缩小 `dtau` 重解一次**（见
    `dtau_control.py`），而不是把状态原样交出去 —— 后者被真实运行证明
    会永久停滞。不"硬着头皮走一步"是刻意的：本项目已经吃过一次"越界
    之后收缩救不回来"的亏（项目记忆
    `adaptive_cfl_four_defects_and_soft_ceiling` 第 12 条）。

    为什么不用标准线搜索（Armijo）：那需要一个 merit function
    （通常 `0.5||R||^2`）与它的方向导数，而伪瞬态解的不是
    `min ||R||^2` 而是 `(I/dtau + J) dU = -R`；在 `dtau` 小的时候
    `dU` 根本不是 `||R||^2` 的下降方向（它是时间推进方向）。所以这里用的
    是"不允许显著恶化"这个更弱、但与 PTC 语义相容的判据。
    """
    theta = float(theta0)
    n_eval = 0
    for _ in range(MAX_BACKTRACK + 1):
        if theta <= 0.0:
            break
        u_try = u0_flat + theta * du_flat
        r_try = residual(u_try)
        n_eval += 1
        if red.all_finite(r_try):
            rn = norm(r_try)
            if rn <= RESIDUAL_ACCEPT_GROWTH * res_norm0:
                return u_try, theta, rn, n_eval
        theta *= 0.5
    return u0_flat, 0.0, res_norm0, n_eval
