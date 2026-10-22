"""AutoFlowCFD V2.0 - Newton-Krylov 稳态的全局化：残差范数与步接受判据。

`jfnk.py::step_newton_krylov` 的每个 Newton 步用同一个残差范数做三件事：残差接受判据
（`accept_step`）、SER 自适应 CFL 的输入（`adaptive_cfl/ser.py`），以及线性求解的最小化
范数（`linear_solve.py::_solve_direction`）。三者必须一致，见 `ResidualNorm` 文档。
"""

from typing import Any, Callable, Tuple

import numpy as np

from ..reductions import LocalReductions

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

#: 线性求解至少要把线性残差降到这个比例以下，Newton 方向才算可信（SU2 自适应 CFL 的同一判据：
#: 线性求解没能把残差减半就缩 CFL）。没达到容差且高于它的方向**不走出去**（`jfnk.py` 按未接受
#: 处理、缩 `dtau` 重解）：这样的 `dU` 只是 Krylov 子空间里让线性残差略降的一个向量，与 Newton
#: 方向无关，而残差接受判据拦不住它——cube_demo SA P1（37.4 万单元，CFL 300 续算）第 3 步 GMRES
#: 用满迭代数、线性残差只到 0.90（恢复 nu_tilde 更新限幅的对照组）/ 0.72（不限幅），步照样被
#: 接受（`R_new/R` 0.92 / 1.05，范数由单个贴壁棱柱的湍流残差主导），一步之内 Cd 0.736 -> 0.457 /
#: 0.636、平均流残差 1.24 -> 3.3 / 1.28 -> 8.4；不限幅组此后 CFL 减到 37 线性求解仍失败，三步后
#: 平均流残差 264、最大速度 394 m/s。越界一次之后收缩救不回来，所以必须在走出去之前拦。
LINEAR_SOLVE_MAX_REL_RESIDUAL = 0.5

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
    半范数（该处文档）。线性求解在同一个加权范数里最小化（`linear_solve.py::_solve_direction`）。

    **逐方程按参考量级无量纲化**（`scales`，2026-10-11）：各列先除以该未知量的参考量级（平均流
    `residual_diagnostics._reference_scales`：rho_inf、rho_inf U_inf、p_inf；湍流列为场的实际量级，
    `coupled_step.py::residual_reference_scales`）再求
    加权平方和。此前直接对有量纲的残差求和：能量方程的残差比质量方程大 `p_inf / rho_inf`（约 1e5）倍、比
    动量方程大 `p_inf / (rho_inf U_inf)`（约 3e3）倍，范数实际只是能量方程的范数——cube_demo SA P1 上能量
    方程占 `sum w r^2` 的 1.000，其余五个方程合计 0.000。接受判据、SER、GMRES 最小化的都是它，质量、动量、
    湍流方程收敛与否不进判据。`scales` 为 None 时不缩放（各列量纲相同的调用方）。
    """

    __slots__ = ("weights", "red", "inv_scales")

    def __init__(self, weights, red: LocalReductions, scales=None):
        self.red = red
        self.weights = None if weights is None else red.xp.asarray(weights, dtype=red.xp.float64).ravel()
        if scales is None:
            self.inv_scales = None
        else:
            s = np.asarray(scales, dtype=np.float64).ravel()
            if not (np.all(np.isfinite(s)) and np.all(s > 0.0)):
                raise ValueError(f"残差范数的参考量级必须为正有限值，收到 {s}")
            self.inv_scales = red.xp.asarray(1.0 / s)

    def restricted(self, columns) -> "ResidualNorm":
        """同一组权重、只取 `columns` 这些列的范数（紧耦合步里平均流 / 湍流各自的残差范数）。"""
        out = ResidualNorm(None, self.red)
        out.weights = self.weights
        out.inv_scales = None if self.inv_scales is None else self.inv_scales[columns]
        return out

    def __call__(self, r) -> float:
        if self.inv_scales is not None:
            r = r * self.inv_scales[None, :]
        if self.weights is None:
            return self.red.rms(r)
        s = self.red.sum(self.weights)
        if s <= 0.0:
            return 0.0
        return float(np.sqrt(self.red.sum(self.weights[:, None] * r * r) / (s * r.shape[1])))


def accept_step(residual: Callable, u0_flat, du_flat, theta0: float, res_norm0: float,
                red: LocalReductions, norm: "ResidualNorm") -> Tuple[Any, float, float, int]:
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
    `time_integration/adaptive_cfl/` 包文档第 8 条）。

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
