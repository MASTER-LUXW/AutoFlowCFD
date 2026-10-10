"""AutoFlowCFD V2.0 - 双时间步的隐式内迭代：BDF 物理时间项并入 Newton-Krylov（四个后端共用）。

## 解的方程

每个物理时间步解（`R` 是空间残差，`dU/dt = -R`；第一个物理步 BDF1，之后 BDF2）

    G(U) = R(U) + (a0 U - b) / dt = 0
    BDF1：a0 = 1，  b = U^n
    BDF2：a0 = 3/2，b = 2 U^n - U^{n-1} / 2

未知量就是稳态 Newton 步的那一组：平均流守恒变量，带输运方程的湍流模型再接上模型的 Newton 未知量
（SST 的 k 与 ln omega、SA 的 nu_tilde），平均流与湍流紧耦合、同一个物理时间。

## 怎么解：复用稳态的 PTC-Newton-Krylov 步

稳态步解 `(I/dtau + J_R) dU = -R`。这里每次内迭代解

    (I/dtau + a0/dt + J_R) dU = -G(U)

物理时间项只是对角上多一个 `a0/dt`：`jfnk.py::step_newton_krylov` 把它并进预处理与 Krylov 算子用的
等效步长 `1/(1/dtau + a0/dt)`，单元块 Jacobian（`J_R`）与块 ILU 原样复用；右端、残差接受判据、SER 的
残差范数用 `G`。伪时间 CFL 从 `DUAL_TIME_PSEUDO_CFL_START` 起步（每个物理步重新起步）：`a0/dt` 已经让
系统对角占优，`dtau` 只是 Newton 被拒时的回退手段。

## 为什么不用显式伪时间内迭代（2026-10-10 以前的做法）

显式内迭代的伪时间步受最小单元的稳定极限约束（CFL 0.03 量级）。带壁面边界层的网格上，默认 20 次内迭代
只把伪残差降不到 1%——cube_demo 的 DDES/SA/LES 瞬态每个物理步都在报"内迭代未收敛"，得到的不是 BDF 方程
的解，时间精度没有保证。隐式内迭代的步数与最小单元无关。

物理时间项还让稳态步解不动的情形变得可解：定常解线性不稳定时 `I/dtau + J_R` 在 `dtau ~ 1/增长率` 处
接近奇异（`mean_flow_step.py::note_linear_growth`），而 `dt` 小于不稳定模态的时间尺度时 `a0/dt + J_R`
是良态的——不稳定模态被时间分辨，而不是被绕开。
"""

import time
from typing import Callable, Optional

from loguru import logger


#: 内迭代里线性求解（线性残差降不到 `LINEAR_SOLVE_MAX_REL_RESIDUAL`）连续失败这么多次就报错停下
MAX_CONSECUTIVE_LINEAR_FAILURES = 3


class BdfTerm:
    """一个物理步的 BDF 物理时间项（模块文档）。`x_now`/`x_prev` 是 Newton 未知量排列的状态
    `(N, n_var)`（`x_prev` 为 None 时用 BDF1）。"""

    __slots__ = ("a0", "dt", "b_over_dt")

    def __init__(self, x_now, x_prev, dt: float):
        if not dt > 0.0:
            raise ValueError(f"物理时间步长必须为正，收到 {dt}")
        self.dt = float(dt)
        if x_prev is None:
            self.a0, b = 1.0, x_now
        else:
            if x_prev.shape != x_now.shape:
                raise ValueError(f"上一时间层形状 {x_prev.shape} 与当前状态 {x_now.shape} 不符")
            self.a0, b = 1.5, 2.0 * x_now - 0.5 * x_prev
        self.b_over_dt = b / self.dt

    @property
    def shift(self) -> float:
        """`dG/dU` 相对 `dR/dU` 多出的对角项 `a0/dt`。"""
        return self.a0 / self.dt

    def bind(self, spatial: Callable, real_rows) -> "BdfResidual":
        return BdfResidual(spatial, self.shift, self.b_over_dt, real_rows)


class BdfResidual:
    """`G(x) = R(x) + (a0 x - b)/dt`。零填充槽位（`real_rows` 为 False 的行）冻结在初值、残差恒为零，
    物理时间项在那里同样置零。"""

    __slots__ = ("spatial", "shift", "b_over_dt", "pad")

    def __init__(self, spatial: Callable, shift: float, b_over_dt, real_rows):
        self.spatial, self.shift, self.b_over_dt = spatial, float(shift), b_over_dt
        self.pad = None if real_rows is None else ~real_rows

    def __call__(self, x):
        out = self.spatial(x) + (self.shift * x - self.b_over_dt)
        if self.pad is not None:
            out[self.pad] = 0.0
        return out


def previous_newton_state(solver, xp, n_cells: int, n_sps: int):
    """上一物理时间层的 Newton 未知量 `(N, 5 + n_turb)`；没有（第一个物理步、或换了湍流模型后只恢复出
    平均流）时 None。平均流与湍流的上一时间层分开存在求解器上（`_dual_time_U_prev`、
    `_dual_time_turb_prev`，checkpoint 按这两个属性读写，`core/utils/checkpoint_time.py`）。"""
    u_prev = getattr(solver, "_dual_time_U_prev", None)
    if u_prev is None:
        return None
    u_prev = xp.asarray(u_prev).reshape(n_cells * n_sps, -1)
    t_prev = getattr(solver, "_dual_time_turb_prev", None)
    if t_prev is None:
        return u_prev
    return xp.concatenate([u_prev] + [xp.asarray(t).reshape(n_cells * n_sps, 1) for t in t_prev], axis=1)


def store_newton_state(solver, x_now, n_cells: int, n_sps: int, n_mean: int) -> None:
    """把本物理步开始时的 Newton 未知量存成下一步的上一时间层（`previous_newton_state` 的逆）。"""
    solver._dual_time_U_prev = x_now[:, :n_mean].copy()
    n_turb = x_now.shape[1] - n_mean
    solver._dual_time_turb_prev = (tuple(x_now[:, n_mean + j].reshape(n_cells, n_sps).copy() for j in range(n_turb))
                                   if n_turb else None)


def advance_physical_step(solver, x_now, x_prev, newton_step: Callable, dt: float) -> dict:
    """推进一个物理时间步：反复调用后端的 Newton 步，直到 `||G||` 降到该物理步起点的
    `time_integrator.dual_time_tolerance` 倍、或一个被完整接受的 Newton 步的更新量小于
    `DUAL_TIME_UPDATE_FLOOR`（`time_integration/base.py`），或用完内迭代次数。

    Args:
        solver: 持有 `time_integrator.dual_time_steps`（内迭代上限）与 `dual_time_tolerance`、
            `_cfl_controller`（伪时间 SER，可为 None 即固定伪时间 CFL）与 Newton 跨步状态的求解器。
        x_now: 本物理步开始时的 Newton 未知量 `(N, n_var)`（调用方保证之后不被原地改写）。
        x_prev: 上一物理时间层（`previous_newton_state`），None 时 BDF1；列数与 `x_now` 不符时
            （换了湍流模型）同样退回 BDF1。
        newton_step: `(BdfTerm, first: bool) -> info`——后端的一个 Newton 步（内部按当前伪时间 CFL
            取 `dtau`、把新状态写回求解器）；`first` 为真时推进产生项渐变（渐变按物理步计）。
        dt: 物理时间步长。

    Returns:
        `{"iterations", "initial", "final", "converged"}`（也记在 `solver._dual_time_inner`）。
    """
    from ..base import DUAL_TIME_UPDATE_FLOOR
    from .mean_flow_step import LINEAR_SOLVE_MAX_REL_RESIDUAL, newton_step_ok

    max_inner = int(solver.time_integrator.dual_time_steps)
    tolerance = float(solver.time_integrator.dual_time_tolerance)
    if max_inner < 1:
        raise ValueError(f"双时间步内迭代次数必须 >= 1，收到 {max_inner}")
    if x_prev is not None and x_prev.shape != x_now.shape:
        x_prev = None
    term = BdfTerm(x_now, x_prev, dt)
    controller = getattr(solver, "_cfl_controller", None)
    if controller is not None:
        controller.restart()
    # forcing 记的是上一步的残差量级：新物理步的 `G` 与上一步收敛时的不可比
    solver._newton_forcing = None

    initial: Optional[float] = None
    final = float("nan")
    iterations = 0
    converged = False
    linear_failures = 0
    for k in range(max_inner):
        t_start = time.perf_counter()
        info = newton_step(term, k == 0)
        iterations = k + 1
        if initial is None:
            initial = float(info["res_norm"])
        final = float(info["res_norm_new"])
        step_ok = newton_step_ok(info)
        if controller is not None:
            controller.update(info["res_norm"], step_ok=step_ok)
        # 每次内迭代一行：物理步的监控行要等内迭代全部结束才打印，大算例上一步可能是几十分钟
        logger.info("[dual-time] inner %d: |G| %.3e -> %.3e | GMRES %d (rel %.1e) | theta %.3g | dtau cuts %d | CFL %s"
                    " | %.1f s"
                    % (iterations, float(info["res_norm"]), final, int(info["gmres_iters"]),
                       float(info["linear_rel_residual"]), float(info["theta"]), int(info["n_dtau_cuts"]),
                       "%.3g" % controller.cfl_number if controller is not None else "fixed",
                       time.perf_counter() - t_start))
        # 线性求解连续失败：伪时间 CFL 已经在逐次减半（SER），仍然解不动说明 `a0/dt + J` 不定——物理时间
        # 步长大于最快不稳定模态的时间尺度。这种系统在伪时间里也不收敛（不稳定模态在伪时间里同样增长），
        # 继续内迭代只是烧时间，带着没解出来的状态进下一个物理步更没有意义
        linear_failures = (linear_failures + 1
                           if float(info["linear_rel_residual"]) > LINEAR_SOLVE_MAX_REL_RESIDUAL else 0)
        if linear_failures >= MAX_CONSECUTIVE_LINEAR_FAILURES:
            raise RuntimeError(
                "双时间步内迭代的线性求解连续 %d 次失败（最后一次线性残差只降到 %.2f，||G|| %.3e -> %.3e）。"
                "物理时间步长 dt=%.3e 大于流动里最快的不稳定模态的时间尺度时 a0/dt + dR/dU 不定，Newton 的"
                "线性系统解不动。请减小 --dt（经验：先减到 1/3）。"
                % (linear_failures, float(info["linear_rel_residual"]), initial, final, term.dt))
        converged = bool(final <= tolerance * initial
                         or (step_ok and float(info["update_norm"]) <= DUAL_TIME_UPDATE_FLOOR))
        if converged:
            break
    summary = dict(iterations=iterations, initial=initial, final=final, converged=converged, a0=term.a0)
    solver._dual_time_inner = summary
    if not converged:
        logger.warning(
            "双时间步内迭代 %d 次后 ||G|| 只从 %.3e 降到 %.3e（判据 %.0e 倍）：这个物理步的 BDF 方程没有解到"
            "容差。加大 --dual-time-inner-iter 或减小 --dt。" % (iterations, initial, final, tolerance))
    return summary


def inner_iteration_suffix(solver) -> str:
    """迭代监控行里本物理步内迭代的摘要；不是隐式双时间步时为空串。"""
    s = getattr(solver, "_dual_time_inner", None)
    if not s:
        return ""
    drop = (s["initial"] / s["final"]) if s["final"] > 0.0 else float("inf")
    return f" | inner={s['iterations']}{'' if s['converged'] else '(未收敛)'} |G| {s['initial']:.3e}->{s['final']:.3e} ({drop:.1e}x)"


__all__ = ["BdfTerm", "BdfResidual", "advance_physical_step",
           "inner_iteration_suffix", "previous_newton_state", "store_newton_state"]
