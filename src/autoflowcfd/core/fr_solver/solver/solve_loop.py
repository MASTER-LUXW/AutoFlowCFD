"""AutoFlowCFD V2.0 - 求解循环（mixin，只含方法）。

从 `core/fr_solver/solver.py` 拆出（2026-09-25）。`SolveLoopMixin` 与后端无关，四个后端共用：CPU `FRSolver`、
单 GPU `GPUFRSolver`（2026-10-04 起）、CPU MPI `DistributedFRSolver` 与多 GPU `MultiGPUDistributedSolver`
（2026-10-05 起）。此前单 GPU、CPU 分布式、多 GPU 各有一份循环：单 GPU 与多 GPU 返回字典，两个分布式循环
的定阶收敛判据是绝对的 `res < tol`（单机与全部 Order Continuation 路径是相对下降 1/tol 倍，同一个 `tol`
两种含义），CPU 分布式不记收敛历史、步数参数名是 `n_steps`。

后端差异只通过这几个接口进入：`step()`（返回全域残差范数，分布式各 rank 一致）、
`_solve_with_order_continuation()`、`host_view()`（每步气动力系数读主机上的状态，只在设置了
`_reference_area` 时调用）、`_scaled_residual_field()`（逐方程缩放残差诊断的数据源）、
`_loop_monitor_suffix()` 与 `_divergence_hint()`（日志与发散报错的后端附注）。日志只在 root rank 打印。
"""

from contextlib import contextmanager
from typing import Optional

import numpy as np

from autoflowcfd.core.mpi import is_root
from autoflowcfd.core.utils.order_continuation.policy import uses_order_continuation
from autoflowcfd.core.time_integration.implicit.mean_flow_step import newton_monitor_suffix
from autoflowcfd.core.fr_solver.residual_diagnostics import check_residual_finite
from autoflowcfd.core.fr_solver.state import SolverResult
from autoflowcfd.core.utils import order_continuation

from .. import step as fr_solver_step
from autoflowcfd.core.time_integration.base import DEFAULT_STEADY_TOL

from .threads import blas_threads_limited
from autoflowcfd.core.turbulence.dual_time import reset_dual_time_history


def format_step_line(solver, order: int, iteration: int, res: float, initial_res: float, elapsed: float,
                     n_steps: int) -> str:
    """每步一行日志（定阶循环 `SolveLoopMixin.solve` 与 Order Continuation 共用，此前两边各写一份）。

    残差、相对本阶段起点的下降倍数、单步耗时、CFL、Newton 诊断、后端附注；设置了 `_reference_area`
    时附压力积分气动力系数（经 `host_view()` 读主机状态）。第 1 步与之后每 10 步再附逐方程缩放残差与最大
    残差定位（参照 Fluent scaled residuals / STAR-CCM+ Max 监视器；最大残差单元的体积分位是区分"退化单元
    机制"与"壁面处理机制"最直接的指标）和累计伪时间预算（残差下降不等于物理场已建立，见
    `pseudotime_budget.py` 模块文档）——数据源由后端钩子 `_scaled_residual_field()` 给出。

    Args:
        order: 本阶段阶数；iteration: 本阶段内步号（1 起）；n_steps: 本次求解累计步数（伪时间预算用）。
    """
    drop = initial_res / max(res, 1e-30)
    msg = f"P{order} Iter {iteration}: Residual = {res:.6e} | Drop: {drop:.1f}x | Time: {elapsed:.2f}s"
    ctrl = getattr(solver, "_cfl_controller", None)
    if ctrl is not None:
        msg += f" | CFL={ctrl.cfl_number:.3f}"
    msg += newton_monitor_suffix(solver) + solver._loop_monitor_suffix()
    ref_area = getattr(solver, "_reference_area", None)
    if ref_area is not None and ref_area > 0:
        from autoflowcfd.postprocess.fr_coefficients import compute_forces_pressure_only

        aero = compute_forces_pressure_only(solver.host_view(), ref_area)
        msg += f" | Cd={aero['Cd']:.4f} Cl={aero['Cl']:.4f} Cs={aero['Cs']:.4f}"
    freestream = getattr(solver, "freestream", None)
    field = solver._scaled_residual_field() if (iteration == 1 or iteration % 10 == 0) else None
    if freestream is not None and field is not None:
        from autoflowcfd.core.fr_solver.residual_diagnostics import (
            compute_scaled_residuals, format_scaled_residual_line,
        )
        from autoflowcfd.core.fr_solver.pseudotime_budget import format_pseudo_time_budget

        msg += " | " + format_scaled_residual_line(
            compute_scaled_residuals(field, freestream),
            cell_volumes=getattr(getattr(solver, "mesh", None), "cell_volumes", None))
        budget = solver._pseudo_time_budget(n_steps=n_steps)
        if budget is not None:
            msg += " | " + format_pseudo_time_budget(budget, compact=True)
    return msg


def report_pseudo_time_summary(solver, n_steps: int) -> None:
    """收尾时把"物理场到底走了多远"完整报一次（两个循环共用）。"""
    budget = solver._pseudo_time_budget(n_steps=n_steps)
    if budget is not None:
        from autoflowcfd.core.fr_solver.pseudotime_budget import format_pseudo_time_budget

        print(format_pseudo_time_budget(budget))


class SolveLoopMixin:
    """与后端无关的求解循环（见模块文档）。"""

    def _loop_monitor_suffix(self) -> str:
        """每步日志末尾的后端信息（CPU 无；GPU 打印显存占用）。"""
        return ""

    def set_dual_time_history(self, U_prev_flat) -> None:
        """恢复 dual-time 的上一物理时间层（续算，见 `core/utils/checkpoint_time.py`）：CPU 后端存主机数组，
        GPU 后端覆盖为设备数组。形状 `(本 rank 单元数 * n_sps, n_vars)`。"""
        self._dual_time_U_prev = np.ascontiguousarray(U_prev_flat, dtype=np.float64)

    def _scaled_residual_field(self):
        """逐方程缩放残差诊断（每 10 步一行）的数据源：完整的 `(n_cells, n_sps, 5)` 主机残差，没有时 None。

        只有 CPU 单机保留着整场物理残差；GPU 不为一行诊断把残差拷回主机，分布式各 rank 只有本分区的
        残差（按本分区算出的分位与最大残差单元会被误读成全域结果）。
        """
        return None

    def _divergence_hint(self) -> str:
        """残差非有限时报错信息的后端附注。"""
        return ""

    def _solve_with_order_continuation(self, max_iter: int, dt: float, tol: float,
                                       checkpoint_callback=None,
                                       phase_max_iter: Optional[int] = None,
                                       residual_drop_threshold: float = 1e2):
        """Order Continuation（四个后端同一个循环，`core/utils/order_continuation/run.py`）。"""
        return order_continuation.run_order_continuation(
            self, max_iter, dt, tol, checkpoint_callback,
            phase_max_iter=phase_max_iter, residual_drop_threshold=residual_drop_threshold)

    def _pseudo_time_budget(self, n_steps: int):
        """本次求解已推进的伪时间与各层物理时标之比；拿不到就返回 None。

        为什么这个量必须被报告：局部时间步进下不存在单一的"当前时间"，
        而残差范数只说"离散方程的不平衡量在变小"，完全不说"物理场走了
        多远"。两者可以同时成立且互不矛盾——plate_demo 上残差单调下降
        350 步、而物理场只走完一个绕板特征时间的 **1.8%**，导致启动暂态的
        压力分布被当成壁面处理缺陷追了好几天。完整记录与那次误判用错的
        定常判据见 `pseudotime_budget.py` 模块文档。

        `L_body` 取 `sqrt(reference_area)`（气动系数已经算过参考面积），
        拿不到时只报全域尺度并在输出里标明，不拿域尺度冒充物体尺度。
        """
        tau = getattr(self, "tau_accum", None)
        if tau is None:
            return None
        try:
            from autoflowcfd.core.fr_solver.pseudotime_budget import (
                _extent, pseudo_time_budget,
            )

            fs = getattr(self, "freestream", None) or {}
            vel_inf = float(fs.get("vel_inf", 0.0) or 0.0)
            if vel_inf <= 0.0:
                return None
            ref_area = getattr(self, "_reference_area", None)
            body_len = float(np.sqrt(ref_area)) if (
                ref_area is not None and ref_area > 0) else None
            mesh = getattr(self, "mesh", None)
            dt_cell = getattr(self, "dt_cell_last", None)
            if dt_cell is None:
                return None
            return pseudo_time_budget(
                dt_cell,
                vel_inf=vel_inf,
                cell_volumes=getattr(mesh, "cell_volumes", None),
                body_length=body_len,
                domain_length=_extent(mesh) if mesh is not None else None,
                tau_accum=tau,
                n_steps=n_steps,
            )
        except Exception:
            # 纯诊断量，任何失败都不该影响求解本身；但**不静默**——
            # 打一次警告，否则"这行诊断怎么不见了"无从查。
            from loguru import logger

            logger.opt(exception=True).warning(
                "伪时间预算诊断计算失败（只影响这行日志，不影响求解）")
            return None

    def solve(self, max_iter: int = 1000, dt: float = 1e-4, tol: float = DEFAULT_STEADY_TOL,
              checkpoint_callback=None,
              phase_max_iter: Optional[int] = None,
              residual_drop_threshold: float = 1e2) -> SolverResult:
        """
        执行稳态/瞬态求解循环。

        Args:
            max_iter: 最大迭代次数
            dt: 时间步长
            tol: 相对收敛容差——残差需相对初始值下降 1/tol 倍才算收敛。
                默认 1e-6 表示下降 6 个量级（与 Order Continuation 各阶段
                的 phase_tol 阶数缩放配合：P0 下降 4 级、P1 下降 5 级、P2 下降 6 级）。
                此前为绝对判据 res < tol，对 RMS ~1e8 的流动永远不可达。
            checkpoint_callback: 可选的中间 checkpoint 回调函数，
                签名为 callback(solver, iteration_number)，每步迭代后调用。
                用于在求解过程中定期保存状态到磁盘。
            phase_max_iter: 仅 Order Continuation（`uses_order_continuation` 为真时）生效，
                非最终阶段（P0/P1/...）各自的最大迭代步数上限，None 时保留
                旧行为（`max_iter // len(orders)` 按阶段数均分）——见
                `order_continuation.run_order_continuation` 同名参数文档。
            residual_drop_threshold: 仅 Order Continuation 生效，单阶段
                判定"可以提前升阶"的残差下降倍数，默认 1e2（降 2 个数量级），
                原来硬编码，现在可配置。

        Returns:
            SolverResult: 包含收敛状态、最终残差和迭代次数的结果对象
        """
        # 累计伪时间的起点（2026-09-24 更正语义）：`tau_accum` 要回答的是
        # **物理场走了多远**，而物理场是跨 `solve resume` 延续的 —— 所以
        # 原来那句"每次 solve() 一律清零"对 resume 接力的长程算例是错的
        # （长程算例正常就是靠 resume 接力跑）。
        #
        # `_tau_accum_seeded` 由 `rebuild_solver_from_checkpoint` 在成功
        # 从 checkpoint 恢复 tau 之后置上；这里**消费掉**这个标记，于是
        # 同一个 solver 对象上第二次全新 `solve()` 仍然会正常清零。
        # 旧版本 checkpoint 没有这个字段时标记不会被置上，行为与改动前
        # 完全一致。打印时始终标注步数（见 `pseudotime_budget.py`）。
        # 物理时间步长（dual-time 的 checkpoint 写出它，续算沿用；见 core/utils/checkpoint_time.py）
        self._physical_dt = float(dt)
        if getattr(self, "_tau_accum_seeded", False):
            self._tau_accum_seeded = False
        else:
            self.tau_accum = None

        report = is_root()
        if report:
            logger_msg = f"Starting solve loop with {self.time_integrator.scheme.value}"
            if self.turb_model_name != "NONE":
                logger_msg += f", turbulence={self.turb_model_name}"
            print(logger_msg)

        # Order Continuation: 从低阶开始逐步提升精度
        if uses_order_continuation(self):
            # Order Continuation 路径在 `core/utils/order_continuation.py`
            # 里自己套 `blas_threads_limited`（求解循环在那边）。
            return self._solve_with_order_continuation(
                max_iter, dt, tol, checkpoint_callback,
                phase_max_iter=phase_max_iter,
                residual_drop_threshold=residual_drop_threshold,
            )

        import time
        converged = False
        final_residual = 1e10
        # 残差下降基准：定阶循环就是单个阶段，与 Order Continuation 共用 `_phase_initial_residual`（随 checkpoint
        # 持久化，`order_continuation/checkpoint_state.py`）。续算时用恢复出的基准做种子——此前定阶循环不写它，
        # 定阶运行（例如 --init-from 起步的瞬态）的 checkpoint 续算时被报成"没有阶段起始残差记录"，相对收敛
        # 判据也从续算的第一步重新起算。
        initial_res = (getattr(self, "_phase_initial_residual", None)
                       if getattr(self, "_resumed_from_checkpoint", False) else None)

        # BLAS 线程数只在求解循环期间限制为 1（性能：求解阶段实测快
        # 9~11%；作用域必须是"循环期间"而不是"构造时一次"，理由见
        # `blas_threads_limited` 文档记录的真实 bug）。
        last_finite = None
        with blas_threads_limited(1):
            for i in range(max_iter):
                t_start = time.time()
                res = self.step(dt)
                t_end = time.time()
                final_residual = res

                # 发散即中止（2026-09-16，真实事故驱动）：此前这条循环
                # 完全没有有限性检查，残差变成 inf/nan 之后照常继续迭代、
                # 照常调用 checkpoint_callback，会把 NaN 状态写进
                # checkpoint 并在收尾时用 NaN 覆盖 final_state.pkl。
                # GPU 单机与多 GPU 路径本来就有这个检查，CPU 路径此前
                # 遗漏——是路径不对等，不是有意设计。检查必须在
                # checkpoint 回调**之前**。
                check_residual_finite(res, i + 1, order=self.order,
                                      last_finite=last_finite, extra_hint=self._divergence_hint())
                last_finite = res

                if initial_res is None:
                    initial_res = res
                self._phase_initial_residual = initial_res

                # 每步打印详细信息（真实功能缺口修复，2026-08-31，用户直接
                # 指出"P0/P1直接运算和P2 order continuation打印的信息应该
                # 一样"）：这条"非 order continuation"常规循环（目标阶数<2，
                # 例如单独求解 P0/P1）此前打印频率（每10步一次）、字段顺序
                # （CFL 在 Time 之前）、前缀（"Iteration N"而非"P{order} Iter
                # N"）、Time 标签（"Time/step"而非"Time"）都与
                # order_continuation.py::run_order_continuation（目标阶数>=2
                # 时走的分阶段路径）不一致——两条路径各自独立发展、从未同步
                # 过格式。这里改成逐字段对齐 order_continuation.py 的格式
                # （以其为准），包括每步都打印、同样的字段顺序与 Cd/Cl/Cs
                # 气动力系数打印。
                if report:
                    print(format_step_line(self, self.order, i + 1, res, initial_res, t_end - t_start, i + 1))

                # 中间 checkpoint 保存
                if checkpoint_callback is not None:
                    checkpoint_callback(self, i + 1)

                # 相对收敛判据：残差相对初始值下降 1/tol 倍
                # tol=1e-6 表示需要下降 6 个量级；tol<=0 表示纯定步数迭代（
                # B-10：transient 命令固定传 tol=0.0，此前 1.0 / tol 在第 2 步
                # 直接 ZeroDivisionError 崩溃），此时不启用收敛判据。
                if i >= 1 and tol > 0.0 and initial_res / max(res, 1e-30) >= 1.0 / tol:
                    converged = True
                    if report:
                        print(f"[OK] Converged at iteration {i+1} with residual {res:.6e} "
                              f"(dropped {initial_res/res:.1e}x)")
                    break

        # 收尾摘要：把"物理场到底走了多远"完整报一次（残差是否收敛与物理场是否建立是两件事）
        if report:
            report_pseudo_time_summary(self, i + 1)

        return SolverResult(converged=converged, iterations=i+1, final_residual=final_residual)


class DistributedSolveLoopMixin(SolveLoopMixin):
    """CPU MPI 与多 GPU 分布式：`step()` 返回全域 allreduce 残差（各 rank 一致，收敛/发散判定同步发生；
    checkpoint 回调是集体操作，全部 rank 都调用）。"""

    def _divergence_hint(self) -> str:
        return f"分布式路径：{self.n_ranks} 个 rank，残差是全域 allreduce 值（各 rank 一致，同时抛出）"


class _SolverSolveMixin(SolveLoopMixin):
    """CPU `FRSolver`：主机视图、Order Continuation 委托、阶数切换与单步推进。"""

    def host_view(self):
        """主机视图：CPU 求解器的状态本来就在主机上，返回自身（单 GPU 的见 `core/gpu/solver/host_view.py`）。"""
        return self

    def _scaled_residual_field(self):
        return self.state.dU_dt

    @contextmanager
    def edit_host_state(self):
        """在主机上修改状态（恢复 checkpoint、`--init-from`）：CPU 直接改自身。"""
        yield self

    def _interpolate_to_new_order(self, new_order: int):
        """换到 `new_order` 阶（Order Continuation 的统一换阶接口，四个后端同一语义）：`new_order == 0` 时用
        来流重置（全新求解器从 P0 起步），否则对解与湍流场做精确多项式延拓；随后算子、网格几何、壁距、
        边界幽灵态（SEM 入口持有构造阶数的通量点坐标）全部切到新阶数，旧阶数的几何缓存释放。

        此前这串步骤内联在 CPU 单机自己的 Order Continuation 循环里，其余后端的同名方法各自完成同一件事。
        """
        from autoflowcfd.core.fr_solver.turbulence import recompute_wall_distance_for_current_order
        from autoflowcfd.fr.operators import generate_fr_operators

        if new_order == 0:
            order_continuation._reset_state_to_p0(self, 1)
            from autoflowcfd.core.time_integration.implicit.mean_flow_step import reset_newton_state

            reset_dual_time_history(self)
            reset_newton_state(self)
        else:
            order_continuation.interpolate_to_new_order_checked(self, new_order)
            self.ops = generate_fr_operators(new_order)
        for o in [o for o in list(self.mesh._order_geometry_cache) if o != new_order]:
            del self.mesh._order_geometry_cache[o]
        if new_order != 0:
            self.mesh.set_order(new_order)
            recompute_wall_distance_for_current_order(self)
        self.boundary_ghost_provider = self._build_boundary_ghost_provider(self.bc_overrides)

    def _limit_prolongated_state(self) -> None:
        """升阶延拓之后在**新阶数**的点集（解点 + 面通量点 + 过积分细点）上施加守恒的
        正性限制器（`time_integration/positivity`，向单元均值收缩、均值不变）。

        低阶多项式只在低阶那组点上被保证可容许；新阶数的点落在别处，延拓后的状态
        可能在那里 rho 或 p 非正——隐式 Newton 的物理性限幅假定出发态处处可容许，
        于是第一步残差就算在非物理态上（plate_demo P1->P2 实测：P2 第 1 步残差
        2.6e27、dtau 缩到下限仍拿不到被接受的步）。必须在新阶数几何就位之后调用。
        """
        from autoflowcfd.core.time_integration.positivity import get_positivity_limiter

        n_cells, n_sps, n_vars = self.state.U.shape
        U = np.ascontiguousarray(self.state.U)
        get_positivity_limiter(self)(U.reshape(n_cells * n_sps, n_vars))
        self.state.U = U
        self.state._update_primitives()

    def step(self, dt: float) -> float:
        """执行一个时间步长 (S-05)。见 fr_solver_step.py::step 文档。"""
        return fr_solver_step.step(self, dt)
