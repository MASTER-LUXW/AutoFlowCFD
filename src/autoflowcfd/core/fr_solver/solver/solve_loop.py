"""AutoFlowCFD V2.0 - `FRSolver` 的求解循环（mixin，只含方法）。

从 `core/fr_solver/solver.py` 拆出（2026-09-25）。
"""

from typing import Optional

import numpy as np

from autoflowcfd.core.time_integration.implicit.mean_flow_step import newton_monitor_suffix
from autoflowcfd.core.fr_solver.residual_diagnostics import check_residual_finite
from autoflowcfd.core.fr_solver.state import SolverResult
from autoflowcfd.core.utils import order_continuation

from .. import step as fr_solver_step
from .threads import blas_threads_limited


class _SolverSolveMixin:
    """求解循环、Order Continuation 委托与单步推进。"""

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

    def solve(self, max_iter: int = 1000, dt: float = 1e-4, tol: float = 1e-6,
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
            phase_max_iter: 仅 Order Continuation（`self.order>=2` 时）生效，
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
        if getattr(self, "_tau_accum_seeded", False):
            self._tau_accum_seeded = False
        else:
            self.tau_accum = None

        logger_msg = f"Starting solve loop with {self.time_integrator.scheme.value}"
        if self.turb_model_name != "NONE":
            logger_msg += f", turbulence={self.turb_model_name}"
        print(logger_msg)

        # Order Continuation: 从低阶开始逐步提升精度
        if self.order_continuation_enabled and self.order >= 2:
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
        initial_res = None
        
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
                self.residual_history.append(res)

                # 发散即中止（2026-09-16，真实事故驱动）：此前这条循环
                # 完全没有有限性检查，残差变成 inf/nan 之后照常继续迭代、
                # 照常调用 checkpoint_callback，会把 NaN 状态写进
                # checkpoint 并在收尾时用 NaN 覆盖 final_state.pkl。
                # GPU 单机与多 GPU 路径本来就有这个检查，CPU 路径此前
                # 遗漏——是路径不对等，不是有意设计。检查必须在
                # checkpoint 回调**之前**。
                check_residual_finite(res, i + 1, order=self.order,
                                      last_finite=last_finite)
                last_finite = res

                if initial_res is None:
                    initial_res = res
            
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
                drop = initial_res / max(res, 1e-30)
                msg = f"P{self.order} Iter {i+1}: Residual = {res:.6e} | Drop: {drop:.1f}x | Time: {t_end - t_start:.2f}s"
                if self._cfl_controller is not None:
                    msg += f" | CFL={self._cfl_controller.cfl_number:.3f}"
                msg += newton_monitor_suffix(self)
                ref_area = getattr(self, '_reference_area', None)
                if ref_area is not None and ref_area > 0:
                    from autoflowcfd.postprocess.fr_coefficients import compute_forces_pressure_only
                    aero = compute_forces_pressure_only(self, ref_area)
                    msg += f" | Cd={aero['Cd']:.4f} Cl={aero['Cl']:.4f} Cs={aero['Cs']:.4f}"
                # 按方程分别归一化残差 + 最大残差定位（与 order_continuation.py
                # 同一处新增，参照 Fluent scaled residuals / STAR-CCM+ Max
                # 监视器，见 residual_diagnostics.py 模块文档"背景"一节）：
                # 只新增打印，不改变本函数自己的 `tol`/`drop` 收敛判据。
                #
                # 打印频率（2026-09-13 用户反馈修复，与 order_continuation.py
                # 同一处、同一理由）：只在第 1 步和其后每 10 步打印一次，避免
                # 正常运行时每步都刷出这行长诊断信息。
                freestream = getattr(self, 'freestream', None)
                if freestream is not None and hasattr(self.state, 'dU_dt') and (i == 0 or (i + 1) % 10 == 0):
                    from autoflowcfd.core.fr_solver.residual_diagnostics import (
                        compute_scaled_residuals, format_scaled_residual_line,
                    )
                    diag = compute_scaled_residuals(self.state.dU_dt, freestream)
                    msg += " | " + format_scaled_residual_line(
                        diag,
                        # 最大残差单元的体积分位：残差被 det(J) 除，体积
                        # 极小的退化单元天然把任何通量不平衡放大若干个
                        # 量级，所以这个数字是区分"退化单元机制"与"壁面
                        # 处理机制"最直接的单个指标（2026-09-16 真实排查
                        # 驱动，见 residual_diagnostics.py::
                        # cell_volume_percentile 文档）。
                        cell_volumes=getattr(
                            getattr(self, "mesh", None), "cell_volumes", None),
                    )
                    # 累计伪时间 / 物体尺度对流时标（2026-09-17）：与上面
                    # 那行诊断同频打印。这个比值此前算得出来但从未被报告，
                    # 结果"残差在降但物理场只走了 1.8% 个特征时间"这件事在
                    # 日志里完全看不出来，直接导致一次把启动暂态误判成壁面
                    # 处理缺陷、追了好几天的事故。完整记录见
                    # `pseudotime_budget.py` 模块文档。
                    _ptb_fn = getattr(
                        self, "_pseudo_time_budget", None)
                    _ptb = (_ptb_fn(n_steps=i + 1)
                            if _ptb_fn is not None else None)
                    if _ptb is not None:
                        from autoflowcfd.core.fr_solver.pseudotime_budget import (
                            format_pseudo_time_budget,
                        )
                        msg += " | " + format_pseudo_time_budget(
                            _ptb, compact=True)
                print(msg)

                # 中间 checkpoint 保存
                if checkpoint_callback is not None:
                    checkpoint_callback(self, i + 1)
                
                # 相对收敛判据：残差相对初始值下降 1/tol 倍
                # tol=1e-6 表示需要下降 6 个量级；tol<=0 表示纯定步数迭代（
                # B-10：transient 命令固定传 tol=0.0，此前 1.0 / tol 在第 2 步
                # 直接 ZeroDivisionError 崩溃），此时不启用收敛判据。
                if i >= 1 and tol > 0.0 and initial_res / max(res, 1e-30) >= 1.0 / tol:
                    converged = True
                    print(f"[OK] Converged at iteration {i+1} with residual {res:.6e} "
                          f"(dropped {initial_res/res:.1e}x)")
                    break

        # 收尾摘要：把"物理场到底走了多远"完整报一次。残差是否收敛与
        # 物理场是否建立是**两件事**，只报前者会让人拿启动暂态的气动力
        # 系数去和文献值比（这件事真实发生过，见 `pseudotime_budget.py`）。
        # `getattr` 而不是直接调用：`test_solver_divergence_abort.py` 用
        # SimpleNamespace 替身直接调用未绑定的 solve()（本仓库既有的测试
        # 手法），那种替身没有这个方法；一行纯诊断不该让它失败。
        _ptb_fn = getattr(self, "_pseudo_time_budget", None)
        _ptb = _ptb_fn(n_steps=i + 1) if _ptb_fn is not None else None
        if _ptb is not None:
            from autoflowcfd.core.fr_solver.pseudotime_budget import (
                format_pseudo_time_budget,
            )
            print(format_pseudo_time_budget(_ptb))

        return SolverResult(converged=converged, iterations=i+1, final_residual=final_residual)
    
    def _solve_with_order_continuation(self, max_iter: int, dt: float, tol: float,
                                        checkpoint_callback=None,
                                        phase_max_iter: Optional[int] = None,
                                        residual_drop_threshold: float = 1e2) -> SolverResult:
        """实现 Order Continuation 策略：从P0逐步提升到目标阶数（委托给 order_continuation）。"""
        return order_continuation.run_order_continuation(
            self, max_iter, dt, tol, checkpoint_callback,
            phase_max_iter=phase_max_iter,
            residual_drop_threshold=residual_drop_threshold,
        )

    def _interpolate_to_new_order(self, new_order: int):
        """将解从当前阶数插值到新的阶数（委托给 order_continuation）。"""
        order_continuation.interpolate_to_new_order_checked(self, new_order)

    def step(self, dt: float) -> float:
        """执行一个时间步长 (S-05)。见 fr_solver_step.py::step 文档。"""
        return fr_solver_step.step(self, dt)
