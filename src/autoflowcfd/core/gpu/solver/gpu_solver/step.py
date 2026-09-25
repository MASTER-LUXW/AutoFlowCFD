"""AutoFlowCFD V2.0 - 单机 GPU 的单步推进与求解循环

从 `src/autoflowcfd/core/gpu/solver/gpu_solver.py` 的 `GPUFRSolver` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `GPUFRSolver` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

import time
import numpy as np
from typing import Optional, Dict, Any
from autoflowcfd.core.fr_solver.residual_diagnostics import check_residual_finite
from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from autoflowcfd.core.time_integration.implicit.mean_flow_step import (
    newton_step_ok,
    step_mean_flow_newton,
)
from autoflowcfd.core.time_integration.implicit.reductions import LocalReductions
from autoflowcfd.core.fr_solver.turbulence.implicit import (
    IMPLICIT_TURBULENCE_MODELS,
    step_turbulence_newton,
)
from autoflowcfd.core.gpu.turbulence.gpu_implicit_turbulence import GpuTurbulenceBackend


class _GPUSolverStepMixin:
    """单机 GPU 的单步推进与求解循环"""

    def step(self, dt: float = 0.0) -> float:
        """执行一个时间步。

        完整流程（与 CPU 版 step() 对应）：
        1. 更新原始变量
        2. 湍流源项求值（算子分裂：湍流走独立显式更新）
        3. 局部 CFL 步长
        4. 平均流残差计算（含湍流涡粘耦合）
        5. SSP-RK / IMEX / DUAL_TIME 时间推进
        6. 湍流场更新（k/ω 正性限制）

        Args:
            dt: 物理时间步长（稳态模式下被局部 CFL 步长覆盖，
                DUAL_TIME 模式下是真正的物理时间步长）

        Returns:
            residual_norm: 残差范数
        """
        cp = get_cupy()
        n_cells = self.mesh.n_cells
        n_sps = self.mesh.n_sps_per_cell

        self._update_primitives_gpu()
        scheme = self.time_integrator.scheme

        # 局部 CFL 步长。启用低马赫数预处理时 dt_local 是**预处理后**的
        # 平均流步长（按 |un|+c_precond 取），dt_physical 是按物理波速那
        # 一份；两者的分工与 CPU 侧 step.py 完全一致。
        if scheme == TimeIntegrationScheme.NEWTON_KRYLOV:
            # 隐式稳态：与 CPU step.py 同一时序——先取步长，k-omega 做一个
            # 分离式 PTC-Newton 步（平均流冻结，显式输运更新在隐式 CFL 下
            # 必然失稳，见 fr_solver/turbulence/implicit.py），平均流再用更新
            # 后的涡粘做它的 Newton 步。
            dt_local, dt_physical = self._compute_local_time_step_gpu(
                return_physical_too=True)
            if (self.turb_model_gpu is not None
                    and self.turb_model_name.upper() in IMPLICIT_TURBULENCE_MODELS):
                step_turbulence_newton(
                    GpuTurbulenceBackend(self),
                    cp.broadcast_to(dt_physical[:, None], (n_cells, n_sps)))
                mu_t_field = self._turbulent_mu_t_gpu()
            else:
                mu_t_field = self.compute_turbulence_source_gpu()
        else:
            # 湍流源项在当前状态下求值（算子分裂）
            mu_t_field = self.compute_turbulence_source_gpu()
            dt_local, dt_physical = self._compute_local_time_step_gpu(
                return_physical_too=True)
        dt_local_full = cp.broadcast_to(
            dt_local[:, None], (n_cells, n_sps)
        ).reshape(n_cells * n_sps)

        # 展平 U 用于时间积分器
        U_flat = self.U_gpu.reshape(n_cells * n_sps, self.n_vars)

        # 构建平均流残差函数（含湍流涡粘耦合）
        def mean_flow_residual_raw(U_flat_trial):
            """未经预处理的原始残差 R（约定 dU/dt = -R）。

            残差监控与自适应 CFL 都必须用这一份**物理**残差：Gamma 可逆、
            两者同时趋零，但量级不同，用预处理值会让打印的残差、收敛判据
            以及与历史算例的对比全部失去可比性（与 CPU 侧 step.py 里
            同名函数一致）。
            """
            U_trial = U_flat_trial.reshape(n_cells, n_sps, self.n_vars)
            inv_res = self.compute_inviscid_residual_gpu(U_trial)
            visc_res = self.compute_viscous_residual_gpu(U_trial, mu_t_field=mu_t_field)
            total = inv_res + visc_res
            return -total

        def mean_flow_residual(U_flat_trial):
            """供时间积分器推进用：启用预处理时返回 `Gamma R`，否则就是 R。

            Gamma 线性，作用在 R 上与作用在 dU/dtau 上等价；它**必须**与
            上面按预处理波速取的 dt 成对出现（见
            core/gpu/gpu_preconditioning.py 与 CPU 侧
            core/utils/preconditioning.py 模块文档）。
            Gamma 需要的原始变量由 `apply_low_mach_preconditioner_gpu`
            从 U_trial 自己推导——GPU 侧不依赖 `self.Q_gpu` 是否与试探态
            同步这条隐式契约（与 CPU 侧的刻意差异，见那份模块文档）。
            """
            res = mean_flow_residual_raw(U_flat_trial)
            if self.low_mach_precond_enabled:
                from autoflowcfd.core.gpu.gpu_preconditioning import (
                    apply_low_mach_preconditioner_gpu,
                )
                # `out=res` 就地写：`res` 是 `mean_flow_residual_raw` 刚
                # 算出来的新数组，stage 内施加完 Gamma 后原始值不再需要，
                # 省掉每个 RK stage 一份全场数组的显存（79 万单元 P2 约
                # 1.2GiB/stage）。`residual0` 那一处不能这样做——那里必须
                # 保留物理残差给残差范数与自适应 CFL 用。
                res = apply_low_mach_preconditioner_gpu(
                    res, U_flat_trial.reshape(n_cells, n_sps, self.n_vars),
                    self.freestream["mach_ref"], out=res,
                )
            return res.reshape(n_cells * n_sps, self.n_vars)

        # 初始残差：`residual0_raw` 供残差范数/自适应 CFL 使用（物理残差），
        # `residual0` 供积分器复用 Stage 0（必要时已施加 Gamma）。
        residual0_raw = mean_flow_residual_raw(U_flat)
        if self.low_mach_precond_enabled:
            from autoflowcfd.core.gpu.gpu_preconditioning import (
                apply_low_mach_preconditioner_gpu,
            )
            residual0 = apply_low_mach_preconditioner_gpu(
                residual0_raw, self.U_gpu, self.freestream["mach_ref"],
            ).reshape(n_cells * n_sps, self.n_vars)
        else:
            residual0 = residual0_raw.reshape(n_cells * n_sps, self.n_vars)

        # 守恒的正性保持限制器：与 CPU 同一个（数组模块无关实现），按阶数缓存。
        from autoflowcfd.core.time_integration.positivity import get_positivity_limiter
        # 隐式步不用它（Newton 步的物理性由 jfnk 的限幅与接受判据保证），不构造。
        positivity_func = (None if scheme == TimeIntegrationScheme.NEWTON_KRYLOV
                           else get_positivity_limiter(self, xp=cp))

        # 根据时间方案选择推进方式
        nk_info = None
        if scheme == TimeIntegrationScheme.NEWTON_KRYLOV:
            # 平均流隐式步：与 CPU 共用同一个实现（implicit/mean_flow_step.py），
            # 这里只提供 GPU 的残差、归约（cupy）与面相邻关系。
            from autoflowcfd.core.fr_solver.residual_diagnostics import _reference_scales
            ff = self.flat_face_gpu
            from autoflowcfd.core.gpu.turbulence.gpu_implicit_turbulence import gpu_cell_colors
            U_new_flat, nk_info = step_mean_flow_newton(
                self, mean_flow_residual, U_flat, dt_local_full,
                _reference_scales(self.freestream, self.n_vars),
                red=LocalReductions(cp),
                cell_is_prism=np.arange(n_cells) < int(self.mesh.n_prism_cells),
                cell_colors=lambda: gpu_cell_colors(cp, ff, n_cells),
                order=int(self.order if getattr(self, "current_order", None) is None
                          else self.current_order),
                filter_active=self.filter_func_gpu is not None)

        elif scheme == TimeIntegrationScheme.DUAL_TIME:
            # DUAL_TIME: 真正时间精度的物理时间推进。`solution_prev=None`
            # （第一个物理步）时积分器自己退化为 BDF1，其后 BDF2。
            U_new_flat = self.time_integrator.step_dual_time(
                U_flat, mean_flow_residual, dt_local_full,
                dt_physical=dt,
                solution_prev=self._dual_time_U_prev,
                max_inner_iter=self.time_integrator.dual_time_steps,
                filter_func=self.filter_func_gpu, positivity_func=positivity_func,
            )
            # 保存当前解作为下一步的 prev
            self._dual_time_U_prev = U_flat.copy()

        elif scheme == TimeIntegrationScheme.IMEX_EULER:
            # IMEX: 显式处理对流，隐式处理粘性
            def convective_residual_only(U_flat_trial):
                U_trial = U_flat_trial.reshape(n_cells, n_sps, self.n_vars)
                inv_res = self.compute_inviscid_residual_gpu(U_trial)
                return -inv_res.reshape(n_cells * n_sps, self.n_vars)

            def diffusive_residual_only(U_flat_trial):
                U_trial = U_flat_trial.reshape(n_cells, n_sps, self.n_vars)
                visc_res = self.compute_viscous_residual_gpu(U_trial, mu_t_field=mu_t_field)
                return -visc_res.reshape(n_cells * n_sps, self.n_vars)

            U_new_flat = self.time_integrator.step_imex(
                U_flat, convective_residual_only, diffusive_residual_only,
                dt_local_full, positivity_func=positivity_func,
            )

        else:
            # SSP-RK2/RK3 or Forward Euler
            U_new_flat = self.time_integrator.step(
                U_flat, mean_flow_residual, dt_local_full,
                residual0=residual0,
                filter_func=self.filter_func_gpu, positivity_func=positivity_func,
            )

        self.U_gpu = U_new_flat.reshape(n_cells, n_sps, self.n_vars)
        self._update_primitives_gpu()

        # SGS（WALE）涡粘系数更新（#7）：必须在状态更新之后调用，供
        # 下一步的粘性残差消费，与 CPU 版 apply_turbulence_corrections
        # 同一个操作分裂时序，见 gpu_solver_io.py::
        # _apply_turbulence_corrections_gpu 文档。
        self._apply_turbulence_corrections_gpu()

        # 残差范数：用**未预处理**的物理残差（见 mean_flow_residual_raw）
        # 显式 ravel：`residual0_raw` 现在是 3 维 (n_cells,n_sps,n_vars)，
        # `linalg.norm` 只在 ord=None 时才隐式对 >2 维做 ravel，依赖那条
        # 特殊语义没必要；ravel 后与改动前那份 2 维输入的范数逐位相同。
        _r = residual0_raw.ravel()
        residual_norm = float(cp.linalg.norm(_r) / max(1, np.sqrt(_r.size)))
        self.residual_history.append(residual_norm)
        self.iteration += 1

        # 自适应 CFL：按物理残差更新（与 CPU 侧 step.py 同一时序——在残差
        # 范数算出来之后、返回之前）
        if self._cfl_controller is not None:
            if nk_info is not None:
                # SER 看 Newton 所解系统的 ||Gamma R||，并区分物理暂态与步失败
                # （adaptive_cfl/ser.py），与 CPU step.py 同一处理
                self._cfl_controller.update(nk_info["res_norm"], step_ok=newton_step_ok(nk_info))
            else:
                self._cfl_controller.update(residual_norm)

        return residual_norm

    def solve(
        self,
        max_iter: int = 1000,
        dt: float = 1e-4,
        tol: float = 1e-6,
        output_interval: int = 10,
        phase_max_iter: Optional[int] = None,
        residual_drop_threshold: float = 1e2,
    ) -> Dict[str, Any]:
        """执行稳态求解循环。

        Order Continuation 自动分派（2026-09-02，见 core/gpu/solver/
        gpu_solver_order_continuation.py 模块文档）：与 CPU `FRSolver.
        solve()`/GPU 分布式版本同一个判据——`self.order`（目标阶数）
        >= 2 时自动改用逐阶爬坡（`run_distributed_order_continuation`，
        尽管函数名带"distributed"，逻辑本身对 solver 只要求
        `step()`/`order`/`current_order`/`_interpolate_to_new_order`
        这几个鸭子类型接口，不依赖任何分布式概念，单机 GPU 复用同一份
        实现，不需要另写一份等价的迭代循环）。

        Args:
            max_iter: 最大迭代次数
            dt: 时间步长（稳态模式下被 CFL 覆盖）
            tol: 收敛容差
            output_interval: 输出间隔
            phase_max_iter, residual_drop_threshold: 仅在触发 Order
                Continuation（`self.order >= 2`）时生效，与单机 CPU
                `run_order_continuation` 同名参数同一含义。

        Returns:
            结果字典
        """
        if getattr(self, 'order_continuation_enabled', True) and self.order >= 2:
            from autoflowcfd.core.mpi.distributed_order_continuation import (
                run_distributed_order_continuation,
            )
            result = run_distributed_order_continuation(
                self, max_iter, dt, tol,
                phase_max_iter=phase_max_iter, residual_drop_threshold=residual_drop_threshold,
            )
            return {
                'converged': result.converged,
                'iterations': result.iterations,
                'final_residual': result.final_residual,
                'residual_history': self.residual_history,
            }

        print(f"Starting GPU solve: max_iter={max_iter}, tol={tol}")
        converged = False
        final_residual = 1e10
        _last_finite = None

        for i in range(max_iter):
            t_start = time.time()
            res = self.step(dt)
            t_end = time.time()
            final_residual = res

            if i == 0 or (i + 1) % output_interval == 0:
                mem = self.array_mgr.get_memory_usage()
                print(
                    f"GPU Iter {i+1}: Residual = {res:.6e} | "
                    f"Time/step: {t_end-t_start:.3f}s | "
                    f"GPU mem: {mem['used_mb']:.0f}/{mem['total_mb']:.0f} MB"
                )

            # 发散即中止（2026-09-16 统一）：此前这里只是 break，于是
            # 调用方拿到的是 converged=False，与"跑满预算仍未收敛"完全
            # 无法区分，收尾还会把 NaN 状态写成结果文件。改用与另外四条
            # 求解循环共享的 SolverDivergedError，让 CLI 非零退出。
            check_residual_finite(res, i + 1, last_finite=_last_finite)
            _last_finite = res

            if res < tol:
                converged = True
                print(f"✅ GPU Converged at iteration {i+1} with residual {res:.6e}")
                break

        return {
            'converged': converged,
            'iterations': self.iteration,
            'final_residual': final_residual,
            'residual_history': self.residual_history,
        }
