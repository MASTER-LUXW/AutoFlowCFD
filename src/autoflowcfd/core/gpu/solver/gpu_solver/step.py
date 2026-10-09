"""AutoFlowCFD V2.0 - 单机 GPU 的单步推进

从 `src/autoflowcfd/core/gpu/solver/gpu_solver.py` 的 `GPUFRSolver` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `GPUFRSolver` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

from functools import partial
import numpy as np
from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from autoflowcfd.core.time_integration.implicit.mean_flow_step import (
    newton_step_ok,
    step_mean_flow_newton,
)
from autoflowcfd.core.time_integration.implicit.reductions import LocalReductions
from autoflowcfd.core.turbulence.registry import has_transport_equations
from autoflowcfd.core.gpu.turbulence.gpu_implicit_turbulence import GpuCoupledBackend
from autoflowcfd.core.time_integration.implicit.coupled_step import step_coupled_newton


class _GPUSolverStepMixin:
    """单机 GPU 的单步推进（求解循环与 CPU 共用 `SolveLoopMixin`）"""

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
        # 合成湍流入口（SEM）每个物理步推进一次（与 CPU 同一个函数；此前单 GPU 从不推进，入口涡结构冻结）
        from autoflowcfd.boundary.synthetic_inlet import advance_synthetic_inlets
        advance_synthetic_inlets(self.boundary_ghost_provider, self.freestream, dt)
        # 本步冻结的问题单元人工扩散系数（与 CPU step.py 同一算子分裂约定；
        # 未启用时 None），同时进入粘性步长限制与全部粘性残差求值
        nu_av = self.compute_artificial_diffusivity_field_gpu()

        # 局部 CFL 步长。启用低马赫数预处理时 dt_local 是**预处理后**的
        # 平均流步长（按 |un|+c_precond 取），dt_physical 是按物理波速那
        # 一份；两者的分工与 CPU 侧 step.py 完全一致。
        coupled_nk = (scheme == TimeIntegrationScheme.NEWTON_KRYLOV and self.turb_model_gpu is not None
                      and has_transport_equations(self.turb_model_name))
        if scheme == TimeIntegrationScheme.NEWTON_KRYLOV:
            # 隐式稳态：与 CPU step.py 同一时序。带 k-omega 时湍流与平均流在下面紧耦合
            # 求解（time_integration/implicit/coupled_step.py），这里只取当前状态的涡粘
            # 供残差监控；否则湍流（代数模型）照原有更新。
            dt_local, dt_physical = self._compute_local_time_step_gpu(
                return_physical_too=True, nu_av=nu_av)
            if coupled_nk:
                mu_t_field = self._turbulent_mu_t_gpu()
            else:
                mu_t_field = self.compute_turbulence_source_gpu(dt_physical[:, None])
        else:
            # 先取步长（粘性 CFL 用上一步的涡粘，与 CPU step.py 同一时序），
            # 再在当前状态下求湍流源项（算子分裂）。湍流步长的规则见
            # compute_turbulence_source_gpu 文档
            dt_local, dt_physical = self._compute_local_time_step_gpu(
                return_physical_too=True, nu_av=nu_av)
            if scheme == TimeIntegrationScheme.DUAL_TIME and self.turb_model_gpu is not None:
                # 湍流方程的双时间步（与 CPU 同一份，`core/turbulence/dual_time.py`）
                from autoflowcfd.core.turbulence.dual_time import advance_turbulence_dual_time

                mu_t_field = advance_turbulence_dual_time(
                    self, self.turb_model_gpu, cp,
                    lambda dtau, term, first: self.compute_turbulence_source_gpu(dtau, term, first),
                    dt_physical[:, None], dt, self.time_integrator.dual_time_steps)
            else:
                mu_t_field = self.compute_turbulence_source_gpu(dt_physical[:, None])
        dt_local_full = cp.broadcast_to(
            dt_local[:, None], (n_cells, n_sps)
        ).reshape(n_cells * n_sps)
        # 累计伪时间（与 CPU step.py 同一个函数；n_cells 个数拷回主机）
        from autoflowcfd.core.fr_solver.pseudotime_budget import accumulate_pseudo_time
        accumulate_pseudo_time(self, cp.asnumpy(dt_local))

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
            visc_res = self.compute_viscous_residual_gpu(U_trial, mu_t_field=mu_t_field, nu_av=nu_av)
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
        # 隐式步用它的点集做物理性限幅（`PositivityLimiter.density_pressure_limits`）。
        from autoflowcfd.core.time_integration.positivity import get_positivity_limiter
        positivity_func = get_positivity_limiter(self, xp=cp)

        # 根据时间方案选择推进方式
        nk_info = None
        if coupled_nk:
            # 平均流 + k-omega 紧耦合隐式步：与 CPU 共用同一个实现（implicit/coupled_step.py），
            # 新状态由适配器写回 U_gpu 与湍流模型
            nk_info = step_coupled_newton(self, GpuCoupledBackend(self, nu_av), dt_local_full,
                                          filter_active=self.filter_func_gpu is not None)
            U_new_flat = self.U_gpu.reshape(n_cells * n_sps, self.n_vars)
        elif scheme == TimeIntegrationScheme.NEWTON_KRYLOV:
            # 平均流隐式步：与 CPU 共用同一个实现（implicit/mean_flow_step.py），
            # 这里只提供 GPU 的残差、归约（cupy）与面相邻关系。
            from autoflowcfd.core.fr_solver.residual_diagnostics import _reference_scales
            ff = self.flat_face_gpu
            from autoflowcfd.core.gpu.turbulence.gpu_implicit_turbulence import gpu_cell_colors, gpu_coupling_graph
            from autoflowcfd.core.fr_residual.jacobian.backend import (
                MeanFlowBlockAssembler, unsupported_reason,
            )
            order_nk = int(self.order if getattr(self, "current_order", None) is None
                           else self.current_order)
            # 解析单元块在主机上装配（与 CPU 同一份实现），块由缓存上传
            block_assembler = None if unsupported_reason(
                order=order_nk, wmles=getattr(self, "wmles_model", None) is not None,
            ) else MeanFlowBlockAssembler(
                mesh=self.mesh, ops=self.ops, ghost_provider=self.boundary_ghost_provider,
                mu=self.mu_molecular, mach_ref=self.freestream["mach_ref"],
                low_mach=self.low_mach_precond_enabled, mu_t=mu_t_field, nu_av=nu_av, n_sps=n_sps)
            U_new_flat, nk_info = step_mean_flow_newton(
                self, mean_flow_residual, U_flat, dt_local_full,
                _reference_scales(self.freestream, self.n_vars),
                red=LocalReductions(cp),
                cell_is_prism=np.arange(n_cells) < int(self.mesh.n_prism_cells),
                cell_colors=lambda: gpu_cell_colors(cp, ff, n_cells),
                coupling_graph=partial(gpu_coupling_graph, cp, ff, n_cells),
                order=order_nk,
                filter_active=self.filter_func_gpu is not None,
                positivity=positivity_func, block_assembler=block_assembler)

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
                visc_res = self.compute_viscous_residual_gpu(U_trial, mu_t_field=mu_t_field, nu_av=nu_av)
                return -visc_res.reshape(n_cells * n_sps, self.n_vars)

            U_new_flat = self.time_integrator.step_imex(
                U_flat, convective_residual_only, diffusive_residual_only,
                dt_local_full, positivity_func=positivity_func,
            )

        else:
            # SSP-RK2/RK3 或前向 Euler
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
