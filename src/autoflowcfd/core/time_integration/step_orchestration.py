"""单步编排：四个后端（单机 CPU / 单 GPU / CPU-MPI / 多 GPU）共用的一个时间步（稳态一个伪时间步、时间精确一个
子步、双时间步一个物理步）。

一步的流程（`advance_step`）：

1. 本步冻结的问题单元人工扩散系数（未启用为 None），进入粘性步长限制与本步全部粘性残差求值；
2. 局部步长：平均流用的（启用低马赫预处理时按预处理波速取）与湍流显式更新用的（物理波速）；
3. 合成湍流入口（SEM）涡核推进一个物理步（`boundary/synthetic_inlet.py`）；
4. 累计伪时间（`fr_solver/pseudotime_budget.py`；双时间步每个单元前进同一个物理步）；
5. 湍流：隐式步（稳态 Newton-Krylov、双时间步的隐式内迭代）里带输运方程的模型与平均流紧耦合求解（这里只取
   当前状态的涡粘）；其余情形按算子分裂在当前状态下先更新。得到本步冻结的涡粘；
6. 平均流残差闭包：物理残差 R（`dU/dt = -R`）与推进用的 `Gamma R`（低马赫预处理；`Gamma` 必须与按预处理波速
   取的步长成对出现，`core/utils/preconditioning.py`）。初始物理残差即本步的监控残差；
7. 推进：隐式步（`implicit/newton/`：紧耦合或只解平均流；双时间步每个物理步若干个带 BDF 物理时间项的
   Newton 步）或显式 `TimeIntegrator`（SSP-RK，守恒的正性限制器在每个 stage 收尾施加）；
8. 步后修正（SGS 涡粘等，供下一步的粘性残差使用）；
9. 残差范数（逐方程，按参考量级归一化，`residual_norm.py`）与显式湍流残差；
10. 自适应 CFL 更新（双时间步的伪时间 CFL 在内迭代里更新）。

后端只提供原语（`StepBackend`）：状态的取用与写回、局部步长、湍流更新、残差、Newton 所需的着色 / 耦合图 /
块装配 / 全局粗校正、正性限制器、归约。2026-10-22（I-112）以前这一流程在四个后端各写一份，已经分叉（伪时间只有
单机后端累计；人工扩散系数 CPU 单机每步求两次、双时间步内迭代里按试探态重算而 GPU 用冻结值；GPU 两个后端另维护
一个从不被读取的迭代计数）。
"""

from typing import Any, Optional, Protocol, Tuple

import numpy as np

from autoflowcfd.core.time_integration.base import TimeIntegrationScheme


class StepBackend(Protocol):
    """一个后端在一步里提供的原语。行数 `n_rows = n_cells * n_sps`（本 rank 的 local 单元）。"""

    xp: Any                    # 数组模块（numpy / cupy）
    n_cells: int
    n_sps: int
    n_vars: int
    reductions: Any            # LocalReductions / MPIReductions
    cell_is_prism: np.ndarray  # (n_cells,) 主机布尔
    order: int
    freestream: dict
    ghost_provider: Any
    turbulence_model: Any      # 输运湍流模型（显式湍流残差、紧耦合判定），没有时 None

    def begin_step(self) -> None:
        """步首：原始变量与当前守恒量同步。"""

    def artificial_diffusivity(self) -> Any:
        """本步冻结的问题单元人工扩散系数（后端自己的表示），未启用为 None。"""

    def local_time_steps(self, nu_av) -> Tuple[Any, Any, np.ndarray]:
        """`(平均流步长 (n_rows,), 湍流显式更新用步长（后端表示）, 每单元伪时间步长 (n_cells,) 主机数组)`。"""

    def pseudo_time_step(self, nu_av) -> Any:
        """双时间步内迭代每个 Newton 步的伪时间步长 `(n_rows,)`（当前状态、本步冻结的人工扩散系数）。"""

    def turbulence(self, dt_turbulence, coupled: bool) -> Any:
        """本步冻结的涡粘（后端表示）。`coupled` 为假时先按算子分裂推进湍流一步。"""

    def state_rows(self) -> Any:
        """当前守恒量 `(n_rows, n_vars)`（视图）。"""

    def set_state_rows(self, rows) -> None:
        """写回守恒量并同步原始变量。"""

    def primitives(self, rows) -> Any:
        """守恒量行 -> 原始变量 `(n_cells, n_sps, 5)`（低马赫预处理用）。"""

    def residual_raw(self, rows, mu_t, nu_av) -> Any:
        """物理残差 R `(n_rows, n_vars)`（`dU/dt = -R`；分布式后端含本次求值的 halo 交换）。"""

    def record_monitor_residual(self, raw) -> None:
        """记下本步的监控残差（CPU 后端存 `state.dU_dt`）。"""

    def coupled_backend(self, nu_av) -> Any:
        """紧耦合 Newton 的后端（`implicit/newton/coupled_step.py` 的约定）。"""

    def newton_options(self, mu_t, nu_av) -> dict:
        """`step_mean_flow_newton` 的后端相关关键字（cell_colors、coupling_graph、block_assembler、global_coarse）。"""

    def positivity(self) -> Any:
        """守恒的正性限制器（按阶数缓存）。"""

    def end_step(self, mu_t, coupled) -> None:
        """步后修正（SGS 涡粘、分布式的上一步涡粘缓存等）。"""


def coupled_implicit(scheme, turbulence_model, turb_model_name: str) -> bool:
    """隐式步里湍流是否与平均流紧耦合求解：带输运方程的湍流模型 + 稳态 Newton-Krylov 或双时间步。"""
    from autoflowcfd.core.turbulence.registry import has_transport_equations

    return (scheme in (TimeIntegrationScheme.NEWTON_KRYLOV, TimeIntegrationScheme.DUAL_TIME)
            and turbulence_model is not None and has_transport_equations(turb_model_name))


def advance_step(solver, backend: StepBackend, dt: float) -> float:
    """推进一步（流程见模块文档），返回本步的残差范数。

    `dt`：稳态显式格式下被忽略（局部伪时间步）；时间精确的显式子步里是该物理步的剩余时间；双时间步里是物理时间
    步长。
    """
    from autoflowcfd.boundary.synthetic_inlet import advance_synthetic_inlets
    from autoflowcfd.core.fr_solver.pseudotime_budget import accumulate_pseudo_time
    from autoflowcfd.core.fr_solver.residual_diagnostics import _reference_scales
    from autoflowcfd.core.time_integration.residual_norm import explicit_turbulence_residual, monitor_residuals
    from autoflowcfd.core.time_integration.time_accurate import sem_time_step

    xp = backend.xp
    n_cells, n_sps, n_vars = backend.n_cells, backend.n_sps, backend.n_vars
    n_rows = n_cells * n_sps
    scheme = solver.time_integrator.scheme
    dual_time = scheme == TimeIntegrationScheme.DUAL_TIME
    implicit = dual_time or scheme == TimeIntegrationScheme.NEWTON_KRYLOV
    coupled = coupled_implicit(scheme, backend.turbulence_model, solver.turb_model_name)
    mach_ref = backend.freestream["mach_ref"]
    scales = _reference_scales(backend.freestream, n_vars)

    backend.begin_step()
    nu_av = backend.artificial_diffusivity()
    dt_rows, dt_turbulence, dt_cells = backend.local_time_steps(nu_av)
    advance_synthetic_inlets(backend.ghost_provider, backend.freestream, sem_time_step(solver, dt))
    accumulate_pseudo_time(solver, np.full(n_cells, float(dt)) if dual_time else dt_cells)
    mu_t = backend.turbulence(dt_turbulence, coupled)

    def residual_raw(rows):
        return backend.residual_raw(rows, mu_t, nu_av)

    def precondition(res, rows, inplace):
        if not solver.low_mach_precond_enabled:
            return res
        from autoflowcfd.core.utils.preconditioning import apply_low_mach_preconditioner

        R3 = res.reshape(n_cells, n_sps, n_vars)
        out = apply_low_mach_preconditioner(R3, backend.primitives(rows), mach_ref, out=R3 if inplace else None)
        return out.reshape(n_rows, n_vars)

    def residual(rows):
        """推进用的残差：启用低马赫预处理时 `Gamma R`，否则 R（就地，`residual_raw` 每次返回新数组）。"""
        return precondition(residual_raw(rows), rows, inplace=True)

    U_rows = backend.state_rows()
    raw0 = residual_raw(U_rows)
    backend.record_monitor_residual(raw0)
    residual_norm, solver._residual_per_equation = monitor_residuals(
        backend.reductions, raw0.reshape(n_cells, n_sps, n_vars), backend.cell_is_prism, backend.order, scales)
    positivity = backend.positivity()

    nk_info: Optional[dict] = None
    coupled_now = None
    if implicit:
        from autoflowcfd.core.time_integration.implicit.newton.coupled_step import step_coupled_newton
        from autoflowcfd.core.time_integration.implicit.newton.mean_flow_step import step_mean_flow_newton

        del raw0          # 隐式步在基态上自己求残差，留着它只会在整个 Newton 步里多占一个全尺寸数组

        def newton_step(dtau_rows, physical_time=None, advance_ramp=True):
            """一个 Newton 步（`dtau_rows` 是伪时间步长天花板，CFL 律见 adaptive_cfl/ser.py），新状态写回后端。"""
            nonlocal coupled_now
            if coupled:
                coupled_now = backend.coupled_backend(nu_av)
                return step_coupled_newton(solver, coupled_now, dtau_rows, physical_time=physical_time,
                                           advance_ramp=advance_ramp)
            u_next, info = step_mean_flow_newton(
                solver, residual, backend.state_rows(), dtau_rows, scales, red=backend.reductions,
                cell_is_prism=backend.cell_is_prism, order=backend.order, positivity=positivity,
                physical_time=physical_time, **backend.newton_options(mu_t, nu_av))
            backend.set_state_rows(u_next)
            return info

        if dual_time:
            from autoflowcfd.core.time_integration.implicit.newton.physical_time import (
                advance_physical_step, previous_newton_state, store_newton_state,
            )

            x_now = backend.coupled_backend(nu_av).state() if coupled else U_rows.copy()
            advance_physical_step(
                solver, x_now, previous_newton_state(solver, xp, n_cells, n_sps),
                lambda term, first: newton_step(backend.pseudo_time_step(nu_av), term, first), dt)
            store_newton_state(solver, x_now, n_cells, n_sps, n_vars)
        else:
            nk_info = newton_step(dt_rows)
    else:
        residual0 = precondition(raw0, U_rows, inplace=True)    # 监控已取完，物理残差不再需要
        backend.set_state_rows(solver.time_integrator.step(
            U_rows, residual, dt_rows, residual0=residual0, positivity_func=positivity))

    backend.end_step(mu_t, coupled_now)
    solver._residual_turbulence = explicit_turbulence_residual(
        backend.reductions, backend.turbulence_model, backend.cell_is_prism, backend.order)

    controller = solver._cfl_controller
    if controller is not None and not dual_time:
        if nk_info is not None:
            # SER 看 Newton 实际在解的系统的残差 ||Gamma R||，并区分"残差上升但步被完整接受"（物理暂态，保持）
            # 与"步没被完整接受"（收缩），见 adaptive_cfl/ser.py
            from autoflowcfd.core.time_integration.implicit.newton.mean_flow_step import (
                newton_step_ok, newton_step_ratio,
            )

            controller.update(nk_info["res_norm"], step_ok=newton_step_ok(nk_info),
                              linear_converged=nk_info["gmres_info"] == 0, step_ratio=newton_step_ratio(nk_info))
        else:
            controller.update(residual_norm)
    return residual_norm
