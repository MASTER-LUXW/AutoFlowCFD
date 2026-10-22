"""AutoFlowCFD V2.0 - 单机 CPU（`FRSolver`）的单步推进：一步的流程由四个后端共用的
`time_integration/step_orchestration.py::advance_step` 编排，这里只提供 CPU 单机的原语（`CpuStepBackend`）。

`dt` 的语义（三种情形）、湍流的算子分裂与紧耦合、低马赫预处理与监控残差的分工见 `step_orchestration.py` 模块
文档。湍流显式更新用逐解点的局部步长（随阶数 / 粘性刚性 / 几何退化收紧的那一份，`cfl.py`）而不是调用方传入的
固定 `dt`：2026-08 以前用固定 `dt`，比局部步长大 178~790 倍，omega 一步放大 23 倍，P0 -> P1 后发散。
"""

from functools import partial
from typing import Any

import numpy as np

from autoflowcfd.core.fr_solver.turbulence.implicit import (
    CpuCoupledBackend,
    single_machine_cell_colors,
    single_machine_coupling_graph,
)
from autoflowcfd.core.time_integration.implicit.reductions import LocalReductions
from autoflowcfd.core.time_integration.step_orchestration import advance_step


class CpuStepBackend:
    """单机 CPU 的单步原语（`step_orchestration.StepBackend`）。"""

    def __init__(self, solver):
        self.solver = solver
        self.xp: Any = np
        self.n_cells, self.n_sps, self.n_vars = solver.state.U.shape
        self.reductions = LocalReductions(np)
        self.cell_is_prism = np.arange(self.n_cells) < int(solver.mesh.n_prism_cells)
        self.order = _current_order(solver)
        self.freestream = solver.freestream
        self.ghost_provider = solver.boundary_ghost_provider
        self.turbulence_model = solver.turb_model

    def begin_step(self) -> None:
        self.solver.state._update_primitives()

    def artificial_diffusivity(self):
        return self.solver.compute_artificial_diffusivity_field()

    def local_time_steps(self, nu_av):
        dt_local, dt_physical = self.solver._compute_local_time_step(return_physical_too=True, nu_av=nu_av)
        dt_local = np.asarray(dt_local)
        cells = (dt_local.reshape(self.n_cells, self.n_sps)[:, 0] if dt_local.ndim > 1
                 else dt_local[::max(self.n_sps, 1)])
        return dt_local.reshape(self.n_cells * self.n_sps), dt_physical, cells

    def pseudo_time_step(self, nu_av):
        return self.solver._compute_local_time_step(nu_av=nu_av).reshape(self.n_cells * self.n_sps)

    def turbulence(self, dt_turbulence, coupled: bool):
        from autoflowcfd.core.fr_solver.turbulence.corrections import get_turbulent_viscosity_field

        if not coupled:
            self.solver.compute_turbulence_source(dt_turbulence)
        return get_turbulent_viscosity_field(self.solver)

    def state_rows(self):
        return self.solver.state.U.reshape(self.n_cells * self.n_sps, self.n_vars)

    def set_state_rows(self, rows) -> None:
        self.solver.state.U = rows.reshape(self.n_cells, self.n_sps, self.n_vars)
        self.solver.state._update_primitives()

    def primitives(self, rows):
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

        return conserved_to_primitive(rows.reshape(self.n_cells, self.n_sps, self.n_vars)[..., :5])

    def residual_raw(self, rows, mu_t, nu_av):
        state = self.solver.state
        saved = state.U
        state.U = rows.reshape(self.n_cells, self.n_sps, self.n_vars)
        try:
            res = self.solver.compute_mean_flow_residual(mu_t_turb=mu_t, nu_av=nu_av)
        finally:
            state.U = saved
        return res.reshape(self.n_cells * self.n_sps, self.n_vars)

    def record_monitor_residual(self, raw) -> None:
        self.solver.state.dU_dt = (-raw).reshape(self.n_cells, self.n_sps, self.n_vars)

    def coupled_backend(self, nu_av):
        return CpuCoupledBackend(self.solver, nu_av)

    def newton_options(self, mu_t, nu_av) -> dict:
        from autoflowcfd.core.fr_residual.jacobian.backend import MeanFlowBlockAssembler, unsupported_reason

        s = self.solver
        assembler = None if unsupported_reason(order=self.order) else MeanFlowBlockAssembler(
            mesh=s.mesh, ops=s.ops, ghost_provider=s.boundary_ghost_provider, mu=s.mu_molecular,
            mach_ref=s.freestream["mach_ref"], low_mach=s.low_mach_precond_enabled,
            mu_t=s._get_turbulent_viscosity_field(mu_t), nu_av=nu_av, n_sps=self.n_sps, wall_model=s.wall_model)
        return dict(cell_colors=lambda: single_machine_cell_colors(s),
                    coupling_graph=partial(single_machine_coupling_graph, s), block_assembler=assembler)

    def positivity(self):
        from autoflowcfd.core.time_integration.positivity import get_positivity_limiter

        return get_positivity_limiter(self.solver)

    def end_step(self, mu_t, coupled) -> None:
        # SGS 涡粘在新状态上更新，供下一步的粘性残差使用（湍流输运场由模型对象自己更新，状态数组只含平均流）
        self.solver.apply_turbulence_corrections()


def step(solver, dt: float) -> float:
    """推进一步，返回残差范数（流程与 `dt` 的语义见 `step_orchestration.advance_step`）。"""
    return advance_step(solver, CpuStepBackend(solver), dt)


def _current_order(solver) -> int:
    order = getattr(solver, "current_order", None)
    return int(order if order is not None else solver.order)
