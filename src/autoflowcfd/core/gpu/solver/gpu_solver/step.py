"""AutoFlowCFD V2.0 - 单机 GPU（`GPUFRSolver`）的单步推进：一步的流程由四个后端共用的
`time_integration/step_orchestration.py::advance_step` 编排，这里只提供单 GPU 的原语（`GpuStepBackend`）。

`GPUFRSolver` 的一组方法（mixin），只含方法，属性由 `GPUFRSolver.__init__` 建立。
"""

from functools import partial
from typing import Any

import numpy as np

from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.time_integration.implicit.reductions import LocalReductions
from autoflowcfd.core.time_integration.step_orchestration import advance_step


class GpuStepBackend:
    """单 GPU 的单步原语（`step_orchestration.StepBackend`）；步长是逐单元的，铺到全部解点。"""

    def __init__(self, solver):
        self.solver = solver
        self.xp = get_cupy()
        self.n_cells = int(solver.mesh.n_cells)
        self.n_sps = int(solver.mesh.n_sps_per_cell)
        self.n_vars = int(solver.n_vars)
        self.reductions = LocalReductions(self.xp)
        self.cell_is_prism = np.arange(self.n_cells) < int(solver.mesh.n_prism_cells)
        self.order = int(solver.order if solver.current_order is None else solver.current_order)
        self.freestream = solver.freestream
        self.ghost_provider = solver.boundary_ghost_provider
        self.turbulence_model = solver.turb_model_gpu

    def _rows(self, per_cell):
        return self.xp.broadcast_to(per_cell[:, None], (self.n_cells, self.n_sps)).reshape(self.n_cells * self.n_sps)

    def begin_step(self) -> None:
        self.solver._update_primitives_gpu()

    def artificial_diffusivity(self):
        return self.solver.compute_artificial_diffusivity_field_gpu()

    def local_time_steps(self, nu_av):
        dt_local, dt_physical = self.solver._compute_local_time_step_gpu(return_physical_too=True, nu_av=nu_av)
        return self._rows(dt_local), dt_physical[:, None], self.xp.asnumpy(dt_local)

    def pseudo_time_step(self, nu_av):
        return self._rows(self.solver._compute_local_time_step_gpu(nu_av=nu_av))

    def turbulence(self, dt_turbulence, coupled: bool):
        if coupled:
            return self.solver._turbulent_mu_t_gpu()
        return self.solver.compute_turbulence_source_gpu(dt_turbulence)

    def state_rows(self):
        return self.solver.U_gpu.reshape(self.n_cells * self.n_sps, self.n_vars)

    def set_state_rows(self, rows) -> None:
        self.solver.U_gpu = rows.reshape(self.n_cells, self.n_sps, self.n_vars)
        self.solver._update_primitives_gpu()

    def primitives(self, rows):
        from autoflowcfd.core.gpu.residual.gpu_flux import conserved_to_primitive_gpu

        return conserved_to_primitive_gpu(rows.reshape(self.n_cells, self.n_sps, self.n_vars)[..., :5])

    def residual_raw(self, rows, mu_t, nu_av):
        U = rows.reshape(self.n_cells, self.n_sps, self.n_vars)
        total = (self.solver.compute_inviscid_residual_gpu(U)
                 + self.solver.compute_viscous_residual_gpu(U, mu_t_field=mu_t, nu_av=nu_av))
        return (-total).reshape(self.n_cells * self.n_sps, self.n_vars)

    def record_monitor_residual(self, raw) -> None:
        return None

    def coupled_backend(self, nu_av):
        from autoflowcfd.core.gpu.turbulence.gpu_implicit_turbulence import GpuCoupledBackend

        return GpuCoupledBackend(self.solver, nu_av)

    def newton_options(self, mu_t, nu_av) -> dict:
        from autoflowcfd.core.fr_residual.jacobian.backend import MeanFlowBlockAssembler, unsupported_reason
        from autoflowcfd.core.gpu.turbulence.gpu_implicit_turbulence import gpu_cell_colors, gpu_coupling_graph

        s, cp, ff = self.solver, self.xp, self.solver.flat_face_gpu
        # 解析单元块在主机上装配（与 CPU 同一份实现），块由缓存上传
        assembler = None if unsupported_reason(order=self.order) else MeanFlowBlockAssembler(
            mesh=s.mesh, ops=s.ops, ghost_provider=s.boundary_ghost_provider, mu=s.mu_molecular,
            mach_ref=s.freestream["mach_ref"], low_mach=s.low_mach_precond_enabled, mu_t=mu_t, nu_av=nu_av,
            n_sps=self.n_sps, wall_model=s.wall_model)
        return dict(cell_colors=lambda: gpu_cell_colors(cp, ff, self.n_cells),
                    coupling_graph=partial(gpu_coupling_graph, cp, ff, self.n_cells), block_assembler=assembler)

    def positivity(self):
        from autoflowcfd.core.time_integration.positivity import get_positivity_limiter

        return get_positivity_limiter(self.solver, xp=self.xp)

    def end_step(self, mu_t, coupled) -> None:
        # SGS（WALE）涡粘在新状态上更新，供下一步的粘性残差使用（`gpu_solver_io.py::_apply_turbulence_corrections_gpu`）
        self.solver._apply_turbulence_corrections_gpu()


class _GPUSolverStepMixin:
    """单机 GPU 的单步推进（求解循环与 CPU 共用 `SolveLoopMixin`）"""

    def step(self: Any, dt: float = 0.0) -> float:
        """推进一步，返回残差范数（流程与 `dt` 的语义见 `step_orchestration.advance_step`）。"""
        return advance_step(self, GpuStepBackend(self), dt)
