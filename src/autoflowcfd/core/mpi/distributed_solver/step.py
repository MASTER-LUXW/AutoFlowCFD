"""AutoFlowCFD V2.0 - CPU MPI 分布式的单步推进与局部时间步长（求解循环在 solve_loop.py）

`DistributedFRSolver` 的一组方法（mixin）。

**只含方法，没有状态**：全部属性由 `DistributedFRSolver` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

from typing import Any

from functools import partial

import numpy as np

from autoflowcfd.core.mpi.reductions import MPIReductions
from autoflowcfd.core.time_integration.step_orchestration import advance_step


class _DistributedStepMixin:
    """CPU MPI 分布式的单步推进与局部时间步长"""

    # 宿主类在 __init__ / 其它 mixin 中提供的属性（仅类型声明）
    _cfl_controller: Any
    _distributed_positivity_limiter: Any
    _newton_last_info: Any
    _prev_mu_t_local: Any
    artificial_viscosity_alpha: Any
    artificial_viscosity_enabled: Any
    ddes_model: Any
    dist_flat_face: Any
    fixed_cfl_number: Any
    halo_exchange: Any
    iddes_h_max_compact: Any
    iddes_h_wn_compact: Any
    local_solver: Any
    low_mach_precond_enabled: Any
    mesh: Any
    ops: Any
    order: Any
    partition: Any
    sgs_model: Any
    state: Any
    time_integrator: Any
    turb_model: Any
    turb_model_name: Any
    wall_distance_compact: Any
    wall_model: Any

    def _compute_distributed_local_time_step(self, U_local, mu_t_local=None,
                                             return_physical_too: bool = False):
        """逐单元局部 CFL 步长（2026-09-14 补齐，取代此前的全局固定 dt）。

        此前这条路径用调用方传入的全局固定 dt 铺满所有单元，源码里记作
        "已接受的简化"。那条"简化"有三重真实代价（步长被全场最苛刻单元
        卡死、逐 SP 的几何/度量 CFL 保护完全失效、自适应 CFL 与低马赫数
        预处理都无从生效），完整论证与实现见
        `core/mpi/distributed_cfl.py` 模块文档。

        需要一次额外的 halo 交换：局部 dt 的谱半径求和必须读到邻居单元
        （含 halo）的速度与声速，这是这项功能的内在需求，不是可以省掉的
        开销（相比之下 RK3 每个 stage 各有一次交换）。
        """
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        from autoflowcfd.core.mpi.comm import allreduce_min
        from autoflowcfd.core.mpi.distributed_cfl import (
            compute_distributed_local_time_step,
        )
        from autoflowcfd.core.time_integration.time_accurate import uniform_time_step

        dist_fc = self.dist_flat_face
        U_extended = self.halo_exchange.exchange(U_local)
        U_compact = U_extended[dist_fc.perm]
        Q_compact = conserved_to_primitive(U_compact[..., :5])

        # 几何量（jacobians/cell_volumes）复用 DistributedMeshAdapter 的
        # 抽取逻辑——它已经处理好"传统模式 vs 完全分布式加载"两种索引
        # 语义的区别（见该类文档），这里不重复那段判断。
        from autoflowcfd.core.mpi.distributed_compute import distributed_mesh_adapter
        adapter = distributed_mesh_adapter(self.partition, dist_fc, self.mesh, self.ops)

        # 时间精确子步：全场同一步长（跨 rank 最小值），见 time_integration/time_accurate.py
        return uniform_time_step(self, compute_distributed_local_time_step(
            U_compact, Q_compact, dist_fc, self.mesh,
            jacobians=adapter.jacobians,
            cell_volumes=adapter.cell_volumes,
            n_local_cells=self.partition.n_local_cells,
            mu_molecular=self.local_solver.mu_molecular,
            freestream=self.local_solver.freestream,
            cfl_controller=self._cfl_controller,
            fixed_cfl_number=self.fixed_cfl_number,
            current_order=getattr(self, "current_order", self.order),
            low_mach_precond_enabled=getattr(
                self, "low_mach_precond_enabled", False),
            mu_t_local=mu_t_local,
            return_physical_too=return_physical_too,
        ), return_physical_too, reduce_min=allreduce_min)

    def _viscous_enabled(self) -> bool:
        """粘性残差是否计算：完全分布式加载的替身显式给出，真实 `FRSolver`（传统模式）恒为真
        （它没有 `.config`，见 `step` 里 2026-09-02 的修复记录）。"""
        cfg = getattr(self.local_solver, "config", None)
        return cfg.physics.enable_viscous if cfg is not None else True

    def _inviscid_dudt_local(self, U_local: np.ndarray) -> np.ndarray:
        """无粘 dU/dt（含本次求值的 halo 交换），`(n_local, n_sps, n_vars)`。"""
        from autoflowcfd.core.mpi.distributed_compute import distributed_compute_inviscid_residual

        ls = self.local_solver
        return distributed_compute_inviscid_residual(
            U_local, self.partition, self.halo_exchange, self.dist_flat_face, self.mesh, self.ops,
            ls.boundary_ghost_provider, mach_ref=ls.freestream["mach_ref"])

    def _viscous_dudt_local(self, U_local: np.ndarray, mu_t_compact, nu_av_compact) -> np.ndarray:
        """粘性（含湍流涡粘耦合与问题单元人工扩散）dU/dt（含本次求值的 halo 交换）。"""
        from autoflowcfd.core.mpi.distributed_compute import distributed_compute_viscous_residual

        ls = self.local_solver
        dudt = distributed_compute_viscous_residual(
            U_local, self.partition, self.halo_exchange, self.dist_flat_face, self.mesh, self.ops,
            ls.mu_molecular, ls.boundary_ghost_provider, mu_t_field_compact=mu_t_compact,
            wall_model=self.wall_model)
        if nu_av_compact is not None:
            from autoflowcfd.core.mpi.distributed_artificial_viscosity import (
                distributed_artificial_diffusion_dudt,
            )
            dudt = dudt + distributed_artificial_diffusion_dudt(
                U_local, nu_av_compact, self.partition, self.halo_exchange, self.dist_flat_face, self.mesh,
                self.ops)
        return dudt

    def _mean_flow_dudt_local(self, U_local: np.ndarray, mu_t_compact, nu_av_compact) -> np.ndarray:
        """平均流 dU/dt（无粘 + 粘性），`(n_local, n_sps, n_vars)`。"""
        dudt = self._inviscid_dudt_local(U_local)
        if self._viscous_enabled():
            dudt = dudt + self._viscous_dudt_local(U_local, mu_t_compact, nu_av_compact)
        return dudt

    def step(self, dt: float) -> float:
        """推进一步（分布式），返回全局残差范数（流程与 `dt` 的语义见 `step_orchestration.advance_step`）。"""
        return advance_step(self, DistributedStepBackend(self), dt)


class DistributedStepBackend:
    """CPU-MPI 分布式（传统模式与完全分布式加载）的单步原语（`step_orchestration.StepBackend`）。

    残差每次求值都重新做 halo 交换（`_mean_flow_dudt_local`）；局部步长需要一次额外的 halo 交换（谱半径求和要读
    邻居单元的速度与声速）；涡粘按紧凑（local + halo）索引空间给出，本步冻结。
    """

    def __init__(self, solver):
        from autoflowcfd.core.mpi.distributed_flat_face import native_cell_is_prism

        self.solver = solver
        self.xp: Any = np
        self.n_cells = int(solver.partition.n_local_cells)
        self.n_sps = int(solver.state.n_sps)
        self.n_vars = int(solver.state.n_vars)
        self.reductions = MPIReductions(np)
        self.cell_is_prism = native_cell_is_prism(solver.dist_flat_face)[:self.n_cells]
        self.order = int(getattr(solver, "current_order", solver.order))
        self.freestream = solver.local_solver.freestream
        self.ghost_provider = solver.local_solver.boundary_ghost_provider
        self.turbulence_model = solver.turb_model
        self._mu_cfl_local = None          # 粘性步长限制用的附加粘度（上一步涡粘 + 本步人工粘性）
        self._coarse = None
        self._coupled = None

    def _rows3(self, rows):
        return rows.reshape(self.n_cells, self.n_sps, self.n_vars)

    def begin_step(self) -> None:
        return None                         # 原始变量随每次写回守恒量同步（set_state_rows）

    def artificial_diffusivity(self):
        s = self.solver
        if not s.artificial_viscosity_enabled:
            return None
        from autoflowcfd.core.mpi.distributed_artificial_viscosity import distributed_artificial_diffusivity

        return distributed_artificial_diffusivity(
            s.state.get_local_U()[..., :5], s.partition, s.halo_exchange, s.dist_flat_face, s.mesh, s.ops,
            order=self.order, alpha_av=s.artificial_viscosity_alpha)

    def local_time_steps(self, nu_av):
        s = self.solver
        U_local = s.state.get_local_U()
        mu_cfl = s._prev_mu_t_local
        if nu_av is not None:
            from autoflowcfd.core.mpi.distributed_artificial_viscosity import local_from_compact

            mu_av = U_local[..., 0] * local_from_compact(nu_av, s.dist_flat_face, self.n_cells)
            mu_cfl = mu_av if mu_cfl is None else mu_cfl + mu_av
        self._mu_cfl_local = mu_cfl
        dt_mean, dt_physical = s._compute_distributed_local_time_step(
            U_local, mu_t_local=mu_cfl, return_physical_too=True)
        dt_mean = np.asarray(dt_mean)
        cells = dt_mean.reshape(self.n_cells, self.n_sps)[:, 0] if dt_mean.ndim > 1 else dt_mean[::self.n_sps]
        return dt_mean.reshape(self.n_cells * self.n_sps), dt_physical, cells

    def pseudo_time_step(self, nu_av):
        s = self.solver
        return s._compute_distributed_local_time_step(
            s.state.get_local_U(), mu_t_local=self._mu_cfl_local).reshape(self.n_cells * self.n_sps)

    def turbulence(self, dt_turbulence, coupled: bool):
        s = self.solver
        if coupled:
            from autoflowcfd.core.mpi.distributed_implicit import DistributedTurbulenceBackend

            turb = DistributedTurbulenceBackend(s, self.cell_is_prism, self.order)
            turb.prepare_inputs()
            turb.rates()
            return turb.trial_mu_t_compact()
        if s.turb_model is not None:
            from autoflowcfd.core.mpi.distributed_turbulence import distributed_compute_turbulence_source_and_viscosity

            return distributed_compute_turbulence_source_and_viscosity(
                s.state.get_local_U()[..., :5], s.partition, s.halo_exchange, s.dist_flat_face, s.mesh, s.ops,
                s.turb_model, s.local_solver.mu_molecular, s.wall_distance_compact, dt_turbulence, ramp_owner=s,
                turb_model_name=s.turb_model_name, ddes_model=s.ddes_model,
                iddes_h_max_compact=s.iddes_h_max_compact, iddes_h_wn_compact=s.iddes_h_wn_compact,
                boundary_ghost_provider=self.ghost_provider)
        if s.sgs_model is not None:
            from autoflowcfd.core.mpi.distributed_turbulence import distributed_compute_les_viscosity

            return distributed_compute_les_viscosity(
                s.state.get_local_U()[..., :5], s.partition, s.halo_exchange, s.dist_flat_face, s.mesh, s.ops,
                s.sgs_model, self.ghost_provider)
        return None

    def state_rows(self):
        return self.solver.state.get_local_U().reshape(self.n_cells * self.n_sps, self.n_vars)

    def set_state_rows(self, rows) -> None:
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

        U = self._rows3(rows)
        self.solver.state.U[:self.n_cells] = U
        self.solver.state.Q[:self.n_cells] = conserved_to_primitive(U[..., :5])

    def primitives(self, rows):
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

        return conserved_to_primitive(self._rows3(rows)[..., :5])

    def residual_raw(self, rows, mu_t, nu_av):
        return (-self.solver._mean_flow_dudt_local(self._rows3(rows), mu_t, nu_av)).reshape(
            self.n_cells * self.n_sps, self.n_vars)

    def record_monitor_residual(self, raw) -> None:
        self.solver.state.dU_dt[:self.n_cells] = -self._rows3(raw)

    def _coarse_context(self):
        if self._coarse is None:
            from autoflowcfd.core.mpi.distributed_coarse import CompactCellValues, coarse_comm_context

            s = self.solver
            self._coarse = coarse_comm_context(
                s.partition, CompactCellValues(s.halo_exchange, s.dist_flat_face.perm, np))
        return self._coarse

    def coupled_backend(self, nu_av):
        if self._coupled is None:
            from autoflowcfd.core.mpi.distributed_implicit import DistributedCoupledBackend

            self._coupled = DistributedCoupledBackend(self.solver, self.cell_is_prism, self.order, nu_av,
                                                      self._coarse_context())
        return self._coupled

    def newton_options(self, mu_t, nu_av) -> dict:
        from autoflowcfd.core.mpi.distributed_implicit import (
            distributed_block_jacobi_colors, distributed_coupling_graph, distributed_mean_flow_assembler,
        )

        s, coarse = self.solver, self._coarse_context()
        return dict(
            cell_colors=lambda: distributed_block_jacobi_colors(s),
            coupling_graph=partial(distributed_coupling_graph, s, coarse), global_coarse=coarse,
            block_assembler=distributed_mean_flow_assembler(
                s, s.local_solver, order=self.order, mu_t_compact=mu_t, exchange=s.halo_exchange.exchange,
                perm=s.dist_flat_face.perm, n_sps=self.n_sps, nu_av_compact=nu_av))

    def positivity(self):
        return self.solver._distributed_positivity_limiter()

    def end_step(self, mu_t, coupled) -> None:
        # 粘性步长限制下一步用本步的涡粘（紧耦合时取 Newton 收尾的涡粘）
        if coupled is not None:
            mu_t = coupled.turb.mu_t_compact
        if mu_t is not None:
            self.solver._prev_mu_t_local = mu_t[self.solver.dist_flat_face.inv_perm][:self.n_cells]
