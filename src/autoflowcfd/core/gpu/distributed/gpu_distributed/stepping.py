"""AutoFlowCFD V2.0 - `MultiGPUDistributedSolver` 的单步推进（mixin，只含方法）。

从 `core/gpu/distributed/gpu_distributed/` 拆出（2026-09-25）。求解循环与其余后端共用
`fr_solver/solver/solve_loop.py::SolveLoopMixin`（2026-10-05 起）。
"""

from typing import Any

from functools import partial

from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.time_integration.step_orchestration import advance_step


class _MultiGPUSteppingMixin:
    """完全分布式加载入口、阶数切换与单步推进。"""

    # 宿主类在 __init__ / 其它 mixin 中提供的属性（仅类型声明）
    _compute_local_time_step_gpu: Any
    _compute_turbulence_source_distributed: Any
    _get_positivity_limiter_gpu: Any
    _inv_perm_gpu: Any
    _perm_gpu: Any
    boundary_ghost_provider: Any
    compute_artificial_diffusivity_compact_gpu: Any
    device_id: Any
    dist_flat_face: Any
    freestream: Any
    gpu_halo: Any
    mesh: Any
    order: Any
    partition: Any
    time_integrator: Any
    turb_model_gpu: Any
    turb_model_name: Any
    U_gpu: Any

    @classmethod
    def from_fully_distributed_package(cls, package: dict, n_ranks: int,
                                       device_id=None, rank=None, root_context=None):
        """真正的"完全分布式加载"构造入口（#1，2026-09-02）——与主
        `__init__`（"传统模式"：每个 rank 独立加载完整全局网格）的关键
        区别：`package` 是 root rank 预先算好、已经按本 rank 的 compact
        索引空间切好的紧凑数据，本 rank 从未持有、也不需要持有完整
        全局网格。与 CPU `DistributedFRSolver.from_fully_distributed_
        package` 共用同一套 `build_fully_distributed_rank_package`/
        `distributed_mesh_load_v2`（package 构造逻辑与后端无关）。

        实现拆到独立模块 `gpu_distributed_fully_distributed/`（控制
        单文件行数，与 `_interpolate_to_new_order`/
        `gpu_distributed_order_continuation.py` 同一个拆分动机），见该
        模块文档。全部湍流模型（`core/turbulence/registry.py::SUPPORTED_MODELS`，与单机相同）、全部时间格式、
        checkpoint 续算与 Order Continuation 均已接入。

        Args:
            package: `build_fully_distributed_rank_package` 的返回值
                （或 `distributed_mesh_load_v2` 经 MPI 收发后本 rank
                收到的那一份）
            n_ranks: MPI rank 总数
            device_id: GPU 设备号（None 时按 rank 轮询分配）
            rank: 当前 rank（None 时从 MPI 获取）
            root_context: 仅 root rank 需要非 None——`distributed_mesh_
                load_v2` 返回的第二个值，供 Order Continuation 使用，
                见 `gpu_distributed_fully_distributed/` 模块文档。

        Returns:
            MultiGPUDistributedSolver 实例
        """
        from autoflowcfd.core.gpu.distributed.gpu_distributed_fully_distributed import (
            build_multi_gpu_solver_from_fully_distributed_package,
        )
        return build_multi_gpu_solver_from_fully_distributed_package(
            cls, package, n_ranks, device_id=device_id, rank=rank, root_context=root_context,
        )

    def _interpolate_to_new_order(self, target_p: int) -> None:
        """阶数切换（2026-09-02，见 core/gpu/distributed/
        gpu_distributed_order_continuation.py 模块文档）——与 CPU
        `DistributedFRSolver._interpolate_to_new_order` 同一个命名/
        调用约定，供 `core/utils/order_continuation/run.py::run_order_continuation`（四个后端
        共用同一份迭代循环）统一调用。"""
        from autoflowcfd.core.gpu.distributed.gpu_distributed_order_continuation import (
            gpu_interpolate_to_new_order,
        )
        gpu_interpolate_to_new_order(self, target_p)

    def set_dual_time_history(self, U_prev_flat) -> None:
        """恢复 dual-time 的上一物理时间层（本 rank local 段，放到本 GPU 上）。"""
        cp = get_cupy()
        with cp.cuda.Device(self.device_id):
            self._dual_time_U_prev = cp.asarray(U_prev_flat, dtype=cp.float64)

    def _limit_prolongated_state(self) -> None:
        """升阶延拓之后在**新阶数**的点集（解点 + 面通量点 + 过积分细点）上施加守恒的
        正性限制器（`time_integration/positivity`，向单元均值收缩、均值不变）。

        低阶多项式只在低阶那组点上被保证可容许；新阶数的点落在别处，延拓后的状态
        可能在那里 rho 或 p 非正——隐式 Newton 的物理性限幅假定出发态处处可容许，
        于是第一步残差就算在非物理态上（plate_demo P1->P2 实测：P2 第 1 步残差
        2.6e27、dtau 缩到下限仍拿不到被接受的步）。必须在新阶数几何就位之后调用。
        """
        cp = get_cupy()
        U = cp.ascontiguousarray(self.U_gpu)
        self._get_positivity_limiter_gpu()(U.reshape(-1, U.shape[-1]))
        self.U_gpu = U

    def step(self, dt: float = 0.0) -> float:
        """推进一步（多 GPU），返回全局残差范数（流程与 `dt` 的语义见 `step_orchestration.advance_step`）。"""
        return advance_step(self, MultiGpuStepBackend(self), dt)


class MultiGpuStepBackend:
    """多 GPU（传统模式与完全分布式加载）的单步原语（`step_orchestration.StepBackend`）。

    状态 `U_gpu` 是本 rank local 单元（原生排列）；残差组装与 halo 交换读 `U_gpu`，试探态临时写进去、算完恢复；
    局部步长与涡粘在紧凑（local + halo）索引空间给出，步长经 `_inv_perm_gpu` 取回 local 段。
    """

    def __init__(self, solver):
        from autoflowcfd.core.mpi.distributed_flat_face import native_cell_is_prism
        from autoflowcfd.core.mpi.reductions import MPIReductions

        self.solver = solver
        self.xp = get_cupy()
        self.n_cells = int(solver.partition.n_local_cells)
        self.n_sps = int(solver.mesh.n_sps_per_cell)
        self.n_vars = 5
        self.reductions = MPIReductions(self.xp)
        self.cell_is_prism = native_cell_is_prism(solver.dist_flat_face)[:self.n_cells]
        self.order = int(getattr(solver, "current_order", solver.order))
        self.freestream = solver.freestream
        self.ghost_provider = solver.boundary_ghost_provider
        self.turbulence_model = solver.turb_model_gpu
        self._coarse = None
        self._coupled = None

    def _rows(self, compact):
        local = compact[self.solver._inv_perm_gpu][:self.n_cells]
        return self.xp.broadcast_to(local[:, None], (self.n_cells, self.n_sps)).reshape(self.n_cells * self.n_sps)

    def begin_step(self) -> None:
        return None

    def artificial_diffusivity(self):
        return self.solver.compute_artificial_diffusivity_compact_gpu()

    def local_time_steps(self, nu_av):
        dt_mean, dt_physical = self.solver._compute_local_time_step_gpu(return_physical_too=True, nu_av_compact=nu_av)
        cells = self.xp.asnumpy(dt_mean[self.solver._inv_perm_gpu][:self.n_cells])
        return self._rows(dt_mean), dt_physical[:, None], cells

    def pseudo_time_step(self, nu_av):
        return self._rows(self.solver._compute_local_time_step_gpu(nu_av_compact=nu_av))

    def turbulence(self, dt_turbulence, coupled: bool):
        s = self.solver
        if coupled:
            from autoflowcfd.core.gpu.distributed.gpu_distributed_implicit import MultiGpuTurbulenceBackend

            turb = MultiGpuTurbulenceBackend(s, self.cell_is_prism, self.order)
            turb.prepare_inputs()
            turb.rates()
            return turb.trial_mu_t_compact()
        return s._compute_turbulence_source_distributed(dt_turbulence)

    def state_rows(self):
        return self.solver.U_gpu.reshape(self.n_cells * self.n_sps, self.n_vars)

    def set_state_rows(self, rows) -> None:
        self.solver.U_gpu = rows.reshape(self.n_cells, self.n_sps, self.n_vars)

    def primitives(self, rows):
        from autoflowcfd.core.gpu.residual.gpu_flux import conserved_to_primitive_gpu

        return conserved_to_primitive_gpu(rows.reshape(self.n_cells, self.n_sps, self.n_vars))

    def residual_raw(self, rows, mu_t, nu_av):
        s = self.solver
        saved = s.U_gpu
        s.U_gpu = rows.reshape(self.n_cells, self.n_sps, self.n_vars)
        try:
            res = s._compute_total_residual_gpu(mu_t_field=mu_t, nu_av_compact=nu_av)
        finally:
            s.U_gpu = saved
        return (-res).reshape(self.n_cells * self.n_sps, self.n_vars)

    def record_monitor_residual(self, raw) -> None:
        return None

    def _coarse_context(self):
        if self._coarse is None:
            from autoflowcfd.core.mpi.distributed_coarse import CompactCellValues, coarse_comm_context

            s = self.solver
            self._coarse = coarse_comm_context(s.partition, CompactCellValues(s.gpu_halo, s._perm_gpu, self.xp))
        return self._coarse

    def coupled_backend(self, nu_av):
        if self._coupled is None:
            from autoflowcfd.core.gpu.distributed.gpu_distributed_implicit import MultiGpuCoupledBackend

            self._coupled = MultiGpuCoupledBackend(self.solver, self.cell_is_prism, self.order, nu_av,
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
                s, s, order=self.order, mu_t_compact=mu_t, exchange=s.gpu_halo.exchange, perm=s._perm_gpu,
                n_sps=self.n_sps, nu_av_compact=nu_av))

    def positivity(self):
        return self.solver._get_positivity_limiter_gpu()

    def end_step(self, mu_t, coupled) -> None:
        return None          # SGS 涡粘在下一步开头的湍流更新里按当前状态求（`_compute_turbulence_source_distributed`）
