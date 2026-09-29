"""AutoFlowCFD V2.0 - CPU-MPI 分布式的隐式稳态（Newton-Krylov）后端适配。

隐式步的算法本身只有一份，这里只回答"分布式上用哪一套求值件"：

* 平均流 Newton 步：`time_integration/implicit/mean_flow_step.py::
  step_mean_flow_newton`，归约对象换成 `core/mpi/reductions.py::
  MPIReductions`（GMRES 内积、残差 RMS、线搜索与物理性限幅都是全局量）；
* 隐式 k-omega：`fr_solver/turbulence/implicit.py` 的 `TurbulenceResidual`
  / `step_turbulence_newton`，本文件提供它要的适配器
  `DistributedTurbulenceBackend`；
* 块 Jacobi 着色：`global_cell_colors`；P0 差分装配耦合块用的距离 2 着色与本地
  模板单元对：`global_cell_colors_d2` / `distributed_coupling_graph`。

## 着色必须全局一致

着色有限差分装配（`implicit/block_jacobi.py`）同时扰动同色的全部单元，
依据是"同色单元不是面邻居"。分布式下 rank A 的单元与 rank B 的单元可以
是面邻居（彼此的 halo），而残差求值里 halo 交换会把 A 上的扰动带进 B 的
残差——所以同色不相邻必须对**全局**邻接关系成立。各 rank 各自对本地邻接
着色做不到这一点。这里对全局面连接关系做一次贪心着色（与单机同一个
`greedy_cell_coloring`），各 rank 取自己 local 单元那一段：

* 传统模式：每个 rank 都持有全局网格，惰性地各算一次（贪心是确定性的，
  各 rank 结果逐位相同）；
* 完全分布式加载：只有 root 持有全局网格，root 算一次、随紧凑包下发
  （`distributed_mesh_loader/package.py` 的 `cell_colors`）。

P0 的差分装配同时截取面邻居耦合块（`implicit/cell_blocks.py`），要的是距离 2
着色（同色单元既不相邻、也不共享面邻居），同一个理由也必须对全局邻接成立，来源
与上面相同（完全分布式加载随包下发 `cell_colors_d2`）。块 ILU 只用两端都在本
rank 的耦合块（rank 间按块 Jacobi 式分解，与解析装配 `select_rows` 同一个近似）。

单元着色只依赖拓扑，与阶数无关，换阶后沿用。

## 隐式 k-omega 的未知量与求值

未知量是本 rank local 单元的 `(k, omega)`（原生排列，与 `turb_model` 一致）。
每次残差求值：local 场经 2 变量 halo 交换写进 compact 视图
（`distributed_turbulence.py::set_view_k_omega`），在视图上调用单机的
`evaluate_turbulence_rates`，结果按 `inv_perm` 换回原生排列、切 local 段。
halo 交换是集体调用：Newton/GMRES/线搜索里一切决定"再求值一次"的判据都
经过全局归约，各 rank 的求值次数一致。
"""

import numpy as np

from autoflowcfd.core.fr_solver.turbulence.init import _update_production_ramp
from autoflowcfd.core.fr_solver.turbulence.source import (
    evaluate_turbulence_rates,
    finalize_turbulence_update,
    prepare_turbulence_inputs,
)
from autoflowcfd.core.mpi.distributed_turbulence import (
    build_distributed_turbulence_view,
    set_view_k_omega,
)
from autoflowcfd.core.mpi.reductions import MPIReductions
from autoflowcfd.core.time_integration.implicit.coloring import (
    CouplingGraph, distance2_cell_coloring, greedy_cell_coloring, stencil_pairs,
)
from autoflowcfd.core.turbulence.transport import omega_wall_cell_targets, prepare_convection_geometry


def global_cell_colors(face_connectivity, n_global_cells: int) -> np.ndarray:
    """全局面连接关系上的贪心距离 1 着色，返回 `(n_global_cells,)`（见模块文档）。"""
    return greedy_cell_coloring(face_connectivity.owner_cell, face_connectivity.neighbor_cell,
                                int(n_global_cells))


class HaloCompactState:
    """local 单元状态 -> 残差用的"棱柱在前"local+halo 紧凑排列（halo 交换 + perm）。

    解析单元块装配器（`fr_residual/jacobian/backend.py`）用它把 Newton 行上的
    状态扩展到分布式残差的单元空间，与 `distributed_compute.py` 的残差同一个
    交换与换序。做成类而不是闭包（项目规范）。交换是集体操作：块缓存在每个
    rank 上同步地装配（与差分装配同一约束）。
    """

    __slots__ = ("exchange", "perm")

    def __init__(self, exchange, perm):
        self.exchange = exchange
        self.perm = perm

    def __call__(self, U_local):
        return self.exchange(U_local)[self.perm]


def distributed_mean_flow_assembler(solver, physics, *, order: int, mu_t_compact, exchange, perm, n_sps: int):
    """分布式后端（CPU-MPI 与多 GPU 共用）本步的解析单元块装配器；不覆盖时返回 None。

    `physics` 提供残差用的物理参数（`mu_molecular`、`boundary_ghost_provider`、
    `freestream`）：CPU-MPI 是其内部的 `local_solver`，多 GPU 是求解器本身。
    """
    from autoflowcfd.core.fr_residual.jacobian.backend import MeanFlowBlockAssembler, unsupported_reason
    from autoflowcfd.core.mpi.distributed_compute import DistributedMeshAdapter

    if unsupported_reason(order=order, wmles=getattr(solver, "wmles_model", None) is not None):
        return None
    dist_fc = solver.dist_flat_face
    n_local = int(solver.partition.n_local_cells)
    return MeanFlowBlockAssembler(
        mesh=DistributedMeshAdapter(solver.partition, dist_fc, solver.mesh, solver.ops), ops=solver.ops,
        ghost_provider=physics.boundary_ghost_provider, mu=physics.mu_molecular,
        mach_ref=physics.freestream["mach_ref"],
        low_mach=solver.low_mach_precond_enabled, mu_t=mu_t_compact, n_sps=n_sps,
        flat=dist_fc.base_flat, compact_state=HaloCompactState(exchange, perm),
        row_compact=np.asarray(dist_fc.inv_perm)[:n_local])


def distributed_block_jacobi_colors(solver) -> np.ndarray:
    """本 rank local 单元（原生排列）的块 Jacobi 着色。"""
    colors = getattr(solver, "_block_jacobi_colors_local", None)
    if colors is None:
        fc = getattr(solver.mesh, "face_connectivity", None)
        if fc is None:
            raise RuntimeError(
                "分布式隐式步需要全局一致的块 Jacobi 着色：传统模式由全局网格的面连接"
                "关系计算，完全分布式加载由 root 随紧凑包下发（time_scheme 为 "
                "newton_krylov 时）——这里两者都没有。")
        part = solver.partition
        colors = global_cell_colors(fc, part.n_global_cells)[part.local_cells]
        solver._block_jacobi_colors_local = colors
    return colors


def global_cell_colors_d2(face_connectivity, n_global_cells: int) -> np.ndarray:
    """全局面连接关系上的贪心距离 2 着色，返回 `(n_global_cells,)`（见模块文档）。"""
    return distance2_cell_coloring(face_connectivity.owner_cell, face_connectivity.neighbor_cell,
                                   int(n_global_cells))


def distributed_coupling_graph(solver) -> CouplingGraph:
    """本 rank 的 P0 差分耦合图：local 单元（原生排列）的距离 2 着色与两端都在本
    rank 的模板单元对。单元对取自残差本身用的紧凑面几何（`dist_flat_face.base_flat`，
    local+halo 紧凑编号），经 `inv_perm` 换回 local 编号。"""
    colors = getattr(solver, "_coupling_colors_local", None)
    if colors is None:
        fc = getattr(solver.mesh, "face_connectivity", None)
        if fc is None:
            raise RuntimeError(
                "分布式 P0 块 ILU 需要全局一致的距离 2 着色：传统模式由全局网格的面连接关系"
                "计算，完全分布式加载由 root 随紧凑包下发（cell_colors_d2）——这里两者都没有。")
        part = solver.partition
        colors = global_cell_colors_d2(fc, part.n_global_cells)[part.local_cells]
        solver._coupling_colors_local = colors
    dist_fc = solver.dist_flat_face
    n_local = int(solver.partition.n_local_cells)
    flat = dist_fc.base_flat
    own = np.asarray(flat.owner_cell, dtype=np.int64)
    nb = np.asarray(flat.neighbor_cell, dtype=np.int64)
    local = -np.ones(max(int(own.max()), int(nb.max())) + 1, dtype=np.int64)
    local[np.asarray(dist_fc.inv_perm)[:n_local]] = np.arange(n_local)
    lo, ln = local[own], np.where(nb >= 0, local[np.maximum(nb, 0)], -1)
    keep = (lo >= 0) & (ln >= 0)
    rows, cols = stencil_pairs(lo[keep], ln[keep], n_local)
    return CouplingGraph(rows=rows, cols=cols, colors=np.asarray(colors, dtype=np.int64))


class _TurbulenceCompactState:
    """local `(k, w)`（未知量，`w = ln omega`）-> 紧凑空间（与湍流残差 `_sync_view` 同一次
    halo 交换与换序；两者都是线性的，对 w 与对 omega 同样适用）。"""

    __slots__ = ("backend",)

    def __init__(self, backend):
        self.backend = backend

    def __call__(self, kw_local):
        be = self.backend
        s = be.solver
        view = be._view
        saved = (view.k_field, view.omega_field)
        try:
            set_view_k_omega(view, s.turb_halo_exchange, s.dist_flat_face,
                             np.ascontiguousarray(kw_local[..., 0]), np.ascontiguousarray(kw_local[..., 1]))
            return np.stack([view.k_field, view.omega_field], axis=-1)
        finally:
            view.k_field, view.omega_field = saved


class DistributedTurbulenceBackend:
    """`DistributedFRSolver`（CPU-MPI）的隐式 k-omega 适配器，接口见
    `fr_solver/turbulence/implicit.py` 模块文档"后端"一节。

    额外产出 `mu_t_compact`：最后一次（`apply_des=True`）求值之后 compact
    空间的涡粘，供本步平均流的粘性残差使用（与显式路径的返回值同一个量）。
    """

    __slots__ = ("solver", "model", "xp", "red", "shape", "cell_is_prism", "order",
                 "_adapter", "_view", "_inputs", "_conv_geom", "mu_t_compact")

    def __init__(self, solver, cell_is_prism: np.ndarray, order: int):
        self.solver = solver
        self.model = solver.turb_model
        self.xp = np
        self.red = MPIReductions(np)
        self.shape = self.model.k_field.shape
        self.cell_is_prism = cell_is_prism
        self.order = int(order)
        self._adapter = self._view = self._inputs = self._conv_geom = None
        self.mu_t_compact = None

    def _to_local(self, a_compact: np.ndarray) -> np.ndarray:
        return a_compact[self.solver.dist_flat_face.inv_perm][:self.shape[0]]

    def _sync_view(self) -> None:
        s = self.solver
        set_view_k_omega(self._view, s.turb_halo_exchange, s.dist_flat_face,
                         self.model.k_field, self.model.omega_field)

    def prepare(self) -> None:
        s = self.solver
        self._adapter, self._view = build_distributed_turbulence_view(
            s.state.get_local_U()[..., :5], s.partition, s.halo_exchange, s.turb_halo_exchange,
            s.dist_flat_face, s.mesh, s.ops, self.model, s.local_solver.mu_molecular,
            s.wall_distance_compact, turb_ramp_step=s._turb_ramp_step,
            turb_ramp_steps=s._turb_production_ramp_steps, turb_model_name=s.turb_model_name,
            ddes_model=s.ddes_model, iddes_h_max_compact=s.iddes_h_max_compact,
            iddes_h_wn_compact=s.iddes_h_wn_compact,
            des_length_scale_halo_exchange=s.des_length_scale_halo_exchange,
            boundary_ghost_provider=s.local_solver.boundary_ghost_provider)
        _update_production_ramp(self._adapter)
        s._turb_ramp_step = self._adapter._turb_ramp_step
        self._inputs = prepare_turbulence_inputs(self._adapter)
        # 平均流在整个 Newton 步内冻结：标量对流几何算一次，各次求值复用
        self._conv_geom = prepare_convection_geometry(
            self._adapter, self._adapter._turbulence_flat_face_override)

    def rates(self, apply_des: bool):
        self._sync_view()
        _, _, dk, dw, tk, tw = evaluate_turbulence_rates(self._adapter, *self._inputs,
                                                         apply_des=apply_des,
                                                         conv_geom=self._conv_geom)
        rate_k = dk if tk is None else dk + tk
        rate_w = dw if tw is None else dw + tw
        if apply_des:
            # 最终场上的求值：涡粘与 DES 长度尺度写回真正的模型（与显式路径
            # `distributed_compute_turbulence_source_and_viscosity` 的写回相同）
            view = self._view
            self.model.nu_t[:] = self._to_local(view.nu_t)
            if self.solver.ddes_model is not None and getattr(view, "des_length_scale", None) is not None:
                self.model.des_length_scale = self._to_local(view.des_length_scale).copy()
            self.mu_t_compact = self._adapter.state.Q[..., 0] * view.nu_t
        return self._to_local(rate_k), self._to_local(rate_w)

    def wall_targets(self):
        adapter = self._adapter
        hit, target = omega_wall_cell_targets(adapter, adapter._turbulence_flat_face_override)
        native = self.solver.dist_flat_face.perm[hit]
        keep = native < self.shape[0]
        return native[keep], target[keep]

    def positivity(self) -> None:
        self.model.apply_positivity_limiter()

    def norm_weights(self):
        """残差范数的逐行守恒权重（与平均流 Newton 步同一份，见 `jfnk.ResidualNorm`）。"""
        return self.solver._distributed_positivity_limiter().W

    def finalize(self, dtau) -> None:
        # 模态滤波在 compact 视图上做（它按"棱柱在前"分块），再写回 local
        self._sync_view()
        finalize_turbulence_update(self._adapter, dtau, omega_wall_relaxation=False)
        self.model.k_field = self._to_local(self._view.k_field).copy()
        self.model.omega_field = self._to_local(self._view.omega_field).copy()

    def cell_colors(self):
        return distributed_block_jacobi_colors(self.solver)

    def block_assembler(self):
        """本步的解析单元块装配器：在与残差同一个紧凑空间视图上装配，按 `inv_perm` 取本 rank 行。"""
        from autoflowcfd.core.turbulence.jacobian import TurbulenceBlockAssembler, turbulence_linearization

        adapter = self._adapter
        flat = adapter._turbulence_flat_face_override
        hit, target = omega_wall_cell_targets(adapter, flat)
        ctx = turbulence_linearization(adapter, self._view, self._inputs, self._conv_geom, flat, hit, target)
        dist_fc = self.solver.dist_flat_face
        return TurbulenceBlockAssembler(
            ctx, self.shape[1], compact_state=_TurbulenceCompactState(self),
            row_compact=np.asarray(dist_fc.inv_perm)[:self.shape[0]])
