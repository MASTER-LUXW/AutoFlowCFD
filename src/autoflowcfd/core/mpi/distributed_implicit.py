"""AutoFlowCFD V2.0 - CPU-MPI 分布式的隐式稳态（Newton-Krylov）后端适配。

隐式步的算法本身只有一份，这里只回答"分布式上用哪一套求值件"：

* 平均流 Newton 步：`time_integration/implicit/mean_flow_step.py::
  step_mean_flow_newton`，归约对象换成 `core/mpi/reductions.py::
  MPIReductions`（GMRES 内积、残差 RMS、线搜索与物理性限幅都是全局量）；
* 隐式 k-omega：`fr_solver/turbulence/implicit.py` 的 `TurbulenceResidual`
  / `step_turbulence_newton`，本文件提供它要的适配器
  `DistributedTurbulenceBackend`；
* 块 Jacobi 着色：`global_cell_colors`。

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
from autoflowcfd.core.time_integration.implicit.block_jacobi import greedy_cell_coloring
from autoflowcfd.core.turbulence.transport import omega_wall_cell_targets


def global_cell_colors(face_connectivity, n_global_cells: int) -> np.ndarray:
    """全局面连接关系上的贪心距离 1 着色，返回 `(n_global_cells,)`（见模块文档）。"""
    return greedy_cell_coloring(face_connectivity.owner_cell, face_connectivity.neighbor_cell,
                                int(n_global_cells))


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


class DistributedTurbulenceBackend:
    """`DistributedFRSolver`（CPU-MPI）的隐式 k-omega 适配器，接口见
    `fr_solver/turbulence/implicit.py` 模块文档"后端"一节。

    额外产出 `mu_t_compact`：最后一次（`apply_des=True`）求值之后 compact
    空间的涡粘，供本步平均流的粘性残差使用（与显式路径的返回值同一个量）。
    """

    __slots__ = ("solver", "model", "xp", "red", "shape", "cell_is_prism", "order",
                 "_adapter", "_view", "_inputs", "mu_t_compact")

    def __init__(self, solver, cell_is_prism: np.ndarray, order: int):
        self.solver = solver
        self.model = solver.turb_model
        self.xp = np
        self.red = MPIReductions(np)
        self.shape = self.model.k_field.shape
        self.cell_is_prism = cell_is_prism
        self.order = int(order)
        self._adapter = self._view = self._inputs = None
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

    def rates(self, apply_des: bool):
        self._sync_view()
        _, _, dk, dw, tk, tw = evaluate_turbulence_rates(self._adapter, *self._inputs,
                                                         apply_des=apply_des)
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

    def finalize(self, dtau) -> None:
        # 模态滤波在 compact 视图上做（它按"棱柱在前"分块），再写回 local
        self._sync_view()
        finalize_turbulence_update(self._adapter, dtau, omega_wall_relaxation=False)
        self.model.k_field = self._to_local(self._view.k_field).copy()
        self.model.omega_field = self._to_local(self._view.omega_field).copy()

    def cell_colors(self):
        return distributed_block_jacobi_colors(self.solver)
