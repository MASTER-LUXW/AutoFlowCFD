"""AutoFlowCFD V2.0 - halo 交换、阶数切换、传感器门控滤波、负载均衡报告

`DistributedFRSolver` 的一组方法（mixin）。

**只含方法，没有状态**：全部属性由 `DistributedFRSolver` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

from typing import Any, Optional

import numpy as np


class _DistributedSupportMixin:
    """halo 交换、阶数切换、传感器门控滤波、负载均衡报告"""

    # 宿主类在 __init__ / 其它 mixin 中提供的属性（仅类型声明）
    _is_fully_distributed: Any
    halo_exchange: Any
    n_ranks: Any
    partition: Any
    rank: Any
    state: Any

    def save_checkpoint_distributed(self, output_dir: str, iteration: int, input_file: str, order: int,
                                    turbulence_model: str, target_order: Optional[int] = None,
                                    surface_mesh: Optional[str] = None):
        """分布式 checkpoint 保存（集体操作，全部 rank 调用）；与多 GPU 同名同签名，CLI 对两个分布式后端
        用同一个回调（`cli/solve/distributed_checkpoint_io.py::distributed_periodic_checkpoint_callback`）。"""
        from autoflowcfd.core.mpi.distributed_checkpoint import distributed_save_checkpoint

        return distributed_save_checkpoint(
            self, output_dir, iteration, input_file, order, turbulence_model, "cpu",
            target_order=target_order, surface_mesh=surface_mesh)

    def load_checkpoint_distributed(self, path: str):
        """把 checkpoint 的状态载入本求解器（集体操作；与多 GPU 同名同签名）。返回 `(metadata, iteration)`。"""
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        from autoflowcfd.core.mpi.distributed_checkpoint import distributed_load_checkpoint

        U_local, metadata, iteration = distributed_load_checkpoint(path, self)
        n_local = self.partition.n_local_cells
        self.state.U[:n_local] = U_local
        self.state.Q[:n_local] = conserved_to_primitive(U_local[..., :5])
        return metadata, iteration

    def local_cell_geometry(self):
        """本 rank local 单元（原生排列，与 `state.U[:n_local]` 同序）的 `(det_jacs (n_local, n_sps), 是否棱柱)`。

        adapter 的 jacobians 在"棱柱在前"紧凑排列，经 inv_perm 换回后切片。正性限制器与写盘的单元积分平均共用。
        """
        from autoflowcfd.core.mpi.distributed_compute import distributed_mesh_adapter
        from autoflowcfd.core.mpi.distributed_flat_face import native_cell_is_prism

        dist_fc = self.dist_flat_face
        n_local = int(self.partition.n_local_cells)
        adapter = distributed_mesh_adapter(self.partition, dist_fc, self.mesh, self.ops)
        n_sps = self.state.U.shape[1]
        det_compact = np.asarray(adapter.jacobians["det_jacs"]).reshape(-1, n_sps)
        return det_compact[dist_fc.inv_perm][:n_local], native_cell_is_prism(dist_fc)[:n_local]

    def _distributed_positivity_limiter(self):
        """正性保持限制器（与单机同一个、同一个核），几何取 `local_cell_geometry`。按阶数缓存。"""
        from autoflowcfd.core.time_integration.positivity import get_positivity_limiter

        return get_positivity_limiter(self, geometry=self.local_cell_geometry)

    def _local_dynamic_eddy_viscosity(self):
        """本 rank local 单元的动力涡粘 rho*nu_t（湍流模型 + 亚格子），层流时 None：最近一步残差实际使用的那一份
        （`step()` 末尾缓存的 `_prev_mu_t_local`）。供 `core/mpi/global_view.py` 汇总。"""
        return None if self._prev_mu_t_local is None else np.asarray(self._prev_mu_t_local)

    def _limit_prolongated_state(self) -> None:
        """升阶延拓之后在**新阶数**的点集（解点 + 面通量点 + 过积分细点）上施加守恒的
        正性限制器（`time_integration/positivity`，向单元均值收缩、均值不变）。

        低阶多项式只在低阶那组点上被保证可容许；新阶数的点落在别处，延拓后的状态
        可能在那里 rho 或 p 非正——隐式 Newton 的物理性限幅假定出发态处处可容许，
        于是第一步残差就算在非物理态上（plate_demo P1->P2 实测：P2 第 1 步残差
        2.6e27、dtau 缩到下限仍拿不到被接受的步）。必须在新阶数几何就位之后调用。
        """
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

        n_local = int(self.partition.n_local_cells)
        U = np.ascontiguousarray(self.state.U[:n_local])
        self._distributed_positivity_limiter()(U.reshape(-1, U.shape[-1]))
        self.state.U[:n_local] = U
        self.state.Q[:n_local] = conserved_to_primitive(U[..., :5])

    def _interpolate_to_new_order(self, target_p: int) -> None:
        """将解从当前阶数插值到新的阶数（分布式 Order Continuation 核心逻辑，见
        core/mpi/distributed_order_continuation/ 包文档）——按本实例的构造方式分派到对应的
        重建函数：'传统模式'（`self._is_fully_distributed is False`，
        每个 rank 持有完整全局网格）复用同一进程内的 `cell_partition`
        重建 partition/dist_flat_face；'完全分布式加载'
        （`self._is_fully_distributed is True`）需要 root 重新计算+
        重新分发紧凑包，见 `redistribute_fully_distributed_for_new_
        order` 文档。
        """
        from autoflowcfd.core.mpi.distributed_order_continuation import (
            cpu_traditional_interpolate_to_new_order,
        )

        if self._is_fully_distributed:
            from autoflowcfd.core.mpi.distributed_mesh_loader import (
                redistribute_fully_distributed_for_new_order,
            )
            redistribute_fully_distributed_for_new_order(self, target_p)
        else:
            cpu_traditional_interpolate_to_new_order(self, target_p)
