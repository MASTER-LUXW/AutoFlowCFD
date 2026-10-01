"""AutoFlowCFD V2.0 - halo 交换、阶数切换、传感器门控滤波、负载均衡报告

从 `src/autoflowcfd/core/mpi/distributed_solver.py` 的 `DistributedFRSolver` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `DistributedFRSolver` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

import numpy as np
from autoflowcfd.core.mpi.comm import allreduce_sum


class _DistributedSupportMixin:
    """halo 交换、阶数切换、传感器门控滤波、负载均衡报告"""

    def _distributed_positivity_limiter(self):
        """正性保持限制器（与单机同一个、同一个核）：几何取 local 段、原生排列——
        adapter 的 jacobians 在紧凑排列，经 inv_perm 换回后切片。按阶数缓存。"""
        from autoflowcfd.core.mpi.distributed_compute import DistributedMeshAdapter
        from autoflowcfd.core.mpi.distributed_flat_face import native_cell_is_prism
        from autoflowcfd.core.time_integration.positivity import get_positivity_limiter

        dist_fc = self.dist_flat_face
        n_local = int(self.partition.n_local_cells)

        def _local_geometry():
            adapter = DistributedMeshAdapter(self.partition, dist_fc, self.mesh, self.ops)
            n_sps = self.state.U.shape[1]
            det_compact = np.asarray(adapter.jacobians["det_jacs"]).reshape(-1, n_sps)
            return det_compact[dist_fc.inv_perm][:n_local], native_cell_is_prism(dist_fc)[:n_local]

        return get_positivity_limiter(self, geometry=_local_geometry)

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
        """将解从当前阶数插值到新的阶数（分布式 Order Continuation
        核心逻辑，2026-09-02，见 core/mpi/distributed_order_
        continuation.py 模块文档）——按本实例的构造方式分派到对应的
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

    def exchange_halo(self, U_local: np.ndarray) -> np.ndarray:
        """执行 halo 交换。

        Args:
            U_local: (n_local_cells, n_sps, n_vars) local cell 数据

        Returns:
            U_extended: (n_total_cells, n_sps, n_vars) 含 halo 的扩展数据
        """
        return self.halo_exchange.exchange(U_local)

    def _build_sensor_gated_filter_func_distributed(
        self, n_local: int, n_sps: int, cell_is_prism: np.ndarray
    ):
        """CPU MPI 路径的**传感器门控**模态滤波回调（2026-09-18 接线）。

        `AFCFD_FILTER_MODE=sensor` 是 2026-09-17 定下的默认档，但当时只有
        单机 CPU 接线，本后端会退回 `project`（全局逐 stage 施加精确投影
        = 功能上等于 legacy，P1 退化成 P0、壁面剪应力恒为零）。这里补齐。

        缺的那一块是 BJ 越界判据要**面邻居的单元均值**，而分区边界上的
        邻居是 halo 单元。三点必须说明：

        1. **不能把分区边界面当边界面排除。** 分区边界不是物理边界，
           排除它等于在任意位置人为切断邻域包络，掩码会随 rank 数变化
           —— 同一个算例换分区数得到不同结果，这对求解器不可接受。
        2. **必须用当前 stage 的解做扩展，不能复用残差求值时缓存的
           `U_extended`。** 滤波在每个 stage 的正定性投影之后施加，那时
           解已经更新过；用缓存值会比单机路径滞后，两条后端就不再逐位
           可比。代价是每个 stage 多一次 halo 交换——与
           `_compute_distributed_local_time_step` 同一个既有权衡（局部 dt
           的谱半径同样需要额外一次交换），而滤波只在 `troubled` 非空时
           才真正动数据，交换本身是 `n_halo * n_sps * n_vars`。
        3. **索引空间**：`state.U` / 滤波回调的 `U_flat` 都是 halo 交换的
           **原生**排列（local 在前、halo 在后），而 `dist_flat_face` 的
           `owner_cell_local`/`neighbor_cell_local` 是"棱柱在前"的**紧凑**
           排列。两者用 `perm` 换算：紧凑下标 k 对应原生下标 `perm[k]`
           （因为 `array_native[perm] == array_permuted`），所以
           `owner_native = perm[owner_cell_local]`。掩码在原生扩展空间上
           算，取前 `n_local` 项；halo 行的邻域不完整、算出的掩码丢弃。

        判据内核与后端无关（`bounds_sensor.compute_bounds_violation_mask`
        是纯数组接口），这里不复制一份实现——只提供索引换算与 halo 扩展。
        """
        from autoflowcfd.core.fr_solver.filter import (
            build_distributed_bounds_conn,
            build_sensor_gated_filter_func_arrays,
        )
        from autoflowcfd.core.fr_operators.bounds_sensor import (
            resolve_troubled_sensor,
        )
        from autoflowcfd.core.mpi.vertex_stencil_mpi import local_vertex_pairs

        sensor = resolve_troubled_sensor()
        conn = {}
        if sensor in ("bounds", "both"):
            # 索引换算、两条自洽性护栏、halo 扩展时机、两张边界表的惰性
            # 构造全部在共享实现里（见 `build_distributed_bounds_conn`），
            # 多 GPU 走的是同一条。
            #
            # provider 取自 `self.local_solver`（惰性属性），它的
            # `group_code` 已经被重切到 `partition.local_faces`——与
            # `dist_flat_face` 同一索引空间，见 `local_solver` 文档里那处
            # "group_code 重映射"的说明。惰性（传 lambda）是因为构造
            # local_solver 本身有代价、且此刻未必已经建好。
            conn = build_distributed_bounds_conn(
                self.dist_flat_face,
                self.partition.n_total_cells,
                lambda: self.halo_exchange,
                lambda: getattr(self.local_solver,
                                "boundary_ghost_provider", None),
                self.freestream,
                local_vertex_pairs(self.mesh, self.partition))

        order = int(getattr(self, "current_order", self.order))
        return build_sensor_gated_filter_func_arrays(
            n_local, n_sps, order,
            self.ops.filter_prism, self.ops.filter_tet,
            cell_is_prism=cell_is_prism, sensor=sensor, **conn)

    def get_load_balance_report(self) -> str:
        """生成分区负载平衡报告。"""
        n_local = self.partition.n_local_cells
        n_halo = self.partition.n_halo
        n_total_global = self.partition.n_global_cells
        ideal = n_total_global / self.n_ranks
        imbalance = abs(n_local - ideal) / ideal * 100

        total_local = allreduce_sum(n_local)
        total_halo = allreduce_sum(n_halo)

        return (
            f"Load balance: rank {self.rank} has {n_local} cells "
            f"(ideal {ideal:.0f}, imbalance {imbalance:.1f}%), "
            f"{n_halo} halo cells. "
            f"Global total: {total_local} cells, {total_halo} halo."
        )
