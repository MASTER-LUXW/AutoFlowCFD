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
            _warn_distributed_face_stencil,
            build_distributed_bounds_conn,
            build_sensor_gated_filter_func_arrays,
        )
        from autoflowcfd.core.fr_operators.bounds_sensor import (
            resolve_troubled_sensor,
        )

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
                self.freestream)

        order = int(getattr(self, "current_order", self.order))
        # 顶点邻域模板（BJ 判据用）在**分布式**路径上还不可用 ——
        # 如实记录为未完成项，不静默当成已修。
        #
        # 缺陷：面邻居在三维四面体上只有 4 个，其单元均值不能把本
        # 单元夹住，于是 O(h|grad u|) 的合法光滑变化被当成越界。
        # 实测标记比例在三档加密上**恒为 100%**（不收敛），换顶点
        # 邻域模板后 100% -> 25% -> 6.18%。完整数据见
        # `core/fr_operators/vertex_stencil.py` 模块文档。
        #
        # 为什么这里不接顶点模板：共享同一顶点的两个单元通过面邻接
        # 可能相隔 **2 个以上**面跳，而本项目的 halo 是 1 层面邻居。
        # 不完整的顶点邻域会让包络在分区边界上变窄，于是**同一算例
        # 换 rank 数得到不同的掩码** —— 结果依赖分区，这是求解器
        # 不可接受的（正是 `test_sensor_gate_distributed.py` 钉住的
        # 那条性质）。
        #
        # 为什么不硬失败：面模板在分布式上**是分区独立的**、现在就
        # 能用，只是带着上面那个过度标记的缺陷（与单机在本次修复前
        # 同一个缺陷）。把它改成崩溃等于删掉一个可用功能。
        #
        # 要在分布式上用顶点模板，需要的是一次**按顶点**的归约交换
        # （每个 rank 先算本地 node_max/node_min，再对共享顶点做
        # allreduce，然后散射回单元），那是一条与现有按单元的 halo
        # 交换不同的通信模式，属于独立一项。
        _warn_distributed_face_stencil(sensor)
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
