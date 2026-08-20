"""
AutoFlowCFD V2.0 - 分布式 FRSolver

将 FRSolver 扩展为 MPI 域分解版本。每个 rank 持有 local cells 的数据，
通过 halo 交换获取邻居信息，独立计算 local cells 的残差。

设计:
- 组合模式：DistributedFRSolver 持有 FRSolver 实例 + 分区/通信基础设施
- 覆盖残差计算方法：先 halo 交换，再调用单机残差（只处理 local cells）
- 全局操作（残差范数、时间步长）通过 MPI Allreduce

使用:
    # 所有 rank 加载同一网格文件
    mesh = load_mesh(grid_file)
    solver = DistributedFRSolver(mesh, ...)
    # 每个 rank 自动获取自己的分区，执行分布式求解
    for step in range(n_steps):
        solver.step(dt)
"""

import numpy as np
from typing import Optional
from loguru import logger

from autoflowcfd.core.mpi import get_rank, get_size, is_root, mpi_available
from autoflowcfd.core.mpi.partition import (
    partition_mesh, build_distributed_partition, DistributedPartition
)
from autoflowcfd.core.mpi.halo import HaloExchange
from autoflowcfd.core.mpi.distributed_state import DistributedFRState
from autoflowcfd.core.mpi.distributed_flat_face import (
    DistributedFlatFaceGeometry, build_distributed_flat_face
)
from autoflowcfd.core.mpi.comm import allreduce_sum, allreduce_min, barrier
from autoflowcfd.core.time_integration.base import TimeIntegrator, TimeIntegrationScheme


class DistributedFRSolver:
    """MPI 域分解分布式 FR 求解器。

    组合 FRSolver（单机残差计算）+ MPI 基础设施（分区、halo 交换、
    全局归约）。每个 rank 独立持有一个 FRSolver 实例（只处理 local cells），
    通过 halo 交换获取邻居信息。

    Attributes:
        partition: 本 rank 的分区信息
        halo_exchange: halo 交换管理器
        state: 分布式状态
        rank: 当前 rank
        n_ranks: 总 rank 数
    """

    def __init__(
        self,
        mesh,
        ops,
        n_ranks: int,
        face_connectivity=None,
        face_connectivity_data=None,
        partition_info=None,
        rank: Optional[int] = None,
        **solver_kwargs,
    ):
        """初始化分布式求解器。

        Args:
            mesh: HighOrderMesh（分布式模式下为局部网格）
            ops: FROperators
            n_ranks: MPI rank 总数
            face_connectivity: FRFaceConnectivity（单机模式，已废弃）
            face_connectivity_data: dict（分布式模式，局部面连接关系数据）
            partition_info: dict（分布式模式，分区信息）
            rank: 当前 rank 编号（默认从 MPI 获取）
            **solver_kwargs: 传递给 FRSolver 的参数

        Raises:
            NotImplementedError: 请求了 'none' 以外的湍流模型时。分布式
                残差/状态目前只接了纯层流（5-var，无 mu_t 耦合）路径——
                湍流模型的 k/omega 输运方程需要 turb_model 实例接入分布式
                状态与逐 RK 子步的 halo 交换，尚未实现（见
                distributed_compute.py::distributed_turbulence_transport
                文档）。静默忽略请求的湍流模型、跑出一个看似正常实际上
                物理不完整的结果，比直接报错更糟——所以这里在构造时就
                拒绝，而不是等到求解中途才发现。单机模式
                （不带 --n-ranks/--np）已完整支持 SST/DDES/WMLES/LES。
        """
        turb_model_name = solver_kwargs.get('turb_model_name', 'none')
        if turb_model_name is not None and str(turb_model_name).upper() != 'NONE':
            raise NotImplementedError(
                f"MPI 分布式求解器（--n-ranks/--np > 1，或 --multi-gpu）目前只支持 "
                f"turbulence_model='none'，收到的是 '{turb_model_name}'。分布式湍流"
                f"输运（k/omega 对流+扩散、wall_distance 分发、SGS/壁面应力模型）"
                f"尚未接入分布式状态与残差计算，详见 "
                f"core/mpi/distributed_compute.py::distributed_turbulence_transport "
                f"文档。请去掉 --turbulence-model（默认 none）或改用单机模式。"
            )

        self.rank = rank if rank is not None else get_rank()
        self.n_ranks = n_ranks
        self.mesh = mesh
        self.ops = ops

        # 分布式模式：使用传入的分区信息
        if partition_info is not None:
            cell_partition = partition_info['cell_partition']
            n_global_cells = partition_info['n_global_cells']
        elif face_connectivity is not None:
            # 兼容旧接口：所有 rank 独立执行分区
            if self.rank == 0:
                logger.info(f"Partitioning mesh into {n_ranks} parts (METIS on root)...")
                cell_partition = partition_mesh(face_connectivity, n_ranks)
            else:
                cell_partition = None
            if n_ranks > 1:
                from autoflowcfd.core.mpi.comm import bcast_from_root
                cell_partition = bcast_from_root(cell_partition)
            n_global_cells = face_connectivity.owner_cell.max() + 1
        else:
            raise ValueError("Either face_connectivity or partition_info must be provided")

        # 构建本 rank 的分区数据结构。
        #
        # build_distributed_partition 的 halo 探测（哪些面跨越分区边界、
        # 因此需要 halo 交换）必须在**全局、未裁剪**的面连接关系上做——
        # 它要用 cell_partition[owner]/cell_partition[neighbor] 判断一条
        # 面两侧是否属于不同 rank，cell_partition 本身是全局数组
        # （下标是全局 cell id）。此前这里传入的是
        # face_connectivity_data（distributed_mesh_load 提取的**局部**
        # 数据，owner_cell/neighbor_cell 已经重映射成本 rank 的局部索引，
        # 跨 rank 的 neighbor 被强制置为 -1，与真正的边界面用同一个
        # 哨兵值、在本 rank 视角下已经无法区分），拿它当 face_connectivity
        # 传给 build_distributed_partition 有两个独立问题：(1) 该
        # 函数第一行就要读 face_connectivity.n_faces，一个用普通
        # dataclass 拼出来的 LocalFaceConnectivity 对象没有这个属性，
        # 必然 AttributeError；(2) 即便补上这个属性，用局部索引去查
        # cell_partition（全局索引空间）也是错的，且跨 rank 邻居已经
        # 提前坍缩成 -1，halo 探测的 `if nc < 0: continue` 会直接跳过
        # 所有真正的分区边界面——halo_cells/send_lists/recv_lists 会
        # 算成空的，粘性/无粘残差在分区边界上完全得不到邻居数据
        # （V2.0 专家组评审逐行核实：这条路径此前从未被真正跑通过）。
        #
        # 修复：用 distributed_mesh_load 随 partition_info 一起广播的
        # **全局**面连接关系（global_owner_cell/global_neighbor_cell/
        # global_is_boundary）构建分区——这才是 build_distributed_
        # partition 设计时假设的输入形态，cell_partition 也是同一个
        # 全局索引空间，两者能正确对齐。
        if face_connectivity_data is not None:
            if partition_info is None or 'global_owner_cell' not in partition_info:
                raise ValueError(
                    "DistributedFRSolver(face_connectivity_data=...) 需要 "
                    "partition_info 里包含 global_owner_cell/global_neighbor_cell/"
                    "global_is_boundary（distributed_mesh_load 的输出）才能正确"
                    "构建分区——不能只用局部（已按 rank 裁剪）的面连接关系，"
                    "见本方法上方注释。"
                )
            from dataclasses import dataclass

            @dataclass
            class GlobalFaceConnectivityView:
                """构建分区专用的最小全局面连接关系视图（只读，不重新
                实例化完整 FRFaceConnectivity，避免要求调用方提供它
                不需要的其余几何字段）。"""
                owner_cell: np.ndarray
                neighbor_cell: np.ndarray
                is_boundary: np.ndarray

                @property
                def n_faces(self) -> int:
                    return len(self.owner_cell)

            global_fc = GlobalFaceConnectivityView(
                owner_cell=np.asarray(partition_info['global_owner_cell']),
                neighbor_cell=np.asarray(partition_info['global_neighbor_cell']),
                is_boundary=np.asarray(partition_info['global_is_boundary']),
            )
            self.partition = build_distributed_partition(
                global_fc, cell_partition, self.rank, n_ranks
            )
        else:
            # 兼容旧接口
            self.partition = build_distributed_partition(
                face_connectivity, cell_partition, self.rank, n_ranks
            )

        if is_root():
            logger.info(
                f"Rank {self.rank}: {self.partition.n_local_cells} local cells, "
                f"{self.partition.n_halo} halo cells, "
                f"{len(self.partition.neighbor_ranks)} neighbors"
            )

        # 3. 初始化分布式状态
        n_sps = mesh.n_sps_per_cell
        n_vars = solver_kwargs.get('n_vars', 5)
        self.state = DistributedFRState(self.partition, n_sps, n_vars)

        # 4. 初始化 halo 交换器
        self.halo_exchange = HaloExchange(self.partition, n_sps, n_vars)

        # 5. 构建分布式面几何
        self.dist_flat_face = build_distributed_flat_face(mesh, ops, self.partition)

        # 6. 保存 solver kwargs 用于创建本地求解器
        self.solver_kwargs = solver_kwargs
        self._local_solver = None  # 延迟初始化

        # 7. 时间推进器：与单机 FRSolver 同一套 Shu-Osher SSP-RK3 stage
        # 实现（core/time_integration/base.py），保证分布式与单机路径
        # 时间精度一致（见 step() 文档：此前这里是自称"RK3"实际执行单步
        # 前向欧拉的简化实现）。dt 在这里只是占位——真正推进用的步长由
        # step() 每次显式构造的 dt_local 数组决定，不读 self.dt。
        self._time_integrator = TimeIntegrator(scheme=TimeIntegrationScheme.SSP_RK3, dt=1.0)

        # 8. 同步
        barrier()

    @property
    def local_solver(self):
        """延迟初始化本地求解器（避免循环依赖）。"""
        if self._local_solver is None:
            from autoflowcfd.core.fr_solver.solver import FRSolver
            self._local_solver = FRSolver(
                self.mesh, self.ops, **self.solver_kwargs
            )
        return self._local_solver

    def exchange_halo(self, U_local: np.ndarray) -> np.ndarray:
        """执行 halo 交换。

        Args:
            U_local: (n_local_cells, n_sps, n_vars) local cell 数据

        Returns:
            U_extended: (n_total_cells, n_sps, n_vars) 含 halo 的扩展数据
        """
        return self.halo_exchange.exchange(U_local)

    def compute_global_residual_norm(self) -> float:
        """全局残差 L2 范数。"""
        return self.state.global_residual_norm()

    def compute_global_min_dt(self, dt_local: np.ndarray) -> float:
        """全局最小时间步长（MPI Allreduce MIN）。"""
        local_min = float(np.min(dt_local[:self.partition.n_local_cells]))
        return allreduce_min(local_min)

    def get_load_balance_report(self) -> str:
        """生成分区负载平衡报告。"""
        n_local = self.partition.n_local_cells
        n_halo = self.partition.n_halo
        n_total_global = self.partition.n_global_cells
        ideal = n_total_global / self.n_ranks
        imbalance = abs(n_local - ideal) / ideal * 100

        from autoflowcfd.core.mpi.comm import allreduce_sum
        total_local = allreduce_sum(n_local)
        total_halo = allreduce_sum(n_halo)

        return (
            f"Load balance: rank {self.rank} has {n_local} cells "
            f"(ideal {ideal:.0f}, imbalance {imbalance:.1f}%), "
            f"{n_halo} halo cells. "
            f"Global total: {total_local} cells, {total_halo} halo."
        )

    def step(self, dt: float) -> float:
        """执行一步时间推进（分布式版本）。

        真正的 3-stage Shu-Osher SSP-RK3（与单机 FRSolver 同一套实现，
        `TimeIntegrator._ssp_rk_stage_step`，见 core/fr_solver/step.py::
        step 的 mean_flow_residual，这里的 `residual_func` 是它的分布式
        版本，两套约定必须严格一致）：每个 stage 都要用该 stage 的中间解
        重新做一次完整的 halo 交换 + 残差求值——不能像旧版本那样只算
        一次残差就套用 RK3 的名字（旧实现自己的注释承认"简化为单步
        Euler"，与 SSP-RK3 的时间精度/稳定域完全不是一回事）。

        两个容易踩错、已用独立数值脚本验证过的约定，都严格照抄
        `fr_solver/step.py::step`：
        1. `_ssp_rk_stage_step` 期望的 `solution`/`dt_local` 是展平成
           `(n_local_cells*n_sps, n_vars)` / `(n_local_cells*n_sps,)` 的
           2D/1D 数组（`dt = dt_local[:, None]` 只能对 2D solution 广播），
           不是 `(n_local_cells, n_sps, n_vars)` 的 3D 数组——直接传 3D
           会在第一个 stage 就因广播形状不匹配抛 ValueError。
        2. `compute_*_residual_fr` 返回的是 dU/dt 本身，`residual_func`
           必须返回其**负值**（`TimeIntegrator` 的约定是 dU/dt=-R(U)）；
           旧的 Euler 实现 `U += dt*total_residual` 直接用未取负的和，
           这一点在那个实现里恰好是自洽的（因为它没有经过 R→L=-R 这层
           转换），但复用 `_ssp_rk_stage_step` 就必须显式取负，否则解会
           往错误的时间方向积分。

        目前只支持纯层流（无 mu_t 耦合）——`__init__` 已经在构造时拒绝
        非 'none' 的湍流模型，这里不需要再判断。

        Args:
            dt: 物理时间步长（分布式路径目前用全局固定步长，不做单机
                路径那种逐 cell 局部 CFL 时间步——旧实现本来就是全局
                dt，这里不新增自适应步长这个单独的功能点）

        Returns:
            residual_norm: 全局残差 L2 范数（RK3 第 0 阶段的 dU/dt 范数，
                与旧实现的报告口径一致，用于跨迭代收敛监控）
        """
        from autoflowcfd.core.mpi.distributed_compute import (
            distributed_compute_inviscid_residual,
            distributed_compute_viscous_residual,
        )
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

        config = self.local_solver.config
        enable_viscous = config.physics.enable_viscous
        mu = self.local_solver.mu_molecular
        boundary_ghost_provider = self.local_solver.boundary_ghost_provider
        n_local = self.partition.n_local_cells
        n_sps = self.state.n_sps
        n_vars = self.state.n_vars

        def residual_func(U_flat_trial: np.ndarray) -> np.ndarray:
            """TimeIntegrator 约定：dU/dt = -residual_func(U)。RK3 每个
            stage 都会调用一次：对该 stage 的中间解重新做 halo 交换 +
            残差求值（halo 数据在每个 stage 之间会变化，不能复用上一个
            stage 交换到的邻居数据）。"""
            U_stage_local = U_flat_trial.reshape(n_local, n_sps, n_vars)
            inviscid_residual = distributed_compute_inviscid_residual(
                U_stage_local, self.partition, self.halo_exchange,
                self.dist_flat_face.connectivity, self.mesh, self.ops,
                boundary_ghost_provider,
            )
            if enable_viscous:
                viscous_residual = distributed_compute_viscous_residual(
                    U_stage_local, self.partition, self.halo_exchange,
                    self.dist_flat_face.connectivity, self.mesh, self.ops,
                    mu, boundary_ghost_provider,
                )
                total_dudt = inviscid_residual + viscous_residual
            else:
                total_dudt = inviscid_residual
            return -total_dudt.reshape(n_local * n_sps, n_vars)

        U_flat = self.state.get_local_U().reshape(n_local * n_sps, n_vars)
        dt_local_flat = np.full(n_local * n_sps, dt)

        # Stage 0 残差单独算一次：既用于收敛监控（与旧实现报告口径一致），
        # 也通过 residual0= 传给 _ssp_rk_stage_step 复用，避免它内部再重复
        # 算一次同样的 R(U^n)。
        residual0 = residual_func(U_flat)
        self.state.dU_dt[:n_local] = (-residual0).reshape(n_local, n_sps, n_vars)

        U_new_flat = self._time_integrator._ssp_rk_stage_step(
            U_flat, residual_func, dt_local_flat, p_floor=1.0, residual0=residual0,
        )

        U_new_local = U_new_flat.reshape(n_local, n_sps, n_vars)
        self.state.U[:n_local] = U_new_local
        self.state.Q[:n_local] = conserved_to_primitive(U_new_local[..., :5])

        return self.compute_global_residual_norm()

    def solve(self, n_steps: int, dt: float, output_interval: int = 100):
        """运行分布式求解循环。

        Args:
            n_steps: 最大时间步数
            dt: 时间步长
            output_interval: 输出间隔
        """
        from autoflowcfd.core.mpi.comm import barrier

        if is_root():
            logger.info(f"Starting distributed solve: {n_steps} steps, dt={dt}")

        for step_idx in range(n_steps):
            # 执行一步
            residual_norm = self.step(dt)

            # 输出进度
            if step_idx % output_interval == 0 and is_root():
                logger.info(
                    f"Step {step_idx}/{n_steps}, "
                    f"residual_norm={residual_norm:.6e}"
                )

            # 同步（可选，用于调试）
            # barrier()

        if is_root():
            logger.info("Distributed solve completed.")
