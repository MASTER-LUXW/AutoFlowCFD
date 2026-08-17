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
        """
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

        # 构建本 rank 的分区数据结构
        # 注意：build_distributed_partition 需要 face_connectivity 对象
        # 分布式模式下我们需要从 face_connectivity_data 重建
        if face_connectivity_data is not None:
            # 分布式模式：从数据构建局部面连接关系
            from autoflowcfd.grid.connectivity.face_connectivity import FRFaceConnectivity
            from dataclasses import dataclass

            @dataclass
            class LocalFaceConnectivity:
                """局部面连接关系（用于分区构建）。"""
                owner_cell: np.ndarray
                neighbor_cell: np.ndarray
                is_boundary: np.ndarray

            local_fc = LocalFaceConnectivity(
                owner_cell=np.array(face_connectivity_data['owner_cell']),
                neighbor_cell=np.array(face_connectivity_data['neighbor_cell']),
                is_boundary=np.array(face_connectivity_data['is_boundary']),
            )
            # 使用局部面连接关系构建分区
            # 注意：这里需要全局 cell 数来构建分区
            self.partition = build_distributed_partition(
                local_fc, cell_partition, self.rank, n_ranks
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

        # 7. 同步
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

        1. Halo 交换（获取邻居 cell 数据）
        2. 计算 local cells 的残差
        3. SSP-RK3 时间推进（只更新 local cells）
        4. 返回全局残差范数

        Args:
            dt: 时间步长

        Returns:
            residual_norm: 全局残差 L2 范数
        """
        from autoflowcfd.core.mpi.distributed_compute import (
            distributed_compute_inviscid_residual,
            distributed_compute_physical_gradient,
            distributed_compute_viscous_residual,
        )
        from autoflowcfd.core.fr_solver.step import ssp_rk3_step

        # 获取当前 local cells 的守恒变量
        U_local = self.state.get_local_U()

        # 1. 计算无粘残差（分布式）
        inviscid_residual = distributed_compute_inviscid_residual(
            U_local,
            self.partition,
            self.halo_exchange,
            self.dist_flat_face.connectivity,
            self.mesh,
            self.ops,
            self.local_solver.boundary_ghost_provider,
        )

        # 2. 计算梯度（分布式）
        grad_U = distributed_compute_physical_gradient(
            U_local,
            self.partition,
            self.halo_exchange,
            self.mesh,
            self.ops,
        )

        # 3. 计算粘性残差（分布式，如果启用了粘性）
        config = self.local_solver.config
        if config.physics.enable_viscous:
            viscous_residual = distributed_compute_viscous_residual(
                U_local,
                grad_U,
                self.partition,
                self.halo_exchange,
                self.dist_flat_face.connectivity,
                self.mesh,
                self.ops,
                config,
            )
            total_residual = inviscid_residual + viscous_residual
        else:
            total_residual = inviscid_residual

        # 4. 存储残差（扩展到 local + halo）
        self.state.dU_dt[:self.partition.n_local_cells] = total_residual

        # 5. SSP-RK3 时间推进（只更新 local cells）
        # 注意：这里简化为单步 Euler，完整 RK3 需要多次残差评估
        # 完整实现需要参考 fr_solver_step.py 的 ssp_rk3_step
        self.state.U[:self.partition.n_local_cells] += dt * total_residual

        # 6. 更新原变量
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        self.state.Q[:self.partition.n_local_cells] = conserved_to_primitive(
            self.state.U[:self.partition.n_local_cells]
        )

        # 7. 返回全局残差范数
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
