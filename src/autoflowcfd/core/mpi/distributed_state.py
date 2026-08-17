"""
AutoFlowCFD V2.0 - 分布式 FRState

将 FRState 扩展为分布式版本：每个 rank 持有 local cells 的状态数据，
加上 halo cells 的副本（通过 halo 交换获取）。

关键设计:
- 扩展数组: (n_local_cells + n_halo_cells, n_sps, n_vars)
- 体积项只在 local cells 上计算（halo cells 的残差被忽略）
- 界面项在 local cells 拥有的所有面上计算（包括 partition boundary 面）
- 残差范数需要 MPI Allreduce（全局 L2 范数）
"""

import numpy as np
from typing import Optional

from autoflowcfd.core.mpi.partition import DistributedPartition
from autoflowcfd.core.mpi.comm import allreduce_sum


class DistributedFRState:
    """分布式 FR 求解器状态。

    管理 local + halo 的守恒变量/原变量存储，提供全局范数计算。

    Attributes:
        partition: 分区信息
        U: (n_total_cells, n_sps, n_vars) 守恒变量（local + halo）
        Q: (n_total_cells, n_sps, n_vars) 原变量
        n_local_cells: 本 rank 的 local cell 数
        n_halo_cells: 本 rank 的 halo cell 数
        n_sps: 每单元解点数
        n_vars: 变量数
    """

    def __init__(self, partition: DistributedPartition, n_sps: int, n_vars: int):
        self.partition = partition
        self.n_local_cells = partition.n_local_cells
        self.n_halo_cells = partition.n_halo
        self.n_sps = n_sps
        self.n_vars = n_vars

        n_total = partition.n_total_cells
        self.U = np.zeros((n_total, n_sps, n_vars))
        self.Q = np.zeros((n_total, n_sps, n_vars))
        self.dU_dt = np.zeros((n_total, n_sps, n_vars))

    @property
    def n_cells(self) -> int:
        """local cell 数（对外接口保持一致性）。"""
        return self.n_local_cells

    @property
    def n_total(self) -> int:
        """local + halo 总数。"""
        return self.n_local_cells + self.n_halo_cells

    def set_local_data(self, U_global: np.ndarray, global_to_local: np.ndarray):
        """从全局数组中提取本 rank 的 local cell 数据。

        Args:
            U_global: (n_global_cells, n_sps, n_vars) 全局守恒变量
            global_to_local: (n_global_cells,) 全局→局部映射
        """
        for g in range(U_global.shape[0]):
            l = global_to_local[g]
            if l >= 0:
                self.U[l] = U_global[g]

    def get_local_U(self) -> np.ndarray:
        """返回 local cells 的守恒变量。"""
        return self.U[:self.n_local_cells].copy()

    def get_local_Q(self) -> np.ndarray:
        """返回 local cells 的原变量。"""
        return self.Q[:self.n_local_cells].copy()

    def local_residual_norm(self) -> float:
        """本 rank 的 local cells 残差 L2 范数。"""
        local_sum = np.sum(self.dU_dt[:self.n_local_cells] ** 2)
        return float(np.sqrt(local_sum))

    def global_residual_norm(self) -> float:
        """全局残差 L2 范数（MPI Allreduce）。"""
        local_sq_sum = np.sum(self.dU_dt[:self.n_local_cells] ** 2)
        global_sq_sum = allreduce_sum(local_sq_sum)
        return float(np.sqrt(global_sq_sum))

    def update_local_from_extended(self, U_extended: np.ndarray):
        """从扩展数组更新 local cells 的数据。

        Args:
            U_extended: (n_total_cells, n_sps, n_vars) 包含 halo 的扩展数组
        """
        self.U[:self.n_local_cells] = U_extended[:self.n_local_cells]
