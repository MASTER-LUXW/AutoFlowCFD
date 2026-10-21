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

from autoflowcfd.core.fr_solver.state import N_MEAN_FLOW_VARS
import numpy as np

from autoflowcfd.core.mpi.partition import DistributedPartition


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
        n_vars: 变量数（平均流守恒变量，`fr_solver/state.py::N_MEAN_FLOW_VARS`）
    """

    def __init__(self, partition: DistributedPartition, n_sps: int):
        self.partition = partition
        self.n_local_cells = partition.n_local_cells
        self.n_halo_cells = partition.n_halo
        self.n_sps = n_sps
        self.n_vars = n_vars = N_MEAN_FLOW_VARS

        n_total = partition.n_total_cells
        self.U = np.zeros((n_total, n_sps, n_vars))
        self.Q = np.zeros((n_total, n_sps, n_vars))
        self.dU_dt = np.zeros((n_total, n_sps, n_vars))

    def initialize_uniform(self, rho: float = 1.0, u: float = 0.0, v: float = 0.0,
                            w: float = 0.0, p: float = 1.0) -> None:
        """用均匀自由流场初始化 local+halo 全部单元（构造后必须调用：`__init__` 只分配零数组，rho=0 会让第一次
        原始变量换算产生 NaN）。公式与单机 `FRState.initialize_uniform` 同一份（`uniform_conservative`）；湍流场
        由湍流模型独立持有，不在 `self.U`/`self.Q` 里（`n_vars` 恒为 5）。
        """
        from autoflowcfd.core.fr_solver.state import uniform_conservative

        self.U[:, :, :5] = uniform_conservative(rho, u, v, w, p)
        self.Q[:, :, 0] = rho
        self.Q[:, :, 1] = u
        self.Q[:, :, 2] = v
        self.Q[:, :, 3] = w
        self.Q[:, :, 4] = p

    @property
    def n_cells(self) -> int:
        """local cell 数（对外接口保持一致性）。"""
        return self.n_local_cells

    @property
    def n_total(self) -> int:
        """local + halo 总数。"""
        return self.n_local_cells + self.n_halo_cells

    def get_local_U(self) -> np.ndarray:
        """返回 local cells 的守恒变量。"""
        return self.U[:self.n_local_cells].copy()

    def get_local_Q(self) -> np.ndarray:
        """返回 local cells 的原变量。"""
        return self.Q[:self.n_local_cells].copy()
