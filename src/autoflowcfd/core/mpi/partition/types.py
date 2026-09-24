"""AutoFlowCFD V2.0 - 分区结果的数据结构

从 `src/autoflowcfd/core/mpi/partition.py`(原 612 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import numpy as np

from dataclasses import dataclass

from typing import Dict


@dataclass
class FaceClassification:
    """面分类结果。

    Attributes:
        interior_mask: (n_local_faces,) bool, True = 内部面
        partition_boundary_mask: (n_local_faces,) bool, True = 分区边界面
        physical_boundary_mask: (n_local_faces,) bool, True = 物理边界面
        halo_owner_mask: (n_local_faces,) bool, True = 本 rank 只持有
            neighbor 侧（owner 是另一个 rank 的 local cell）的面——模块
            文档"面分类"一节的第四类"halo"，2026-09-02 修复前从未真正
            被选进 `local_faces`（见 build_distributed_partition 同名
            注释的真实 bug 记录），现在真正生效。
        interior_indices: 内部面的局部索引
        partition_boundary_indices: 分区边界面的局部索引
        physical_boundary_indices: 物理边界面的局部索引
        halo_owner_indices: halo_owner_mask 为 True 的局部索引
    """
    interior_mask: np.ndarray
    partition_boundary_mask: np.ndarray
    physical_boundary_mask: np.ndarray
    interior_indices: np.ndarray
    partition_boundary_indices: np.ndarray
    physical_boundary_indices: np.ndarray
    halo_owner_mask: np.ndarray = None
    halo_owner_indices: np.ndarray = None


@dataclass
class DistributedPartition:
    """单个 rank 的分区信息。

    Attributes:
        rank: 当前 rank 编号
        n_ranks: 总 rank 数
        n_global_cells: 全局单元总数
        local_cells: (n_local_cells,) 本 rank 拥有的 cell 全局索引
        n_local_cells: 本 rank 的 local cell 数
        local_to_global: (n_local_cells,) 局部索引 → 全局索引（= local_cells）
        global_to_local: (n_global_cells,) 全局索引 → local+halo 扩展索引
            空间（[0,n_local) 为 local cell，[n_local,n_local+n_halo) 为
            halo cell；-1 = 既非 local 也非 halo，真正不可达）
        halo_cells: (n_halo,) halo 层 cell 的全局索引
        n_halo: halo cell 数
        halo_owners: (n_halo,) 每个 halo cell 来自哪个 rank
        halo_to_local_offset: halo cell 在扩展数组中的局部偏移
            （local cells 在前 [0, n_local_cells)，halo cells 在后）
        send_lists: dict[rank] → 发送给该 rank 的 local cell 局部索引
        recv_lists: dict[rank] → 从该 rank 接收的 halo cell 全局索引
        neighbor_ranks: 相邻 rank 列表（有 halo 交换关系的 rank）
        face_classification: 面分类
        local_faces: (n_local_faces,) 本 rank 负责的面在全局面数组中的索引
    """
    rank: int
    n_ranks: int
    n_global_cells: int
    local_cells: np.ndarray
    n_local_cells: int
    local_to_global: np.ndarray
    global_to_local: np.ndarray
    halo_cells: np.ndarray
    n_halo: int
    halo_owners: np.ndarray
    halo_to_local_offset: np.ndarray
    send_lists: Dict[int, np.ndarray]
    recv_lists: Dict[int, np.ndarray]
    neighbor_ranks: list
    face_classification: FaceClassification
    local_faces: np.ndarray

    @property
    def n_total_cells(self) -> int:
        """local + halo 的总 cell 数（扩展数组的大小）。"""
        return self.n_local_cells + self.n_halo
