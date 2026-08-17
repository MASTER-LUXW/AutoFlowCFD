"""
AutoFlowCFD V2.0 - 网格分区（METIS 接口）

将全局网格分割为若干子域，每个 MPI rank 负责一个子域。

分区算法:
    1. 从 FRFaceConnectivity 构建单元邻接图（cell → cell via shared face）
    2. 调用 METIS 的 part_graph 将图分成 n_parts 个分区
    3. 输出 DistributedPartition 数据结构

分区质量:
    METIS 的图分区算法最小化分区间的切割边数（= 跨 rank halo 交换量），
    同时保持各分区大小近似均衡（负载平衡）。对结构良好的非结构网格，
    切割边数通常为 O(n_faces^(2/3))，远优于随机分区。

面分类:
    分区后每个 rank 的面分为四类：
    - interior: owner 和 neighbor 都是 local cell
    - partition_boundary: owner 是 local，neighbor 在另一个 rank
    - physical_boundary: 原始物理边界面（is_boundary=True）
    - halo: neighbor 是 local，owner 在另一个 rank（用于接收校正）

    注意：物理边界面优先级高于 partition_boundary——如果一个面既是
    物理边界又是分区边界，它被归类为 physical_boundary（因为 neighbor=-1，
    不需要 halo 交换）。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from loguru import logger

from autoflowcfd.core.mpi import mpi_available


@dataclass
class FaceClassification:
    """面分类结果。

    Attributes:
        interior_mask: (n_local_faces,) bool, True = 内部面
        partition_boundary_mask: (n_local_faces,) bool, True = 分区边界面
        physical_boundary_mask: (n_local_faces,) bool, True = 物理边界面
        interior_indices: 内部面的局部索引
        partition_boundary_indices: 分区边界面的局部索引
        physical_boundary_indices: 物理边界面的局部索引
    """
    interior_mask: np.ndarray
    partition_boundary_mask: np.ndarray
    physical_boundary_mask: np.ndarray
    interior_indices: np.ndarray
    partition_boundary_indices: np.ndarray
    physical_boundary_indices: np.ndarray


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
        global_to_local: (n_global_cells,) 全局索引 → 局部索引（-1 = 非 local）
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


def build_cell_adjacency_graph(owner_cell: np.ndarray, neighbor_cell: np.ndarray,
                                is_boundary: np.ndarray, n_cells: int):
    """从面连接关系构建单元邻接图（CSR 格式）。

    两个单元通过内部面（非边界面）相连。用于 METIS 分区。

    Args:
        owner_cell: (n_faces,)
        neighbor_cell: (n_faces,)，边界面为 -1
        is_boundary: (n_faces,) bool
        n_cells: 全局单元数

    Returns:
        (adj_indptr, adj_indices): CSR 格式的邻接表
            adj_indptr[i]:adj_indptr[i+1] 给出 cell i 的邻居在 adj_indices 中的范围
    """
    # 统计每个 cell 的邻居数
    degree = np.zeros(n_cells, dtype=np.int64)
    for f in range(len(owner_cell)):
        if is_boundary[f]:
            continue
        oc = owner_cell[f]
        nc = neighbor_cell[f]
        if nc >= 0:
            degree[oc] += 1
            degree[nc] += 1

    # CSR 构建
    adj_indptr = np.zeros(n_cells + 1, dtype=np.int64)
    adj_indptr[1:] = np.cumsum(degree)
    adj_indices = np.empty(adj_indptr[-1], dtype=np.int64)
    pos = adj_indptr[:-1].copy()

    for f in range(len(owner_cell)):
        if is_boundary[f]:
            continue
        oc = owner_cell[f]
        nc = neighbor_cell[f]
        if nc >= 0:
            adj_indices[pos[oc]] = nc
            pos[oc] += 1
            adj_indices[pos[nc]] = oc
            pos[nc] += 1

    return adj_indptr, adj_indices


def partition_mesh(face_connectivity, n_parts: int) -> np.ndarray:
    """将网格分成 n_parts 个分区。

    Args:
        face_connectivity: FRFaceConnectivity 实例
        n_parts: 分区数（= MPI rank 数）

    Returns:
        cell_partition: (n_cells,) int32, 每个 cell 所属的分区编号
    """
    n_cells = int(np.max(face_connectivity.owner_cell)) + 1
    adj_indptr, adj_indices = build_cell_adjacency_graph(
        face_connectivity.owner_cell,
        face_connectivity.neighbor_cell,
        face_connectivity.is_boundary,
        n_cells,
    )

    try:
        import pymetis
        # pymetis 使用邻接列表格式
        adjacency = []
        for i in range(n_cells):
            adjacency.append(adj_indices[adj_indptr[i]:adj_indptr[i+1]].tolist())
        _, cell_partition = pymetis.part_graph(n_parts, adjacency=adjacency)
        cell_partition = np.array(cell_partition, dtype=np.int32)
    except ImportError:
        logger.warning("pymetis not available, using simple block partitioning")
        cell_partition = np.zeros(n_cells, dtype=np.int32)
        cells_per_part = n_cells // n_parts
        for p in range(n_parts):
            start = p * cells_per_part
            end = start + cells_per_part if p < n_parts - 1 else n_cells
            cell_partition[start:end] = p

    return cell_partition


def build_distributed_partition(
    face_connectivity,
    cell_partition: np.ndarray,
    rank: int,
    n_ranks: int,
) -> DistributedPartition:
    """为指定 rank 构建分区数据结构。

    Args:
        face_connectivity: FRFaceConnectivity
        cell_partition: (n_cells,) 每个 cell 的分区编号
        rank: 当前 rank
        n_ranks: 总 rank 数

    Returns:
        DistributedPartition 实例
    """
    n_cells = len(cell_partition)
    n_faces = face_connectivity.n_faces

    # 1. 确定 local cells
    local_cells = np.flatnonzero(cell_partition == rank).astype(np.int64)
    n_local_cells = len(local_cells)

    # 全局→局部映射
    global_to_local = np.full(n_cells, -1, dtype=np.int64)
    global_to_local[local_cells] = np.arange(n_local_cells, dtype=np.int64)

    # 2. 确定 halo cells（通过分区边界面找到邻居 rank 的 cells）
    halo_set = set()
    halo_owner_map = {}  # global_cell → owner_rank

    for f in range(n_faces):
        if face_connectivity.is_boundary[f]:
            continue
        oc = face_connectivity.owner_cell[f]
        nc = face_connectivity.neighbor_cell[f]
        if nc < 0:
            continue
        oc_part = cell_partition[oc]
        nc_part = cell_partition[nc]

        # owner 是 local，neighbor 不是 → neighbor 是 halo
        if oc_part == rank and nc_part != rank:
            halo_set.add(int(nc))
            halo_owner_map[int(nc)] = int(nc_part)
        # neighbor 是 local，owner 不是 → owner 是 halo
        if nc_part == rank and oc_part != rank:
            halo_set.add(int(oc))
            halo_owner_map[int(oc)] = int(oc_part)

    halo_cells = np.array(sorted(halo_set), dtype=np.int64)
    n_halo = len(halo_cells)
    halo_owners = np.array([halo_owner_map[int(c)] for c in halo_cells], dtype=np.int32)

    # halo cell 在扩展数组中的偏移（local cells 在前，halo 在后）
    halo_to_local_offset = np.arange(n_local_cells, n_local_cells + n_halo, dtype=np.int64)

    # 3. 构建 send/recv lists
    send_lists = {}
    recv_lists = {}
    neighbor_ranks_set = set()

    for i, hc in enumerate(halo_cells):
        owner_rank = int(halo_owners[i])
        neighbor_ranks_set.add(owner_rank)
        if owner_rank not in recv_lists:
            recv_lists[owner_rank] = []
        recv_lists[owner_rank].append(int(hc))

    # send_lists[r] = 本 rank 需要发给 rank r 的 local cell 局部索引
    # （rank r 需要这些 cell 的数据来填充它的 halo）
    for other_rank in range(n_ranks):
        if other_rank == rank:
            continue
        # 哪些 local cell 是 other_rank 的 halo？
        # 即：other_rank 的 recv_lists[rank] 中的 cell
        # 这需要通过全局通信确定——首版简化：通过面连接直接判断
        send_cells = []
        for f in range(n_faces):
            if face_connectivity.is_boundary[f]:
                continue
            oc = face_connectivity.owner_cell[f]
            nc = face_connectivity.neighbor_cell[f]
            if nc < 0:
                continue
            oc_part = cell_partition[oc]
            nc_part = cell_partition[nc]
            # local cell 的 neighbor 在 other_rank → 需要发送给 other_rank
            if oc_part == rank and nc_part == other_rank:
                local_idx = int(global_to_local[oc])
                if local_idx not in send_cells:
                    send_cells.append(local_idx)
            # local cell 的 owner 在 other_rank → 需要发送给 other_rank
            if nc_part == rank and oc_part == other_rank:
                local_idx = int(global_to_local[nc])
                if local_idx not in send_cells:
                    send_cells.append(local_idx)
        if send_cells:
            send_lists[other_rank] = np.array(send_cells, dtype=np.int64)
            neighbor_ranks_set.add(other_rank)

    # 转换 recv_lists 为 numpy 数组
    for r in recv_lists:
        recv_lists[r] = np.array(recv_lists[r], dtype=np.int64)

    neighbor_ranks = sorted(neighbor_ranks_set)

    # 4. 确定本 rank 负责的面（owner 是 local cell 的面）
    local_faces = np.flatnonzero(
        np.isin(face_connectivity.owner_cell, local_cells)
    ).astype(np.int64)

    # 5. 面分类
    interior_mask = np.zeros(len(local_faces), dtype=bool)
    partition_boundary_mask = np.zeros(len(local_faces), dtype=bool)
    physical_boundary_mask = np.zeros(len(local_faces), dtype=bool)

    for i, f in enumerate(local_faces):
        if face_connectivity.is_boundary[f]:
            physical_boundary_mask[i] = True
        else:
            nc = face_connectivity.neighbor_cell[f]
            if nc >= 0 and cell_partition[nc] == rank:
                interior_mask[i] = True
            else:
                partition_boundary_mask[i] = True

    face_classification = FaceClassification(
        interior_mask=interior_mask,
        partition_boundary_mask=partition_boundary_mask,
        physical_boundary_mask=physical_boundary_mask,
        interior_indices=np.flatnonzero(interior_mask),
        partition_boundary_indices=np.flatnonzero(partition_boundary_mask),
        physical_boundary_indices=np.flatnonzero(physical_boundary_mask),
    )

    return DistributedPartition(
        rank=rank,
        n_ranks=n_ranks,
        n_global_cells=n_cells,
        local_cells=local_cells,
        n_local_cells=n_local_cells,
        local_to_global=local_cells,
        global_to_local=global_to_local,
        halo_cells=halo_cells,
        n_halo=n_halo,
        halo_owners=halo_owners,
        halo_to_local_offset=halo_to_local_offset,
        send_lists=send_lists,
        recv_lists=recv_lists,
        neighbor_ranks=neighbor_ranks,
        face_classification=face_classification,
        local_faces=local_faces,
    )
