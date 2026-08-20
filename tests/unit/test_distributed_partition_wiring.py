"""DistributedFRSolver 分区构建的单元测试。

核心回归判据：build_distributed_partition 的 halo 探测必须在**全局**
面连接关系上进行——用 distributed_mesh_load/extract_local_mesh_data
产出的**局部**（已按 rank 裁剪、跨 rank 邻居坍缩成 -1）面连接关系去
调用它，会让分区边界上的所有面被误判成"没有远端邻居"，halo 列表算成
全空，残差在分区边界上完全得不到邻居数据（V2.0 专家组评审逐行核实：
DistributedFRSolver 的 face_connectivity_data 路径此前从未被真正跑
通过——旧代码甚至连 build_distributed_partition 要求的 n_faces 属性
都没提供，直接 AttributeError）。

这些测试不需要真实多进程 MPI 环境：build_distributed_partition 是
纯函数，接受显式的 rank/n_ranks 参数，可以在单进程里模拟多个 rank
各自的视角直接验证。
"""

import numpy as np
import pytest

from autoflowcfd.core.mpi.partition import build_distributed_partition


class _FaceConnectivityView:
    """最小面连接关系视图，匹配 build_distributed_partition 的期望接口。"""

    def __init__(self, owner_cell, neighbor_cell, is_boundary):
        self.owner_cell = np.asarray(owner_cell, dtype=np.int64)
        self.neighbor_cell = np.asarray(neighbor_cell, dtype=np.int64)
        self.is_boundary = np.asarray(is_boundary, dtype=bool)

    @property
    def n_faces(self):
        return len(self.owner_cell)


def _synthetic_two_rank_mesh():
    """4 个单元的合成网格：0-1 同属 rank0，2-3 同属 rank1，1-2 是唯一
    跨分区的内部面（真正需要 halo 交换的地方）。

    面列表（全局）：
        f0: owner=0, neighbor=1, interior (both rank 0)
        f1: owner=1, neighbor=2, interior (rank 0 <-> rank 1, CROSSES)
        f2: owner=2, neighbor=3, interior (both rank 1)
        f3: owner=0, neighbor=-1, boundary
        f4: owner=3, neighbor=-1, boundary
    """
    owner_cell = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    neighbor_cell = np.array([1, 2, 3, -1, -1], dtype=np.int64)
    is_boundary = np.array([False, False, False, True, True], dtype=bool)
    cell_partition = np.array([0, 0, 1, 1], dtype=np.int32)
    return owner_cell, neighbor_cell, is_boundary, cell_partition


class TestBuildDistributedPartitionOnGlobalData:
    """修复后的正确用法：用全局面连接关系构建分区。"""

    def test_rank0_sees_cell2_as_halo(self):
        owner_cell, neighbor_cell, is_boundary, cell_partition = _synthetic_two_rank_mesh()
        global_fc = _FaceConnectivityView(owner_cell, neighbor_cell, is_boundary)

        partition = build_distributed_partition(global_fc, cell_partition, rank=0, n_ranks=2)

        assert partition.n_local_cells == 2
        assert set(partition.local_cells.tolist()) == {0, 1}
        assert partition.n_halo == 1, "跨分区面 f1(1<->2) 必须让 cell 2 成为 rank0 的 halo cell"
        assert partition.halo_cells.tolist() == [2]
        assert partition.halo_owners.tolist() == [1]
        assert 1 in partition.neighbor_ranks

    def test_rank1_sees_cell1_as_halo(self):
        owner_cell, neighbor_cell, is_boundary, cell_partition = _synthetic_two_rank_mesh()
        global_fc = _FaceConnectivityView(owner_cell, neighbor_cell, is_boundary)

        partition = build_distributed_partition(global_fc, cell_partition, rank=1, n_ranks=2)

        assert partition.n_local_cells == 2
        assert set(partition.local_cells.tolist()) == {2, 3}
        assert partition.n_halo == 1
        assert partition.halo_cells.tolist() == [1]
        assert partition.halo_owners.tolist() == [0]
        assert 0 in partition.neighbor_ranks


class TestLocallyClippedDataCorruptsHaloDetection:
    """回归判据：证明"用局部裁剪后的面连接关系代替全局面连接关系"
    这个此前实际发生过的错误用法，会让本应存在的 halo 关系凭空消失
    ——不是走这条路径就崩溃（旧代码是先在别处崩溃），而是即便勉强能
    跑，产出的分区在数值上就是错的（更隐蔽）。
    """

    def test_local_clipping_hides_the_cross_rank_neighbour(self):
        owner_cell, neighbor_cell, is_boundary, cell_partition = _synthetic_two_rank_mesh()

        # 模拟 extract_local_mesh_data 对 rank0 的裁剪：
        # 只保留 owner 属于 rank0 的面（f0, f1, f3），
        # neighbor 不在 local_cells={0,1} 里的一律坍缩成 -1。
        local_cells = np.array([0, 1])
        global_to_local = np.full(4, -1, dtype=np.int64)
        global_to_local[local_cells] = np.arange(2)

        owner_is_local = np.isin(owner_cell, local_cells)
        face_idx = np.where(owner_is_local)[0]
        clipped_owner = global_to_local[owner_cell[face_idx]]
        neighbor_global = neighbor_cell[face_idx]
        clipped_neighbor = np.where(
            neighbor_global >= 0, global_to_local[np.maximum(neighbor_global, 0)], -1
        )
        clipped_is_boundary = is_boundary[face_idx]

        # f1 (owner=1(local 1), neighbor=2(不在 local_cells 里)) 的
        # neighbor 被坍缩成 -1 —— 和真正的边界面 f3 用了同一个哨兵值，
        # 在这个"局部视角"里已经无法区分。
        f1_pos = int(np.where(face_idx == 1)[0][0])
        assert clipped_neighbor[f1_pos] == -1, "跨 rank 边被坍缩成 -1（复现根因）"
        assert clipped_is_boundary[f1_pos] == False, "但它本不是真正的物理边界面"

        clipped_fc = _FaceConnectivityView(clipped_owner, clipped_neighbor, clipped_is_boundary)
        # 用局部裁剪后的数据、局部 cell_partition（全 0，因为只看得到
        # 自己的两个 cell）调用 build_distributed_partition——
        # is_boundary 已经把 f1 也标记为边界，halo 探测的
        # `if is_boundary[f]: continue` 直接跳过它，halo 算成空。
        local_cell_partition = np.zeros(2, dtype=np.int32)
        broken_partition = build_distributed_partition(
            clipped_fc, local_cell_partition, rank=0, n_ranks=1
        )
        assert broken_partition.n_halo == 0, (
            "用局部裁剪数据构建分区时，本应存在的 halo 关系凭空消失——"
            "这正是修复前 DistributedFRSolver 会静默产出的错误状态"
        )
