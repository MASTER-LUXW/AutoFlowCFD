"""AutoFlowCFD V2.0 - 面通量点跨引用所需的 halo 扩展

从 `src/autoflowcfd/core/mpi/partition.py`(原 612 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import numpy as np


from typing import Dict

from loguru import logger

from .types import DistributedPartition


def extend_halo_for_flux_point_cross_references(
    partition: DistributedPartition,
    cell_partition: np.ndarray,
    face_connectivity,
    extra_dep_arrays,
) -> DistributedPartition:
    """扩展 halo 层以覆盖 FR Flux Point 交叉插值（src0/src1）依赖的单元。

    第四次评审发现（真实网格验证）：`build_distributed_partition` 的
    halo 层只由 `face_connectivity` 的直接 owner/neighbor 1-ring 邻接
    决定，但棱柱四边形侧面的多源交叉插值（`fr/face_flux_points/merge.py`
    的 src0/src1 机制，用于重建跨单元 Flux Point 状态）会引用**不是**
    这个面的直接 neighbor 的其它单元——在 cube_demo 真实网格（37万单元，
    4-rank 简单 block 分区）上实测：16%~19% 的单元存在这类未被基础
    1-ring halo 覆盖的额外依赖。此前 `build_distributed_flat_face` 对
    这类引用只是重映射失败就报错（见该文件文档）——现在改为在报错之前
    先尝试这里的 halo 扩展，从根源修复而不是停在"检测到问题"。

    实现要点：每个 rank 独立执行同一套确定性计算（所有 rank 都拿到完整
    的 `cell_partition`/`face_connectivity`/`extra_dep_arrays`，不需要
    实际 MPI 通信就能算出一致的结果——与 `build_distributed_partition`
    本身"兼容旧接口：所有 rank 独立执行分区"的既有设计原则相同）：
    对每个面 f，"请求方 rank" = cell_partition[face 的 owner_cell]（该面
    的插值计算发生在这个 rank 上），若某个 extra_dep_arrays 条目在 f 处
    引用了单元 c 且 c 的 owner rank 与请求方不同，则：
    - 若本 rank 就是请求方：c 需要被加入本 rank 的 halo（若尚不是）；
    - 若本 rank 就是 c 的 owner：本 rank 需要把 c 发送给请求方 rank。

    Args:
        partition: build_distributed_partition 的输出（原地扩展并返回）
        cell_partition: (n_global_cells,) 每个单元的分区编号
        face_connectivity: 全局 FRFaceConnectivity
        extra_dep_arrays: List[np.ndarray]，每个形状 (n_faces,)，每个面
            额外依赖的单元全局索引（-1 表示无）。调用方负责把 src1 的
            紧凑数组展开成这种 per-face 形式（见
            `distributed_flat_face.py::_expand_compact_src1`）。

    Returns:
        原地扩展后的 partition（同一个对象，为了链式调用方便返回它）
    """
    rank = partition.rank
    owner_cell_g = face_connectivity.owner_cell
    owner_rank_of_face = cell_partition[owner_cell_g]  # (n_faces,)

    local_cell_set = set(int(c) for c in partition.local_cells)
    halo_set = set(int(c) for c in partition.halo_cells)

    new_halo_owner: Dict[int, int] = {}      # 新 halo cell -> owner rank
    new_send: Dict[int, set] = {}            # other_rank -> {本 rank 需要发送的 local cell}

    for dep_arr in extra_dep_arrays:
        valid = dep_arr >= 0
        if not np.any(valid):
            continue
        dep_cells = dep_arr[valid].astype(np.int64)
        requester_ranks = owner_rank_of_face[valid]
        owner_ranks = cell_partition[dep_cells]
        cross = requester_ranks != owner_ranks
        if not np.any(cross):
            continue
        dep_cells_c = dep_cells[cross]
        requester_c = requester_ranks[cross]
        owner_c = owner_ranks[cross]

        # 本 rank 是请求方：额外 halo 需求
        mask_req = requester_c == rank
        for c, o in zip(dep_cells_c[mask_req].tolist(), owner_c[mask_req].tolist()):
            if c not in local_cell_set and c not in halo_set and c not in new_halo_owner:
                new_halo_owner[c] = int(o)

        # 本 rank 是 owner：需要发送给请求方
        mask_own = owner_c == rank
        for c, r in zip(dep_cells_c[mask_own].tolist(), requester_c[mask_own].tolist()):
            if c in local_cell_set:
                new_send.setdefault(int(r), set()).add(c)

    if not new_halo_owner and not new_send:
        return partition  # 基础 1-ring halo 已完全覆盖，无需扩展

    n_old_halo = partition.n_halo
    new_halo_cells_arr = np.array(sorted(new_halo_owner.keys()), dtype=np.int64)
    new_halo_owners_arr = np.array(
        [new_halo_owner[int(c)] for c in new_halo_cells_arr], dtype=np.int32
    )
    n_new_halo = len(new_halo_cells_arr)

    # 关键正确性约束：halo.py::HaloExchange.exchange()/exchange_scalar() 和
    # gpu_halo_exchange.py 在填入接收到的 halo 数据时，都用
    # `np.searchsorted(partition.halo_cells, gc)` 定位某个全局单元 id 在
    # halo_cells 中的位置——np.searchsorted 要求数组*已排序*，否则是未定义
    # 行为（不会报错，只会静默返回错误的位置，导致部分/全部新增 halo 数据
    # 被写到错误位置或完全找不到匹配而被静默丢弃）。此前这里只是简单
    # `np.concatenate([旧halo_cells(已排序), 新halo_cells(各自已排序)])`，
    # 拼接后的整体数组不保证全局有序（新 halo cell 的全局 id 可能小于旧
    # halo_cells 里的部分元素）——第五次评审自查发现的真实 bug，必须在
    # 拼接后重新整体排序，并让 halo_owners 与 global_to_local 都同步这个
    # 新的排序位置（不只是给新增的 halo cell 赋值，旧 halo cell 的位置在
    # 重排后也可能变化）。
    combined_halo_cells = np.concatenate([partition.halo_cells, new_halo_cells_arr])
    combined_halo_owners = np.concatenate([partition.halo_owners, new_halo_owners_arr])
    sort_order = np.argsort(combined_halo_cells)
    partition.halo_cells = combined_halo_cells[sort_order]
    partition.halo_owners = combined_halo_owners[sort_order]
    partition.n_halo = n_old_halo + n_new_halo
    partition.halo_to_local_offset = np.arange(
        partition.n_local_cells, partition.n_local_cells + partition.n_halo, dtype=np.int64
    )
    # global_to_local 必须对*全部*（旧+新）halo cell 按重排后的最终位置
    # 重建，不能只更新新增的部分——排序会打乱旧 halo cell 原来的位置。
    for i, c in enumerate(partition.halo_cells):
        partition.global_to_local[int(c)] = partition.n_local_cells + i

    # recv_lists/send_lists 扩展：关键正确性约束（第五次评审自查发现，
    # 与上面 halo_cells 排序是同一类问题、同一个根因）——halo.py::
    # HaloExchange.exchange() 的通信协议里，发送方按 send_lists[r] 的
    # 顺序把数据打包进连续 buffer，接收方按自己 recv_lists[发送方] 的
    # 顺序原样解包，两端必须使用同一个顺序约定，数据本身不携带单元 id。
    # build_distributed_partition 的基础构造用"按全局 cell id 升序"作为
    # 这个约定（recv_lists 天然如此，因为是遍历已排序的 halo_cells 构建；
    # 本轮同步修复了 send_lists 让它也按全局 id 升序，见上方该函数内的
    # 注释）。这里的扩展逻辑必须遵守同一个约定，此前的实现里 recv_lists
    # 按 dict 插入顺序追加、send_lists 按局部索引数值排序，两者都不是
    # "按全局 id 升序"，会重演"顺序不一致导致 halo 数据被静默对应到
    # 错误单元"的缺陷。修复：合并后按全局 cell id 整体重新排序（不是
    # 只对新增部分排序后简单追加在旧数据后面）。
    new_recv_by_owner: Dict[int, list] = {}
    for c, o in new_halo_owner.items():
        new_recv_by_owner.setdefault(o, []).append(c)
    for o, new_cells in new_recv_by_owner.items():
        existing_recv = partition.recv_lists.get(o, np.empty(0, dtype=np.int64))
        combined_recv = np.concatenate([existing_recv, np.array(new_cells, dtype=np.int64)])
        partition.recv_lists[o] = np.sort(combined_recv)

    for r, cells in new_send.items():
        new_global_ids = np.array(sorted(cells), dtype=np.int64)
        existing_local = partition.send_lists.get(r, np.empty(0, dtype=np.int64))
        existing_global = (
            partition.local_to_global[existing_local] if len(existing_local) else np.empty(0, dtype=np.int64)
        )
        # np.unique 同时完成排序与去重（existing/new 理论上不应重叠，
        # 保留 unique 只是为了在意外重叠时不产生重复条目，而不是依赖
        # "两者必然不相交"这个假设本身）。
        combined_global = np.unique(np.concatenate([existing_global, new_global_ids]))
        partition.send_lists[r] = partition.global_to_local[combined_global].astype(np.int64)

    # neighbor_ranks 扩展
    neighbor_set = set(partition.neighbor_ranks)
    neighbor_set.update(int(o) for o in new_halo_owner.values())
    neighbor_set.update(int(r) for r in new_send.keys())
    partition.neighbor_ranks = sorted(neighbor_set)

    logger.info(
        f"Rank {rank}: extended halo for FR flux-point cross-references - "
        f"+{n_new_halo} halo cells (was {n_old_halo}), "
        f"+{sum(len(v) for v in new_send.values())} extra send entries"
    )

    return partition
