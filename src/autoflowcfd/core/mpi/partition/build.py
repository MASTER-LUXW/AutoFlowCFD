"""AutoFlowCFD V2.0 - 邻接图构造与分区

从 `src/autoflowcfd/core/mpi/partition.py`(原 612 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import numpy as np


from typing import Optional

from loguru import logger

from .types import DistributedPartition, FaceClassification


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


def partition_mesh(face_connectivity, n_parts: int, n_cells: Optional[int] = None) -> np.ndarray:
    """将网格分成 n_parts 个分区。

    Args:
        face_connectivity: FRFaceConnectivity 实例
        n_parts: 分区数（= MPI rank 数）
        n_cells: 网格真实单元总数（强烈建议显式传入，通常是
            `mesh.n_cells`）。第四次评审发现真实崩溃 bug：省略时退回的
            `max(owner_cell)+1` 推断不可靠——owner/neighbor 角色分配依赖
            面提取时的排序，全局编号最后一个单元完全可能从未被选为任何
            面的 owner（只当 neighbor），此时 `max(owner_cell)` 会漏掉它，
            n_cells 被低估 1，后续 `degree[nc] += 1` 用真实存在的 neighbor
            单元索引访问这个偏小的数组直接 IndexError 崩溃（cube_demo
            真实网格上复现：374352 个单元，`max(owner_cell)+1` 算出
            374351）。仅在真的没有 mesh 对象可用时才退回这个不可靠的
            推断（保留向后兼容，但已知有此风险）。
    Returns:
        cell_partition: (n_cells,) int32, 每个 cell 所属的分区编号
    """
    if n_cells is None:
        n_cells = int(max(
            np.max(face_connectivity.owner_cell),
            np.max(face_connectivity.neighbor_cell),
        )) + 1
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

    # global_to_local 补齐 halo cell 的位置（第五次评审自查发现的一致性
    # 缺口）：此前这里只给 local_cells 赋过值，halo cell 的位置永远是
    # 初始化时的 -1 哨兵——`extend_halo_for_flux_point_cross_references`
    # 扩展 halo 层时会顺带把这个数组补全，但没有触发扩展的 rank（例如
    # 分区恰好不需要额外交叉引用依赖）永远不会补上这一步，导致同一个
    # partition 对象在"是否触发过扩展"上出现不一致的 global_to_local
    # 语义（历史上 distributed_flat_face.py 通过自建一份独立的扩展映射
    # 绕开了这个问题，没有直接依赖这个字段，因此这个缺口本身不影响当前
    # 唯一的真实消费点，但作为 partition 对象自身字段的语义完整性缺陷，
    # 留着会是未来新增消费者的地雷）。这里在基础构造阶段就直接补齐，
    # 与扩展逻辑保持一致，不依赖"后面会不会被扩展一次"这种偶然性。
    global_to_local[halo_cells] = n_local_cells + np.arange(n_halo, dtype=np.int64)

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
    # （rank r 需要这些 cell 的数据来填充它的 halo）。
    #
    # **这段计算是精确的，不是近似**（2026-09-14 更正）：此前这里的注释
    # 写"这需要通过全局通信确定——首版简化：通过面连接直接判断"，把一个
    # 精确计算错标成了简化。实际上 `rank r 的 halo` 完全由
    # `(face_connectivity, cell_partition)` 这两份**每个 rank 都已经持有
    # 的全局数据**唯一决定：一个 local cell 需要发给 rank r，当且仅当它
    # 通过某个内部面与一个属于 rank r 的 cell 相邻。因此不需要任何通信，
    # 也没有任何精度损失——绕开全局通信是这份数据可得性带来的结果，
    # 不是"用近似换简单"。
    #
    # 但原实现有两处**真实的性能缺陷**（2026-09-14 向量化重写）：
    #   1. 外层按 rank、内层按面的双层 Python 循环：O(n_ranks × n_faces)
    #      次解释器迭代。79 万单元 cube_demo 有 188 万个面，16 rank 下是
    #      3000 万次迭代，光这一步就要分钟级。
    #   2. `if local_idx not in send_cells` 对一个 **list** 做成员测试，
    #      整体是 O(n_send²)。一个 rank 的分区边界单元数上万时这一项
    #      就是数亿次比较。
    # 现在改成纯 numpy：一次性取出所有内部面的 (owner_part, neigh_part)，
    # 按两个方向分别筛出"本 rank 一侧 + 对端 rank 一侧"的单元，用
    # `np.unique` 同时完成去重与排序。
    #
    # 全局 id 升序这条约束必须保留（第五次评审自查发现的真实 bug）：
    # `halo.py::HaloExchange.exchange()` 的通信协议是"发送方按
    # send_lists[r] 的顺序把数据打包进一段连续 buffer，接收方按自己的
    # recv_lists[发送方 rank] 的顺序原样解包"——没有随数据传输任何单元
    # id，完全依赖两端按同一顺序约定摆放数据。recv_lists 是按已排序的
    # halo_cells 构建的、天然全局 id 升序；原实现的 send_cells 按"哪个面
    # 先发现这个单元"的遍历顺序收集，与全局 id 升序无关。用一个 3 rank、
    # 6x6 网格的合成算例实测复现过：发送方顺序 [22,23,18,21,19,20] vs
    # 接收方期望 [18,19,20,21,22,23]——集合相同但顺序不同，会把 halo
    # 数据静默对应到错误的单元上（不报错，只是物理结果错误）。
    # `np.unique` 返回的就是升序的**全局** id，正好满足这个约定，
    # 所以下面直接对全局 id 排序、再映射成局部索引。
    internal_mask = (~np.asarray(face_connectivity.is_boundary)) & (
        np.asarray(face_connectivity.neighbor_cell) >= 0)
    oc_int = np.asarray(face_connectivity.owner_cell)[internal_mask]
    nc_int = np.asarray(face_connectivity.neighbor_cell)[internal_mask]
    oc_part = cell_partition[oc_int]
    nc_part = cell_partition[nc_int]

    # 本 rank 在 owner 侧 / neighbor 侧两个方向（一个面可能两侧都不是本
    # rank，也不可能两侧都是本 rank 又跨 rank——下面按对端 rank 分组）
    own_here = (oc_part == rank)
    nbr_here = (nc_part == rank)

    for other_rank in range(n_ranks):
        if other_rank == rank:
            continue
        # 方向一：本 rank 的 owner cell，其 neighbor 属于 other_rank
        g1 = oc_int[own_here & (nc_part == other_rank)]
        # 方向二：本 rank 的 neighbor cell，其 owner 属于 other_rank
        g2 = nc_int[nbr_here & (oc_part == other_rank)]
        if g1.size == 0 and g2.size == 0:
            continue
        # np.unique：去重 + 按**全局 id 升序**排序（通信协议要求，见上）
        send_global = np.unique(np.concatenate([g1, g2]))
        send_lists[other_rank] = global_to_local[send_global].astype(np.int64)
        neighbor_ranks_set.add(other_rank)

    # 转换 recv_lists 为 numpy 数组
    for r in recv_lists:
        recv_lists[r] = np.array(recv_lists[r], dtype=np.int64)

    neighbor_ranks = sorted(neighbor_ranks_set)

    # 4. 确定本 rank 负责的面。
    #
    # 真实 bug 修复（2026-09-02，实现分布式湍流模型时用非均匀流场端到端
    # 测试发现——本项目此前所有分布式残差验证都用均匀自由流场，均匀场下
    # 任何面的真实解析跳跃恒为零，"漏掉一个本该算出来也是零的贡献"和
    # "正确算出这个贡献（结果也是零）"在数值上完全无法区分，这个 bug
    # 因此被完全掩盖，从未被现有测试捕捉到）：此前这里只选
    # `owner_cell` 是 local cell 的面（模块文档"面分类"一节其实早就
    # 写明了应该存在第四类"halo: neighbor 是 local，owner 在另一个
    # rank"，但从未真正被选进 `local_faces`——是文档与实现不一致，
    # 不是本次新发现的需求）。对一个跨 rank 分区边界的面，如果本 rank
    # 只持有 neighbor 侧的 local cell（owner 侧是另一个 rank 的 local
    # cell、对本 rank 而言是 halo），这个面完全不会出现在
    # `local_faces` 里——真实残差计算（inviscid_kernel.py/
    # viscous_flux_kernel.py/turbulence transport 的
    # owner-primary/neighbor-primary 两段式累加）因此从未算出这个面对
    # 本 rank 那个 local cell（neighbor 角色）的贡献，也没有任何其它
    # rank 会替它算（owner 所在的 rank 只关心 owner 侧的贡献，写不到
    # 属于另一个 rank 的 local cell 上）——等价于这类面对相关 local
    # cell 完全"消失"，真实合成网格端到端验证：非均匀流场下该 cell
    # 的无粘残差相对误差达 1239 倍（rel_diff=1.24e3，vs 单机路径），
    # 不是量级噪声。
    #
    # 修复：`local_faces` 改为"owner 或 neighbor 任一侧是 local cell"
    # 的并集——owner 侧仍按原逻辑分类（interior/partition_boundary/
    # physical_boundary）；新增的"仅 neighbor 是 local"这批面额外标记
    # 好，供下游 `owner_is_primary`/`neighbor_is_primary` 两段式累加
    # 分别处理（这批面的 owner 是 halo，只需要神经它们的
    # neighbor-primary 贡献，owner-primary 贡献留给拥有该 owner 的
    # 另一个 rank 自己算）。
    owner_local_mask = np.isin(face_connectivity.owner_cell, local_cells)
    neighbor_local_mask = (
        (face_connectivity.neighbor_cell >= 0)
        & np.isin(face_connectivity.neighbor_cell, local_cells)
    )
    local_faces = np.flatnonzero(owner_local_mask | neighbor_local_mask).astype(np.int64)

    # 5. 面分类
    interior_mask = np.zeros(len(local_faces), dtype=bool)
    partition_boundary_mask = np.zeros(len(local_faces), dtype=bool)
    physical_boundary_mask = np.zeros(len(local_faces), dtype=bool)
    # halo_owner_mask：本 rank 只持有 neighbor 侧（owner 是另一个 rank
    # 的 local cell）的这批面——模块文档"面分类"一节里一直存在、但此前
    # 从未真正生效的第四类。
    halo_owner_mask = np.zeros(len(local_faces), dtype=bool)

    for i, f in enumerate(local_faces):
        if not owner_local_mask[f]:
            # owner 不是本 rank 的 local cell——本 rank 只是因为
            # neighbor 是 local cell 才把这个面纳入（见上方修复），
            # 不适用原有的"owner 是 local"分类体系。
            halo_owner_mask[i] = True
            continue
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
        halo_owner_mask=halo_owner_mask,
        halo_owner_indices=np.flatnonzero(halo_owner_mask),
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
