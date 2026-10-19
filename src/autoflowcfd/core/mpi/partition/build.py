"""AutoFlowCFD V2.0 - 邻接图构造与分区。

全部按面的操作都是 numpy 向量化的（2026-10-19，I-96）：此前邻接图、halo 判定、面分类是对**全局**面数的 Python
循环，且每个 rank 各跑一遍，千万单元级网格上分区阶段本身成为瓶颈。结果与此前逐位相同（邻接表行内顺序保持按面
序号，METIS 输入不变）。
"""

from typing import Tuple

import numpy as np

from .types import DistributedPartition, FaceClassification

#: `partition_mesh` 的分区方法
PARTITION_METHODS = ("metis", "block")


def build_cell_adjacency_graph(owner_cell: np.ndarray, neighbor_cell: np.ndarray,
                                is_boundary: np.ndarray, n_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    """从面连接关系构建单元邻接图（CSR 格式）。

    两个单元通过内部面（非边界面）相连。用于 METIS 分区。每个单元的邻居按面序号排列（与此前逐面循环的
    填充顺序相同）。

    Args:
        owner_cell: (n_faces,)
        neighbor_cell: (n_faces,)，边界面为 -1
        is_boundary: (n_faces,) bool
        n_cells: 全局单元数

    Returns:
        (adj_indptr, adj_indices): CSR 格式的邻接表
            adj_indptr[i]:adj_indptr[i+1] 给出 cell i 的邻居在 adj_indices 中的范围
    """
    owner_cell = np.asarray(owner_cell, dtype=np.int64)
    neighbor_cell = np.asarray(neighbor_cell, dtype=np.int64)
    internal = (~np.asarray(is_boundary, dtype=bool)) & (neighbor_cell >= 0)
    oc, nc = owner_cell[internal], neighbor_cell[internal]
    # 每个面依次贡献 (oc -> nc)、(nc -> oc)；按起点稳定排序后，各行内即按面序号排列
    src = np.stack([oc, nc], axis=1).ravel()
    dst = np.stack([nc, oc], axis=1).ravel()
    order = np.argsort(src, kind="stable")
    adj_indptr = np.zeros(n_cells + 1, dtype=np.int64)
    adj_indptr[1:] = np.cumsum(np.bincount(src, minlength=n_cells))
    return adj_indptr, dst[order]


def partition_mesh(face_connectivity, n_parts: int, n_cells: int, method: str = "metis") -> np.ndarray:
    """将网格分成 n_parts 个分区。

    Args:
        face_connectivity: FRFaceConnectivity 实例
        n_parts: 分区数（= MPI rank 数）
        n_cells: 网格真实单元总数（`mesh.n_cells`）。2026-10-19（I-96）以前可省略、按 max(owner, neighbor)+1
            推断，而最后一个单元可能从不出现在面上（cube_demo 真实网格上少算 1 个、随后 IndexError）。
        method: "metis"（图分区，最小化切割面数并均衡负载，需要 pymetis）；"block"（按单元编号等分，只用于测试：
            棱柱在前 / 四面体在后的编号下分区不连通、halo 体量失控）。

    Raises:
        ImportError: method="metis" 且 n_parts > 1 时没有 pymetis。2026-10-19（I-96）以前只打一行 warning 就退回
            按单元编号切块。

    Returns:
        cell_partition: (n_cells,) int32, 每个 cell 所属的分区编号
    """
    if method not in PARTITION_METHODS:
        raise ValueError(f"未知分区方法 {method!r}（{PARTITION_METHODS}）")
    if n_parts == 1:
        return np.zeros(n_cells, dtype=np.int32)
    if method == "block":
        cell_partition = np.zeros(n_cells, dtype=np.int32)
        cells_per_part = n_cells // n_parts
        for p in range(n_parts):
            start = p * cells_per_part
            cell_partition[start:start + cells_per_part if p < n_parts - 1 else n_cells] = p
        return cell_partition
    try:
        import pymetis
    except ImportError as exc:
        raise ImportError(
            f"分成 {n_parts} 个分区需要 METIS 图分区（pip install pymetis）。按单元编号切块在棱柱在前 / 四面体在后的"
            f"编号下分区不连通、halo 交换量失控，不作为替代") from exc
    adj_indptr, adj_indices = build_cell_adjacency_graph(
        face_connectivity.owner_cell, face_connectivity.neighbor_cell, face_connectivity.is_boundary, n_cells)
    _, cell_partition = pymetis.part_graph(n_parts, xadj=adj_indptr, adjncy=adj_indices)
    return np.asarray(cell_partition, dtype=np.int32)


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

    # 1. 确定 local cells
    local_cells = np.flatnonzero(cell_partition == rank).astype(np.int64)
    n_local_cells = len(local_cells)

    # 全局→局部映射
    global_to_local = np.full(n_cells, -1, dtype=np.int64)
    global_to_local[local_cells] = np.arange(n_local_cells, dtype=np.int64)

    # 2. 确定 halo cells：经内部面与本 rank 单元相邻、属于其它 rank 的单元（全局 id 升序），所属 rank 即其分区号
    owner_all = np.asarray(face_connectivity.owner_cell, dtype=np.int64)
    neighbor_all = np.asarray(face_connectivity.neighbor_cell, dtype=np.int64)
    is_boundary_all = np.asarray(face_connectivity.is_boundary, dtype=bool)
    internal_mask = (~is_boundary_all) & (neighbor_all >= 0)
    oc_int = owner_all[internal_mask]
    nc_int = neighbor_all[internal_mask]
    oc_part = cell_partition[oc_int]
    nc_part = cell_partition[nc_int]
    own_here = (oc_part == rank)
    nbr_here = (nc_part == rank)
    halo_cells = np.unique(np.concatenate([nc_int[own_here & ~nbr_here], oc_int[nbr_here & ~own_here]]))
    n_halo = len(halo_cells)
    halo_owners = cell_partition[halo_cells].astype(np.int32)

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

    # 按在 halo_cells 中首次出现的顺序建表（与此前逐个追加相同），各表内为全局 id 升序
    _, first = np.unique(halo_owners, return_index=True)
    for owner_rank in halo_owners[np.sort(first)]:
        recv_lists[int(owner_rank)] = halo_cells[halo_owners == owner_rank]
        neighbor_ranks_set.add(int(owner_rank))

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
    # 本 rank 在 owner 侧 / neighbor 侧两个方向（own_here / nbr_here 见第 2 步；下面按对端 rank 分组）
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
    owner_local_mask = cell_partition[owner_all] == rank
    neighbor_local_mask = (neighbor_all >= 0) & (cell_partition[np.maximum(neighbor_all, 0)] == rank)
    local_faces = np.flatnonzero(owner_local_mask | neighbor_local_mask).astype(np.int64)

    # 5. 面分类。halo_owner：本 rank 只持有 neighbor 侧（owner 是另一个 rank 的 local cell）的面，不适用"owner 是
    # local"的另三类；其余按物理边界 > 本 rank 内部 > 分区边界分类。
    owner_local = owner_local_mask[local_faces]
    halo_owner_mask = ~owner_local
    physical_boundary_mask = owner_local & is_boundary_all[local_faces]
    interior_mask = owner_local & ~is_boundary_all[local_faces] & neighbor_local_mask[local_faces]
    partition_boundary_mask = owner_local & ~is_boundary_all[local_faces] & ~interior_mask

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
