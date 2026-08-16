"""从体网格单元连接关系提取节点级邻接表。

目前唯一的调用方是 core/wall_distance.py 的 Eikonal（图最短路径近似）壁面
距离求解器——它需要"节点 i 的邻居是哪些节点"这张图，而不是单元连接关系
本身。这是纯拓扑操作（每个单元自己的边就是节点邻接关系），和求解器/壁面
距离这些概念无关，所以放在 grid 模块而不是 core 模块，供其它将来需要节点
图的场景（例如别的图拉普拉斯类平滑/传播算法）直接复用。
"""

import numpy as np

# 四面体的 6 条边（4 个顶点两两相连，四面体本身就是"任意两点都有一条边"
# 的单纯形，所以是全部 C(4,2)=6 对，不是子集）。
_TET_EDGE_LOCAL_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# 三棱柱的 9 条边：底面三角形 3 条 + 顶面三角形 3 条 + 3 条竖向边——不是
# C(6,2)=15 对全部组合，棱柱底面和顶面各自的对角线（例如 v0-w1）不是真实
# 网格边。节点顺序约定 (v0,v1,v2,w0,w1,w2)，与 quality_metrics.py 的
# prism_edge_lengths、mesh_prism_to_tet.py 的棱柱节点顺序约定完全一致。
_PRISM_EDGE_LOCAL_PAIRS = [
    (0, 1), (1, 2), (2, 0),  # 底面三角形
    (3, 4), (4, 5), (5, 3),  # 顶面三角形
    (0, 3), (1, 4), (2, 5),  # 竖向边
]


def build_node_adjacency(
    n_nodes: int,
    tet_connectivity: 'np.ndarray | None' = None,
    prism_connectivity: 'np.ndarray | None' = None,
) -> np.ndarray:
    """从四面体/棱柱连接关系构建节点邻接表。

    Args:
        n_nodes: 网格总节点数（邻接表的行数，独立于连接关系数组自己实际
            引用到的最大节点号 - 允许存在未被任何单元引用的孤立节点，其
            对应行全部为 -1）
        tet_connectivity: (n_tet, 4) 四面体节点连接关系，或 None
        prism_connectivity: (n_prism, 6) 棱柱节点连接关系（顺序
            v0,v1,v2,w0,w1,w2），或 None

    Returns:
        (n_nodes, max_degree) int64 数组，每行是该节点的邻居节点号列表，
        不足 max_degree 个邻居的位置补 -1。max_degree 由实际数据决定（网格
        里度数最高的节点决定这一列宽），不是任意猜的固定上限——猜小了会
        截断真实邻居（悄悄丢失图连通性，Eikonal 传播会算错但不报错），猜
        大了纯粹浪费内存，两者都不如直接用数据本身的真实值。
    """
    edge_blocks = []

    if tet_connectivity is not None and len(tet_connectivity) > 0:
        tet = np.asarray(tet_connectivity)
        for i, j in _TET_EDGE_LOCAL_PAIRS:
            edge_blocks.append(np.stack([tet[:, i], tet[:, j]], axis=1))

    if prism_connectivity is not None and len(prism_connectivity) > 0:
        prism = np.asarray(prism_connectivity)
        for i, j in _PRISM_EDGE_LOCAL_PAIRS:
            edge_blocks.append(np.stack([prism[:, i], prism[:, j]], axis=1))

    if not edge_blocks:
        return np.full((n_nodes, 0), -1, dtype=np.int64)

    edges = np.vstack(edge_blocks).astype(np.int64)
    # 邻接关系是双向的：每条边同时贡献 a->b 和 b->a 两条有向记录，再去重
    # （同一条无向边可能被多个共享它的单元各贡献一次，例如两个相邻四面体
    # 共享一条边）。
    directed_edges = np.vstack([edges, edges[:, ::-1]])
    directed_edges = np.unique(directed_edges, axis=0)

    order = np.argsort(directed_edges[:, 0], kind='stable')
    sorted_edges = directed_edges[order]

    unique_src, start_idx, counts = np.unique(
        sorted_edges[:, 0], return_index=True, return_counts=True
    )
    max_degree = int(counts.max())

    adjacency = np.full((n_nodes, max_degree), -1, dtype=np.int64)
    # 全向量化 scatter：group_id 标出每条已排序的边属于第几个 unique_src
    # 分组，pos_in_group 是它在自己分组内的偏移（0..count-1），两者结合
    # 就能一次性把 sorted_edges 的目标节点号写进 adjacency 对应的行、列，
    # 不需要逐节点 Python 循环。
    group_id = np.repeat(np.arange(len(unique_src)), counts)
    pos_in_group = np.arange(len(sorted_edges)) - np.repeat(start_idx, counts)
    adjacency[unique_src[group_id], pos_in_group] = sorted_edges[:, 1]

    return adjacency
