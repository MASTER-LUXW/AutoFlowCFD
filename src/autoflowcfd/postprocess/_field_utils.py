"""后处理模块共用的场插值辅助函数。

vtk_export.py（从单元中心的 FVM 解数据写出 POINT_DATA）和
transient_stats.py（从同一份单元中心数据累积节点分辨率的时间统计量）
都需要，两者要的是完全相同的单元->节点转换，所以放在这里共用一份，
而不是各写一份。
"""

import numpy as np


def cell_to_node(
    connectivity: np.ndarray,
    cell_values: np.ndarray,
    n_points: int,
    volumes: np.ndarray = None,
    fallback: float = 0.0,
) -> np.ndarray:
    """把逐单元的标量场插值成逐节点的值。

    对每个节点相连的单元做体积加权平均（向量化的 bincount scatter），
    而不是简单的不加权平均——按单元体积加权可以避免节点值被随便哪个
    相邻单元（恰好体积特别大或特别小）拉偏，这在边界层附近尤其重要，
    因为相邻四面体的体积可能相差好几个数量级。

    Args:
        connectivity: (n_cells, nodes_per_cell) 整数单元-节点数组
        cell_values: (n_cells,) 逐单元场值
        n_points: 网格节点数
        volumes: (n_cells,) 加权用的单元体积；None 表示不加权（等权重）平均
        fallback: 赋给没有相连单元的节点的值（对正常的体网格不应该发生，
            但万一发生了可以避免除零）

    Returns:
        (n_points,) 逐节点插值结果
    """
    conn = np.asarray(connectivity)
    n_cells, nodes_per_cell = conn.shape
    if volumes is None:
        weights = np.ones(n_cells, dtype=np.float64)
    else:
        weights = np.maximum(np.asarray(volumes, dtype=np.float64), 1e-30)

    node_ids = conn.ravel()
    cell_ids = np.repeat(np.arange(n_cells), nodes_per_cell)
    w = weights[cell_ids]

    weighted_sum = np.bincount(node_ids, weights=cell_values[cell_ids] * w, minlength=n_points)
    weight_sum = np.bincount(node_ids, weights=w, minlength=n_points)

    node_values = np.full(n_points, fallback, dtype=np.float64)
    has_data = weight_sum > 0
    node_values[has_data] = weighted_sum[has_data] / weight_sum[has_data]
    return node_values
