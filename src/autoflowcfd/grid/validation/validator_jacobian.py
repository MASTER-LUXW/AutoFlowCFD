"""GridValidator 的雅可比行列式 / 三角形绕向一致性检查。

从 validator.py 中拆分出来（原文件超过 400 行的项目约定上限）：
GridValidator._check_jacobian 及其私有辅助方法 _count_flipped_triangles
是一个自成一体的算法阶段——检测相邻三角形绕向是否互相矛盾（曲面网格
翻转单元的判据），除了读取 self.grid_data 之外不与类的其它检查共享任何
状态，因此按"取 grid_data 为参数的模块级函数"整体搬移，validator.py 里
只保留转调用的薄包装方法，逻辑/数值结果完全不变。
"""

from typing import Dict

import numpy as np
from loguru import logger

from ..structures import GridData


def check_jacobian(grid_data: GridData) -> Dict[str, float]:
    """检查雅可比行列式

    计算每个单元的雅可比行列式。雅可比衡量
    坐标变换的局部缩放因子。

    Returns:
        Dict: 包含 'max'、'avg'、'min' 键的统计字典

    注意:
        正雅可比表示有效的单元方向。
        负或接近零的雅可比表示反转或退化的单元。
    """
    logger.debug("Computing Jacobian determinants...")

    connectivity = grid_data.cells.connectivity
    nodes = grid_data.nodes

    # 获取节点坐标
    cell_coords = np.stack([
        nodes.get_coordinates(connectivity[:, i])
        for i in range(3)
    ], axis=1)

    # 计算从节点 0 到节点 1 和 2 的向量
    v0 = cell_coords[:, 1] - cell_coords[:, 0]
    v1 = cell_coords[:, 2] - cell_coords[:, 0]

    # 计算叉积（得到法向量）
    cross = np.cross(v0, v1)

    # 雅可比行列式是叉积的模
    jacobians = np.linalg.norm(cross, axis=1)

    # 单个三角形的绕向在没有外部参考系的情况下没有绝对符号，
    # 因此这里的 "jacobians < 0"（np.linalg.norm 永远不为负）
    # 对任何网格都不会触发——无论输入如何，反转三角形对此检查
    # 都是静默不可见的。在没有外部参考的情况下有明确定义的是
    # 相邻三角形是否彼此一致：在方向一致的流形表面上，
    # 共享一条边的两个三角形必须以相反方向遍历该共享边。
    # 如果它们以相同方向遍历，则其中一个相对于邻居翻转——
    # 通过有向边符号累加而非模的符号来检测。
    negative_count = _count_flipped_triangles(grid_data, connectivity)

    # 计算统计量
    stats = {
        'max': float(np.max(jacobians)),
        'avg': float(np.mean(jacobians)),
        'min': float(np.min(jacobians)),
        'std': float(np.std(jacobians)),
        'negative_count': negative_count
    }

    logger.debug(
        f"Jacobian - Max: {stats['max']:.6f}, "
        f"Avg: {stats['avg']:.6f}, Min: {stats['min']:.6f}, "
        f"Negative: {stats['negative_count']}"
    )

    return stats


def _count_flipped_triangles(grid_data: GridData, connectivity: np.ndarray) -> int:
    """计数绕向与邻居不一致的三角形。

    对于每条共享（流形内部）边——恰好与两个三角形相邻的边——
    方向一致的表面必须从每条边以相反方向遍历它。
    不以相同方向遍历其共享边的两个三角形不可能都方向正确；
    两者都被标记。被非 2 个三角形共享的边（开放边界或非流形）
    无法以此方式检查方向，此处跳过——这是一个不同的网格完整性
    问题，不属于此度量的关注范围。
    """
    n1, n2, n3 = connectivity[:, 0], connectivity[:, 1], connectivity[:, 2]
    directed_edges = np.concatenate([
        np.stack([n1, n2], axis=1),
        np.stack([n2, n3], axis=1),
        np.stack([n3, n1], axis=1),
    ], axis=0)
    owner_cells = np.tile(np.arange(len(connectivity)), 3)

    n_nodes = grid_data.nodes.count
    lo = np.minimum(directed_edges[:, 0], directed_edges[:, 1]).astype(np.int64)
    hi = np.maximum(directed_edges[:, 0], directed_edges[:, 1]).astype(np.int64)
    edge_key = lo * n_nodes + hi
    sign = np.where(directed_edges[:, 0] < directed_edges[:, 1], 1, -1)

    order = np.argsort(edge_key, kind='stable')
    sorted_keys = edge_key[order]
    sorted_signs = sign[order]
    sorted_owners = owner_cells[order]

    group_end = np.flatnonzero(
        np.concatenate([sorted_keys[1:] != sorted_keys[:-1], [True]])
    )
    group_start = np.concatenate([[0], group_end[:-1] + 1])
    group_size = group_end - group_start + 1

    pair_mask = group_size == 2
    pair_first = group_start[pair_mask]
    same_direction = sorted_signs[pair_first] == sorted_signs[pair_first + 1]
    bad_first = pair_first[same_direction]

    flipped_cells = np.unique(np.concatenate([
        sorted_owners[bad_first], sorted_owners[bad_first + 1]
    ])) if bad_first.size else np.array([], dtype=owner_cells.dtype)

    return int(flipped_cells.size)
