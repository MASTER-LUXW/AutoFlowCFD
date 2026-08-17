"""
AutoFlowCFD V2.0 - 面图着色（消除 scatter-add 写冲突）

界面 kernel 的 `correction[owner_cell[f], s, v] += ...` 是 scatter-add：
同一个 cell 被多个面共享，不同 f 会写同一个 `correction[cell]`。当前方案
（fr_residual_inviscid_kernel.py）用 per-thread buffer 避免冲突，内存
O(n_threads * n_cells * n_sps * n_vars)，是扩展性的根本瓶颈。

面图着色提供替代方案：将面分成若干颜色组，同色面之间无 owner_cell 冲突
（不存在两个同色面共享同一个 owner cell）。每色内可安全直接 prange +
写入共享 buffer，无需 per-thread buffer。内存从 O(n_threads * N) 降至
O(N)（一个共享 buffer），按色循环处理。

算法：贪心着色（按面序遍历，每个面取最小的不与已着色邻居冲突的颜色）。
着色数取决于网格拓扑，典型 4-8 色（四面体网格）到 10-15 色（混合网格）。

性能权衡：
- 优势：内存降 n_threads 倍，消除归约开销，NUMA 友好
- 代价：按色循环处理，每色只处理 ~n_faces/n_colors 个面，
  每色内可 prange 并行；总工作量不变但循环次数增加 n_colors 倍
- 适用场景：大网格（内存受限）或高线程数（per-thread buffer 爆炸）

与 per-thread buffer 方案的选择：
- 小网格 + 低线程数（<=4）：per-thread buffer 更快（无循环开销）
- 大网格 + 高线程数（>8）：图着色更优（内存可控，无归约）
- 通过 benchmark 选择最优策略
"""

import numpy as np
from typing import Dict, List


def build_cell_face_ownership(owner_cell: np.ndarray, n_cells: int) -> Dict[int, List[int]]:
    """构建 cell → 以其为 owner 的面索引列表。

    Args:
        owner_cell: (n_faces,) 每个面的 owner cell 索引
        n_cells: 总单元数

    Returns:
        cell_to_faces: dict[cell_id] = [face_id, ...]
    """
    cell_to_faces: Dict[int, List[int]] = {}
    for f in range(len(owner_cell)):
        oc = int(owner_cell[f])
        if oc not in cell_to_faces:
            cell_to_faces[oc] = []
        cell_to_faces[oc].append(f)
    return cell_to_faces


def build_face_conflict_graph(owner_cell: np.ndarray, n_cells: int) -> List[List[int]]:
    """构建面冲突邻接表。

    两个面冲突当且仅当它们共享同一个 owner_cell。对每个 cell，以其为
    owner 的所有面两两冲突（完全子图）。

    注意：不显式构建完整邻接矩阵（O(n_faces^2) 内存），只构建邻接表。
    对贪心着色来说，只需要知道每个面的邻居（冲突面）集合。

    Args:
        owner_cell: (n_faces,)
        n_cells: 总单元数

    Returns:
        adj: list of lists, adj[f] = [face_id, ...] 与面 f 冲突的面列表
    """
    n_faces = len(owner_cell)
    cell_to_faces = build_cell_face_ownership(owner_cell, n_cells)

    # 对每个 cell，其所有 face 两两冲突
    adj = [[] for _ in range(n_faces)]
    for cell_id, face_list in cell_to_faces.items():
        n_f = len(face_list)
        if n_f <= 1:
            continue
        for i in range(n_f):
            for j in range(i + 1, n_f):
                fi, fj = face_list[i], face_list[j]
                adj[fi].append(fj)
                adj[fj].append(fi)
    return adj


def greedy_face_coloring(owner_cell: np.ndarray, n_cells: int) -> np.ndarray:
    """贪心面图着色。

    按面序（range(n_faces)）遍历，每个面取最小的不与任何已着色冲突邻居
    相同颜色。保持与原始面序一致（不重排），满足退化 Jacobian 敏感性约束。

    Args:
        owner_cell: (n_faces,)
        n_cells: 总单元数

    Returns:
        colors: (n_faces,) int32, 每个面的颜色编号（0, 1, ..., n_colors-1）
    """
    n_faces = len(owner_cell)
    cell_to_faces = build_cell_face_ownership(owner_cell, n_cells)

    colors = np.full(n_faces, -1, dtype=np.int32)

    # 按 cell 分组处理：对每个 cell 的所有 face，它们互相冲突，
    # 必须着不同颜色。贪心策略：按面序遍历，取最小可用颜色。
    # 但面之间还有跨 cell 的隐式冲突（通过共享 neighbor cell 的
    # scatter-add 间接冲突）。当前简化版本只考虑 owner_cell 冲突。
    for cell_id, face_list in cell_to_faces.items():
        for f in face_list:
            # 收集已着色冲突邻居的颜色
            neighbor_colors = set()
            oc = int(owner_cell[f])
            # 冲突邻居 = 同一个 owner cell 的其他面
            for f2 in cell_to_faces[oc]:
                if f2 != f and colors[f2] >= 0:
                    neighbor_colors.add(int(colors[f2]))
            # 取最小可用颜色
            c = 0
            while c in neighbor_colors:
                c += 1
            colors[f] = c

    return colors


def get_color_masks(colors: np.ndarray, n_colors: int) -> list:
    """将颜色数组转为布尔 mask 列表。

    Args:
        colors: (n_faces,) int32
        n_colors: 颜色总数

    Returns:
        masks: list of (n_faces,) bool arrays, masks[c] = (colors == c)
    """
    return [colors == c for c in range(n_colors)]


def color_faces(flat) -> tuple:
    """对 FlatFaceGeometry 的面进行图着色。

    便捷入口：从 FlatFaceGeometry 提取 owner_cell，执行贪心着色，
    返回颜色数组和 mask 列表。

    Args:
        flat: FlatFaceGeometry

    Returns:
        (colors, color_masks, n_colors)
    """
    n_cells_est = int(np.max(flat.owner_cell)) + 1
    colors = greedy_face_coloring(flat.owner_cell, n_cells_est)
    n_colors = int(np.max(colors)) + 1
    color_masks = get_color_masks(colors, n_colors)
    return colors, color_masks, n_colors
