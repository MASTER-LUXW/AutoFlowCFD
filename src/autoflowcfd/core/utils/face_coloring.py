"""
AutoFlowCFD V2.0 - 面图着色（消除 scatter-add 写冲突）

界面 kernel 的 `correction[owner_cell[f], s, v] += ...` 是 scatter-add：
同一个 cell 被多个面共享，不同 f 会写同一个 `correction[cell]`。当前方案
（fr_residual_inviscid_kernel.py）用 per-thread buffer 避免冲突，内存
O(n_threads * n_cells * n_sps * n_vars)，是扩展性的根本瓶颈。

面图着色提供替代方案：将面分成若干颜色组，同色面之间无写冲突单元
（不存在两个同色面会 scatter-add 到同一个单元）。每色内可安全直接
prange + 写入共享 buffer，无需 per-thread buffer。内存从 O(n_threads * N)
降至 O(N)（一个共享 buffer），按色循环处理。

冲突判定必须覆盖 owner 侧和 neighbor 侧两次写入：图着色版本 kernel
（inviscid_kernel_colored.py / viscous_flux_kernel.py 的 _colored 变体）
对每个内部面都会执行两次独立的 scatter-add——`correction[owner_cell[f]] +=`
（owner_is_primary 分支）和 `correction[neighbor_cell[f]] +=`
（neighbor_is_primary 分支）。若只按 owner_cell 分组着色，两个面 A、B
只要满足 "A 的 owner 恰好是 B 的 neighbor"（或 A/B 的 neighbor 相同）就
可能被分到同一颜色，而该颜色组在 prange 并行循环里会对同一个单元做
非原子并发写入，产生静默丢失更新的竞争条件。因此正确的冲突图必须以
"面触及的单元集合"（owner_cell 且，非边界面时还有 neighbor_cell）为
单位构建，而不能只用 owner_cell。

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


def build_cell_face_ownership(
    owner_cell: np.ndarray,
    n_cells: int,
    neighbor_cell: np.ndarray = None,
    is_boundary: np.ndarray = None,
) -> Dict[int, List[int]]:
    """构建 cell → 触及该 cell（会向其 scatter-add 写入）的面索引列表。

    图着色 kernel 对内部面会做两次独立写入：owner 侧
    `correction[owner_cell[f]] +=` 与 neighbor 侧
    `correction[neighbor_cell[f]] +=`。因此一个面"触及"的单元集合，
    非边界面时是 {owner_cell[f], neighbor_cell[f]}，边界面时只有
    {owner_cell[f]}（边界面没有 neighbor 侧写入）。

    Args:
        owner_cell: (n_faces,) 每个面的 owner cell 索引
        n_cells: 总单元数
        neighbor_cell: (n_faces,) 每个面的 neighbor cell 索引（边界面处为 -1
            或占位值，由 is_boundary 判定是否有效）；为 None 时视为无 neighbor
            侧写入（等价于旧的仅 owner_cell 行为，仅供向后兼容/单元测试用）
        is_boundary: (n_faces,) bool，True 表示该面是边界面（无 neighbor 侧写入）

    Returns:
        cell_to_faces: dict[cell_id] = [face_id, ...]（同一 face 若同时触及
        owner 与 neighbor 两侧，会在两个 cell 的列表中各出现一次）
    """
    cell_to_faces: Dict[int, List[int]] = {}
    n_faces = len(owner_cell)
    for f in range(n_faces):
        oc = int(owner_cell[f])
        cell_to_faces.setdefault(oc, []).append(f)
        if neighbor_cell is not None:
            boundary = bool(is_boundary[f]) if is_boundary is not None else False
            if not boundary:
                nc = int(neighbor_cell[f])
                if nc >= 0 and nc != oc:
                    cell_to_faces.setdefault(nc, []).append(f)
    return cell_to_faces


def build_face_conflict_graph(
    owner_cell: np.ndarray,
    n_cells: int,
    neighbor_cell: np.ndarray = None,
    is_boundary: np.ndarray = None,
) -> List[List[int]]:
    """构建面冲突邻接表。

    两个面冲突当且仅当它们触及同一个单元（见 `build_cell_face_ownership`
    的写入语义：owner 侧 + 非边界面的 neighbor 侧）。对每个 cell，触及它
    的所有面两两冲突（完全子图）。

    注意：不显式构建完整邻接矩阵（O(n_faces^2) 内存），只构建邻接表。
    对贪心着色来说，只需要知道每个面的邻居（冲突面）集合。

    Args:
        owner_cell: (n_faces,)
        n_cells: 总单元数
        neighbor_cell: (n_faces,)，见 `build_cell_face_ownership`
        is_boundary: (n_faces,) bool，见 `build_cell_face_ownership`

    Returns:
        adj: list of lists, adj[f] = [face_id, ...] 与面 f 冲突的面列表
    """
    n_faces = len(owner_cell)
    cell_to_faces = build_cell_face_ownership(owner_cell, n_cells, neighbor_cell, is_boundary)

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


def greedy_face_coloring(
    owner_cell: np.ndarray,
    n_cells: int,
    neighbor_cell: np.ndarray = None,
    is_boundary: np.ndarray = None,
) -> np.ndarray:
    """贪心面图着色。

    按面序（range(n_faces)）遍历，每个面取最小的不与任何已着色冲突邻居
    相同颜色。保持与原始面序一致（不重排），满足退化 Jacobian 敏感性约束。

    冲突判定覆盖 owner 侧与非边界面的 neighbor 侧两次 scatter-add 写入
    （见 `build_cell_face_ownership` docstring）。传 `neighbor_cell`/
    `is_boundary` 是必须的——图着色版本 kernel 对内部面在 owner_is_primary
    与 neighbor_is_primary 分支下分别向 owner_cell 与 neighbor_cell 做非
    原子并发写入，只按 owner_cell 分组会漏掉 "A 的 owner 是 B 的 neighbor"
    这类冲突，导致同色并行循环里出现静默丢失更新的竞争条件。

    Args:
        owner_cell: (n_faces,)
        n_cells: 总单元数
        neighbor_cell: (n_faces,)，为 None 时退化为仅按 owner_cell 冲突
            （不安全，仅供向后兼容的单元测试使用，生产调用必须传入）
        is_boundary: (n_faces,) bool

    Returns:
        colors: (n_faces,) int32, 每个面的颜色编号（0, 1, ..., n_colors-1）
    """
    n_faces = len(owner_cell)
    cell_to_faces = build_cell_face_ownership(owner_cell, n_cells, neighbor_cell, is_boundary)

    # 每个面触及 1~2 个 cell（owner，非边界面时还有 neighbor）；
    # 面 f 的冲突面集合 = 其触及的所有 cell 各自面列表的并集。
    face_to_cells: Dict[int, List[int]] = {}
    for cell_id, face_list in cell_to_faces.items():
        for f in face_list:
            face_to_cells.setdefault(f, []).append(cell_id)

    colors = np.full(n_faces, -1, dtype=np.int32)

    for f in range(n_faces):
        neighbor_colors = set()
        for cid in face_to_cells.get(f, []):
            for f2 in cell_to_faces[cid]:
                if f2 != f and colors[f2] >= 0:
                    neighbor_colors.add(int(colors[f2]))
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

    便捷入口：从 FlatFaceGeometry 提取 owner_cell/neighbor_cell/is_boundary，
    执行贪心着色（owner 侧 + 非边界面 neighbor 侧冲突均覆盖），返回颜色
    数组和 mask 列表。

    Args:
        flat: FlatFaceGeometry

    Returns:
        (colors, color_masks, n_colors)
    """
    n_cells_est = int(np.max(flat.owner_cell)) + 1
    colors = greedy_face_coloring(flat.owner_cell, n_cells_est, flat.neighbor_cell, flat.is_boundary)
    n_colors = int(np.max(colors)) + 1
    color_masks = get_color_masks(colors, n_colors)
    return colors, color_masks, n_colors
