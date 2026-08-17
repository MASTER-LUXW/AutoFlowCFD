"""tetgen 核心域填充：填充后的清理与修复。

从 mesh_tetgen_core.py 拆分出来，负责 fill_core_volume 产出的四面体在
拼接/导出前需要做的收尾处理：重合点合并、超大四面体细分、非流形面修复、
以及从 tetgen 自带的 facet marker 反推每个单元所属的边界分组。
"""

from typing import List, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from loguru import logger


def _dedupe_coincident_points(
    points: np.ndarray,
    faces: np.ndarray,
    tolerance: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """合并重合点（在容差内）并重映射 faces（或四面体单元——
    `faces` 只是一个 (n, k) 索引数组，通过普通花式索引重映射，
    所以 k=3 或 k=4 均可直接使用）。

    同时返回 `remap`（shape=(len(points),)，旧索引 -> 新索引），
    这样持有指向同一原始 `points` 的任意其他索引数组的调用方
    （例如 fill_core_volume 单独读取的 `tgen.trifaces`）可以应用
    相同的重映射并保持一致——传 `remap[some_other_array]` 即可。
    `remap` 在没有重合点时为恒等映射。

    完全传递闭包（使用 scipy connected_components 处理重合图，
    而不是单跳合并），与本包其他地方旧的 `merge_conforming_meshes`
    节点去重逻辑不同。

    两个调用场景：一是 fill_core_volume 中的原始回退，用于 tetgen
    未返回完全保形边界的情况；二是 mesh_background.generate_hybrid_mesh
    对整体合并网格的最终防御性传递，用于一个在真实案例中发现的
    不同失败模式——当 mesh_repair.compute_bl_thickness_limit_override
    的响应式 BL 厚度上限需要限制非常大比例的表面顶点时（这本身
    就是上游产生广泛而非局部的坏单元的症状），许多节点的
    `remaining_budget` 在相同的几层内恰好达到零，将它们冻结在
    后续每层的相同坐标上——但每个仍然获得自己的独立全局节点
    索引（每层无条件一个新索引），所以结果是在不同索引下产生
    大量几何重合的点。这不会触发 repair_nonmanifold_cells（它按
    精确节点索引而非几何匹配），也不能可靠地触发退化体积过滤器
    （混合冻结节点和仍在增长的邻居的四面体可以有小但不可忽略的
    体积）——这是一个静默的拓扑撕裂（两组几何相同的面在不同索引
    下，各自独立计为法向边界面的），而不是崩溃，所以上游没有任何
    地方能捕获它。
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    pairs = tree.query_pairs(tolerance)

    n_points = len(points)
    if not pairs:
        return points, faces, np.arange(n_points, dtype=np.int64)

    rows = [p[0] for p in pairs]
    cols = [p[1] for p in pairs]
    # 确保数据数组为整数类型以避免 connected_components 产生浮点数标签
    graph = coo_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(n_points, n_points))
    n_components, labels = connected_components(graph, directed=False)

    # 确保 labels 为整数类型以用于索引
    labels = labels.astype(np.int64)

    # 使用每个分量中最小的原始索引作为代表。
    representative = np.full(n_components, n_points, dtype=np.int64)
    np.minimum.at(representative, labels, np.arange(n_points))

    new_index_of_label = np.arange(n_components)
    unique_points = points[representative]
    remap = new_index_of_label[labels]

    new_faces = remap[faces]
    logger.warning(
        f"Coincident-point fallback stitch: {n_points} -> {len(unique_points)} points "
        f"({n_points - len(unique_points)} merged)"
    )
    return unique_points, new_faces, remap


def _tet_volumes(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Unsigned tetrahedron volumes (orientation-independent)."""
    p0 = nodes[cells[:, 0]]
    p1 = nodes[cells[:, 1]]
    p2 = nodes[cells[:, 2]]
    p3 = nodes[cells[:, 3]]
    return np.abs(np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0


# subdivide_oversized_tetrahedra 自身深度限制的体积比安全上限：
# 每次质心细分精确地将四面体体积分为四分之一（见该函数文档字符串
# 中的证明），所以 8 层是 4**8 = 65,536 倍体积缩减——远超在 cube_demo
# 上直接测量到的最严重的逃逸四面体情况（约 16,000 倍目标值）。
_MAX_SUBDIVIDE_DEPTH = 8


def subdivide_oversized_tetrahedra(
    nodes: np.ndarray,
    tets: np.ndarray,
    max_volume: float,
    max_depth: int = _MAX_SUBDIVIDE_DEPTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """递归细分每个体积超过 `max_volume` 的四面体，方法是插入其
    自身质心作为新顶点，将其替换为以该质心和其 4 个原始面形成的
    4 个子四面体。

    存在原因：tetgen 自身的基于体积的精化（fill_core_volume 的
    `regions`/`varvolume`）不能可靠地覆盖每个单元：其精化队列是
    形状质量优先、体积第二，一个形状完美但过大且所有 4 个顶点都
    在 PLC 边界上的四面体（附近没有触发进一步插入的东西）可以
    完全不被处理——在 cube_demo 上直接确认（一个 14.15 m^3 的
    四面体从入口跨到出口，约 16,000 倍区域自身目标，无论
    volume_cap_fraction 是收紧到 0.1 还是放松到 0.5，或者单区域
    种子被替换为约 27 个分散种子都完全一样——见 mesh_background_merge.py
    自身历史了解该调查过程），以及在更大规模上通过 mesh_overlap_check.py
    在真实运行中记录到数千个异常大（0.1-3 m^2）的边界面的。
    本函数是一个确定性的、不依赖 tetgen 的保底措施，不依赖 tetgen
    的精化队列选择配合。

    选择质心细分（而非例如最长边二分）是因为它不需要与邻居单元
    协调：对于任意四面体 (A, B, C, D) 和质心 G = (A+B+C+D)/4，
    4 个子四面体 (A,B,C,G)、(A,B,D,G)、(A,C,D,G)、(B,C,D,G) 各自
    恰好有原始体积的 1/4，与原始四面体的形状无关（可直接证明：
    令 u=B-A, v=C-A, w=D-A，则 det(u, v, (u+v+w)/4) = det(u,v,w)/4，
    因为 (u+v+w)/4 的 u 和 v 分量在行列式中消去——所以
    Volume(A,B,C,G) = Volume(A,B,C,D)/4 精确成立），每个子体恰好
    保留原始四面体的 4 个面之一完全不变。共享该面的邻居看到的是
    未受影响的、仍然保形的边界——没有悬挂节点，不需要也去细分
    邻居，没有全局闭合/传播过程（不像最长边二分需要这些来保持
    保形）。这也意味着基于面的边界归属（attribute_cells_from_trifaces，
    按排序节点三元组匹配 tetgen 自身的 facet 标记）在结果上仍然
    无需修改地工作：继承了被标记边界面的子体仍然通过相同的匹配
    找到，且绕向对于这个匹配和本函数自身的（无符号）体积计算都
    不重要——任何下游方向要求都在稍后对整个合并网格统一归一化
    （mesh_background.py 的 orient_tetrahedra 调用）。

    Args:
        nodes: (n, 3) float64 节点坐标（仅 `tets` 实际引用的那些；
            例如 fill_core_volume 的返回值，不是与其他无关单元共享
            的数组——新质心顶点追加在末尾，所以任何指向同一原始
            `nodes` 数组的其他索引数组仍然有效，但在此调用之前
            不存在引用索引 >= len(nodes) 的东西）
        tets: (m, 4) int64 四面体连接关系，索引指向 `nodes`
        max_volume: 分裂阈值，单位与 `nodes` 坐标的立方相同
            （例如 m^3）
        max_depth: 每个原始超大四面体的递归分裂次数安全上限——
            超过这个层数后仍然超过 `max_volume` 的四面体保持原样
            （记录日志）而不是无限分裂

    Returns:
        (new_nodes, new_tets)：new_nodes 是 `nodes` 加上每插入一个
        质心追加的一行；new_tets 与 `tets` 总体积相同（质心细分是
        精确划分，不是近似），但长度不保持——每个分裂的四面体变成
        4 行，所以行顺序/计数与输入不同，不能假设位置对应。
    """
    nodes_arr = np.asarray(nodes, dtype=np.float64)
    pending = np.asarray(tets, dtype=np.int64)
    finished_chunks: List[np.ndarray] = []
    n_split_total = 0
    worst_before = float(_tet_volumes(nodes_arr, pending).max()) if len(pending) else 0.0

    for _ in range(max_depth):
        if len(pending) == 0:
            break
        vols = _tet_volumes(nodes_arr, pending)
        oversized = vols > max_volume
        if not np.any(oversized):
            finished_chunks.append(pending)
            pending = np.empty((0, 4), dtype=np.int64)
            break

        finished_chunks.append(pending[~oversized])
        to_split = pending[oversized]
        n_split_total += len(to_split)

        centroids = nodes_arr[to_split].mean(axis=1)
        base_idx = len(nodes_arr)
        centroid_idx = np.arange(base_idx, base_idx + len(to_split), dtype=np.int64)
        nodes_arr = np.vstack([nodes_arr, centroids])

        a, b, c, d = to_split[:, 0], to_split[:, 1], to_split[:, 2], to_split[:, 3]
        pending = np.concatenate([
            np.stack([a, b, c, centroid_idx], axis=1),
            np.stack([a, b, d, centroid_idx], axis=1),
            np.stack([a, c, d, centroid_idx], axis=1),
            np.stack([b, c, d, centroid_idx], axis=1),
        ], axis=0)

    if len(pending):
        logger.warning(
            f"subdivide_oversized_tetrahedra: {len(pending)} cell(s) still "
            f"exceed max_volume={max_volume:.4g} after {max_depth} levels "
            f"(worst {float(_tet_volumes(nodes_arr, pending).max()):.4g}) - "
            f"kept as-is rather than split indefinitely"
        )
        finished_chunks.append(pending)

    new_tets = np.vstack(finished_chunks) if finished_chunks else np.asarray(tets, dtype=np.int64)
    if n_split_total:
        logger.info(
            f"subdivide_oversized_tetrahedra: split {n_split_total} oversized "
            f"cell(s) (worst {worst_before:.4g} -> target {max_volume:.4g}), "
            f"{len(new_tets) - len(tets)} net new cells"
        )
    return nodes_arr, new_tets


def repair_nonmanifold_cells(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """检测使某些三角面被超过 2 个单元共享（非流形）的四面体，
    并标记多余的单元以便移除。

    主要已知原因已在源头修复：隔离的嵌入实体（例如车身）需要
    tetgen hole 种子，否则其内部会被填充与已占据该空间的 BL 棱柱
    重叠的伪四面体（见 mesh_domain_classify.find_point_inside_closed_shell
    和 fill_core_volume 的 `holes` 参数）。本函数作为安全网保留，
    用于捕获该机制未覆盖的情况（例如一个非常非凸的实体找不到
    hole 点，或者一个真正紧密的 BL seam 产生的近退化边界面上
    tetgen 的 `nobisect=True` 核心填充无法通过插入边界点来解决）。
    如果不修复，这是真实守恒违反：有限体积面提取只能将一个共享面
    归属到 3+ 个接触单元中的 2 个，静默丢弃通过该面的通量（见
    face_extractor.py 在恰好这种情况下的硬失败）。

    一个三角面在每一侧最多有一个合法的邻居单元（四面体的第 4 个
    顶点，即"顶点"，位于该面平面的那一侧）。当多个单元共享同一
    侧时，它们是物理重叠的副本——只保留体积最大的一个，丢弃其余，
    每个超共享面独立处理。

    Args:
        nodes: (n_nodes, 3) 节点坐标
        cells: (n_cells, 4) 四面体连接关系

    Returns:
        布尔保留掩码，shape=(n_cells,)；False 标记要移除的单元
    """
    n_cells = len(cells)
    keep = np.ones(n_cells, dtype=bool)
    if n_cells == 0:
        return keep

    face_templates = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ], dtype=np.int64)
    apex_of_face = np.array([3, 2, 1, 0], dtype=np.int64)

    all_faces = cells[:, face_templates].reshape(-1, 3)
    apex_nodes = cells[:, apex_of_face].reshape(-1)
    cell_of_face = np.repeat(np.arange(n_cells), 4)

    sorted_faces = np.sort(all_faces, axis=1)
    face_dtype = np.dtype((np.void, sorted_faces.dtype.itemsize * 3))
    face_voids = np.ascontiguousarray(sorted_faces).view(face_dtype).reshape(-1)

    order = np.argsort(face_voids, kind='stable')
    sorted_voids = face_voids[order]
    sorted_cells = cell_of_face[order]
    sorted_apex = apex_nodes[order]
    sorted_face_nodes = sorted_faces[order]

    change = np.flatnonzero(sorted_voids[1:] != sorted_voids[:-1]) + 1
    group_starts = np.concatenate([[0], change])
    group_ends = np.concatenate([change, [len(sorted_voids)]])
    counts = group_ends - group_starts

    invalid_groups = np.flatnonzero(counts > 2)
    if len(invalid_groups) == 0:
        return keep

    volumes = _tet_volumes(nodes, cells)
    n_removed = 0

    for gi in invalid_groups:
        s, e = group_starts[gi], group_ends[gi]
        face_cells = sorted_cells[s:e]
        n0, n1, n2 = sorted_face_nodes[s]
        p0 = nodes[n0]
        normal = np.cross(nodes[n1] - p0, nodes[n2] - p0)

        apexes = sorted_apex[s:e]
        signed_dist = (nodes[apexes] - p0) @ normal

        for side_mask in (signed_dist > 0, signed_dist <= 0):
            side_cells = face_cells[side_mask]
            if len(side_cells) <= 1:
                continue
            best = side_cells[np.argmax(volumes[side_cells])]
            for c in side_cells:
                if c != best and keep[c]:
                    keep[c] = False
                    n_removed += 1

    if n_removed:
        logger.warning(
            f"Repaired {len(invalid_groups)} non-manifold faces by removing "
            f"{n_removed} redundant overlapping tetrahedra"
        )
    return keep


def attribute_cells_from_trifaces(
    cells: np.ndarray,
    trifaces: np.ndarray,
    triface_markers: np.ndarray,
    marker_to_name: dict,
) -> np.ndarray:
    """从 fill_core_volume 的 facet 标记恢复每个单元的源边界分组，
    用于拥有一个被标记边界面的单元。

    当 fill_core_volume 以 nobisect=False（分级 max-cell-size 区域）
    运行时需要：tetgen 可能将一个输入边界细分为许多子面以满足
    大小上限，所以这些子面的节点索引不再存在于填充前的表面网格
    中，简单的节点索引匹配（先前存在的 mesh_boundary.py 回退）不再
    能找到它们。tetgen 自身的 facet 标记被无论怎么细分的每个被标记
    输入面的子面继承，所以按节点集合（而不是某个外部数组的索引）
    匹配单元自身的边界面对标记集可以无条件工作。

    Args:
        cells: (n_cells, 4) 四面体连接关系，与 `trifaces` 在同一索引
            空间中（即在任何节点重索引之前调用——重索引只改变节点
            索引的含义，从不改变哪个单元拥有哪一行，所以返回的每行
            分组赋值在后续重映射后仍然有效）
        trifaces: (n_tri, 3) 来自 fill_core_volume 的边界三角
        triface_markers: (n_tri,) int32，0 = 无标记（仅内部面，
            例如 BL/core 接口——永远不是真正的外边界，所以不归属
            也没问题）
        marker_to_name: 将非零标记值映射回其边界分组名称

    Returns:
        (n_cells,) str 数组，单元不拥有被标记边界面处为 ''
    """
    n_cells = len(cells)
    cell_groups = np.full(n_cells, '', dtype=object)

    nonzero = triface_markers != 0
    if not np.any(nonzero):
        return cell_groups

    marked_tri = np.sort(trifaces[nonzero], axis=1)
    marked_markers = triface_markers[nonzero]
    tri_dtype = np.dtype((np.void, marked_tri.dtype.itemsize * 3))
    marked_hash = np.ascontiguousarray(marked_tri).view(tri_dtype).reshape(-1)

    order = np.argsort(marked_hash, kind='stable')
    sorted_hash = marked_hash[order]
    sorted_marker = marked_markers[order]

    face_templates = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    all_faces = cells[:, face_templates].reshape(-1, 3)
    cell_of_face = np.repeat(np.arange(n_cells), 4)
    face_hash = np.ascontiguousarray(np.sort(all_faces, axis=1)).view(tri_dtype).reshape(-1)

    pos = np.clip(np.searchsorted(sorted_hash, face_hash), 0, len(sorted_hash) - 1)
    matched = sorted_hash[pos] == face_hash

    for cell_idx, marker in zip(cell_of_face[matched].tolist(), sorted_marker[pos[matched]].tolist()):
        cell_groups[cell_idx] = marker_to_name[marker]

    return cell_groups
