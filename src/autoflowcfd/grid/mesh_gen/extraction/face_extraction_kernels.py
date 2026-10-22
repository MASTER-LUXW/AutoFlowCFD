"""面提取底层 kernel：Numba / numpy 面构建原语（numba 是硬依赖，不设无 numba 回退）。

从 face_extractor.py 拆分出来，只保留和具体 FaceExtractor API 无关的、
纯粹的面枚举/编码/排序去重/单元质心计算这些底层构建块，供
face_extractor.py 的 FaceExtractor 类和 repair_nonmanifold_mixed 复用。
"""

import numpy as np
from typing import Tuple

from numba import njit

from ...schema.grid_nodes import NodeArray
from ..utils.prism_split import prism_face_triangles


@njit(parallel=False)
def _build_face_dict_numba(
    cell_connectivity: np.ndarray,
    n_cells: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """使用 Numba 加速方法构建面数组，采用排序友好的编码。

    本函数从四面体单元生成所有面并将每个排序三元组的最小两个节点索引
    编码到单个 int64 主键；第三个（最大）索引保留为单独的 tie-break 数组
    而不是打包到同一个 word 中。

    Args:
        cell_connectivity: 单元-节点连接，形状=(n_cells, 4)，dtype=int32
        n_cells: 单元数量

    Returns:
        元组：
        - face_key1: 每个面的编码主键 (min<<32 | mid)，形状=(n_faces_raw,)
        - face_max: 每个面 3 个排序节点索引中最大的（tie-break），形状=(n_faces_raw,)
        - face_cell_map: 每个面出现对应的单元索引，形状=(n_faces_raw,)
        - n_faces_raw: 面出现总数（去重前）
    """
    # 每个 tet 有 4 个面，所以最多 4*n_cells 个面出现
    max_faces = n_cells * 4
    face_key1 = np.zeros(max_faces, dtype=np.int64)
    face_max = np.zeros(max_faces, dtype=np.int32)
    face_cell_map = np.zeros(max_faces, dtype=np.int32)

    face_idx = 0

    for cell_idx in range(n_cells):
        n0 = cell_connectivity[cell_idx, 0]
        n1 = cell_connectivity[cell_idx, 1]
        n2 = cell_connectivity[cell_idx, 2]
        n3 = cell_connectivity[cell_idx, 3]

        # 生成 4 个面，排序节点索引。只将 (min, mid) 打包到 int64 主键
        # 通过 (min << 32) | mid：因为节点 ID 是 int32（< 2^31），这对任何
        # 节点数都安全，不会溢出（之前的 20 位/分量 3 路打包在超过 2^20（~1M）
        # 节点的网格上会默默损坏面键——将不相关的节点三元组错误地混为一谈——
        # 真实混合/BL 汽车气动网格经常超过这个数）。第三个（max）索引单独
        # 保留，在调用方通过 np.lexsort 用作排序 tie-breaker，而不是打包进去。

        # Face 0: nodes 0,1,2
        a, b, c = n0, n1, n2
        if a > b:
            a, b = b, a
        if b > c:
            b, c = c, b
        if a > b:
            a, b = b, a
        face_key1[face_idx] = (np.int64(a) << 32) | np.int64(b)
        face_max[face_idx] = c
        face_cell_map[face_idx] = cell_idx
        face_idx += 1

        # Face 1: nodes 0,1,3
        a, b, c = n0, n1, n3
        if a > b:
            a, b = b, a
        if b > c:
            b, c = c, b
        if a > b:
            a, b = b, a
        face_key1[face_idx] = (np.int64(a) << 32) | np.int64(b)
        face_max[face_idx] = c
        face_cell_map[face_idx] = cell_idx
        face_idx += 1

        # Face 2: nodes 0,2,3
        a, b, c = n0, n2, n3
        if a > b:
            a, b = b, a
        if b > c:
            b, c = c, b
        if a > b:
            a, b = b, a
        face_key1[face_idx] = (np.int64(a) << 32) | np.int64(b)
        face_max[face_idx] = c
        face_cell_map[face_idx] = cell_idx
        face_idx += 1

        # Face 3: nodes 1,2,3
        a, b, c = n1, n2, n3
        if a > b:
            a, b = b, a
        if b > c:
            b, c = c, b
        if a > b:
            a, b = b, a
        face_key1[face_idx] = (np.int64(a) << 32) | np.int64(b)
        face_max[face_idx] = c
        face_cell_map[face_idx] = cell_idx
        face_idx += 1

    return face_key1[:face_idx], face_max[:face_idx], face_cell_map[:face_idx], face_idx


@njit(parallel=False)
def _scan_sorted_faces_numba(
    sorted_key1: np.ndarray,
    sorted_max: np.ndarray,
    sorted_cells: np.ndarray,
    n_faces_raw: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """去重面并通过一次扫描已按字典序排序的 (key1, max) 对构建连接。

    排序本身（按 (face_key1, face_max)，face_key1 为主）在调用方用普通
    NumPy 的 np.lexsort 完成，因为 Numba 不支持 np.lexsort；本函数只做
    O(n) 扫描，几乎所有逐面出现的工作都在这里。

    Args:
        sorted_key1: 已排序的 face_key1 值（主键），形状=(n_faces_raw,)
        sorted_max: 相同排序顺序的 face_max 值（tie-break），形状=(n_faces_raw,)
        sorted_cells: 相同排序顺序的单元索引，形状=(n_faces_raw,)
        n_faces_raw: 面出现数

    Returns:
        元组：
        - face_nodes_decoded: 解码的节点三元组，形状=(n_unique, 3)
        - face_connectivity: 每个唯一面的 [left_cell, right_cell]
        - face_occurrence_count: 每个唯一面的计数
        - n_unique_faces: 唯一面数
        - n_interior: 内部面数（count==2）
    """
    # 关键修复：Numba 在 njit 函数中不支持 np.concatenate
    # 使用足够大的预分配代替动态调整大小
    # 为安全起见，分配完整大小（最坏情况：所有面都是唯一的）
    alloc_size = n_faces_raw  # 保守取法：按完整尺寸分配
    unique_key1_temp = np.zeros(alloc_size, dtype=np.int64)
    unique_max_temp = np.zeros(alloc_size, dtype=np.int32)
    face_conn_temp = np.full((alloc_size, 2), -1, dtype=np.int32)
    occurrence_count_temp = np.zeros(alloc_size, dtype=np.int32)

    uniq_idx = 0
    unique_key1_temp[0] = sorted_key1[0]
    unique_max_temp[0] = sorted_max[0]
    face_conn_temp[0, 0] = sorted_cells[0]
    occurrence_count_temp[0] = 1

    for i in range(1, n_faces_raw):
        if sorted_key1[i] != sorted_key1[i-1] or sorted_max[i] != sorted_max[i-1]:
            # 找到新的唯一面
            uniq_idx += 1    # uniq_idx <= i < n_faces_raw = alloc_size，不会越界
            unique_key1_temp[uniq_idx] = sorted_key1[i]
            unique_max_temp[uniq_idx] = sorted_max[i]
            face_conn_temp[uniq_idx, 0] = sorted_cells[i]
            occurrence_count_temp[uniq_idx] = 1
        else:
            # 相同面如前，添加第二个单元
            if occurrence_count_temp[uniq_idx] < 2:
                face_conn_temp[uniq_idx, occurrence_count_temp[uniq_idx]] = sorted_cells[i]
            occurrence_count_temp[uniq_idx] += 1

    n_unique_faces = uniq_idx + 1

    # 用切片修剪数组到实际大小（Numba 兼容）
    unique_key1 = unique_key1_temp[:n_unique_faces]
    unique_max = unique_max_temp[:n_unique_faces]
    face_conn = face_conn_temp[:n_unique_faces]
    occurrence_count = occurrence_count_temp[:n_unique_faces]

    # 计数内部 vs 边界
    n_interior = 0
    for i in range(n_unique_faces):
        if occurrence_count[i] == 2:
            n_interior += 1

    # 将面键解码回节点三元组。不需要掩码：key1 精确打包 (min << 32) | mid，
    # 对任何 int32 节点 ID 没有重叠风险，max 从未打包。
    face_nodes_decoded = np.zeros((n_unique_faces, 3), dtype=np.int32)
    for i in range(n_unique_faces):
        key1 = unique_key1[i]
        n0 = np.int32(key1 >> 32)
        n1 = np.int32(key1 & 0xFFFFFFFF)
        n2 = unique_max[i]
        face_nodes_decoded[i, 0] = n0
        face_nodes_decoded[i, 1] = n1
        face_nodes_decoded[i, 2] = n2

    return face_nodes_decoded, face_conn, occurrence_count, n_unique_faces, n_interior


def _compute_tet_cell_centers(cell_connectivity: np.ndarray, nodes: NodeArray) -> np.ndarray:
    """Vertex-average centroid of every tetrahedron, 形状=(n_cells, 3)."""
    x, y, z = nodes.x, nodes.y, nodes.z
    centers = np.zeros((len(cell_connectivity), 3), dtype=np.float64)
    for k in range(4):
        idx = cell_connectivity[:, k]
        centers[:, 0] += x[idx]
        centers[:, 1] += y[idx]
        centers[:, 2] += z[idx]
    centers /= 4.0
    return centers


def _compute_prism_cell_centers(prism_connectivity: np.ndarray, nodes: NodeArray) -> np.ndarray:
    """每个三角棱柱的顶点平均质心，形状=(n_cells, 3)。

    与 _compute_tet_cell_centers 相同的顶点平均约定（不是真正的体积质心）——
    与本模块其余部分对待四面体“中心”用于方向翻转的方式一致；只用于决定
    面的哪一侧是“内部”拥有单元，不需要任何体积上精确的量。
    """
    if len(prism_connectivity) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    x, y, z = nodes.x, nodes.y, nodes.z
    centers = np.zeros((len(prism_connectivity), 3), dtype=np.float64)
    for k in range(6):
        idx = prism_connectivity[:, k]
        centers[:, 0] += x[idx]
        centers[:, 1] += y[idx]
        centers[:, 2] += z[idx]
    centers /= 6.0
    return centers


def _encode_face_keys(face_nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """_build_face_dict_numba 每面编码的向量化 numpy 等价物：排序每个面
    的 3 个节点索引，将 (min, mid) 打包到一个 int64 键（min<<32 | mid），
    保持 max 单独作为 lexsort tie-break。
    face_nodes: (n_faces, 3) int32/int64 -> (key1, max)，每个 (n_faces,)。
    """
    sorted_nodes = np.sort(face_nodes.astype(np.int64), axis=1)
    key1 = (sorted_nodes[:, 0] << 32) | sorted_nodes[:, 1]
    return key1, sorted_nodes[:, 2].astype(np.int32)


def _build_prism_face_occurrences(
    prism_connectivity: np.ndarray, cell_index_offset: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每个棱柱的 8 个边界三角形（2 个帽 + 3 个侧面四边形各拆成 2 个，对角线规则见
    `mesh_gen/utils/prism_split.py`：过四边形最小全局编号顶点，与底 / 顶指定无关）的面键与所属单元。

    Args:
        prism_connectivity: (n_prism, 6)，(v0,v1,v2,w0,w1,w2)
        cell_index_offset: 加到每个所属单元下标上（棱柱占全局单元编号的开头时为 0）

    Returns:
        (key1, max, owner)：与 _build_face_dict_numba 相同的编码，可与四面体的面出现直接拼接后排序去重。
    """
    n_prism = len(prism_connectivity)
    if n_prism == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64))

    faces_flat = prism_face_triangles(prism_connectivity).reshape(-1, 3)
    owner = np.repeat(np.arange(n_prism, dtype=np.int64) + cell_index_offset, 8)

    # BL 挤出在恰好一个底部顶点停止增长的棱柱（v_i == w_i——有效的
    # “坍缩为楔形”单元，总体积仍然非零因为其他 2 个角有真实高度）恰好
    # 产生这 8 个面中的 2 个编号重复的零面积三角形（配对 v_i 与 w_i 的
    # 两个侧面四边形拆分面），必须显式过滤，否则会作为零面积面到达
    # FaceData 的正面积检查（真实案例上确认过 78426 个这样的面）。
    degenerate = (
        (faces_flat[:, 0] == faces_flat[:, 1])
        | (faces_flat[:, 0] == faces_flat[:, 2])
        | (faces_flat[:, 1] == faces_flat[:, 2])
    )
    if np.any(degenerate):
        faces_flat = faces_flat[~degenerate]
        owner = owner[~degenerate]

    key1, fmax = _encode_face_keys(faces_flat)
    return key1, fmax, owner
