"""面提取底层 kernel：Numba/numpy 面构建原语。

从 face_extractor.py 拆分出来，只保留和具体 FaceExtractor API 无关的、
纯粹的面枚举/编码/排序去重/单元质心计算这些底层构建块，供
face_extractor.py 的 FaceExtractor 类和 repair_nonmanifold_mixed 复用。
"""

import numpy as np
from typing import Tuple
from loguru import logger

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available, face extraction will be slower")
    # 提供 numba 不可用时的回退
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

from ...schema.grid_nodes import NodeArray


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
        if a > b: a, b = b, a
        if b > c: b, c = c, b
        if a > b: a, b = b, a
        face_key1[face_idx] = (np.int64(a) << 32) | np.int64(b)
        face_max[face_idx] = c
        face_cell_map[face_idx] = cell_idx
        face_idx += 1

        # Face 1: nodes 0,1,3
        a, b, c = n0, n1, n3
        if a > b: a, b = b, a
        if b > c: b, c = c, b
        if a > b: a, b = b, a
        face_key1[face_idx] = (np.int64(a) << 32) | np.int64(b)
        face_max[face_idx] = c
        face_cell_map[face_idx] = cell_idx
        face_idx += 1

        # Face 2: nodes 0,2,3
        a, b, c = n0, n2, n3
        if a > b: a, b = b, a
        if b > c: b, c = c, b
        if a > b: a, b = b, a
        face_key1[face_idx] = (np.int64(a) << 32) | np.int64(b)
        face_max[face_idx] = c
        face_cell_map[face_idx] = cell_idx
        face_idx += 1

        # Face 3: nodes 1,2,3
        a, b, c = n1, n2, n3
        if a > b: a, b = b, a
        if b > c: b, c = c, b
        if a > b: a, b = b, a
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
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
    alloc_size = n_faces_raw  # Conservative: use full size
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
            uniq_idx += 1
            # 安全检柋（alloc_size = n_faces_raw 时不应触发）
            if uniq_idx >= alloc_size:
                break  # Defensive: stop if we somehow exceed allocation

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
    """直接枚举每个棱柱的 8 个边界三角形（2 个帽 + 3 个侧面四边形，
    每个分割为 2 个三角形）——见 extract_faces_mixed 的文档字符串了解
    为什么这等价于但比实例化 3 个子四面体然后事后合并更便宜更简单。

    每个侧面四边形的对角线规则（推导自并且要求精确匹配
    mesh_prism_to_tet.convert_layers_to_tetrahedra 自己的“v0-w1、v1-w2、
    v0-w2”规则，使棱柱的面与旧的拆分为四面体路径产生的完全位相同）：
    将底部三角形顶点排序到 v0<v1<v2（并将相同的行置换携带到顶部三角形，
    使 w_i 保持在 v_i “上方”）后，8 个面为：
        底部帽：(v0, v1, v2)
        顶部帽：(w0, w1, w2)
        quad(v0,v1/w0,w1): (v0, v1, w1), (v0, w0, w1)
        quad(v1,v2/w1,w2): (v1, v2, w2), (v1, w1, w2)
        quad(v0,v2/w0,w2): (v0, v2, w2), (v0, w0, w2)

    Args:
        prism_connectivity: (n_prism, 6) int32/int64，(v0,v1,v2,w0,w1,w2)——
            不要求已底部排序；这里排序。
        cell_index_offset: 加到每个拥有者索引（如果棱柱占据全局单元索引
            空间的开始则为 0，按本模块约定它们总是如此——保留为参数而不
            是硬编码 0，原因与代码库中每个其他每区域偏移都是显式的一样，
            不是假设）

    Returns:
        (key1, max, owner)：每个形状=(n_prism*8,)——与 _build_face_dict_numba
        产生相同的编码，准备好与 tet 出现列表拼接并直接送入现有的 lexsort +
        去重扫描。
    """
    n_prism = len(prism_connectivity)
    if n_prism == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64))

    bottom = prism_connectivity[:, 0:3].astype(np.int64)
    top = prism_connectivity[:, 3:6].astype(np.int64)
    order = np.argsort(bottom, axis=1)
    row_idx = np.arange(n_prism)[:, None]
    sb = bottom[row_idx, order]
    st = top[row_idx, order]
    v0, v1, v2 = sb[:, 0], sb[:, 1], sb[:, 2]
    w0, w1, w2 = st[:, 0], st[:, 1], st[:, 2]

    faces = np.stack([
        np.stack([v0, v1, v2], axis=1),
        np.stack([w0, w1, w2], axis=1),
        np.stack([v0, v1, w1], axis=1),
        np.stack([v0, w0, w1], axis=1),
        np.stack([v1, v2, w2], axis=1),
        np.stack([v1, w1, w2], axis=1),
        np.stack([v0, v2, w2], axis=1),
        np.stack([v0, w0, w2], axis=1),
    ], axis=1)  # (n_prism, 8, 3)

    faces_flat = faces.reshape(-1, 3)
    owner = np.repeat(np.arange(n_prism, dtype=np.int64) + cell_index_offset, 8)

    # BL 挤出在恰好一个底部顶点停止增长的棱柱（v_i == w_i——有效的
    # “坍缩为楔形”单元，总体积仍然非零因为其他 2 个角有真实高度）恰好
    # 产生这 8 个面中的 2 个零面积重复顶点三角形（配对 v_i 与 w_i 的
    # 两个侧面四边形对角面）。旧的拆分为 3 个子四面体路径从不遇到这个，
    # 因为它默默丢弃相应的近零体积子四面体；这个直接枚举必须显式过滤
    # 它们，否则它们会到达 FaceData.__post_init__ 的正面积检查作为硬
    # 零面积面（已在真实案例上确认：78426 个这样的面）。
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


def _build_tet_face_occurrences_numpy(
    tet_connectivity: np.ndarray, cell_index_offset: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建 tet 面出现的向量化 numpy 回退，当 numba 不可用时
    （镜像 _build_face_dict_numba 的每 tet 4 面枚举，仅被
    extract_faces_mixed 的无 numba 路径使用）。
    """
    n_tet = len(tet_connectivity)
    if n_tet == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64))
    c = tet_connectivity.astype(np.int64)
    faces = np.stack([
        c[:, [0, 1, 2]], c[:, [0, 1, 3]], c[:, [0, 2, 3]], c[:, [1, 2, 3]],
    ], axis=1).reshape(-1, 3)
    key1, fmax = _encode_face_keys(faces)
    owner = np.repeat(np.arange(n_tet, dtype=np.int64) + cell_index_offset, 4)
    return key1, fmax, owner


def _scan_sorted_faces_python(
    sorted_key1: np.ndarray, sorted_max: np.ndarray, sorted_cells: np.ndarray, n_faces_raw: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """_scan_sorted_faces_numba 的纯 Python 直译（相同算法），用于
    extract_faces_mixed 的无 numba 回退。
    """
    alloc_size = n_faces_raw
    unique_key1 = np.zeros(alloc_size, dtype=np.int64)
    unique_max = np.zeros(alloc_size, dtype=np.int32)
    face_conn = np.full((alloc_size, 2), -1, dtype=np.int32)
    occurrence_count = np.zeros(alloc_size, dtype=np.int32)

    uniq_idx = 0
    unique_key1[0] = sorted_key1[0]
    unique_max[0] = sorted_max[0]
    face_conn[0, 0] = sorted_cells[0]
    occurrence_count[0] = 1

    for i in range(1, n_faces_raw):
        if sorted_key1[i] != sorted_key1[i - 1] or sorted_max[i] != sorted_max[i - 1]:
            uniq_idx += 1
            unique_key1[uniq_idx] = sorted_key1[i]
            unique_max[uniq_idx] = sorted_max[i]
            face_conn[uniq_idx, 0] = sorted_cells[i]
            occurrence_count[uniq_idx] = 1
        else:
            if occurrence_count[uniq_idx] < 2:
                face_conn[uniq_idx, occurrence_count[uniq_idx]] = sorted_cells[i]
            occurrence_count[uniq_idx] += 1

    n_unique_faces = uniq_idx + 1
    unique_key1 = unique_key1[:n_unique_faces]
    unique_max = unique_max[:n_unique_faces]
    face_conn = face_conn[:n_unique_faces]
    occurrence_count = occurrence_count[:n_unique_faces]

    n_interior = int(np.sum(occurrence_count == 2))

    face_nodes_decoded = np.zeros((n_unique_faces, 3), dtype=np.int32)
    face_nodes_decoded[:, 0] = (unique_key1 >> 32).astype(np.int32)
    face_nodes_decoded[:, 1] = (unique_key1 & 0xFFFFFFFF).astype(np.int32)
    face_nodes_decoded[:, 2] = unique_max

    return face_nodes_decoded, face_conn, occurrence_count, n_unique_faces, n_interior
