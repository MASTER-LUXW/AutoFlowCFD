"""四面体网格的面提取模块。

本模块提供从四面体体积网格中高效提取面的功能，
生成高阶 FR 求解器所需的面连接和几何数据。

核心功能:
    - 从四面体单元提取所有三角面
    - 识别内部面（共享2个单元）和边界脸（1个单元）
    - 计算具有一致方向的面面积矢量
    - 将边界条件映射到提取的面

性能优化:
    - 使用 Numba JIT 编译关键循环
    - 尽可能使用向量化 numpy 操作
    - 内存高效的数据结构

注意：底层 Numba/numpy 面构建原语在 face_extraction_kernels.py；
面积/法向/中心的收尾几何计算与校验在 face_geometry_finalize.py；
本文件只保留 FaceExtractor 的公开 API 编排。

Example:
    >>> from autoflowcfd.grid.mesh_gen.extraction.face_extractor import FaceExtractor
    >>> face_data = FaceExtractor.extract_faces(
    ...     cell_connectivity=cells.connectivity,
    ...     nodes=grid.nodes,
    ...     boundary_groups=boundaries.groups
    ... )
    >>> print(f"Extracted {face_data.count} faces")
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger

from ...schema.grid_nodes import NodeArray
from ...schema.grid_faces import FaceData
from ...validation.quality_metrics import (
    compute_prism_volumes,
    compute_tetrahedron_volumes,
)
from .face_extraction_kernels import (
    NUMBA_AVAILABLE,
    _build_face_dict_numba,
    _scan_sorted_faces_numba,
    _scan_sorted_faces_python,
    _compute_tet_cell_centers,
    _compute_prism_cell_centers,
    _build_prism_face_occurrences,
    _build_tet_face_occurrences_numpy,
)
from ..background.face_geometry_finalize import finalize_face_data, validate_face_data


def repair_nonmanifold_mixed(
    nodes: NodeArray,
    prism_connectivity: np.ndarray,
    tet_connectivity: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """检测混合 prism+tet 网格中被超过 2 个单元共享的面，并通过仅保留
    最大体积的拥有者来解析每个，丢弃其余——与 mesh_tetgen_core.
    repair_nonmanifold_cells 已使用的相同“重复是重叠副本，保留最大”
    理念，泛化到跨单元类型。

    mesh_tetgen_core.repair_nonmanifold_cells 是 tet 专用的（硬编码
    4 面/ apex 顶点逻辑），只看到混合网格的 tet 部分——被例如 2 个 tet +
    1 个棱柱共享的面（或任何涉及棱柱的重数）对它完全不可见。这已被确认
    为真实的、不仅理论上的缺口：在真实案例上，37 个这样的面在整个生成/
    修复流程中未被检测到，仅在最终某物第一次尝试构建完整混合网格的面图
    时作为硬 RuntimeError 出现在 FaceExtractor.extract_faces_mixed 自身
    的一致性检查中。

    Args:
        nodes: 完整节点数组（两种单元类型的共享坐标空间）
        prism_connectivity, tet_connectivity: 当前单元数组

    Returns:
        (prism_keep_mask, tet_keep_mask): bool 数组，False 标记要丢弃的
        单元。如果未发现过度共享的面则两者全 True（无操作）。
    """
    n_prism = len(prism_connectivity)
    n_tet = len(tet_connectivity)
    prism_keep = np.ones(n_prism, dtype=bool)
    tet_keep = np.ones(n_tet, dtype=bool)
    if n_prism + n_tet == 0:
        return prism_keep, tet_keep

    prism_key1, prism_max, prism_owner = _build_prism_face_occurrences(prism_connectivity, cell_index_offset=0)
    if NUMBA_AVAILABLE:
        tet_key1, tet_max, tet_owner_local, _ = _build_face_dict_numba(
            tet_connectivity.astype(np.int32), n_tet
        )
        tet_owner = tet_owner_local.astype(np.int64) + n_prism
    else:
        tet_key1, tet_max, tet_owner = _build_tet_face_occurrences_numpy(tet_connectivity, cell_index_offset=n_prism)

    key1 = np.concatenate([prism_key1, tet_key1])
    fmax = np.concatenate([prism_max, tet_max])
    owner = np.concatenate([prism_owner, tet_owner])
    if len(key1) == 0:
        return prism_keep, tet_keep

    order = np.lexsort((fmax, key1))
    key1_s, fmax_s, owner_s = key1[order], fmax[order], owner[order]

    change = np.ones(len(key1_s), dtype=bool)
    change[1:] = (key1_s[1:] != key1_s[:-1]) | (fmax_s[1:] != fmax_s[:-1])
    run_start = np.flatnonzero(change)
    run_len = np.diff(np.append(run_start, len(key1_s)))

    over_shared = np.flatnonzero(run_len > 2)
    if len(over_shared) == 0:
        return prism_keep, tet_keep

    pts = np.column_stack([nodes.x, nodes.y, nodes.z])
    n_dropped = 0
    for r in over_shared:
        start = int(run_start[r])
        length = int(run_len[r])
        cand_cells = owner_s[start:start + length]
        vols = np.empty(length)
        for i, c in enumerate(cand_cells):
            c = int(c)
            if c < n_prism:
                vols[i] = compute_prism_volumes(pts, prism_connectivity[c:c + 1])[0]
            else:
                t = c - n_prism
                vols[i] = abs(float(compute_tetrahedron_volumes(pts, tet_connectivity[t:t + 1])[0]))
        best = int(np.argmax(vols))
        for i, c in enumerate(cand_cells):
            if i == best:
                continue
            c = int(c)
            if c < n_prism:
                prism_keep[c] = False
            else:
                # 全局索引（tet 从 n_prism 开始，与上方 vols 循环里
                # `t = c - n_prism` 的换算相同），不是本地 tet_keep 数组的
                # 索引——`tet_keep[c] = False` 会在 c 超过 n_tet 时越界
                # （已在真实 cube_demo 网格上直接确认：IndexError，c 高达
                # 364726 对一个只有 202949 个 tet 的数组），且即便不越界，
                # 用错索引也会静默标记错误的 tet 为丢弃。上方第 126-127 行
                # 算体积时已经正确用了 `t = c - n_prism`，这里漏了同一个
                # 换算。
                tet_keep[c - n_prism] = False
            n_dropped += 1

    logger.warning(
        f"Mixed-mesh non-manifold repair: {len(over_shared)} face(s) shared by >2 cells "
        f"(spanning prism+tet, invisible to the tet-only repair_nonmanifold_cells check) - "
        f"dropped {n_dropped} redundant cell(s), keeping the largest-volume owner per face"
    )
    return prism_keep, tet_keep


class FaceExtractor:
    """从四面体网格提取面数据，用于 FVM 计算。

    本类将四面体单元连接转换为基于面的表示，
    有限体积法通量计算所需。

    提取过程：
    1. 枚举四面体单元的所有三角面
    2. 识别唯一面（按排序节点索引）
    3. 确定面类型：内部（2 个单元）或边界（1 个单元）
    4. 计算几何属性：面积向量、中心
    5. 确保一致的法向方向

    Attributes:
        None（无状态工具类）

    Example:
        >>> extractor = FaceExtractor()
        >>> face_data = extractor.extract_faces(
        ...     cell_connectivity=cells.connectivity,
        ...     nodes=mesh.nodes,
        ...     boundary_groups=boundaries.groups
        ... )
    """

    @staticmethod
    def extract_faces(
        cell_connectivity: np.ndarray,
        nodes: NodeArray,
        boundary_groups: Optional[Dict[str, np.ndarray]] = None,
        strict: bool = False,
    ) -> FaceData:
        """使用优化的基数排序方法从四面体网格提取完整面数据。

        这个优化版本用以下方法替代了慢速的 Python dict + np.unique：
        1. 位编码面键用于快速比较
        2. Numba 加速的基于 argsort 的去重
        3. 向量化几何计算

        性能提升：大网格（>1M 单元）约 10-20 倍

        Args:
            cell_connectivity: 单元-节点连接数组，形状=(n_cells, 4)，dtype=int32
            nodes: 节点坐标数组，具有 x, y, z 属性
            boundary_groups: 未使用；FaceData 不携带每面边界类型字段，
                调用方必须通过拥有者单元对照 BoundaryMap.groups
                分类边界（见 bc_handler.py）
            strict: 如果任何面被超过 2 个单元共享（无效拓扑）则抛出
                RuntimeError，而不是警告并继续。对生成和修复期间的
                中间/探索性调用方默认为 False，其中瞬态非流形状态是
                预期的并通过后续修复阶段解决——仅在真正的最终门控
                （见 GridData.ensure_faces_exist）传递 True，在所有
                修复阶段运行之后。

        Returns:
            FaceData: FVM 的完整面数据结构

        Raises:
            ValueError: 输入数组形状或类型无效
            RuntimeError: 面提取遇到拓扑错误
        """
        # 验证 inputs
        if len(cell_connectivity.shape) != 2 or cell_connectivity.shape[1] != 4:
            raise ValueError(
                f"cell_connectivity must be 2D array with shape (n_cells, 4), "
                f"got {cell_connectivity.shape}"
            )

        if cell_connectivity.dtype != np.int32:
            raise ValueError(f"cell_connectivity must be int32, got {cell_connectivity.dtype}")

        n_cells = cell_connectivity.shape[0]
        logger.info(f"Extracting faces from {n_cells} tetrahedral cells...")

        # 步骤 1：使用优化的 Numba 函数构建面数组
        if NUMBA_AVAILABLE:
            logger.debug("Using optimized radix-sort face extraction")
            face_key1_raw, face_max_raw, face_cell_map_raw, n_faces_raw = _build_face_dict_numba(
                cell_connectivity, n_cells
            )

            # 步骤 2：按 (face_key1, face_max) 排序——face_key1 为主，
            # face_max 作为字典序 tie-break。在普通 NumPy 中完成，因为
            # Numba 不支持 np.lexsort；这仍然是向量化的 O(n log n) 操作，
            # 不是 Python 循环。
            logger.debug("Sorting faces via lexsort...")
            sort_indices = np.lexsort((face_max_raw, face_key1_raw))
            sorted_key1 = face_key1_raw[sort_indices]
            sorted_max = face_max_raw[sort_indices]
            sorted_cells = face_cell_map_raw[sort_indices]

            # 步骤 3：通过单次扫描去重并构建连接
            logger.debug("Deduplicating faces via single-pass scan...")
            (face_nodes_sorted, face_connectivity,
             occurrence_count, n_unique_faces, n_interior) = \
                _scan_sorted_faces_numba(
                    sorted_key1, sorted_max, sorted_cells, n_faces_raw
                )
        else:
            logger.warning("Numba not available, falling back to slower Python implementation")
            # 回退到原始 Python 实现（为兼容性保留）
            face_dict: Dict[Tuple[int, int, int], List[int]] = {}

            for cell_idx in range(n_cells):
                nodes_idx = cell_connectivity[cell_idx]

                # 四面体有 4 个三角面
                faces = [
                    tuple(sorted([nodes_idx[0], nodes_idx[1], nodes_idx[2]])),
                    tuple(sorted([nodes_idx[0], nodes_idx[1], nodes_idx[3]])),
                    tuple(sorted([nodes_idx[0], nodes_idx[2], nodes_idx[3]])),
                    tuple(sorted([nodes_idx[1], nodes_idx[2], nodes_idx[3]]))
                ]

                for face_nodes in faces:
                    if face_nodes not in face_dict:
                        face_dict[face_nodes] = []
                    face_dict[face_nodes].append(cell_idx)

            # 将 dict 转换为数组
            n_unique_faces = len(face_dict)
            face_nodes_sorted = np.zeros((n_unique_faces, 3), dtype=np.int32)
            face_connectivity = np.full((n_unique_faces, 2), -1, dtype=np.int32)
            occurrence_count = np.zeros(n_unique_faces, dtype=np.int32)

            for idx, (face_nodes, cell_list) in enumerate(face_dict.items()):
                face_nodes_sorted[idx] = list(face_nodes)
                for i, cell_idx in enumerate(cell_list[:2]):  # 每面最多 2 个单元
                    face_connectivity[idx, i] = cell_idx
                occurrence_count[idx] = len(cell_list)

            n_interior = np.sum(occurrence_count == 2)

        all_cell_centers = _compute_tet_cell_centers(cell_connectivity, nodes)
        return finalize_face_data(
            face_nodes_sorted, face_connectivity, occurrence_count,
            n_unique_faces, n_interior, n_faces_raw, nodes, all_cell_centers, n_cells,
            strict=strict,
        )

    @staticmethod
    def extract_faces_mixed(
        prism_connectivity: np.ndarray,
        tet_connectivity: np.ndarray,
        nodes: NodeArray,
        strict: bool = False,
    ) -> FaceData:
        """从混合棱柱(BL) + 四面体(core) 网格提取面数据。

        全局单元索引约定（匹配 mesh_gen/mesh_repair.py 中全程使用的
        现有 n_bl_cells 约定）：棱柱占据 [0, n_prism)，四面体占据
        [n_prism, n_prism + n_tet)。

        每个棱柱直接贡献其 8 个边界三角形（2 个帽 + 3 个侧面四边形，
        每个四边形沿相同的“底部低索引到顶部高索引”对角线分割，
        mesh_prism_to_tet.convert_layers_to_tetrahedra 已使用的）而不是
        实例化为 3 个独立四面体然后事后合并——见 _build_prism_face_
        occurrences 了解推导。这保证共享侧面的两个棱柱选择相同的对角线
        （规则仅依赖全局节点索引比较，不是每棱柱选择），棱柱的帽面与
        相邻棱柱或四面体的面纯粹通过匹配排序节点三元组去重，与此处
        任何其他面相同——棱柱/核心四面体接口不需要特殊处理。

        Args:
            prism_connectivity: (n_prism, 6) int32，见 PrismCells 文档
                字符串了解 (v0,v1,v2,w0,w1,w2) 约定
            tet_connectivity: (n_tet, 4) int32
            nodes: 节点坐标
            strict: 见 extract_faces 的 `strict` 文档字符串

        Returns:
            FaceData，owner/邻居单元索引在上述组合全局索引空间中
        """
        n_prism = len(prism_connectivity)
        n_tet = len(tet_connectivity)
        n_cells = n_prism + n_tet
        logger.info(
            f"Extracting faces from {n_prism} prism + {n_tet} tetrahedral cells "
            f"({n_cells} total)..."
        )

        prism_key1, prism_max, prism_owner = _build_prism_face_occurrences(
            prism_connectivity, cell_index_offset=0
        )

        if NUMBA_AVAILABLE:
            tet_key1, tet_max, tet_owner_local, n_tet_faces_raw = _build_face_dict_numba(
                tet_connectivity.astype(np.int32), n_tet
            )
            tet_owner = tet_owner_local.astype(np.int64) + n_prism
        else:
            # 回退：复用与棱柱路径相同的向量化方法（numba 不可用）
            # 而不是第三次复制慢速的 Python-dict 回退
            tet_key1, tet_max, tet_owner = _build_tet_face_occurrences_numpy(
                tet_connectivity, cell_index_offset=n_prism
            )

        face_key1_raw = np.concatenate([prism_key1, tet_key1])
        face_max_raw = np.concatenate([prism_max, tet_max])
        face_cell_map_raw = np.concatenate([prism_owner, tet_owner]).astype(np.int32)
        n_faces_raw = len(face_key1_raw)

        logger.debug("Sorting faces via lexsort...")
        sort_indices = np.lexsort((face_max_raw, face_key1_raw))
        sorted_key1 = face_key1_raw[sort_indices]
        sorted_max = face_max_raw[sort_indices]
        sorted_cells = face_cell_map_raw[sort_indices]

        logger.debug("Deduplicating faces via single-pass scan...")
        if NUMBA_AVAILABLE:
            (face_nodes_sorted, face_connectivity,
             occurrence_count, n_unique_faces, n_interior) = _scan_sorted_faces_numba(
                sorted_key1, sorted_max, sorted_cells, n_faces_raw
            )
        else:
            (face_nodes_sorted, face_connectivity,
             occurrence_count, n_unique_faces, n_interior) = _scan_sorted_faces_python(
                sorted_key1, sorted_max, sorted_cells, n_faces_raw
            )

        all_cell_centers = np.vstack([
            _compute_prism_cell_centers(prism_connectivity, nodes),
            _compute_tet_cell_centers(tet_connectivity, nodes),
        ]) if n_prism > 0 else _compute_tet_cell_centers(tet_connectivity, nodes)

        return finalize_face_data(
            face_nodes_sorted, face_connectivity, occurrence_count,
            n_unique_faces, n_interior, n_faces_raw, nodes, all_cell_centers, n_cells,
            strict=strict,
        )

    @staticmethod
    def validate_face_data(face_data: FaceData, n_cells: int) -> bool:
        """验证提取的面数据的一致性——见
        face_geometry_finalize.validate_face_data 了解实现
        （保留为 FaceExtractor 静态方法，因为它是本类已建立的公开 API 的一部分）。
        """
        return validate_face_data(face_data, n_cells)


# 便捷函数，供直接使用
def extract_faces_from_tetrahedra(
    cell_connectivity: np.ndarray,
    nodes: NodeArray,
    boundary_groups: Optional[Dict[str, np.ndarray]] = None
) -> FaceData:
    """面提取的便捷包装器。

    Args:
        cell_connectivity: 单元-节点连接，形状=(n_cells, 4)
        nodes: 节点坐标
        boundary_groups: 可选边界条件映射

    Returns:
        FaceData: 完整面信息
    """
    return FaceExtractor.extract_faces(cell_connectivity, nodes, boundary_groups)
