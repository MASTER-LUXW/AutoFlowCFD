"""Face extraction module for tetrahedral meshes.

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
    >>> from autoflowcfd.grid.mesh_gen.face_extractor import FaceExtractor
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

from ..structures import NodeArray, FaceData
from ..validation import quality_metrics as _qm
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
from .face_geometry_finalize import finalize_face_data, validate_face_data


def repair_nonmanifold_mixed(
    nodes: NodeArray,
    prism_connectivity: np.ndarray,
    tet_connectivity: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect faces shared by more than 2 cells across a MIXED prism+tet
    mesh and resolve each by keeping only the largest-volume owner,
    dropping the rest - the same "duplicates are overlapping copies, keep
    the biggest" philosophy mesh_tetgen_core.repair_nonmanifold_cells
    already uses, generalized across cell types.

    mesh_tetgen_core.repair_nonmanifold_cells is tet-specific (hardcoded
    4-face/apex-vertex logic) and only ever sees the tet portion of a
    mixed mesh - a face shared by, say, 2 tets + 1 prism (or any
    multiplicity involving a prism) is completely invisible to it. This
    was confirmed as a real, not just theoretical, gap: on a real case,
    37 such faces survived the entire generation/repair pipeline
    undetected and only surfaced as a hard RuntimeError in
    FaceExtractor.extract_faces_mixed's own conformality check, at the
    point something finally tried to build a face graph over the FULL
    mixed mesh for the first time.

    Args:
        nodes: full node array (shared coordinate space for both cell types)
        prism_connectivity, tet_connectivity: current cell arrays

    Returns:
        (prism_keep_mask, tet_keep_mask): bool arrays, False marks a cell
        to drop. Both all-True (no-op) if no over-shared face was found.
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
                vols[i] = _qm.compute_prism_volumes(pts, prism_connectivity[c:c + 1])[0]
            else:
                t = c - n_prism
                vols[i] = abs(float(_qm.compute_tetrahedron_volumes(pts, tet_connectivity[t:t + 1])[0]))
        best = int(np.argmax(vols))
        for i, c in enumerate(cand_cells):
            if i == best:
                continue
            c = int(c)
            if c < n_prism:
                prism_keep[c] = False
            else:
                tet_keep[c] = False
            n_dropped += 1

    logger.warning(
        f"Mixed-mesh non-manifold repair: {len(over_shared)} face(s) shared by >2 cells "
        f"(spanning prism+tet, invisible to the tet-only repair_nonmanifold_cells check) - "
        f"dropped {n_dropped} redundant cell(s), keeping the largest-volume owner per face"
    )
    return prism_keep, tet_keep


class FaceExtractor:
    """Extract face data from tetrahedral meshes for FVM computations.

    This class converts tetrahedral cell connectivity into face-based representation
    required for Finite Volume Method flux calculations.

    The extraction process:
    1. Enumerate all triangular faces from tetrahedral cells
    2. Identify unique faces (by sorted node indices)
    3. Determine face type: interior (2 cells) or boundary (1 cell)
    4. Compute geometric properties: area vectors, centers
    5. Ensure consistent normal orientation

    Attributes:
        None (stateless utility class)

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
        """Extract complete face data from tetrahedral mesh using optimized radix-sort approach.

        This optimized version replaces the slow Python dict + np.unique approach with:
        1. Bit-encoded face keys for fast comparison
        2. Numba-accelerated argsort-based deduplication
        3. Vectorized geometric computations

        Performance improvement: ~10-20x faster for large meshes (>1M cells)

        Args:
            cell_connectivity: Cell-node connectivity array, shape=(n_cells, 4), dtype=int32
            nodes: Node coordinate array with x, y, z attributes
            boundary_groups: Unused; FaceData carries no per-face boundary-type
                field, so callers must classify boundary faces via their
                owner cell against BoundaryMap.groups (see bc_handler.py)
            strict: Raise RuntimeError if any face is shared by more than 2
                cells (invalid topology) instead of warning and proceeding.
                Default False for intermediate/exploratory callers during
                generation and repair, where a transient non-manifold state
                is expected and gets resolved by a later repair stage - pass
                True only at a genuine final gate (see GridData.
                ensure_faces_exist), after all repair stages have run.

        Returns:
            FaceData: Complete face data structure for FVM

        Raises:
            ValueError: If input arrays have invalid shapes or types
            RuntimeError: If face extraction encounters topology errors
        """
        # Validate inputs
        if len(cell_connectivity.shape) != 2 or cell_connectivity.shape[1] != 4:
            raise ValueError(
                f"cell_connectivity must be 2D array with shape (n_cells, 4), "
                f"got {cell_connectivity.shape}"
            )

        if cell_connectivity.dtype != np.int32:
            raise ValueError(f"cell_connectivity must be int32, got {cell_connectivity.dtype}")

        n_cells = cell_connectivity.shape[0]
        logger.info(f"Extracting faces from {n_cells} tetrahedral cells...")

        # Step 1: Build face arrays using optimized Numba function
        if NUMBA_AVAILABLE:
            logger.debug("Using optimized radix-sort face extraction")
            face_key1_raw, face_max_raw, face_cell_map_raw, n_faces_raw = _build_face_dict_numba(
                cell_connectivity, n_cells
            )

            # Step 2: Sort by (face_key1, face_max) - face_key1 primary,
            # face_max as the lexicographic tie-break. Done in plain NumPy
            # since Numba doesn't support np.lexsort; this is still a
            # vectorized O(n log n) op, not a Python loop.
            logger.debug("Sorting faces via lexsort...")
            sort_indices = np.lexsort((face_max_raw, face_key1_raw))
            sorted_key1 = face_key1_raw[sort_indices]
            sorted_max = face_max_raw[sort_indices]
            sorted_cells = face_cell_map_raw[sort_indices]

            # Step 3: Deduplicate and build connectivity via single-pass scan
            logger.debug("Deduplicating faces via single-pass scan...")
            (face_nodes_sorted, face_connectivity,
             occurrence_count, n_unique_faces, n_interior) = \
                _scan_sorted_faces_numba(
                    sorted_key1, sorted_max, sorted_cells, n_faces_raw
                )
        else:
            logger.warning("Numba not available, falling back to slower Python implementation")
            # Fallback to original Python implementation (kept for compatibility)
            face_dict: Dict[Tuple[int, int, int], List[int]] = {}

            for cell_idx in range(n_cells):
                nodes_idx = cell_connectivity[cell_idx]

                # Tetrahedron has 4 triangular faces
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

            # Convert dict to arrays
            n_unique_faces = len(face_dict)
            face_nodes_sorted = np.zeros((n_unique_faces, 3), dtype=np.int32)
            face_connectivity = np.full((n_unique_faces, 2), -1, dtype=np.int32)
            occurrence_count = np.zeros(n_unique_faces, dtype=np.int32)

            for idx, (face_nodes, cell_list) in enumerate(face_dict.items()):
                face_nodes_sorted[idx] = list(face_nodes)
                for i, cell_idx in enumerate(cell_list[:2]):  # Max 2 cells per face
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
        """Extract face data from a mixed prism(BL) + tetrahedron(core) mesh.

        Global cell index convention (matches the pre-existing n_bl_cells
        convention used throughout mesh_gen/mesh_repair.py): prisms occupy
        [0, n_prism), tets occupy [n_prism, n_prism + n_tet).

        Each prism contributes its 8 boundary triangles directly (2 caps +
        3 side quads, each quad split along the same "lower bottom-index to
        higher corresponding top-index" diagonal mesh_prism_to_tet.
        convert_layers_to_tetrahedra already uses) rather than being
        materialized as 3 separate tets and merged after the fact - see
        _build_prism_face_occurrences for the derivation. This guarantees
        two prisms sharing a side face pick the same diagonal (the rule
        depends only on global node-index comparison, not per-prism
        choice), and a prism's cap face dedupes against a neighbouring
        prism's or tet's face purely by matching sorted node triple, same
        as any other face here - no special-casing needed for the prism/
        core-tet interface.

        Args:
            prism_connectivity: (n_prism, 6) int32, see PrismCells docstring
                for the (v0,v1,v2,w0,w1,w2) convention
            tet_connectivity: (n_tet, 4) int32
            nodes: node coordinates
            strict: see extract_faces' `strict` docstring.

        Returns:
            FaceData with owner/neighbor cell indices in the combined
            global index space described above.
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
            # Fallback: reuse the same vectorized approach as the prism
            # path (numba unavailable) rather than duplicating the slow
            # Python-dict fallback a third time.
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
        """Validate extracted face data for consistency - see
        face_geometry_finalize.validate_face_data for the implementation
        (kept as a FaceExtractor staticmethod too since it's part of this
        class's established public API)."""
        return validate_face_data(face_data, n_cells)


# Convenience function for direct use
def extract_faces_from_tetrahedra(
    cell_connectivity: np.ndarray,
    nodes: NodeArray,
    boundary_groups: Optional[Dict[str, np.ndarray]] = None
) -> FaceData:
    """Convenience wrapper for face extraction.

    Args:
        cell_connectivity: Cell-node connectivity, shape=(n_cells, 4)
        nodes: Node coordinates
        boundary_groups: Optional boundary condition mapping

    Returns:
        FaceData: Complete face information
    """
    return FaceExtractor.extract_faces(cell_connectivity, nodes, boundary_groups)
