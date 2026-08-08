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
    # Provide fallback for when numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

from ..structures import NodeArray


@njit(parallel=False)
def _build_face_dict_numba(
    cell_connectivity: np.ndarray,
    n_cells: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build face arrays using Numba-accelerated approach with a sort-friendly encoding.

    This function generates all faces from tetrahedral cells and encodes the
    two lowest node indices of each sorted triple into a single int64 primary
    key; the third (largest) index is kept as a separate tie-break array
    rather than being packed into the same word.

    Args:
        cell_connectivity: Cell-node connectivity, shape=(n_cells, 4), dtype=int32
        n_cells: Number of cells

    Returns:
        Tuple of:
        - face_key1: Encoded primary key (min<<32 | mid) per face, shape=(n_faces_raw,)
        - face_max: The largest of the 3 sorted node indices per face (tie-break), shape=(n_faces_raw,)
        - face_cell_map: Cell indices for each face occurrence, shape=(n_faces_raw,)
        - n_faces_raw: Total number of face occurrences (before deduplication)
    """
    # Each tet has 4 faces, so maximum 4*n_cells face occurrences
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

        # Generate 4 faces with sorted node indices. Pack only (min, mid)
        # into the int64 primary key via (min << 32) | mid: since node IDs
        # are int32 (< 2^31), this is safe for ANY node count without
        # overflow (the previous 20-bits-per-component 3-way packing
        # silently corrupted face keys - aliasing unrelated node triples
        # together - for any mesh with >2^20 (~1M) nodes, which real
        # hybrid/BL automotive-aero meshes routinely exceed). The third
        # (max) index is kept separate and used as the sort tie-breaker
        # via np.lexsort in the caller instead of being packed in.

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
    """Deduplicate faces and build connectivity via a single pass over
    already lexicographically-sorted (key1, max) pairs.

    The sort itself (by (face_key1, face_max), face_key1 primary) is done in
    plain NumPy via np.lexsort in the caller, since Numba does not support
    np.lexsort; this function only does the O(n) scan, which is where
    almost all of the per-face-occurrence work actually lives.

    Args:
        sorted_key1: face_key1 values already sorted (primary key), shape=(n_faces_raw,)
        sorted_max: face_max values in the same sorted order (tie-break), shape=(n_faces_raw,)
        sorted_cells: cell indices in the same sorted order, shape=(n_faces_raw,)
        n_faces_raw: Number of face occurrences

    Returns:
        Tuple of:
        - face_nodes_decoded: Decoded node triples, shape=(n_unique, 3)
        - face_connectivity: [left_cell, right_cell] for each unique face
        - face_occurrence_count: Count per unique face
        - n_unique_faces: Number of unique faces
        - n_interior: Number of interior faces (count==2)
    """
    # CRITICAL FIX: Numba doesn't support np.concatenate in njit functions
    # Use a large enough pre-allocation instead of dynamic resizing
    # For safety, allocate full size (worst case: all faces are unique)
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
            # New unique face found
            uniq_idx += 1
            # Safety check (should never trigger with alloc_size = n_faces_raw)
            if uniq_idx >= alloc_size:
                break  # Defensive: stop if we somehow exceed allocation

            unique_key1_temp[uniq_idx] = sorted_key1[i]
            unique_max_temp[uniq_idx] = sorted_max[i]
            face_conn_temp[uniq_idx, 0] = sorted_cells[i]
            occurrence_count_temp[uniq_idx] = 1
        else:
            # Same face as previous, add second cell
            if occurrence_count_temp[uniq_idx] < 2:
                face_conn_temp[uniq_idx, occurrence_count_temp[uniq_idx]] = sorted_cells[i]
            occurrence_count_temp[uniq_idx] += 1

    n_unique_faces = uniq_idx + 1

    # Trim arrays to actual size using slicing (Numba-compatible)
    unique_key1 = unique_key1_temp[:n_unique_faces]
    unique_max = unique_max_temp[:n_unique_faces]
    face_conn = face_conn_temp[:n_unique_faces]
    occurrence_count = occurrence_count_temp[:n_unique_faces]

    # Count interior vs boundary
    n_interior = 0
    for i in range(n_unique_faces):
        if occurrence_count[i] == 2:
            n_interior += 1

    # Decode face keys back to node triples. No masking needed: key1 packs
    # exactly (min << 32) | mid with no overlap risk for any int32 node ID,
    # and max was never packed at all.
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
    """Vertex-average centroid of every tetrahedron, shape=(n_cells, 3)."""
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
    """Vertex-average centroid of every triangular prism, shape=(n_cells, 3).

    Same vertex-average convention as _compute_tet_cell_centers (not a true
    volumetric centroid) - consistent with how the rest of this module
    already treats a tet's "center" for orientation-flip purposes; only
    used to decide which side of a face is "inside" the owner cell, not
    for any quantity that needs to be volumetrically exact.
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
    """Vectorized numpy equivalent of _build_face_dict_numba's per-face
    encoding: sort each face's 3 node indices, pack (min, mid) into one
    int64 key (min<<32 | mid), keep max separate as the lexsort tie-break.
    face_nodes: (n_faces, 3) int32/int64 -> (key1, max), each (n_faces,)."""
    sorted_nodes = np.sort(face_nodes.astype(np.int64), axis=1)
    key1 = (sorted_nodes[:, 0] << 32) | sorted_nodes[:, 1]
    return key1, sorted_nodes[:, 2].astype(np.int32)


def _build_prism_face_occurrences(
    prism_connectivity: np.ndarray, cell_index_offset: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate the 8 boundary triangles of every prism (2 caps + 3 side
    quads, each split into 2 triangles) directly - see extract_faces_mixed's
    docstring for why this is equivalent to, but cheaper and simpler than,
    materializing 3 sub-tets per prism and merging them after the fact.

    Diagonal rule for each side quad (derived from, and required to exactly
    match, mesh_prism_to_tet.convert_layers_to_tetrahedra's own "v0-w1,
    v1-w2, v0-w2" rule so a prism's faces are bit-identical to what the old
    split-to-tets path would have produced): after sorting the bottom
    triangle's vertices to v0<v1<v2 (and carrying the SAME row permutation
    over to the top triangle, so w_i stays "above" v_i), the 8 faces are:
        bottom cap:  (v0, v1, v2)
        top cap:     (w0, w1, w2)
        quad(v0,v1/w0,w1):  (v0, v1, w1), (v0, w0, w1)
        quad(v1,v2/w1,w2):  (v1, v2, w2), (v1, w1, w2)
        quad(v0,v2/w0,w2):  (v0, v2, w2), (v0, w0, w2)

    Args:
        prism_connectivity: (n_prism, 6) int32/int64, (v0,v1,v2,w0,w1,w2) -
            NOT required to already be bottom-sorted; sorted here.
        cell_index_offset: added to every owner index (0 if prisms occupy
            the start of the global cell index space, as they always do
            per this module's convention - kept as a parameter rather than
            a hardcoded 0 for the same reason every other per-region
            offset in this codebase is explicit, not assumed)

    Returns:
        (key1, max, owner): each shape=(n_prism*8,) - same encoding
        _build_face_dict_numba produces, ready to concatenate with a tet
        occurrence list and feed straight into the existing lexsort +
        dedup scan.
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

    # A prism whose BL extrusion stopped growing at exactly one base vertex
    # (v_i == w_i - a valid "collapsed to wedge" cell, total volume still
    # nonzero since the other 2 corners have real height) makes exactly 2 of
    # these 8 faces zero-area duplicate-vertex triangles (the two side-quad
    # diagonal faces that pair v_i with w_i). The old split-into-3-sub-tets
    # path never hit this because it silently dropped the corresponding
    # near-zero-volume sub-tet; this direct enumeration has to filter them
    # explicitly or they reach FaceData.__post_init__'s positive-area check
    # as hard zero-area faces (confirmed on a real case: 78426 such faces).
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
    """Vectorized-numpy fallback for building tet face occurrences when
    numba is unavailable (mirrors _build_face_dict_numba's 4-faces-per-tet
    enumeration, only used by extract_faces_mixed's no-numba path)."""
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
    """Pure-Python transliteration of _scan_sorted_faces_numba (same
    algorithm), for the no-numba fallback used by extract_faces_mixed."""
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
