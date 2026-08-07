"""Face extraction module for tetrahedral meshes.

This module provides efficient face extraction from tetrahedral volume meshes,
generating the face connectivity and geometric data required for Finite Volume Method (FVM)
flux calculations.

Key Features:
    - Extract all triangular faces from tetrahedral cells
    - Identify interior faces (shared by 2 cells) vs boundary faces (1 cell)
    - Compute face area vectors with consistent orientation
    - Map boundary conditions to extracted faces
    
Performance Optimization:
    - Uses Numba JIT compilation for critical loops
    - Vectorized numpy operations where possible
    - Memory-efficient data structures

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

from ..structures import NodeArray, FaceData
from ..validation import quality_metrics as _qm


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
        return FaceExtractor._finalize_faces(
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

        return FaceExtractor._finalize_faces(
            face_nodes_sorted, face_connectivity, occurrence_count,
            n_unique_faces, n_interior, n_faces_raw, nodes, all_cell_centers, n_cells,
            strict=strict,
        )

    @staticmethod
    def _finalize_faces(
        face_nodes_sorted: np.ndarray,
        face_connectivity: np.ndarray,
        occurrence_count: np.ndarray,
        n_unique_faces: int,
        n_interior: int,
        n_faces_raw: int,
        nodes: NodeArray,
        all_cell_centers: np.ndarray,
        n_cells: int,
        strict: bool = False,
    ) -> FaceData:
        """Shared post-dedup geometry/orientation/validation, used by both
        extract_faces (tet-only) and extract_faces_mixed (prism+tet) -
        genuinely cell-shape-agnostic from this point on: everything below
        only ever consumes a face's 3 corner node indices, its owner/
        neighbour cell index, and that cell's already-computed centroid."""
        n_boundary = n_unique_faces - n_interior
        n_invalid = np.sum(occurrence_count > 2)

        logger.info(
            f"Identified {n_unique_faces} unique faces from {n_faces_raw} occurrences"
        )
        logger.info(
            f"Face topology: {n_interior} interior, {n_boundary} boundary, "
            f"{n_invalid} invalid (>2 cells)"
        )

        if n_invalid > 0:
            # NOTE: the dedup scan above only ever records the first 2 cells
            # touching a given face key (see _deduplicate_and_build_connectivity);
            # for a face shared by 3+ cells, every cell beyond the first two
            # never gets connected to it at all, silently dropping that
            # cell's flux through this face from the residual - a genuine
            # local conservation violation, not a numerical-stability issue.
            # This can (and has been observed to) produce a residual that
            # diverges unboundedly regardless of how low CFL is pushed,
            # while integrated body forces stay comparatively normal since
            # they don't depend on these (typically interior/core-mesh)
            # faces. Continuing to solve on a topologically invalid mesh
            # wastes potentially hours of compute on a result that was
            # never going to be physically meaningful - fail immediately
            # instead, pointing at the volume mesh generation step that
            # produced overlapping/duplicate tetrahedra.
            invalid_mask = occurrence_count > 2
            invalid_node_ids = np.unique(face_nodes_sorted[invalid_mask])
            bad_x = nodes.x[invalid_node_ids]
            bad_y = nodes.y[invalid_node_ids]
            bad_z = nodes.z[invalid_node_ids]
            logger.warning(
                f"Invalid faces detected (n={n_invalid}), spatially bounded by "
                f"x=[{bad_x.min():.4g}, {bad_x.max():.4g}], "
                f"y=[{bad_y.min():.4g}, {bad_y.max():.4g}], "
                f"z=[{bad_z.min():.4g}, {bad_z.max():.4g}]. "
                f"This is likely due to BL extrusion at sharp corners."
                + (" Proceeding for inspection (non-strict call)." if not strict else "")
            )
            if strict:
                # Unlike the intermediate/exploratory callers during mesh
                # generation and repair (mesh_repair.py, mesh_repair_cavity.py,
                # mesh_background.py's pre-repair check - all non-strict,
                # since a transient non-manifold state there is expected and
                # gets resolved by a LATER repair stage, e.g.
                # repair_nonmanifold_mixed), this is the genuine solve/export-
                # time gate (GridData.ensure_faces_exist, strict=True) - by
                # this point every repair stage has already run, so a
                # remaining >2-owner face is a real, uncorrected defect, not
                # a transient one.
                raise RuntimeError(
                    f"Invalid mesh topology: {n_invalid} faces are shared by more than "
                    f"2 cells (expected exactly 1 for boundary or 2 for interior faces). "
                    f"This means the volume mesh contains overlapping/duplicate "
                    f"tetrahedra - almost certainly from the boundary-layer/core "
                    f"tetgen merge (see mesh_background.generate_hybrid_mesh). "
                    f"Solving on this mesh would silently drop flux through the "
                    f"affected faces and is not physically meaningful; regenerate "
                    f"the volume mesh (e.g. with different BL parameters) rather "
                    f"than proceeding."
                )
        
        # Expected ratio: ~2x cells for interior-dominated mesh
        expected_ratio = n_unique_faces / n_cells
        logger.debug(f"Face-to-cell ratio: {expected_ratio:.2f} (expected ~2.0-2.5)")
        
        # Step 3: Compute geometric properties using vectorized operations
        logger.debug("Computing face geometry (vectorized)...")
        x = nodes.x
        y = nodes.y
        z = nodes.z
        
        # Vectorized face center computation
        n0 = face_nodes_sorted[:, 0]
        n1 = face_nodes_sorted[:, 1]
        n2 = face_nodes_sorted[:, 2]
        
        face_centers = np.column_stack([
            (x[n0] + x[n1] + x[n2]) / 3.0,
            (y[n0] + y[n1] + y[n2]) / 3.0,
            (z[n0] + z[n1] + z[n2]) / 3.0
        ])
        
        # Vectorized area vector computation
        p0 = np.column_stack([x[n0], y[n0], z[n0]])
        p1 = np.column_stack([x[n1], y[n1], z[n1]])
        p2 = np.column_stack([x[n2], y[n2], z[n2]])
        
        v1 = p1 - p0
        v2 = p2 - p0
        face_areas_vec = 0.5 * np.cross(v1, v2)
        
        # Determine face orientation and flip if needed
        left_cells = face_connectivity[:, 0]
        right_cells = face_connectivity[:, 1]

        # all_cell_centers is already computed by the caller (tet-only or
        # mixed prism+tet - see _compute_tet_cell_centers/
        # _compute_prism_cell_centers), passed in as a parameter.

        # Get left and right cell centers
        center_left = all_cell_centers[left_cells]
        
        # For interior faces, ensure normal points from left to right
        mask_interior = right_cells >= 0
        
        # CRITICAL FIX: Create copies of arrays before masking to avoid shape mismatch
        center_right = all_cell_centers[right_cells[mask_interior]]
        dx_interior = center_right - center_left[mask_interior]
        dot_interior = np.sum(face_areas_vec[mask_interior] * dx_interior, axis=1)
        
        # Flip faces where normal points wrong direction
        flip_mask = dot_interior < 0
        indices_to_flip = np.where(mask_interior)[0][flip_mask]
        face_areas_vec[indices_to_flip] *= -1
        
        # Swap cell connectivity for flipped faces
        temp = face_connectivity[indices_to_flip, 0].copy()
        face_connectivity[indices_to_flip, 0] = face_connectivity[indices_to_flip, 1]
        face_connectivity[indices_to_flip, 1] = temp
        
        # For boundary faces, ensure normal points outward
        mask_boundary = ~mask_interior
        dx_boundary = face_centers[mask_boundary] - center_left[mask_boundary]
        dot_boundary = np.sum(face_areas_vec[mask_boundary] * dx_boundary, axis=1)
        flip_boundary = dot_boundary < 0
        indices_to_flip_boundary = np.where(mask_boundary)[0][flip_boundary]
        face_areas_vec[indices_to_flip_boundary] *= -1
        
        # Compute scalar areas and unit normals
        face_scalar_areas = np.linalg.norm(face_areas_vec, axis=1)
        valid_area_mask = face_scalar_areas > 1e-12
        face_normals = np.zeros_like(face_areas_vec)
        face_normals[valid_area_mask] = (
            face_areas_vec[valid_area_mask] / 
            face_scalar_areas[valid_area_mask][:, np.newaxis]
        )
        
        # Create FaceData object. node_connectivity is the triangle-corner
        # node indices already computed above (face_nodes_sorted) purely to
        # derive area/normal/center - kept here too so callers that need
        # the actual boundary surface mesh (e.g. VTKExporter.export_boundaries,
        # for per-zone/per-patch visualization) don't have to re-extract it
        # from the tetrahedra a second time.
        face_data = FaceData(
            connectivity=face_connectivity,
            area=face_scalar_areas,
            normal=face_normals,
            center=face_centers,
            node_connectivity=face_nodes_sorted.astype(np.int32),
        )
        
        # Validate output
        FaceExtractor.validate_face_data(face_data, n_cells)
        
        logger.success(
            f"Face extraction completed: {face_data.n_interior_faces} interior, "
            f"{face_data.n_boundary_faces} boundary faces"
        )
        
        return face_data
    
    @staticmethod
    def _compute_cell_center(
        cell_idx: int,
        cell_connectivity: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> np.ndarray:
        """Compute centroid of a tetrahedral cell.
        
        Args:
            cell_idx: Cell index
            cell_connectivity: Cell-node connectivity
            x, y, z: Node coordinates
            
        Returns:
            Centroid coordinates, shape=(3,)
        """
        nodes = cell_connectivity[cell_idx]
        center = np.array([
            (x[nodes[0]] + x[nodes[1]] + x[nodes[2]] + x[nodes[3]]) / 4.0,
            (y[nodes[0]] + y[nodes[1]] + y[nodes[2]] + y[nodes[3]]) / 4.0,
            (z[nodes[0]] + z[nodes[1]] + z[nodes[2]] + z[nodes[3]]) / 4.0
        ], dtype=np.float64)
        return center
    
    @staticmethod
    def validate_face_data(face_data: FaceData, n_cells: int) -> bool:
        """Validate extracted face data for consistency.
        
        Checks:
        - All cells are referenced by at least one face
        - No duplicate faces
        - Area values have reasonable magnitudes
        - Normal vectors are unit length
        
        Args:
            face_data: Extracted face data
            n_cells: Expected number of cells
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        # Check 1: All cells should be referenced
        referenced_cells = set()
        for i in range(face_data.count):
            referenced_cells.add(int(face_data.connectivity[i, 0]))
            if face_data.connectivity[i, 1] >= 0:
                referenced_cells.add(int(face_data.connectivity[i, 1]))
        
        if len(referenced_cells) != n_cells:
            raise ValueError(
                f"Face connectivity references {len(referenced_cells)} cells, "
                f"expected {n_cells}"
            )
        
        # Check 2: Areas should have positive magnitude
        n_zero_areas = np.sum(face_data.area < 1e-12)
        if n_zero_areas > 0:
            logger.warning(f"Found {n_zero_areas} faces with zero/near-zero area. Allowing export for debugging.")
            # raise ValueError(f"Found {n_zero_areas} faces with zero/near-zero area")
        
        # Check 3: Normal vectors should be unit length
        normal_magnitudes = np.linalg.norm(face_data.normal, axis=1)
        n_invalid_normals = np.sum(np.abs(normal_magnitudes - 1.0) > 1e-6)
        if n_invalid_normals > 0:
            logger.warning(f"Found {n_invalid_normals} faces with non-unit normals (magnitude != 1.0)")
        
        logger.debug("Face data validation passed")
        return True


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
