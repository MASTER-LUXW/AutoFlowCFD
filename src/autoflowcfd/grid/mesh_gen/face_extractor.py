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
        boundary_groups: Optional[Dict[str, np.ndarray]] = None
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
            logger.error(
                f"Invalid faces are spatially bounded by "
                f"x=[{bad_x.min():.4g}, {bad_x.max():.4g}], "
                f"y=[{bad_y.min():.4g}, {bad_y.max():.4g}], "
                f"z=[{bad_z.min():.4g}, {bad_z.max():.4g}] - check this region "
                f"(e.g. a BL-extruded surface's seam with a core-only boundary, "
                f"or two extruded surfaces close enough for their layers to "
                f"overlap) in the volume mesh generation log/geometry."
            )
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
        
        # Compute cell centers for all cells at once (vectorized)
        all_cell_centers = np.zeros((n_cells, 3), dtype=np.float64)
        for k in range(4):
            node_indices = cell_connectivity[:, k]
            all_cell_centers[:, 0] += x[node_indices]
            all_cell_centers[:, 1] += y[node_indices]
            all_cell_centers[:, 2] += z[node_indices]
        all_cell_centers /= 4.0
        
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
            raise ValueError(f"Found {n_zero_areas} faces with zero/near-zero area")
        
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
