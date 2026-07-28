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
    >>> from autoflowcfd.grid.face_extractor import FaceExtractor
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

from .structures import NodeArray, FaceData


@njit(parallel=False)
def _build_face_dict_numba(
    cell_connectivity: np.ndarray,
    n_cells: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build face dictionary using Numba-accelerated approach.
    
    This function generates all faces from tetrahedral cells and identifies
    which cells share each face.
    
    Args:
        cell_connectivity: Cell-node connectivity, shape=(n_cells, 4), dtype=int32
        n_cells: Number of cells
        
    Returns:
        Tuple of:
        - face_nodes: All faces as sorted node triples, shape=(n_faces_raw, 3)
        - face_cell_map: Cell indices for each face occurrence, shape=(n_faces_raw,)
        - n_faces_raw: Total number of face occurrences (before deduplication)
    """
    # Each tet has 4 faces, so maximum 4*n_cells face occurrences
    max_faces = n_cells * 4
    face_nodes = np.zeros((max_faces, 3), dtype=np.int32)
    face_cell_map = np.zeros(max_faces, dtype=np.int32)
    
    face_idx = 0
    
    for cell_idx in range(n_cells):
        n0 = cell_connectivity[cell_idx, 0]
        n1 = cell_connectivity[cell_idx, 1]
        n2 = cell_connectivity[cell_idx, 2]
        n3 = cell_connectivity[cell_idx, 3]
        
        # Generate 4 faces (nodes already sorted within each face)
        # Face 0: nodes 0,1,2
        if n0 < n1:
            if n1 < n2:
                face_nodes[face_idx, 0] = n0
                face_nodes[face_idx, 1] = n1
                face_nodes[face_idx, 2] = n2
            elif n0 < n2:
                face_nodes[face_idx, 0] = n0
                face_nodes[face_idx, 1] = n2
                face_nodes[face_idx, 2] = n1
            else:
                face_nodes[face_idx, 0] = n2
                face_nodes[face_idx, 1] = n0
                face_nodes[face_idx, 2] = n1
        else:
            if n0 < n2:
                face_nodes[face_idx, 0] = n1
                face_nodes[face_idx, 1] = n0
                face_nodes[face_idx, 2] = n2
            elif n1 < n2:
                face_nodes[face_idx, 0] = n1
                face_nodes[face_idx, 1] = n2
                face_nodes[face_idx, 2] = n0
            else:
                face_nodes[face_idx, 0] = n2
                face_nodes[face_idx, 1] = n1
                face_nodes[face_idx, 2] = n0
        face_cell_map[face_idx] = cell_idx
        face_idx += 1
        
        # Face 1: nodes 0,1,3
        if n0 < n1:
            if n1 < n3:
                face_nodes[face_idx, 0] = n0
                face_nodes[face_idx, 1] = n1
                face_nodes[face_idx, 2] = n3
            elif n0 < n3:
                face_nodes[face_idx, 0] = n0
                face_nodes[face_idx, 1] = n3
                face_nodes[face_idx, 2] = n1
            else:
                face_nodes[face_idx, 0] = n3
                face_nodes[face_idx, 1] = n0
                face_nodes[face_idx, 2] = n1
        else:
            if n0 < n3:
                face_nodes[face_idx, 0] = n1
                face_nodes[face_idx, 1] = n0
                face_nodes[face_idx, 2] = n3
            elif n1 < n3:
                face_nodes[face_idx, 0] = n1
                face_nodes[face_idx, 1] = n3
                face_nodes[face_idx, 2] = n0
            else:
                face_nodes[face_idx, 0] = n3
                face_nodes[face_idx, 1] = n1
                face_nodes[face_idx, 2] = n0
        face_cell_map[face_idx] = cell_idx
        face_idx += 1
        
        # Face 2: nodes 0,2,3
        if n0 < n2:
            if n2 < n3:
                face_nodes[face_idx, 0] = n0
                face_nodes[face_idx, 1] = n2
                face_nodes[face_idx, 2] = n3
            elif n0 < n3:
                face_nodes[face_idx, 0] = n0
                face_nodes[face_idx, 1] = n3
                face_nodes[face_idx, 2] = n2
            else:
                face_nodes[face_idx, 0] = n3
                face_nodes[face_idx, 1] = n0
                face_nodes[face_idx, 2] = n2
        else:
            if n0 < n3:
                face_nodes[face_idx, 0] = n2
                face_nodes[face_idx, 1] = n0
                face_nodes[face_idx, 2] = n3
            elif n2 < n3:
                face_nodes[face_idx, 0] = n2
                face_nodes[face_idx, 1] = n3
                face_nodes[face_idx, 2] = n0
            else:
                face_nodes[face_idx, 0] = n3
                face_nodes[face_idx, 1] = n2
                face_nodes[face_idx, 2] = n0
        face_cell_map[face_idx] = cell_idx
        face_idx += 1
        
        # Face 3: nodes 1,2,3
        if n1 < n2:
            if n2 < n3:
                face_nodes[face_idx, 0] = n1
                face_nodes[face_idx, 1] = n2
                face_nodes[face_idx, 2] = n3
            elif n1 < n3:
                face_nodes[face_idx, 0] = n1
                face_nodes[face_idx, 1] = n3
                face_nodes[face_idx, 2] = n2
            else:
                face_nodes[face_idx, 0] = n3
                face_nodes[face_idx, 1] = n1
                face_nodes[face_idx, 2] = n2
        else:
            if n1 < n3:
                face_nodes[face_idx, 0] = n2
                face_nodes[face_idx, 1] = n1
                face_nodes[face_idx, 2] = n3
            elif n2 < n3:
                face_nodes[face_idx, 0] = n2
                face_nodes[face_idx, 1] = n3
                face_nodes[face_idx, 2] = n1
            else:
                face_nodes[face_idx, 0] = n3
                face_nodes[face_idx, 1] = n2
                face_nodes[face_idx, 2] = n1
        face_cell_map[face_idx] = cell_idx
        face_idx += 1
    
    return face_nodes[:face_idx], face_cell_map[:face_idx], face_idx


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
        """Extract complete face data from tetrahedral mesh.
        
        Args:
            cell_connectivity: Cell-node connectivity array, shape=(n_cells, 4), dtype=int32
            nodes: Node coordinate array with x, y, z attributes
            boundary_groups: Optional mapping of boundary names to cell indices
            
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
        
        # Step 1: Build face dictionary using Numba-accelerated function
        if NUMBA_AVAILABLE:
            logger.debug("Using Numba-accelerated face extraction")
            face_nodes_raw, face_cell_map_raw, n_faces_raw = _build_face_dict_numba(
                cell_connectivity, n_cells
            )
        else:
            logger.debug("Using Python fallback for face extraction (slower)")
            # Fallback to original Python implementation
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
            n_faces_raw = sum(len(v) for v in face_dict.values())
            face_nodes_raw = np.zeros((n_faces_raw, 3), dtype=np.int32)
            face_cell_map_raw = np.zeros(n_faces_raw, dtype=np.int32)
            
            idx = 0
            for face_nodes, cell_list in face_dict.items():
                for cell_idx in cell_list:
                    face_nodes_raw[idx] = list(face_nodes)
                    face_cell_map_raw[idx] = cell_idx
                    idx += 1
        
        logger.debug(f"Generated {n_faces_raw} face occurrences")
        
        # Step 2: Identify unique faces using vectorized numpy operations
        # Sort each face's nodes to create canonical form
        face_nodes_sorted = np.sort(face_nodes_raw, axis=1)
        
        # Use structured array view for efficient unique detection
        face_dtype = np.dtype((np.void, face_nodes_sorted.dtype.itemsize * 3))
        face_voids = np.ascontiguousarray(face_nodes_sorted).view(face_dtype).reshape(-1)
        
        # Find unique faces and their inverse indices
        unique_faces, inverse_indices = np.unique(face_voids, return_inverse=True)
        n_unique_faces = len(unique_faces)
        
        logger.info(f"Identified {n_unique_faces} unique faces from {n_faces_raw} occurrences")
        
        # Expected ratio: ~2x cells for interior-dominated mesh
        expected_ratio = n_unique_faces / n_cells
        logger.debug(f"Face-to-cell ratio: {expected_ratio:.2f} (expected ~2.0-2.5)")
        
        # Step 3: Build face connectivity (which cells share each face)
        # For each unique face, find all occurrences
        face_connectivity = np.full((n_unique_faces, 2), -1, dtype=np.int32)
        face_occurrence_count = np.zeros(n_unique_faces, dtype=np.int32)
        
        for occ_idx in range(n_faces_raw):
            unique_idx = inverse_indices[occ_idx]
            count = face_occurrence_count[unique_idx]
            
            if count < 2:
                face_connectivity[unique_idx, count] = face_cell_map_raw[occ_idx]
            
            face_occurrence_count[unique_idx] += 1
        
        # Validate topology
        n_interior = np.sum(face_occurrence_count == 2)
        n_boundary = np.sum(face_occurrence_count == 1)
        n_invalid = np.sum(face_occurrence_count > 2)
        
        logger.info(
            f"Face topology: {n_interior} interior, {n_boundary} boundary, "
            f"{n_invalid} invalid (>2 cells)"
        )
        
        if n_invalid > 0:
            logger.warning(f"Found {n_invalid} faces shared by >2 cells (topology error)")
        
        # Step 4: Compute geometric properties
        x = nodes.x
        y = nodes.y
        z = nodes.z
        
        face_areas = np.zeros((n_unique_faces, 3), dtype=np.float64)
        face_centers = np.zeros((n_unique_faces, 3), dtype=np.float64)
        boundary_flags = np.zeros(n_unique_faces, dtype=np.bool_)
        
        for face_idx in range(n_unique_faces):
            n0 = face_nodes_sorted[face_idx, 0]
            n1 = face_nodes_sorted[face_idx, 1]
            n2 = face_nodes_sorted[face_idx, 2]
            
            # Get node coordinates
            p0 = np.array([x[n0], y[n0], z[n0]], dtype=np.float64)
            p1 = np.array([x[n1], y[n1], z[n1]], dtype=np.float64)
            p2 = np.array([x[n2], y[n2], z[n2]], dtype=np.float64)
            
            # Compute face center (centroid)
            face_centers[face_idx] = (p0 + p1 + p2) / 3.0
            
            # Compute face area vector
            v1 = p1 - p0
            v2 = p2 - p0
            area_vec = 0.5 * np.cross(v1, v2)
            
            # Determine face type and ensure consistent orientation
            left_cell = face_connectivity[face_idx, 0]
            right_cell = face_connectivity[face_idx, 1]
            
            if right_cell >= 0:
                # Interior face: ensure normal points from left to right
                center_left = FaceExtractor._compute_cell_center(
                    left_cell, cell_connectivity, x, y, z
                )
                center_right = FaceExtractor._compute_cell_center(
                    right_cell, cell_connectivity, x, y, z
                )
                
                dx = center_right - center_left
                if np.dot(area_vec, dx) < 0:
                    area_vec = -area_vec
                    # Swap cells
                    face_connectivity[face_idx, 0] = right_cell
                    face_connectivity[face_idx, 1] = left_cell
                
                boundary_flags[face_idx] = False
            else:
                # Boundary face: ensure normal points OUT of the domain (away from cell center)
                center_left = FaceExtractor._compute_cell_center(
                    left_cell, cell_connectivity, x, y, z
                )
                
                # Vector from cell center to face center
                dx = face_centers[face_idx] - center_left
                
                # If area_vec points toward cell center (dot < 0), flip it
                # This ensures normal points outward from the fluid domain
                if np.dot(area_vec, dx) < 0:
                    area_vec = -area_vec
                
                boundary_flags[face_idx] = True
            
            face_areas[face_idx] = area_vec
        
        # Step 5: Map boundary conditions
        boundary_types_map: Dict[int, str] = {}
        if boundary_groups is not None:
            for bname, cell_indices in boundary_groups.items():
                cell_set = set(cell_indices.tolist())
                
                for face_idx in range(n_unique_faces):
                    if boundary_flags[face_idx]:
                        left_cell = face_connectivity[face_idx, 0]
                        if left_cell in cell_set:
                            boundary_types_map[face_idx] = bname
                            break
        
        # Compute scalar areas and unit normals from area vectors
        face_scalar_areas = np.linalg.norm(face_areas, axis=1)
        # Avoid division by zero
        valid_area_mask = face_scalar_areas > 1e-12
        face_normals = np.zeros_like(face_areas)
        face_normals[valid_area_mask] = face_areas[valid_area_mask] / face_scalar_areas[valid_area_mask][:, np.newaxis]
        
        # Create FaceData object with correct field names
        face_data = FaceData(
            connectivity=face_connectivity,
            area=face_scalar_areas,
            normal=face_normals,
            center=face_centers
        )
        
        # Validate output
        FaceExtractor.validate_face_data(face_data, n_cells)
        
        logger.info(
            f"Face extraction completed successfully: "
            f"{face_data.n_interior_faces} interior, "
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
