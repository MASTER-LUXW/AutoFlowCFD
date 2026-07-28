"""FVM face extraction from tetrahedral mesh.

Extracts face connectivity and geometry from unstructured tetrahedral mesh
using integer hashing for efficient duplicate detection.
"""

import numpy as np
from typing import Dict
from loguru import logger


class FVMFaceExtractor:
    """Extracts face connectivity and geometry from tetrahedral mesh."""
    
    def __init__(self):
        self.face_connectivity = None
        self.face_areas = None
        self.face_centers = None
        self.face_normals = None
        self.boundary_flags = None
    
    def build_from_tetrahedra(self, connectivity: np.ndarray, nodes: np.ndarray) -> Dict[str, np.ndarray]:
        """Build face data structure using integer hashing for efficiency.
        
        Args:
            connectivity: Cell connectivity, shape=(n_cells, 4)
            nodes: Node coordinates, shape=(n_nodes, 3)
            
        Returns:
            Dictionary with face data arrays
        """
        n_cells = len(connectivity)
        logger.info(f"Building face connectivity from {n_cells} tetrahedra...")
        
        # Extract all faces (4 per tetrahedron)
        faces_per_cell = np.array([
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ], dtype=np.int64)
        
        # Generate all faces: shape=(n_cells*4, 3)
        all_faces = connectivity[:, faces_per_cell].reshape(-1, 3)
        n_total_faces = len(all_faces)
        
        # Create sorted face keys
        sorted_faces = np.sort(all_faces, axis=1)
        
        # Use integer hashing instead of structured array (much faster)
        logger.info("Computing face hashes...")
        max_node_id = np.max(connectivity) + 1
        
        # Compute unique hash for each face: n0 + n1*M + n2*M^2
        # Use uint64 to avoid overflow
        M = np.uint64(max_node_id)
        face_hashes = (
            np.uint64(sorted_faces[:, 0]) + 
            np.uint64(sorted_faces[:, 1]) * M + 
            np.uint64(sorted_faces[:, 2]) * M * M
        )
        
        # Find unique faces using hash
        logger.info("Finding unique faces...")
        unique_hashes, inverse_indices, counts = np.unique(
            face_hashes, 
            return_inverse=True, 
            return_counts=True
        )
        n_faces = len(unique_hashes)
        
        logger.info(f"Extracted {n_faces} unique faces from {n_total_faces} total faces")
        
        # Initialize output arrays
        self.face_connectivity = np.full((n_faces, 2), -1, dtype=np.int64)
        self.face_areas = np.zeros(n_faces, dtype=np.float64)
        self.face_centers = np.zeros((n_faces, 3), dtype=np.float64)
        self.face_normals = np.zeros((n_faces, 3), dtype=np.float64)
        self.boundary_flags = np.zeros(n_faces, dtype=np.int32)
        
        # Create cell index array
        cell_indices = np.repeat(np.arange(n_cells), 4)
        
        # Process boundary faces (count == 1) - fully vectorized
        boundary_mask = counts[inverse_indices] == 1
        if np.any(boundary_mask):
            logger.info(f"Processing {np.sum(boundary_mask)} boundary faces...")
            
            # Get positions of boundary faces in the original array
            boundary_positions = np.where(boundary_mask)[0]
            
            # Map to unique face indices
            boundary_face_idx = inverse_indices[boundary_positions]
            
            # For each unique boundary face, get the first occurrence
            unique_bf, bf_first_pos = np.unique(boundary_face_idx, return_index=True)
            
            # Get actual positions in original array
            bf_original_pos = boundary_positions[bf_first_pos]
            
            # Set connectivity and flags
            self.face_connectivity[unique_bf, 0] = cell_indices[bf_original_pos]
            self.face_connectivity[unique_bf, 1] = -1
            self.boundary_flags[unique_bf] = 1
            
            # Compute geometry for all boundary faces at once
            bf_nodes = all_faces[bf_original_pos]
            p0 = nodes[bf_nodes[:, 0]]
            p1 = nodes[bf_nodes[:, 1]]
            p2 = nodes[bf_nodes[:, 2]]
            
            self.face_centers[unique_bf] = (p0 + p1 + p2) / 3.0
            
            v1 = p1 - p0
            v2 = p2 - p0
            normals = np.cross(v1, v2)
            areas = 0.5 * np.linalg.norm(normals, axis=1)
            
            # Normalize normals
            valid_mask = areas > 1e-15
            norm_factors = np.where(valid_mask, areas * 2.0, 1.0)
            self.face_normals[unique_bf] = normals / norm_factors[:, np.newaxis]
            self.face_normals[unique_bf[~valid_mask]] = [0, 0, 1]
            
            self.face_areas[unique_bf] = areas
        
        # Process internal faces (count == 2) - use grouping approach
        internal_mask = counts[inverse_indices] == 2
        if np.any(internal_mask):
            logger.info(f"Processing {np.sum(internal_mask)} internal faces...")
            
            # Get positions of internal faces
            internal_positions = np.where(internal_mask)[0]
            internal_face_idx = inverse_indices[internal_positions]
            internal_cell_idx = cell_indices[internal_positions]
            internal_face_nodes = all_faces[internal_positions]
            
            # Sort by face index to group pairs together
            sort_order = np.argsort(internal_face_idx)
            sorted_face_idx = internal_face_idx[sort_order]
            sorted_cell_idx = internal_cell_idx[sort_order]
            sorted_face_nodes = internal_face_nodes[sort_order]
            
            # Each internal face appears exactly twice, process in pairs
            n_internal_pairs = len(sorted_face_idx) // 2
            
            if n_internal_pairs > 0:
                # Get face indices (every other element after sorting)
                face_indices = sorted_face_idx[::2]
                
                # Get cell pairs
                cell0 = sorted_cell_idx[::2]
                cell1 = sorted_cell_idx[1::2]
                
                # Get face nodes (use first of each pair)
                face_nodes = sorted_face_nodes[::2]
                
                # Set connectivity with consistent ordering
                mask0_first = cell0 < cell1
                self.face_connectivity[face_indices[mask0_first], 0] = cell0[mask0_first]
                self.face_connectivity[face_indices[mask0_first], 1] = cell1[mask0_first]
                self.face_connectivity[face_indices[~mask0_first], 0] = cell1[~mask0_first]
                self.face_connectivity[face_indices[~mask0_first], 1] = cell0[~mask0_first]
                
                self.boundary_flags[face_indices] = 0
                
                # Compute geometry for all internal faces at once
                p0 = nodes[face_nodes[:, 0]]
                p1 = nodes[face_nodes[:, 1]]
                p2 = nodes[face_nodes[:, 2]]
                
                self.face_centers[face_indices] = (p0 + p1 + p2) / 3.0
                
                v1 = p1 - p0
                v2 = p2 - p0
                normals = np.cross(v1, v2)
                areas = 0.5 * np.linalg.norm(normals, axis=1)
                
                # Normalize normals
                valid_mask = areas > 1e-15
                norm_factors = np.where(valid_mask, areas * 2.0, 1.0)
                self.face_normals[face_indices] = normals / norm_factors[:, np.newaxis]
                self.face_normals[face_indices[~valid_mask]] = [0, 0, 1]
                
                self.face_areas[face_indices] = areas
        
        # Statistics
        n_boundary = np.sum(self.boundary_flags)
        n_internal = n_faces - n_boundary
        logger.info(f"Face mapping: {n_faces} total ({n_internal} internal, {n_boundary} boundary)")
        
        return self.get_face_data()
    
    
    def get_face_data(self) -> Dict[str, np.ndarray]:
        """Get all face data."""
        return {
            'connectivity': self.face_connectivity,
            'areas': self.face_areas,
            'centers': self.face_centers,
            'normals': self.face_normals,
            'boundary_flags': self.boundary_flags
        }
