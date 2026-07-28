"""Gradient computation and limiting for MUSCL reconstruction.

Implements Green-Gauss gradient calculation and slope limiting using sparse adjacency.
Optimized with Numba JIT compilation for high performance on large meshes.
"""

import numpy as np
from typing import Tuple
from loguru import logger


class GradientComputer:
    """Compute cell-centered gradients using Green-Gauss theorem.
    
    For unstructured meshes, computes gradients via:
        grad_phi_i = (1/V_i) * sum(phi_face * A_face * n_face)
    
    where the sum is over all faces of cell i.
    """
    
    @staticmethod
    def compute_gradients_green_gauss(
        solution: np.ndarray,
        connectivity: np.ndarray,
        face_areas: np.ndarray,
        face_normals: np.ndarray,
        cell_volumes: np.ndarray
    ) -> np.ndarray:
        """Compute gradients using Green-Gauss theorem (vectorized).
        
        For unstructured meshes, computes gradients via:
            grad_phi_i = (1/V_i) * sum(phi_f * A_f * n_f)
        
        where the sum is over all faces of cell i, and phi_f is the 
        face-interpolated value.
        
        Args:
            solution: Cell-centered solutions, shape=(n_cells, n_vars)
            connectivity: Face-to-cell connectivity, shape=(n_faces, 2)
                         connectivity[f] = [left_cell, right_cell]
            face_areas: Face areas, shape=(n_faces,)
            face_normals: Unit face normals pointing from left to right, shape=(n_faces, 3)
            cell_volumes: Cell volumes, shape=(n_cells,)
            
        Returns:
            Gradients, shape=(n_cells, n_vars, 3)
            gradients[i, v, :] = [d(phi_v)/dx, d(phi_v)/dy, d(phi_v)/dz]
        """
        n_cells = solution.shape[0]
        n_vars = solution.shape[1]
        n_faces = connectivity.shape[0]
        
        # Initialize gradients to zero
        gradients = np.zeros((n_cells, n_vars, 3), dtype=np.float64)
        
        # Vectorized face contributions
        # Filter valid faces (both cells exist)
        left_cells = connectivity[:, 0]
        right_cells = connectivity[:, 1]
        
        valid_mask = (left_cells >= 0) & (left_cells < n_cells) & \
                     (right_cells >= 0) & (right_cells < n_cells)
        
        valid_left = left_cells[valid_mask]
        valid_right = right_cells[valid_mask]
        
        # Face area vectors: A * n (shape: n_valid_faces x 3)
        area_vecs = face_areas[valid_mask][:, np.newaxis] * face_normals[valid_mask]  # (n_valid, 3)
        
        # Interpolate solution at face (arithmetic mean)
        # phi_face shape: (n_valid_faces, n_vars)
        phi_face = 0.5 * (solution[valid_left, :] + solution[valid_right, :])
        
        # Contribution per face per variable: phi_face * area_vec
        # Shape broadcasting: (n_valid, n_vars, 1) * (n_valid, 1, 3) -> (n_valid, n_vars, 3)
        contributions = phi_face[:, :, np.newaxis] * area_vecs[:, np.newaxis, :]
        
        # Accumulate into gradients using np.add.at (handles repeated indices)
        # Left cell: add contribution
        np.add.at(gradients, (valid_left, slice(None), slice(None)), contributions)
        
        # Right cell: subtract contribution (normal points inward)
        np.add.at(gradients, (valid_right, slice(None), slice(None)), -contributions)
        
        # Divide by cell volume: grad_φ = (1/V) * sum(phi_f * A_f * n_f)
        vol_inv = np.where(cell_volumes > 1e-12, 1.0 / cell_volumes, 0.0)
        gradients *= vol_inv[:, np.newaxis, np.newaxis]
        
        return gradients
    
    @staticmethod
    def build_sparse_adjacency(
        connectivity: np.ndarray,
        n_cells: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build sparse adjacency representation using NumPy.
        
        Creates CSR-format adjacency without scipy dependency.
        
        Args:
            connectivity: Face connectivity, shape=(n_faces, 2)
            n_cells: Number of cells
            
        Returns:
            row_ptr: CSR row pointers, shape=(n_cells+1,)
            col_idx: Column indices (neighbor lists)
        """
        # Extract valid connections
        left = connectivity[:, 0]
        right = connectivity[:, 1]
        
        # Filter valid connections
        valid_mask = (left >= 0) & (right >= 0) & (left < n_cells) & (right < n_cells)
        left_valid = left[valid_mask]
        right_valid = right[valid_mask]
        
        # Build symmetric adjacency (both directions)
        rows = np.concatenate([left_valid, right_valid])
        cols = np.concatenate([right_valid, left_valid])
        
        # Sort by row index for CSR format
        sort_idx = np.argsort(rows)
        rows_sorted = rows[sort_idx]
        cols_sorted = cols[sort_idx]
        
        # Build row_ptr (CSR format)
        row_ptr = np.zeros(n_cells + 1, dtype=np.int64)
        unique_rows, counts = np.unique(rows_sorted, return_counts=True)
        
        # Fill row_ptr
        for i, row_idx in enumerate(unique_rows):
            row_ptr[row_idx + 1] = counts[i]
        
        # Cumulative sum to get actual pointers
        row_ptr = np.cumsum(row_ptr)
        
        return row_ptr, cols_sorted


# Numba-accelerated gradient limiting kernel
try:
    from numba import njit, prange
    
    @njit(parallel=True, cache=True)
    def _gradient_limiting_kernel_numba(
        gradients: np.ndarray,
        row_ptr: np.ndarray,
        col_idx: np.ndarray,
        limiter_func
    ) -> np.ndarray:
        """Numba-accelerated gradient limiting with Van Leer limiter.
        
        Args:
            gradients: Raw gradients, shape=(n_cells, n_vars, 3)
            row_ptr: CSR row pointers for adjacency
            col_idx: Column indices (neighbors)
            limiter_func: Limiter function (must be Numba-compatible)
            
        Returns:
            Limited gradients, same shape as input
        """
        n_cells = gradients.shape[0]
        n_vars = gradients.shape[1]
        
        limited_grads = gradients.copy()
        
        for i in prange(n_cells):
            start = row_ptr[i]
            end = row_ptr[i + 1]
            
            if start == end:
                continue
            
            for v in range(n_vars):
                grad_center = gradients[i, v, :]
                
                # Compute gradient magnitude at center
                grad_mag_center = 0.0
                for d in range(3):
                    grad_mag_center += grad_center[d] * grad_center[d]
                grad_mag_center = np.sqrt(grad_mag_center)
                
                if grad_mag_center < 1e-10:
                    for d in range(3):
                        limited_grads[i, v, d] = 0.0
                    continue
                
                # Find minimum limiter value across all neighbors
                phi_min = 1.0
                
                for idx in range(start, end):
                    j = col_idx[idx]
                    
                    if j < 0 or j >= n_cells:
                        continue
                    
                    grad_neighbor = gradients[j, v, :]
                    
                    # Compute neighbor gradient magnitude
                    grad_mag_neighbor = 0.0
                    for d in range(3):
                        grad_mag_neighbor += grad_neighbor[d] * grad_neighbor[d]
                    grad_mag_neighbor = np.sqrt(grad_mag_neighbor)
                    
                    # Compute limiter ratio
                    r = grad_mag_neighbor / grad_mag_center
                    phi = limiter_func(r)
                    
                    if phi < phi_min:
                        phi_min = phi
                
                # Apply limiter to gradient
                for d in range(3):
                    limited_grads[i, v, d] *= phi_min
        
        return limited_grads
        
except ImportError:
    logger.warning("Numba not available for gradient limiting, using fallback")
    
    def _gradient_limiting_kernel_numba(
        gradients: np.ndarray,
        row_ptr: np.ndarray,
        col_idx: np.ndarray,
        limiter_func
    ) -> np.ndarray:
        """Fallback gradient limiting without Numba."""
        n_cells = gradients.shape[0]
        n_vars = gradients.shape[1]
        limited_grads = gradients.copy()
        
        for i in range(n_cells):
            start = row_ptr[i]
            end = row_ptr[i + 1]
            
            if start == end:
                continue
            
            for v in range(n_vars):
                grad_center = gradients[i, v, :]
                grad_mag_center = np.linalg.norm(grad_center)
                
                if grad_mag_center < 1e-10:
                    limited_grads[i, v, :] = 0.0
                    continue
                
                phi_min = 1.0
                for idx in range(start, end):
                    j = col_idx[idx]
                    if j < 0 or j >= n_cells:
                        continue
                    
                    grad_mag_neighbor = np.linalg.norm(gradients[j, v, :])
                    r = grad_mag_neighbor / grad_mag_center
                    phi = limiter_func(r)
                    phi_min = min(phi_min, phi)
                
                limited_grads[i, v, :] *= phi_min
        
        return limited_grads
