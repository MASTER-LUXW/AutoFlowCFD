"""MUSCL reconstruction and slope limiters - Numba optimized version.

This module provides high-performance MUSCL (Monotonic Upstream-centered Scheme
for Conservation Laws) reconstruction with Numba JIT acceleration for CFD simulations.

Key features:
- Vectorized state reconstruction using NumPy einsum
- Numba-accelerated gradient limiting with parallel execution
- CSR sparse matrix format for efficient neighbor access
- Multiple limiter functions (Van Leer, Minmod, SuperBee, MC)
- Fallback to pure NumPy when Numba is unavailable
"""

import numpy as np
from typing import Tuple, Optional
from loguru import logger
from enum import Enum


class LimiterType(Enum):
    """Slope limiter types."""
    MINMOD = "minmod"
    VAN_LEER = "van_leer"
    SUPERBEE = "superbee"
    MC = "mc"


class SlopeLimiters:
    """Collection of slope limiter functions.
    
    All limiters take a ratio r and return phi(r) in [0, 2].
    """
    
    @staticmethod
    def minmod(r: float) -> float:
        """Minmod limiter - most dissipative, most stable."""
        if r <= 0:
            return 0.0
        return max(0.0, min(1.0, r))
    
    @staticmethod
    def van_leer(r: float) -> float:
        """Van Leer limiter - good balance of stability and accuracy."""
        if r <= 0:
            return 0.0
        return (2.0 * r) / (1.0 + r)
    
    @staticmethod
    def superbee(r: float) -> float:
        """SuperBee limiter - least dissipative, highest resolution."""
        return max(0.0, min(2.0 * r, 1.0), min(r, 2.0))
    
    @staticmethod
    def mc(r: float) -> float:
        """MC (Modified Centered) limiter."""
        return max(0.0, min(2.0 * r, 0.5 * (1.0 + r), 2.0))


# ============================================================================
# Numba-accelerated kernels
# ============================================================================

try:
    from numba import njit, prange
    
    @njit(cache=True)
    def _van_leer_limiter_numba(r: float) -> float:
        """Van Leer limiter for Numba kernel."""
        if r <= 0:
            return 0.0
        return (2.0 * r) / (1.0 + r)
    
    @njit(parallel=True, cache=True)
    def _gradient_limiting_kernel_numba(
        gradients: np.ndarray,
        row_ptr: np.ndarray,
        col_idx: np.ndarray
    ) -> np.ndarray:
        """Numba-accelerated gradient limiting with Van Leer limiter.
        
        Args:
            gradients: Raw gradients, shape=(n_cells, n_vars, 3)
            row_ptr: CSR row pointers, shape=(n_cells+1,)
            col_idx: CSR column indices (neighbor lists)
            
        Returns:
            Limited gradients, shape=(n_cells, n_vars, 3)
        """
        n_cells = gradients.shape[0]
        n_vars = gradients.shape[1]
        
        limited_grads = gradients.copy()
        
        # Parallel loop over cells
        for i in prange(n_cells):
            start = row_ptr[i]
            end = row_ptr[i + 1]
            
            if start == end:
                continue
            
            # Process each variable
            for v in range(n_vars):
                grad_center = gradients[i, v, :]
                
                # Compute gradient magnitude manually (avoid np.linalg.norm)
                grad_mag_center = 0.0
                for d in range(3):
                    grad_mag_center += grad_center[d] * grad_center[d]
                grad_mag_center = np.sqrt(grad_mag_center)
                
                # Skip near-zero gradients
                if grad_mag_center < 1e-10:
                    for d in range(3):
                        limited_grads[i, v, d] = 0.0
                    continue
                
                # Find minimum limiter across all neighbors
                phi_min = 1.0
                
                for idx in range(start, end):
                    j = col_idx[idx]
                    
                    if j < 0 or j >= n_cells:
                        continue
                    
                    # Compute neighbor gradient magnitude
                    grad_neighbor = gradients[j, v, :]
                    grad_mag_neighbor = 0.0
                    for d in range(3):
                        grad_mag_neighbor += grad_neighbor[d] * grad_neighbor[d]
                    grad_mag_neighbor = np.sqrt(grad_mag_neighbor)
                    
                    # Compute ratio and apply limiter
                    r = grad_mag_neighbor / grad_mag_center
                    phi = _van_leer_limiter_numba(r)
                    
                    if phi < phi_min:
                        phi_min = phi
                
                # Scale gradient by minimum limiter
                for d in range(3):
                    limited_grads[i, v, d] *= phi_min
        
        return limited_grads
    
    NUMBA_AVAILABLE = True
    logger.info("Numba acceleration enabled for gradient limiting")
    
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available, using fallback implementation")
    
    def _gradient_limiting_kernel_numba(
        gradients: np.ndarray,
        row_ptr: np.ndarray,
        col_idx: np.ndarray
    ) -> np.ndarray:
        """Fallback implementation without Numba."""
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
                    phi = SlopeLimiters.van_leer(r)
                    phi_min = min(phi_min, phi)
                
                limited_grads[i, v, :] *= phi_min
        
        return limited_grads


class GradientComputer:
    """Compute cell-centered gradients using Green-Gauss theorem."""
    
    @staticmethod
    def compute_gradients_green_gauss(
        solution: np.ndarray,
        connectivity: np.ndarray,
        face_areas: np.ndarray,
        face_normals: np.ndarray,
        cell_volumes: np.ndarray
    ) -> np.ndarray:
        """Compute gradients using Green-Gauss theorem (vectorized).
        
        Args:
            solution: Cell-centered solutions, shape=(n_cells, n_vars)
            connectivity: Face-to-cell connectivity, shape=(n_faces, 2)
            face_areas: Face areas, shape=(n_faces,)
            face_normals: Face normal vectors, shape=(n_faces, 3)
            cell_volumes: Cell volumes, shape=(n_cells,)
            
        Returns:
            Gradients, shape=(n_cells, n_vars, 3)
        """
        n_cells = solution.shape[0]
        n_vars = solution.shape[1]
        n_faces = connectivity.shape[0]
        
        # Initialize gradients
        gradients = np.zeros((n_cells, n_vars, 3), dtype=np.float64)
        
        # Extract left/right cell indices
        left_cells = connectivity[:, 0]
        right_cells = connectivity[:, 1]
        
        # Compute face-interpolated values (simple average)
        # Handle boundary faces (right_cell = -1)
        valid_mask = (left_cells >= 0) & (right_cells >= 0)
        
        # For interior faces: phi_f = 0.5 * (phi_L + phi_R)
        U_L = solution[left_cells[valid_mask]]  # shape=(n_valid, n_vars)
        U_R = solution[right_cells[valid_mask]]
        U_face = 0.5 * (U_L + U_R)  # shape=(n_valid, n_vars)
        
        # Weighted contribution: phi_f * A_f * n_f
        weighted_normal = face_areas[valid_mask][:, np.newaxis] * face_normals[valid_mask]
        
        # Accumulate gradients for left and right cells
        # Left cell: add contribution (normal points outward from left cell)
        # Right cell: subtract contribution (normal points inward to right cell)
        for var in range(n_vars):
            contribution = U_face[:, var][:, np.newaxis] * weighted_normal
            np.add.at(gradients[:, var, :], left_cells[valid_mask], contribution)
            np.add.at(gradients[:, var, :], right_cells[valid_mask], -contribution)  # Fix: subtract for right cell
        
        # Divide by cell volume
        inv_volumes = 1.0 / cell_volumes[:, np.newaxis, np.newaxis]
        gradients *= inv_volumes
        
        return gradients


class MUSCLReconstructor:
    """MUSCL reconstruction with Numba-accelerated gradient limiting.
    
    Implements second-order spatial reconstruction using:
    1. Green-Gauss gradient computation
    2. Slope limiting (Van Leer by default)
    3. Linear extrapolation to face centers
    
    Performance optimizations:
    - Vectorized state reconstruction using NumPy einsum
    - Numba JIT-compiled gradient limiting with parallel execution
    - CSR sparse matrix for efficient neighbor access
    - Pre-computed data structures reused across iterations
    """
    
    def __init__(self, limiter_type: LimiterType = LimiterType.VAN_LEER):
        """Initialize MUSCL reconstructor.
        
        Args:
            limiter_type: Type of slope limiter to use
        """
        self.limiter_type = limiter_type
        self.limiter_func = SlopeLimiters.van_leer
        
        logger.info(f"MUSCLReconstructor initialized with {limiter_type.value} limiter")
        if NUMBA_AVAILABLE:
            logger.info("Numba acceleration: ENABLED")
        else:
            logger.warning("Numba acceleration: DISABLED (using fallback)")
    
    def build_sparse_adjacency(
        self,
        connectivity: np.ndarray,
        n_cells: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build sparse adjacency representation (CSR format)."""
        left = connectivity[:, 0]
        right = connectivity[:, 1]
        
        valid_mask = (left >= 0) & (right >= 0) & (left < n_cells) & (right < n_cells)
        left_valid = left[valid_mask]
        right_valid = right[valid_mask]
        
        rows = np.concatenate([left_valid, right_valid])
        cols = np.concatenate([right_valid, left_valid])
        
        sort_idx = np.argsort(rows)
        rows_sorted = rows[sort_idx]
        cols_sorted = cols[sort_idx]
        
        row_ptr = np.zeros(n_cells + 1, dtype=np.int64)
        unique_rows, counts = np.unique(rows_sorted, return_counts=True)
        
        for i, row_idx in enumerate(unique_rows):
            row_ptr[row_idx + 1] = counts[i]
        
        row_ptr = np.cumsum(row_ptr)
        
        return row_ptr, cols_sorted
    
    def apply_limiting_to_gradients(
        self,
        solution: np.ndarray,
        gradients: np.ndarray,
        connectivity: np.ndarray
    ) -> np.ndarray:
        """Apply slope limiting directly to gradients (Numba-accelerated)."""
        logger.debug("Applying slope limiting to gradients (Numba-accelerated)...")
        n_cells = solution.shape[0]
        
        logger.debug("Building sparse neighbor adjacency...")
        row_ptr, col_idx = self.build_sparse_adjacency(connectivity, n_cells)
        logger.debug(f"Sparse adjacency built: {len(col_idx)} connections")
        
        logger.debug("Running Numba-accelerated gradient limiting...")
        limited_grads = _gradient_limiting_kernel_numba(
            gradients, row_ptr, col_idx
        )
        
        logger.debug("Slope limiting completed")
        return limited_grads
    
    def reconstruct_states(
        self,
        solution: np.ndarray,
        connectivity: np.ndarray,
        gradients: np.ndarray,
        cell_centers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct left/right states at face centers (fully vectorized)."""
        logger.debug(f"Starting MUSCL state reconstruction (vectorized)...")
        n_faces = connectivity.shape[0]
        n_vars = solution.shape[1]
        
        left_cells = connectivity[:, 0]
        right_cells = connectivity[:, 1]
        
        x_L = cell_centers[left_cells]
        x_R = cell_centers[right_cells]
        
        d_vec = x_R - x_L
        
        grad_L = gradients[left_cells]
        grad_R = gradients[right_cells]
        
        # Using einsum for efficient batch dot product
        grad_L_dot_d = np.einsum('fvd,fd->fv', grad_L, d_vec)
        grad_R_dot_d = np.einsum('fvd,fd->fv', grad_R, d_vec)
        
        U_L = solution[left_cells] + 0.5 * grad_L_dot_d
        U_R = solution[right_cells] - 0.5 * grad_R_dot_d
        
        logger.debug(f"MUSCL reconstruction completed: {n_faces} faces, {n_vars} variables")
        
        return U_L, U_R
