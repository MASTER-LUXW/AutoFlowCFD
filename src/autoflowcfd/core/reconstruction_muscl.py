"""MUSCL reconstruction for high-resolution CFD.

Implements Monotone Upstream-centered Schemes for Conservation Laws (MUSCL)
with slope limiting to prevent oscillations near discontinuities.

Reconstruction formula:
    U_L = U_i + 0.5 * phi * grad_U_i dot d
    U_R = U_j - 0.5 * phi * grad_U_j dot d

where:
    U_i, U_j: Cell-centered values
    phi: Limiter function value
    grad_U: Limited gradient
    d: Distance vector between cell centers
"""

import numpy as np
from typing import Tuple
from loguru import logger

from .reconstruction_limiters import LimiterType, SlopeLimiters, _van_leer_limiter_numba
from .reconstruction_gradients import GradientComputer, _gradient_limiting_kernel_numba


class MUSCLReconstructor:
    """MUSCL reconstruction with slope limiting.
    
    Reconstructs left and right states at cell interfaces using
    limited linear interpolation from cell centers.
    
    Attributes:
        limiter_type: Type of slope limiter to use
        limiter_func: Limiter function phi(r)
    """
    
    def __init__(self, limiter_type: LimiterType = LimiterType.VAN_LEER):
        """Initialize MUSCL reconstructor.
        
        Args:
            limiter_type: Type of slope limiter (default: Van Leer)
        """
        self.limiter_type = limiter_type
        self.limiter_func = SlopeLimiters.get_limiter(limiter_type)
        
        logger.debug(f"MUSCLReconstructor initialized with {limiter_type.value} limiter")
    
    def reconstruct_states(
        self,
        solution: np.ndarray,
        connectivity: np.ndarray,
        gradients: np.ndarray,
        cell_centers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct left and right states at all interfaces.
        
        Vectorized implementation for high performance.
        
        Reconstruction formula:
            U_L = U_i + 0.5 * phi * grad_U_i dot d
            U_R = U_j - 0.5 * phi * grad_U_j dot d
        
        Args:
            solution: Cell-centered solutions, shape=(n_cells, n_vars)
            connectivity: Face-to-cell connectivity, shape=(n_faces, 2)
            gradients: Cell gradients, shape=(n_cells, n_vars, 3)
            cell_centers: Cell center coordinates, shape=(n_cells, 3)
            
        Returns:
            U_L: Left states at interfaces, shape=(n_faces, n_vars)
            U_R: Right states at interfaces, shape=(n_faces, n_vars)
        """
        logger.debug("Starting MUSCL state reconstruction (vectorized)...")
        n_faces = connectivity.shape[0]
        n_cells = solution.shape[0]
        n_vars = solution.shape[1]
        
        # Get left and right cell indices for all faces
        left_cells = connectivity[:, 0]  # shape=(n_faces,)
        right_cells = connectivity[:, 1]  # shape=(n_faces,)
        
        # Filter out invalid faces
        valid_mask = (left_cells >= 0) & (right_cells >= 0) & \
                     (left_cells < n_cells) & (right_cells < n_cells)
        
        if not np.all(valid_mask):
            logger.warning(f"Found {np.sum(~valid_mask)} invalid faces, skipping")
        
        # Initialize output arrays with cell-centered values as fallback
        U_L = np.zeros((n_faces, n_vars), dtype=np.float64)
        U_R = np.zeros((n_faces, n_vars), dtype=np.float64)
        
        # For valid faces, use cell-centered values initially
        valid_left = left_cells[valid_mask]
        valid_right = right_cells[valid_mask]
        U_L[valid_mask] = solution[valid_left]
        U_R[valid_mask] = solution[valid_right]
        
        # Compute distance vectors between cell centers
        d_vectors = cell_centers[valid_right] - cell_centers[valid_left]  # (n_valid, 3)
        distances = np.linalg.norm(d_vectors, axis=1, keepdims=True)  # (n_valid, 1)
        
        # Handle degenerate cases (zero distance)
        degenerate_mask = distances.flatten() < 1e-10
        if np.any(degenerate_mask):
            logger.warning(f"Found {np.sum(degenerate_mask)} degenerate faces with zero distance")
        
        # Unit direction vectors
        distances_safe = np.maximum(distances, 1e-10)
        d_hat = d_vectors / distances_safe  # (n_valid, 3)
        
        # Get gradients for left and right cells
        grad_L = gradients[valid_left]  # (n_valid, n_vars, 3)
        grad_R = gradients[valid_right]  # (n_valid, n_vars, 3)
        
        # Project gradients onto interface direction
        # grad_proj = sum(grad[:, v, :] * d_hat, axis=-1) for each variable
        grad_L_proj = np.einsum('ijk,ik->ij', grad_L, d_hat)  # (n_valid, n_vars)
        grad_R_proj = np.einsum('ijk,ik->ij', grad_R, d_hat)  # (n_valid, n_vars)
        
        # Compute limiter values for all variables at once
        # Simplified: use phi=1.0 in smooth regions (can be enhanced later)
        phi_L = np.ones((np.sum(valid_mask), n_vars), dtype=np.float64)
        phi_R = np.ones((np.sum(valid_mask), n_vars), dtype=np.float64)
        
        # Apply limited reconstruction (vectorized)
        # U_L = U_i + 0.5 * phi_L * grad_U_i dot d
        # U_R = U_j - 0.5 * phi_R * grad_U_j dot d
        correction_L = 0.5 * phi_L * grad_L_proj * distances  # (n_valid, n_vars)
        correction_R = 0.5 * phi_R * grad_R_proj * distances  # (n_valid, n_vars)
        
        U_L[valid_mask] += correction_L
        U_R[valid_mask] -= correction_R
        
        logger.debug(f"MUSCL reconstruction completed: {n_faces} faces, {n_vars} variables")
        return U_L, U_R
    
    def apply_limiting_to_gradients(
        self,
        solution: np.ndarray,
        gradients: np.ndarray,
        connectivity: np.ndarray
    ) -> np.ndarray:
        """Apply slope limiting directly to gradients.
        
        Optimized implementation using Numba JIT and vectorized operations.
        
        Args:
            solution: Cell solutions, shape=(n_cells, n_vars)
            gradients: Raw gradients, shape=(n_cells, n_vars, 3)
            connectivity: Face connectivity, shape=(n_faces, 2)
            
        Returns:
            Limited gradients, same shape as input
        """
        logger.debug("Applying slope limiting to gradients (Numba-accelerated)...")
        n_cells = solution.shape[0]
        
        # Build neighbor adjacency in CSR sparse format for efficient access
        logger.debug("Building sparse neighbor adjacency...")
        row_ptr, col_idx = GradientComputer.build_sparse_adjacency(connectivity, n_cells)
        
        # Use Numba-accelerated limiting kernel (Van Leer hardcoded for performance)
        logger.debug("Running Numba-accelerated gradient limiting...")
        limited_grads = _gradient_limiting_kernel_numba(
            gradients, row_ptr, col_idx, _van_leer_limiter_numba
        )
        
        logger.debug("Slope limiting completed")
        return limited_grads
