"""FVM residual computation with Numba acceleration.

Implements cell-centered residual calculation using HLLC flux solver.
Optimized with Numba JIT for high performance on large meshes.

NOT CURRENTLY USED: FRSolver.solve() (solver_steady.py) computes residuals
via ViscousRANSResidual (fvm_viscous_residual.py) directly - first-order,
inviscid-only, with no MUSCL/viscous/turbulence terms, unlike that path.
FVMResidualComputer is constructed nowhere in the live solve; the Numba
kernel here (_compute_residuals_kernel) never runs. If this is revived
(e.g. as a GPU/Numba fast path), it needs the viscous+SST terms added to
match ViscousRANSResidual before it can replace it.
"""

import numpy as np
from typing import Callable
from loguru import logger

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback for environments without Numba
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

from .fvm_flux import FVMFluxCalculator


@njit(parallel=False)  # Temporarily disable parallel to debug
def _compute_residuals_kernel(solution, face_connectivity, face_normals, 
                               face_areas, cell_volumes, boundary_states,
                               n_faces, n_cells):
    """Numba-accelerated residual computation kernel with numerical stability safeguards.
    
    Args:
        solution: Solution array, shape=(n_cells, 7)
        face_connectivity: Face connectivity, shape=(n_faces, 2)
        face_normals: Face normals, shape=(n_faces, 3)
        face_areas: Face areas, shape=(n_faces,)
        cell_volumes: Cell volumes, shape=(n_cells,)
        boundary_states: Pre-computed ghost cell states for boundary faces, shape=(n_faces, 7)
                        Only used when right_cell < 0
        n_faces: Number of faces
        n_cells: Number of cells
        
    Returns:
        Residual array, shape=(n_cells, 7)
    """
    n_vars = 7
    residuals = np.zeros((n_cells, n_vars), dtype=np.float64)
    gamma = 1.4
    
    # Numerical stability constants
    EPSILON = 1e-10
    MAX_DENSITY = 100.0  # kg/m^3 (air at sea level is ~1.2)
    MIN_PRESSURE = 100.0  # Pa
    MAX_VELOCITY = 500.0  # m/s
    MAX_ENERGY = 1e8  # J
    
    # Loop over all faces
    for face_idx in range(n_faces):
        left_cell = face_connectivity[face_idx, 0]
        right_cell = face_connectivity[face_idx, 1]
        
        # Get left state with safety checks
        rho_L = max(min(solution[left_cell, 0], MAX_DENSITY), EPSILON)
        rhou_L = solution[left_cell, 1]
        rhov_L = solution[left_cell, 2]
        rhow_L = solution[left_cell, 3]
        E_L = min(max(solution[left_cell, 4], 0.0), MAX_ENERGY)
        k_L = max(solution[left_cell, 5], 0.0)
        omega_L = max(solution[left_cell, 6], 0.0)
        
        # Get right state (for boundary faces, use pre-computed ghost cell state)
        if right_cell >= 0:
            rho_R = max(min(solution[right_cell, 0], MAX_DENSITY), EPSILON)
            rhou_R = solution[right_cell, 1]
            rhov_R = solution[right_cell, 2]
            rhow_R = solution[right_cell, 3]
            E_R = min(max(solution[right_cell, 4], 0.0), MAX_ENERGY)
            k_R = max(solution[right_cell, 5], 0.0)
            omega_R = max(solution[right_cell, 6], 0.0)
        else:
            # For boundary faces, use pre-computed ghost cell state
            rho_R = max(min(boundary_states[face_idx, 0], MAX_DENSITY), EPSILON)
            rhou_R = boundary_states[face_idx, 1]
            rhov_R = boundary_states[face_idx, 2]
            rhow_R = boundary_states[face_idx, 3]
            E_R = min(max(boundary_states[face_idx, 4], 0.0), MAX_ENERGY)
            k_R = max(boundary_states[face_idx, 5], 0.0)
            omega_R = max(boundary_states[face_idx, 6], 0.0)
        
        # Primitive variables with velocity limiting
        u_L = max(min(rhou_L / rho_L, MAX_VELOCITY), -MAX_VELOCITY)
        v_L = max(min(rhov_L / rho_L, MAX_VELOCITY), -MAX_VELOCITY)
        w_L = max(min(rhow_L / rho_L, MAX_VELOCITY), -MAX_VELOCITY)
        
        u_R = max(min(rhou_R / rho_R, MAX_VELOCITY), -MAX_VELOCITY)
        v_R = max(min(rhov_R / rho_R, MAX_VELOCITY), -MAX_VELOCITY)
        w_R = max(min(rhow_R / rho_R, MAX_VELOCITY), -MAX_VELOCITY)
        
        # Pressure calculation with clamping
        kinetic_L = 0.5 * rho_L * (u_L**2 + v_L**2 + w_L**2)
        kinetic_R = 0.5 * rho_R * (u_R**2 + v_R**2 + w_R**2)
        
        p_L = max((gamma - 1.0) * (E_L - kinetic_L), MIN_PRESSURE)
        p_R = max((gamma - 1.0) * (E_R - kinetic_R), MIN_PRESSURE)
        
        # Speed of sound
        a_L = np.sqrt(gamma * p_L / rho_L)
        a_R = np.sqrt(gamma * p_R / rho_R)
        
        # Normal velocity
        normal_x = face_normals[face_idx, 0]
        normal_y = face_normals[face_idx, 1]
        normal_z = face_normals[face_idx, 2]
        
        u_n_L = u_L * normal_x + v_L * normal_y + w_L * normal_z
        u_n_R = u_R * normal_x + v_R * normal_y + w_R * normal_z
        
        # HLLC wave speeds
        S_L = min(u_n_L - a_L, u_n_R - a_R)
        S_R = max(u_n_L + a_L, u_n_R + a_R)
        
        # Contact speed with division-by-zero protection
        denom = rho_L * (S_L - u_n_L) - rho_R * (S_R - u_n_R)
        if abs(denom) > EPSILON:
            S_star = (p_R - p_L + rho_L * u_n_L * (S_L - u_n_L) - 
                     rho_R * u_n_R * (S_R - u_n_R)) / denom
        else:
            S_star = 0.5 * (u_n_L + u_n_R)
        
        # Limit S_star to reasonable range
        S_star = max(min(S_star, MAX_VELOCITY), -MAX_VELOCITY)
        
        # Compute flux using HLLC
        area = face_areas[face_idx]
        flux = np.zeros(7)
        
        if S_L >= 0:
            # Left flux
            u_n = u_n_L
            flux[0] = rho_L * u_n
            flux[1] = rho_L * u_L * u_n + p_L * normal_x
            flux[2] = rho_L * v_L * u_n + p_L * normal_y
            flux[3] = rho_L * w_L * u_n + p_L * normal_z
            flux[4] = (E_L + p_L) * u_n
            flux[5] = k_L * u_n
            flux[6] = omega_L * u_n
        elif S_R <= 0:
            # Right flux
            u_n = u_n_R
            flux[0] = rho_R * u_n
            flux[1] = rho_R * u_R * u_n + p_R * normal_x
            flux[2] = rho_R * v_R * u_n + p_R * normal_y
            flux[3] = rho_R * w_R * u_n + p_R * normal_z
            flux[4] = (E_R + p_R) * u_n
            flux[5] = k_R * u_n
            flux[6] = omega_R * u_n
        else:
            # Star region flux
            if S_star >= 0:
                # Left star state
                denom_star = S_L - S_star
                if abs(denom_star) > EPSILON:
                    rho_star = rho_L * (S_L - u_n_L) / denom_star
                    rho_star = max(min(rho_star, MAX_DENSITY), EPSILON)
                    
                    u_star = S_star * normal_x + (u_L - u_n_L * normal_x)
                    v_star = S_star * normal_y + (v_L - u_n_L * normal_y)
                    w_star = S_star * normal_z + (w_L - u_n_L * normal_z)
                    
                    # Limit star velocities
                    u_star = max(min(u_star, MAX_VELOCITY), -MAX_VELOCITY)
                    v_star = max(min(v_star, MAX_VELOCITY), -MAX_VELOCITY)
                    w_star = max(min(w_star, MAX_VELOCITY), -MAX_VELOCITY)
                    
                    p_star = p_L + rho_L * (S_L - u_n_L) * (S_star - u_n_L)
                    p_star = max(p_star, MIN_PRESSURE)
                    
                    u_n_star = u_star * normal_x + v_star * normal_y + w_star * normal_z
                    
                    flux[0] = rho_star * u_n_star
                    flux[1] = rho_star * u_star * u_n_star + p_star * normal_x
                    flux[2] = rho_star * v_star * u_n_star + p_star * normal_y
                    flux[3] = rho_star * w_star * u_n_star + p_star * normal_z
                    
                    # Total energy flux with safety check
                    e_star = p_star / ((gamma - 1.0) * rho_star)
                    kinetic_star = 0.5 * (u_star**2 + v_star**2 + w_star**2)
                    flux[4] = rho_star * (e_star + kinetic_star + p_star / rho_star) * u_n_star
                    
                    flux[5] = k_L * (S_L - u_n_L) / denom_star * u_n_star
                    flux[6] = omega_L * (S_L - u_n_L) / denom_star * u_n_star
                else:
                    # Fallback to left state
                    flux[0] = rho_L * u_n_L
                    flux[1] = rho_L * u_L * u_n_L + p_L * normal_x
                    flux[2] = rho_L * v_L * u_n_L + p_L * normal_y
                    flux[3] = rho_L * w_L * u_n_L + p_L * normal_z
                    flux[4] = (E_L + p_L) * u_n_L
                    flux[5] = k_L * u_n_L
                    flux[6] = omega_L * u_n_L
            else:
                # Right star state
                denom_star = S_R - S_star
                if abs(denom_star) > EPSILON:
                    rho_star = rho_R * (S_R - u_n_R) / denom_star
                    rho_star = max(min(rho_star, MAX_DENSITY), EPSILON)
                    
                    u_star = S_star * normal_x + (u_R - u_n_R * normal_x)
                    v_star = S_star * normal_y + (v_R - u_n_R * normal_y)
                    w_star = S_star * normal_z + (w_R - u_n_R * normal_z)
                    
                    # Limit star velocities
                    u_star = max(min(u_star, MAX_VELOCITY), -MAX_VELOCITY)
                    v_star = max(min(v_star, MAX_VELOCITY), -MAX_VELOCITY)
                    w_star = max(min(w_star, MAX_VELOCITY), -MAX_VELOCITY)
                    
                    p_star = p_R + rho_R * (S_R - u_n_R) * (S_star - u_n_R)
                    p_star = max(p_star, MIN_PRESSURE)
                    
                    u_n_star = u_star * normal_x + v_star * normal_y + w_star * normal_z
                    
                    flux[0] = rho_star * u_n_star
                    flux[1] = rho_star * u_star * u_n_star + p_star * normal_x
                    flux[2] = rho_star * v_star * u_n_star + p_star * normal_y
                    flux[3] = rho_star * w_star * u_n_star + p_star * normal_z
                    
                    # Total energy flux with safety check
                    e_star = p_star / ((gamma - 1.0) * rho_star)
                    kinetic_star = 0.5 * (u_star**2 + v_star**2 + w_star**2)
                    flux[4] = rho_star * (e_star + kinetic_star + p_star / rho_star) * u_n_star
                    
                    flux[5] = k_R * (S_R - u_n_R) / denom_star * u_n_star
                    flux[6] = omega_R * (S_R - u_n_R) / denom_star * u_n_star
                else:
                    # Fallback to right state
                    flux[0] = rho_R * u_n_R
                    flux[1] = rho_R * u_R * u_n_R + p_R * normal_x
                    flux[2] = rho_R * v_R * u_n_R + p_R * normal_y
                    flux[3] = rho_R * w_R * u_n_R + p_R * normal_z
                    flux[4] = (E_R + p_R) * u_n_R
                    flux[5] = k_R * u_n_R
                    flux[6] = omega_R * u_n_R
        
        flux *= area
        
        # Clip flux to prevent overflow
        for i in range(n_vars):
            flux[i] = max(min(flux[i], 1e10), -1e10)
        
        # Accumulate residuals
        residuals[left_cell, 0] -= flux[0]
        residuals[left_cell, 1] -= flux[1]
        residuals[left_cell, 2] -= flux[2]
        residuals[left_cell, 3] -= flux[3]
        residuals[left_cell, 4] -= flux[4]
        residuals[left_cell, 5] -= flux[5]
        residuals[left_cell, 6] -= flux[6]
        
        if right_cell >= 0:
            residuals[right_cell, 0] += flux[0]
            residuals[right_cell, 1] += flux[1]
            residuals[right_cell, 2] += flux[2]
            residuals[right_cell, 3] += flux[3]
            residuals[right_cell, 4] += flux[4]
            residuals[right_cell, 5] += flux[5]
            residuals[right_cell, 6] += flux[6]
    
    # Normalize by volume with additional safety
    for cell_idx in range(n_cells):
        vol = max(cell_volumes[cell_idx], 1e-15)
        for v in range(n_vars):
            res_val = residuals[cell_idx, v] / vol
            # Clip residuals to prevent explosion
            residuals[cell_idx, v] = max(min(res_val, 1e8), -1e8)
    
    return residuals


class FVMResidualComputer:
    """Computes residuals for all cells using FVM formulation."""
    
    def __init__(self, flux_calculator: FVMFluxCalculator):
        self.flux_calculator = flux_calculator
    
    def compute_residuals(self, solution: np.ndarray, 
                         face_connectivity: np.ndarray,
                         face_normals: np.ndarray,
                         face_areas: np.ndarray,
                         boundary_flags: np.ndarray,
                         cell_volumes: np.ndarray,
                         apply_bc_func) -> np.ndarray:
        """Compute residuals for all cells using Numba-accelerated kernel.
        
        Args:
            solution: Solution array, shape=(n_cells, 7)
            face_connectivity: Face connectivity, shape=(n_faces, 2)
            face_normals: Face normals, shape=(n_faces, 3)
            face_areas: Face areas, shape=(n_faces,)
            boundary_flags: Boundary flags, shape=(n_faces,)
            cell_volumes: Cell volumes, shape=(n_cells,)
            apply_bc_func: Function to apply boundary conditions
            
        Returns:
            Residual array, shape=(n_cells, 7)
        """
        n_cells = len(solution)
        n_faces = len(face_connectivity)
        
        # Pre-compute boundary ghost cell states for all boundary faces
        boundary_states = np.zeros((n_faces, 7), dtype=np.float64)
        n_boundary_faces = 0
        
        for face_idx in range(n_faces):
            right_cell = face_connectivity[face_idx, 1]
            if right_cell < 0:
                # This is a boundary face, compute ghost cell state
                left_cell = face_connectivity[face_idx, 0]
                boundary_states[face_idx] = apply_bc_func(left_cell, face_idx)
                n_boundary_faces += 1
        
        logger.debug(f"Pre-computed {n_boundary_faces} boundary ghost cell states")
        
        if NUMBA_AVAILABLE:
            # Use Numba-accelerated kernel with pre-computed boundary states
            logger.debug(f"Computing residuals with Numba (n_faces={n_faces}, n_cells={n_cells})")
            residuals = _compute_residuals_kernel(
                solution, face_connectivity, face_normals, 
                face_areas, cell_volumes, boundary_states,
                n_faces, n_cells
            )
        else:
            # Fallback to Python loop (slow)
            logger.warning("Numba not available, using slow Python loop for residuals")
            n_vars = 7
            residuals = np.zeros((n_cells, n_vars), dtype=np.float64)
            
            for face_idx in range(n_faces):
                left_cell = face_connectivity[face_idx, 0]
                right_cell = face_connectivity[face_idx, 1]
                
                U_left = solution[left_cell].copy()
                
                if right_cell >= 0:
                    U_right = solution[right_cell].copy()
                else:
                    U_right = apply_bc_func(left_cell, face_idx)
                
                normal = face_normals[face_idx]
                area = face_areas[face_idx]
                
                flux = self.flux_calculator.compute_flux(U_left, U_right, normal, area)
                
                residuals[left_cell] -= flux
                
                if right_cell >= 0:
                    residuals[right_cell] += flux
            
            # Normalize by volume
            for cell_idx in range(n_cells):
                vol = max(cell_volumes[cell_idx], 1e-15)
                residuals[cell_idx] /= vol
        
        return residuals
