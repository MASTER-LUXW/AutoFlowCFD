"""Boundary condition handler for FVM solver.

This module handles boundary condition application for different boundary types
in the finite volume method solver.

Key Components:
    - BoundaryConditionHandler: Applies BCs to ghost cells
"""

import numpy as np
from typing import Dict
from loguru import logger


class BoundaryConditionHandler:
    """Handles boundary condition application for FVM solver."""
    
    def __init__(self, grid_data, face_extractor):
        self.grid_data = grid_data
        self.face_extractor = face_extractor
        
        # Ramp mechanism for smooth velocity transition
        self.ramp_factor = 0.0  # Start from 0, will increase to 1.0
        self.base_inlet_velocity = 30.0  # Base inlet velocity (m/s)
        self.base_farfield_velocity = 30.0  # Base farfield velocity (m/s)
        self.ramp_iterations = 0  # Will be set during solve
    
    def update_ramp_factor(self, iteration: int, max_iter: int):
        """Update ramp factor for smooth velocity transition.
        
        Gradually increases the inlet/farfield velocity from 0 to full value over
        the first 20% of iterations to avoid numerical instability.
        
        Args:
            iteration: Current iteration number
            max_iter: Maximum iterations
        """
        self.ramp_iterations = max(10, int(max_iter * 0.2))  # At least 10 iterations, or 20% of total
        
        if iteration <= self.ramp_iterations:
            # Linear ramp from 0 to 1
            self.ramp_factor = iteration / self.ramp_iterations
        else:
            self.ramp_factor = 1.0
        
        return self.ramp_factor
    
    def get_current_inlet_velocity(self) -> float:
        """Get current inlet velocity with ramp applied."""
        return self.base_inlet_velocity * self.ramp_factor
    
    def get_current_farfield_velocity(self) -> float:
        """Get current farfield velocity with ramp applied."""
        return self.base_farfield_velocity * self.ramp_factor

    def apply_boundary_condition(self, solution: np.ndarray, 
                                 cell_idx: int, face_idx: int) -> np.ndarray:
        """Apply boundary condition for a boundary face.
        
        Args:
            solution: Solution array
            cell_idx: Interior cell index
            face_idx: Boundary face index
            
        Returns:
            Ghost cell conservative variables
        """
        U_interior = solution[cell_idx].copy()
        rho, rhou, rhov, rhow, E, k, omega = U_interior
        
        u = rhou / max(rho, 1e-10)
        v = rhov / max(rho, 1e-10)
        w = rhow / max(rho, 1e-10)
        
        gamma = 1.4
        p = (gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2 + w**2))
        p = max(p, 100.0)
        
        normal = self.face_extractor.face_normals[face_idx]
        boundary_type = self._get_face_boundary_type(face_idx)
        
        if boundary_type in ["WALL", "GROUND"]:
            return self._wall_bc(rho, u, v, w, p, k, omega, normal, boundary_type)
        elif boundary_type in ["INLET", "FARFIELD"]:
            return self._inlet_farfield_bc()
        elif boundary_type == "OUTLET":
            return self._outlet_bc(rho, u, v, w, p, k, omega)
        elif boundary_type == "SYMMETRY":
            return self._symmetry_bc(rho, u, v, w, E, k, omega, normal)
        else:
            return U_interior.copy()
    
    def _wall_bc(self, rho: float, u: float, v: float, w: float,
                p: float, k: float, omega: float, 
                normal: np.ndarray, wall_type: str) -> np.ndarray:
        """Wall boundary condition (no-slip)."""
        u_n = u * normal[0] + v * normal[1] + w * normal[2]
        
        u_ghost = u - 2.0 * u_n * normal[0]
        v_ghost = v - 2.0 * u_n * normal[1]
        w_ghost = w - 2.0 * u_n * normal[2]
        
        # Moving ground with ramp factor
        if wall_type == "GROUND":
            ground_velocity = self.get_current_farfield_velocity()
            u_ghost += ground_velocity
        
        gamma = 1.4
        rho_ghost = rho
        rhou_ghost = rho_ghost * u_ghost
        rhov_ghost = rho_ghost * v_ghost
        rhow_ghost = rho_ghost * w_ghost
        E_ghost = p / (gamma - 1.0) + 0.5 * rho_ghost * (u_ghost**2 + v_ghost**2 + w_ghost**2)
        
        return np.array([rho_ghost, rhou_ghost, rhov_ghost, rhow_ghost, 
                        E_ghost, k, omega])
    
    def _inlet_farfield_bc(self) -> np.ndarray:
        """Inlet/farfield boundary condition with ramp factor."""
        gamma = 1.4
        rho_inf = 1.225
        
        # Apply ramp factor to velocity
        u_inf = self.get_current_inlet_velocity()
        p_inf = 101325.0
        
        rhou_inf = rho_inf * u_inf
        E_inf = p_inf / (gamma - 1.0) + 0.5 * rho_inf * u_inf**2
        
        return np.array([rho_inf, rhou_inf, 0.0, 0.0, E_inf, 0.001, 1.0])
    
    def _outlet_bc(self, rho: float, u: float, v: float, w: float,
                  p: float, k: float, omega: float) -> np.ndarray:
        """Outlet boundary condition (pressure specified)."""
        gamma = 1.4
        p_outlet = 101325.0
        
        rhou = rho * u
        rhov = rho * v
        rhow = rho * w
        E = p_outlet / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        
        return np.array([rho, rhou, rhov, rhow, E, k, omega])
    
    def _symmetry_bc(self, rho: float, u: float, v: float, w: float,
                    E: float, k: float, omega: float, normal: np.ndarray) -> np.ndarray:
        """Symmetry boundary condition."""
        u_n = u * normal[0] + v * normal[1] + w * normal[2]
        
        u_ghost = u - 2.0 * u_n * normal[0]
        v_ghost = v - 2.0 * u_n * normal[1]
        w_ghost = w - 2.0 * u_n * normal[2]
        
        rhou_ghost = rho * u_ghost
        rhov_ghost = rho * v_ghost
        rhow_ghost = rho * w_ghost
        
        return np.array([rho, rhou_ghost, rhov_ghost, rhow_ghost, E, k, omega])
    
    def _get_face_boundary_type(self, face_idx: int) -> str:
        """Get boundary type for a face."""
        if not self.face_extractor.boundary_flags[face_idx]:
            return "INTERIOR"
        
        left_cell = self.face_extractor.face_connectivity[face_idx, 0]
        
        for boundary_name in self.grid_data.boundaries.boundary_names:
            boundary_cells = self.grid_data.boundaries.get_cell_indices(boundary_name)
            if left_cell in boundary_cells:
                name_upper = boundary_name.upper()
                if "BODY" in name_upper or "CAR" in name_upper:
                    return "WALL"
                elif "GROUND" in name_upper:
                    return "GROUND"
                elif "INLET" in name_upper:
                    return "INLET"
                elif "OUTLET" in name_upper:
                    return "OUTLET"
                elif "SYMMETRY" in name_upper:
                    return "SYMMETRY"
                elif "TUNNEL" in name_upper or "FARFIELD" in name_upper:
                    return "FARFIELD"
        
        return "WALL"
