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
    
    def __init__(self, grid_data, face_extractor, rho_inf: float = 1.225, p_inf: float = 101325.0):
        self.grid_data = grid_data
        self.face_extractor = face_extractor

        # Thermodynamic constants
        self.gamma = 1.4  # Ratio of specific heats for air

        # Freestream reference conditions (shared with FRSolver._initialize_solution
        # and AeroCoefficientCalculator via SteadyConfig - single source of truth).
        self.rho_inf = rho_inf
        self.p_inf = p_inf

        # Ramp mechanism for smooth velocity transition
        self.ramp_factor = 0.0  # Start from 0, will increase to 1.0
        self.base_inlet_velocity = 30.0  # Base inlet velocity (m/s); overwritten by FRSolver from config
        self.base_farfield_velocity = 30.0  # Base farfield velocity (m/s); overwritten by FRSolver from config
        self.ramp_iterations = 0  # Will be set during solve
        # Cached boundary-face -> type map (built lazily).
        self._face_types = None
    
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
        rho, rhou, rhov, rhow, E, rhok, rhow_sst = U_interior

        u = rhou / max(rho, 1e-10)
        v = rhov / max(rho, 1e-10)
        w = rhow / max(rho, 1e-10)
        # Turbulence variables are carried in conservative form (rho*k, rho*omega).
        k = rhok / max(rho, 1e-10)
        omega = rhow_sst / max(rho, 1e-10)

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
        """Viscous no-slip wall boundary condition.

        The ghost state mirrors the *full* interior velocity (not just its
        normal component) so that the face-interpolated velocity 0.5*(u_i+u_g)
        equals the prescribed wall velocity.  For a stationary wall that target
        is zero; for a moving ground it is the belt speed in +x.  This is what
        makes the viscous shear stress at the wall non-zero, which is the
        physical origin of skin-friction drag.
        """
        # Target wall velocity.
        u_wall, v_wall, w_wall = 0.0, 0.0, 0.0
        if wall_type == "GROUND":
            u_wall = self.get_current_farfield_velocity()

        # Ghost = 2*wall - interior  => average = wall.
        u_ghost = 2.0 * u_wall - u
        v_ghost = 2.0 * v_wall - v
        w_ghost = 2.0 * w_wall - w

        gamma = 1.4
        rho_ghost = rho
        rhou_ghost = rho_ghost * u_ghost
        rhov_ghost = rho_ghost * v_ghost
        rhow_ghost = rho_ghost * w_ghost
        # Zero-gradient pressure at the wall (dp/dn = 0).
        E_ghost = p / (gamma - 1.0) + 0.5 * rho_ghost * (u_ghost**2 + v_ghost**2 + w_ghost**2)

        # Turbulence (conservative): k -> 0 at the wall. NOTE: this used to be
        # mirrored as -rho*k (intending a zero face-average), but every
        # consumer decodes ghost states through to_primitive(), which clamps
        # k = max(rho_k/rho, 0.0) - the negative mirror value was always
        # floored back to 0 before use, so the actual boundary value is (and
        # always effectively was) a direct k=0 Dirichlet ghost, not a mirror.
        # omega is convected from interior (zero-gradient), not mirrored.
        rhok_ghost = 0.0
        rhow_ghost_sst = rho_ghost * omega
        return np.array([rho_ghost, rhou_ghost, rhov_ghost, rhow_ghost,
                        E_ghost, rhok_ghost, rhow_ghost_sst])
    
    def _inlet_farfield_bc(self) -> np.ndarray:
        """Inlet/farfield boundary condition with ramp factor.

        Freestream turbulence (conservative form): a low k and a moderate omega
        giving a small free-stream eddy viscosity.
        """
        gamma = 1.4
        rho_inf = self.rho_inf

        # Apply ramp factor to velocity
        u_inf = self.get_current_inlet_velocity()
        p_inf = self.p_inf

        rhou_inf = rho_inf * u_inf
        E_inf = p_inf / (gamma - 1.0) + 0.5 * rho_inf * u_inf**2

        # k_inf, omega_inf as primitive -> convert to conservative.
        k_inf = 1.5 * (0.01 * max(u_inf, 1.0))**2   # 1% turbulence intensity
        omega_inf = 5.0 * max(u_inf, 1.0) / 0.1     # length scale ~0.1 m
        return np.array([rho_inf, rhou_inf, 0.0, 0.0, E_inf,
                        rho_inf * k_inf, rho_inf * omega_inf])

    def _outlet_bc(self, rho: float, u: float, v: float, w: float,
                  p: float, k: float, omega: float) -> np.ndarray:
        """Outlet boundary condition (static pressure specified, rest extrapolated)."""
        gamma = 1.4
        p_outlet = self.p_inf

        rhou = rho * u
        rhov = rho * v
        rhow = rho * w
        E = p_outlet / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)

        return np.array([rho, rhou, rhov, rhow, E, rho * k, rho * omega])

    def _symmetry_bc(self, rho: float, u: float, v: float, w: float,
                    E: float, k: float, omega: float, normal: np.ndarray) -> np.ndarray:
        """Symmetry boundary condition (mirror normal velocity)."""
        u_n = u * normal[0] + v * normal[1] + w * normal[2]

        u_ghost = u - 2.0 * u_n * normal[0]
        v_ghost = v - 2.0 * u_n * normal[1]
        w_ghost = w - 2.0 * u_n * normal[2]

        rhou_ghost = rho * u_ghost
        rhov_ghost = rho * v_ghost
        rhow_ghost = rho * w_ghost

        return np.array([rho, rhou_ghost, rhov_ghost, rhow_ghost, E,
                        rho * k, rho * omega])
    
    def _get_face_boundary_type(self, face_idx: int) -> str:
        """Get boundary type for a face (cached O(1) lookup)."""
        if not self.face_extractor.boundary_flags[face_idx]:
            return "INTERIOR"
        if self._face_types is None:
            self._precompute_face_types()
        return self._face_types.get(int(face_idx), "WALL")

    @staticmethod
    def _classify(name_upper: str) -> str:
        if "BODY" in name_upper or "CAR" in name_upper:
            return "WALL"
        elif "GROUND" in name_upper:
            return "GROUND"
        elif "INLET" in name_upper or "INFLOW" in name_upper:
            return "INLET"
        elif "OUTLET" in name_upper:
            return "OUTLET"
        elif "SYMMETRY" in name_upper:
            return "SYMMETRY"
        elif "TUNNEL" in name_upper or "FARFIELD" in name_upper:
            return "FARFIELD"
        return "WALL"

    def _precompute_face_types(self) -> None:
        """Build a face_idx -> boundary-type map once (was O(N^2) per call).

        A cell -> type lookup is built from the boundary groups, then applied to
        each boundary face via its owner cell.  This removes the per-face,
        per-iteration linear scan over every boundary group.
        """
        cell_type: Dict[int, str] = {}
        for boundary_name in self.grid_data.boundaries.boundary_names:
            btype = self._classify(boundary_name.upper())
            for c in self.grid_data.boundaries.get_cell_indices(boundary_name):
                cell_type[int(c)] = btype

        flags = self.face_extractor.boundary_flags
        conn = self.face_extractor.face_connectivity
        self._face_types = {}
        for face_idx in np.where(flags)[0]:
            owner = int(conn[face_idx, 0])
            self._face_types[int(face_idx)] = cell_type.get(owner, "WALL")

    def build_boundary_states(self, solution: np.ndarray) -> np.ndarray:
        """Return ghost conservative states for every face (n_faces, 7).

        Interior-face rows are left as zeros (unused by the residual); boundary
        rows hold the ghost state from :meth:`apply_boundary_condition`.
        
        OPTIMIZED: Vectorized implementation replacing Python loop over boundary faces.
        Processes all boundary faces grouped by type for maximum performance.
        """
        n_faces = len(self.face_extractor.boundary_flags)
        states = np.zeros((n_faces, 7), dtype=np.float64)
        
        # Get all boundary face indices at once
        bface_mask = self.face_extractor.boundary_flags
        if not np.any(bface_mask):
            return states
        
        bfaces = np.where(bface_mask)[0]
        n_bfaces = len(bfaces)
        
        # Batch extract owner cell indices for all boundary faces
        owner_indices = self.face_extractor.face_connectivity[bfaces, 0].astype(np.int32)
        
        # Batch extract interior conservative states for owner cells
        U_interior = solution[owner_indices]  # (n_bfaces, 7)
        
        # Decompose to primitive variables for all boundary faces at once
        rho = np.maximum(U_interior[:, 0], 1e-9)
        vel = U_interior[:, 1:4] / rho[:, None]  # (n_bfaces, 3)
        u, v, w = vel[:, 0], vel[:, 1], vel[:, 2]
        ke = 0.5 * rho * np.sum(vel**2, axis=1)
        p = np.maximum((self.gamma - 1.0) * (U_interior[:, 4] - ke), 100.0)
        k = np.maximum(U_interior[:, 5] / rho, 0.0)
        omega = np.maximum(U_interior[:, 6] / rho, 1e-6)
        
        # Get face normals for all boundary faces
        normals = self.face_extractor.face_normals[bfaces]  # (n_bfaces, 3)
        
        # Get boundary types for all faces (vectorized lookup)
        if self._face_types is None:
            self._precompute_face_types()
        
        # Create mapping from face index to array position
        face_to_idx = {int(f): i for i, f in enumerate(bfaces)}
        btypes = np.array([self._face_types.get(int(f), "WALL") for f in bfaces])
        
        # Process each boundary type separately (vectorized within each group)
        unique_types = np.unique(btypes)
        
        for btype in unique_types:
            type_mask = (btypes == btype)
            if not np.any(type_mask):
                continue
            
            # Indices within bfaces array for this boundary type
            type_indices_in_bfaces = np.where(type_mask)[0]
            # Actual face indices
            type_face_indices = bfaces[type_indices_in_bfaces]
            
            # Extract data for this boundary type
            rho_t = rho[type_mask]
            u_t, v_t, w_t = u[type_mask], v[type_mask], w[type_mask]
            p_t = p[type_mask]
            k_t = k[type_mask]
            omega_t = omega[type_mask]
            normals_t = normals[type_mask]
            U_int_t = U_interior[type_indices_in_bfaces]
            
            # Apply boundary condition based on type
            if btype in ["WALL", "GROUND"]:
                ghost_states = self._wall_bc_vectorized(
                    rho_t, u_t, v_t, w_t, p_t, k_t, omega_t, 
                    normals_t, btype
                )
            elif btype in ["INLET", "FARFIELD"]:
                # Inlet/farfield uses fixed freestream values
                n_type = np.sum(type_mask)
                ghost_states = np.tile(self._inlet_farfield_bc(), (n_type, 1))
            elif btype == "OUTLET":
                ghost_states = self._outlet_bc_vectorized(
                    rho_t, u_t, v_t, w_t, p_t, k_t, omega_t
                )
            elif btype == "SYMMETRY":
                # Need total energy E for symmetry BC
                E_t = U_int_t[:, 4]
                ghost_states = self._symmetry_bc_vectorized(
                    rho_t, u_t, v_t, w_t, E_t, k_t, omega_t, normals_t
                )
            else:
                # Default: copy interior state
                ghost_states = U_int_t.copy()
            
            # Assign ghost states to output array
            states[type_face_indices] = ghost_states
        
        return states
    
    def _wall_bc_vectorized(self, rho: np.ndarray, u: np.ndarray, v: np.ndarray,
                           w: np.ndarray, p: np.ndarray, k: np.ndarray,
                           omega: np.ndarray, normals: np.ndarray,
                           wall_type: str = "WALL") -> np.ndarray:
        """Vectorized wall boundary condition with numerical stability protection."""
        gamma = self.gamma
        
        # === NUMERICAL STABILITY: Clip velocity to prevent blow-up ===
        MAX_VELOCITY = 1e4  # 10 km/s physical upper bound
        vel_mag = np.sqrt(u**2 + v**2 + w**2)
        clip_factor = np.minimum(1.0, MAX_VELOCITY / np.maximum(vel_mag, 1e-12))
        
        u = u * clip_factor
        v = v * clip_factor
        w = w * clip_factor
        
        # Recompute kinetic energy with clipped velocities
        ke = 0.5 * rho * (u**2 + v**2 + w**2)
        
        # Ensure pressure positivity and reasonable bounds
        p = np.maximum(p, 100.0)
        p = np.minimum(p, 1e8)  # Prevent extreme pressure
        
        # Target wall velocity
        u_wall, v_wall, w_wall = 0.0, 0.0, 0.0
        if wall_type == "GROUND":
            u_wall = self.get_current_farfield_velocity()
        
        # Ghost = 2*wall - interior (mirror reflection)
        u_ghost = 2.0 * u_wall - u
        v_ghost = 2.0 * v_wall - v
        w_ghost = 2.0 * w_wall - w
        
        # Clip ghost velocities as well
        vel_ghost_mag = np.sqrt(u_ghost**2 + v_ghost**2 + w_ghost**2)
        ghost_clip = np.minimum(1.0, MAX_VELOCITY / np.maximum(vel_ghost_mag, 1e-12))
        u_ghost *= ghost_clip
        v_ghost *= ghost_clip
        w_ghost *= ghost_clip
        
        rho_ghost = rho
        rhou_ghost = rho_ghost * u_ghost
        rhov_ghost = rho_ghost * v_ghost
        rhow_ghost = rho_ghost * w_ghost
        
        # Compute ghost energy with clipped values
        E_ghost = p / (gamma - 1.0) + 0.5 * rho_ghost * (u_ghost**2 + v_ghost**2 + w_ghost**2)
        
        # Turbulence: k -> 0 at wall (direct Dirichlet ghost - see the scalar
        # _wall_bc for why mirroring doesn't survive the to_primitive clamp),
        # omega extrapolated from interior.
        rhok_ghost = np.zeros_like(rho_ghost)
        rhow_ghost_sst = rho_ghost * omega
        
        return np.column_stack([
            rho_ghost, rhou_ghost, rhov_ghost, rhow_ghost,
            E_ghost, rhok_ghost, rhow_ghost_sst
        ])
    
    def _outlet_bc_vectorized(self, rho: np.ndarray, u: np.ndarray, v: np.ndarray,
                             w: np.ndarray, p: np.ndarray, k: np.ndarray,
                             omega: np.ndarray) -> np.ndarray:
        """Vectorized outlet boundary condition."""
        gamma = self.gamma
        p_outlet = self.p_inf
        
        rhou = rho * u
        rhov = rho * v
        rhow = rho * w
        E = p_outlet / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        
        return np.column_stack([rho, rhou, rhov, rhow, E, rho * k, rho * omega])
    
    def _symmetry_bc_vectorized(self, rho: np.ndarray, u: np.ndarray, v: np.ndarray,
                               w: np.ndarray, E: np.ndarray, k: np.ndarray,
                               omega: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Vectorized symmetry boundary condition."""
        # Normal velocity component
        u_n = u * normal[:, 0] + v * normal[:, 1] + w * normal[:, 2]
        
        # Mirror normal velocity: u_ghost = u - 2*u_n*n
        u_ghost = u - 2.0 * u_n * normal[:, 0]
        v_ghost = v - 2.0 * u_n * normal[:, 1]
        w_ghost = w - 2.0 * u_n * normal[:, 2]
        
        rhou_ghost = rho * u_ghost
        rhov_ghost = rho * v_ghost
        rhow_ghost = rho * w_ghost
        
        return np.column_stack([
            rho, rhou_ghost, rhov_ghost, rhow_ghost, E,
            rho * k, rho * omega
        ])
