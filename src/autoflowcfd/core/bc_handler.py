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
        elif boundary_type == "INLET":
            return self._inlet_bc()
        elif boundary_type == "FARFIELD":
            return self._farfield_bc(rho, u, v, w, p, k, omega, normal)
        elif boundary_type == "OUTLET":
            return self._outlet_bc(rho, u, v, w, p, k, omega, normal)
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
    
    def _inlet_bc(self) -> np.ndarray:
        """Prescribed-velocity inlet boundary condition with ramp factor.

        Unlike FARFIELD, an INLET is an explicit prescribed-inflow face (the
        user named it "inlet"), so a hard Dirichlet freestream state is the
        physically intended condition here - there is no ambiguity about
        flow direction to resolve.

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

    def _farfield_bc(self, rho: float, u: float, v: float, w: float, p: float,
                     k: float, omega: float, normal: np.ndarray) -> np.ndarray:
        """Characteristic (Riemann-invariant) subsonic far-field boundary condition.

        A box-shaped far-field/tunnel boundary is local inflow on the
        front/sides but local outflow on the rear/top - imposing the full
        freestream Dirichlet state everywhere (the previous behaviour) forces
        mass through what are physically outflow faces and biases the
        pressure field there, which feeds back into Cd/Cl and slows/
        destabilizes convergence. This applies the standard 1-D
        Riemann-invariant extrapolation along the face normal: the outgoing
        invariant R+ is taken from the interior, the incoming invariant R-
        is taken from the fixed freestream state, and whichever side the
        resulting normal velocity implies (inflow vs. outflow) supplies the
        tangential velocity and entropy (rho, p), matching the standard
        subsonic far-field BC used in e.g. SU2/OpenFOAM's characteristic
        far-field condition. Assumes the freestream direction is +x, matching
        `_inlet_bc`/`_precompute_face_types`'s convention (v_inf = w_inf = 0).
        """
        gamma = self.gamma
        rho_inf = self.rho_inf
        p_inf = self.p_inf
        u_inf = self.get_current_farfield_velocity()

        rho_safe = max(rho, 1e-10)
        c = np.sqrt(gamma * max(p, 100.0) / rho_safe)
        c_inf = np.sqrt(gamma * p_inf / max(rho_inf, 1e-10))

        vel = np.array([u, v, w])
        vel_inf = np.array([u_inf, 0.0, 0.0])

        un = float(np.dot(vel, normal))
        un_inf = float(np.dot(vel_inf, normal))

        R_plus = un + 2.0 * c / (gamma - 1.0)
        R_minus = un_inf - 2.0 * c_inf / (gamma - 1.0)

        un_b = 0.5 * (R_plus + R_minus)
        c_b = max((gamma - 1.0) / 4.0 * (R_plus - R_minus), 1e-6)

        if un_b < 0.0:
            # Local inflow: tangential velocity & entropy come from freestream.
            vel_tang = vel_inf - un_inf * normal
            rho_side, p_side = rho_inf, p_inf
            k_b = 1.5 * (0.01 * max(u_inf, 1.0))**2
            omega_b = 5.0 * max(u_inf, 1.0) / 0.1
        else:
            # Local outflow: tangential velocity & entropy extrapolated from interior.
            vel_tang = vel - un * normal
            rho_side, p_side = rho_safe, max(p, 100.0)
            k_b, omega_b = k, omega

        s = p_side / (rho_side ** gamma)
        rho_b = max(c_b**2 / (gamma * s), 1e-10) ** (1.0 / (gamma - 1.0))
        p_b = rho_b * c_b**2 / gamma

        vel_b = vel_tang + un_b * normal
        u_b, v_b, w_b = vel_b

        rhou_b = rho_b * u_b
        rhov_b = rho_b * v_b
        rhow_b = rho_b * w_b
        E_b = p_b / (gamma - 1.0) + 0.5 * rho_b * (u_b**2 + v_b**2 + w_b**2)

        return np.array([rho_b, rhou_b, rhov_b, rhow_b, E_b, rho_b * k_b, rho_b * omega_b])

    def _outlet_bc(self, rho: float, u: float, v: float, w: float,
                  p: float, k: float, omega: float, normal: np.ndarray) -> np.ndarray:
        """Outlet boundary condition (static pressure specified, rest extrapolated).

        Backflow-safe: if the local interior velocity actually points INTO
        the domain through this outflow-only face (un < 0 - common
        whenever a separated/vortex-shedding wake, e.g. a bluff body's,
        hasn't settled into attached axial flow by the time it reaches the
        outlet plane), a plain zero-gradient extrapolation of velocity and
        density directly re-injects that reversed, wake-disturbed state
        back into the domain with no bound - a self-reinforcing
        instability (observed directly: density piling up to >10x
        freestream exactly at the outlet plane on a cube case whose wake
        reached the outlet only ~7 body-widths downstream). On backflow,
        fall back to freestream density and zero velocity - a safe,
        bounded "stagnant reservoir" assumption - instead of extrapolating
        the disturbed interior state.
        """
        gamma = 1.4
        p_outlet = self.p_inf

        un = u * normal[0] + v * normal[1] + w * normal[2]
        if un < 0.0:
            rho_g, u_g, v_g, w_g = self.rho_inf, 0.0, 0.0, 0.0
            # The "stagnant reservoir" assumption above only reset density/
            # velocity/pressure - k/omega were left as the raw interior
            # (wake-disturbed) values, silently re-injecting the wake's own
            # turbulence signature through the one channel the density/
            # velocity reset was specifically added to close off. A
            # reservoir has freestream turbulence, not wake turbulence -
            # same 1% intensity / 0.1 m length-scale formula _inlet_bc uses.
            u_ref = max(self.base_inlet_velocity, 1.0)
            k_g = 1.5 * (0.01 * u_ref) ** 2
            omega_g = 5.0 * u_ref / 0.1
        else:
            rho_g, u_g, v_g, w_g = rho, u, v, w
            k_g, omega_g = k, omega

        rhou = rho_g * u_g
        rhov = rho_g * v_g
        rhow = rho_g * w_g
        E = p_outlet / (gamma - 1.0) + 0.5 * rho_g * (u_g**2 + v_g**2 + w_g**2)

        return np.array([rho_g, rhou, rhov, rhow, E, rho_g * k_g, rho_g * omega_g])

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
        elif "TUNNEL" in name_upper:
            # A named "tunnel" boundary is a physical (if frictionless) duct
            # wall - zero-penetration, free-slip - not an open domain
            # boundary. Reuses the SYMMETRY ghost state (mirror the normal
            # velocity component, no viscous shear enforced), which is
            # mathematically identical to a slip wall for an inviscid-wall
            # treatment. Previously grouped with FARFIELD (an open,
            # characteristic boundary letting mass freely cross), which is
            # the wrong physics for an actual tunnel wall and let flow
            # leak through what should have been a solid boundary.
            return "SYMMETRY"
        elif "FARFIELD" in name_upper:
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
            elif btype == "INLET":
                # Prescribed inflow: fixed freestream Dirichlet state.
                n_type = np.sum(type_mask)
                ghost_states = np.tile(self._inlet_bc(), (n_type, 1))
            elif btype == "FARFIELD":
                ghost_states = self._farfield_bc_vectorized(
                    rho_t, u_t, v_t, w_t, p_t, k_t, omega_t, normals_t
                )
            elif btype == "OUTLET":
                ghost_states = self._outlet_bc_vectorized(
                    rho_t, u_t, v_t, w_t, p_t, k_t, omega_t, normals_t
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
    
    def _farfield_bc_vectorized(self, rho: np.ndarray, u: np.ndarray, v: np.ndarray,
                               w: np.ndarray, p: np.ndarray, k: np.ndarray,
                               omega: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Vectorized characteristic far-field BC - see `_farfield_bc` for the
        per-face derivation (Riemann-invariant inflow/outflow split)."""
        gamma = self.gamma
        rho_inf = self.rho_inf
        p_inf = self.p_inf
        u_inf = self.get_current_farfield_velocity()

        rho_safe = np.maximum(rho, 1e-10)
        p_safe = np.maximum(p, 100.0)
        c = np.sqrt(gamma * p_safe / rho_safe)
        c_inf = np.sqrt(gamma * p_inf / max(rho_inf, 1e-10))

        nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
        un = u * nx + v * ny + w * nz
        un_inf = u_inf * nx  # freestream direction is +x: v_inf = w_inf = 0

        R_plus = un + 2.0 * c / (gamma - 1.0)
        R_minus = un_inf - 2.0 * c_inf / (gamma - 1.0)

        un_b = 0.5 * (R_plus + R_minus)
        c_b = np.maximum((gamma - 1.0) / 4.0 * (R_plus - R_minus), 1e-6)

        inflow = un_b < 0.0

        u_tang = u - un * nx
        v_tang = v - un * ny
        w_tang = w - un * nz
        u_tang_inf = u_inf - un_inf * nx
        v_tang_inf = -un_inf * ny
        w_tang_inf = -un_inf * nz

        u_tang_b = np.where(inflow, u_tang_inf, u_tang)
        v_tang_b = np.where(inflow, v_tang_inf, v_tang)
        w_tang_b = np.where(inflow, w_tang_inf, w_tang)

        rho_side = np.where(inflow, rho_inf, rho_safe)
        p_side = np.where(inflow, p_inf, p_safe)
        k_inf = 1.5 * (0.01 * max(u_inf, 1.0))**2
        omega_inf = 5.0 * max(u_inf, 1.0) / 0.1
        k_b = np.where(inflow, k_inf, k)
        omega_b = np.where(inflow, omega_inf, omega)

        s = p_side / rho_side ** gamma
        rho_b = np.maximum(c_b**2 / (gamma * s), 1e-10) ** (1.0 / (gamma - 1.0))
        p_b = rho_b * c_b**2 / gamma

        u_b = u_tang_b + un_b * nx
        v_b = v_tang_b + un_b * ny
        w_b = w_tang_b + un_b * nz

        rhou_b = rho_b * u_b
        rhov_b = rho_b * v_b
        rhow_b = rho_b * w_b
        E_b = p_b / (gamma - 1.0) + 0.5 * rho_b * (u_b**2 + v_b**2 + w_b**2)

        return np.column_stack([rho_b, rhou_b, rhov_b, rhow_b, E_b, rho_b * k_b, rho_b * omega_b])

    def _outlet_bc_vectorized(self, rho: np.ndarray, u: np.ndarray, v: np.ndarray,
                             w: np.ndarray, p: np.ndarray, k: np.ndarray,
                             omega: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Vectorized outlet boundary condition - see the scalar `_outlet_bc`
        for why the backflow clamp is needed."""
        gamma = self.gamma
        p_outlet = self.p_inf

        un = u * normals[:, 0] + v * normals[:, 1] + w * normals[:, 2]
        backflow = un < 0.0

        rho_g = np.where(backflow, self.rho_inf, rho)
        u_g = np.where(backflow, 0.0, u)
        v_g = np.where(backflow, 0.0, v)
        w_g = np.where(backflow, 0.0, w)

        # See the scalar `_outlet_bc` for why backflow also resets k/omega
        # to freestream turbulence, not just density/velocity/pressure.
        u_ref = max(self.base_inlet_velocity, 1.0)
        k_inf = 1.5 * (0.01 * u_ref) ** 2
        omega_inf = 5.0 * u_ref / 0.1
        k_g = np.where(backflow, k_inf, k)
        omega_g = np.where(backflow, omega_inf, omega)

        rhou = rho_g * u_g
        rhov = rho_g * v_g
        rhow = rho_g * w_g
        E = p_outlet / (gamma - 1.0) + 0.5 * rho_g * (u_g**2 + v_g**2 + w_g**2)

        return np.column_stack([rho_g, rhou, rhov, rhow, E, rho_g * k_g, rho_g * omega_g])
    
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
