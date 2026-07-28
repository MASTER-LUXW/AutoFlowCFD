"""Aerodynamic coefficient calculator for FVM solver.

This module computes aerodynamic coefficients (Cd, Cl, etc.) by integrating
pressure over body surfaces.

Key Components:
    - AeroCoefficientCalculator: Computes drag and lift coefficients
"""

import numpy as np
from typing import Tuple
from loguru import logger


class AeroCoefficientCalculator:
    """Computes aerodynamic coefficients from solution field."""
    
    def __init__(self, grid_data, face_extractor):
        """Initialize aerodynamic coefficient calculator.
        
        Args:
            grid_data: Volume mesh data (VolumeMeshData)
            face_extractor: Face extractor for volume mesh
        """
        self.grid_data = grid_data
        self.face_extractor = face_extractor
        
        # Cache for reference area to avoid recomputation
        self._cached_ref_area = None
        self._ref_area_computed = False

    def compute_coefficients(self, solution: np.ndarray, iteration: int = 0) -> Tuple[float, float]:
        """Compute drag and lift coefficients.
        
        Args:
            solution: Solution array, shape=(n_cells, 7)
            iteration: Current iteration number (for debugging)
            
        Returns:
            Tuple of (Cd, Cl)
        """
        try:
            # Extract primitive variables
            rho = solution[:, 0]
            rhou = solution[:, 1]
            rhov = solution[:, 2]
            rhow = solution[:, 3]
            E = solution[:, 4]
            
            gamma = 1.4
            velocity_x = rhou / np.maximum(rho, 1e-10)
            velocity_y = rhov / np.maximum(rho, 1e-10)
            velocity_z = rhow / np.maximum(rho, 1e-10)
            
            V_squared = velocity_x**2 + velocity_y**2 + velocity_z**2
            pressure = (gamma - 1.0) * (E - 0.5 * rho * V_squared)
            
            # Freestream conditions
            rho_inf = 1.225
            vel_inf = 30.0
            q_inf = 0.5 * rho_inf * vel_inf**2
            
            if q_inf < 1e-6:
                logger.warning("Dynamic pressure too small")
                return 0.0, 0.0
            
            # Identify body faces
            body_face_indices = self._identify_body_faces()
            
            if len(body_face_indices) == 0:
                logger.warning(f"[Iter {iteration}] No body faces found - returning Cd=0, Cl=0")
                return 0.0, 0.0
            
            # Get face data
            face_normals = self.face_extractor.face_normals[body_face_indices]
            face_areas = self.face_extractor.face_areas[body_face_indices]
            
            # Get pressure on body surface
            body_cell_indices = self.face_extractor.face_connectivity[body_face_indices, 0]
            p_body = pressure[body_cell_indices]
            p_ref = 101325.0
            
            dp = p_body - p_ref
            
            # Debug output for first few iterations
            if iteration <= 5 or iteration % 10 == 0:
                logger.info(f"[Iter {iteration}] Body surface analysis:")
                logger.info(f"  Body faces: {len(body_face_indices)}")
                logger.info(f"  Pressure stats: min={p_body.min():.2f}, max={p_body.max():.2f}, mean={p_body.mean():.2f}")
                logger.info(f"  Pressure diff stats: min={dp.min():.2f}, max={dp.max():.2f}, mean={dp.mean():.2f}")
                logger.info(f"  Total area: {face_areas.sum():.6f} m²")
            
            # Force components
            Fx = -np.sum(dp * face_normals[:, 0] * face_areas)
            Fz = -np.sum(dp * face_normals[:, 2] * face_areas)
            
            # Reference area
            ref_area = self._compute_reference_area(body_face_indices)
            
            # Coefficients
            Cd = Fx / (q_inf * ref_area)
            Cl = Fz / (q_inf * ref_area)
            
            # Validate
            if not np.isfinite(Cd):
                logger.warning("Cd is not finite")
                Cd = 0.0
            if not np.isfinite(Cl):
                logger.warning("Cl is not finite")
                Cl = 0.0
            
            # Debug output
            if iteration <= 5 or iteration % 10 == 0:
                logger.info(f"[Iter {iteration}] Forces: Fx={Fx:.2f} N, Fz={Fz:.2f} N")
                logger.info(f"[Iter {iteration}] Coefficients: Cd={Cd:.6f}, Cl={Cl:.6f} (A_ref={ref_area:.6f} m²)")
            
            return float(Cd), float(Cl)
            
        except Exception as e:
            logger.error(f"Failed to compute coefficients: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.0, 0.0
    
    def _identify_body_faces(self) -> np.ndarray:
        """Identify body surface face indices using vectorized operations.
        
        Uses NumPy set operations instead of Python loops for performance.
        
        Returns:
            Array of face indices belonging to body boundaries
        """
        # Find all boundary names containing 'body'
        body_boundary_names = [
            name for name in self.grid_data.boundaries.boundary_names
            if 'body' in name.lower()
        ]
        
        if not body_boundary_names:
            logger.warning("No body boundary found")
            return np.array([], dtype=np.int64)
        
        logger.info(f"Searching for body boundaries: {body_boundary_names}")
        
        # Collect all body cell indices using set union (vectorized)
        body_cell_set = set()
        for boundary_name in body_boundary_names:
            cells = self.grid_data.boundaries.get_cell_indices(boundary_name)
            logger.info(f"  Boundary '{boundary_name}': {len(cells)} cells")
            body_cell_set.update(cells)
        
        if not body_cell_set:
            logger.warning(f"Body boundary found but no cells identified: {body_boundary_names}")
            return np.array([], dtype=np.int64)
        
        # Convert to numpy array for fast lookup
        body_cells_array = np.array(list(body_cell_set), dtype=np.int64)
        logger.info(f"Total body cells: {len(body_cells_array)}")
        
        # Get all boundary faces
        boundary_mask = self.face_extractor.boundary_flags
        
        # Get left cell indices for all faces
        left_cells = self.face_extractor.face_connectivity[:, 0]
        
        # Use numpy isin for vectorized membership test (much faster than Python loop)
        is_body_face = np.isin(left_cells, body_cells_array) & boundary_mask
        
        # Get indices where condition is True
        body_face_indices = np.where(is_body_face)[0]
        
        logger.info(f"Identified {len(body_face_indices)} body surface faces from {len(body_cells_array)} cells")
        
        return body_face_indices.astype(np.int64)

    def _compute_reference_area(self, body_face_indices: np.ndarray) -> float:
        """Compute reference frontal area using bounding box estimation.
        
        Uses the body boundary's bounding box dimensions to estimate the
        frontal projected area, avoiding issues with volume mesh extrusion layers.
        
        Args:
            body_face_indices: Indices of body surface faces (not used in this implementation)
            
        Returns:
            Reference area in m^2
        """
        # Use cached value if already computed
        if self._ref_area_computed and self._cached_ref_area is not None:
            return self._cached_ref_area
        
        try:
            # Compute reference area from bounding box
            ref_area = self._compute_ref_area_from_surface_mesh()
            
            if ref_area > 0 and np.isfinite(ref_area):
                self._cached_ref_area = ref_area
                self._ref_area_computed = True
                return ref_area
            
            # Fallback to old method if bounding box estimation fails
            logger.warning("Bounding box estimation failed, using volume mesh fallback")
            ref_area = self._compute_ref_area_from_volume_mesh(body_face_indices)
            
            self._cached_ref_area = ref_area
            self._ref_area_computed = True
            
            return ref_area
            
        except Exception as e:
            logger.error(f"Failed to compute reference area: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 1.0
    
    def _compute_ref_area_from_surface_mesh(self) -> float:
        """Compute reference area from original surface mesh geometry.
        
        This method directly uses the surface triangles from the NAS file,
        avoiding any issues with volume mesh extrusion layers or boundary pollution.
        
        Returns:
            Reference area in m^2, or 0.0 if computation fails
        """
        try:
            # Check if surface mesh is available
            if not hasattr(self.grid_data, 'surface_mesh') or self.grid_data.surface_mesh is None:
                logger.warning("Surface mesh not available in grid_data, using fallback method")
                return 0.0
            
            surface_mesh = self.grid_data.surface_mesh
            surface_nodes = surface_mesh.get('nodes')  # shape=(n_nodes, 3)
            surface_faces = surface_mesh.get('faces')  # shape=(n_faces, 3)
            surface_boundaries = surface_mesh.get('boundaries')  # BoundaryMap
            
            if surface_nodes is None or surface_faces is None:
                logger.warning("Surface mesh nodes or faces not available")
                return 0.0
            
            # Get body boundary face indices from surface mesh
            if surface_boundaries is None:
                logger.warning("Surface mesh boundaries not available")
                return 0.0
            
            body_boundary_names = [
                name for name in surface_boundaries.boundary_names
                if 'body' in name.lower()
            ]
            
            if not body_boundary_names:
                logger.warning("No body boundary found in surface mesh")
                return 0.0
            
            # Collect all body face indices from surface mesh
            body_face_indices = []
            for boundary_name in body_boundary_names:
                face_indices = surface_boundaries.get_cell_indices(boundary_name)
                body_face_indices.extend(face_indices)
            
            if len(body_face_indices) == 0:
                logger.warning("No body faces found in surface mesh")
                return 0.0
            
            body_face_indices = np.array(body_face_indices, dtype=np.int64)
            
            logger.info(f"Surface mesh body analysis:")
            logger.info(f"  Body faces: {len(body_face_indices)}")
            
            # Extract node coordinates for body faces
            v0 = surface_nodes[surface_faces[body_face_indices, 0]]
            v1 = surface_nodes[surface_faces[body_face_indices, 1]]
            v2 = surface_nodes[surface_faces[body_face_indices, 2]]
            
            # Compute face normals and areas
            e1 = v1 - v0
            e2 = v2 - v0
            normals = np.cross(e1, e2)
            areas = 0.5 * np.linalg.norm(normals, axis=1)
            
            # Normalize normals to unit vectors
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # Avoid division by zero
            unit_normals = normals / norms
            
            # Debug output
            logger.info(f"  Total area: {areas.sum():.6f} m²")
            logger.info(f"  Mean face area: {areas.mean():.6e} m²")
            logger.info(f"  Min/Max face area: {areas.min():.6e} / {areas.max():.6e} m²")
            
            # Compute projected area in X-direction (freestream direction)
            x_component = unit_normals[:, 0]
            
            # Only count upstream-facing faces (normal pointing against flow, n_x < 0)
            upstream_mask = x_component < 0
            projected_areas = -x_component[upstream_mask] * areas[upstream_mask]
            ref_area = np.sum(projected_areas)
            
            # Validate
            if ref_area <= 0 or not np.isfinite(ref_area):
                logger.warning(f"Invalid reference area from surface mesh: {ref_area:.6e}")
                # Fallback: use absolute projection divided by 2 (for symmetric bodies)
                projected_areas_all = np.abs(x_component) * areas
                ref_area_fallback = np.sum(projected_areas_all) / 2.0
                if ref_area_fallback > 0 and np.isfinite(ref_area_fallback):
                    logger.info(f"Fallback reference area (|n_x|/2): {ref_area_fallback:.6f} m^2")
                    return float(ref_area_fallback)
                return 0.0
            
            # Sanity check for Ahmed Body
            if ref_area < 0.01 or ref_area > 1.0:
                logger.warning(f"Reference area {ref_area:.4f} m^2 outside expected range (0.1-0.3 m^2)")
                logger.warning(f"  Upstream-facing faces: {np.sum(upstream_mask)} / {len(body_face_indices)}")
                logger.warning(f"  Mean |n_x|: {np.mean(np.abs(x_component)):.4f}")
            
            logger.info(f"Reference area (from surface mesh): {ref_area:.6f} m^2")
            logger.info(f"  Upstream-facing ratio: {np.sum(upstream_mask) / len(body_face_indices) * 100:.1f}%")
            logger.info(f"  Mean projected area per upstream face: {ref_area / max(1, np.sum(upstream_mask)):.6e} m^2")
            
            return float(ref_area)
            
        except Exception as e:
            logger.error(f"Failed to compute reference area from surface mesh: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.0
    
    def _compute_ref_area_from_volume_mesh(self, body_face_indices: np.ndarray) -> float:
        """Fallback: compute reference area from volume mesh boundary faces.
        
        This method is used when surface mesh is not available.
        
        Args:
            body_face_indices: Indices of body surface faces in volume mesh
            
        Returns:
            Reference area in m^2
        """
        if len(body_face_indices) == 0:
            logger.warning("No body faces identified for reference area calculation")
            return 1.0
        
        face_normals = self.face_extractor.face_normals[body_face_indices]
        face_areas = self.face_extractor.face_areas[body_face_indices]
        
        # Debug: Print body geometry statistics
        logger.info(f"Volume mesh body analysis:")
        logger.info(f"  Body faces: {len(body_face_indices)}")
        logger.info(f"  Total area: {face_areas.sum():.6f} m²")
        logger.info(f"  Mean face area: {face_areas.mean():.6e} m²")
        logger.info(f"  Min/Max face area: {face_areas.min():.6e} / {face_areas.max():.6e} m²")
        
        # Projected area in X-direction (freestream direction)
        x_component = face_normals[:, 0]
        
        # Only count upstream-facing faces (more accurate)
        upstream_mask = x_component < 0
        projected_areas = -x_component[upstream_mask] * face_areas[upstream_mask]
        ref_area = np.sum(projected_areas)
        
        # Validate
        if ref_area <= 0 or not np.isfinite(ref_area):
            logger.warning(f"Invalid reference area: {ref_area:.6e}, using fallback")
            projected_areas_all = np.abs(x_component) * face_areas
            ref_area_fallback = np.sum(projected_areas_all) / 2.0
            if ref_area_fallback > 0 and np.isfinite(ref_area_fallback):
                logger.info(f"Fallback reference area: {ref_area_fallback:.6f} m^2")
                return float(ref_area_fallback)
            return 1.0
        
        # Sanity check
        if ref_area > 10.0:
            logger.warning(f"Reference area {ref_area:.2f} m^2 seems too large for Ahmed Body")
            logger.warning(f"  Number of body faces: {len(body_face_indices)}")
            logger.warning(f"  Mean |n_x|: {np.mean(np.abs(x_component)):.4f}")
            logger.warning(f"  Upstream-facing ratio: {np.sum(upstream_mask) / len(body_face_indices) * 100:.1f}%")
            logger.warning(f"  This may indicate incorrect unit conversion or face identification")
        
        logger.info(f"Reference area (volume mesh): {ref_area:.6f} m^2")
        logger.info(f"  Upstream-facing faces: {np.sum(upstream_mask)} / {len(body_face_indices)}")
        logger.info(f"  Mean projected area per face: {ref_area / max(1, np.sum(upstream_mask)):.6e} m^2")
        
        return float(ref_area)
