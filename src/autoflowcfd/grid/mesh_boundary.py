"""Boundary identification and mapping module.

Identifies boundary faces from volume mesh and maps surface boundaries
to volume mesh cells.
"""

import numpy as np
from typing import Dict, Optional
from loguru import logger


def identify_boundaries_from_surface(
    volume_cells: np.ndarray,
    surface_faces: np.ndarray,
    surface_boundaries: Optional['BoundaryMap'] = None,
    direct_cell_groups: Optional[np.ndarray] = None,
) -> 'BoundaryMap':
    """Identify boundary faces from volume mesh and inherit surface boundaries.

    Boundary faces are those that belong to only one cell (exterior faces).
    This method maps the original surface face boundaries to the volume mesh.

    Args:
        volume_cells: Tetrahedral connectivity, shape=(n_cells, 4)
        surface_faces: Original surface face connectivity with boundary info
        surface_boundaries: Optional boundary mapping from surface mesh
        direct_cell_groups: Optional (n_cells,) str array giving each cell's
            source boundary-group name directly (empty string if unknown),
            e.g. from mesh_background.generate_hybrid_mesh's BL-extrusion
            face tracking. Takes priority over node-index matching below,
            which cannot work for BL-extruded groups (see map_surface_boundaries).

    Returns:
        BoundaryMap object with identified boundary groups
    """
    from .structures import BoundaryMap
    
    logger.info("Identifying boundary conditions from surface mesh...")
    
    # Extract tetrahedron faces (each tet has 4 triangular faces)
    n_tets = len(volume_cells)
    
    # Vectorized generation of all tetrahedron faces - much faster than loop
    logger.info(f"Extracting faces from {n_tets} tetrahedra (vectorized)...")
    
    # Use advanced indexing to generate all faces at once
    face_templates = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ], dtype=np.int64)
    
    # Generate all faces: shape=(n_tets*4, 3)
    tet_faces = volume_cells[:, face_templates].reshape(-1, 3)
    
    # Generate cell IDs for each face
    tet_face_cell_ids = np.repeat(np.arange(n_tets), 4)
    
    logger.info(f"Generated {len(tet_faces)} total faces")
    
    # Sort nodes in each face to enable comparison (canonical form)
    tet_faces_sorted = np.sort(tet_faces, axis=1)
    
    # Find faces that appear only once (boundary faces) - Fully vectorized approach
    logger.info("Finding boundary faces (vectorized)...")
    
    # Convert each row to a single void type for hashing
    face_dtype = np.dtype((np.void, tet_faces_sorted.dtype.itemsize * 3))
    face_voids = np.ascontiguousarray(tet_faces_sorted).view(face_dtype).reshape(-1)
    
    # Count occurrences using np.unique
    unique_faces, inverse_indices, counts = np.unique(
        face_voids, 
        return_inverse=True, 
        return_counts=True
    )
    
    # Boundary faces appear exactly once
    boundary_face_mask = counts[inverse_indices] == 1
    boundary_faces = tet_faces[boundary_face_mask]
    boundary_cell_indices = tet_face_cell_ids[boundary_face_mask]
    
    logger.info(
        f"Found {len(boundary_faces)} boundary faces on "
        f"{len(np.unique(boundary_cell_indices))} cells"
    )
    
    # If surface boundaries are provided, try to map them to volume mesh
    if surface_boundaries is not None and len(surface_boundaries.groups) > 0:
        logger.info(
            f"Inheriting {len(surface_boundaries.groups)} boundary groups "
            f"from surface mesh"
        )
        return map_surface_boundaries(
            boundary_faces, boundary_cell_indices,
            surface_faces, surface_boundaries,
            direct_cell_groups=direct_cell_groups,
        )
    
    # Fallback: create a single "wall" boundary group with all boundary cells
    groups = {}
    bc_types = {}
    
    if len(boundary_cell_indices) > 0:
        unique_boundary_cells = np.unique(boundary_cell_indices)
        # Convert to numpy int32 array (required by BoundaryMap)
        groups['wall'] = unique_boundary_cells.astype(np.int32)
        bc_types['wall'] = 'WALL'
        logger.info(f"Created 'wall' boundary group with {len(unique_boundary_cells)} cells")
    
    boundaries = BoundaryMap(groups=groups, bc_types=bc_types)
    logger.info(f"Boundary identification completed: {len(groups)} boundary groups")
    
    return boundaries


def map_surface_boundaries(
    boundary_faces: np.ndarray,
    boundary_cell_indices: np.ndarray,
    surface_faces: np.ndarray,
    surface_boundaries: 'BoundaryMap',
    direct_cell_groups: Optional[np.ndarray] = None,
) -> 'BoundaryMap':
    """Map surface mesh boundaries to volume mesh boundary cells.

    Uses node-based matching to identify which volume boundary cells
    correspond to which surface boundary groups. Node-based matching only
    works for cells whose boundary-face nodes are literally unchanged from
    the input surface mesh; it structurally cannot match BL-extruded faces,
    since extrusion displaces their nodes to new coordinates/indices. For
    those, `direct_cell_groups` (built during BL extrusion in
    mesh_domain_classify.classify_boundary_groups / mesh_background) gives
    each cell's source group directly and is used first, in preference to
    node matching.

    Args:
        boundary_faces: Volume mesh boundary faces, shape=(n_faces, 3)
        boundary_cell_indices: Cell indices for each boundary face
        surface_faces: Original surface faces, shape=(n_surf_faces, 3)
        surface_boundaries: Surface mesh boundary mapping
        direct_cell_groups: Optional (n_cells,) str array, empty string
            where unknown; takes priority over node matching below

    Returns:
        BoundaryMap with inherited boundary groups
    """
    from .structures import BoundaryMap

    logger.info("Mapping surface boundaries to volume mesh...")

    # Build a mapping from surface face nodes to boundary groups
    # For efficiency, use a dictionary keyed by sorted node tuples
    surface_face_to_boundary = {}
    for boundary_name, cell_indices in surface_boundaries.groups.items():
        for cell_idx in cell_indices:
            if cell_idx < len(surface_faces):
                face_nodes = tuple(sorted(surface_faces[cell_idx]))
                surface_face_to_boundary[face_nodes] = boundary_name

    # Map volume boundary faces to surface boundaries
    volume_cell_to_boundary = {}  # cell_idx -> boundary_name

    n_direct = 0
    if direct_cell_groups is not None:
        for cell_idx in np.unique(boundary_cell_indices):
            if cell_idx < len(direct_cell_groups):
                name = direct_cell_groups[cell_idx]
                if name:
                    volume_cell_to_boundary[cell_idx] = name
                    n_direct += 1
        if n_direct:
            logger.info(
                f"  {n_direct} boundary cells attributed directly from "
                f"BL-extrusion group tracking"
            )

    for i, face in enumerate(boundary_faces):
        cell_idx = boundary_cell_indices[i]
        if cell_idx in volume_cell_to_boundary:
            continue  # already attributed directly (BL-extruded cell)
        face_key = tuple(sorted(face))
        if face_key in surface_face_to_boundary:
            boundary_name = surface_face_to_boundary[face_key]
            volume_cell_to_boundary[cell_idx] = boundary_name

    # Group cells by boundary name
    groups = {}
    bc_types = {}
    
    for cell_idx, boundary_name in volume_cell_to_boundary.items():
        if boundary_name not in groups:
            groups[boundary_name] = []
        groups[boundary_name].append(cell_idx)
        
        # Inherit boundary type from surface
        if boundary_name in surface_boundaries.bc_types:
            bc_types[boundary_name] = surface_boundaries.bc_types[boundary_name]
        else:
            bc_types[boundary_name] = 'WALL'  # Default
    
    # Convert lists to numpy arrays
    for boundary_name in groups:
        groups[boundary_name] = np.array(groups[boundary_name], dtype=np.int32)
    
    boundaries = BoundaryMap(groups=groups, bc_types=bc_types)
    logger.info(
        f"Surface boundary mapping completed: {len(groups)} boundary groups, "
        f"{sum(len(cells) for cells in groups.values())} total cells"
    )
    
    return boundaries
