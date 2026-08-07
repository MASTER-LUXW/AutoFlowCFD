"""Boundary identification and mapping module.

Identifies boundary faces from volume mesh and maps surface boundaries
to volume mesh cells.
"""

import numpy as np
from typing import Dict, Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ..structures import BoundaryMap, GridData, VolumeMeshData


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
    from ..structures import BoundaryMap
    
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
    from ..structures import BoundaryMap

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

    # A boundary cell matched by neither direct_cell_groups nor node-triplet
    # lookup used to just silently vanish from every group - it still has an
    # exterior face in the mesh, but no boundary condition at all, and
    # nothing downstream (the solver's BC handler) would know why. Put such
    # cells in an explicit catch-all group instead so a solver setup that
    # can't find a BC for some cells has a concrete, loud reason.
    unique_boundary_cells = np.unique(boundary_cell_indices)
    unmatched = np.setdiff1d(unique_boundary_cells, np.fromiter(
        volume_cell_to_boundary.keys(), dtype=np.int64, count=len(volume_cell_to_boundary)
    ), assume_unique=True)
    if len(unmatched) > 0:
        logger.warning(
            f"{len(unmatched)}/{len(unique_boundary_cells)} boundary cells matched "
            f"neither BL-extrusion group tracking nor a surface boundary face "
            f"(likely a remeshed/subdivided face whose nodes no longer match the "
            f"original surface) - placed in an 'UNCLASSIFIED' group as WALL "
            f"instead of being silently dropped from every boundary condition"
        )
        for cell_idx in unmatched:
            volume_cell_to_boundary[int(cell_idx)] = 'UNCLASSIFIED'

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


def map_boundaries_by_geometry(
    volume_mesh: 'VolumeMeshData',
    surface_grid: 'GridData',
    distance_tolerance_factor: float = 0.75,
) -> 'BoundaryMap':
    """Attribute boundary groups to an EXTERNALLY-generated volume mesh's
    own exterior faces, by nearest-centroid geometric matching against a
    companion surface mesh's boundary groups.

    Unlike map_surface_boundaries (node-INDEX matching - only correct
    when both meshes share the same node numbering, which holds for this
    project's own generation pipeline but not for a volume mesh some
    other tool produced, e.g. ANSA's own volume export: its node ids have
    no relationship at all to the original surface .nas file's), this
    matches by POSITION: every exterior face of `volume_mesh` is matched
    to whichever surface boundary group has a face closest to it. Exact
    coincidence isn't expected (a volume mesher may retriangulate/insert
    Steiner points, so a volume boundary face is rarely identical to any
    single original surface face) - only proximity, gated by
    `distance_tolerance_factor` so a face that's suspiciously far from
    every surface boundary face (e.g. a genuine tetgen/mesher interior
    artifact incorrectly exposed, or a mismatched pair of files) falls
    through to 'UNCLASSIFIED' instead of being silently mis-attributed to
    whatever happens to be geometrically nearest.

    Args:
        volume_mesh: The externally-parsed volume mesh (e.g.
            nas_parser_volume.parse_volume_mesh_nas's own output) -
            `ensure_faces_exist()` is called on it if faces aren't
            already computed.
        surface_grid: The companion surface mesh (NASParser.parse()'s
            own output) - its `boundaries.groups` supplies the inlet/
            outlet/wall/... groups to match against, and its `bc_types`
            is inherited unchanged for any matched group.
        distance_tolerance_factor: A volume boundary face's nearest
            surface-boundary-face centroid must be within this many
            multiples of that surface face's own circumradius to count
            as a match - scales with local mesh density automatically
            instead of a single fixed absolute distance, since a fine
            region's surface faces are much smaller (and so need a much
            tighter tolerance) than a coarse region's.

    Returns:
        BoundaryMap with cell indices in `volume_mesh`'s own global
        mixed-cell convention (prisms [0, n_prism), tets
        [n_prism, n_prism + n_tet) - see face_extractor.extract_faces_
        mixed's own docstring), same convention map_surface_boundaries'
        own output already uses. Unmatched exterior faces' owning cells
        go into 'UNCLASSIFIED' (WALL), the same fallback
        map_surface_boundaries uses for its own unmatched case.
    """
    from scipy.spatial import cKDTree
    from ..structures import BoundaryMap

    logger.info("Mapping surface boundaries to external volume mesh by geometry...")

    faces = volume_mesh.ensure_faces_exist()
    boundary_face_idx = faces.get_boundary_face_indices()
    if len(boundary_face_idx) == 0:
        logger.warning("External volume mesh has no exterior faces at all - returning empty BoundaryMap")
        return BoundaryMap(groups={}, bc_types={})

    vol_nodes = np.column_stack([volume_mesh.nodes.x, volume_mesh.nodes.y, volume_mesh.nodes.z])
    vol_face_verts = faces.node_connectivity[boundary_face_idx]
    vol_face_centroids = vol_nodes[vol_face_verts].mean(axis=1)
    vol_face_owner = faces.connectivity[boundary_face_idx, 0]

    surf_nodes = np.column_stack([
        surface_grid.nodes.x, surface_grid.nodes.y, surface_grid.nodes.z
    ])
    surf_faces = surface_grid.cells.connectivity

    surf_centroids_list = []
    surf_radius_list = []
    surf_group_list = []
    for name, face_idx in surface_grid.boundaries.groups.items():
        face_idx = face_idx[face_idx < len(surf_faces)]
        if len(face_idx) == 0:
            continue
        verts = surf_faces[face_idx]
        pts = surf_nodes[verts]
        centroids = pts.mean(axis=1)
        # Circumradius proxy: max distance from centroid to any of its
        # own 3 vertices - a cheap, sufficient local-scale estimate (no
        # need for the exact circumradius, just something proportional
        # to "how big is this face").
        radius = np.linalg.norm(pts - centroids[:, None, :], axis=2).max(axis=1)
        surf_centroids_list.append(centroids)
        surf_radius_list.append(radius)
        surf_group_list.extend([name] * len(face_idx))

    if not surf_centroids_list:
        logger.warning(
            "Surface mesh has no boundary groups at all - every external "
            "volume mesh exterior face will fall through to UNCLASSIFIED"
        )
        groups = {'UNCLASSIFIED': np.unique(vol_face_owner).astype(np.int32)}
        bc_types = {'UNCLASSIFIED': 'WALL'}
        return BoundaryMap(groups=groups, bc_types=bc_types)

    surf_centroids = np.vstack(surf_centroids_list)
    surf_radius = np.concatenate(surf_radius_list)
    surf_group_arr = np.array(surf_group_list, dtype=object)

    tree = cKDTree(surf_centroids)
    dist, nearest_idx = tree.query(vol_face_centroids)
    tolerance = np.maximum(surf_radius[nearest_idx] * distance_tolerance_factor, 1e-12)
    matched = dist <= tolerance

    volume_cell_to_boundary: Dict[int, str] = {}
    for i in np.flatnonzero(matched):
        volume_cell_to_boundary[int(vol_face_owner[i])] = str(surf_group_arr[nearest_idx[i]])

    unique_owners = np.unique(vol_face_owner)
    n_matched_cells = len(volume_cell_to_boundary)
    n_unmatched = len(unique_owners) - n_matched_cells
    if n_unmatched > 0:
        logger.warning(
            f"{n_unmatched}/{len(unique_owners)} exterior-face-owning cells matched no "
            f"surface boundary group within tolerance - placed in an 'UNCLASSIFIED' "
            f"group as WALL instead of being silently dropped from every boundary condition"
        )
        for cell_idx in unique_owners:
            if int(cell_idx) not in volume_cell_to_boundary:
                volume_cell_to_boundary[int(cell_idx)] = 'UNCLASSIFIED'

    groups: Dict[str, list] = {}
    bc_types: Dict[str, str] = {}
    for cell_idx, name in volume_cell_to_boundary.items():
        groups.setdefault(name, []).append(cell_idx)
        if name not in bc_types:
            bc_types[name] = surface_grid.boundaries.bc_types.get(name, 'WALL')

    groups_arr = {name: np.array(idx, dtype=np.int32) for name, idx in groups.items()}
    boundaries = BoundaryMap(groups=groups_arr, bc_types=bc_types)
    logger.info(
        f"Geometric boundary mapping completed: {len(groups_arr)} boundary groups, "
        f"{sum(len(c) for c in groups_arr.values())} total cells "
        f"({n_matched_cells} matched by proximity, {max(n_unmatched, 0)} UNCLASSIFIED)"
    )
    return boundaries
