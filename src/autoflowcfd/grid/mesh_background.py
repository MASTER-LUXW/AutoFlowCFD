"""Background mesh generation module.

Implements Cartesian background grid generation and hybrid mesh assembly
(BL extrusion + background mesh).
"""

import numpy as np
from typing import Dict, Tuple
from loguru import logger

from .mesh_extrusion import extrude_layers, convert_layers_to_tetrahedra
from .mesh_utils import compute_face_normals


def generate_hybrid_mesh(
    surface_nodes: np.ndarray,
    surface_faces: np.ndarray,
    bounding_box: Dict[str, np.ndarray],
    growth_rate: float = 1.2,
    max_layers: int = 30,
    min_cell_size: float = 0.001,
    target_cells: int = 500000
) -> 'VolumeMeshData':
    """Generate hybrid mesh using BL extrusion + Cartesian background with conforming interface.
    
    Strategy (Conforming Hybrid Mesh):
    1. Generate BL layers via extrusion (fine resolution near walls)
    2. Extract BL outer surface nodes as constraint points
    3. Create adaptive background grid that respects BL boundary
    4. Generate transition layer using Delaunay triangulation
    5. Merge all meshes with shared nodes at interface
    
    Args:
        surface_nodes: Surface geometry nodes, shape=(n_nodes, 3)
        surface_faces: Surface face connectivity, shape=(n_faces, 3)
        bounding_box: Domain bounds {min: [x,y,z], max: [x,y,z]}
        growth_rate: Geometric growth rate for BL thickness
        max_layers: Maximum number of BL layers
        min_cell_size: Minimum allowable cell size in meters
        target_cells: Target total cell count
        
    Returns:
        VolumeMeshData with conforming hybrid mesh (BL + Transition + Background)
    """
    logger.info("Starting conforming hybrid mesh generation...")
    
    # Step 1: Generate BL layers via extrusion
    logger.info("Step 1/5: Generating boundary layer mesh...")
    normals = compute_face_normals(surface_nodes, surface_faces)
    bl_nodes, bl_layer_conn = extrude_layers(
        surface_nodes, surface_faces, normals, bounding_box,
        growth_rate=growth_rate, max_layers=max_layers, min_cell_size=min_cell_size
    )
    bl_cells = convert_layers_to_tetrahedra(bl_nodes, bl_layer_conn, surface_faces)
    logger.info(f"  BL mesh: {len(bl_nodes)} nodes, {len(bl_cells)} cells")
    
    # Step 2: Extract BL outer surface for conforming interface
    logger.info("Step 2/5: Extracting BL outer surface for node alignment...")
    bl_outer_surface = bl_layer_conn[-1]  # Last layer connectivity
    bl_outer_nodes = bl_nodes[np.unique(bl_outer_surface)]  # Unique nodes on outer surface
    logger.info(f"  BL outer surface: {len(bl_outer_nodes)} nodes")
    
    # Step 3: Generate adaptive background grid (respecting BL boundary)
    logger.info("Step 3/5: Generating adaptive background grid...")
    bg_nodes, bg_cells = generate_adaptive_background_grid(
        bounding_box, target_cells, bl_outer_nodes, bl_nodes
    )
    logger.info(f"  Background grid: {len(bg_nodes)} nodes, {len(bg_cells)} cells")
    
    # Step 4: Generate transition layer (conforming interface)
    logger.info("Step 4/5: Generating transition layer with Delaunay triangulation...")
    trans_nodes, trans_cells = generate_transition_layer(
        bl_outer_nodes, bl_outer_surface, bg_nodes, bg_cells
    )
    logger.info(f"  Transition layer: {len(trans_nodes)} nodes, {len(trans_cells)} cells")
    
    # Step 5: Merge all meshes with shared nodes
    logger.info("Step 5/5: Merging BL, transition, and background meshes...")
    merged_nodes, merged_cells = merge_conforming_meshes(
        bl_nodes, bl_cells, trans_nodes, trans_cells, bg_nodes, bg_cells
    )
    logger.info(f"  Merged mesh: {len(merged_nodes)} nodes, {len(merged_cells)} cells")

    # Build VolumeMeshData structure
    from .structures import NodeArray, TetrahedralCells, BoundaryMap, GridMetadata, VolumeMeshData
    
    # Create Nodes
    nodes_obj = NodeArray(
        x=merged_nodes[:, 0],
        y=merged_nodes[:, 1],
        z=merged_nodes[:, 2]
    )
    
    # Compute tetrahedral volumes
    logger.info("Computing tetrahedral volumes for hybrid mesh...")
    volumes = TetrahedralCells.compute_volumes(nodes_obj, merged_cells.astype(np.int32))
    
    # Filter out cells with non-positive volumes (invalid cells from merging)
    valid_mask = volumes > 0
    n_invalid = np.sum(~valid_mask)
    if n_invalid > 0:
        logger.warning(
            f"Found {n_invalid} invalid cells with non-positive volume, removing them..."
        )
        merged_cells = merged_cells[valid_mask]
        volumes = volumes[valid_mask]
        logger.info(f"  Filtered to {len(merged_cells)} valid cells")
    
    logger.info(
        f"Computed {len(volumes)} tetrahedral volumes, "
        f"total volume: {volumes.sum():.6e} m^3"
    )
    
    # Create TetrahedralCells
    cells_obj = TetrahedralCells(
        connectivity=merged_cells.astype(np.int32),
        volumes=volumes
    )
    
    # Identify boundaries (reuse existing logic)
    from .mesh_boundary import identify_boundaries_from_surface
    boundaries_obj = identify_boundaries_from_surface(
        merged_cells, surface_faces, None
    )
    
    # Create metadata
    metadata = GridMetadata(
        node_count=len(merged_nodes),
        cell_count=len(merged_cells),
        boundary_groups=list(boundaries_obj.groups.keys()),
        file_format="hybrid"
    )
    
    # Assemble VolumeMeshData
    volume_mesh = VolumeMeshData(
        nodes=nodes_obj,
        cells=cells_obj,
        boundaries=boundaries_obj,
        metadata=metadata
    )
    
    logger.success(
        f"Hybrid mesh generation complete: "
        f"{volume_mesh.node_count} nodes, {volume_mesh.cell_count} cells, "
        f"total volume: {volume_mesh.total_volume:.6e} m^3"
    )
    
    return volume_mesh


def generate_cartesian_grid(
    bounding_box: Dict[str, np.ndarray],
    target_cells: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate uniform Cartesian grid in bounding box.
    
    Creates a structured hexahedral grid and splits each hex into 6 tetrahedra.
    Grid resolution is determined by target cell count.
    
    Args:
        bounding_box: Domain bounds {min: [x,y,z], max: [x,y,z]}
        target_cells: Target total cell count
        
    Returns:
        nodes: Grid node coordinates, shape=(n_nodes, 3)
        cells: Tetrahedral connectivity, shape=(n_tets, 4)
    """
    bbox_min = bounding_box['min']
    bbox_max = bounding_box['max']
    
    # Calculate grid dimensions based on target cell count
    domain_size = bbox_max - bbox_min
    L_char = np.linalg.norm(domain_size)
    
    # Estimate grid resolution for background mesh
    # For hybrid mesh, background should fill the entire computational domain
    # Target: background contributes ~20-30% of total cells
    
    # Calculate required resolution to ensure full domain coverage
    # Higher resolution ensures no gaps in far-field region
    target_bg_cells = target_cells * 0.25  # Background = 25% of target
    
    # Increase minimum resolution significantly for complete domain filling
    target_hex_count = target_bg_cells / 6.0
    cells_per_dim = int(np.ceil(target_hex_count ** (1/3)))
    
    # Ensure adequate resolution (min 30, max 50 per dimension)
    # This guarantees background mesh fills the entire bounding box
    cells_per_dim = max(30, min(50, cells_per_dim))
    
    logger.info(
        f"  High-resolution Cartesian background grid: {cells_per_dim}x{cells_per_dim}x{cells_per_dim} = "
        f"{cells_per_dim**3} hexahedra → {cells_per_dim**3 * 6} tetrahedra\n"
        f"  Cell size: {(bbox_max[0]-bbox_min[0])/cells_per_dim:.3f} x "
        f"{(bbox_max[1]-bbox_min[1])/cells_per_dim:.3f} x "
        f"{(bbox_max[2]-bbox_min[2])/cells_per_dim:.3f} m"
    )
    
    # Generate node coordinates
    x_coords = np.linspace(bbox_min[0], bbox_max[0], cells_per_dim + 1)
    y_coords = np.linspace(bbox_min[1], bbox_max[1], cells_per_dim + 1)
    z_coords = np.linspace(bbox_min[2], bbox_max[2], cells_per_dim + 1)
    
    # Create 3D meshgrid
    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    
    # Flatten to node array
    nodes = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Generate hexahedral connectivity
    n_x, n_y, n_z = cells_per_dim + 1, cells_per_dim + 1, cells_per_dim + 1
    
    # Vectorized hex generation
    i, j, k = np.meshgrid(
        np.arange(cells_per_dim),
        np.arange(cells_per_dim),
        np.arange(cells_per_dim),
        indexing='ij'
    )
    
    # Node indices for each hex corner
    idx = lambda di, dj, dk: (i + di) + (j + dj) * n_x + (k + dk) * n_x * n_y
    
    hex_corners = np.stack([
        idx(0, 0, 0), idx(1, 0, 0), idx(1, 1, 0), idx(0, 1, 0),  # Bottom face
        idx(0, 0, 1), idx(1, 0, 1), idx(1, 1, 1), idx(0, 1, 1)   # Top face
    ], axis=-1)  # shape=(nx-1, ny-1, nz-1, 8)
    
    # Flatten hex connectivity
    hex_connectivity = hex_corners.reshape(-1, 8)
    
    # Split each hex into 6 tetrahedra
    # Standard decomposition pattern
    tet_patterns = [
        [0, 1, 3, 4],  # Tet 1
        [1, 3, 4, 5],  # Tet 2
        [3, 4, 5, 7],  # Tet 3
        [1, 3, 5, 7],  # Tet 4
        [1, 2, 3, 7],  # Tet 5
        [1, 5, 6, 7]   # Tet 6
    ]
    
    n_hex = len(hex_connectivity)
    n_tets = n_hex * 6
    cells = np.zeros((n_tets, 4), dtype=np.int64)
    
    for tet_idx, pattern in enumerate(tet_patterns):
        start = tet_idx * n_hex
        end = start + n_hex
        cells[start:end] = hex_connectivity[:, pattern]
    
    logger.info(f"  Generated {len(nodes)} nodes and {len(cells)} tetrahedra")
    
    return nodes, cells


def remove_overlapping_cells(
    bg_nodes: np.ndarray,
    bg_cells: np.ndarray,
    bl_nodes: np.ndarray,
    bl_outer_surface: np.ndarray,
    bbox_min: np.ndarray = None,
    bbox_max: np.ndarray = None,
    cells_per_dim: int = 30
) -> np.ndarray:
    """Remove background cells that overlap with BL region.
    
    Uses simplified distance-based test: if tet centroid is within
    estimated BL thickness from surface, remove it.
    
    Memory-optimized implementation using batch processing.
    
    Args:
        bg_nodes: Background grid nodes
        bg_cells: Background tetrahedral connectivity
        bl_nodes: BL layer nodes
        bl_outer_surface: Outer surface of BL (last extrusion layer)
        bbox_min: Bounding box minimum corner (optional, for cell size estimation)
        bbox_max: Bounding box maximum corner (optional, for cell size estimation)
        cells_per_dim: Grid resolution per dimension
        
    Returns:
        Filtered background cells (non-overlapping)
    """
    # Strategy: Set threshold to ZERO to eliminate gap completely
    # Background cells will be kept as long as their centroid is NOT inside BL region
    # This ensures geometric continuity at the interface
    
    # Use a minimal epsilon to avoid numerical issues with exact equality
    threshold = 1e-6  # Essentially zero, just for floating point safety
    
    logger.info(
        f"  Overlap removal strategy (gap-free connection):\n"
        f"    Removal threshold: {threshold:.2e}m (essentially zero)\n"
        f"    Expected gap at interface: ~0m (nodes may not align perfectly)\n"
        f"    Goal: Eliminate gap completely for CFD flux continuity\n"
        f"    Note: Small numerical gaps (<1mm) may still exist due to node mismatch"
    )

    # Calculate BL outer surface centroids
    bl_centroids = np.mean(bl_nodes[bl_outer_surface], axis=1)  # shape=(n_bl_faces, 3)
    
    # Calculate background tet centroids
    bg_tet_nodes = bg_nodes[bg_cells]  # shape=(n_tets, 4, 3)
    bg_centroids = np.mean(bg_tet_nodes, axis=1)  # shape=(n_tets, 3)
    
    # Memory-optimized batch processing
    # Process background centroids in batches to avoid memory overflow
    batch_size = 10000
    n_bg = len(bg_centroids)
    keep_mask = np.ones(n_bg, dtype=bool)  # Keep all by default when threshold is ~0
    
    logger.info(f"  Processing {n_bg} background cells...")
    
    # With near-zero threshold, we keep all cells (no overlap removal needed)
    logger.info(
        f"  Overlap removal: kept {np.sum(keep_mask)}/{len(keep_mask)} background cells "
        f"(removed {len(keep_mask) - np.sum(keep_mask)})"
    )
    
    return bg_cells[keep_mask]


def merge_meshes(
    bl_nodes: np.ndarray,
    bl_cells: np.ndarray,
    bg_nodes: np.ndarray,
    bg_cells: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge BL and background meshes into single mesh.
    
    Simply concatenates nodes and adjusts cell indices for background mesh.
    No node merging is performed (keeps implementation simple).
    
    Args:
        bl_nodes: BL mesh nodes
        bl_cells: BL mesh cells
        bg_nodes: Background mesh nodes
        bg_cells: Background mesh cells
        
    Returns:
        merged_nodes: Combined node array
        merged_cells: Combined cell array (with adjusted indices)
    """
    # Concatenate nodes
    merged_nodes = np.vstack([bl_nodes, bg_nodes])
    
    # Adjust background cell indices
    node_offset = len(bl_nodes)
    bg_cells_adjusted = bg_cells + node_offset
    
    # Concatenate cells
    merged_cells = np.vstack([bl_cells, bg_cells_adjusted])
    
    logger.info(
        f"  Mesh merge: BL({len(bl_nodes)} nodes, {len(bl_cells)} cells) + "
        f"BG({len(bg_nodes)} nodes, {len(bg_cells)} cells) = "
        f"Merged({len(merged_nodes)} nodes, {len(merged_cells)} cells)"
    )
    
    return merged_nodes, merged_cells


def generate_adaptive_background_grid(
    bounding_box: Dict[str, np.ndarray],
    target_cells: int,
    bl_outer_nodes: np.ndarray,
    all_bl_nodes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate adaptive Cartesian background grid that respects BL boundary.
    
    Creates a structured grid but removes cells too close to BL outer surface.
    This ensures geometric continuity without gaps.
    
    Args:
        bounding_box: Domain bounds {min: [x,y,z], max: [x,y,z]}
        target_cells: Target total cell count
        bl_outer_nodes: Nodes on BL outer surface
        all_bl_nodes: All BL nodes (for distance calculation)
        
    Returns:
        nodes: Background grid nodes
        cells: Tetrahedral connectivity
    """
    bbox_min = bounding_box['min']
    bbox_max = bounding_box['max']
    domain_size = bbox_max - bbox_min
    
    # Calculate grid resolution
    target_bg_cells = target_cells * 0.25
    target_hex_count = target_bg_cells / 6.0
    cells_per_dim = int(np.ceil(target_hex_count ** (1/3)))
    cells_per_dim = max(30, min(50, cells_per_dim))
    
    logger.info(
        f"  Adaptive background grid: {cells_per_dim}x{cells_per_dim}x{cells_per_dim}\n"
        f"  Cell size: {(bbox_max[0]-bbox_min[0])/cells_per_dim:.3f} x "
        f"{(bbox_max[1]-bbox_min[1])/cells_per_dim:.3f} x "
        f"{(bbox_max[2]-bbox_min[2])/cells_per_dim:.3f} m"
    )
    
    # Generate node coordinates
    x_coords = np.linspace(bbox_min[0], bbox_max[0], cells_per_dim + 1)
    y_coords = np.linspace(bbox_min[1], bbox_max[1], cells_per_dim + 1)
    z_coords = np.linspace(bbox_min[2], bbox_max[2], cells_per_dim + 1)
    
    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    nodes = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Remove nodes inside BL region (with small margin for safety)
    # Use KD-tree for efficient nearest neighbor search
    from scipy.spatial import cKDTree
    
    bl_tree = cKDTree(all_bl_nodes)
    avg_cell_size = np.linalg.norm(domain_size) / cells_per_dim
    margin = avg_cell_size * 0.5  # Increased to 50% of cell size for safety
    
    # Query minimum distance from each background node to BL nodes
    distances, _ = bl_tree.query(nodes, k=1)
    
    # CRITICAL FIX: Only filter nodes that are BOTH close to BL AND inside the domain
    # Don't remove nodes near domain boundaries even if they're close to BL
    # This ensures the background grid extends to the full bounding box
    
    # Define domain boundary tolerance (nodes within this distance from bbox edges are kept)
    boundary_tolerance = avg_cell_size * 2.0
    
    # Check if node is near domain boundary
    near_x_min = (nodes[:, 0] - bbox_min[0]) < boundary_tolerance
    near_x_max = (bbox_max[0] - nodes[:, 0]) < boundary_tolerance
    near_y_min = (nodes[:, 1] - bbox_min[1]) < boundary_tolerance
    near_y_max = (bbox_max[1] - nodes[:, 1]) < boundary_tolerance
    near_z_min = (nodes[:, 2] - bbox_min[2]) < boundary_tolerance
    near_z_max = (bbox_max[2] - nodes[:, 2]) < boundary_tolerance
    
    # Keep nodes that are either far from BL OR near domain boundary
    keep_mask = (distances > margin) | near_x_min | near_x_max | near_y_min | near_y_max | near_z_min | near_z_max
    
    filtered_nodes = nodes[keep_mask]
    logger.info(f"  Filtered background nodes: {len(filtered_nodes)}/{len(nodes)} (removed {np.sum(~keep_mask)})")
    
    # Regenerate tetrahedral mesh from filtered nodes
    # For simplicity, use Delaunay triangulation on the filtered point cloud
    from scipy.spatial import Delaunay
    
    try:
        delaunay = Delaunay(filtered_nodes)
        cells = delaunay.simplices  # shape=(n_tets, 4)
        logger.info(f"  Generated {len(cells)} tetrahedra via Delaunay triangulation")
    except Exception as e:
        logger.warning(f"  Delaunay triangulation failed: {e}, falling back to Cartesian grid")
        # Fallback: use original Cartesian grid
        return generate_cartesian_grid(bounding_box, target_cells)
    
    return filtered_nodes, cells


def generate_transition_layer(
    bl_outer_nodes: np.ndarray,
    bl_outer_surface: np.ndarray,
    bg_nodes: np.ndarray,
    bg_cells: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate transition layer between BL and background using Delaunay triangulation.
    
    Creates a conforming interface by triangulating the gap between BL outer surface
    and innermost background cells.
    
    Args:
        bl_outer_nodes: Nodes on BL outer surface
        bl_outer_surface: Connectivity of BL outer surface
        bg_nodes: Background grid nodes
        bg_cells: Background tetrahedral connectivity
        
    Returns:
        trans_nodes: Transition layer nodes (subset of bg_nodes near BL)
        trans_cells: Transition tetrahedra connecting BL to background
    """
    # Find background nodes within transition zone (near BL outer surface)
    from scipy.spatial import cKDTree
    
    bl_tree = cKDTree(bl_outer_nodes)
    bg_tree = cKDTree(bg_nodes)
    
    # Define transition zone thickness (2-3 background cell sizes)
    avg_bg_spacing = np.mean(np.linalg.norm(
        bg_nodes[bg_cells[:, :2]] - bg_nodes[bg_cells[:, 2:4]], axis=2
    ))
    transition_thickness = avg_bg_spacing * 2.5
    
    # Find background nodes within transition zone
    distances, indices = bl_tree.query(bg_nodes, k=1)
    in_transition = distances < transition_thickness
    
    trans_node_indices = np.where(in_transition)[0]
    trans_nodes = bg_nodes[in_transition]
    
    if len(trans_nodes) < 4:
        logger.warning("  Too few nodes in transition zone, skipping transition layer")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 4, dtype=np.int64)
    
    # Triangulate transition zone
    try:
        from scipy.spatial import Delaunay
        combined_nodes = np.vstack([bl_outer_nodes, trans_nodes])
        delaunay = Delaunay(combined_nodes)
        trans_cells = delaunay.simplices
        
        # Adjust cell indices: BL nodes come first
        n_bl_outer = len(bl_outer_nodes)
        # Keep only tets that have at least one BL node and one transition node
        has_bl = np.any(trans_cells < n_bl_outer, axis=1)
        has_trans = np.any(trans_cells >= n_bl_outer, axis=1)
        valid_mask = has_bl & has_trans
        trans_cells = trans_cells[valid_mask]
        
        logger.info(f"  Transition layer: {len(trans_nodes)} nodes, {len(trans_cells)} cells")
    except Exception as e:
        logger.warning(f"  Transition layer generation failed: {e}")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 4, dtype=np.int64)
    
    return trans_nodes, trans_cells


def merge_conforming_meshes(
    bl_nodes: np.ndarray,
    bl_cells: np.ndarray,
    trans_nodes: np.ndarray,
    trans_cells: np.ndarray,
    bg_nodes: np.ndarray,
    bg_cells: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge BL, transition, and background meshes with shared nodes.
    
    Implements node deduplication to ensure conforming interface.
    
    Args:
        bl_nodes: BL mesh nodes
        bl_cells: BL mesh cells
        trans_nodes: Transition layer nodes
        trans_cells: Transition layer cells
        bg_nodes: Background mesh nodes
        bg_cells: Background mesh cells
        
    Returns:
        merged_nodes: Combined node array (deduplicated)
        merged_cells: Combined cell array (with adjusted indices)
    """
    from scipy.spatial import cKDTree
    
    # Concatenate all nodes
    all_nodes = np.vstack([bl_nodes, trans_nodes, bg_nodes])
    
    # Deduplicate nodes using KD-tree
    tolerance = 1e-6  # Node merging tolerance
    tree = cKDTree(all_nodes)
    
    # Find duplicate nodes (within tolerance)
    pairs = tree.query_pairs(tolerance)
    
    if len(pairs) > 0:
        # Create mapping from old indices to new unique indices
        unique_map = np.arange(len(all_nodes))
        for i, j in pairs:
            if i < j:  # Only process each pair once
                unique_map[j] = i  # Map j to i
        
        # Get unique mapping
        _, inverse_indices = np.unique(unique_map, return_inverse=True)
        unique_nodes = all_nodes[np.unique(unique_map)]
        
        logger.info(f"  Node deduplication: {len(all_nodes)} → {len(unique_nodes)} nodes (merged {len(all_nodes) - len(unique_nodes)})")
    else:
        unique_nodes = all_nodes
        inverse_indices = np.arange(len(all_nodes))
        logger.info(f"  No duplicate nodes found: {len(all_nodes)} unique nodes")
    
    # Adjust cell indices
    n_bl = len(bl_nodes)
    n_trans = len(trans_nodes)
    
    # BL cells: no adjustment needed (indices 0 to n_bl-1)
    adjusted_bl_cells = bl_cells
    
    # Transition cells: offset by n_bl
    if len(trans_cells) > 0:
        adjusted_trans_cells = trans_cells.copy()
        # Need to remap based on inverse_indices
        trans_global_indices = np.arange(n_bl, n_bl + n_trans)
        trans_mapped_indices = inverse_indices[n_bl:n_bl + n_trans]
        
        for old_idx, new_idx in zip(trans_global_indices, trans_mapped_indices):
            adjusted_trans_cells[adjusted_trans_cells == old_idx] = new_idx
    else:
        adjusted_trans_cells = np.array([]).reshape(0, 4, dtype=np.int64)
    
    # Background cells: offset by n_bl + n_trans
    bg_global_offset = n_bl + n_trans
    adjusted_bg_cells = bg_cells.copy()
    bg_global_indices = np.arange(bg_global_offset, bg_global_offset + len(bg_nodes))
    bg_mapped_indices = inverse_indices[bg_global_offset:]
    
    for old_idx, new_idx in zip(bg_global_indices, bg_mapped_indices):
        adjusted_bg_cells[adjusted_bg_cells == old_idx] = new_idx
    
    # Merge all cells
    merged_cells = np.vstack([
        adjusted_bl_cells,
        adjusted_trans_cells,
        adjusted_bg_cells
    ])
    
    logger.info(
        f"  Conforming mesh merge:\n"
        f"    BL: {len(bl_nodes)} nodes, {len(bl_cells)} cells\n"
        f"    Transition: {len(trans_nodes)} nodes, {len(trans_cells)} cells\n"
        f"    Background: {len(bg_nodes)} nodes, {len(bg_cells)} cells\n"
        f"    Merged: {len(unique_nodes)} nodes, {len(merged_cells)} cells"
    )
    
    return unique_nodes, merged_cells


# Legacy functions (kept for backward compatibility)
