"""Domain-conforming hybrid mesh assembly (BL extrusion + tetgen core fill).

Generates a volume mesh that fills exactly the closed cavity the input
surface mesh encloses: boundary-layer (BL) prisms are extruded only from
wall-type surfaces (mesh_domain_classify picks these out via topology, not
boundary names), and the remaining interior volume is filled by a
constrained tetrahedralization (tetgen) of the exact outer boundary - the BL
outer surface plus the unmodified non-wall surfaces (inlet/outlet/tunnel/
symmetry-like boundaries). Because the tetgen fill is bounded by the real
closed surface rather than a padded bounding box, the result can never
extend outside the domain the input surface actually describes.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger

from .mesh_extrusion import extrude_layers, convert_layers_to_tetrahedra, orient_tetrahedra
from .mesh_utils import compute_face_normals
from .mesh_domain_classify import classify_boundary_groups
from .mesh_tetgen_core import build_seam_taper_scale, fill_core_volume


def generate_hybrid_mesh(
    surface_nodes: np.ndarray,
    surface_faces: np.ndarray,
    bounding_box: Dict[str, np.ndarray],
    growth_rate: float = 1.2,
    max_layers: int = 30,
    min_cell_size: float = 0.001,
    target_cells: int = 500000,
    surface_boundaries: Optional['BoundaryMap'] = None
) -> 'VolumeMeshData':
    """Generate a volume mesh that conforms exactly to the closed input surface.

    Strategy:
    1. Classify boundary groups into BL-extrude-eligible vs. core-only, using
       topology (closed-shell / bounding-box-touch analysis) rather than
       boundary names (mesh_domain_classify.classify_boundary_groups).
    2. Extrude BL layers only from eligible faces. Nodes shared with
       core-only faces (e.g. where a ground plane meets the tunnel wall) are
       pinned so the BL tapers to zero there instead of tearing the mesh
       open at that seam.
    3. Build the closed PLC (BL outer surface + unmodified core-only faces)
       and constrained-tetrahedralize the remaining core volume with tetgen
       - this can never extend outside the domain the input surface
       actually encloses (mesh_tetgen_core.fill_core_volume).
    4. Merge BL tets + core tets, defensively re-orient, drop degenerate
       cells, and inherit boundary groups from the original surface mesh
       (mesh_boundary.identify_boundaries_from_surface, unchanged).

    Args:
        surface_nodes: Surface geometry nodes, shape=(n_nodes, 3)
        surface_faces: Surface face connectivity, shape=(n_faces, 3)
        bounding_box: The input surface's exact (unpadded) extent
            {min: [x,y,z], max: [x,y,z]} - used only to decide which
            boundary groups touch the domain's outer shell and as a BL
            growth-cap reference, never to define fill geometry
        growth_rate: Geometric growth rate for BL thickness
        max_layers: Maximum number of BL layers
        min_cell_size: Minimum allowable cell size in meters
        target_cells: Unused by the tetgen-based core fill (quality is
            controlled by tetgen's own radius-edge/dihedral bounds instead of
            a cell budget) - kept for CLI/API backward compatibility
        surface_boundaries: Boundary mapping from the surface mesh (inlet/
            outlet/wall/symmetry groups); required to classify which faces
            get BL-extruded and to inherit boundary groups on the result

    Returns:
        VolumeMeshData with a domain-conforming hybrid mesh (BL + core)
    """
    if surface_boundaries is None or not surface_boundaries.groups:
        raise ValueError(
            "generate_hybrid_mesh requires surface_boundaries with at least "
            "one boundary group, used to classify wall-type surfaces for BL "
            "extrusion versus the outer domain shell"
        )

    logger.info("Starting domain-conforming hybrid mesh generation...")

    bbox_min = np.asarray(bounding_box['min'], dtype=np.float64)
    bbox_max = np.asarray(bounding_box['max'], dtype=np.float64)

    logger.info("Step 1/4: Classifying boundary groups (extrude vs. core-only)...")
    extrude_faces, core_faces, extruded_groups = classify_boundary_groups(
        surface_nodes, surface_faces, surface_boundaries, bbox_min, bbox_max
    )

    if len(extrude_faces) == 0:
        logger.warning(
            "No boundary group was eligible for BL extrusion; filling the "
            "entire closed surface directly with tetgen (no boundary layer)"
        )
        core_nodes, core_tets = fill_core_volume(surface_nodes, surface_faces)
        merged_nodes, merged_cells = core_nodes, core_tets
    else:
        logger.info(
            f"Step 2/4: Extruding BL layers from {len(extrude_faces)} faces "
            f"(groups: {extruded_groups})..."
        )
        n_surface_nodes = len(surface_nodes)
        taper_scale = build_seam_taper_scale(n_surface_nodes, extrude_faces, core_faces)

        normals = compute_face_normals(surface_nodes, extrude_faces)
        bl_nodes, bl_layer_conn = extrude_layers(
            surface_nodes, extrude_faces, normals,
            bounding_box={'min': bbox_min, 'max': bbox_max},
            growth_rate=growth_rate, max_layers=max_layers, min_cell_size=min_cell_size,
            taper_scale=taper_scale
        )
        bl_cells = convert_layers_to_tetrahedra(bl_nodes, bl_layer_conn, extrude_faces)
        logger.info(f"  BL mesh: {len(bl_nodes)} nodes, {len(bl_cells)} cells")

        n_layers = len(bl_layer_conn)
        nodes_per_layer = len(bl_nodes) // n_layers
        outer_offset = (n_layers - 1) * nodes_per_layer

        # Layer 0 keeps bare surface-node indices unchanged (see
        # convert_layers_to_tetrahedra); every later layer's block uses the
        # SAME local numbering, just offset by layer_idx * nodes_per_layer.
        # So the outer (last) layer's node-index space already matches
        # surface_faces'/core_faces' own indexing directly - no arithmetic
        # needed to align them.
        outer_nodes = bl_nodes[outer_offset:outer_offset + nodes_per_layer]
        bl_outer_surface = bl_layer_conn[-1]

        logger.info(
            f"Step 3/4: Tetrahedralizing core volume "
            f"({len(core_faces)} core-only faces + BL outer surface)..."
        )
        plc_points = outer_nodes
        plc_faces = np.vstack([bl_outer_surface, core_faces])
        core_nodes, core_tets = fill_core_volume(plc_points, plc_faces)

        # Merge: BL nodes/cells keep their own indexing untouched. core_tets
        # reference the shared boundary (indices < n_surface_nodes) plus any
        # new interior Steiner points (indices >= n_surface_nodes) tetgen
        # added. A shared index maps to its outer-layer position if that
        # node was actually displaced by extrusion (taper_scale > 0), or to
        # its bare/layer-0 position otherwise (core-only-exclusive nodes and
        # exact-seam nodes with taper_scale == 0 - both cases hold the
        # identical coordinate at either position, so routing them to the
        # bare index is what lets identify_boundaries_from_surface keep
        # matching core-only boundary faces against the original
        # surface_faces indices unchanged).
        # taper_scale defaults to 1.0 for nodes never touched by
        # extrude_faces at all (core-only-exclusive) - restrict "moved" to
        # nodes actually referenced by an extruded face, or they'd get
        # incorrectly routed to the outer-layer slot despite never moving.
        in_extrude = np.zeros(n_surface_nodes, dtype=bool)
        in_extrude[np.unique(extrude_faces)] = True
        moved_mask = in_extrude & (taper_scale > 0.0)
        shared_target = np.where(
            moved_mask,
            outer_offset + np.arange(n_surface_nodes),
            np.arange(n_surface_nodes),
        )

        n_shared = n_surface_nodes
        core_tets_remapped = core_tets.copy()
        is_shared = core_tets < n_shared
        core_tets_remapped[is_shared] = shared_target[core_tets[is_shared]]
        core_tets_remapped[~is_shared] = core_tets[~is_shared] - n_shared + len(bl_nodes)

        new_core_nodes = core_nodes[n_shared:]
        merged_nodes = np.vstack([bl_nodes, new_core_nodes])
        merged_cells = np.vstack([bl_cells, core_tets_remapped])
        logger.info(
            f"  Merged mesh: {len(merged_nodes)} nodes, {len(merged_cells)} cells "
            f"({len(bl_cells)} BL + {len(core_tets_remapped)} core)"
        )

    # Build VolumeMeshData structure
    from .structures import NodeArray, TetrahedralCells, GridMetadata, VolumeMeshData

    nodes_obj = NodeArray(
        x=merged_nodes[:, 0],
        y=merged_nodes[:, 1],
        z=merged_nodes[:, 2]
    )

    logger.info("Step 4/4: Re-orienting and computing tetrahedral volumes...")
    merged_cells = orient_tetrahedra(merged_nodes, merged_cells.astype(np.int64))
    volumes = TetrahedralCells.compute_volumes(nodes_obj, merged_cells.astype(np.int32))

    # Drop any still-degenerate (near-zero volume) cells - these cannot be
    # fixed by re-orientation and indicate collapsed/duplicate geometry
    # (e.g. a fully-tapered BL prism right at a pinned seam).
    valid_mask = volumes > 1e-15
    n_invalid = np.sum(~valid_mask)
    if n_invalid > 0:
        logger.warning(f"Found {n_invalid} degenerate (near-zero volume) cells, removing them...")
        merged_cells = merged_cells[valid_mask]
        volumes = volumes[valid_mask]

    cells_obj = TetrahedralCells(
        connectivity=merged_cells.astype(np.int32),
        volumes=volumes
    )

    from .mesh_boundary import identify_boundaries_from_surface
    boundaries_obj = identify_boundaries_from_surface(
        merged_cells, surface_faces, surface_boundaries
    )

    metadata = GridMetadata(
        node_count=len(merged_nodes),
        cell_count=len(merged_cells),
        boundary_groups=list(boundaries_obj.groups.keys()),
        file_format="hybrid"
    )

    volume_mesh = VolumeMeshData(
        nodes=nodes_obj,
        cells=cells_obj,
        boundaries=boundaries_obj,
        metadata=metadata
    )

    logger.success(
        f"Domain-conforming hybrid mesh generation complete: "
        f"{volume_mesh.node_count} nodes, {volume_mesh.cell_count} cells, "
        f"total volume: {volume_mesh.total_volume:.6e} m^3"
    )

    return volume_mesh
