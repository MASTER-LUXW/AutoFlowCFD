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
from .mesh_tetgen_core import (
    build_seam_taper_scale, fill_core_volume,
    compute_local_thickness_limit, repair_nonmanifold_cells,
    attribute_cells_from_trifaces,
)


def generate_hybrid_mesh(
    surface_nodes: np.ndarray,
    surface_faces: np.ndarray,
    bounding_box: Dict[str, np.ndarray],
    growth_rate: float = 1.2,
    max_layers: int = 30,
    min_cell_size: float = 0.001,
    target_cells: int = 500000,
    surface_boundaries: Optional['BoundaryMap'] = None,
    max_cell_size: Optional[float] = None,
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
        max_cell_size: Optional hard cap (meters) on core-region cell size,
            applied uniformly across the whole core fill via a single
            tetgen region constraint. tetgen's core fill otherwise has no
            size cap at all beyond shape-quality bounds, so cells can grow
            as large as whatever coarse far-field input facet (e.g. a
            sparsely-triangulated tunnel/inlet/outlet wall) happens to
            allow. None disables the cap entirely (unbounded core cell
            size, matching prior behavior exactly). A distance-graded
            (fine near the wall, coarsening outward) version was tried and
            abandoned - tetgen's per-region refinement does not reliably
            converge multiple simultaneous regions at real-world scale;
            see the comment at this parameter's use site below for the
            specific evidence.

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
    (extrude_faces, core_faces, extruded_groups, extrude_face_groups,
     hole_points, core_face_groups, _is_closed_solid_face) = classify_boundary_groups(
        surface_nodes, surface_faces, surface_boundaries, bbox_min, bbox_max
    )

    # Marker IDs for tetgen's facet-marker mechanism (attribute_cells_from_trifaces)
    # - only needed when max_cell_size grading is active (it's the only thing
    # that switches fill_core_volume's nobisect off, which is what breaks the
    # plain node-index-matching boundary attribution for subdivided facets).
    # 0 is reserved by tetgen for "no marker" (an unmarked/interior facet).
    group_name_to_marker = {name: i + 1 for i, name in enumerate(surface_boundaries.groups.keys())}
    marker_to_name = {v: k for k, v in group_name_to_marker.items()}

    if len(extrude_faces) == 0:
        logger.warning(
            "No boundary group was eligible for BL extrusion; filling the "
            "entire closed surface directly with tetgen (no boundary layer)"
        )
        face_markers = None
        regions = None
        if max_cell_size is not None:
            face_group_name = np.full(len(surface_faces), '', dtype=object)
            for name, idx in surface_boundaries.groups.items():
                face_group_name[idx] = name
            face_markers = np.array(
                [group_name_to_marker.get(n, 0) for n in face_group_name], dtype=np.int32
            )
            center = surface_nodes.mean(axis=0)
            regions = [(center, 1, max_cell_size ** 3 * 0.15)]
        core_nodes, core_tets, trifaces, triface_markers = fill_core_volume(
            surface_nodes, surface_faces, holes=hole_points,
            regions=regions, face_markers=face_markers,
        )
        merged_nodes, merged_cells = core_nodes, core_tets
        if face_markers is not None:
            cell_groups = attribute_cells_from_trifaces(
                core_tets, trifaces, triface_markers, marker_to_name
            )
        else:
            cell_groups = np.full(len(merged_cells), '', dtype=object)
    else:
        logger.info(
            f"Step 2/4: Extruding BL layers from {len(extrude_faces)} faces "
            f"(groups: {extruded_groups})..."
        )
        n_surface_nodes = len(surface_nodes)
        taper_scale = build_seam_taper_scale(n_surface_nodes, extrude_faces, core_faces)

        # Cap each node's cumulative BL thickness near tight facing features
        # (e.g. a body's underbody close to the ground) so the two fronts
        # freeze before they can cross, instead of relying entirely on
        # repair_nonmanifold_cells to clean up the resulting overlap after
        # the fact (see compute_local_thickness_limit's own docstring for
        # why this is a strong mitigation, not a formal guarantee).
        domain_size = float(np.linalg.norm(bbox_max - bbox_min))
        thickness_limit = compute_local_thickness_limit(
            surface_nodes, extrude_faces, np.unique(extrude_faces), domain_size
        )

        normals = compute_face_normals(surface_nodes, extrude_faces)
        bl_nodes, bl_layer_conn = extrude_layers(
            surface_nodes, extrude_faces, normals,
            bounding_box={'min': bbox_min, 'max': bbox_max},
            growth_rate=growth_rate, max_layers=max_layers, min_cell_size=min_cell_size,
            taper_scale=taper_scale, thickness_limit=thickness_limit
        )
        bl_cells = convert_layers_to_tetrahedra(bl_nodes, bl_layer_conn, extrude_faces)
        logger.info(f"  BL mesh: {len(bl_nodes)} nodes, {len(bl_cells)} cells")

        # Attribute each BL cell directly back to its source boundary group
        # via position, not node-index matching against the pre-extrusion
        # surface: convert_layers_to_tetrahedra emits, for every (layer,
        # quad) pair, one contiguous block of len(extrude_faces) tets in the
        # SAME face order as extrude_faces - so cell i always corresponds to
        # extrude_faces[i % len(extrude_faces)], regardless of how many
        # layers or quads there are. This is exact for every BL cell,
        # including the vast majority of body/ground's own outer surface
        # that node-index matching can never reach (see mesh_boundary.py -
        # those nodes get a brand-new offset index once genuinely displaced
        # by extrusion, so their post-extrusion face can't match anything in
        # a lookup built from the original, pre-extrusion node indices).
        n_base_faces = len(extrude_faces)
        n_tets_per_face = len(bl_cells) // n_base_faces
        bl_cell_groups = np.tile(extrude_face_groups, n_tets_per_face)

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

        # bl_outer_surface's portion is marked with its own source group
        # too (extrude_face_groups), not left at 0/unmarked. It's normally
        # a purely internal BL/core interface (a face there is shared
        # between one BL cell and one core cell, never exposed to the
        # domain exterior) so this is usually redundant with bl_cell_groups
        # - but a wall entirely pinned by the seam taper (e.g. every node
        # of a coarse, corner-only wall facet happens to sit within
        # taper_rings hops of a seam) can collapse to zero BL thickness,
        # at which point its "outer surface" IS the real exposed boundary.
        # Marking it directly makes attribute_cells_from_trifaces recover
        # that case correctly too, instead of only the node-index-matching
        # fallback (which breaks once nobisect=False lets tetgen subdivide
        # that now-oversized, previously-2-triangle wall facet, producing
        # sub-triangles the pre-fill node lookup was never built to match).
        face_markers = None
        regions = None
        if max_cell_size is not None:
            # A distance-graded multi-tier sphere scheme was tried and
            # abandoned: tetgen's per-region variable-volume refinement
            # does not reliably converge multiple simultaneous regions to
            # their own targets when they compete for one shared Steiner
            # budget - small/aggressive inner regions can starve a
            # necessary large outer/far-field region of the points it
            # needs (and vice versa). A single flat region covering the
            # entire core sidesteps that competition entirely (only one
            # region to fund), and its actual accuracy is governed by
            # fill_core_volume's own Steiner-point budget, which is now
            # sized from the region's real volume/max_cell_size ratio
            # rather than a fixed constant (a fixed budget was accurate
            # to ~1.2x target only for domain/max_cell_size ratios small
            # enough for it to be sufficient; measured directly on a
            # 5.5x3x3 m domain capped at 0.05 m, a fixed 300,000-point
            # budget left a worst-case cell ~5.8x over target - see
            # mesh_tetgen_core.fill_core_volume's steinerleft comment).
            bl_outer_markers = np.array(
                [group_name_to_marker.get(n, 0) for n in extrude_face_groups], dtype=np.int32
            )
            core_markers = np.array(
                [group_name_to_marker.get(n, 0) for n in core_face_groups], dtype=np.int32
            )
            face_markers = np.concatenate([bl_outer_markers, core_markers])
            regions = [(outer_nodes.mean(axis=0), 1, max_cell_size ** 3 * 0.15)]

        core_nodes, core_tets, trifaces, triface_markers = fill_core_volume(
            plc_points, plc_faces, holes=hole_points, regions=regions, face_markers=face_markers,
        )
        core_cell_groups = (
            attribute_cells_from_trifaces(core_tets, trifaces, triface_markers, marker_to_name)
            if face_markers is not None
            else np.full(len(core_tets), '', dtype=object)
        )

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
        # core_cell_groups (from facet markers, populated above whenever
        # max_cell_size grading is active) already identifies core cells'
        # source groups directly. When grading isn't active, it's all ''
        # placeholders instead - those core-only groups (tunnel/inlet/
        # outlet-type, never displaced by extrusion) still go through
        # identify_boundaries_from_surface's node-index matching below,
        # which works correctly for them since their nodes were never
        # remapped in that case (nobisect=True preserves them verbatim).
        cell_groups = np.concatenate([bl_cell_groups, core_cell_groups])
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
    #
    # The threshold must scale with the mesh's own cell size, not be a fixed
    # absolute constant: an automotive-scale mesh (min_cell_size ~1e-3 to
    # 1e-2 m) has legitimate cell volumes around 1e-8 to 1e-6 m^3, so a fixed
    # 1e-15 m^3 cutoff is 7-8 orders of magnitude below the smallest real
    # cell - it was effectively a no-op, letting genuinely collapsed slivers
    # (volume many orders of magnitude below any legitimate cell, but still
    # "> 1e-15") through as if they were valid. Anchor it to min_cell_size
    # instead, generously below (1e-6x) the smallest intended cell volume so
    # it only catches true degeneracies, never a legitimately small cell.
    degenerate_threshold = (min_cell_size ** 3) * 1e-6
    valid_mask = volumes > degenerate_threshold
    n_invalid = np.sum(~valid_mask)
    if n_invalid > 0:
        logger.warning(
            f"Found {n_invalid} degenerate (near-zero volume, "
            f"< {degenerate_threshold:.3e} m^3) cells, removing them..."
        )
        merged_cells = merged_cells[valid_mask]
        volumes = volumes[valid_mask]
        cell_groups = cell_groups[valid_mask]

    # Detect and repair non-manifold faces (a face shared by more than 2
    # cells) - observed from tetgen's core fill producing a small local
    # cluster of overlapping tetrahedra at tight BL-extrusion seam features
    # (nobisect=True prevents it from resolving this by inserting boundary
    # points, see mesh_tetgen_core.fill_core_volume). Left unrepaired, this
    # is a real conservation violation: face_extractor.py can only ever
    # attribute a shared face to 2 of the 3+ cells touching it, and now
    # raises a hard error rather than silently dropping flux through it.
    nonmanifold_keep = repair_nonmanifold_cells(merged_nodes, merged_cells)
    if not nonmanifold_keep.all():
        merged_cells = merged_cells[nonmanifold_keep]
        volumes = volumes[nonmanifold_keep]
        cell_groups = cell_groups[nonmanifold_keep]

    cells_obj = TetrahedralCells(
        connectivity=merged_cells.astype(np.int32),
        volumes=volumes
    )

    from .mesh_boundary import identify_boundaries_from_surface
    boundaries_obj = identify_boundaries_from_surface(
        merged_cells, surface_faces, surface_boundaries, direct_cell_groups=cell_groups
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
