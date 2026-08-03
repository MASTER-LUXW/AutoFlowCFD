"""Merged BL + tetgen-core mesh assembly for one generation attempt.

_build_merged_mesh does the actual per-attempt work generate_hybrid_mesh
(mesh_background.py) orchestrates: classify boundary groups, extrude BL
layers, tetgen-fill the remaining core volume, and splice the two into one
merged (nodes, cells) pair with per-cell source-group attribution. Split
into its own module purely to keep mesh_background.py's own file size
down - there is no independent reuse of this function outside
generate_hybrid_mesh's own retry loop (Stage B), which is why it stays
private (leading underscore) and lives right next to its only caller's
module.
"""

import numpy as np
from typing import Dict, Optional, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ..structures import BoundaryMap

from .mesh_extrusion import extrude_layers
from .mesh_prism_to_tet import convert_layers_to_tetrahedra, orient_tetrahedra
from .mesh_utils import compute_face_normals
from .mesh_domain_classify import classify_boundary_groups
from .mesh_tetgen_core import (
    build_seam_taper_scale, fill_core_volume,
    compute_local_thickness_limit, repair_nonmanifold_cells,
    attribute_cells_from_trifaces,
)
from .mesh_repair import compute_bl_thickness_limit_override


def _build_merged_mesh(
    surface_nodes: np.ndarray,
    surface_faces: np.ndarray,
    bounding_box: Dict[str, np.ndarray],
    growth_rate: float,
    max_layers: int,
    min_cell_size: float,
    surface_boundaries: 'BoundaryMap',
    max_cell_size: Optional[float],
    extra_thickness_limit: Optional[np.ndarray],
    bl_layers: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Steps 1-3 of generate_hybrid_mesh: classify boundary groups, extrude
    BL layers, tetgen-fill the core, and merge - everything up to (but not
    including) re-orientation/volume computation/quality repair.

    Split into its own function (rather than inline in generate_hybrid_mesh)
    so every large intermediate array it allocates (bl_nodes, core_tets,
    plc_points/plc_faces, etc. - each comparable in size to the final
    merged mesh) is scoped to *this* call and gets freed the moment it
    returns, instead of lingering in generate_hybrid_mesh's own frame.
    That distinction matters specifically for the Stage B retry there:
    `return generate_hybrid_mesh(...)` from inside a single monolithic
    function keeps every one of these intermediates alive on the call
    stack for the retry's entire duration, since Python has no tail-call
    elimination - observed directly on a real 2.6M-cell case, process RSS
    climbing past 11GB and a step that normally takes ~2s
    (compute_local_thickness_limit) taking 10+ minutes under the resulting
    memory pressure. With the mesh-building work isolated here,
    generate_hybrid_mesh only ever holds the *current* attempt's merged
    mesh, not every attempt's raw intermediates simultaneously.

    Returns:
        (merged_nodes, merged_cells, cell_groups, n_bl_cells, source_vertex,
        bl_extrude_faces) - source_vertex is (nodes_per_BL_layer,), identity
        (np.arange) - kept as an explicit return (rather than the caller
        just assuming identity) so Stage B's own node-to-vertex bookkeeping
        (mesh_repair.compute_bl_thickness_limit_override) stays correct if
        a future BL-generation path ever needs a non-identity mapping again.
        bl_extrude_faces is the BL surface connectivity in that same
        layer-local index space - empty (0, 3) when there's no BL region at
        all - the caller needs it too, to taper Stage B's own cap across
        mesh neighbours instead of a hard cliff (see
        compute_bl_thickness_limit_override's local_surface_faces doc).
    """
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
        n_bl_cells = 0
        source_vertex = np.arange(len(surface_nodes))
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
        if extra_thickness_limit is not None:
            thickness_limit = np.minimum(thickness_limit, extra_thickness_limit)

        # source_vertex[i] == i: local (within-a-layer) index i is already
        # an original-surface vertex index, no translation needed. Kept as
        # an explicit identity array (rather than the caller assuming it)
        # so Stage B's own node-to-vertex bookkeeping
        # (mesh_repair.compute_bl_thickness_limit_override) has a real
        # array to index with regardless of how BL nodes were produced.
        source_vertex = np.arange(n_surface_nodes)

        normals = compute_face_normals(surface_nodes, extrude_faces)
        bl_nodes, bl_layer_conn = extrude_layers(
            surface_nodes, extrude_faces, normals,
            bounding_box={'min': bbox_min, 'max': bbox_max},
            growth_rate=growth_rate, max_layers=max_layers, min_cell_size=min_cell_size,
            taper_scale=taper_scale, thickness_limit=thickness_limit,
            # Couples the transition layers' growth to the core fill's own
            # target cell size (see extrude_layers' target_handoff_size doc)
            # so the BL/core interface facet size and the core's target
            # size aren't an uncoordinated, potentially large jump - a
            # common source of skewed/sliver cells right at that interface.
            # None when max_cell_size itself is unset, matching prior
            # (fixed growth_rate*1.25) behavior exactly.
            target_handoff_size=max_cell_size,
            bl_layers=bl_layers,
        )
        bl_cells, bl_face_of_cell = convert_layers_to_tetrahedra(
            bl_nodes, bl_layer_conn, extrude_faces
        )
        n_bl_cells = len(bl_cells)
        logger.info(f"  BL mesh: {len(bl_nodes)} nodes, {len(bl_cells)} cells")

        # Attribute each BL cell directly back to its source boundary group
        # via position, not node-index matching against the pre-extrusion
        # surface: convert_layers_to_tetrahedra's own bl_face_of_cell maps
        # every surviving cell back to its extrude_faces row directly (a
        # plain tile - one contiguous block of len(extrude_faces) tets per
        # (layer, quad) pair, cell i -> extrude_faces[i % len(extrude_faces)]
        # - no longer holds now that function can drop analytically
        # zero-volume connector-face tets, see its own docstring). This is
        # exact for every surviving BL cell, including the vast majority of
        # body/ground's own outer surface that node-index matching can never
        # reach (see mesh_boundary.py - those nodes get a brand-new offset
        # index once genuinely displaced by extrusion, so their
        # post-extrusion face can't match anything in a lookup built from
        # the original, pre-extrusion node indices).
        bl_cell_groups = extrude_face_groups[bl_face_of_cell]

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
            # needs (and vice versa). Stage B's core-side local repair
            # regions hit the same limitation from the other direction
            # (removed - see mesh_repair.py's module docstring: a handful
            # of small local regions alongside this one could balloon the
            # whole core fill several-fold instead of staying local). A
            # single flat region covering the entire core sidesteps that
            # competition entirely (only one region to fund, always true
            # now), and its actual accuracy is governed by
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

        # n_shared must match len(plc_points) (== outer_nodes ==
        # nodes_per_layer) - always equal to n_surface_nodes in practice,
        # but computed from outer_nodes directly rather than assumed, since
        # plc_points is exactly outer_nodes and that's what core_tets'
        # shared-index range actually corresponds to.
        n_shared = len(outer_nodes)
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

    return merged_nodes, merged_cells, cell_groups, n_bl_cells, source_vertex, extrude_faces
