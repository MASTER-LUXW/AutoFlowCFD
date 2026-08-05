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
from .mesh_prism_to_tet import convert_layers_to_prisms, convert_layers_to_tetrahedra, orient_tetrahedra
from .mesh_utils import compute_face_normals
from .mesh_domain_classify import classify_boundary_groups
from .mesh_tetgen_core import (
    build_seam_taper_scale, fill_core_volume,
    compute_local_thickness_limit, repair_nonmanifold_cells,
    attribute_cells_from_trifaces,
    CORE_TETGEN_MINRATIO, CORE_TETGEN_MINDIHEDRAL, CORE_VOLUME_CAP_FRACTION,
)
from .mesh_repair import compute_bl_thickness_limit_override


# CORE_TETGEN_MINRATIO/CORE_TETGEN_MINDIHEDRAL/CORE_VOLUME_CAP_FRACTION now
# live in mesh_tetgen_core.py (imported above) rather than here, so
# mesh_repair_cavity.py's Stage B' can use the SAME tightened standard for
# its own, much smaller fill_core_volume calls - see that constant's own
# docstring for why the inconsistency mattered in practice. Re-imported
# under these names (not aliased) purely so every existing reference below
# keeps working unchanged.


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, int]:
    """Steps 1-3 of generate_hybrid_mesh: classify boundary groups, extrude
    BL layers, tetgen-fill the core, and merge - everything up to (but not
    including) re-orientation/volume computation/quality repair.

    Only the fine near-wall stage (the first `bl_layers` of `max_layers`,
    see extrude_layers' own Stage-1/Stage-2 split) becomes genuine
    triangular prisms (PrismCells-shaped, (n,6)), kept SEPARATE from the
    tet array. The faster-growing TRANSITION stage (`bl_layers` up to
    `max_layers`, bridging up toward the core fill's own cell size) stays
    tetrahedra (mesh_prism_to_tet.convert_layers_to_tetrahedra), merged
    into `tet_cells` ahead of the core fill's own tets - this is the
    original, pre-existing design (confirmed - see ProjectFiles Part6/7):
    an earlier revision of this function mistakenly extended the true-prism
    conversion to the ENTIRE extruded stack including the transition layers,
    which is what this now restores.

    Only the true-prism portion bypasses generate_hybrid_mesh's repair
    pipeline (Stage A smoothing, Stage B' cavity re-tiling, Stage D edge
    collapse) - that pipeline is written throughout for a single (n,4) tet
    connectivity array and n_bl_cells as a row-index split within it, none
    of which understands a 6-node prism cell (prisms still go through their
    own quality validation - see quality_validator.py's prism branch - and
    Stage C's coarse global-parameter backoff one level up still applies to
    a bad prism mesh the same as ever; prism-specific repair is its own
    follow-up evaluation, not bundled into this change). The transition
    tets are ordinary members of `tet_cells`, going through the exact same
    repair pipeline the core fill's own tets do, tagged as BL-origin via
    the returned `n_transition_cells` (see below) purely for the
    quality validator's separate, more permissive BL-region aspect-ratio
    threshold (a transition-stage cell is still expected to be more
    stretched than an isotropic core cell) - same as this whole stack was
    treated before true prisms existed at all.

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
        (merged_nodes, prism_cells, tet_cells, cell_groups, n_bl_prisms,
        source_vertex, bl_extrude_faces, bl_cell_groups, n_transition_cells):
        - prism_cells: (n_prism, 6) int64, true-BL cells as genuine prisms -
          empty (0, 6) when there's no BL region at all
        - tet_cells: (n_tet, 4) int64, transition-stage tets FOLLOWED BY the
          core tetgen fill's own tets (transition cells first, matching the
          pre-existing "BL-origin cells first" convention the repair
          pipeline's n_bl_cells split already assumes)
        - cell_groups: parallel to tet_cells (transition + core)
        - n_bl_prisms: len(prism_cells), kept as an explicit return (not
          just re-derived as len(prism_cells) at the call site) purely for
          symmetry with the pre-existing n_bl_cells naming other code
          greps for
        - n_transition_cells: len of the transition-tet portion at the
          front of tet_cells - the caller's real n_bl_cells for the
          tet-only repair pipeline (0 when bl_layers >= max_layers, i.e.
          no transition stage at all)
        source_vertex is (nodes_per_BL_layer,), identity
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
            regions = [(center, 1, max_cell_size ** 3 * CORE_VOLUME_CAP_FRACTION)]
        core_nodes, core_tets, trifaces, triface_markers = fill_core_volume(
            surface_nodes, surface_faces, holes=hole_points,
            regions=regions, face_markers=face_markers,
            minratio=CORE_TETGEN_MINRATIO, mindihedral=CORE_TETGEN_MINDIHEDRAL,
        )
        merged_nodes, tet_cells = core_nodes, core_tets
        prism_cells = np.zeros((0, 6), dtype=np.int64)
        bl_cell_groups = np.zeros(0, dtype=object)
        n_transition_cells = 0
        if face_markers is not None:
            cell_groups = attribute_cells_from_trifaces(
                core_tets, trifaces, triface_markers, marker_to_name
            )
        else:
            cell_groups = np.full(len(tet_cells), '', dtype=object)
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
        # extrude_layers' own Stage-1(BL)/Stage-2(transition) split point -
        # recomputed here identically (extrude_layers doesn't hand the
        # clamped value back) since it also governs the prism/tet cell-type
        # split: only the fine near-wall Stage-1 layers become true prisms
        # (see this function's own docstring for why the transition stage
        # stays tetrahedra - this was always the design, restored here).
        _effective_bl_layers = bl_layers if bl_layers is not None else min(8, max_layers)
        _effective_bl_layers = int(np.clip(_effective_bl_layers, 0, max_layers))

        n_layers = len(bl_layer_conn)
        nodes_per_layer = len(bl_nodes) // n_layers
        outer_offset = (n_layers - 1) * nodes_per_layer
        bl_split_offset = _effective_bl_layers * nodes_per_layer

        bl_prisms, bl_face_of_cell = convert_layers_to_prisms(
            bl_nodes[:bl_split_offset + nodes_per_layer],
            bl_layer_conn[:_effective_bl_layers + 1],
            extrude_faces,
        )
        n_bl_cells = len(bl_prisms)
        logger.info(f"  BL mesh: {len(bl_nodes)} nodes, {len(bl_prisms)} prism cells")

        # Attribute each BL cell directly back to its source boundary group
        # via position, not node-index matching against the pre-extrusion
        # surface: convert_layers_to_prisms' own bl_face_of_cell maps
        # every surviving cell back to its extrude_faces row directly (a
        # plain tile - one contiguous block of len(extrude_faces) prisms per
        # layer - no longer holds exactly now that function can drop
        # analytically zero-volume collapsed-layer prisms, see its own
        # docstring). This is exact for every surviving BL cell, including
        # the vast majority of
        # body/ground's own outer surface that node-index matching can never
        # reach (see mesh_boundary.py - those nodes get a brand-new offset
        # index once genuinely displaced by extrusion, so their
        # post-extrusion face can't match anything in a lookup built from
        # the original, pre-extrusion node indices).
        #
        # Only LAYER 0's own prisms (the ones whose bottom cap is the actual
        # physical wall) are tagged with the source group name - every other
        # layer gets '' instead, even though bl_face_of_cell would happily
        # tell us their source face too. This matters concretely, not just
        # cosmetically: a BL column can terminate early at a sharp/complex
        # geometry feature (local thickness cap triggered - see
        # compute_local_thickness_limit), and the LAST surviving prism's own
        # top cap then becomes a legitimate, unavoidable terminal boundary
        # face - a real face, not a bug in face-extraction - but it is NOT
        # the physical wall, it is an artifact of where this specific
        # column happened to stop. Tagging every layer identically (the
        # previous behaviour, unchanged since before true prisms existed)
        # attributed that terminal face to the same "body"/WALL group as the
        # genuine wall, which would wrongly apply a no-slip condition across
        # what should be open interior space. Confirmed as a real, not
        # theoretical, effect on a real case (ProjectFiles Part6/7 P21):
        # 33,448 such faces, concentrated at sharp cube edges, spread across
        # layers 1-3 - NOT at the BL/transition seam as first suspected,
        # confirming this is a pre-existing BL-extrusion characteristic
        # unrelated to the prism/tet split, just never previously isolated
        # from the wall group it was silently merged into.
        #
        # Layer-0 detection is a plain node-index range check, not a return
        # value from convert_layers_to_prisms: layer L's own nodes always
        # occupy bl_nodes[L*nodes_per_layer : (L+1)*nodes_per_layer]
        # (extrude_layers' own node layout, unchanged since before this
        # session), so a prism's bottom cap (v0) being < nodes_per_layer is
        # both necessary and sufficient for "this prism's bottom is layer 0"
        # - no need to plumb a new return value through convert_layers_to_
        # prisms just to re-derive information already implicit in the node
        # indices it returns.
        is_layer0_prism = bl_prisms[:, 0] < nodes_per_layer
        bl_cell_groups = np.where(is_layer0_prism, extrude_face_groups[bl_face_of_cell], '')

        # Transition stage (bl_layers .. max_layers): stays tetrahedra, via
        # the SAME diagonal-consistency rule (sorted_base derived from
        # extrude_faces alone, identical regardless of which layer slice is
        # passed - see convert_layers_to_tetrahedra's own docstring), so the
        # prism/transition-tet seam at bl_layers is automatically conformal
        # without any extra coordination between the two calls. Node indices
        # come back relative to the SLICED bl_nodes sub-array passed in, so
        # bl_split_offset is added back to make them valid global indices
        # into the full bl_nodes/merged_nodes array.
        if _effective_bl_layers < max_layers:
            _transition_tets_local, _transition_face_of_cell = convert_layers_to_tetrahedra(
                bl_nodes[bl_split_offset:], bl_layer_conn[_effective_bl_layers:], extrude_faces
            )
            transition_tets = _transition_tets_local + bl_split_offset
            # Same reasoning as bl_cell_groups just above: only tag a tet
            # with the source group if it's genuinely touching the physical
            # wall (layer 0), not every layer uniformly. In the ordinary
            # case (_effective_bl_layers > 0, a real prism region exists)
            # this is always empty - transition tets start at bl_layers by
            # construction and can never reach layer 0 - so this line only
            # does something in the bl_layers==0 edge case (no BL region at
            # all, pure-tetrahedra fallback), where it correctly reduces to
            # the same "tag layer 0 only" rule. Column 0 of every tet is
            # always its own v0 (orient_tetrahedra only ever swaps columns
            # 2/3), and v0 is always the lowest-layer vertex of any of
            # T1/T2/T3's definitions, so checking it alone is sufficient.
            is_layer0_tet = transition_tets[:, 0] < nodes_per_layer
            transition_cell_groups = np.where(
                is_layer0_tet, extrude_face_groups[_transition_face_of_cell], ''
            )
        else:
            transition_tets = np.zeros((0, 4), dtype=np.int64)
            transition_cell_groups = np.zeros(0, dtype=object)
        n_transition_cells = len(transition_tets)
        logger.info(
            f"  Transition mesh: {n_transition_cells} tet cells "
            f"(layers {_effective_bl_layers}-{max_layers})"
        )

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

        # Which LOCAL (per-layer) nodes may legitimately have their outer-
        # layer position treated as a real boundary, for the marking step
        # just below - two disjoint legitimate cases, tested separately and
        # OR'd together:
        #   (a) still genuinely growing in the very last slab (outer layer
        #       position differs from the second-to-last layer's)
        #   (b) pinned at the bare/layer-0 position for its ENTIRE
        #       extrusion (taper_scale==0 from the start - "a wall entirely
        #       pinned by the seam taper", see the marking comment below) -
        #       its outer-layer position IS still the true, unmoved wall
        # What's EXCLUDED by neither (a) nor (b): a column that grew for a
        # while then froze early (thickness_limit capped mid-extrusion,
        # see compute_local_thickness_limit) - its outer-layer position is
        # a repeated/coincident copy of wherever it froze, no longer
        # changing, but NOT equal to its original layer-0 position either.
        second_last_offset = outer_offset - nodes_per_layer
        prev_layer_nodes = bl_nodes[second_last_offset:second_last_offset + nodes_per_layer]
        node_reached_outer_layer = (
            np.linalg.norm(outer_nodes - prev_layer_nodes, axis=1) > min_cell_size * 1e-6
        )
        node_never_moved = (
            np.linalg.norm(outer_nodes - surface_nodes, axis=1) <= min_cell_size * 1e-6
        )
        node_marker_ok = node_reached_outer_layer | node_never_moved

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
        #
        # Only a facet whose all 3 vertices reached the outer layer (or,
        # symmetrically, a facet with an EARLIER zero-thickness column -
        # see node_reached_outer_layer's own docstring, that case still
        # sits exactly at the true unmoved wall) may inherit the group
        # marker. A facet with a MIXED or entirely-frozen-early vertex set
        # represents an early-terminated column's exposed remnant, not the
        # real wall or the intended BL/core interface - marking it "body"
        # would tag whatever core tet touches it (via attribute_cells_
        # from_trifaces, below) as the physical wall too. Confirmed as a
        # real, large-scale effect on a real case, not a theoretical one -
        # see ProjectFiles Part7 P21: 33,612 such faces on cube_demo,
        # spatially coincident with the same sharp-cube-edge columns whose
        # BL-side termination P21's other half already fixed. Leaving an
        # excluded facet unmarked (0) lets it fall through to mesh_
        # boundary.py's own node-index matching (which also won't match,
        # since it isn't a bare-surface position either) and finally to
        # that module's own UNCLASSIFIED catch-all - the same resolution
        # already applied to the BL/transition-side half of this gap.
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
            # See node_marker_ok's own docstring above: only a facet whose
            # all 3 vertices are legitimately at the true wall or the
            # intended full-height interface keeps its marker; an early-
            # terminated column's facet is zeroed out (unmarked) so it
            # falls through to the UNCLASSIFIED catch-all downstream
            # instead of being attributed to the physical wall group.
            facet_marker_ok = node_marker_ok[bl_outer_surface].all(axis=1)
            bl_outer_markers = np.where(facet_marker_ok, bl_outer_markers, 0)
            core_markers = np.array(
                [group_name_to_marker.get(n, 0) for n in core_face_groups], dtype=np.int32
            )
            face_markers = np.concatenate([bl_outer_markers, core_markers])
            regions = [(outer_nodes.mean(axis=0), 1, max_cell_size ** 3 * CORE_VOLUME_CAP_FRACTION)]

        core_nodes, core_tets, trifaces, triface_markers = fill_core_volume(
            plc_points, plc_faces, holes=hole_points, regions=regions, face_markers=face_markers,
            minratio=CORE_TETGEN_MINRATIO, mindihedral=CORE_TETGEN_MINDIHEDRAL,
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
        # but computed from outer_nodes directly rather than assumed.
        n_shared = len(outer_nodes)
        core_tets_remapped = core_tets.copy()
        is_shared = core_tets < n_shared
        core_tets_remapped[is_shared] = shared_target[core_tets[is_shared]]
        core_tets_remapped[~is_shared] = core_tets[~is_shared] - n_shared + len(bl_nodes)

        new_core_nodes = core_nodes[n_shared:]
        merged_nodes = np.vstack([bl_nodes, new_core_nodes])
        # Prisms and tets are kept as two SEPARATE connectivity arrays (see
        # this function's docstring) rather than one vstacked array - a
        # prism's (n,6) shape can't share a row layout with a tet's (n,4)
        # anyway. bl_prisms'/transition_tets' node indices are already valid
        # in merged_nodes' global space unchanged (bl_nodes occupies
        # merged_nodes[:len(bl_nodes)] verbatim), same as bl_cells never
        # needed remapping before. transition_tets goes FIRST in tet_cells,
        # matching the pre-existing "BL-origin cells first" convention the
        # tet-only repair pipeline's n_bl_cells split already assumes.
        prism_cells = bl_prisms
        tet_cells = np.vstack([transition_tets, core_tets_remapped])
        # core_cell_groups (from facet markers, populated above whenever
        # max_cell_size grading is active) already identifies core cells'
        # source groups directly. When grading isn't active, it's all ''
        # placeholders instead - those core-only groups (tunnel/inlet/
        # outlet-type, never displaced by extrusion) still go through
        # identify_boundaries_from_surface's node-index matching below,
        # which works correctly for them since their nodes were never
        # remapped in that case (nobisect=True preserves them verbatim).
        # transition_cell_groups is always known exactly (from
        # extrude_face_groups, same as bl_cell_groups), independent of
        # whether facet-marker grading is active.
        cell_groups = np.concatenate([transition_cell_groups, core_cell_groups])
        logger.info(
            f"  Merged mesh: {len(merged_nodes)} nodes, "
            f"{len(prism_cells) + len(tet_cells)} cells "
            f"({len(prism_cells)} BL prisms + {n_transition_cells} transition tets + "
            f"{len(core_tets_remapped)} core tets)"
        )

    return (
        merged_nodes, prism_cells, tet_cells, cell_groups, n_bl_cells,
        source_vertex, extrude_faces, bl_cell_groups, n_transition_cells,
    )
