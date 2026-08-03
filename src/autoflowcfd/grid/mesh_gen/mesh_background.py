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

The actual per-attempt assembly work (classify -> extrude -> tetgen-fill ->
splice) lives in mesh_background_merge._build_merged_mesh; this module is
the retry/repair orchestration around it (Stage A smoothing, Stage B/B'
targeted repair, Stage C-adjacent backoff-retry recursion) - see
generate_hybrid_mesh's own docstring.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ..structures import BoundaryMap, VolumeMeshData

from .mesh_extrusion import extrude_layers
from .mesh_prism_to_tet import orient_tetrahedra
from .mesh_domain_classify import classify_boundary_groups
from .mesh_tetgen_core import (
    fill_core_volume, compute_local_thickness_limit, repair_nonmanifold_cells,
    _dedupe_coincident_points,
)
from .mesh_repair import smooth_bad_cells, compute_bl_thickness_limit_override, remesh_core_cavity
from .mesh_background_merge import _build_merged_mesh


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
    extra_thickness_limit: Optional[np.ndarray] = None,
    bl_layers: Optional[int] = None,
    _is_stage_b_retry: bool = False,
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
        extra_thickness_limit: Stage B (mesh quality repair) internal use -
            (n_surface_nodes,) per-vertex cumulative-BL-thickness cap,
            merged via elementwise minimum into the thickness_limit this
            function computes on its own (see compute_local_thickness_limit)
            - forces early BL termination at specific vertices implicated
            in still-bad cells after Stage A smoothing, without affecting
            the rest of the surface. None (default) leaves the internally-
            computed limit untouched.
        bl_layers: Optional override for how many of max_layers count as
            "Stage 1 (BL)" before switching to the transition growth rate
            (see mesh_extrusion.extrude_layers' own bl_layers doc). None
            (default) keeps the previous hardcoded `min(8, max_layers)`
            split - in particular, any max_layers <= 8 then leaves ZERO
            layers for the transition stage, silently disabling
            target_handoff_size's size-matching behavior above regardless
            of max_cell_size.
        _is_stage_b_retry: internal - True on the single recursive call
            this function makes to itself after computing extra_thickness_
            limit from Stage A's residual bad cells, to
            cap retry depth at exactly 1 rather than potentially recursing
            forever if a Stage B attempt doesn't fully resolve every bad
            cell (Stage C's coarser global-parameter backoff, one level up
            in volume_mesh_generator.py, is the fallback beyond this).

    Returns:
        VolumeMeshData with a domain-conforming hybrid mesh (BL + core).
        Mesh quality (see quality_validator.MeshQualityValidator) is
        checked and repaired (Stage A smoothing, Stage B targeted
        regeneration - see module mesh_repair.py) before this returns;
        the final quality report (including any repair actions taken) is
        logged, not returned - callers that need the gate result run their
        own MeshQualityValidator().validate_volume_mesh() pass, which is
        cheap relative to generation itself.
    """
    if surface_boundaries is None or not surface_boundaries.groups:
        raise ValueError(
            "generate_hybrid_mesh requires surface_boundaries with at least "
            "one boundary group, used to classify wall-type surfaces for BL "
            "extrusion versus the outer domain shell"
        )

    logger.info("Starting domain-conforming hybrid mesh generation...")

    (merged_nodes, merged_cells, cell_groups, n_bl_cells, bl_source_vertex,
     bl_extrude_faces) = _build_merged_mesh(
        surface_nodes, surface_faces, bounding_box,
        growth_rate, max_layers, min_cell_size, surface_boundaries, max_cell_size,
        extra_thickness_limit, bl_layers,
    )

    # Build VolumeMeshData structure
    from ..structures import NodeArray, TetrahedralCells, GridMetadata, VolumeMeshData

    logger.info("Step 4/4: Re-orienting and computing tetrahedral volumes...")
    merged_cells = orient_tetrahedra(merged_nodes, merged_cells.astype(np.int64))
    _nodes_obj_tmp = NodeArray(
        x=merged_nodes[:, 0].copy(), y=merged_nodes[:, 1].copy(), z=merged_nodes[:, 2].copy()
    )
    volumes = TetrahedralCells.compute_volumes(_nodes_obj_tmp, merged_cells.astype(np.int32))

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
        n_bl_cells = int(np.sum(valid_mask[:n_bl_cells]))
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
        n_bl_cells = int(np.sum(nonmanifold_keep[:n_bl_cells]))
        merged_cells = merged_cells[nonmanifold_keep]
        volumes = volumes[nonmanifold_keep]
        cell_groups = cell_groups[nonmanifold_keep]

    # ------------------------------------------------------------------
    # Mesh quality repair: Stage A (quality-gated smoothing, in-place on
    # merged_nodes - see mesh_repair.py) then, if bad cells remain, Stage B
    # (one targeted regeneration retry with a local BL thickness cap or
    # core refinement region derived from exactly which cells are still
    # bad). Stage C (global parameter backoff) is one level up, in
    # VolumeMeshGenerator.generate_from_surface, for when even Stage B
    # isn't enough.
    # ------------------------------------------------------------------
    from ..validation.quality_validator import MeshQualityValidator
    from .face_extractor import FaceExtractor

    merged_cells = merged_cells.astype(np.int32)
    validator = MeshQualityValidator()
    logger.info("Checking volume mesh quality (pre-repair)...")
    # Extracted once and handed to both validate() and smooth_bad_cells()
    # below - both would otherwise independently re-extract faces from this
    # exact same (unmoved) geometry, a real duplicated cost on a multi-
    # million-cell mesh (face extraction alone measured at several seconds
    # on a ~2-7M cell mesh).
    _pre_repair_node_arr = NodeArray(
        x=merged_nodes[:, 0].copy(), y=merged_nodes[:, 1].copy(), z=merged_nodes[:, 2].copy()
    )
    pre_repair_faces = FaceExtractor.extract_faces(merged_cells, _pre_repair_node_arr)
    initial_report = validator.validate(
        merged_nodes, merged_cells, cell_type="tetrahedron", faces=pre_repair_faces
    )

    # Cells physically overlapping a different, non-adjacent cell
    # (mesh_overlap_check.py, run as part of validate() above - reusing its
    # result rather than checking again) are folded into Stage A/B''s own
    # "bad cell" criteria alongside skew/orthogonality/volume-ratio, so an
    # overlap actually drives repair action instead of only being visible
    # in the quality report.
    overlap_bad_mask = None
    if initial_report.overlapping_cell_ids is not None and len(initial_report.overlapping_cell_ids):
        overlap_bad_mask = np.zeros(len(merged_cells), dtype=bool)
        overlap_bad_mask[initial_report.overlapping_cell_ids] = True

    nodes_before_repair = merged_nodes
    merged_nodes, bad_mask, repair_actions = smooth_bad_cells(
        merged_nodes, merged_cells, validator, max_passes=5, initial_faces=pre_repair_faces,
        extra_bad_mask=overlap_bad_mask, n_bl_cells=n_bl_cells,
    )
    mesh_changed_by_repair = not np.array_equal(nodes_before_repair, merged_nodes)

    if np.any(bad_mask):
        # Stage B': local cavity remesh (core cells, and BL cells including
        # ones touching the wall - see mesh_repair.py's module docstring).
        # Not gated on _is_stage_b_retry (unlike the BL-
        # side retry below): it never recurses into a fresh
        # generate_hybrid_mesh call, it's a one-shot local patch on the
        # mesh already in hand, so there's no unbounded-recursion risk from
        # also attempting it on the single BL-side retry pass.
        # pre_repair_faces' CONNECTIVITY (owner/neighbor, node_connectivity)
        # is still valid here even though Stage A may have moved node
        # coordinates - smoothing never changes cell/face topology, only
        # positions.
        n_cells_before_cavity = len(merged_cells)
        merged_nodes, merged_cells, cell_groups, bad_mask, cavity_actions = remesh_core_cavity(
            merged_nodes, merged_cells, cell_groups, n_bl_cells, pre_repair_faces, bad_mask, validator,
        )
        repair_actions.extend(cavity_actions)
        if len(merged_cells) != n_cells_before_cavity:
            mesh_changed_by_repair = True

        # Re-check for non-manifold faces (the same check already run once
        # above, right after degenerate-cell removal) - Stage B' is a new
        # potential SOURCE of this failure mode that didn't exist when that
        # first check was scoped to run only once, before any repair stage:
        # a cavity's own local retile can pass its own boundary-point-
        # preservation check (verbatim node positions) while still
        # producing tets that overlap the kept mesh just outside the
        # cavity, if the cavity's fixed boundary itself sits on nearly-
        # degenerate geometry (observed directly: a BL-side cavity retiled
        # near a Stage B thickness-capped vertex, where capping had already
        # collapsed a neighbouring layer to near-zero height, produced 12
        # such faces on a real case - not caught here previously because
        # Stage B' extending to BL/wall-adjacent cells is new; a core-only
        # cavity's boundary is never that close to a capped-thickness BL
        # seam). Left unrepaired, this crashes downstream (face_extractor
        # raises a hard error on a >2-cell-shared face) instead of failing
        # gracefully via the quality gate like every other residual defect
        # here does.
        nonmanifold_keep = repair_nonmanifold_cells(merged_nodes, merged_cells)
        if not nonmanifold_keep.all():
            n_bl_cells = int(np.sum(nonmanifold_keep[:n_bl_cells]))
            merged_cells = merged_cells[nonmanifold_keep]
            cell_groups = cell_groups[nonmanifold_keep]
            bad_mask = bad_mask[nonmanifold_keep]
            mesh_changed_by_repair = True

    if np.any(bad_mask) and not _is_stage_b_retry:
        n_bad = int(np.sum(bad_mask))
        cap_thickness = min_cell_size * 3.0
        extra_limit, bl_verts = compute_bl_thickness_limit_override(
            bad_mask, n_bl_cells, merged_cells, len(surface_nodes), cap_thickness,
            nodes_per_layer=len(bl_source_vertex), node_original_vertex=bl_source_vertex,
            local_surface_faces=bl_extrude_faces,
        )
        # Core-side local repair regions were tried here too and removed:
        # tetgen's per-region refinement does not confine itself to a small
        # added region's local footprint when a domain-wide grading region
        # is also active in the same connected volume - a handful of small
        # local regions ballooned the whole core fill several-fold (1.2M ->
        # 6.1M tets on a real case) without improving quality there. See
        # mesh_repair.py's module docstring for the full account. BL-side
        # thickness capping has no equivalent failure mode, so it remains.
        if extra_limit is not None:
            logger.warning(
                f"Stage A left {n_bad} cells still bad ({len(bl_verts)} BL vertices "
                f"implicated) - retrying generation once with a targeted local BL "
                f"thickness cap (Stage B)..."
            )
            # Explicitly drop this attempt's large arrays before recursing:
            # `return generate_hybrid_mesh(...)` keeps this frame (and every
            # local in it - merged_nodes/merged_cells/volumes/cell_groups,
            # plus the earlier bl_*/core_*/plc_* intermediates, all sized to
            # a multi-million-cell mesh) alive on the call stack for the
            # entire duration of the retry, since Python doesn't do tail-
            # call elimination. Observed directly on a real 2.6M-cell case:
            # left implicit, process RSS climbed past 11GB and a step that
            # normally takes ~2s (compute_local_thickness_limit) started
            # taking 10+ minutes, consistent with the retry now running
            # under severe memory pressure from a frame it never needed to
            # keep alive - only extra_limit (a tiny, derived array) is
            # actually needed past this point.
            del merged_nodes, merged_cells, volumes, cell_groups, bad_mask, initial_report
            import gc
            gc.collect()
            return generate_hybrid_mesh(
                surface_nodes, surface_faces, bounding_box,
                growth_rate=growth_rate, max_layers=max_layers, min_cell_size=min_cell_size,
                target_cells=target_cells, surface_boundaries=surface_boundaries,
                max_cell_size=max_cell_size,
                extra_thickness_limit=extra_limit,
                bl_layers=bl_layers,
                _is_stage_b_retry=True,
            )
        else:
            logger.warning(
                f"Stage A left {n_bad} cells still bad, but none are traceable to a "
                f"specific BL vertex Stage B can target - leaving as-is for the "
                f"caller's own quality gate to report"
            )

    # ------------------------------------------------------------------
    # Final defensive pass: merge any geometrically-coincident points that
    # ended up under different global node indices (see
    # mesh_tetgen_core._dedupe_coincident_points's docstring for the
    # specific failure mode - a widespread Stage B thickness cap freezing
    # many BL nodes at an identical coordinate for their remaining layers,
    # each still getting its own index). Cheap to run unconditionally (a
    # KD-tree query_pairs at a tight 1e-9 tolerance is a no-op cost-wise
    # when there's nothing to merge) and this is the only point in the
    # pipeline that looks at the WHOLE merged mesh's geometry rather than
    # one local piece of it.
    # ------------------------------------------------------------------
    n_nodes_before_merge = len(merged_nodes)
    merged_nodes, merged_cells, _remap = _dedupe_coincident_points(merged_nodes, merged_cells)
    if len(merged_nodes) != n_nodes_before_merge:
        mesh_changed_by_repair = True
        merged_cells = merged_cells.astype(np.int32)

        # Merging can turn a cell degenerate (2+ of its 4 vertices now
        # identical) - same threshold/reasoning as the earlier degenerate-
        # cell removal above.
        _tmp_nodes_obj = NodeArray(
            x=merged_nodes[:, 0].copy(), y=merged_nodes[:, 1].copy(), z=merged_nodes[:, 2].copy()
        )
        post_merge_volumes = TetrahedralCells.compute_volumes(_tmp_nodes_obj, merged_cells)
        degenerate_threshold = (min_cell_size ** 3) * 1e-6
        valid_mask = post_merge_volumes > degenerate_threshold
        n_invalid = int(np.sum(~valid_mask))
        if n_invalid > 0:
            logger.warning(
                f"Coincident-point merge left {n_invalid} newly-degenerate cells "
                f"(2+ vertices merged into the same node), removing them..."
            )
            merged_cells = merged_cells[valid_mask]
            cell_groups = cell_groups[valid_mask]

        # And merging can reveal non-manifold structure that was previously
        # hidden behind two different index sets (see the docstring above -
        # the whole point of this pass is to expose exactly that).
        nonmanifold_keep = repair_nonmanifold_cells(merged_nodes, merged_cells)
        if not nonmanifold_keep.all():
            merged_cells = merged_cells[nonmanifold_keep]
            cell_groups = cell_groups[nonmanifold_keep]

    nodes_obj = NodeArray(
        x=merged_nodes[:, 0].copy(), y=merged_nodes[:, 1].copy(), z=merged_nodes[:, 2].copy()
    )
    volumes = TetrahedralCells.compute_volumes(nodes_obj, merged_cells)

    cells_obj = TetrahedralCells(
        connectivity=merged_cells,
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

    if mesh_changed_by_repair:
        # log_summary=False: the explicit logger.info(final_report.summary())
        # below prints a strictly more complete version of the same report
        # (with the before/after comparison and repair-actions sections
        # attached, set just after this call) - letting validate() also log
        # its own copy here would print the same content twice in a row.
        final_report = validator.validate(
            merged_nodes, merged_cells, cell_type="tetrahedron", log_summary=False
        )
    else:
        # Stage A made no changes (the common case whenever it can't find a
        # safe move at all - see mesh_repair.smooth_bad_cells) - the mesh is
        # bit-for-bit identical to what initial_report already evaluated, so
        # re-running a full validate() (another face extraction + all
        # quality checks) would just recompute the exact same numbers.
        # dataclasses.replace makes an independent copy rather than
        # aliasing initial_report itself, so the before/after fields set
        # below don't turn it into a self-reference.
        from dataclasses import replace as _dc_replace
        final_report = _dc_replace(initial_report)
    final_report.repair_stages_applied = repair_actions
    final_report.initial_report = initial_report
    logger.info(f"\n{final_report.summary()}")

    return volume_mesh
