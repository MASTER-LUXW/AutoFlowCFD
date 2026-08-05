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


def _prism_aware_overlap_bad_tet_mask(
    merged_nodes: np.ndarray, prism_cells: np.ndarray, merged_cells: np.ndarray,
) -> Optional[np.ndarray]:
    """Physical-overlap check over the FULL mixed (prism+tet) face set,
    returning a (len(merged_cells),) bool mask of which TET cells are
    implicated - the only side Stage A/B'/D can act on (true prisms are
    entirely outside their scope, see this module's own n_bl_cells comment
    above - merged_cells here is transition tets followed by core tets,
    never prisms).

    Why this is necessary and not redundant with the ordinary tet-only
    overlap check already run as part of validator.validate(...): a
    tet-only check can only ever see merged_cells (transition + core
    tets), so a badly-graded sliver tet at the BL/core interface (see
    ProjectFiles Part6's P14 for how common these are - abrupt tetgen
    size-grading transitions right at the interface) that's elongated
    enough to physically reach back into a nearby true-prism cell is
    invisible to it as an overlap defect - it has no OTHER tet in
    merged_cells to overlap with from its own region. And since prisms are
    outside smooth_bad_cells'/collapse_bad_cells' own skew/orthogonality/
    adjacent-ratio criteria too (those only ever see merged_cells), nothing
    in Stage A/B'/D would otherwise ever learn this cell is bad from the
    overlap angle at all - it would silently pass every mid-pipeline check
    and only show up in the FINAL validate_mixed report, too late for any
    repair stage to act on.
    Confirmed as a large, not marginal, real-case effect: before this
    function existed, the final report flagged 72,209 cells (12,541 prism +
    59,668 tet) - every sampled example a prism-tet pair with ZERO shared
    node indices at a genuine millimetre-scale (not floating-point-
    tolerance-scale) geometric distance, i.e. real physical collisions
    between an oversized core sliver and the BL region, not a coincident-
    point indexing artifact.

    Returns None if there are no prism cells (nothing this adds over the
    ordinary tet-only check) or nothing was found.
    """
    n_prism = len(prism_cells)
    if n_prism == 0:
        return None
    from ..schema.grid_nodes import NodeArray as _NodeArray
    from .face_extractor import FaceExtractor as _FaceExtractor
    from ..validation.mesh_overlap_check import check_face_overlap_and_proximity as _check_overlap

    node_arr = _NodeArray(
        x=np.ascontiguousarray(merged_nodes[:, 0]),
        y=np.ascontiguousarray(merged_nodes[:, 1]),
        z=np.ascontiguousarray(merged_nodes[:, 2]),
    )
    mixed_faces = _FaceExtractor.extract_faces_mixed(
        prism_cells, merged_cells.astype(np.int64), node_arr
    )
    mixed_report = _check_overlap(merged_nodes, merged_cells, faces=mixed_faces)
    if not len(mixed_report.overlapping_cell_ids):
        return None
    global_ids = mixed_report.overlapping_cell_ids
    tet_ids = global_ids[global_ids >= n_prism] - n_prism
    if len(tet_ids) == 0:
        return None
    mask = np.zeros(len(merged_cells), dtype=bool)
    mask[tet_ids] = True
    logger.warning(
        f"Prism-aware overlap check: {len(tet_ids)} core tet cell(s) physically "
        f"overlap a BL prism or another tet across the full mixed mesh - "
        f"invisible to the tet-only overlap check, added to the repair target set"
    )
    return mask


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

    (merged_nodes, prism_cells, merged_cells, cell_groups, n_bl_prisms,
     bl_source_vertex, bl_extrude_faces, bl_cell_groups, n_bl_cells) = _build_merged_mesh(
        surface_nodes, surface_faces, bounding_box,
        growth_rate, max_layers, min_cell_size, surface_boundaries, max_cell_size,
        extra_thickness_limit, bl_layers,
    )
    # Only the fine near-wall BL stage is genuine prisms now (prism_cells,
    # tracked separately - see PrismCells/_build_merged_mesh's docstrings),
    # entirely outside merged_cells; every repair stage below (Stage
    # A/B'/D) - all written for a single (n,4) tet array with n_bl_cells as
    # a row-index split within it - therefore never sees a prism cell.
    # This is a deliberate scope boundary (prisms bypass this repair
    # pipeline for now, see _build_merged_mesh's docstring), not a bug.
    #
    # The faster-growing TRANSITION stage (bl_layers..max_layers) is still
    # tetrahedra, occupying merged_cells[:n_bl_cells] (see
    # _build_merged_mesh's docstring for why - this is the original,
    # pre-existing design; an earlier revision of this function mistakenly
    # hardcoded n_bl_cells=0 unconditionally here, treating the ENTIRE
    # extruded stack as prism-only and leaving the transition tets with no
    # BL-origin protection/threshold at all - confirmed and restored).
    # n_bl_cells is exactly 0 only when there's genuinely no transition
    # stage (bl_layers >= max_layers) or no BL region at all - both already
    # the pre-existing "no BL region" code path every one of these
    # functions already handles correctly.

    # Build VolumeMeshData structure
    from ..structures import NodeArray, TetrahedralCells, PrismCells, GridMetadata, VolumeMeshData

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

    # Merge geometrically-coincident points BEFORE the pre-repair quality
    # check below - the BL extrusion and the core tetgen fill are two
    # independently-generated pieces stitched together right above
    # (_build_merged_mesh), and their shared interface routinely ends up
    # with two coincident-but-differently-indexed points on either side of
    # the seam. initial_report's overlap check (right below) has no
    # tolerance for that: two faces that are actually the same physical
    # face but reference different node indices look like a genuine
    # physical overlap. Confirmed directly as a real, severe effect, not a
    # theoretical one: on a real case this produced 208,512 cells flagged
    # "overlapping" (of ~2.7M total) before this dedup pass existed - two
    # orders of magnitude more than the ~2 cells every fully-processed
    # (post-repair, post-dedup) quality report on the same case ever
    # showed. Left unfixed, that inflated overlap_bad_mask (below) feeds
    # straight into Stage A as extra_bad_mask, so EVERY downstream repair
    # stage (A, B', B, D) inherits a bad_cell count dominated by this one
    # false-positive class rather than genuine skew/orthogonality/volume-
    # ratio defects - directly responsible for repair actions elsewhere in
    # this pipeline reporting hundreds of thousands of "bad cells" that
    # were never actually bad. See the near-identical pass just before
    # Stage D, and the "Final defensive pass" at the very end, for the
    # same fix applied at the two other points a fresh coincidence can be
    # introduced (Stage B' cavity splicing, Stage B thickness capping).
    n_nodes_before_seam_merge = len(merged_nodes)
    merged_nodes, merged_cells, _seam_remap = _dedupe_coincident_points(merged_nodes, merged_cells)
    if len(merged_nodes) != n_nodes_before_seam_merge:
        merged_cells = merged_cells.astype(np.int32)
        # prism_cells lives entirely outside merged_cells (Stage A/B'/D's
        # view) but can share nodes at the BL/core interface with what this
        # dedup pass just merged - its indices must follow the same remap
        # or they silently point at the wrong (or, if a node was actually
        # dropped by index-collapse, a stale) row of the now-shorter
        # merged_nodes array.
        prism_cells = _seam_remap[prism_cells]
        _tmp_nodes_obj_seam = NodeArray(
            x=merged_nodes[:, 0].copy(), y=merged_nodes[:, 1].copy(), z=merged_nodes[:, 2].copy()
        )
        post_merge_volumes_seam = TetrahedralCells.compute_volumes(_tmp_nodes_obj_seam, merged_cells)
        degenerate_threshold_seam = (min_cell_size ** 3) * 1e-6
        valid_mask_seam = post_merge_volumes_seam > degenerate_threshold_seam
        if int(np.sum(~valid_mask_seam)) > 0:
            logger.warning(
                f"Pre-repair coincident-point merge (BL/core seam) left "
                f"{int(np.sum(~valid_mask_seam))} newly-degenerate cells, removing them..."
            )
            n_bl_cells = int(np.sum(valid_mask_seam[:n_bl_cells]))
            merged_cells = merged_cells[valid_mask_seam]
            volumes = volumes[valid_mask_seam]
            cell_groups = cell_groups[valid_mask_seam]
        nonmanifold_keep_seam = repair_nonmanifold_cells(merged_nodes, merged_cells)
        if not nonmanifold_keep_seam.all():
            n_bl_cells = int(np.sum(nonmanifold_keep_seam[:n_bl_cells]))
            merged_cells = merged_cells[nonmanifold_keep_seam]
            volumes = volumes[nonmanifold_keep_seam]
            cell_groups = cell_groups[nonmanifold_keep_seam]

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
    from .face_extractor import FaceExtractor, repair_nonmanifold_mixed

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

    # Supplement with the prism-aware check (see its own docstring) - a bad
    # core tet overlapping a BL prism is invisible to the tet-only check
    # above but still needs to reach Stage A/D's target set.
    prism_overlap_mask = _prism_aware_overlap_bad_tet_mask(merged_nodes, prism_cells, merged_cells)
    if prism_overlap_mask is not None:
        if overlap_bad_mask is None:
            overlap_bad_mask = prism_overlap_mask
        else:
            overlap_bad_mask |= prism_overlap_mask

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
            del prism_cells, bl_cell_groups
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

    # Merge geometrically-coincident points BEFORE the overlap re-check
    # below - Stage B' cavity splices routinely leave two cavities' new
    # interior points sitting at (or extremely near) the same physical
    # location under different node indices, since each cavity's own
    # standalone tetgen call has no visibility into any other cavity's
    # result. check_face_overlap_and_proximity has no tolerance for that:
    # two faces that are actually the same physical face but reference
    # different-indexed-but-coincident points look like a genuine
    # overlap. Confirmed directly as the actual cause of a wildly inflated
    # count on a real case: 208,512 cells flagged "overlapping" here
    # before this dedup pass existed - two orders of magnitude more than
    # the ~2 cells every fully-processed (post-dedup) quality report on
    # the same mesh ever showed. This duplicates the logic of the "Final
    # defensive pass" further below (which still needs to run again at
    # the very end - Stage D's own edge collapses are watertight by
    # construction and shouldn't reintroduce coincident points, but this
    # is cheap enough to not skip out of caution).
    n_nodes_before_early_merge = len(merged_nodes)
    merged_nodes, merged_cells, _early_remap = _dedupe_coincident_points(merged_nodes, merged_cells)
    if len(merged_nodes) != n_nodes_before_early_merge:
        mesh_changed_by_repair = True
        merged_cells = merged_cells.astype(np.int32)
        prism_cells = _early_remap[prism_cells]
        _tmp_nodes_obj_early = NodeArray(
            x=merged_nodes[:, 0].copy(), y=merged_nodes[:, 1].copy(), z=merged_nodes[:, 2].copy()
        )
        post_merge_volumes_early = TetrahedralCells.compute_volumes(_tmp_nodes_obj_early, merged_cells)
        degenerate_threshold_early = (min_cell_size ** 3) * 1e-6
        valid_mask_early = post_merge_volumes_early > degenerate_threshold_early
        if int(np.sum(~valid_mask_early)) > 0:
            logger.warning(
                f"Pre-Stage-D coincident-point merge left {int(np.sum(~valid_mask_early))} "
                f"newly-degenerate cells, removing them..."
            )
            n_bl_cells = int(np.sum(valid_mask_early[:n_bl_cells]))
            merged_cells = merged_cells[valid_mask_early]
            cell_groups = cell_groups[valid_mask_early]
            bad_mask = bad_mask[valid_mask_early]
        nonmanifold_keep_early = repair_nonmanifold_cells(merged_nodes, merged_cells)
        if not nonmanifold_keep_early.all():
            n_bl_cells = int(np.sum(nonmanifold_keep_early[:n_bl_cells]))
            merged_cells = merged_cells[nonmanifold_keep_early]
            cell_groups = cell_groups[nonmanifold_keep_early]
            bad_mask = bad_mask[nonmanifold_keep_early]

    # Re-check physical cell overlap fresh, on the CURRENT mesh (post Stage
    # A/B'/nonmanifold-repair), before deciding whether Stage D has
    # anything to do. The overlap signal Stage A originally received
    # (overlap_bad_mask, above) does not survive this far: Stage B''s own
    # internal "is this cavity still bad" re-evaluation only re-derives
    # skew/orthogonality/adjacent-ratio (mesh_repair._bad_cell_mask's own
    # criteria - a spatial overlap check is too expensive to re-run per
    # cavity), so a cell that was never touched by Stage A/B' but is still
    # physically overlapping a distant cell silently drops out of bad_mask.
    # Reusing the ORIGINAL overlap_bad_mask by index isn't safe either -
    # cell indices have shifted from nonmanifold-repair/Stage B' cell-count
    # changes since it was computed - so this re-derives it from scratch on
    # the mesh as it stands right now, the only way to get CURRENT indices
    # right. Confirmed as a real, not just theoretical, gap: on a real
    # case Stage D left 2 CRITICAL-severity overlapping cells completely
    # untouched end to end for exactly this reason.
    collapse_faces = FaceExtractor.extract_faces(merged_cells, NodeArray(
        x=merged_nodes[:, 0].copy(), y=merged_nodes[:, 1].copy(), z=merged_nodes[:, 2].copy()
    ))
    from ..validation.mesh_overlap_check import check_face_overlap_and_proximity
    current_overlap_report = check_face_overlap_and_proximity(merged_nodes, merged_cells, faces=collapse_faces)
    if len(current_overlap_report.overlapping_cell_ids):
        bad_mask = bad_mask.copy()
        bad_mask[current_overlap_report.overlapping_cell_ids] = True
        logger.warning(
            f"Stage D: {len(current_overlap_report.overlapping_cell_ids)} cell(s) still "
            f"physically overlap a distant cell after Stage A/B' - adding them to Stage D's "
            f"target set (skew/orthogonality/adjacent-ratio checks alone wouldn't have caught them)"
        )

    # Same prism-aware supplement as the pre-Stage-A check above, re-derived
    # fresh (cell indices have shifted since then from Stage A/B'/nonmanifold
    # repair, same reasoning as current_overlap_report just above).
    prism_overlap_mask_d = _prism_aware_overlap_bad_tet_mask(merged_nodes, prism_cells, merged_cells)
    if prism_overlap_mask_d is not None:
        bad_mask = bad_mask.copy()
        bad_mask |= prism_overlap_mask_d

    if np.any(bad_mask):
        # --- TEMPORARY DIAGNOSTIC: break down bad_mask by criterion, region,
        # and BL/core-interface adjacency, to root-cause the core-region
        # share of bad cells (tetgen grading vs BL-splitting artifacts).
        # Remove once the core-region investigation concludes.
        skew_arr = validator.compute_cell_skewness(merged_nodes, merged_cells)
        diag = validator.compute_face_diagnostics(merged_nodes, merged_cells, collapse_faces)
        ortho_bad_face = diag['angle_deg'] > validator.thresholds['max_orthogonality_angle']
        adjratio_bad_face = diag['volume_ratio'] > validator.thresholds['max_adjacent_volume_ratio']
        cell_ortho_bad = np.zeros(len(merged_cells), dtype=bool)
        cell_ortho_bad[diag['owner'][ortho_bad_face]] = True
        cell_ortho_bad[diag['neighbor'][ortho_bad_face]] = True
        cell_adjratio_bad = np.zeros(len(merged_cells), dtype=bool)
        cell_adjratio_bad[diag['owner'][adjratio_bad_face]] = True
        cell_adjratio_bad[diag['neighbor'][adjratio_bad_face]] = True
        cell_skew_bad = skew_arr > validator.thresholds['max_skewness']

        conn = collapse_faces.connectivity
        owner_f, neighbor_f = conn[:, 0], conn[:, 1]
        interior_f = neighbor_f >= 0
        crosses_interface = interior_f & ((owner_f < n_bl_cells) != (neighbor_f < n_bl_cells))
        interface_cells = np.zeros(len(merged_cells), dtype=bool)
        interface_cells[owner_f[crosses_interface]] = True
        interface_cells[neighbor_f[crosses_interface]] = True

        core_bad_idx = np.flatnonzero(bad_mask & (np.arange(len(merged_cells)) >= n_bl_cells))
        bl_bad_idx = np.flatnonzero(bad_mask & (np.arange(len(merged_cells)) < n_bl_cells))
        core_vol = validator._compute_tetrahedron_volumes(merged_nodes, merged_cells)

        def _summ(idx, label):
            if len(idx) == 0:
                logger.info(f"Stage D DIAGNOSTIC [{label}]: 0 cells")
                return
            n_skew = int(np.sum(cell_skew_bad[idx]))
            n_ortho = int(np.sum(cell_ortho_bad[idx]))
            n_adjr = int(np.sum(cell_adjratio_bad[idx]))
            n_iface = int(np.sum(interface_cells[idx]))
            vol_idx = core_vol[idx]
            logger.info(
                f"Stage D DIAGNOSTIC [{label}]: n={len(idx)}, "
                f"skew-bad={n_skew} ({100*n_skew/len(idx):.0f}%), "
                f"ortho-bad={n_ortho} ({100*n_ortho/len(idx):.0f}%), "
                f"adjratio-bad={n_adjr} ({100*n_adjr/len(idx):.0f}%), "
                f"touches BL/core interface={n_iface} ({100*n_iface/len(idx):.0f}%), "
                f"volume range=[{vol_idx.min():.3e}, {vol_idx.max():.3e}], "
                f"mean={vol_idx.mean():.3e}"
            )

        _summ(bl_bad_idx, "BL-region bad cells")
        _summ(core_bad_idx, "core-region bad cells")
        # Same breakdown restricted to core cells NOT touching the interface
        # (i.e. genuinely "deep core", away from any BL-splicing seam) - if
        # this count is small relative to core_bad_idx, the core-region
        # problem is concentrated at the seam, not spread through tetgen's
        # far-field fill.
        core_bad_far = core_bad_idx[~interface_cells[core_bad_idx]]
        _summ(core_bad_far, "core-region bad cells NOT touching BL interface")
        # --- END TEMPORARY DIAGNOSTIC

        # Stage D: last-resort local edge collapse (mesh_repair_collapse.py)
        # for cells that are STILL bad at this point - i.e. Stage A
        # (smoothing) and Stage B' (cavity re-tiling) already couldn't fix
        # them, and Stage B's full regeneration retry either doesn't apply
        # here (extra_limit was None above) or has already been spent (this
        # call itself IS that one retry, _is_stage_b_retry=True, so it will
        # never recurse into another). Unlike every stage above, this
        # changes cell COUNT (not just node positions or a cavity's local
        # tetrahedralization) - see that module's docstring for why an edge
        # collapse is safe here (never touches physical-boundary geometry,
        # never introduces a negative-volume or newly-bad neighbour cell,
        # stays watertight by construction) where naively deleting a cell
        # would not be. NOTE: an edge collapse is a purely local, shape-
        # driven operation - it has no notion of "does this fix the
        # overlap with some distant cell" and isn't guaranteed to (the
        # overlap might not even involve a short edge at all). Cells added
        # to bad_mask above purely for overlap, with no accompanying
        # skew/orthogonality/adjacent-ratio defect, likely have no short
        # edge to collapse and will end up in collapse_bad_cells' own
        # "unresolved" count - left for the caller's quality gate to still
        # report, same as any other unfixable defect.
        from .mesh_repair_collapse import collapse_bad_cells
        n_cells_before_collapse = len(merged_cells)
        n_nodes_before_collapse = len(merged_nodes)
        # prism_cells is entirely outside collapse_bad_cells' own view (it
        # only ever operates on the tet portion) but can share BL/core-
        # interface nodes with it - protect them from its final compaction
        # dropping a node whose only surviving references are prism cells
        # (confirmed as a real crash on a real case: PrismCells.compute_
        # volumes later indexing past the end of the compacted node array).
        prism_referenced_nodes = np.unique(prism_cells.ravel()) if len(prism_cells) else None
        merged_nodes, merged_cells, cell_groups, n_bl_cells, collapse_actions, collapse_node_remap = collapse_bad_cells(
            merged_nodes, merged_cells, cell_groups, n_bl_cells, collapse_faces, bad_mask, validator,
            extra_referenced_nodes=prism_referenced_nodes,
        )
        repair_actions.extend(collapse_actions)
        if len(merged_nodes) != n_nodes_before_collapse:
            prism_cells = collapse_node_remap[prism_cells]
        if len(merged_cells) != n_cells_before_collapse:
            mesh_changed_by_repair = True
            merged_cells = merged_cells.astype(np.int32)

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
        prism_cells = _remap[prism_cells]

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

    # Non-manifold check across the FULL mixed mesh (prism+tet together) -
    # the tet-only repair_nonmanifold_cells pass(es) above structurally
    # cannot see a face shared by, e.g., 2 tets + 1 prism (or any
    # multiplicity involving a prism cell at all), since they only ever
    # look at merged_cells. Confirmed as a real gap on a real case: 37
    # such faces survived every check above and only surfaced as a hard
    # crash in FaceExtractor.extract_faces_mixed the first time anything
    # tried to build a face graph over the true combined mesh. Cheap when
    # clean (mirrors the tet-only checks' own no-op cost when nothing is
    # wrong) so run unconditionally rather than only after this mesh is
    # already suspected bad.
    if len(prism_cells):
        prism_keep_mm, tet_keep_mm = repair_nonmanifold_mixed(nodes_obj, prism_cells, merged_cells.astype(np.int64))
        if not prism_keep_mm.all() or not tet_keep_mm.all():
            mesh_changed_by_repair = True
            prism_cells = prism_cells[prism_keep_mm]
            bl_cell_groups = bl_cell_groups[prism_keep_mm]
            merged_cells = merged_cells[tet_keep_mm]
            cell_groups = cell_groups[tet_keep_mm]

    volumes = TetrahedralCells.compute_volumes(nodes_obj, merged_cells)

    cells_obj = TetrahedralCells(
        connectivity=merged_cells,
        volumes=volumes
    )

    prism_cells = prism_cells.astype(np.int32)
    n_prism = len(prism_cells)
    prism_cells_obj = None
    if n_prism > 0:
        prism_volumes = PrismCells.compute_volumes(nodes_obj, prism_cells)
        prism_cells_obj = PrismCells(connectivity=prism_cells, volumes=prism_volumes)

    from .mesh_boundary import identify_boundaries_from_surface
    tet_boundaries = identify_boundaries_from_surface(
        merged_cells, surface_faces, surface_boundaries, direct_cell_groups=cell_groups
    )

    # Merge prism-side boundary groups into tet_boundaries' global index
    # space. Prisms occupy [0, n_prism) already (this module's global-index
    # convention, matching FaceExtractor.extract_faces_mixed); tet_boundaries'
    # own cell indices are LOCAL to merged_cells (tets only) and must be
    # shifted by +n_prism to land in the same shared space. Built directly
    # from bl_cell_groups (known exactly per prism cell from BL-extrusion
    # tracking - see _build_merged_mesh) rather than routed through
    # identify_boundaries_from_surface at all: that function's own boundary-
    # face derivation is tet-specific (4-face templates), and re-deriving
    # it for prisms is unnecessary work when the group name is already
    # known per-cell directly, the same reasoning direct_cell_groups
    # already uses for BL-extruded tet cells prior to this change.
    groups: Dict[str, np.ndarray] = {}
    bc_types: Dict[str, str] = {}
    final_mixed_faces = None
    if n_prism > 0:
        for name in np.unique(bl_cell_groups):
            if not name:
                continue
            idx = np.flatnonzero(bl_cell_groups == name).astype(np.int32)
            groups[name] = idx
            bc_types[name] = surface_boundaries.bc_types.get(name, 'WALL')

        # Untagged (bl_cell_groups=='') prisms that still own a genuine
        # boundary face are the early-BL-column-termination artifacts
        # described where bl_cell_groups is built above (a column that
        # stopped growing at a sharp/complex feature, layers past the first
        # deliberately left untagged) - a real exterior face, but not the
        # physical wall. Route them into the same 'UNCLASSIFIED' catch-all
        # mesh_boundary.map_surface_boundaries already uses for the
        # analogous tet-side gap, instead of leaving them in no group at
        # all - the exact "silently dropped from every boundary condition"
        # failure mode that fallback exists to prevent (confirmed as a
        # real, not theoretical, gap on a real case - see ProjectFiles
        # Part6/7 P21: 33,448 such faces on cube_demo, concentrated at
        # sharp cube edges).
        untagged_mask = bl_cell_groups == ''
        if untagged_mask.any():
            final_mixed_faces = FaceExtractor.extract_faces_mixed(
                prism_cells, merged_cells.astype(np.int64), nodes_obj
            )
            boundary_idx = final_mixed_faces.get_boundary_face_indices()
            owners = final_mixed_faces.connectivity[boundary_idx, 0]
            prism_owners = owners[owners < n_prism]
            orphaned_prisms = np.unique(prism_owners[untagged_mask[prism_owners]]).astype(np.int32)
            if len(orphaned_prisms):
                logger.warning(
                    f"{len(orphaned_prisms)} prism cell(s) own a boundary face from "
                    f"early BL-column termination (sharp/complex geometry feature) "
                    f"but aren't the physical wall - placed in 'UNCLASSIFIED' instead "
                    f"of the real wall group, same fallback mesh_boundary.py already "
                    f"uses for the analogous tet-side gap"
                )
                if 'UNCLASSIFIED' in groups:
                    groups['UNCLASSIFIED'] = np.union1d(groups['UNCLASSIFIED'], orphaned_prisms).astype(np.int32)
                else:
                    groups['UNCLASSIFIED'] = orphaned_prisms
                    bc_types['UNCLASSIFIED'] = 'WALL'
    for name, idx in tet_boundaries.groups.items():
        shifted = (idx.astype(np.int64) + n_prism).astype(np.int32)
        if name in groups:
            groups[name] = np.union1d(groups[name], shifted).astype(np.int32)
        else:
            groups[name] = shifted
            bc_types[name] = tet_boundaries.bc_types.get(name, 'WALL')

    from ..structures import BoundaryMap
    boundaries_obj = BoundaryMap(groups=groups, bc_types=bc_types)

    metadata = GridMetadata(
        node_count=len(merged_nodes),
        cell_count=n_prism + len(merged_cells),
        boundary_groups=list(boundaries_obj.groups.keys()),
        file_format="hybrid"
    )

    volume_mesh = VolumeMeshData(
        nodes=nodes_obj,
        cells=cells_obj,
        boundaries=boundaries_obj,
        metadata=metadata,
        prism_cells=prism_cells_obj,
        # Reuse the mixed face graph already built just above (to find
        # orphaned-prism boundary faces for the UNCLASSIFIED fallback)
        # instead of paying for extract_faces_mixed a second time the
        # moment anything calls ensure_faces_exist() - None when there was
        # nothing untagged to check (no BL region, or every prism already
        # correctly grouped), in which case ensure_faces_exist() computes
        # it lazily as before.
        faces=final_mixed_faces,
    )

    logger.success(
        f"Domain-conforming hybrid mesh generation complete: "
        f"{volume_mesh.node_count} nodes, {volume_mesh.cell_count} cells "
        f"({n_prism} BL prisms + {len(merged_cells)} core tets), "
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
