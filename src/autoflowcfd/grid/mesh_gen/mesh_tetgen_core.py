"""Constrained tetrahedralization of the domain core using tetgen.

Fills the volume enclosed by a closed piecewise-linear complex (PLC) - the
boundary-layer (BL) outer surface plus the unmodified outer-shell faces
(inlet/outlet/tunnel/symmetry-like boundaries) - with tetgen, instead of the
old arbitrary padded-bounding-box + Cartesian background grid. The PLC is by
construction exactly the closed surface the input mesh already describes, so
the result can never extend outside the real domain.

本文件只保留 fill_core_volume 本体和相关常量。拆分出去的部分：
    - mesh_tetgen_seam.py：BL/core 交界（seam）过渡缩放、局部厚度限制
    - mesh_tetgen_postprocess.py：填充后清理（重合点合并、超大四面体细分、
      非流形修复、边界分组反推）
    - mesh_tetgen_seeding.py：Steiner 点预算估算、远场背景点播种
下面统一从这三个文件转出，外部代码一律仍从 `mesh_tetgen_core` 导入即可，
不需要关心内部是怎么拆的。
"""

from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from .mesh_tetgen_seam import (
    build_seam_taper_scale,
    compute_local_thickness_limit,
)
from .mesh_tetgen_postprocess import (
    _dedupe_coincident_points,
    _tet_volumes,
    subdivide_oversized_tetrahedra,
    repair_nonmanifold_cells,
    attribute_cells_from_trifaces,
)
from .mesh_tetgen_seeding import (
    estimate_steinerleft,
    generate_core_background_points,
)

__all__ = [
    'CORE_TETGEN_MINRATIO',
    'CORE_TETGEN_MINDIHEDRAL',
    'CORE_VOLUME_CAP_FRACTION',
    'build_seam_taper_scale',
    'compute_local_thickness_limit',
    'subdivide_oversized_tetrahedra',
    'repair_nonmanifold_cells',
    'attribute_cells_from_trifaces',
    'estimate_steinerleft',
    'generate_core_background_points',
    'fill_core_volume',
]

# Core-fill tetgen quality/grading knobs, shared by every caller of
# fill_core_volume that wants this project's own tightened standard rather
# than tetgen's out-of-the-box defaults (minratio~2.0, mindihedral~0
# effectively unconstrained). Originally lived only in
# mesh_background_merge.py (the main core fill's own caller) - moved here,
# the lowest-level module every one of fill_core_volume's callers already
# imports from, specifically so mesh_repair_cavity.py's Stage B' (local
# cavity re-tiling) can use the SAME standard for its own, much smaller
# fill_core_volume calls instead of silently falling back to tetgen's
# looser defaults. That inconsistency was a real, measured gap, not
# theoretical: Stage B' was rejecting ~72% of its own cavity retile
# attempts as "not an improvement" on a real case, and the retile itself
# had no reason to actually BE an improvement over the original (already
# badly-graded) cavity while using looser shape-quality bounds than what
# produced that cavity's own neighbours in the first place.
CORE_TETGEN_MINRATIO = 1.15  # was 1.4; tetgen default ~2.0 (lower = stricter)
CORE_TETGEN_MINDIHEDRAL = 15.0  # unchanged - dihedral wasn't the implicated metric
CORE_VOLUME_CAP_FRACTION = 0.08  # was 0.15, of max_cell_size**3


# Rough conversion from a target edge length to a tetgen maxvolume cap
# (regular-tet volume/edge^3 is ~0.118; Delaunay-refined tets are less
# regular and tetgen's own region cap isn't strictly tight in practice -
# so this is deliberately generous, not exact).
_VOLUME_SHAPE_FACTOR = 0.15


# NOTE: an earlier version of this module graded the core fill's max
# cell size outward from the wall via nested icosphere regions
# (build_graded_regions/_generate_icosphere). It was abandoned - tetgen's
# per-region variable-volume refinement does not reliably converge
# multiple simultaneous regions to their own targets when they compete
# for one shared Steiner budget (see fill_core_volume's `regions` doc) -
# in favor of the single flat region mesh_background.py builds directly.
# Removed rather than left unreferenced to avoid it being wired back in
# without that context.

def fill_core_volume(
    points: np.ndarray,
    faces: np.ndarray,
    minratio: float = 1.4,
    mindihedral: float = 15.0,
    holes: Optional[List[np.ndarray]] = None,
    regions: Optional[List[Tuple[np.ndarray, int, float]]] = None,
    face_markers: Optional[np.ndarray] = None,
    background_points: Optional[np.ndarray] = None,
    verbose: bool = True,
    force_preserve_boundary: bool = False,
    allow_boundary_bisect: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Constrained-tetrahedralize the volume enclosed by a closed PLC.

    Args:
        points: (n_points, 3) float64 PLC vertices
        faces: (n_faces, 3) int32 PLC triangles (closed, watertight)
        minratio: max radius-edge ratio quality bound (tetgen convention;
            lower = higher quality, 1.0 is a perfect tet)
        mindihedral: min dihedral angle quality bound (degrees)
        allow_boundary_bisect: explicitly forces nobisect=False even when
            `regions` is unset (the default, no-`regions` behaviour is
            nobisect=True - see this function's own nobisect comment
            below). Use when the given boundary is only an ESTIMATE that
            may not be perfectly valid (e.g. a near-self-intersecting
            proxy surface) and tetgen's own boundary-recovery robustness
            handling (Steiner-point insertion, coincident-point
            resolution) is preferred over hard-failing - the caller must
            then treat the RETURNED boundary as authoritative (via this
            function's own `conformal` check/fallback) rather than
            assuming the input `points` survive verbatim as an exact
            prefix. Takes priority over force_preserve_boundary if both
            are somehow set (mutually contradictory intents - this one
            wins since it was requested last/more specifically by design).
        force_preserve_boundary: forces tetgen's own `-Y` switch
            (nobisect=True) even when `regions` is set - the ordinary
            behaviour (see this function's own nobisect comment below)
            allows region-based grading only by ALSO permitting tetgen to
            insert Steiner points on the given boundary itself, which is
            fine when the caller doesn't need that exact boundary
            preserved elsewhere. Set this when the boundary given here is
            ALSO used, unchanged, as a fixed input to another, separate
            tetrahedralization that must match it exactly - e.g. the
            "fill, don't extrude" transition-region strategy in
            mesh_background_merge._build_merged_mesh, where the SAME
            estimated core-side surface is handed to both this call (as
            its own outer boundary) and a separate transition-gap fill (as
            ITS inner boundary): if either call let tetgen subdivide that
            shared surface independently, the two meshes would no longer
            agree on it and the splice between them would tear. Grading
            still works normally with this on - region-based interior
            refinement (regionattrib/varvolume below) only ever inserts
            points in the tet INTERIOR, never on the boundary, so -Y
            doesn't suppress it (verified directly: near/far tet volume
            ratio unaffected - see mesh_tetgen_core.py's own historical
            comment on nobisect+regions, since corrected, for the
            unrelated coupling bug that used to make it LOOK like -Y broke
            grading).
        holes: points, one strictly inside each isolated embedded solid in
            the PLC (mesh_domain_classify.find_point_inside_closed_shell).
            Without these, tetgen has no way to know an internal closed
            surface bounds a solid rather than just another constraint -
            it fills the fluid region around it AND that solid's own
            (BL-extruded) interior, producing spurious tetrahedra that
            overlap the BL prisms already occupying that cavity.
        regions: (seed_point, region_id, maxvolume) tuples (built by the
            caller, mesh_background.py) for capping max cell size per
            graded tier. Note: tetgen's own background-mesh sizing (`bgmesh`/
            `metric`, tetgen 0.8.4) is not used here - it segfaults
            unconditionally in this environment and package version
            regardless of settings (reproducible on a trivial cube,
            matching an unresolved upstream issue with no test coverage
            for that path) - region-based grading is used instead: proven
            stable, if less smoothly continuous.

            Passing `regions` switches off `nobisect`: enforcing a max
            cell size near a coarse far-field boundary facet (e.g. a
            sparsely-triangulated tunnel/inlet/outlet wall) requires
            tetgen to be allowed to subdivide that facet itself - with
            `nobisect` on (the default, no `regions`), any region touching
            the domain's own outer boundary is provably unaffected by its
            volume cap at all (verified: identical output with and
            without the cap on a boundary-adjacent region), because
            `nobisect` forbids inserting points on or near boundary
            facets and that blocks volume-based splitting of the
            boundary-adjacent cells too, not just the facets themselves.
        face_markers: (n_faces,) int32, one marker per input face, required
            together with `regions` - the boundary attribution mechanism
            (mesh_background.py) can no longer match subdivided boundary
            faces back to their source group by node index (nobisect=False
            means those indices no longer exist verbatim in the input), so
            it uses tetgen's own facet markers instead, which are inherited
            by every sub-facet a marked facet gets split into and are
            returned via this function's 3rd/4th outputs.
        background_points: (q, 3) optional extra points, NOT referenced by
            any row of `faces`, appended to `points` before tetgen ever
            runs (see `generate_core_background_points` above for how to
            build these for the sparse-far-field-escaped-tet problem).
            tetgen accepts free (non-facet) points as ordinary input
            vertices and incorporates them into its initial Delaunay step
            verbatim - confirmed directly on a synthetic cube-PLC-plus-3-
            interior-points test, all 3 appeared in the output node array
            at their exact input coordinates and were referenced by 60/102
            output tets. Left as None (unchanged default) for every
            existing caller that doesn't pass it.
        verbose: log this call's own routine per-call progress (boundary
            point/face counts, Steiner budget, completion) at INFO level
            (default, matching prior behavior exactly). False drops those
            same lines entirely (not merely demoted to DEBUG - this
            project's default loguru sink shows DEBUG and above, so a
            demotion alone would not actually reduce visible output) - for
            a caller that makes many small calls in a loop
            (mesh_repair.remesh_core_cavity, one call per repaired cavity)
            where each individual call's own progress isn't interesting on
            its own, only the caller's own summary is. Warnings (non-
            conformal boundary, self-intersection) always stay at their
            normal level regardless of this flag - they indicate something
            a caller needs to see, not routine progress.

    Returns:
        (nodes, tets, trifaces, triface_markers): nodes shape=(n, 3)
        float64 (input points preserved verbatim as the first len(points)
        rows, even under subdivision - verified empirically, tetgen only
        appends new points, it never reorders/replaces existing ones),
        tets shape=(m, 4) int64. trifaces/triface_markers are None unless
        `face_markers` was given, else the tetrahedralized boundary
        triangles (shape=(p, 3) int64, indices into `nodes`) and their
        inherited markers (shape=(p,) int32).
    """
    import tetgen

    points = np.ascontiguousarray(points, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)

    # Appended AFTER every point `faces` can reference, so none of the
    # bounds/degenerate-face checks just below (which only look at
    # `faces`/the ORIGINAL `points`) need to change. See this function's
    # own `background_points` doc for why these are safe/useful as free
    # (non-facet) input points.
    if background_points is not None and len(background_points) > 0:
        points = np.vstack([points, np.ascontiguousarray(background_points, dtype=np.float64)])
        logger.info(f"Adding {len(background_points)} background points to seed the initial tetrahedralization")

    # Relax quality constraints slightly to ensure convergence on a complex
    # BL surface.
    effective_minratio = max(1.1, minratio - 0.2)
    effective_mindihedral = max(5.0, mindihedral - 10.0)

    # nobisect=True (no regions) was unconditional here for a while, to
    # route around a real TetGen hang on THIS project's own BL outer
    # surface - but that surface was, at the time, coming out of
    # mesh_corner_split.py's corner-splitting/bevel-cap construction with
    # real defects of its own (see mesh_corner_split.py's and
    # mesh_layer_step.py's own docstrings - the valence-3+ corner handling
    # this project's own later work, P27/P28 in ProjectFiles' 3-3 Part8
    # report, specifically rebuilt). With `regions` (max_cell_size) unset,
    # nobisect=True is still forced unconditionally below (no behaviour
    # change from before for that case). With `regions` set, nobisect is
    # now allowed OFF - required for a max_cell_size region touching the
    # domain's own outer boundary to have any effect at all (see this
    # function's own `regions` doc) - now that the BL outer surface this
    # sits on is the geometry P27/P28 already fixed, not the one that
    # caused the original hang.
    #
    # Tried forcing this to True UNCONDITIONALLY (every caller, every
    # region) as a fix for a confirmed-real defect (726-882 of 22,830
    # BL/transition-outer interface facets coming back subdivided by
    # tetgen under nobisect=False, a genuine triangulation mismatch at the
    # interface) - verified directly that -Y does eliminate that
    # subdivision (0/22,830 afterward) WITHOUT disabling max_cell_size
    # grading (near/far tet volume ratio still ~15,000x) - but the actual
    # reported defects (166 X-junction boundary edges at sharp corners, a
    # disconnected ~24,000-face phantom boundary shell in the wake region)
    # were completely unchanged by it, since the extrusion-based
    # transition stage's own outer surface (what was being protected) was
    # never actually the thing tetgen disagreed with. Reverted as a
    # blanket default; the same -Y mechanism now exists as the OPT-IN
    # `force_preserve_boundary` parameter instead (see its own docstring
    # above) for the specific case that DOES need it: a boundary this call
    # is given that is ALSO independently used as a fixed input elsewhere
    # (mesh_background_merge's "fill, don't extrude" transition strategy).
    force_nobisect = ((not bool(regions)) or force_preserve_boundary) and not allow_boundary_bisect
    log = logger.info if verbose else (lambda *_a, **_k: None)

    log(
        f"Tetrahedralizing core volume: {len(points)} boundary points, "
        f"{len(faces)} boundary faces (tetgen, nobisect={force_nobisect}, "
        f"minratio={effective_minratio:.1f}, mindihedral={effective_mindihedral:.1f})..."
    )

    # Defensive check: ensure all face indices are within bounds and unique
    if np.any(faces < 0) or np.any(faces >= len(points)):
        raise RuntimeError(
            f"Invalid face indices detected in PLC boundary. "
            f"Faces range [{faces.min()}, {faces.max()}], but points count is {len(points)}."
        )

    # Check for degenerate faces (faces with duplicate vertices)
    sorted_faces = np.sort(faces, axis=1)
    degenerate_mask = (
        (sorted_faces[:, 0] == sorted_faces[:, 1]) |
        (sorted_faces[:, 1] == sorted_faces[:, 2]) |
        (sorted_faces[:, 0] == sorted_faces[:, 2])
    )
    n_degenerate = int(np.sum(degenerate_mask))
    if n_degenerate > 0:
        logger.warning(
            f"Found {n_degenerate} degenerate faces in PLC boundary. "
            f"Removing them before TetGen call to prevent hangs."
        )
        faces = faces[~degenerate_mask]
        if face_markers is not None:
            face_markers = face_markers[~degenerate_mask]

    if face_markers is not None:
        tgen = tetgen.TetGen(points, faces, np.ascontiguousarray(face_markers, dtype=np.int32))
    else:
        tgen = tetgen.TetGen(points, faces)
    if holes:
        for hole_pt in holes:
            tgen.add_hole(hole_pt)
        log(f"Marked {len(holes)} tetgen hole seed(s) for isolated embedded solids")
    # Registered whenever given, independent of force_nobisect (see
    # regionattrib/varvolume's own comment below for why -Y doesn't
    # conflict with region-based interior refinement).
    if regions:
        for seed_pt, region_id, maxvol in regions:
            tgen.add_region(region_id, seed_pt, maxvol)
        log(f"Marked {len(regions)} graded max-cell-size region(s)")

    steinerleft = estimate_steinerleft(points, regions)
    # Optimization: For sharp-corner models, increase the Steiner point budget
    steinerleft = max(steinerleft, 500_000)
    log(f"Steiner-point budget: {steinerleft:,}")

    try:
        nodes, elems, _attr, _markers = tgen.tetrahedralize(
            plc=True, nobisect=force_nobisect, quality=True,
            minratio=effective_minratio, mindihedral=effective_mindihedral,
            # Depends on `regions` alone, NOT on force_nobisect - region-
            # based interior refinement only ever inserts Steiner points in
            # the tet interior (never on the boundary), so it is orthogonal
            # to -Y regardless of why nobisect ended up True (no `regions`
            # at all, or force_preserve_boundary's own opt-in - see that
            # parameter's own docstring for why an earlier version of this
            # line, ANDed with `not force_nobisect`, silently broke grading
            # any time nobisect was forced True for an unrelated reason).
            regionattrib=bool(regions),
            varvolume=bool(regions),
            steinerleft=steinerleft,
            # Was hardcoded True regardless of this function's own
            # `verbose` param - meant every caller got tetgen's own raw
            # C-level console output (memorypool sizing, per-phase
            # progress, Steiner-point counts...) unconditionally, even
            # mesh_repair_cavity.remesh_core_cavity's own `verbose=False`
            # calls (one per cavity cluster, potentially hundreds per
            # repair pass) - exactly the console spam `verbose=False` was
            # supposed to suppress but couldn't, since it only gated this
            # function's own log() calls, never tetgen's native output.
            verbose=verbose,
        )
    except RuntimeError as e:
        if "self-intersection" in str(e).lower():
            raise RuntimeError(
                f"{e}. The BL outer surface self-intersects at a tight local "
                f"feature (common at small welded contact patches with sharp "
                f"edges). Try fewer/thinner BL layers (--bl-layers, "
                f"--min-cell-size) - naive normal-offset extrusion has no "
                f"per-feature thickness limiting yet, so cumulative BL "
                f"thickness must stay well under the tightest local gap in "
                f"the geometry."
            ) from e
        if "removevertexbyflips" in str(e).lower() or "internal tetgen error" in str(e).lower():
            # Observed on a real case when Stage B's reactive BL thickness
            # cap (mesh_repair.compute_bl_thickness_limit_override) needs to
            # cap a very large fraction of surface vertices - itself already
            # a symptom of Stage A leaving widespread, not localized, bad
            # cells - producing a boundary facet with enough near-coincident
            # points to exceed tetgen's own numerical robustness limits
            # internally (a tetgen implementation limitation, not a
            # meshing-strategy error on this codebase's side) rather than
            # failing with a clearer diagnostic like the self-intersection
            # case above.
            raise RuntimeError(
                f"{e}. tetgen hit an internal robustness limit - on a case "
                f"seen directly, this followed a very widespread Stage B "
                f"BL-thickness cap (a sign Stage A already found bad cells "
                f"across much of the surface, not just a few corners). Try "
                f"loosening --growth-rate/--min-cell-size/--bl-layers so "
                f"Stage A has fewer bad cells to begin with."
            ) from e
        raise

    trifaces = None
    triface_markers = None
    if face_markers is not None:
        trifaces = tgen.trifaces.astype(np.int64)
        triface_markers = tgen.triface_markers.astype(np.int32)

    n_input = len(points)
    conformal = nodes.shape[0] >= n_input and np.array_equal(nodes[:n_input], points)

    if not conformal:
        logger.warning(
            "tetgen did not preserve all boundary points verbatim "
            "(likely near-duplicate/degenerate input facets); "
            "falling back to coincident-point stitching"
        )
        nodes, elems, remap = _dedupe_coincident_points(nodes, elems)
        if trifaces is not None:
            # trifaces was read from tgen.trifaces in the PRE-dedupe index
            # space (same node array `nodes` was in before the line above).
            # Left unremapped, it desynced from the now-renumbered
            # nodes/elems - mesh_background.attribute_cells_from_trifaces
            # matches trifaces against core_tets by sorted-node-triple, so
            # a stale index space made that matching silently miss or
            # misattribute boundary cells whenever this fallback and
            # face_markers (i.e. max_cell_size) were both active.
            trifaces = remap[trifaces]

    log(f"Core tetrahedralization complete: {len(nodes)} nodes, {len(elems)} tets")

    return nodes.astype(np.float64), elems.astype(np.int64), trifaces, triface_markers
