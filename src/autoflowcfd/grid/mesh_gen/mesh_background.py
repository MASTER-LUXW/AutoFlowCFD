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

import sys
import traceback
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
from .mesh_repair_cavity import patch_nonmanifold_cavity
from .mesh_repair_nonmanifold_mixed import patch_nonmanifold_cavity_mixed, demote_invalid_prisms_to_tets
from .mesh_background_merge import _build_merged_mesh

# Import refactored repair stages
from .mesh_overlap_handler import compute_extra_bad_mask
from .mesh_repair_stage_a import run_stage_a_repair
from .mesh_repair_stage_b import run_stage_b_repair


def generate_hybrid_mesh(
    surface_nodes: np.ndarray,
    surface_faces: np.ndarray,
    bounding_box: Dict[str, np.ndarray],
    growth_rate: float = 1.2,
    min_cell_size: float = 0.001,
    target_cells: int = 500000,
    surface_boundaries: Optional['BoundaryMap'] = None,
    max_cell_size: Optional[float] = None,
    extra_thickness_limit: Optional[np.ndarray] = None,
    bl_layers: Optional[int] = None,
    _is_stage_b_retry: bool = False,
    export_bl_only: bool = False,
    export_bl_only_path: Optional[str] = None,
    export_core_only: bool = False,
    export_core_only_path: Optional[str] = None,
) -> 'VolumeMeshData':
    """Generate a volume mesh that conforms exactly to the closed input surface."""
    try:
        if surface_boundaries is None or not surface_boundaries.groups:
            raise ValueError(
                "generate_hybrid_mesh requires surface_boundaries with at least "
                "one boundary group, used to classify wall-type surfaces for BL "
                "extrusion versus the outer domain shell"
            )

        logger.info("Starting domain-conforming hybrid mesh generation...")

        (merged_nodes, prism_cells, merged_cells, cell_groups, n_bl_prisms,
         bl_source_vertex, bl_extrude_faces, bl_cell_groups, n_bl_cells) = _build_merged_mesh(
            surface_nodes, surface_faces, bounding_box, surface_boundaries,
            growth_rate, min_cell_size, max_cell_size,
            extra_thickness_limit, bl_layers,
            export_bl_only=export_bl_only,
            export_bl_only_path=export_bl_only_path,
            export_core_only=export_core_only,
            export_core_only_path=export_core_only_path,
        )

        # If we are in *-only export mode, _build_merged_mesh will have already saved and exited
        # or returned a special signal. For now, we assume it returns normally but we skip TetGen logic.
        if export_bl_only or export_core_only:
            logger.success("Partial-pipeline export completed. Exiting.")
            import sys
            sys.exit(0)

        from ..structures import NodeArray, TetrahedralCells, PrismCells, GridMetadata, VolumeMeshData

        logger.info("Step 4/4: Re-orienting and computing tetrahedral volumes...")
        merged_cells = orient_tetrahedra(merged_nodes, merged_cells.astype(np.int64))
        _nodes_obj_tmp = NodeArray(
            x=merged_nodes[:, 0].copy(), y=merged_nodes[:, 1].copy(), z=merged_nodes[:, 2].copy()
        )
        volumes = TetrahedralCells.compute_volumes(_nodes_obj_tmp, merged_cells.astype(np.int32))

        # Drop degenerate cells
        degenerate_threshold = (min_cell_size ** 3) * 1e-6
        valid_mask = volumes > degenerate_threshold
        n_invalid = np.sum(~valid_mask)
        if n_invalid > 0:
            logger.warning(f"Found {n_invalid} degenerate cells, removing them...")
            n_bl_cells = int(np.sum(valid_mask[:n_bl_cells]))
            merged_cells = merged_cells[valid_mask]
            volumes = volumes[valid_mask]
            cell_groups = cell_groups[valid_mask]

        # Repair non-manifold faces - try a local retile first (fills the
        # gap a plain "keep largest, drop rest" repair would otherwise
        # leave when the extra cells came from two different regions
        # legitimately meeting at a sharp corner, not genuine duplicates -
        # see patch_nonmanifold_cavity's own docstring for the real
        # measured case, 0.189 m^3 of missing volume, that motivated this).
        nonmanifold_keep = repair_nonmanifold_cells(merged_nodes, merged_cells)
        if not nonmanifold_keep.all():
            merged_nodes, merged_cells, cell_groups, n_bl_cells, _ = patch_nonmanifold_cavity(
                merged_nodes, merged_cells, nonmanifold_keep, cell_groups, n_bl_cells,
            )
            # patch_nonmanifold_cavity falls back to returning its inputs
            # UNCHANGED (still non-manifold) when it can't safely patch -
            # re-run the plain keep-mask deletion in that case, same as
            # before this fix existed, so a defect it can't fix still
            # gets cleaned up rather than left in the mesh.
            nonmanifold_keep = repair_nonmanifold_cells(merged_nodes, merged_cells)
            # A cluster the default n_buffer_rings=1 attempt couldn't retile
            # often just needed a bigger, better-defined local boundary, not
            # because it's unfixable - escalate once before falling back to
            # plain deletion, which leaves a REAL hole (confirmed directly:
            # unconditional deletion at exactly this point, on a real
            # cube_demo run, produced a disconnected tet-only "phantom"
            # boundary shell enclosing genuinely empty space - see the mixed-
            # mesh non-manifold check further below, which had the identical
            # pattern and the same fix).
            if not nonmanifold_keep.all():
                merged_nodes, merged_cells, cell_groups, n_bl_cells, _ = patch_nonmanifold_cavity(
                    merged_nodes, merged_cells, nonmanifold_keep, cell_groups, n_bl_cells,
                    n_buffer_rings=4, max_cavity_cells=15_000,
                )
                nonmanifold_keep = repair_nonmanifold_cells(merged_nodes, merged_cells)
            if not nonmanifold_keep.all():
                n_deleted = int((~nonmanifold_keep).sum())
                del_pts = merged_nodes[np.unique(merged_cells[~nonmanifold_keep])]
                logger.warning(
                    f"Non-manifold tet repair: {n_deleted} cell(s) still unpatched after "
                    f"retry with a larger buffer ring - deleting as a last resort "
                    f"(bbox min={del_pts.min(axis=0)}, max={del_pts.max(axis=0)}); this "
                    f"leaves a real gap at that location, not just missing volume"
                )
                n_bl_cells = int(np.sum(nonmanifold_keep[:n_bl_cells]))
                merged_cells = merged_cells[nonmanifold_keep]
                cell_groups = cell_groups[nonmanifold_keep]
            _tmp_nodes_obj_nm = NodeArray(
                x=merged_nodes[:, 0].copy(), y=merged_nodes[:, 1].copy(), z=merged_nodes[:, 2].copy()
            )
            volumes = TetrahedralCells.compute_volumes(_tmp_nodes_obj_nm, merged_cells.astype(np.int32))

        # Merge coincident points (seam merge)
        n_nodes_before_seam_merge = len(merged_nodes)
        merged_nodes, merged_cells, _seam_remap = _dedupe_coincident_points(merged_nodes, merged_cells)
        if len(merged_nodes) != n_nodes_before_seam_merge:
            merged_cells = merged_cells.astype(np.int32)
            prism_cells = _seam_remap[prism_cells]
            _tmp_nodes_obj_seam = NodeArray(
                x=merged_nodes[:, 0].copy(), y=merged_nodes[:, 1].copy(), z=merged_nodes[:, 2].copy()
            )
            post_merge_volumes_seam = TetrahedralCells.compute_volumes(_tmp_nodes_obj_seam, merged_cells)
            degenerate_threshold_seam = (min_cell_size ** 3) * 1e-6
            valid_mask_seam = post_merge_volumes_seam > degenerate_threshold_seam
            if int(np.sum(~valid_mask_seam)) > 0:
                logger.warning(f"Seam merge left {int(np.sum(~valid_mask_seam))} newly-degenerate cells, removing them...")
                n_bl_cells = int(np.sum(valid_mask_seam[:n_bl_cells]))
                merged_cells = merged_cells[valid_mask_seam]
                volumes = volumes[valid_mask_seam]
                cell_groups = cell_groups[valid_mask_seam]
            nonmanifold_keep_seam = repair_nonmanifold_cells(merged_nodes, merged_cells)
            if not nonmanifold_keep_seam.all():
                merged_nodes, merged_cells, cell_groups, n_bl_cells, _ = patch_nonmanifold_cavity(
                    merged_nodes, merged_cells, nonmanifold_keep_seam, cell_groups, n_bl_cells,
                )
                nonmanifold_keep_seam = repair_nonmanifold_cells(merged_nodes, merged_cells)
                # Same escalate-before-delete fix as the pre-seam-merge check above.
                if not nonmanifold_keep_seam.all():
                    merged_nodes, merged_cells, cell_groups, n_bl_cells, _ = patch_nonmanifold_cavity(
                        merged_nodes, merged_cells, nonmanifold_keep_seam, cell_groups, n_bl_cells,
                        n_buffer_rings=4, max_cavity_cells=15_000,
                    )
                    nonmanifold_keep_seam = repair_nonmanifold_cells(merged_nodes, merged_cells)
                if not nonmanifold_keep_seam.all():
                    n_deleted = int((~nonmanifold_keep_seam).sum())
                    del_pts = merged_nodes[np.unique(merged_cells[~nonmanifold_keep_seam])]
                    logger.warning(
                        f"Non-manifold tet repair (post seam-merge): {n_deleted} cell(s) still "
                        f"unpatched after retry with a larger buffer ring - deleting as a last "
                        f"resort (bbox min={del_pts.min(axis=0)}, max={del_pts.max(axis=0)}); "
                        f"this leaves a real gap at that location, not just missing volume"
                    )
                    n_bl_cells = int(np.sum(nonmanifold_keep_seam[:n_bl_cells]))
                    merged_cells = merged_cells[nonmanifold_keep_seam]
                    cell_groups = cell_groups[nonmanifold_keep_seam]
                _tmp_nodes_obj_nm2 = NodeArray(
                    x=merged_nodes[:, 0].copy(), y=merged_nodes[:, 1].copy(), z=merged_nodes[:, 2].copy()
                )
                volumes = TetrahedralCells.compute_volumes(_tmp_nodes_obj_nm2, merged_cells.astype(np.int32))

        # ------------------------------------------------------------------
        # Mesh quality repair pipeline
        # ------------------------------------------------------------------
        from ..validation.quality_validator import MeshQualityValidator
        from .face_extractor import FaceExtractor, repair_nonmanifold_mixed

        merged_cells = merged_cells.astype(np.int32)
        validator = MeshQualityValidator()
        logger.info("Checking volume mesh quality (pre-repair)...")
        
        _pre_repair_node_arr = NodeArray(
            x=merged_nodes[:, 0].copy(), y=merged_nodes[:, 1].copy(), z=merged_nodes[:, 2].copy()
        )
        pre_repair_faces = FaceExtractor.extract_faces(merged_cells, _pre_repair_node_arr)
        initial_report = validator.validate(
            merged_nodes, merged_cells, cell_type="tetrahedron", faces=pre_repair_faces
        )

        # Compute extra bad mask (overlap detection)
        overlap_bad_mask = compute_extra_bad_mask(validator, initial_report, merged_nodes, prism_cells, merged_cells)

        # Run Stage A Repair
        nodes_before_repair = merged_nodes
        merged_nodes, bad_mask, repair_actions = run_stage_a_repair(
            merged_nodes, merged_cells, validator, pre_repair_faces, 
            overlap_bad_mask, n_bl_cells
        )
        mesh_changed_by_repair = not np.array_equal(nodes_before_repair, merged_nodes)

        # Run Stage B Repair (Cavity remesh + BL thickness capping)
        if np.any(bad_mask):
            (merged_nodes, merged_cells, cell_groups, bad_mask, stage_b_actions,
             extra_limit, bl_verts) = run_stage_b_repair(
                merged_nodes, merged_cells, cell_groups, n_bl_cells, pre_repair_faces,
                bad_mask, validator, min_cell_size, bl_source_vertex, bl_extrude_faces, surface_nodes
            )
            repair_actions.extend(stage_b_actions)

            # Handle Stage B retry logic (recursive call) - reuses the
            # extra_limit/bl_verts run_stage_b_repair already computed
            # (dijkstra-based, not free) instead of recomputing them here
            # with identical arguments.
            if np.any(bad_mask) and not _is_stage_b_retry:
                if extra_limit is not None:
                    logger.warning(f"Stage B: Retrying generation with targeted local BL thickness cap...")
                    del merged_nodes, merged_cells, volumes, cell_groups, bad_mask, initial_report
                    del prism_cells, bl_cell_groups
                    import gc
                    gc.collect()
                    return generate_hybrid_mesh(
                        surface_nodes, surface_faces, bounding_box,
                        growth_rate=growth_rate, min_cell_size=min_cell_size,
                        target_cells=target_cells, surface_boundaries=surface_boundaries,
                        max_cell_size=max_cell_size,
                        extra_thickness_limit=extra_limit,
                        bl_layers=bl_layers,
                        _is_stage_b_retry=True,
                    )

        # Final defensive pass: merge coincident points and repair non-manifold
        n_nodes_before_merge = len(merged_nodes)
        merged_nodes, merged_cells, _remap = _dedupe_coincident_points(merged_nodes, merged_cells)
        if len(merged_nodes) != n_nodes_before_merge:
            mesh_changed_by_repair = True
            merged_cells = merged_cells.astype(np.int32)
            prism_cells = _remap[prism_cells]
            
            # Check for new degeneracies after merge
            _tmp_nodes_obj = NodeArray(
                x=merged_nodes[:, 0].copy(), y=merged_nodes[:, 1].copy(), z=merged_nodes[:, 2].copy()
            )
            post_merge_volumes = TetrahedralCells.compute_volumes(_tmp_nodes_obj, merged_cells)
            degenerate_threshold = (min_cell_size ** 3) * 1e-6
            valid_mask = post_merge_volumes > degenerate_threshold
            if int(np.sum(~valid_mask)) > 0:
                logger.warning(f"Final merge left {int(np.sum(~valid_mask))} newly-degenerate cells, removing them...")
                merged_cells = merged_cells[valid_mask]
                cell_groups = cell_groups[valid_mask]

        # Build final objects
        nodes_obj = NodeArray(
            x=merged_nodes[:, 0].copy(), y=merged_nodes[:, 1].copy(), z=merged_nodes[:, 2].copy()
        )

        # Non-manifold check across mixed mesh - try a local retile first
        # (same rationale as the tet-only patch above: a plain "keep
        # largest, drop rest" repair leaves a hole when the extra cells
        # came from two different regions legitimately meeting at a sharp
        # corner rather than genuine duplicates; this is where the
        # REMAINDER of a real measured 0.189 m^3 deficit on cube_demo -
        # 0.147 m^3 still missing after the earlier tet-only patch already
        # fixed what it could - was traced to, since this check runs past
        # Stage A/B/C on the full prism+tet mesh and had no patch of its
        # own until now).
        if len(prism_cells):
            prism_keep_mm, tet_keep_mm = repair_nonmanifold_mixed(nodes_obj, prism_cells, merged_cells.astype(np.int64))
            if not prism_keep_mm.all() or not tet_keep_mm.all():
                merged_nodes, prism_cells, merged_cells, bl_cell_groups, cell_groups = patch_nonmanifold_cavity_mixed(
                    merged_nodes, prism_cells, merged_cells.astype(np.int64),
                    prism_keep_mm, tet_keep_mm, bl_cell_groups, cell_groups,
                )
                nodes_obj = NodeArray(
                    x=merged_nodes[:, 0].copy(), y=merged_nodes[:, 1].copy(), z=merged_nodes[:, 2].copy()
                )
                prism_keep_mm, tet_keep_mm = repair_nonmanifold_mixed(nodes_obj, prism_cells, merged_cells)

                # A cluster that failed the default n_buffer_rings=1 attempt
                # (tetgen exception, or its own retile turning out no better
                # than the original - see patch_nonmanifold_cavity_mixed's own
                # per-cluster loop) doesn't mean the defect is unfixable, just
                # that THAT cavity's boundary was too tight/oddly-shaped for
                # tetgen to work with. Escalate with a much larger buffer ring
                # (pulls in more surrounding good cells, giving tetgen a
                # better-defined boundary) before falling back to deletion -
                # unconditional deletion below leaves a REAL hole (confirmed
                # directly: on a real cube_demo run, this exact fallback
                # deleting ~48-65 failed clusters' worth of tets produced a
                # disconnected, tet-only "phantom" boundary shell enclosing
                # genuinely empty space in the wake region - not just missing
                # volume but a hole an outside viewer like ANSA can walk into,
                # since the surrounding survivors' newly-exposed faces close
                # up into their own self-consistent little manifold, passing
                # even the water-tightness open-edge check).
                if not prism_keep_mm.all() or not tet_keep_mm.all():
                    merged_nodes, prism_cells, merged_cells, bl_cell_groups, cell_groups = patch_nonmanifold_cavity_mixed(
                        merged_nodes, prism_cells, merged_cells.astype(np.int64),
                        prism_keep_mm, tet_keep_mm, bl_cell_groups, cell_groups,
                        n_buffer_rings=4, max_cavity_cells=15_000,
                    )
                    nodes_obj = NodeArray(
                        x=merged_nodes[:, 0].copy(), y=merged_nodes[:, 1].copy(), z=merged_nodes[:, 2].copy()
                    )
                    prism_keep_mm, tet_keep_mm = repair_nonmanifold_mixed(nodes_obj, prism_cells, merged_cells)

                if not prism_keep_mm.all() or not tet_keep_mm.all():
                    n_prism_del = int((~prism_keep_mm).sum())
                    n_tet_del = int((~tet_keep_mm).sum())
                    # Log WHERE this is happening, not just how many - a bare
                    # count gives no way to tell whether this run's deletions
                    # are a few scattered slivers (harmless) or, as measured
                    # on a real run, a large contiguous pocket (a real hole).
                    del_pts = []
                    if n_tet_del:
                        del_pts.append(merged_nodes[np.unique(merged_cells[~tet_keep_mm])])
                    if n_prism_del:
                        del_pts.append(merged_nodes[np.unique(prism_cells[~prism_keep_mm])])
                    if del_pts:
                        bbox = np.vstack(del_pts)
                        logger.warning(
                            f"Non-manifold mixed-cavity patch: {n_prism_del} prism(s) + "
                            f"{n_tet_del} tet(s) still unpatched after retry with a larger "
                            f"buffer ring - deleting as a last resort (bbox min={bbox.min(axis=0)}, "
                            f"max={bbox.max(axis=0)}); this leaves a real gap at that location, "
                            f"not just missing volume"
                        )
                    prism_cells = prism_cells[prism_keep_mm]
                    bl_cell_groups = bl_cell_groups[prism_keep_mm]
                    merged_cells = merged_cells[tet_keep_mm]
                    cell_groups = cell_groups[tet_keep_mm]
                mesh_changed_by_repair = True

        # BL prism aspect-ratio repair: Stage A/B/B' above only ever
        # operate on merged_cells (transition/core tets) - prism_cells is
        # never touched by any of them, so a severely thin "collapsed-
        # corner" prism (a BL column whose growth froze at exactly one
        # base vertex - see quality_metrics.compute_prism_aspect_ratios'
        # own docstring, "ProjectFiles Part6 Bug 4", a valid nonzero-
        # volume cell, not a generation error) has NO repair path at all
        # today and survives unconditionally, however extreme (measured
        # directly: max BL aspect ratio pinned at that function's own
        # 1e6 reporting cap, i.e. a min edge under a millionth of the
        # cell's own longest edge). Reuses the exact same local-cavity
        # patch machinery the non-manifold fix above uses - the seed
        # condition there is just "this cell is marked for removal",
        # which a bad-aspect-ratio keep-mask satisfies identically to a
        # non-manifold one; the retile that comes back replaces the
        # collapsed prism(s) with ordinary tets, which can represent an
        # arbitrarily thin corner without the extreme-ratio artifact a
        # prism's fixed cap/side-quad topology forces on a frozen column.
        if len(prism_cells):
            from ..validation.quality_metrics import compute_prism_aspect_ratios
            prism_ar = compute_prism_aspect_ratios(merged_nodes, prism_cells)
            # Deliberately much looser than the quality report's own
            # bl_max_aspect_ratio=50 threshold (an ordinary BL cell is
            # SUPPOSED to be elongated - see compute_prism_aspect_ratios'
            # own docstring) - this pass targets only the genuinely
            # collapsed/degenerate outliers a local retile can actually
            # improve on, not every merely-stretched-but-fine BL cell.
            ar_keep = prism_ar <= 500.0
            if not ar_keep.all():
                n_bad_ar = int((~ar_keep).sum())
                logger.warning(
                    f"{n_bad_ar} BL prism(s) with extreme aspect ratio "
                    f"(collapsed-corner columns, max={float(prism_ar.max()):.3g}) - "
                    f"attempting local cavity patch"
                )
                tet_keep_allones = np.ones(len(merged_cells), dtype=bool)
                merged_nodes, prism_cells, merged_cells, bl_cell_groups, cell_groups = patch_nonmanifold_cavity_mixed(
                    merged_nodes, prism_cells, merged_cells.astype(np.int64),
                    ar_keep, tet_keep_allones, bl_cell_groups, cell_groups,
                )
                # A successful patch appends new interior nodes to
                # merged_nodes - nodes_obj (built before this block) must
                # be rebuilt from the possibly-larger array before anything
                # downstream indexes into it, or a cell referencing one of
                # those new nodes indexes past the end of the stale array.
                # Confirmed directly, not theoretical: this exact gap
                # crashed the very next line (TetrahedralCells.
                # compute_volumes) on a real run once clusters were large
                # enough to actually need new interior points.
                nodes_obj = NodeArray(
                    x=merged_nodes[:, 0].copy(), y=merged_nodes[:, 1].copy(), z=merged_nodes[:, 2].copy()
                )
                mesh_changed_by_repair = True

        # Deterministic fallback for whatever the tetgen-based AR patch just
        # above could not fix: any prism still referencing the same node
        # twice among its own 6 vertices is a malformed CPENTA record (not
        # merely low quality - external tools validate this and reject it
        # outright; confirmed directly against a real ANSA 21.0.1 import,
        # which rejected ~21,000 such records with "invalid node
        # combination", one per collapsed-corner prism the AR patch above
        # left untouched because tetgen cannot retile a near-zero-volume
        # cavity). Pure arithmetic, cannot fail the way the tetgen patch
        # can, so this must run unconditionally as a final invariant check,
        # not only when the AR patch above reports remaining failures.
        if len(prism_cells):
            prism_cells, bl_cell_groups, extra_tets, extra_tet_groups = demote_invalid_prisms_to_tets(
                prism_cells, bl_cell_groups
            )
            if len(extra_tets):
                # _split_prisms_to_tets' fixed template assumes a well-formed
                # prism's own bottom/top winding; a collapsed-corner prism's
                # near-zero geometry can flip that near-degenerate case,
                # so re-orient explicitly rather than trust the template -
                # same convention line ~93 already applies to merged_cells
                # right after its first construction.
                extra_tets = orient_tetrahedra(merged_nodes, extra_tets.astype(np.int64))
                merged_cells = np.vstack([merged_cells.astype(np.int64), extra_tets])
                cell_groups = np.concatenate([cell_groups, extra_tet_groups])
                mesh_changed_by_repair = True

        # TetrahedralCells enforces int32 connectivity strictly; the patch
        # path above works in int64 throughout (matching the .astype(np.
        # int64) cast fed into it, needed since patch_nonmanifold_cavity_
        # mixed's own remap arithmetic can produce indices during
        # construction that transiently exceed int32 range on a very large
        # mesh) and must be cast back before building the final object.
        merged_cells = merged_cells.astype(np.int32)
        volumes = TetrahedralCells.compute_volumes(nodes_obj, merged_cells)
        cells_obj = TetrahedralCells(connectivity=merged_cells, volumes=volumes)

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

        # Merge boundary groups
        groups: Dict[str, np.ndarray] = {}
        bc_types: Dict[str, str] = {}
        if n_prism > 0:
            for name in np.unique(bl_cell_groups):
                if not name: continue
                idx = np.flatnonzero(bl_cell_groups == name).astype(np.int32)
                groups[name] = idx
                bc_types[name] = surface_boundaries.bc_types.get(name, 'WALL')

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

        return VolumeMeshData(
            nodes=nodes_obj,
            cells=cells_obj,
            boundaries=boundaries_obj,
            metadata=metadata,
            prism_cells=prism_cells_obj,
        )
    except Exception as e:
        logger.error(f"Error in generate_hybrid_mesh: {e}")
        traceback.print_exc()
        raise