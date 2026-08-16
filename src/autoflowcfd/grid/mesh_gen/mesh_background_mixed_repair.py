"""generate_hybrid_mesh 里 Stage A/B/B' 结束之后、最终装配之前的
"混合网格（棱柱+四面体）收尾修补"阶段：跨类型非流形面拼接、BL 棱柱畸变
长细比局部重铺、collapsed-corner 棱柱降级为四面体。

从 mesh_background.py 拆分出来（原文件超过 400 行上限），逐字搬移
generate_hybrid_mesh 原来紧接在"Final defensive pass: merge coincident
points and repair non-manifold"之后的那一整段，未改动任何数值逻辑——
三个子步骤共享同一组滚动状态（merged_nodes/prism_cells/merged_cells/
bl_cell_groups/cell_groups/nodes_obj/mesh_changed_by_repair），所以作为
一个整体一起搬移，而不是拆成三个更小的函数。
"""

import numpy as np
from typing import Tuple
from loguru import logger


def _repair_mixed_mesh_post_stage_c(
    merged_nodes: np.ndarray,
    prism_cells: np.ndarray,
    merged_cells: np.ndarray,
    bl_cell_groups: np.ndarray,
    cell_groups: np.ndarray,
    nodes_obj,
    mesh_changed_by_repair: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, bool]:
    """跨棱柱+四面体的非流形面修补、BL 棱柱长细比修补、collapsed-corner
    棱柱降级为四面体——见本模块文档字符串。逐字对应
    mesh_background.generate_hybrid_mesh 原来这一段代码，未改动任何数值
    逻辑。

    Returns:
        (merged_nodes, prism_cells, merged_cells, bl_cell_groups,
        cell_groups, nodes_obj, mesh_changed_by_repair) - 与传入参数一一
        对应，反映本阶段可能施加的任意次原地重建。
    """
    # 延迟导入，避免循环导入（约定见本项目 core/fr_solver_cfl.py 等）。
    from ..structures import NodeArray
    from .face_extractor import repair_nonmanifold_mixed
    from .mesh_prism_to_tet import orient_tetrahedra
    from .mesh_repair_nonmanifold_mixed import patch_nonmanifold_cavity_mixed, demote_invalid_prisms_to_tets

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

    return merged_nodes, prism_cells, merged_cells, bl_cell_groups, cell_groups, nodes_obj, mesh_changed_by_repair
