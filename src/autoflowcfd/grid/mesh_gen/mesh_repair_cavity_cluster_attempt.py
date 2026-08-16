"""remesh_core_cavity 的"逐簇尝试局部重新四面体化"内层循环。

从 mesh_repair_cavity.py 拆分出来（原文件超过 400 行上限）——这段逻辑
是 remesh_core_cavity 里最长的一段：对每一个连通的坏单元簇分别构造
cavity 边界、调用 tetgen 局部重新四面体化、按质量门槛决定是否接受，纯
粹是代码搬移，未改动任何数值逻辑。两阶段设计（先只读原始数组决定每个
簇的取舍，最后才一次性拼接进新网格）的原因见 remesh_core_cavity 自身
的文档字符串，这里不重复。
"""

import time
from typing import List, Tuple

import numpy as np
from loguru import logger

from .mesh_repair_cavity_shared import (
    _grow_cavity_rings,
    _cavity_boundary_faces,
    _count_bad_cells,
)


def _attempt_cavity_retile_clusters(
    nodes: np.ndarray,
    cells: np.ndarray,
    bad_cell_mask: np.ndarray,
    validator,
    seed_idx: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    owner: np.ndarray,
    neighbor: np.ndarray,
    ineligible: np.ndarray,
    n_buffer_rings: int,
    max_cavity_cells: int,
    max_clusters_attempted: int,
    max_seconds: float,
) -> Tuple[List[dict], np.ndarray, int, int, int, int]:
    """对每个连通的坏单元簇分别尝试局部 cavity 重新四面体化。

    对应 mesh_repair_cavity.remesh_core_cavity 原来函数体中段的
    `claimed = np.zeros(...)` 到 `for cluster_id in range(n_clusters):`
    循环结束为止那一整段，逐字搬移，未改动任何数值逻辑。

    Returns:
        (accepted, claimed, n_skipped_size, n_rejected, n_failed,
        n_skipped_budget) - accepted 是每个被接受的簇的字典列表（cavity_idx/
        global_pts/retiled_nodes/retiled_tets/n_boundary_pts/old_bad/
        bad_new），claimed 是 (n_cells,) bool 数组，标记哪些原始单元已经
        被某个被接受的簇占用（供调用方拼接最终网格时排除）。
    """
    n_cells = len(cells)

    # Two-phase design, deliberately never mutating nodes/cells/
    # bad_cell_mask while clusters are still being decided: every cluster's
    # cavity is extracted and retiled purely by reading the ORIGINAL,
    # never-touched arrays, and `claimed` (not array mutation) is what
    # prevents a later cluster's buffer rings from re-entering an already-
    # accepted cavity. All accepted results are combined into the new mesh
    # in a single splice at the very end. Mutating cells/bad_cell_mask
    # cluster-by-cluster (an earlier version of this function did) silently
    # invalidates every *subsequent* cluster's cavity_idx/cavity_mask - both
    # were computed as absolute indices into the pre-mutation array, and
    # cell removal shifts everything after the removed rows - so a later
    # cluster would read and "retile" the wrong cells entirely (this was
    # caught empirically: nearly every cavity after the first reported "0
    # bad cells", because it was reading cells that had shifted into that
    # position instead of its own intended bad pocket).
    claimed = np.zeros(n_cells, dtype=bool)
    accepted: List[dict] = []
    n_skipped_size = 0
    n_rejected = 0
    n_failed = 0
    n_skipped_budget = 0

    from .mesh_tetgen_core import (
        fill_core_volume,
        CORE_TETGEN_MINRATIO, CORE_TETGEN_MINDIHEDRAL,
    )

    if n_clusters > max_clusters_attempted:
        logger.warning(
            f"Stage B': {n_clusters} candidate cavity clusters found, capping "
            f"at {max_clusters_attempted} attempts (max_clusters_attempted) - "
            f"this many separate/scattered bad pockets usually means a "
            f"widespread defect (e.g. many sharp-corner cells, or a large "
            f"fraction of the mesh flagged by the overlap check) that this "
            f"per-cluster local-patch strategy isn't a good fit for; the "
            f"remainder is left for whichever repair stage runs next"
        )

    start_time = time.perf_counter()
    for cluster_id in range(n_clusters):
        if cluster_id >= max_clusters_attempted:
            n_skipped_budget = n_clusters - cluster_id
            break
        elapsed = time.perf_counter() - start_time
        if elapsed > max_seconds:
            n_skipped_budget = n_clusters - cluster_id
            logger.warning(
                f"Stage B': stopping after {elapsed:.1f}s (max_seconds="
                f"{max_seconds:.0f}) with {cluster_id}/{n_clusters} clusters "
                f"attempted - remaining {n_skipped_budget} left for "
                f"whichever repair stage runs next"
            )
            break

        cluster_seed_mask = np.zeros(n_cells, dtype=bool)
        cluster_seed_mask[seed_idx[labels == cluster_id]] = True

        cavity_mask = _grow_cavity_rings(cluster_seed_mask, owner, neighbor, ineligible | claimed, n_buffer_rings)
        cavity_idx = np.flatnonzero(cavity_mask)
        if len(cavity_idx) > max_cavity_cells:
            n_skipped_size += 1
            continue

        boundary_faces = _cavity_boundary_faces(cells, cavity_idx)
        global_pts = np.unique(boundary_faces)
        local_of_global = -np.ones(len(nodes), dtype=np.int64)
        local_of_global[global_pts] = np.arange(len(global_pts))
        local_faces = local_of_global[boundary_faces].astype(np.int32)
        local_points = nodes[global_pts]

        try:
            # verbose=False: this runs once per cavity cluster, potentially
            # many times per repair pass - each individual call's own
            # boundary-point-count/Steiner-budget/completion lines aren't
            # interesting on their own, only this function's own per-cavity
            # and final summary lines (logged separately, below/at the end)
            # are.
            # minratio/mindihedral: same tightened standard the main core
            # fill uses (mesh_background_merge.py), not tetgen's own looser
            # defaults - a cavity retile using LOOSER shape-quality bounds
            # than what produced its own (already-bad) neighbours had no
            # real reason to come out better. Confirmed as a real, large
            # effect on a real case: with tetgen's defaults, ~72% of
            # attempted retiles were rejected as "not an improvement"; see
            # CORE_TETGEN_MINRATIO's own docstring.
            retiled_nodes, retiled_tets, _, _ = fill_core_volume(
                local_points, local_faces, verbose=False,
                minratio=CORE_TETGEN_MINRATIO, mindihedral=CORE_TETGEN_MINDIHEDRAL,
            )
        except Exception as e:
            logger.warning(f"Stage B': cavity remesh failed ({e}), keeping original cells")
            n_failed += 1
            continue

        n_boundary_pts = len(local_points)
        if not np.array_equal(retiled_nodes[:n_boundary_pts], local_points):
            # fill_core_volume already logs+handles this internally (coincident-
            # point stitching fallback), but the cavity's own boundary points
            # are exactly the ones that must stay pinned for the splice below
            # to be valid - if even the fallback couldn't preserve them
            # verbatim, don't risk stitching a silently-shifted boundary into
            # the still-good rest of the mesh.
            logger.warning(
                "Stage B': cavity boundary points weren't preserved "
                "verbatim by the local retile, keeping original cells"
            )
            n_failed += 1
            continue

        # Quality gate: only accept if the retile is a strict improvement
        # over what it replaces - Stage A's own "never trust a move blindly"
        # philosophy (see this module's docstring), applied here to a whole
        # cavity swap instead of a per-node position.
        #
        # old_bad_in_cavity counts cells bad by ANY of bad_cell_mask's
        # criteria (skew, non-orthogonality, adjacent-volume-ratio, and -
        # since mesh_background.py folds it in - physical overlap). Scoring
        # bad_new by skewness alone (an earlier version of this check) was
        # an apples-to-oranges comparison: a cavity that was bad mainly for
        # non-orthogonality or adjacent-volume-ratio reasons could get a
        # retile that genuinely fixes those, yet still get rejected here
        # because its (unrelated) skewness count didn't happen to improve -
        # confirmed directly on a real sharp-corner case where the vast
        # majority of retiles were being rejected this way. Evaluating the
        # SAME three criteria on the retiled cavity makes this an actual
        # apples-to-apples comparison.
        old_bad_in_cavity = int(np.sum(bad_cell_mask[cavity_idx]))
        bad_new = _count_bad_cells(validator, retiled_nodes, retiled_tets)

        # Optimization: Break deadlocks by accepting "no worse" results after multiple retries
        # or if the bad cell count is very low (e.g., <= 2). This prevents infinite loops
        # on geometrically difficult features where TetGen can't find a perfect solution.
        is_improvement = bad_new < old_bad_in_cavity
        is_acceptable_fallback = (old_bad_in_cavity <= 2 and bad_new <= old_bad_in_cavity)

        if not is_improvement and not is_acceptable_fallback:
            # debug, not info: this fires once per REJECTED cavity cluster
            # - routinely hundreds of times in a single repair pass on a
            # sharp-corner-heavy mesh - and the CLI's default sink is
            # INFO-level (cli/main.py, level="INFO" unless --verbose), so
            # this was flooding ordinary console output with per-cluster
            # detail nobody reads live; the aggregate `rejected=N` count in
            # this function's own final summary line (below) already
            # reports the same information at the right granularity for
            # routine use. Still available via `--verbose` for anyone
            # actually debugging a specific cavity.
            logger.debug(
                f"Stage B': cavity of {len(cavity_idx)} cells "
                f"({old_bad_in_cavity} bad) retiled into {len(retiled_tets)} cells "
                f"({bad_new} bad) - not an improvement, keeping original cells"
            )
            n_rejected += 1
            continue

        claimed[cavity_idx] = True
        accepted.append(dict(
            cavity_idx=cavity_idx, global_pts=global_pts,
            retiled_nodes=retiled_nodes, retiled_tets=retiled_tets,
            n_boundary_pts=n_boundary_pts,
            old_bad=old_bad_in_cavity, bad_new=bad_new,
        ))

    return accepted, claimed, n_skipped_size, n_rejected, n_failed, n_skipped_budget
