"""generate_hybrid_mesh 用到的、重复出现两次的非流形面修复+兜底逻辑。

generate_hybrid_mesh（mesh_background.py）在 seam merge 前、seam merge 后
各跑一遍几乎一样的"repair_nonmanifold_cells -> 不行就局部重铺
（patch_nonmanifold_cavity）-> 还不行就放大缓冲环再试一次 -> 实在不行就
兜底删除"流程，抽成这一个共享函数，避免同一段逻辑维护两份。
"""

from typing import Tuple

import numpy as np
from loguru import logger

from .mesh_tetgen_core import repair_nonmanifold_cells
from .mesh_repair_cavity import patch_nonmanifold_cavity


def repair_nonmanifold_tets_with_escalation(
    nodes: np.ndarray,
    cells: np.ndarray,
    cell_groups: np.ndarray,
    n_bl_cells: int,
    context_suffix: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    """Repair non-manifold tets: local retile first (fills the gap a plain
    "keep largest, drop rest" repair would otherwise leave when the extra
    cells came from two different regions legitimately meeting at a sharp
    corner, not genuine duplicates - see patch_nonmanifold_cavity's own
    docstring for the real measured case, 0.189 m^3 of missing volume, that
    motivated this), escalating to a larger buffer ring once before finally
    falling back to plain deletion (which leaves a REAL hole - confirmed
    directly: unconditional deletion at this point, on a real cube_demo
    run, produced a disconnected tet-only "phantom" boundary shell
    enclosing genuinely empty space).

    Args:
        nodes, cells: current merged mesh
        cell_groups: (n_cells,) str array parallel to cells
        n_bl_cells: BL cells occupy cells[:n_bl_cells]
        context_suffix: appended to the last-resort-deletion warning
            message, to distinguish the pre-seam-merge and post-seam-merge
            call sites in the log

    Returns:
        (nodes, cells, cell_groups, n_bl_cells, changed) - changed is True
        iff anything was actually repaired/deleted, so the caller knows
        whether it needs to recompute cell volumes.
    """
    nonmanifold_keep = repair_nonmanifold_cells(nodes, cells)
    if nonmanifold_keep.all():
        return nodes, cells, cell_groups, n_bl_cells, False

    nodes, cells, cell_groups, n_bl_cells, _ = patch_nonmanifold_cavity(
        nodes, cells, nonmanifold_keep, cell_groups, n_bl_cells,
    )
    # patch_nonmanifold_cavity falls back to returning its inputs UNCHANGED
    # (still non-manifold) when it can't safely patch - re-run the plain
    # keep-mask check in that case, same as before this fix existed, so a
    # defect it can't fix still gets cleaned up rather than left in the mesh.
    nonmanifold_keep = repair_nonmanifold_cells(nodes, cells)
    # A cluster the default n_buffer_rings=1 attempt couldn't retile often
    # just needed a bigger, better-defined local boundary, not because it's
    # unfixable - escalate once before falling back to plain deletion.
    if not nonmanifold_keep.all():
        nodes, cells, cell_groups, n_bl_cells, _ = patch_nonmanifold_cavity(
            nodes, cells, nonmanifold_keep, cell_groups, n_bl_cells,
            n_buffer_rings=4, max_cavity_cells=15_000,
        )
        nonmanifold_keep = repair_nonmanifold_cells(nodes, cells)
    if not nonmanifold_keep.all():
        n_deleted = int((~nonmanifold_keep).sum())
        del_pts = nodes[np.unique(cells[~nonmanifold_keep])]
        logger.warning(
            f"Non-manifold tet repair{context_suffix}: {n_deleted} cell(s) still unpatched "
            f"after retry with a larger buffer ring - deleting as a last resort "
            f"(bbox min={del_pts.min(axis=0)}, max={del_pts.max(axis=0)}); this "
            f"leaves a real gap at that location, not just missing volume"
        )
        n_bl_cells = int(np.sum(nonmanifold_keep[:n_bl_cells]))
        cells = cells[nonmanifold_keep]
        cell_groups = cell_groups[nonmanifold_keep]

    return nodes, cells, cell_groups, n_bl_cells, True
