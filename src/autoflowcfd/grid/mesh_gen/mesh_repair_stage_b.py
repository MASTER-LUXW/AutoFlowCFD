"""Stage B：结合 BL 厚度封顶与 cavity 重新铺网的定向再生成。

从 mesh_background.py 拆分出来以控制行数。
"""

import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ..schema.grid_faces import FaceData
    from ..validation.quality_validator import MeshQualityValidator


def run_stage_b_repair(
    merged_nodes: np.ndarray,
    merged_cells: np.ndarray,
    cell_groups: np.ndarray,
    n_bl_cells: int,
    pre_repair_faces: 'FaceData',
    bad_mask: np.ndarray,
    validator: 'MeshQualityValidator',
    min_cell_size: float,
    bl_source_vertex: np.ndarray,
    bl_extrude_faces: np.ndarray,
    surface_nodes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """Run Stage B: local cavity remesh and/or BL thickness capping.

    Args:
        merged_nodes: Node coordinates.
        merged_cells: Cell connectivity.
        cell_groups: Cell group labels.
        n_bl_cells: Number of BL cells.
        pre_repair_faces: Pre-extracted face data.
        bad_mask: Mask of bad cells from Stage A.
        validator: Quality validator instance.
        min_cell_size: Minimum cell size parameter.
        bl_source_vertex: Mapping from BL nodes to surface vertices.
        bl_extrude_faces: Faces used for BL extrusion.
        surface_nodes: Original surface nodes.

    Returns:
        Tuple of (new_nodes, new_cells, new_cell_groups, new_bad_mask,
        repair_actions, extra_limit, bl_verts) - the last two are the
        thickness-cap override computed below (None, None if Stage B's
        cavity remesh alone already cleared every bad cell), returned so
        the caller (mesh_background.generate_hybrid_mesh) can reuse it for
        the retry call instead of recomputing the identical, non-free
        (dijkstra-based, see compute_bl_thickness_limit_override) result a
        second time with the same arguments.
    """
    from .mesh_repair import remesh_core_cavity, compute_bl_thickness_limit_override
    from .face_extractor import FaceExtractor
    from ..schema.grid_nodes import NodeArray

    repair_actions = []

    if not np.any(bad_mask):
        return merged_nodes, merged_cells, cell_groups, bad_mask, repair_actions, None, None

    # Stage B': Local cavity remesh
    max_b_prime_attempts = 3
    b_prime_attempt_count = 0

    while np.any(bad_mask) and b_prime_attempt_count < max_b_prime_attempts:
        merged_nodes, merged_cells, cell_groups, bad_mask, cavity_actions = remesh_core_cavity(
            merged_nodes, merged_cells, cell_groups, n_bl_cells, pre_repair_faces, bad_mask, validator,
        )
        repair_actions.extend(cavity_actions)

        b_prime_attempt_count += 1
        if np.any(bad_mask):
            logger.warning(f"Stage B' attempt {b_prime_attempt_count}/{max_b_prime_attempts} completed, "
                           f"{int(np.sum(bad_mask))} bad cells remain.")

        # remesh_core_cavity splices new cells in place of the removed
        # cavity ones - merged_cells' own size/content just changed
        # (possibly even at the SAME length: a cavity can be replaced
        # 1-for-1 by a different tiling, so a length check alone isn't a
        # reliable "did it actually change" test), but pre_repair_faces
        # (its owner/neighbour cell indices) still reflects whatever
        # topology was current BEFORE this call. remesh_core_cavity's own
        # use of `faces` (both to find which cells touch a physical
        # boundary and to walk interior owner/neighbour pairs) requires it
        # to index validly into the CURRENT `cells` array - reusing the
        # stale one here was a real, not theoretical, crash: confirmed
        # directly - a second cavity-remesh iteration's owner index landed
        # one past the (by-then-different) cell count, an IndexError
        # inside remesh_core_cavity itself (touches_physical_boundary[...]
        # = True), aborting the whole repair pipeline before it could
        # report or export anything. Recomputed unconditionally every
        # iteration (not just when a change is detected) since it must
        # also be current for the compute_bl_thickness_limit_override call
        # below the loop, and getting the "did anything change" check
        # wrong is a worse failure mode than one redundant extraction.
        node_arr = NodeArray(
            x=merged_nodes[:, 0].copy(), y=merged_nodes[:, 1].copy(), z=merged_nodes[:, 2].copy()
        )
        pre_repair_faces = FaceExtractor.extract_faces(merged_cells.astype(np.int32), node_arr)

    if np.any(bad_mask):
        logger.warning(f"Stage B' reached max attempts ({max_b_prime_attempts}), "
                       f"{int(np.sum(bad_mask))} bad cells remain.")

    # Stage B: BL thickness capping retry
    extra_limit, bl_verts = None, None
    if np.any(bad_mask):
        n_bad = int(np.sum(bad_mask))
        cap_thickness = min_cell_size * 3.0
        extra_limit, bl_verts = compute_bl_thickness_limit_override(
            bad_mask, n_bl_cells, merged_cells, len(surface_nodes), cap_thickness,
            nodes_per_layer=len(bl_source_vertex), node_original_vertex=bl_source_vertex,
            local_surface_faces=bl_extrude_faces,
        )

        if extra_limit is not None:
            logger.warning(
                f"Stage A/B' left {n_bad} cells still bad ({len(bl_verts)} BL vertices "
                f"implicated) - triggering Stage B: targeted local BL thickness cap."
            )
            # Note: The actual retry logic (calling generate_hybrid_mesh again)
            # is handled by the orchestrator in mesh_background.py, using the
            # extra_limit/bl_verts returned below.
            repair_actions.append(f"Stage B: computed thickness limit for {len(bl_verts)} vertices")

    return merged_nodes, merged_cells, cell_groups, bad_mask, repair_actions, extra_limit, bl_verts
