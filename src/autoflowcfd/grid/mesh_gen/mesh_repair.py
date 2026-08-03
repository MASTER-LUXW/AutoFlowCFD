"""Stage A post-generation volume mesh repair: quality-gated smoothing.

Stage A (this module's main entry point, `smooth_bad_cells`): quality-gated
Laplacian smoothing, restricted to nodes that don't lie on any physical
boundary surface (body/tunnel/inlet/outlet) or on the BL/core interface -
covers BL-interior layer nodes and core-region tetgen Steiner points, since
neither carries physical geometry meaning and neither is load-bearing for
already-generated neighbouring geometry the way the interface is (see
compute_movable_node_mask's own docstring).

Stage B (BL-side thickness capping) and Stage B' (local cavity re-tiling)
live in mesh_repair_cavity.py, split out purely to keep this file's size
down - re-exported from here (see the bottom of this file) so existing
callers (`from .mesh_repair import smooth_bad_cells,
compute_bl_thickness_limit_override, remesh_core_cavity`) keep working
unchanged. See mesh_repair_cavity.py's own module docstring for the full
Stage B/B' picture, including why a previous core-side region-based
counterpart to Stage B was removed (net-harmful in practice - tetgen's
per-region refinement leaking outward, see mesh_background.py's own
history), and why Stage B' now covers BL cells too, not just core-only
ones.
"""

from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from ..schema.grid_faces import FaceData
    from ..validation.quality_validator import MeshQualityValidator


def compute_movable_node_mask(
    n_nodes: int, faces: 'FaceData', n_bl_cells: Optional[int] = None,
) -> np.ndarray:
    """Nodes safe for Stage A to relocate: everything NOT part of any
    boundary face, AND (when `n_bl_cells` is given) not on the BL/core
    interface either.

    The interface was originally treated as freely movable too, on the
    reasoning that it's "an internal mesh seam once merged, not a
    physical boundary" - true, but incomplete: unlike an ordinary
    interior node, an interface node is load-bearing for TWO already-
    finalized pieces of geometry that were built independently and never
    reconciled with each other - the BL side's own extrusion (mesh_
    front_collision.py's reactive checks already guarantee that side is
    self-consistent) AND tetgen's core fill, which received the
    interface's position as a FIXED PLC boundary constraint and
    triangulated the entire core volume against it. Moving an interface
    node during smoothing improves the BL cell(s) touching it but leaves
    the core tets on the other side still built against the node's OLD
    position - confirmed directly as a real, reproducible defect on
    cube_demo: BL cells and core cells with no shared node at all
    (genuinely non-adjacent) ending up spatially overlapping, every
    single example traced back to exactly this mismatch. Excluding these
    nodes is the conservative fix - some BL cells whose bad shape
    stems from an interface node's position may go unsmoothed, but a
    CRITICAL-severity overlap is worse than a HIGH-severity skewness/
    orthogonality warning Stage B'/C can still address without touching
    the interface.

    Args:
        n_nodes: total node count
        faces: FaceData for the current (nodes, cells) geometry
        n_bl_cells: Optional - if given, cell indices [0, n_bl_cells) are
            treated as BL-origin and the rest as core-origin (matches
            mesh_background._build_merged_mesh's own convention: BL cells
            first, core cells appended after). None (default) skips the
            interface exclusion entirely - only safe for a caller that
            has no BL region at all.
    """
    if faces.node_connectivity is None:
        raise ValueError(
            "faces.node_connectivity is required (see FaceExtractor.extract_faces) "
            "to determine which nodes lie on a physical boundary"
        )
    boundary_face_idx = faces.get_boundary_face_indices()
    boundary_nodes = np.unique(faces.node_connectivity[boundary_face_idx].ravel())
    movable = np.ones(n_nodes, dtype=bool)
    movable[boundary_nodes] = False

    if n_bl_cells is not None:
        owner = faces.connectivity[:, 0]
        neighbor = faces.connectivity[:, 1]
        interior = neighbor >= 0
        crosses_interface = interior & ((owner < n_bl_cells) != (neighbor < n_bl_cells))
        interface_face_idx = np.flatnonzero(crosses_interface)
        if len(interface_face_idx):
            interface_nodes = np.unique(faces.node_connectivity[interface_face_idx].ravel())
            movable[interface_nodes] = False

    return movable


def _bad_cell_mask(
    validator: 'MeshQualityValidator',
    nodes: np.ndarray,
    cells: np.ndarray,
    faces: 'FaceData',
    extra_bad_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Which cells trip any of the skewness/orthogonality/adjacent-volume-
    ratio thresholds - the same three checks MeshQualityValidator gates on,
    evaluated per-cell instead of aggregated.

    Args:
        extra_bad_mask: Optional (n_cells,) bool array OR'd into the
            result - e.g. cells implicated in a physical overlap with a
            different, non-adjacent cell (mesh_overlap_check.py). Overlap
            is a static, one-time-computed fact about the mesh BEFORE this
            pass started (recomputing the broad-phase spatial search every
            smoothing pass would be needlessly expensive - see
            smooth_bad_cells' own doc); folding it in here means a cell an
            earlier overlap check flagged keeps being treated as a
            legitimate smoothing candidate for as long as Stage A runs,
            and - if still bad in the mask smooth_bad_cells returns - is
            then a legitimate Stage B' cavity-remesh candidate too, same as
            any other bad cell.
    """
    n_cells = len(cells)
    bad = np.zeros(n_cells, dtype=bool)

    skew = validator.compute_cell_skewness(nodes, cells)
    bad |= skew > validator.thresholds['max_skewness']

    diag = validator.compute_face_diagnostics(nodes, cells, faces)
    if len(diag['angle_deg']) > 0:
        face_bad = (
            (diag['angle_deg'] > validator.thresholds['max_orthogonality_angle'])
            | (diag['volume_ratio'] > validator.thresholds['max_adjacent_volume_ratio'])
        )
        bad[diag['owner'][face_bad]] = True
        bad[diag['neighbor'][face_bad]] = True

    if extra_bad_mask is not None:
        bad |= extra_bad_mask

    return bad


def _node_target_positions(nodes: np.ndarray, cells: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted average of each node's incident cell centroids (the classic
    "smart Laplacian" smoothing target) - vectorized scatter-sum, same
    pattern as mesh_extrusion.py's per-node normal averaging."""
    centroids = nodes[cells].mean(axis=1)
    n_nodes = len(nodes)
    node_sum = np.zeros((n_nodes, 3))
    node_weight = np.zeros(n_nodes)
    flat_nodes = cells.ravel()
    flat_centroids = np.repeat(centroids, cells.shape[1], axis=0)
    flat_weights = np.repeat(np.maximum(weights, 1e-300), cells.shape[1])
    np.add.at(node_sum, flat_nodes, flat_centroids * flat_weights[:, None])
    np.add.at(node_weight, flat_nodes, flat_weights)

    target = nodes.copy()
    has_weight = node_weight > 0
    target[has_weight] = node_sum[has_weight] / node_weight[has_weight, None]
    return target


def smooth_bad_cells(
    nodes: np.ndarray,
    cells: np.ndarray,
    validator: 'MeshQualityValidator',
    max_passes: int = 5,
    initial_faces: Optional['FaceData'] = None,
    extra_bad_mask: Optional[np.ndarray] = None,
    n_bl_cells: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Stage A: quality-gated Laplacian smoothing of skewed/non-orthogonal/
    volume-mismatched cells, restricted to movable (non-boundary) nodes.

    Each pass: identify bad cells -> collect their movable nodes -> propose
    the volume-weighted-centroid target position for each -> apply all
    proposed moves simultaneously -> if that introduces any negative-volume
    cell, halve the displacement and retry (up to 4 halvings) rather than
    discarding the whole pass - a standard "relaxation + line-search
    damping" simplification of smart Laplacian smoothing: cheaper than a
    strict per-node accept/reject scheme (which would need re-validating
    the whole affected neighbourhood after every single node's move) while
    keeping its core safety guarantee - a pass is only ever committed if it
    introduces zero negative-volume cells.

    Args:
        nodes: (n_nodes, 3) float64, not mutated - a new array is returned
        cells: (n_cells, 4) int32 tetrahedron connectivity
        validator: MeshQualityValidator instance (reused for its threshold
            config and per-cell/per-face diagnostic methods)
        max_passes: stop after this many passes regardless of outcome
        initial_faces: Optional FaceData already extracted from this exact
            (nodes, cells) pair (e.g. by a caller that just ran its own
            pre-repair validate() on the same unmoved geometry) - reused
            for pass 0 only, instead of re-extracting from scratch, since
            pass 0 evaluates the mesh before any node has moved. A real
            saving on large meshes: face extraction is a non-trivial cost,
            and the common case is pass 0 finding no safe move at all
            (nothing to smooth), where this is the only extraction Stage A
            would otherwise redundantly duplicate against the caller's own
            pre-repair check.
        extra_bad_mask: Optional (n_cells,) bool array of additional cells
            to always treat as bad regardless of what the skew/orthogonality/
            volume-ratio checks say this pass - see _bad_cell_mask's own
            doc (e.g. cells flagged by mesh_overlap_check.py). Evaluated
            against the ORIGINAL cell indexing throughout - safe because
            this function never adds/removes cells, only moves node
            positions.
        n_bl_cells: Optional - cell indices [0, n_bl_cells) are BL-origin,
            the rest core-origin (see compute_movable_node_mask's own
            docstring for why this excludes the BL/core interface from
            smoothing). None (default) leaves the interface movable -
            only correct for a caller with no BL region at all.

    Returns:
        (new_nodes, bad_cell_mask_after, action_log) - bad_cell_mask_after
        is evaluated fresh on the final geometry (empty array in, if no
        passes ran because the mesh started with no bad cells at all).
    """
    from ..schema.grid_nodes import NodeArray
    from .face_extractor import FaceExtractor

    nodes = nodes.copy()
    actions: List[str] = []
    bad_mask = np.zeros(len(cells), dtype=bool)

    for pass_idx in range(max_passes):
        if pass_idx == 0 and initial_faces is not None:
            faces = initial_faces
        else:
            node_arr = NodeArray(
                x=np.ascontiguousarray(nodes[:, 0]),
                y=np.ascontiguousarray(nodes[:, 1]),
                z=np.ascontiguousarray(nodes[:, 2]),
            )
            faces = FaceExtractor.extract_faces(cells.astype(np.int32), node_arr)
        movable_mask = compute_movable_node_mask(len(nodes), faces, n_bl_cells)

        bad_mask = _bad_cell_mask(validator, nodes, cells, faces, extra_bad_mask=extra_bad_mask)
        if not np.any(bad_mask):
            if pass_idx == 0:
                actions.append("Stage A: mesh already within thresholds, no smoothing needed")
            break

        candidate_mask = np.zeros(len(nodes), dtype=bool)
        candidate_mask[cells[bad_mask].ravel()] = True
        candidate_mask &= movable_mask

        n_bad = int(np.sum(bad_mask))
        if not np.any(candidate_mask):
            actions.append(
                f"Stage A pass {pass_idx + 1}: {n_bad} bad cells remain, but none of "
                f"their nodes are movable (all on a physical boundary) - stopping"
            )
            break

        current_volumes = validator._compute_tetrahedron_volumes(nodes, cells)
        target = _node_target_positions(nodes, cells, np.abs(current_volumes))

        # Safety criterion is "never turn a currently-valid cell negative",
        # not "guarantee zero negative cells mesh-wide" - the latter can
        # never be satisfied if something upstream of this pass (e.g. an
        # already-degenerate cell elsewhere untouched by this move) put the
        # mesh in that state first, which would then block Stage A from
        # fixing anything at all, including cells it *can* legitimately
        # improve.
        already_bad = current_volumes <= 0

        relax = 1.0
        accepted = False
        nodes_trial = nodes
        for _damp_iter in range(4):
            nodes_trial = nodes.copy()
            nodes_trial[candidate_mask] = (
                nodes[candidate_mask] + relax * (target[candidate_mask] - nodes[candidate_mask])
            )
            trial_volumes = validator._compute_tetrahedron_volumes(nodes_trial, cells)
            newly_negative = (trial_volumes <= 0) & ~already_bad
            if not np.any(newly_negative):
                accepted = True
                break
            relax *= 0.5

        if not accepted:
            actions.append(
                f"Stage A pass {pass_idx + 1}: no safe move found for {int(np.sum(candidate_mask))} "
                f"candidate nodes even after damping - stopping"
            )
            break

        n_moved = int(np.sum(candidate_mask))
        nodes = nodes_trial
        actions.append(
            f"Stage A pass {pass_idx + 1}: {n_bad} bad cells -> moved {n_moved} nodes "
            f"(relax={relax:.3f})"
        )

    else:
        actions.append(f"Stage A: reached max_passes={max_passes} limit")

    return nodes, bad_mask, actions


# Stage B / B' - see mesh_repair_bl_thickness.py / mesh_repair_cavity.py.
# Re-exported here so `from .mesh_repair import ...` keeps working for
# existing callers (mesh_background.py, tests/unit/test_mesh_repair.py)
# unchanged.
from .mesh_repair_bl_thickness import compute_bl_thickness_limit_override  # noqa: E402
from .mesh_repair_cavity import remesh_core_cavity  # noqa: E402
