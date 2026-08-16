"""棱柱->四面体的降级拆分 (从 mesh_repair_nonmanifold_mixed.py 拆分)。

从 mesh_repair_nonmanifold_mixed.py 拆出来（该文件原有 440 行，超过
400 行硬性拆分阈值）：`_split_prisms_to_tets` + `demote_invalid_prisms_to_tets`
只互相依赖、不依赖同文件里 `patch_nonmanifold_cavity_mixed` 的任何状态，
是清晰的拆分边界（`patch_nonmanifold_cavity_mixed` 反过来从这里导入
`_split_prisms_to_tets`，因为它也要用同一套拆分规则）。纯代码搬移，
不改变任何行为。
"""

from typing import Tuple

import numpy as np
from loguru import logger


# A prism (v0,v1,v2,w0,w1,w2) splits into exactly these 3 tets - same
# diagonal-consistency rule convert_layers_to_tetrahedra uses, so a
# prism's boundary faces here are bit-identical to what that function
# would have produced for the same slab, and therefore automatically
# conformal with whatever un-split neighbour (prism or tet) still borders
# this patch - PROVIDED v0<v1<v2 by global node index (the bottom
# triangle's own vertices, sorted, with the SAME row permutation carried
# over to the top triangle so w_i stays "above" v_i - convert_layers_to_
# prisms' own convention when it FIRST builds a prism).
#
# That precondition does NOT survive downstream node remapping: mesh_
# background.generate_hybrid_mesh calls _dedupe_coincident_points (seam
# merge, final defensive pass) multiple times after prisms are built,
# each of which can reassign a node's GLOBAL index to an arbitrary
# representative of its coincident-point group - nothing about that
# remap preserves "v0's NEW index < v1's NEW index < v2's NEW index" just
# because it held for the OLD indices. Confirmed directly, not
# theoretical: calling this function on real post-remap prisms without
# re-sorting produced ~23,000 phantom "non-manifold" face groups in an
# ad-hoc diagnostic script - face_extractor.repair_nonmanifold_mixed's
# own _build_prism_face_occurrences (which DOES re-sort every call, see
# its own docstring) found zero on the exact same mesh, proving the
# ~23,000 was entirely an artifact of this function's missing sort, not a
# real defect. Sorting here unconditionally (cheap, always correct
# whether or not the caller's input happens to already be sorted) is
# both the fix and the safe default going forward.
def _split_prisms_to_tets(prisms: np.ndarray) -> np.ndarray:
    bottom = prisms[:, 0:3]
    top = prisms[:, 3:6]
    order = np.argsort(bottom, axis=1)
    row_idx = np.arange(len(prisms))[:, None]
    sb = bottom[row_idx, order]
    st = top[row_idx, order]
    v0, v1, v2 = sb[:, 0], sb[:, 1], sb[:, 2]
    w0, w1, w2 = st[:, 0], st[:, 1], st[:, 2]
    return np.concatenate([
        np.stack([v0, v1, v2, w2], axis=1),
        np.stack([v0, v1, w1, w2], axis=1),
        np.stack([v0, w0, w1, w2], axis=1),
    ], axis=0)


def demote_invalid_prisms_to_tets(
    prism_cells: np.ndarray,
    bl_cell_groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Guarantee no exported CPENTA references the same node twice.

    A "collapsed-corner" prism (growth frozen at exactly one base vertex,
    v_i == w_i - see quality_metrics.compute_prism_aspect_ratios' own
    docstring) is a valid nonzero-volume cell by this project's own
    tolerance, but as a CPENTA record it repeats one GRID id in two of its
    6 slots - a malformed element by Nastran's own definition, not merely
    a quality issue. Confirmed directly against a real cube_demo export:
    ANSA 21.0.1 rejected ~21,000 such CPENTA records ("invalid node
    combination"), one per collapsed-corner prism, which is what actually
    produced the reported "empty" patches in the imported mesh - not the
    small tet-volume deficit this project chased earlier (see ProjectFiles
    Part10 P39).

    patch_nonmanifold_cavity_mixed (the aspect-ratio repair pass in
    mesh_background.py already routes every one of these through it, via
    prism_ar <= 500.0) is a best-effort tetgen retile and silently leaves
    the cavity untouched whenever no cluster is `accepted` - and tetgen
    reliably fails or is skipped on cavities built from near-zero-volume
    geometry, which is exactly what a collapsed-corner prism's boundary
    is. Confirmed directly: on that same real export, 100% of the
    ~21,000 flagged prisms were still present, unpatched, with the
    original duplicate node id, despite the AR-based patch call having
    run. This function is the deterministic fallback with no failure
    mode: a collapsed prism splits into exactly 3 tets via the same
    diagonal-consistent rule used everywhere else in this module
    (_split_prisms_to_tets), of which exactly the one referencing the
    repeated node twice is degenerate and dropped; the other 2 are
    ordinary, valid, non-degenerate tets covering the same volume the
    prism did - pure arithmetic, cannot fail the way a tetgen call can.

    Args:
        prism_cells: (n_prism, 6) prism connectivity
        bl_cell_groups: (n_prism,) str array parallel to prism_cells -
            each demoted prism's group name is carried onto its surviving
            tets directly (as `cell_groups`/`direct_cell_groups`), so the
            wall boundary group the prism used to belong to is not lost.

    Returns:
        (new_prism_cells, new_bl_cell_groups, extra_tets, extra_tet_groups)
        - extra_tets/extra_tet_groups are empty arrays (not None) when
        nothing needed demoting, so the caller can always np.vstack/
        np.concatenate them onto merged_cells/cell_groups unconditionally.
    """
    empty_tets = np.empty((0, 4), dtype=prism_cells.dtype)
    empty_groups = np.empty((0,), dtype=object)
    if len(prism_cells) == 0:
        return prism_cells, bl_cell_groups, empty_tets, empty_groups

    has_dup = np.zeros(len(prism_cells), dtype=bool)
    for i in range(6):
        for j in range(i + 1, 6):
            has_dup |= prism_cells[:, i] == prism_cells[:, j]

    if not has_dup.any():
        return prism_cells, bl_cell_groups, empty_tets, empty_groups

    bad_idx = np.flatnonzero(has_dup)
    split_tets = _split_prisms_to_tets(prism_cells[bad_idx])  # (3*n_bad, 4), block layout: all T1s, then T2s, then T3s
    degenerate = (
        (split_tets[:, 0] == split_tets[:, 1]) | (split_tets[:, 0] == split_tets[:, 2]) |
        (split_tets[:, 0] == split_tets[:, 3]) | (split_tets[:, 1] == split_tets[:, 2]) |
        (split_tets[:, 1] == split_tets[:, 3]) | (split_tets[:, 2] == split_tets[:, 3])
    )
    valid_tets = split_tets[~degenerate]
    # np.tile (not np.repeat) matches _split_prisms_to_tets' block layout -
    # row r of the (3*n_bad,4) output belongs to source prism bad_idx[r % n_bad].
    source_idx = np.tile(bad_idx, 3)[~degenerate]

    logger.warning(
        f"{len(bad_idx)} prism(s) with a duplicate node id among their own 6 "
        f"vertices (collapsed-corner, invalid as a CPENTA record) - demoting "
        f"to {len(valid_tets)} plain tet(s), the deterministic fallback for "
        f"whatever the tetgen-based aspect-ratio patch above did not resolve"
    )

    keep_mask = ~has_dup
    return (
        prism_cells[keep_mask],
        bl_cell_groups[keep_mask],
        valid_tets.astype(prism_cells.dtype),
        bl_cell_groups[source_idx],
    )
