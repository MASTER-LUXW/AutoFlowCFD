"""Volume-mesh cell overlap / near-touching-face detection.

Detects the one class of mesh defect none of this project's existing
checks target directly: two DIFFERENT, non-topologically-adjacent cells
whose faces physically overlap in 3D space, or sit close enough together
that they are one small parameter change away from overlapping (e.g. two
BL extrusion fronts approaching each other across a tight gap - see
mesh_tetgen_core.compute_local_thickness_limit, which only *prevents* this
during generation and is explicitly documented there as a heuristic, not a
guarantee).

What this complements, and does not replace:
    - repair_nonmanifold_cells (mesh_tetgen_core.py) detects a *symptom*
      of overlap (a face shared by >2 cells) after the fact - it can't see
      an overlap that doesn't happen to produce that specific topological
      signature (e.g. two cells that interpenetrate without any face being
      literally coincident).
    - fill_core_volume's tetgen self-intersection error only checks the BL
      outer surface (a single 2D shell) for self-intersection before the
      core is even filled - it says nothing about the final 3D volume mesh.
    - This module checks the actual, final set of cells directly, via exact
      triangle-triangle intersection/distance tests (overlap_geometry.py),
      not a proxy signal.

Two faces are only compared if they share ZERO nodes - faces sharing a
node (an edge, a single vertex, or being the same face from opposite
sides) are ordinary, correct mesh topology, not a defect, and are excluded
before any geometric test runs.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from loguru import logger

from .overlap_geometry import triangle_triangle_intersect, triangle_triangle_min_distance

if TYPE_CHECKING:
    from ..schema.grid_faces import FaceData

# A single outlier-huge boundary face (e.g. a coarse farfield/domain-shell
# panel many times larger than the typical boundary face - see
# check_face_overlap_and_proximity's own search_radius docstring for why
# radius scales with each face's OWN size) gets a broad-phase search
# radius scaled to that huge size, and its query_ball_point call can then
# return a candidate list numbering in the hundreds of thousands - purely
# because of its own size, not genuine proximity risk. Measured on a real
# case (cube_demo's coarse domain-shell/farfield panels): a single such
# face returned 142,944 candidates, blowing its containing 500-face chunk
# out to 5.58M candidate pairs and making the whole check take 6+ minutes
# and several GB of memory. A genuine near-miss or overlap always shows up
# among the CLOSEST few candidates - the defect threshold
# (proximity_fraction * min(size_i, size_j)) is far tighter than
# search_radius (search_multiplier's own docstring explains the required
# headroom between them) - so capping any one face's candidate set at its
# CAP nearest neighbours (instead of every point within its oversized
# radius) keeps every genuine candidate while dropping only the excess,
# far-within-radius-but-nowhere-near-the-actual-threshold ones that were
# never going to be flagged anyway.
CANDIDATE_CAP_PER_FACE = 2000


@dataclass
class OverlapProximityReport:
    """Result of check_face_overlap_and_proximity.

    Attributes:
        n_faces_checked: Total distinct triangular faces considered
        n_candidate_pairs: Face pairs that survived broad-phase + node-
            sharing filtering and were actually given the exact geometric
            test - informational, for judging how selective the broad
            phase was
        n_overlapping_pairs: Face pairs that genuinely intersect
        n_close_pairs: Face pairs that don't intersect but are closer than
            `proximity_threshold_used` (a per-pair, locally-scaled
            distance, not a single global constant - see the check
            function's own docstring)
        overlapping_cell_ids: Unique cell indices that own at least one
            face involved in a genuine overlap
        close_cell_ids: Unique cell indices that own at least one face
            involved in a near-touching (but not yet overlapping) pair
        min_gap_found: Smallest non-overlapping face-to-face distance
            found among all close pairs (None if none found)
        overlap_examples: Up to `max_examples` (cell_a, cell_b) pairs, for
            a human-readable report - not exhaustive on a mesh with many
            overlaps
        close_examples: Up to `max_examples` (cell_a, cell_b, distance)
            tuples
        elapsed_seconds: Wall-clock time this check took - included
            because, unlike this project's other O(n) vectorized quality
            checks, this one's cost scales with local mesh density, not
            purely cell count (see module docstring)
    """
    n_faces_checked: int = 0
    n_candidate_pairs: int = 0
    n_overlapping_pairs: int = 0
    n_close_pairs: int = 0
    overlapping_cell_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    close_cell_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    min_gap_found: Optional[float] = None
    overlap_examples: List[Tuple[int, int]] = field(default_factory=list)
    close_examples: List[Tuple[int, int, float]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def has_overlaps(self) -> bool:
        return self.n_overlapping_pairs > 0

    def bad_cell_mask(self, n_cells: int) -> np.ndarray:
        """Boolean mask, True for any cell implicated in a genuine overlap
        (NOT close-but-not-overlapping - see mesh_repair.py's use of this:
        only an actual overlap is bad enough to warrant repair action;
        "close" is reported for visibility but is not itself a defect)."""
        mask = np.zeros(n_cells, dtype=bool)
        if len(self.overlapping_cell_ids):
            mask[self.overlapping_cell_ids] = True
        return mask

    def summary(self) -> str:
        lines = [
            "Cell Overlap / Proximity Check:",
            f"  Faces checked: {self.n_faces_checked:,}  "
            f"Candidate pairs tested: {self.n_candidate_pairs:,}  "
            f"({self.elapsed_seconds:.2f}s)",
            f"  Overlapping face pairs: {self.n_overlapping_pairs} "
            f"({len(self.overlapping_cell_ids)} cells implicated)",
            f"  Near-touching (not yet overlapping) face pairs: {self.n_close_pairs} "
            f"({len(self.close_cell_ids)} cells implicated)"
            + (f", min gap {self.min_gap_found:.3e} m" if self.min_gap_found is not None else ""),
        ]
        if self.overlap_examples:
            preview = ", ".join(f"({a},{b})" for a, b in self.overlap_examples[:10])
            more = f" (+{len(self.overlap_examples) - 10} more)" if len(self.overlap_examples) > 10 else ""
            lines.append(f"  Overlapping cell pairs (owner ids): {preview}{more}")
        if self.close_examples:
            preview = ", ".join(f"({a},{b}, {d:.2e}m)" for a, b, d in self.close_examples[:10])
            more = f" (+{len(self.close_examples) - 10} more)" if len(self.close_examples) > 10 else ""
            lines.append(f"  Near-touching cell pairs: {preview}{more}")
        return "\n".join(lines)


def _extract_faces(nodes: np.ndarray, cells: np.ndarray) -> 'FaceData':
    # Lazy-imported: mesh_gen -> validation is a one-way dependency
    # elsewhere in this package (see quality_validator.py's identical
    # _extract_faces) - importing the other direction only at call time
    # avoids ever needing to reason about import order.
    from ..mesh_gen.face_extractor import FaceExtractor
    from ..schema.grid_nodes import NodeArray

    node_arr = NodeArray(
        x=np.ascontiguousarray(nodes[:, 0]),
        y=np.ascontiguousarray(nodes[:, 1]),
        z=np.ascontiguousarray(nodes[:, 2]),
    )
    return FaceExtractor.extract_faces(cells.astype(np.int32), node_arr)


def check_face_overlap_and_proximity(
    nodes: np.ndarray,
    cells: np.ndarray,
    faces: Optional['FaceData'] = None,
    proximity_fraction: float = 0.1,
    search_multiplier: float = 3.0,
    max_examples: int = 20,
    chunk_size: int = 500,
    boundary_faces_only: bool = True,
) -> OverlapProximityReport:
    """Detect genuinely overlapping and near-touching faces between
    different, non-adjacent cells.

    Uses FaceData's already-deduplicated face list (one entry per distinct
    triangular face, whether interior or boundary - see grid_faces.py) as
    the candidate set, rather than re-deriving all 4*n_cells raw tet faces:
    two cells that legitimately share a face already collapse to a single
    FaceData entry with both an owner and a neighbour, so this never has to
    separately special-case "the two faces are actually the same face".

    By default (`boundary_faces_only=True`) only the mesh's true boundary
    faces (no neighbour cell) are candidates. This is a deliberate scope
    restriction, not just a performance shortcut: the defects this check
    exists for (see module docstring - two BL extrusion fronts crossing, a
    core-fill splice artifact) both manifest at the OUTER surface of the
    respective regions colliding, never buried inside an already-correctly-
    formed BL layer stack's interior. Checking interior faces too was tried
    first and found to be both wrong and impractically expensive on a real
    automotive mesh: a BL stack's own consecutive layers are, BY DESIGN,
    packed far closer together (first-layer thickness can be a few mm) than
    a face's own lateral size, so `proximity_fraction`-scaled "closeness"
    flagged nearly every ordinary layer-to-layer transition in the entire
    BL volume as a false positive - millions of them on a 2.4M-cell case,
    which is also what made the check itself slow/memory-heavy for no
    diagnostic benefit. Boundary faces alone were ~36x fewer on that same
    case (135,914 of 4,886,259 total) and are exactly where a genuine
    cross-region collision would actually show up.

    Broad phase: for each face, query a KD-tree of face centroids within
    `search_multiplier * sqrt(own_area)` of that face's own centroid - a
    LOCAL, per-face radius (not a single domain-scale constant). This
    matters on a BL-extruded automotive mesh where near-wall cells can be
    orders of magnitude smaller than far-field core cells (see
    mesh_tetgen_core.compute_local_thickness_limit's own doc for the exact
    same lesson, learned from a measured multi-minute regression when an
    earlier version of that function used a domain-scale radius
    unconditionally). Querying from every face (not just small ones) with
    its OWN radius, then taking the union of all pairs found, naturally
    also catches a small face near a much larger one even though the small
    face's own radius alone wouldn't reach the large face's centroid - the
    large face's own (larger) query catches it from the other direction.
    A rare outlier-huge face's own candidate list is capped at
    CANDIDATE_CAP_PER_FACE nearest neighbours rather than left unbounded -
    see that constant's module-level docstring for why this doesn't lose
    any genuine defect.

    Narrow phase, per surviving candidate pair (excludes any pair sharing a
    node - see module docstring): exact triangle_triangle_intersect first;
    if not intersecting, triangle_triangle_min_distance, flagged as "close"
    if below `proximity_fraction * min(sqrt(area_i), sqrt(area_j))` (a
    per-pair, locally-scaled threshold, not a single global distance).

    Args:
        nodes: (n_nodes, 3) float64 node coordinates
        cells: (n_cells, 4) int32/int64 tetrahedral connectivity
        faces: Optional precomputed FaceData - reused if the caller already
            has it (this project's mesh generation/repair pipeline commonly
            does), else derived internally
        proximity_fraction: "close" threshold as a fraction of the smaller
            of the two candidate faces' own characteristic size
        search_multiplier: broad-phase KD-tree query radius as a multiple
            of each face's own characteristic size - larger catches more
            candidates (safer, slower); must exceed proximity_fraction for
            the close-pair threshold to ever be reachable, and in practice
            needs headroom beyond that since two faces can be centroid-
            further-apart than their close-distance while their nearest
            EDGES are still within range
        max_examples: cap on how many concrete (cell, cell[, distance])
            examples are kept for the human-readable report - counts are
            never capped, only the example list
        chunk_size: KD-tree queries AND candidate-pair deduplication are
            batched this many faces at a time (same rationale as
            mesh_tetgen_core.compute_local_thickness_limit's chunking -
            materializing every face's full candidate list, or every
            candidate pair ever seen, at once does not scale on a fine
            mesh; a global cross-chunk `set()` of every pair seen so far
            was tried first and grew into tens of GB on a real 4.9M-face
            mesh). Deduplication is per-chunk only (vectorized via
            np.unique, not a Python-level set) - a genuine defect pair can
            therefore be found and geometrically tested twice if it's
            reachable from two different chunks, a bounded 2x compute cost
            accepted in exchange for bounded (not unbounded) memory; the
            final overlap/close pair lists are deduplicated again at the
            end regardless (cheap, bounded by the number of actual
            findings, not by candidate-pair count)
        boundary_faces_only: see above - False checks every face
            (interior + boundary), which is both slower and noisier on a
            BL-extruded mesh; only meaningful for a caller that has
            verified its own mesh has no BL region at all (e.g. a bare
            tetgen background fill), where "interior" carries none of the
            BL-stacking density this default is scoped around.

    Returns:
        OverlapProximityReport
    """
    from scipy.spatial import cKDTree

    start = time.perf_counter()

    if faces is None:
        faces = _extract_faces(nodes, cells)

    if faces.node_connectivity is None:
        raise ValueError(
            "faces.node_connectivity is required (see FaceExtractor.extract_faces) "
            "to determine which faces share a node"
        )

    owner_full = faces.connectivity[:, 0]
    neighbor_full = faces.connectivity[:, 1]

    if boundary_faces_only:
        face_idx = faces.get_boundary_face_indices().astype(np.int64)
    else:
        face_idx = np.arange(faces.count, dtype=np.int64)

    n_faces = len(face_idx)
    centroids = faces.center[face_idx]
    face_nodes = faces.node_connectivity[face_idx]
    face_size = np.sqrt(np.maximum(faces.area[face_idx], 1e-300))
    owner = owner_full[face_idx]
    neighbor = neighbor_full[face_idx]

    tree = cKDTree(centroids)
    search_radius = search_multiplier * face_size

    overlap_pairs: List[Tuple[int, int]] = []
    close_pairs: List[Tuple[int, int, float]] = []
    n_candidate_pairs = 0
    min_gap_found: Optional[float] = None
    n_chunks = (n_faces + chunk_size - 1) // chunk_size
    progress_every = max(1, n_chunks // 20)

    for chunk_num, start_idx in enumerate(range(0, n_faces, chunk_size)):
        end_idx = min(start_idx + chunk_size, n_faces)
        idx_chunk = np.arange(start_idx, end_idx)
        neighbor_lists = tree.query_ball_point(
            centroids[idx_chunk], r=search_radius[idx_chunk], workers=-1
        )

        # Fully vectorized candidate-pair construction - NOT a Python-level
        # loop over every (face, candidate) combination, and NOT an
        # ever-growing cross-chunk Python `set()` of every pair seen so
        # far. Both were tried first and, on a real multi-million-face
        # mesh (Ahmed Body, 4.9M faces), the set grew into tens of GB and
        # the loop made no visible progress for 10+ minutes before being
        # killed - the same class of unbounded-accumulation performance
        # trap this project has hit before (see this doc's own Part4,
        # P1/P2). Deduplication is instead done PER CHUNK only, via
        # np.unique on a small (chunk-local) array; a pair can still be
        # tested twice total if it's found from both directions across two
        # DIFFERENT chunks (bounded 2x extra work, not unbounded memory) -
        # a deliberate, cheap trade-off, not an oversight.
        counts = np.fromiter(
            (len(lst) for lst in neighbor_lists), dtype=np.int64, count=len(neighbor_lists)
        )

        # See CANDIDATE_CAP_PER_FACE's module-level docstring. This only
        # ever fires for rare outlier-huge faces - normal-scale faces
        # (the vast majority) have small counts and are completely
        # unaffected.
        over_cap = np.flatnonzero(counts > CANDIDATE_CAP_PER_FACE)
        if len(over_cap):
            k = min(CANDIDATE_CAP_PER_FACE, n_faces)
            for local_i in over_cap:
                face_i = int(idx_chunk[local_i])
                original_count = int(counts[local_i])
                nn_dists, nn_idx = tree.query(
                    centroids[face_i], k=k, distance_upper_bound=search_radius[face_i]
                )
                valid = np.isfinite(nn_dists)
                capped = nn_idx[valid].tolist()
                neighbor_lists[local_i] = capped
                counts[local_i] = len(capped)
                logger.warning(
                    f"Overlap check: face {face_i} had an oversized broad-phase candidate set "
                    f"({original_count}+ within radius {search_radius[face_i]:.3e}); capped to the "
                    f"{len(capped)} nearest to keep the check tractable - likely a large outlier "
                    f"face (size={face_size[face_i]:.3e}) relative to the mesh's typical boundary "
                    f"face scale."
                )

        if counts.sum() == 0:
            if chunk_num % progress_every == 0:
                logger.debug(f"Overlap check: {chunk_num}/{n_chunks} chunks, 0 candidates so far")
            continue

        row_idx = np.repeat(idx_chunk, counts)
        col_idx = np.concatenate(
            [np.asarray(lst, dtype=np.int64) for lst in neighbor_lists if len(lst) > 0]
        )
        keep_self = row_idx != col_idx
        row_idx, col_idx = row_idx[keep_self], col_idx[keep_self]
        if len(row_idx) == 0:
            continue

        pairs = np.stack([np.minimum(row_idx, col_idx), np.maximum(row_idx, col_idx)], axis=1)
        pairs = np.unique(pairs, axis=0)
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]

        # Exclude any pair sharing a node (legitimate topology, not overlap).
        shares_node = np.zeros(len(i_idx), dtype=bool)
        ni, nj = face_nodes[i_idx], face_nodes[j_idx]
        for a in range(3):
            for b in range(3):
                shares_node |= ni[:, a] == nj[:, b]
        keep = ~shares_node
        if chunk_num % progress_every == 0:
            logger.debug(
                f"Overlap check: {chunk_num}/{n_chunks} chunks, "
                f"{n_candidate_pairs:,} candidate pairs tested so far"
            )
        if not np.any(keep):
            continue

        i_idx, j_idx = i_idx[keep], j_idx[keep]
        n_candidate_pairs += len(i_idx)

        a_nodes = nodes[face_nodes[i_idx]]  # (M, 3, 3)
        b_nodes = nodes[face_nodes[j_idx]]

        intersects = triangle_triangle_intersect(
            a_nodes[:, 0], a_nodes[:, 1], a_nodes[:, 2],
            b_nodes[:, 0], b_nodes[:, 1], b_nodes[:, 2],
        )

        for k in np.flatnonzero(intersects):
            fi, fj = int(i_idx[k]), int(j_idx[k])
            cells_i = [owner[fi]] + ([neighbor[fi]] if neighbor[fi] >= 0 else [])
            cells_j = [owner[fj]] + ([neighbor[fj]] if neighbor[fj] >= 0 else [])
            for ci in cells_i:
                for cj in cells_j:
                    overlap_pairs.append((int(ci), int(cj)))

        non_intersecting = np.flatnonzero(~intersects)
        if len(non_intersecting):
            ni_idx, nj_idx = i_idx[non_intersecting], j_idx[non_intersecting]
            an, bn = a_nodes[non_intersecting], b_nodes[non_intersecting]
            dists = triangle_triangle_min_distance(
                an[:, 0], an[:, 1], an[:, 2], bn[:, 0], bn[:, 1], bn[:, 2]
            )
            threshold = proximity_fraction * np.minimum(face_size[ni_idx], face_size[nj_idx])
            close_mask = dists < threshold
            for k in np.flatnonzero(close_mask):
                fi, fj = int(ni_idx[k]), int(nj_idx[k])
                d = float(dists[k])
                min_gap_found = d if min_gap_found is None else min(min_gap_found, d)
                cells_i = [owner[fi]] + ([neighbor[fi]] if neighbor[fi] >= 0 else [])
                cells_j = [owner[fj]] + ([neighbor[fj]] if neighbor[fj] >= 0 else [])
                for ci in cells_i:
                    for cj in cells_j:
                        close_pairs.append((int(ci), int(cj), d))

    # A genuine defect pair can be found twice (once from each face's own
    # chunk - see the per-chunk-only dedup note above); collapse to unique
    # (cell_a, cell_b) entries here rather than double-reporting/double-
    # counting it. Cheap: bounded by the number of actual findings, not by
    # the (potentially huge) total candidate-pair count.
    overlap_pairs = list(dict.fromkeys(overlap_pairs))
    close_pairs = list({(a, b): (a, b, d) for a, b, d in close_pairs}.values())

    overlap_cell_ids = np.unique(np.array([p for pair in overlap_pairs for p in pair], dtype=np.int64)) \
        if overlap_pairs else np.array([], dtype=np.int64)
    close_cell_ids = np.unique(np.array([p for pair in close_pairs for p in pair[:2]], dtype=np.int64)) \
        if close_pairs else np.array([], dtype=np.int64)

    elapsed = time.perf_counter() - start
    report = OverlapProximityReport(
        n_faces_checked=n_faces,
        n_candidate_pairs=n_candidate_pairs,
        n_overlapping_pairs=len(overlap_pairs),
        n_close_pairs=len(close_pairs),
        overlapping_cell_ids=overlap_cell_ids,
        close_cell_ids=close_cell_ids,
        min_gap_found=min_gap_found,
        overlap_examples=overlap_pairs[:max_examples],
        close_examples=close_pairs[:max_examples],
        elapsed_seconds=elapsed,
    )

    if report.has_overlaps:
        logger.warning(
            f"Overlap check: {report.n_overlapping_pairs} overlapping face pair(s) "
            f"found across {len(overlap_cell_ids)} cells ({elapsed:.2f}s)"
        )
    else:
        logger.debug(
            f"Overlap check: no overlapping faces found among {n_faces} faces "
            f"({n_candidate_pairs} candidate pairs tested, {elapsed:.2f}s)"
        )

    return report
