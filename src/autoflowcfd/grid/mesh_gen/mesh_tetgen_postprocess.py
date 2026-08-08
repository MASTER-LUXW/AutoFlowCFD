"""tetgen 核心域填充：填充后的清理与修复。

从 mesh_tetgen_core.py 拆分出来，负责 fill_core_volume 产出的四面体在
拼接/导出前需要做的收尾处理：重合点合并、超大四面体细分、非流形面修复、
以及从 tetgen 自带的 facet marker 反推每个单元所属的边界分组。
"""

from typing import List, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from loguru import logger


def _dedupe_coincident_points(
    points: np.ndarray,
    faces: np.ndarray,
    tolerance: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse coincident points (within tolerance) and remap faces (or
    tetrahedral cells - `faces` is just an (n, k) index array, remapped via
    plain fancy indexing, so this works unmodified for k=3 or k=4).

    Also returns `remap` (shape=(len(points),), old index -> new index) so
    a caller holding ANY OTHER index array into the same original `points`
    (e.g. fill_core_volume's separately-read `tgen.trifaces`) can apply the
    identical remapping and stay consistent with the returned
    `unique_points`/`new_faces` - passing `remap[some_other_array]` does
    that. `remap` is the identity when no coincident points were found.

    Fully transitive (uses scipy connected_components over the coincidence
    graph, not a one-hop union), unlike the older `merge_conforming_meshes`
    node-dedup logic elsewhere in this package.

    Two call sites: the original fallback here in fill_core_volume, for
    when tetgen doesn't return a fully conformal boundary; and
    mesh_background.generate_hybrid_mesh's final defensive pass over the
    WHOLE merged mesh, for a distinct failure mode found on a real case -
    when mesh_repair.compute_bl_thickness_limit_override's reactive BL
    thickness cap needs to cap a very large fraction of surface vertices
    (itself a symptom of something upstream producing widespread, not
    localized, bad cells), many nodes' `remaining_budget` hits exactly zero
    within the same few layers, freezing them at an identical coordinate
    for every subsequent layer - each still gets its own distinct global
    node index (one new index per layer, unconditionally), so the result
    is a large number of geometrically-coincident points under different
    indices. That doesn't trip repair_nonmanifold_cells (which matches by
    exact node index, not geometry) and doesn't reliably trip the
    degenerate-volume filter either (a tet mixing a frozen node with a
    still-growing neighbour can have a small but non-negligible volume) -
    it's a silent topological tear (two geometrically-identical faces under
    different index sets, each independently counted as a normal boundary
    face) rather than a crash, so nothing upstream of this catches it.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    pairs = tree.query_pairs(tolerance)

    n_points = len(points)
    if not pairs:
        return points, faces, np.arange(n_points, dtype=np.int64)

    rows = [p[0] for p in pairs]
    cols = [p[1] for p in pairs]
    # Ensure the data array is integer type to avoid float labels from connected_components
    graph = coo_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(n_points, n_points))
    n_components, labels = connected_components(graph, directed=False)

    # Ensure labels is integer type for indexing
    labels = labels.astype(np.int64)

    # Use the smallest original index in each component as the representative.
    representative = np.full(n_components, n_points, dtype=np.int64)
    np.minimum.at(representative, labels, np.arange(n_points))

    new_index_of_label = np.arange(n_components)
    unique_points = points[representative]
    remap = new_index_of_label[labels]

    new_faces = remap[faces]
    logger.warning(
        f"Coincident-point fallback stitch: {n_points} -> {len(unique_points)} points "
        f"({n_points - len(unique_points)} merged)"
    )
    return unique_points, new_faces, remap


def _tet_volumes(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Unsigned tetrahedron volumes (orientation-independent)."""
    p0 = nodes[cells[:, 0]]
    p1 = nodes[cells[:, 1]]
    p2 = nodes[cells[:, 2]]
    p3 = nodes[cells[:, 3]]
    return np.abs(np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0


# Volume-ratio safety cap for subdivide_oversized_tetrahedra's own depth
# limit: each centroid split quarters a tet's volume EXACTLY (see that
# function's own docstring for the proof), so 8 levels is a 4**8 =
# 65,536x volume reduction - comfortably beyond the worst escaped-tet case
# measured directly on cube_demo (~16,000x the target).
_MAX_SUBDIVIDE_DEPTH = 8


def subdivide_oversized_tetrahedra(
    nodes: np.ndarray,
    tets: np.ndarray,
    max_volume: float,
    max_depth: int = _MAX_SUBDIVIDE_DEPTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recursively split every tetrahedron whose volume exceeds
    `max_volume` by inserting its own centroid as a new vertex, replacing
    it with the 4 sub-tetrahedra formed by that centroid and each of its
    4 original faces.

    Exists because tetgen's own volume-based refinement (`fill_core_volume`'s
    `regions`/`varvolume`) does not reliably reach every cell: its
    refinement queue is shape-quality-first, volume-second, and a
    perfectly well-shaped-but-oversized tet with all 4 vertices already on
    the PLC boundary (nothing to trigger further insertion nearby) can be
    left completely untouched - confirmed directly on cube_demo (a single
    14.15 m^3 tet spanning inlet-to-outlet, ~16,000x the region's own
    target, identical whether volume_cap_fraction was tightened 0.5->0.1
    or the single region seed was replaced with ~27 scattered ones - see
    mesh_background_merge.py's own history for that investigation) and,
    at larger scale, via mesh_overlap_check.py logging thousands of
    anomalously large (0.1-3 m^2) boundary faces on a real run. This
    function is a deterministic, tetgen-independent backstop that doesn't
    depend on tetgen's refinement queue choosing to cooperate.

    Centroid subdivision (rather than e.g. longest-edge bisection) was
    chosen specifically because it needs no coordination with neighbouring
    cells: for any tetrahedron (A, B, C, D) with centroid
    G = (A+B+C+D)/4, the 4 sub-tets (A,B,C,G), (A,B,D,G), (A,C,D,G),
    (B,C,D,G) each have EXACTLY 1/4 of the original volume regardless of
    the original tet's shape (provable directly: substituting u=B-A,
    v=C-A, w=D-A gives det(u, v, (u+v+w)/4) = det(u,v,w)/4, since the u
    and v components of (u+v+w)/4 drop out of the determinant - so
    Volume(A,B,C,G) = Volume(A,B,C,D)/4 exactly), and each child retains
    exactly one of the original tet's 4 faces completely unchanged. A
    neighbour sharing that face sees an untouched, still-conformal
    boundary - no hanging nodes, no need to also split the neighbour, no
    global closure/propagation pass (unlike longest-edge bisection, which
    requires exactly that to stay conformal). This also means face-based
    boundary attribution (attribute_cells_from_trifaces, matching by
    sorted node-triple against tetgen's own facet markers) keeps working
    unmodified on the result: a child that inherits a marked boundary face
    is still found by that same matching, and winding doesn't matter for
    either that matching or this function's own (unsigned) volume
    computation - any downstream orientation requirement is normalized
    later, once, over the whole merged mesh (mesh_background.py's
    orient_tetrahedra call).

    Args:
        nodes: (n, 3) float64 node coordinates (only ones actually
            referenced by `tets`; e.g. fill_core_volume's own return
            value, not a shared array with other unrelated cells - new
            centroid vertices are appended past the end, so any OTHER
            index array into the same original `nodes` array stays valid,
            but nothing referencing indices >= len(nodes) existed before
            this call to begin with)
        tets: (m, 4) int64 tetrahedral connectivity, indices into `nodes`
        max_volume: split threshold in the same volume units as `nodes`'
            coordinates cubed (e.g. m^3)
        max_depth: safety cap on recursive splits per originally-oversized
            tet - a tet still over `max_volume` after this many levels is
            left as-is (logged) rather than split indefinitely

    Returns:
        (new_nodes, new_tets): new_nodes is `nodes` plus one appended row
        per centroid inserted; new_tets has the same total volume as
        `tets` (centroid subdivision is an exact partition, not an
        approximation) but is NOT length-preserving - each split tet
        becomes 4 rows, so row order/count differs from the input and
        must not be assumed to correspond positionally.
    """
    nodes_arr = np.asarray(nodes, dtype=np.float64)
    pending = np.asarray(tets, dtype=np.int64)
    finished_chunks: List[np.ndarray] = []
    n_split_total = 0
    worst_before = float(_tet_volumes(nodes_arr, pending).max()) if len(pending) else 0.0

    for _ in range(max_depth):
        if len(pending) == 0:
            break
        vols = _tet_volumes(nodes_arr, pending)
        oversized = vols > max_volume
        if not np.any(oversized):
            finished_chunks.append(pending)
            pending = np.empty((0, 4), dtype=np.int64)
            break

        finished_chunks.append(pending[~oversized])
        to_split = pending[oversized]
        n_split_total += len(to_split)

        centroids = nodes_arr[to_split].mean(axis=1)
        base_idx = len(nodes_arr)
        centroid_idx = np.arange(base_idx, base_idx + len(to_split), dtype=np.int64)
        nodes_arr = np.vstack([nodes_arr, centroids])

        a, b, c, d = to_split[:, 0], to_split[:, 1], to_split[:, 2], to_split[:, 3]
        pending = np.concatenate([
            np.stack([a, b, c, centroid_idx], axis=1),
            np.stack([a, b, d, centroid_idx], axis=1),
            np.stack([a, c, d, centroid_idx], axis=1),
            np.stack([b, c, d, centroid_idx], axis=1),
        ], axis=0)

    if len(pending):
        logger.warning(
            f"subdivide_oversized_tetrahedra: {len(pending)} cell(s) still "
            f"exceed max_volume={max_volume:.4g} after {max_depth} levels "
            f"(worst {float(_tet_volumes(nodes_arr, pending).max()):.4g}) - "
            f"kept as-is rather than split indefinitely"
        )
        finished_chunks.append(pending)

    new_tets = np.vstack(finished_chunks) if finished_chunks else np.asarray(tets, dtype=np.int64)
    if n_split_total:
        logger.info(
            f"subdivide_oversized_tetrahedra: split {n_split_total} oversized "
            f"cell(s) (worst {worst_before:.4g} -> target {max_volume:.4g}), "
            f"{len(new_tets) - len(tets)} net new cells"
        )
    return nodes_arr, new_tets


def repair_nonmanifold_cells(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Detect tetrahedra that make some triangular face shared by more than
    2 cells (non-manifold) and mark the redundant ones for removal.

    The main known cause is fixed at the source now: an isolated embedded
    solid (e.g. a car body) needs a tetgen hole seed or its own interior
    gets filled with spurious tetrahedra that overlap the BL prisms already
    occupying that space (see mesh_domain_classify.find_point_inside_closed_shell
    and fill_core_volume's `holes` parameter). This function remains as a
    safety net for whatever that doesn't catch (e.g. a hole point that
    couldn't be found for a very non-convex solid, or a genuinely tight BL
    seam producing a near-degenerate boundary facet that tetgen's
    `nobisect=True` core fill can't resolve by inserting a boundary point
    there). Left unrepaired, this is a real conservation violation: a
    finite-volume face extraction can only ever attribute a shared face to
    2 of the 3+ cells touching it, silently dropping flux through it for
    the rest (see face_extractor.py's hard failure on exactly this
    condition).

    A triangular face has at most one legitimate neighbouring cell per
    side (the tet whose 4th vertex, the "apex", lies on that side of the
    face's plane). When more than one cell shares an apex-side, they are
    physically overlapping duplicates of each other - keep only the
    largest-volume one and drop the rest, independently per over-shared
    face.

    Args:
        nodes: (n_nodes, 3) node coordinates
        cells: (n_cells, 4) tetrahedral connectivity

    Returns:
        Boolean keep-mask, shape=(n_cells,); False marks a cell to remove
    """
    n_cells = len(cells)
    keep = np.ones(n_cells, dtype=bool)
    if n_cells == 0:
        return keep

    face_templates = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ], dtype=np.int64)
    apex_of_face = np.array([3, 2, 1, 0], dtype=np.int64)

    all_faces = cells[:, face_templates].reshape(-1, 3)
    apex_nodes = cells[:, apex_of_face].reshape(-1)
    cell_of_face = np.repeat(np.arange(n_cells), 4)

    sorted_faces = np.sort(all_faces, axis=1)
    face_dtype = np.dtype((np.void, sorted_faces.dtype.itemsize * 3))
    face_voids = np.ascontiguousarray(sorted_faces).view(face_dtype).reshape(-1)

    order = np.argsort(face_voids, kind='stable')
    sorted_voids = face_voids[order]
    sorted_cells = cell_of_face[order]
    sorted_apex = apex_nodes[order]
    sorted_face_nodes = sorted_faces[order]

    change = np.flatnonzero(sorted_voids[1:] != sorted_voids[:-1]) + 1
    group_starts = np.concatenate([[0], change])
    group_ends = np.concatenate([change, [len(sorted_voids)]])
    counts = group_ends - group_starts

    invalid_groups = np.flatnonzero(counts > 2)
    if len(invalid_groups) == 0:
        return keep

    volumes = _tet_volumes(nodes, cells)
    n_removed = 0

    for gi in invalid_groups:
        s, e = group_starts[gi], group_ends[gi]
        face_cells = sorted_cells[s:e]
        n0, n1, n2 = sorted_face_nodes[s]
        p0 = nodes[n0]
        normal = np.cross(nodes[n1] - p0, nodes[n2] - p0)

        apexes = sorted_apex[s:e]
        signed_dist = (nodes[apexes] - p0) @ normal

        for side_mask in (signed_dist > 0, signed_dist <= 0):
            side_cells = face_cells[side_mask]
            if len(side_cells) <= 1:
                continue
            best = side_cells[np.argmax(volumes[side_cells])]
            for c in side_cells:
                if c != best and keep[c]:
                    keep[c] = False
                    n_removed += 1

    if n_removed:
        logger.warning(
            f"Repaired {len(invalid_groups)} non-manifold faces by removing "
            f"{n_removed} redundant overlapping tetrahedra"
        )
    return keep


def attribute_cells_from_trifaces(
    cells: np.ndarray,
    trifaces: np.ndarray,
    triface_markers: np.ndarray,
    marker_to_name: dict,
) -> np.ndarray:
    """Recover each cell's source boundary group from fill_core_volume's
    facet markers, for cells that own a marked boundary face.

    Needed once fill_core_volume runs with nobisect=False (graded max-cell-
    size regions): tetgen may subdivide an input boundary facet into many
    sub-facets to satisfy the size cap, so those sub-facets' node indices
    no longer exist in the pre-fill surface mesh, and plain node-index
    matching (the pre-existing mesh_boundary.py fallback) can no longer
    find them. tetgen's own facet markers are inherited by every
    sub-facet of a marked input facet regardless of subdivision, so
    matching a cell's own boundary face against the marker set (by node
    SET, not index into some external array) works unconditionally.

    Args:
        cells: (n_cells, 4) tetrahedral connectivity, in the SAME index
            space as `trifaces` (i.e. call this before any node reindexing
            - reindexing only changes what a node index means, never which
            cell owns which row, so the returned per-row group assignment
            stays valid across a later remap)
        trifaces: (n_tri, 3) boundary triangles from fill_core_volume
        triface_markers: (n_tri,) int32, 0 = no marker (interior-only
            facet, e.g. the BL/core interface - never a real exterior
            boundary so it's fine to leave unattributed)
        marker_to_name: maps a nonzero marker value back to its boundary
            group name

    Returns:
        (n_cells,) str array, '' where the cell owns no marked boundary
        face
    """
    n_cells = len(cells)
    cell_groups = np.full(n_cells, '', dtype=object)

    nonzero = triface_markers != 0
    if not np.any(nonzero):
        return cell_groups

    marked_tri = np.sort(trifaces[nonzero], axis=1)
    marked_markers = triface_markers[nonzero]
    tri_dtype = np.dtype((np.void, marked_tri.dtype.itemsize * 3))
    marked_hash = np.ascontiguousarray(marked_tri).view(tri_dtype).reshape(-1)

    order = np.argsort(marked_hash, kind='stable')
    sorted_hash = marked_hash[order]
    sorted_marker = marked_markers[order]

    face_templates = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    all_faces = cells[:, face_templates].reshape(-1, 3)
    cell_of_face = np.repeat(np.arange(n_cells), 4)
    face_hash = np.ascontiguousarray(np.sort(all_faces, axis=1)).view(tri_dtype).reshape(-1)

    pos = np.clip(np.searchsorted(sorted_hash, face_hash), 0, len(sorted_hash) - 1)
    matched = sorted_hash[pos] == face_hash

    for cell_idx, marker in zip(cell_of_face[matched].tolist(), sorted_marker[pos[matched]].tolist()):
        cell_groups[cell_idx] = marker_to_name[marker]

    return cell_groups
