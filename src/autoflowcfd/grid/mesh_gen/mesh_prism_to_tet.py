"""把分层棱柱网格（来自 mesh_extrusion.extrude_layers）转换成保形的四面体网格。

从 mesh_extrusion.py 拆出（该文件保留分层生成循环
extrude_layers/extrude_single_layer），纯粹为了让两个文件都控制在 450
行以内；两个模块彼此没有依赖关系。
"""

import numpy as np
from typing import List, Optional, Tuple
from loguru import logger

# Same relative-to-cell-size convention mesh_background.py's own
# degenerate-tet cleanup already uses (`(min_cell_size ** 3) * 1e-6`) -
# reused here so a fully-collapsed prism (an entire base face's own
# taper_scale/budget stalled to ~0, not just floating-point-exact 0) is
# recognised as degenerate too, not just the literal-zero case the old
# fixed 1e-20 threshold caught. Deliberately still many orders of
# magnitude below a genuine wedge prism's own volume (one vertical edge
# near-zero, the other two normal) - see convert_layers_to_prisms' own
# Returns doc for why that distinction matters (a wedge is real geometry,
# dropping it tears a hole; a fully-collapsed prism contributes nothing).
DEGENERATE_VOLUME_FRACTION = 1e-6


def convert_layers_to_tetrahedra(
    all_nodes: np.ndarray,
    layer_connectivity: List[np.ndarray],
    base_faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert layered prism mesh to a *conformal* tetrahedral mesh.

    Each triangular prism between two consecutive layers is split into 3
    tetrahedra.  The split is chosen so that neighbouring prisms agree on the
    diagonal of every shared quadrilateral face, which is what makes the
    resulting mesh conformal (every interior face is shared by exactly two
    cells).  A fixed template applied blindly does *not* have this property and
    produces hanging faces that a finite-volume solver then mistakes for
    boundary faces.

    Rule: sort the three base vertices by global node index, v0 < v1 < v2, and
    let w_i be the corresponding vertices on the next layer.  Emit

        T1 = (v0, v1, v2, w2)
        T2 = (v0, v1, w1, w2)
        T3 = (v0, w0, w1, w2)

    The diagonals this induces on the three quad faces are v0-w1, v1-w2 and
    v0-w2, i.e. always "lower-indexed bottom vertex to higher-indexed top
    vertex".  That rule depends only on the two vertices of the shared edge, so
    two prisms sharing an edge necessarily pick the same diagonal.

    Tetrahedra are additionally oriented to have positive signed volume, then
    any tet whose signed volume is analytically zero (not just small - see
    "dropped tets" below) is removed from the output entirely.

    Args:
        all_nodes: All nodes from all layers, shape=(total_nodes, 3)
        layer_connectivity: one entry per extrusion STEP (n_layers - 1
            entries for n_layers node-layers), matching the convention
            used everywhere else this project derives a node-layer count
            from an extrude_layers-produced connectivity list (see
            mesh_background_merge._build_merged_mesh's own
            `n_layers = len(bl_layer_conn) + 1` and its docstring for why
            - extrude_layers.all_layer_nodes starts with the layer-0
            block then appends one block per step, so the node array
            always has one MORE layer than there are steps). Only this
            list's LENGTH is used (its actual per-layer face-index
            contents are never read - every layer shares base_faces' own
            local face topology, just offset by nodes_per_layer).
        base_faces: Original surface faces, shape=(n_faces, 3)

    Returns:
        (tetrahedra, face_of_tet): tetrahedra connectivity, shape=(n_tets, 4)
        - n_tets may be LESS than n_base_faces*(n_layers-1)*3 (see "dropped
        tets" below); face_of_tet, shape=(n_tets,), maps each surviving tet
        back to its base_faces row index. A caller that used to assume a
        fixed tile (n_tets_per_face = n_tets // n_base_faces, cell i ->
        base_faces[i % n_base_faces]) must use face_of_tet instead now that
        cells can be dropped.

        Dropped tets: exactly (to floating-point noise, |det| < 1e-20) zero
        volume - e.g. a prism fully collapsed by a seam taper_scale of 0
        (mesh_extrusion.extrude_layers' taper_scale), which pins a node at
        its original position every layer so the prism between two
        identical layer positions has zero thickness. Such a tet's one
        non-degenerate face is always shared only internally with one of
        that same prism's other 2 tets (T1's real face is literally one of
        T2's own faces - the prism's own internal split diagonal), never
        with an external neighbour, so dropping it loses no real geometry
        and cannot orphan a face some other prism still expects matched
        (verified empirically: the merged mesh's own conformality check -
        every interior face shared by exactly 2 cells - still passes after
        dropping).
    """
    # +1: layer_connectivity holds one entry per extrusion STEP, not per
    # node-layer - see this function's own `layer_connectivity` doc. Using
    # len(layer_connectivity) directly here (as this function did until
    # this fix) under-counts n_layers by exactly one, which in turn makes
    # nodes_per_layer = n_total_nodes // n_layers compute a value LARGER
    # than the true per-layer node count - every off_lo/off_hi below then
    # lands on the wrong absolute node index, silently connecting a vertex
    # from one layer to a same-local-index vertex several layers away
    # (confirmed directly: on a real case this produced tetrahedra with
    # vertices from the near-wall transition region AND the domain's own
    # inlet/outlet/farfield walls in the same cell - a single ~14 m^3 tet
    # spanning the full domain length, and this wasn't a rare outlier:
    # roughly half of all transition-stage tets on that case exceeded a
    # sane volume bound). The prism counterpart below
    # (convert_layers_to_prisms) has the identical fix; its own caller in
    # mesh_background_merge.py compensated with an ad-hoc slice-time +1
    # that masked this exact bug there, but nothing analogous existed for
    # this function's own caller.
    n_layers = len(layer_connectivity) + 1
    n_base_faces = len(base_faces)

    if n_layers < 2:
        raise ValueError("Need at least 2 layers to create volume")

    n_total_nodes = len(all_nodes)
    nodes_per_layer = n_total_nodes // n_layers

    logger.info(f"Converting {n_layers-1} layer pairs to conformal tetrahedra...")

    # Sort each base triangle's vertices by global index once; the relative
    # order is identical on every layer (index = base + layer*nodes_per_layer),
    # so one sort is valid for the whole stack.
    sorted_base = np.sort(base_faces, axis=1)          # (n_faces, 3) -> v0<v1<v2

    n_tets = n_base_faces * (n_layers - 1) * 3
    tetrahedra = np.empty((n_tets, 4), dtype=np.int64)
    face_of_tet = np.empty(n_tets, dtype=np.int64)
    face_range = np.arange(n_base_faces)

    tet_idx = 0
    for layer_idx in range(n_layers - 1):
        off_lo = layer_idx * nodes_per_layer
        off_hi = (layer_idx + 1) * nodes_per_layer

        v0 = off_lo + sorted_base[:, 0]
        v1 = off_lo + sorted_base[:, 1]
        v2 = off_lo + sorted_base[:, 2]
        w0 = off_hi + sorted_base[:, 0]
        w1 = off_hi + sorted_base[:, 1]
        w2 = off_hi + sorted_base[:, 2]

        for quad in ((v0, v1, v2, w2),
                     (v0, v1, w1, w2),
                     (v0, w0, w1, w2)):
            sl = slice(tet_idx, tet_idx + n_base_faces)
            tetrahedra[sl, 0] = quad[0]
            tetrahedra[sl, 1] = quad[1]
            tetrahedra[sl, 2] = quad[2]
            tetrahedra[sl, 3] = quad[3]
            face_of_tet[sl] = face_range
            tet_idx += n_base_faces

    # Enforce positive signed volume (swap two vertices where inverted) so that
    # downstream code can rely on orientation instead of taking |det|.
    tetrahedra = orient_tetrahedra(all_nodes, tetrahedra)

    # Drop degenerate/near-degenerate connector-artifact tets (see "Dropped
    # tets" above) - recomputed post-orientation since orient_tetrahedra
    # only flips sign, never changes magnitude.
    p0 = all_nodes[tetrahedra[:, 0]]
    p1 = all_nodes[tetrahedra[:, 1]]
    p2 = all_nodes[tetrahedra[:, 2]]
    p3 = all_nodes[tetrahedra[:, 3]]
    e1, e2, e3 = p1 - p0, p2 - p0, p3 - p0
    det = np.einsum('ij,ij->i', e1, np.cross(e2, e3))
    drop = np.abs(det) < 1e-20

    n_dropped = int(np.count_nonzero(drop))
    if n_dropped:
        logger.info(
            f"Dropped {n_dropped} exactly-zero-volume tetrahedra "
            f"(see this function's own docstring)"
        )
        keep = ~drop
        tetrahedra = tetrahedra[keep]
        face_of_tet = face_of_tet[keep]

    logger.info(f"Total tetrahedra generated: {len(tetrahedra)}")
    return tetrahedra, face_of_tet


def orient_tetrahedra(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Flip inverted tetrahedra so every cell has positive signed volume.

    Signed volume = det(p1-p0, p2-p0, p3-p0) / 6.  Swapping two vertices flips
    the sign, so inverted cells are repaired in place.  Cells that are exactly
    degenerate (zero volume) cannot be repaired and are reported.

    Args:
        nodes: Node coordinates, shape=(n_nodes, 3)
        tets: Tetrahedral connectivity, shape=(n_tets, 4)

    Returns:
        Connectivity with all signed volumes >= 0.
    """
    p0 = nodes[tets[:, 0]]
    p1 = nodes[tets[:, 1]]
    p2 = nodes[tets[:, 2]]
    p3 = nodes[tets[:, 3]]
    det = np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))

    inverted = det < 0.0
    n_inv = int(np.count_nonzero(inverted))
    if n_inv:
        # Swap last two vertices to restore positive orientation.
        tets[inverted, 2], tets[inverted, 3] = (
            tets[inverted, 3].copy(), tets[inverted, 2].copy()
        )
        logger.info(f"Re-oriented {n_inv} inverted tetrahedra")

    n_degen = int(np.count_nonzero(np.abs(det) < 1e-20))
    if n_degen:
        logger.warning(
            f"{n_degen} degenerate (zero-volume) tetrahedra detected; these "
            f"cannot be fixed by re-orientation and indicate collapsed layers"
        )

    return tets


def convert_layers_to_prisms(
    all_nodes: np.ndarray,
    layer_connectivity: List[np.ndarray],
    base_faces: np.ndarray,
    min_cell_size: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert layered prism mesh into genuine triangular-prism cells - the
    true-prism counterpart to convert_layers_to_tetrahedra, kept in this
    same module since the two share the exact same per-layer node-
    correspondence bookkeeping (only the final cell shape emitted differs).

    Emits ONE prism per (layer, base face) pair, using the SAME sorted-
    vertex convention (v0<v1<v2 by global node index, w_i the corresponding
    vertex one layer up) the old tet path already relies on for diagonal
    consistency - see PrismCells' and face_extractor.extract_faces_mixed's
    docstrings for why this makes a prism's 8 boundary faces bit-identical
    to what convert_layers_to_tetrahedra's 3-tet split of the same slab
    would have produced, and therefore automatically conformal with a
    neighbouring prism (or, at the BL/core interface, a neighbouring core
    tet) without this function needing any cross-cell coordination beyond
    the same global-index sort every prism applies independently.

    Args:
        all_nodes: All nodes from all layers, shape=(total_nodes, 3)
        layer_connectivity: one entry per extrusion STEP, not per
            node-layer - see convert_layers_to_tetrahedra's own
            `layer_connectivity` doc for the full explanation (this
            function had the identical off-by-one bug until the same fix
            was applied here). Only its LENGTH is used - every layer
            shares base_faces' own local face topology, just offset by
            nodes_per_layer.
        base_faces: Original surface faces, shape=(n_faces, 3)
        min_cell_size: Optional target cell size (meters) used to scale the
            degenerate-volume drop threshold below to
            `(min_cell_size**3) * DEGENERATE_VOLUME_FRACTION` instead of a
            fixed 1e-20 - see that constant's own comment for why a plain
            float-noise epsilon under-catches a prism that has fully
            collapsed (all 3 vertical edges ~0, an entire base face
            stalled) but not to EXACTLY floating-point zero. None (the
            default) keeps the old fixed 1e-20 threshold, for any caller
            that doesn't have a size reference handy.

    Returns:
        (prisms, face_of_prism): prism connectivity, shape=(n_prisms, 6) as
        (v0,v1,v2,w0,w1,w2); face_of_prism, shape=(n_prisms,), maps each
        surviving prism back to its base_faces row index (n_prisms may be
        less than n_base_faces*(n_layers-1) - prisms whose volume is
        negligible relative to the degenerate-volume threshold above
        (whether from a taper_scale of 0 collapsing a whole base face to
        zero thickness, or - see that threshold's own comment - a whole-
        face stall that lands close to but not exactly at zero) are
        dropped. A prism with only ONE vertical edge near-zero and the
        other two still growing normally (a genuine wedge, not a stall) is
        NOT touched by this - its volume stays comparable to a normal
        prism's, far above this threshold regardless of min_cell_size, by
        construction (dropping it WOULD tear a real hole in the outer
        boundary, unlike the fully-collapsed case: this project's own
        Part12 history is full of exactly that class of defect from
        deleting cells too aggressively). The resulting coordinate-
        duplicate-but-index-distinct seam a fully-collapsed drop leaves
        behind is cleaned up by the caller's coincident-point merge pass,
        same as it already was for the equivalent tet case.
    """
    # +1: see convert_layers_to_tetrahedra's identical fix/comment above -
    # layer_connectivity holds one entry per extrusion STEP, not per
    # node-layer.
    n_layers = len(layer_connectivity) + 1
    n_base_faces = len(base_faces)

    if n_layers < 2:
        raise ValueError("Need at least 2 layers to create volume")

    n_total_nodes = len(all_nodes)
    nodes_per_layer = n_total_nodes // n_layers

    logger.info(f"Converting {n_layers-1} layer pairs to {n_base_faces} boundary-layer prism(s) each...")

    sorted_base = np.sort(base_faces, axis=1)  # (n_faces, 3) -> v0<v1<v2, same per layer

    n_prisms = n_base_faces * (n_layers - 1)
    prisms = np.empty((n_prisms, 6), dtype=np.int64)
    face_of_prism = np.empty(n_prisms, dtype=np.int64)
    face_range = np.arange(n_base_faces)

    prism_idx = 0
    for layer_idx in range(n_layers - 1):
        off_lo = layer_idx * nodes_per_layer
        off_hi = (layer_idx + 1) * nodes_per_layer
        sl = slice(prism_idx, prism_idx + n_base_faces)
        prisms[sl, 0] = off_lo + sorted_base[:, 0]
        prisms[sl, 1] = off_lo + sorted_base[:, 1]
        prisms[sl, 2] = off_lo + sorted_base[:, 2]
        prisms[sl, 3] = off_hi + sorted_base[:, 0]
        prisms[sl, 4] = off_hi + sorted_base[:, 1]
        prisms[sl, 5] = off_hi + sorted_base[:, 2]
        face_of_prism[sl] = face_range
        prism_idx += n_base_faces

    # Drop fully-collapsed (whole-base-face) prisms - see
    # DEGENERATE_VOLUME_FRACTION's own comment for why this threshold is
    # scaled to min_cell_size rather than a fixed float-noise epsilon, and
    # this function's own Returns doc for why a volume-based (not per-
    # edge) check is what keeps a genuine wedge prism safe from being
    # dropped.
    from ..validation.quality_metrics import compute_prism_volumes
    volumes = compute_prism_volumes(all_nodes, prisms)
    degenerate_threshold = (
        (min_cell_size ** 3) * DEGENERATE_VOLUME_FRACTION if min_cell_size is not None
        else 1e-20
    )
    drop = volumes < degenerate_threshold

    n_dropped = int(np.count_nonzero(drop))
    if n_dropped:
        logger.info(
            f"Dropped {n_dropped} fully-collapsed prisms (volume < "
            f"{degenerate_threshold:.3e} m^3, an entire base face stalled)"
        )
        keep = ~drop
        prisms = prisms[keep]
        face_of_prism = face_of_prism[keep]

    logger.info(f"Total prisms generated: {len(prisms)}")
    return prisms, face_of_prism
