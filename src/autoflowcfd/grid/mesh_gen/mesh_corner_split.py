"""BL 挤出前的尖锐拐角顶点拆分。

概述：把接触到 3 个及以上曲面片的硬边/拐角顶点，按每个曲面片各复制一份、
沿各自真实法向偏移，再用 bevel/cap 三角形把裂开的缝隙缝合——单一平均法向
无法正确表达 valence-3+ 拐角，容易在挤出时自相交。下面详细说明原因与
实现细节（cap 扇形必须按真实几何环绕顺序连接，否则会产生扭曲的连接面）。

extrude_single_layer's per-node averaged normal (mesh_layer_step.py) and
its miter-join compensation model a SINGLE two-patch sharp edge correctly
(a fixed compensation factor along one blended direction), but cannot
represent a genuine valence-3+ corner - three or more patches meeting at
one point - without risking self-intersection: no single blended
direction simultaneously offsets three independent planes correctly, and
mesh_front_collision.py's reactive freeze then rolls the offending nodes
back and permanently stops them, producing degenerate (zero-volume,
dropped) cells for the remainder of the run. Confirmed directly on
cube_demo (a literal box body): freezing starts on the very FIRST BL
layer, exactly at the box's own edge/corner-adjacent nodes, and cascades
to affect the majority of the surface within a handful of layers.

split_sharp_corners takes the alternative real BL meshers (Pointwise's
T-Rex, ANSA) use for this: duplicate a hard-edge/corner vertex into one
copy PER smooth patch touching it, offset each copy along its own patch's
true (un-blended) normal, and stitch the resulting gaps closed with extra
"bevel" triangles along every hard edge, plus a fan of "cap" triangles at
any valence-3+ corner where 3+ patches meet at one point (a 2-patch edge
alone doesn't need a cap - the single bevel quad already closes it
completely). Each of the pieces (patch A's own offset, patch B's own
offset, the flat bevel/cap connecting them) is individually incapable of
folding over at a CONVEX feature, unlike a single blended-normal miter
estimate.

Correctness requirement for capping: the cap fan MUST connect the
patches in their true geometric cyclic order around the vertex, walked
from the LOCAL hard-edge adjacency (each patch neighbours exactly 2
others in a simple closed fan). A naive fan in arbitrary (e.g. patch-id)
order is only safe for k==3 (any 3 points form one valid triangle
regardless of order) - for k>3 it can connect geometrically non-adjacent
patches directly, producing a badly twisted/oversized connector once the
copies diverge past layer 0 (confirmed directly: an earlier, order-naive
version of this function caused a 1.4-million-pair mesh overlap explosion
on a real, non-cube body with several k>3 vertices). Whenever the local
topology at a k>=3 vertex ISN'T a simple closed fan (non-manifold input,
or the vertex sits where the mesh's own patch structure is more tangled
than a plain star), this module does not split that vertex at all -
falling back to the pre-split single-averaged-normal behaviour there -
rather than risk an incorrectly-ordered connector or an actually unclosed
gap.

A vertex on the boundary of the whole extrude-faces patch (part of an
edge with only one adjacent face - a seam with a non-extruded, e.g.
core-only, boundary group) is split normally but never capped: its
patches don't form a closed fan (the two ends of the open fan just run to
the boundary independently, nothing to connect), and
mesh_tetgen_core.build_seam_taper_scale already handles that seam
separately by tapering displacement to zero there.
"""

import numpy as np
from typing import Tuple
from loguru import logger

FEATURE_ANGLE_THRESHOLD_RAD = np.deg2rad(20.0)


def _face_normals(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = nodes[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cross_norm = np.linalg.norm(cross, axis=1)
    return cross / np.maximum(cross_norm, 1e-300)[:, np.newaxis]


def _implied_edge_radius(edge_length: float, dihedral_angle: float) -> float:
    """Local radius of curvature implied by one triangulated edge, treating
    it as one chord of a circular arc of unknown radius swept through
    `dihedral_angle` (the standard chord-to-radius relation: chord c,
    subtended angle theta, radius r = c / (2 sin(theta/2))).

    The point of this estimate: for a genuinely SHARP CAD crease (two
    flat faces meeting at a fixed G0-discontinuous angle), that angle is a
    property of the two flat faces themselves - it does NOT shrink as the
    mesh is refined, so the radius this formula implies shrinks toward 0
    in direct proportion to edge_length (r = c / const). For an ordinary
    CURVED surface of true physical radius R that is merely under-
    tessellated (few facets across the curve), the SAME formula recovers
    approximately R itself, regardless of edge_length, PROVIDED
    edge_length is small relative to R (a large chord on a fine curve
    starts to noticeably underestimate R - not a concern here since this
    is only ever evaluated on edges that already registered as locally
    "sharp", i.e. small chords by construction). Distinguishing the two
    is exactly "is the implied radius comparable to the probing edge
    length itself (shrinks toward 0 with it - a real crease) or much
    larger than it (roughly resolution-independent - a real, if coarse,
    curve)" - see split_sharp_corners' own min_feature_radius parameter
    for how this is actually used.
    """
    half_angle = dihedral_angle / 2.0
    sin_half = np.sin(half_angle)
    if sin_half < 1e-9:
        return np.inf
    return edge_length / (2.0 * sin_half)


def _unique_edges(faces: np.ndarray):
    """Yield (v0, v1, face_idx_array) per distinct undirected edge - 1
    face for a patch-boundary edge, 2 for a normal interior edge, more
    only for non-manifold input.
    """
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edge_face_idx = np.tile(np.arange(len(faces)), 3)
    sorted_edges = np.sort(edges, axis=1)

    order = np.lexsort((sorted_edges[:, 1], sorted_edges[:, 0]))
    se = sorted_edges[order]
    efi = edge_face_idx[order]

    is_new = np.ones(len(se), dtype=bool)
    is_new[1:] = np.any(se[1:] != se[:-1], axis=1)
    boundaries = np.flatnonzero(is_new)
    boundaries = np.append(boundaries, len(se))

    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        yield se[lo, 0], se[lo, 1], efi[lo:hi]


def split_sharp_corners(
    nodes: np.ndarray,
    faces: np.ndarray,
    threshold: float = FEATURE_ANGLE_THRESHOLD_RAD,
    min_feature_radius: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split every sharp-corner/hard-edge vertex of `faces` into one node
    copy per smooth patch, adding bevel/cap triangles to keep the result
    watertight.

    Args:
        nodes: Full node array (this function only ever APPENDS rows,
            every existing index stays valid), shape=(n_nodes, 3)
        faces: Triangle connectivity to split, shape=(n_faces, 3) - a
            self-contained sub-mesh (e.g. classify_boundary_groups' own
            extrude_faces), not the whole surface
        threshold: Dihedral angle (radians) above which an edge is "hard"
            and its two faces are treated as different patches unless
            connected some other way
        min_feature_radius: an edge whose dihedral angle exceeds
            `threshold` is still NOT treated as hard if the LOCAL RADIUS
            OF CURVATURE its own geometry implies is at or above this
            value (meters) - see _implied_edge_radius's own docstring for
            why this distinguishes a genuinely sharp (near-zero-radius)
            CAD crease from an ordinary curved surface (a fillet, a
            rounded corner) that is merely under-tessellated relative to
            its own true radius. Deliberately a single-edge heuristic,
            not a guarantee: an actually-refined input mesh would let the
            plain `threshold` check alone tell the two apart correctly in
            general, but naive (flat, non-surface-fitting) subdivision
            does NOT reduce a coarse curve's own per-facet angle at all
            (sub-triangles of a flat triangle stay exactly coplanar with
            it - confirmed directly: 3 rounds of subdivision on cube_demo
            changed its patch count not at all, while making self-
            intersection markedly worse from the extra small triangles
            crowding the same tight corner), so that path isn't available
            without real surface-fitting resampling this project doesn't
            have. 0.0 (default) preserves the plain angle-only behaviour.

    Returns:
        new_nodes: nodes with copy rows appended, shape=(n_nodes + n_copies, 3)
            - every copy starts at the SAME position as its original
            vertex (only later BL extrusion layers make copies diverge)
        topology_faces: faces.copy() with vertex indices remapped to the
            correct per-patch copy, followed by the new bevel/cap
            triangles, shape=(n_faces + n_extra, 3)
        real_face_mask: bool, shape=(len(topology_faces),) - True for the
            first n_faces rows (the original, remapped triangles - used
            for per-node normal averaging), False for the appended
            bevel/cap rows (pure connectivity, never contribute to normal
            averaging - their own corner nodes already draw a correct
            normal from their real patch's own faces)
        orig_of_node: int64, shape=(len(new_nodes),) - maps every node
            (original and copy) back to its ORIGINAL vertex index in the
            input `nodes` array, for expanding any other per-original-
            vertex array (taper_scale, thickness_limit) the same way
        bevel_source_face: int64, shape=(n_extra,) - for every appended
            row, which original `faces` row (0-indexed) it should inherit
            a boundary-group attribution from
    """
    n_nodes = len(nodes)
    n_faces = len(faces)
    if n_faces == 0:
        return (
            nodes.copy(), faces.copy(), np.ones(0, dtype=bool),
            np.arange(n_nodes), np.zeros(0, dtype=np.int64),
        )

    face_normals = _face_normals(nodes, faces)

    parent = np.arange(n_faces)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    boundary_edge_verts = []  # (v0, v1) with only 1 adjacent face
    hard_edge_list = []  # (v0, v1, fa, fb) with 2 adjacent faces, angle > threshold

    for v0, v1, fidx in _unique_edges(faces):
        if len(fidx) == 1:
            boundary_edge_verts.append((int(v0), int(v1)))
            continue
        if len(fidx) != 2:
            continue  # non-manifold edge - leave unioned-apart (safest: treat as hard, no bevel)
        fa, fb = int(fidx[0]), int(fidx[1])
        cosang = np.clip(np.dot(face_normals[fa], face_normals[fb]), -1.0, 1.0)
        angle = np.arccos(cosang)
        is_hard = angle > threshold
        # An edge that crosses the plain angle threshold is still treated
        # as an ordinary smooth edge if its own geometry implies a local
        # curvature radius at or above min_feature_radius - see this
        # function's own min_feature_radius docstring for why (and its
        # documented limits).
        if is_hard and min_feature_radius > 0.0:
            edge_length = float(np.linalg.norm(nodes[v0] - nodes[v1]))
            implied_radius = _implied_edge_radius(edge_length, angle)
            if implied_radius >= min_feature_radius:
                is_hard = False
        if is_hard:
            hard_edge_list.append((v0, v1, fa, fb))
        else:
            union(fa, fb)

    patch_id_raw = np.array([find(f) for f in range(n_faces)], dtype=np.int64)
    _, patch_id = np.unique(patch_id_raw, return_inverse=True)
    n_patches = int(patch_id.max()) + 1 if n_faces else 0

    boundary_verts = set()
    for v0, v1 in boundary_edge_verts:
        boundary_verts.add(v0)
        boundary_verts.add(v1)

    hard_edges = np.array(hard_edge_list, dtype=np.int64).reshape(-1, 4)
    he_pa = patch_id[hard_edges[:, 2]] if len(hard_edges) else np.zeros(0, dtype=np.int64)
    he_pb = patch_id[hard_edges[:, 3]] if len(hard_edges) else np.zeros(0, dtype=np.int64)
    he_differ = he_pa != he_pb
    hard_edges = hard_edges[he_differ]
    he_pa, he_pb = he_pa[he_differ], he_pb[he_differ]

    vertex_patch_adj: dict = {}
    for i in range(len(hard_edges)):
        v0i, v1i, pai, pbi = int(hard_edges[i, 0]), int(hard_edges[i, 1]), int(he_pa[i]), int(he_pb[i])
        for v in (v0i, v1i):
            d = vertex_patch_adj.setdefault(v, {})
            d.setdefault(pai, set()).add(pbi)
            d.setdefault(pbi, set()).add(pai)

    # --- Preliminary (vertex, patch) grouping, using the RAW patch_id, to
    # decide per-vertex whether splitting is safe (see module docstring).
    flat_vert = faces.ravel().astype(np.int64)
    flat_patch = np.repeat(patch_id, 3)
    key0 = flat_vert * n_patches + flat_patch
    uk0, _ = np.unique(key0, return_inverse=True)
    uk0_vertex = uk0 // n_patches
    uk0_patch = uk0 % n_patches

    group_starts = np.flatnonzero(np.concatenate([[True], uk0_vertex[1:] != uk0_vertex[:-1]]))
    group_ends = np.append(group_starts[1:], len(uk0_vertex))

    force_single = np.zeros(n_nodes, dtype=bool)
    vertex_cyclic_order: dict = {}
    n_skipped_irregular = 0

    for gs, ge in zip(group_starts, group_ends):
        k = ge - gs
        if k < 3:
            continue
        v = int(uk0_vertex[gs])
        if v in boundary_verts:
            continue  # open fan - split is fine, just never capped (below)
        patches_here = uk0_patch[gs:ge].tolist()
        patches_set = set(patches_here)
        adj = vertex_patch_adj.get(v)
        regular = adj is not None
        if regular:
            for p in patches_here:
                neigh = adj.get(p, set())
                if len(neigh) != 2 or not neigh.issubset(patches_set):
                    regular = False
                    break
        if regular:
            cyclic = [patches_here[0]]
            prev, cur = None, patches_here[0]
            ok = True
            for _ in range(k - 1):
                neigh = list(adj[cur])
                nxt = neigh[0] if neigh[0] != prev else neigh[1]
                if nxt in cyclic:
                    ok = False
                    break
                cyclic.append(nxt)
                prev, cur = cur, nxt
            if ok and len(cyclic) == k and patches_here[0] in adj[cyclic[-1]]:
                vertex_cyclic_order[v] = cyclic
            else:
                regular = False
        if not regular:
            force_single[v] = True
            n_skipped_irregular += 1

    if n_skipped_irregular:
        logger.warning(
            f"Sharp-corner splitting: {n_skipped_irregular} valence-3+ vertex/vertices "
            f"had irregular local patch topology (not a simple closed fan) - left "
            f"unsplit (falls back to the pre-split averaged-normal behaviour there) "
            f"rather than risk an incorrectly-ordered connector"
        )

    # --- Final (vertex, patch) grouping: force_single vertices collapse
    # every one of their face-corners to a single synthetic patch (0) -
    # i.e. no split at all for that vertex, regardless of how many real
    # patches actually touch it.
    final_flat_patch = np.where(force_single[flat_vert], 0, flat_patch)
    key = flat_vert * n_patches + final_flat_patch
    unique_keys, inverse = np.unique(key, return_inverse=True)
    uk_vertex = unique_keys // n_patches
    uk_patch = unique_keys % n_patches

    is_first_for_vertex = np.ones(len(unique_keys), dtype=bool)
    is_first_for_vertex[1:] = uk_vertex[1:] != uk_vertex[:-1]

    new_node_index = np.empty(len(unique_keys), dtype=np.int64)
    new_node_index[is_first_for_vertex] = uk_vertex[is_first_for_vertex]
    n_extra_copies = int(np.sum(~is_first_for_vertex))
    new_node_index[~is_first_for_vertex] = n_nodes + np.arange(n_extra_copies)

    orig_of_node = np.concatenate([
        np.arange(n_nodes, dtype=np.int64),
        uk_vertex[~is_first_for_vertex],
    ])
    new_nodes = np.vstack([nodes, nodes[uk_vertex[~is_first_for_vertex]]])

    topology_faces_real = new_node_index[inverse].reshape(n_faces, 3)

    n_split_vertices = int(np.sum(np.bincount(uk_vertex, minlength=n_nodes) > 1))
    if n_extra_copies:
        logger.info(
            f"Sharp-corner splitting: {n_split_vertices} vertices split into "
            f"{n_extra_copies} extra copies ({n_patches} smooth patches total)"
        )

    def lookup_copy(vertex_arr: np.ndarray, patch_arr: np.ndarray) -> np.ndarray:
        eff_patch = np.where(force_single[vertex_arr], 0, patch_arr)
        k = vertex_arr.astype(np.int64) * n_patches + eff_patch.astype(np.int64)
        idx = np.searchsorted(unique_keys, k)
        idx = np.clip(idx, 0, len(unique_keys) - 1)
        return new_node_index[idx]

    # --- Bevel strips along every hard edge whose two faces landed in
    # genuinely different patches (a hard edge unioned back together via
    # some OTHER path elsewhere on the mesh already shares copies at both
    # ends - no gap, no bevel needed there). A force_single endpoint
    # collapses one side of the resulting quad to a point (both its
    # triangles then share 2 identical vertices on that side) - filtered
    # out below as a degenerate triangle, same as any other.
    bevel_tris = []
    bevel_source = []
    if len(hard_edges):
        v0, v1, fa, fb = hard_edges[:, 0], hard_edges[:, 1], hard_edges[:, 2], hard_edges[:, 3]
        c_v0_a = lookup_copy(v0, he_pa)
        c_v1_a = lookup_copy(v1, he_pa)
        c_v0_b = lookup_copy(v0, he_pb)
        c_v1_b = lookup_copy(v1, he_pb)
        bevel_tris.append(np.stack([c_v0_a, c_v1_a, c_v1_b], axis=1))
        bevel_source.append(fa)
        bevel_tris.append(np.stack([c_v0_a, c_v1_b, c_v0_b], axis=1))
        bevel_source.append(fa)

    # --- Corner caps: fan-triangulate the true cyclic order computed
    # above for every regular valence-3+ vertex. force_single is False
    # for every such v by construction (only added to
    # vertex_cyclic_order when regular splitting applies), so lookup_copy
    # resolves each (v, patch) pair to its real per-patch copy, not the
    # collapsed single index.
    for v, cyclic in vertex_cyclic_order.items():
        v_arr = np.full(len(cyclic), v, dtype=np.int64)
        p_arr = np.array(cyclic, dtype=np.int64)
        copies = lookup_copy(v_arr, p_arr)
        apex = int(copies[0])
        for i in range(1, len(cyclic) - 1):
            bevel_tris.append(np.array([[apex, int(copies[i]), int(copies[i + 1])]], dtype=np.int64))
            bevel_source.append(np.array([-1], dtype=np.int64))  # filled in below

    if bevel_tris:
        extra_faces = np.vstack(bevel_tris)
        bevel_source_face = np.concatenate(bevel_source)

        # Drop degenerate rows (a force_single endpoint on an otherwise
        # split edge collapses one side of its bevel quad to a repeated
        # vertex - see the bevel-strip comment above).
        degenerate = (
            (extra_faces[:, 0] == extra_faces[:, 1]) |
            (extra_faces[:, 1] == extra_faces[:, 2]) |
            (extra_faces[:, 0] == extra_faces[:, 2])
        )
        if np.any(degenerate):
            extra_faces = extra_faces[~degenerate]
            bevel_source_face = bevel_source_face[~degenerate]

        # Corner-cap rows were appended with a placeholder -1 source face
        # (a cap triangle isn't "part of" any single original face) - fall
        # back to whichever face happens to touch its apex vertex, so
        # boundary-group inheritance still resolves to the right group
        # name rather than crashing on a -1 index.
        need_fallback = bevel_source_face < 0
        if np.any(need_fallback):
            vertex_to_any_face = np.full(n_nodes + n_extra_copies, -1, dtype=np.int64)
            vertex_to_any_face[faces[:, 0]] = np.arange(n_faces)
            vertex_to_any_face[faces[:, 1]] = np.arange(n_faces)
            vertex_to_any_face[faces[:, 2]] = np.arange(n_faces)
            apex_nodes = extra_faces[need_fallback, 0]
            apex_orig = orig_of_node[apex_nodes]
            resolved = vertex_to_any_face[apex_orig]
            resolved = np.where(resolved < 0, 0, resolved)
            bevel_source_face[need_fallback] = resolved
    else:
        extra_faces = np.zeros((0, 3), dtype=np.int64)
        bevel_source_face = np.zeros(0, dtype=np.int64)

    topology_faces = np.vstack([topology_faces_real, extra_faces]).astype(np.int64)
    real_face_mask = np.zeros(len(topology_faces), dtype=bool)
    real_face_mask[:n_faces] = True

    if len(extra_faces):
        logger.info(
            f"Sharp-corner splitting: added {len(extra_faces)} bevel/cap "
            f"triangle(s) to keep the split surface watertight"
        )

    return new_nodes, topology_faces, real_face_mask, orig_of_node, bevel_source_face
