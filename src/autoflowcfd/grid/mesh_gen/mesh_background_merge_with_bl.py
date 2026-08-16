"""_build_merged_mesh 的"有 BL"主分支：挤出边界层棱柱，再用 tetgen
从 BL 的真实外表面直接填充剩余体积。

从 mesh_background_merge.py 拆分出来（原文件超过 400 行上限），纯粹是
代码搬移——逻辑与 _build_merged_mesh 里原来的 `else:` 分支（对应
`if len(extrude_faces) == 0:` 的否定分支）完全一致，只是把它变成一个独
立的模块级函数，由 mesh_background_merge._build_merged_mesh 在该分支下
直接调用并原样返回其结果。`--bl-only` 调试导出那一小段进一步拆到了
mesh_background_merge_bl_export.py（自包含的"构造导出对象 -> 写文件 ->
sys.exit(0)"流程，与本文件其余逻辑没有共享状态）。
"""

import sys
import numpy as np
from typing import Dict, Optional, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ..structures import BoundaryMap

from .mesh_extrusion import extrude_layers
from .mesh_prism_to_tet import convert_layers_to_prisms
from .mesh_utils import compute_face_normals
from .mesh_corner_split import split_sharp_corners
from .mesh_background_merge_bl_export import _export_bl_only_and_exit
from .mesh_background_merge_utils import _export_partial_mesh_and_exit
from .mesh_tetgen_core import (
    build_seam_taper_scale, fill_core_volume,
    compute_local_thickness_limit,
    attribute_cells_from_trifaces, generate_core_background_points,
    subdivide_oversized_tetrahedra,
    CORE_TETGEN_MINRATIO, CORE_TETGEN_MINDIHEDRAL,
)


def _build_merged_mesh_with_bl(
    surface_nodes: np.ndarray,
    surface_boundaries: 'BoundaryMap',
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    extrude_faces: np.ndarray,
    core_faces: np.ndarray,
    extruded_groups,
    extrude_face_groups: np.ndarray,
    hole_points,
    core_face_groups: np.ndarray,
    group_name_to_marker: dict,
    marker_to_name: dict,
    growth_rate: float,
    min_cell_size: float,
    max_cell_size: Optional[float],
    extra_thickness_limit: Optional[np.ndarray],
    bl_layers: Optional[int],
    export_bl_only: bool,
    export_bl_only_path: Optional[str],
    export_core_only: bool,
    export_core_only_path: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, int]:
    """至少有一个曲面组适合挤出边界层时的主路径：挤出 BL 棱柱，再用
    tetgen 从 BL 的真实外表面直接填充剩余体积（不再有独立的"transition"
    阶段）。

    对应 mesh_background_merge._build_merged_mesh 里原来的 `else:` 分
    支，逐字搬移，未改动任何数值逻辑。

    Returns:
        与 _build_merged_mesh 自身完全相同的 9 元组：
        (merged_nodes, prism_cells, tet_cells, cell_groups, n_bl_cells,
        source_vertex, topology_faces, bl_cell_groups, n_transition_cells)
    """
    # OVERSIZED_TET_FACTOR / CORE_FILL_VOLUME_CAP_FRACTION 定义在
    # mesh_background_merge.py（本函数唯一的调用者所在文件），延迟导入
    # 以避免循环导入——本模块被 mesh_background_merge.py 导入，不能在
    # 模块顶层反向导入它。
    from .mesh_background_merge import OVERSIZED_TET_FACTOR, CORE_FILL_VOLUME_CAP_FRACTION

    logger.info(
        f"Step 2/4: Extruding BL layers from {len(extrude_faces)} faces "
        f"(groups: {extruded_groups})..."
    )
    n_surface_nodes = len(surface_nodes)
    taper_scale = build_seam_taper_scale(n_surface_nodes, extrude_faces, core_faces)

    # Cap each node's cumulative BL thickness near tight facing features
    # (e.g. a body's underbody close to the ground) so the two fronts
    # freeze before they can cross, instead of relying entirely on
    # repair_nonmanifold_cells to clean up the resulting overlap after
    # the fact (see compute_local_thickness_limit's own docstring for
    # why this is a strong mitigation, not a formal guarantee).
    domain_size = float(np.linalg.norm(bbox_max - bbox_min))
    thickness_limit = compute_local_thickness_limit(
        surface_nodes, extrude_faces, np.unique(extrude_faces), domain_size
    )
    if extra_thickness_limit is not None:
        thickness_limit = np.minimum(thickness_limit, extra_thickness_limit)

    # Split every sharp-corner/hard-edge vertex of the extrude-eligible
    # sub-mesh into one copy per smooth patch BEFORE extrusion - see
    # mesh_corner_split's own module docstring for why a single
    # averaged-normal-per-node offset cannot represent a genuine
    # valence-3+ corner without risking self-intersection (confirmed
    # directly on cube_demo, a literal box body: cascading collision
    # freezes - mesh_front_collision.freeze_self_colliding_nodes -
    # starting on the very FIRST BL layer, exactly at the body's own
    # sharp edges/corners, affecting the majority of the surface within
    # a handful of layers). taper_scale/thickness_limit/
    # extrude_face_groups are per-ORIGINAL-vertex/face arrays - expand
    # them the same way (a copy inherits its source's value/group)
    # before they reach extrude_layers/downstream cell attribution.
    # min_feature_radius=min_cell_size: an edge whose own geometry
    # implies a curvature radius at or above the BL's own target
    # near-wall cell size is treated as an ordinary curved surface
    # (however coarsely tessellated) rather than a sharp crease to
    # split - see split_sharp_corners' own docstring. Below that
    # scale, further mesh resolution wouldn't meaningfully change how
    # the BL sees the feature anyway, so it stays classified as hard.
    split_nodes, topology_faces, real_face_mask, orig_of_node, bevel_source_face = (
        split_sharp_corners(
            surface_nodes, extrude_faces, min_feature_radius=min_cell_size
        )
    )
    taper_scale = taper_scale[orig_of_node]
    thickness_limit = thickness_limit[orig_of_node]
    extrude_face_groups = np.concatenate(
        [extrude_face_groups, extrude_face_groups[bevel_source_face]]
    )

    # source_vertex maps a split-local (post-modulo) node index back to
    # the ORIGINAL surface vertex it represents - identity below
    # n_surface_nodes (untouched by splitting), and to whichever
    # vertex a split copy was duplicated from above that. Stage B's
    # own node-to-vertex bookkeeping (mesh_repair_bl_thickness.
    # compute_bl_thickness_limit_override) already supports a
    # non-identity mapping via its node_original_vertex/
    # local_surface_faces parameters - built for exactly this
    # possibility even before splitting existed.
    source_vertex = orig_of_node

    normal_faces = topology_faces[real_face_mask]
    normals = compute_face_normals(split_nodes, normal_faces)
    # Geometric extrusion stops at the end of the BL stage - the
    # remaining volume is filled directly from the BL's own real outer
    # surface in one unstructured, graded tetgen pass instead (see this
    # function's own "Fill directly from the BL's own real outer
    # surface" section below, right after the BL prism/export block,
    # for the full rationale - ProjectFiles Part12 P45/P46 and the
    # architecture history that led here).
    bl_nodes, bl_layer_conn = extrude_layers(
        split_nodes, topology_faces, normals,
        bounding_box={'min': bbox_min, 'max': bbox_max},
        growth_rate=growth_rate, min_cell_size=min_cell_size,
        taper_scale=taper_scale, thickness_limit=thickness_limit,
        max_cell_size=max_cell_size,
        bl_layers=bl_layers,
        normal_faces=normal_faces,
    )
    # extrude_layers' own BL layer count - the clip against the actual
    # generated count is kept regardless: the BL stage itself can still
    # stop early (domain boundary/self-collision freeze) before
    # reaching the requested bl_layers.
    # bl_layer_conn has one entry per extrusion STEP (n_layers_generated
    # in extrude_layers' own terms), but bl_nodes (np.vstack'd from
    # extrude_layers' all_layer_nodes) holds n_layers_generated + 1 node
    # blocks - the starting layer-0 block plus one appended per step
    # (see extrude_layers' own all_layer_nodes = [current_nodes] then
    # .append(new_nodes) per step). Using len(bl_layer_conn) directly as
    # the node-layer count is off by one and corrupts every stride
    # derived from it below (nodes_per_layer, bl_split_offset,
    # outer_offset, ...): confirmed directly on cube_demo, where it
    # made node-index arithmetic land BL "layer 1" copies of body-wall
    # nodes on completely unrelated far-field (tunnel-outlet-scale)
    # coordinates instead of a few mm away, producing what looked like
    # shattered/self-intersecting BL geometry (and is almost certainly
    # what the resulting corrupted BL-outer-surface PLC was feeding
    # TetGen's "Recovering segments" hang further downstream).
    n_layers = len(bl_layer_conn) + 1
    _effective_bl_layers = bl_layers if bl_layers is not None else 8
    _effective_bl_layers = int(np.clip(_effective_bl_layers, 0, n_layers - 1))

    nodes_per_layer = len(bl_nodes) // n_layers
    outer_offset = (n_layers - 1) * nodes_per_layer
    bl_split_offset = _effective_bl_layers * nodes_per_layer

    # bl_layer_conn[:_effective_bl_layers] (not +1): convert_layers_to_
    # prisms now internally accounts for layer_connectivity holding one
    # entry per STEP (see its own docstring/fix) - passing the +1 here
    # too would double-count and silently break this call site the
    # same way the transition-tet call site below was broken until
    # that fix (see this project's own investigation: a domain-
    # spanning ~14 m^3 transition tet, not a tetgen defect).
    bl_prisms, bl_face_of_cell = convert_layers_to_prisms(
        bl_nodes[:bl_split_offset + nodes_per_layer],
        bl_layer_conn[:_effective_bl_layers],
        topology_faces,
        min_cell_size=min_cell_size,
    )
    n_bl_cells = len(bl_prisms)
    logger.info(f"  BL mesh: {len(bl_nodes)} nodes, {len(bl_prisms)} prism cells")

    if export_bl_only:
        _export_bl_only_and_exit(
            export_bl_only_path, bl_nodes, bl_prisms, n_bl_cells,
            bl_split_offset, nodes_per_layer, _effective_bl_layers,
        )

    # Attribute each BL cell directly back to its source boundary group
    # via position, not node-index matching against the pre-extrusion
    # surface: convert_layers_to_prisms' own bl_face_of_cell maps
    # every surviving cell back to its extrude_faces row directly (a
    # plain tile - one contiguous block of len(extrude_faces) prisms per
    # layer - no longer holds exactly now that function can drop
    # analytically zero-volume collapsed-layer prisms, see its own
    # docstring). This is exact for every surviving BL cell, including
    # the vast majority of
    # body/ground's own outer surface that node-index matching can never
    # reach (see mesh_boundary.py - those nodes get a brand-new offset
    # index once genuinely displaced by extrusion, so their
    # post-extrusion face can't match anything in a lookup built from
    # the original, pre-extrusion node indices).
    #
    # Only LAYER 0's own prisms (the ones whose bottom cap is the actual
    # physical wall) are tagged with the source group name - every other
    # layer gets '' instead, even though bl_face_of_cell would happily
    # tell us their source face too. This matters concretely, not just
    # cosmetically: a BL column can terminate early at a sharp/complex
    # geometry feature (local thickness cap triggered - see
    # compute_local_thickness_limit), and the LAST surviving prism's own
    # top cap then becomes a legitimate, unavoidable terminal boundary
    # face - a real face, not a bug in face-extraction - but it is NOT
    # the physical wall, it is an artifact of where this specific
    # column happened to stop. Tagging every layer identically (the
    # previous behaviour, unchanged since before true prisms existed)
    # attributed that terminal face to the same "body"/WALL group as the
    # genuine wall, which would wrongly apply a no-slip condition across
    # what should be open interior space. Confirmed as a real, not
    # theoretical, effect on a real case (ProjectFiles Part6/7 P21):
    # 33,448 such faces, concentrated at sharp cube edges, spread across
    # layers 1-3 - NOT at the BL/transition seam as first suspected,
    # confirming this is a pre-existing BL-extrusion characteristic
    # unrelated to the prism/tet split, just never previously isolated
    # from the wall group it was silently merged into.
    #
    # Layer-0 detection is a plain node-index range check, not a return
    # value from convert_layers_to_prisms: layer L's own nodes always
    # occupy bl_nodes[L*nodes_per_layer : (L+1)*nodes_per_layer]
    # (extrude_layers' own node layout, unchanged since before this
    # session), so a prism's bottom cap (v0) being < nodes_per_layer is
    # both necessary and sufficient for "this prism's bottom is layer 0"
    # - no need to plumb a new return value through convert_layers_to_
    # prisms just to re-derive information already implicit in the node
    # indices it returns.
    is_layer0_prism = bl_prisms[:, 0] < nodes_per_layer
    # Ensure bl_face_of_cell is integer type for indexing
    bl_face_of_cell = bl_face_of_cell.astype(np.int64) if not np.issubdtype(bl_face_of_cell.dtype, np.integer) else bl_face_of_cell
    bl_cell_groups = np.where(is_layer0_prism, extrude_face_groups[bl_face_of_cell], '')

    # Layer 0 keeps bare surface-node indices unchanged; the BL's own
    # true final layer (now always the last one extrude_layers
    # actually generated, since bl_only=True) occupies bl_nodes' own
    # last block. core_faces' own node indices are only ever valid
    # against outer_nodes because a seam node shared with core_faces
    # has taper_scale==0 and so never moves off its original (layer-0)
    # position.
    outer_nodes = bl_nodes[outer_offset:outer_offset + nodes_per_layer]
    bl_outer_surface = bl_layer_conn[-1]
    if not np.issubdtype(bl_outer_surface.dtype, np.integer):
        logger.warning(f"bl_outer_surface dtype is {bl_outer_surface.dtype}, converting to int64")
        bl_outer_surface = bl_outer_surface.astype(np.int64)

    # --- Fill directly from the BL's own real outer surface, no
    # separate transition stage at all (neither extruded nor
    # estimated). Tried building a SEPARATE transition-region fill
    # against an ESTIMATED core-side boundary first (a plausible-
    # looking design: protect both interfaces independently) - that
    # estimated surface proved to be a genuinely hard computational-
    # geometry problem on cube_demo's own sharp 90-degree corners (a
    # box has valence-3+ corners everywhere): six different mitigation
    # strategies (plain averaged-normal offset, the same least-squares
    # miter-join direction real BL extrusion uses, multi-step
    # incremental extrusion with mesh_front_collision.py's own proven
    # per-step freeze mechanism, post-hoc shrink/pull-back/local-
    # smoothing repair loops, and finally letting tetgen's own
    # boundary-recovery robustness handling try to fix a still-
    # imperfect estimate) all left SOME residual self-intersection
    # that tetgen's own hard, nobisect-independent input-validity
    # precondition rejects outright (confirmed directly: this holds
    # regardless of nobisect - that switch only governs whether an
    # already-VALID input may be further subdivided for quality, not
    # whether a genuinely self-intersecting input is tolerated at
    # all). This simpler alternative sidesteps the entire problem: no
    # surface needs to be estimated or built at all, since outer_nodes
    # is the REAL, already-extruded BL surface and was independently
    # confirmed self-intersection-free on the same real run (0 hits
    # from mesh_front_collision.find_self_colliding_faces). One
    # unstructured, graded tetgen fill now covers the entire remaining
    # volume (what used to be "transition" is just the near-wall
    # portion of this same graded fill, not a structurally distinct
    # region anymore).
    logger.info(
        f"Step 3/4: Tetrahedralizing core volume "
        f"({len(core_faces)} core-only faces + BL outer surface)..."
    )
    core_plc_points = outer_nodes.copy()
    core_plc_faces = np.vstack([topology_faces, core_faces])

    face_markers = None
    regions = None
    background_points = None
    if max_cell_size is not None:
        # bl_outer_surface's own portion is marked with its source
        # group too (extrude_face_groups) - normally redundant with
        # bl_cell_groups (a BL/core interface face is never itself
        # exposed to the domain exterior), but a column entirely
        # pinned by the seam taper (collapsed to zero BL thickness)
        # has its "outer surface" become the real exposed wall - see
        # attribute_cells_from_trifaces' own caller docs. A facet
        # whose vertices are a MIX of genuinely-grown and early-frozen
        # nodes is left unmarked instead of guessing, falling through
        # to mesh_boundary.py's own UNCLASSIFIED catch-all rather than
        # being silently mis-attributed to the physical wall group.
        bl_outer_markers = np.array(
            [group_name_to_marker.get(n, 0) for n in extrude_face_groups], dtype=np.int32
        )
        core_markers = np.array(
            [group_name_to_marker.get(n, 0) for n in core_face_groups], dtype=np.int32
        )
        face_markers = np.concatenate([bl_outer_markers, core_markers])
        target_edge_length = max_cell_size
        # See this module's own CORE_FILL_VOLUME_CAP_FRACTION comment
        # (top of file) for the tuning history of this value.
        volume_cap_fraction = CORE_FILL_VOLUME_CAP_FRACTION
        regions = [(core_plc_points.mean(axis=0), 1, target_edge_length ** 3 * volume_cap_fraction)]
        background_points = generate_core_background_points(
            core_plc_points, core_plc_faces, target_edge_length
        )
        logger.info(f"TetGen constraint: target_edge_length={target_edge_length:.4f}m, volume_cap={volume_cap_fraction}")

    core_nodes, core_tets, trifaces, triface_markers = fill_core_volume(
        core_plc_points, core_plc_faces, holes=hole_points, regions=regions, face_markers=face_markers,
        background_points=background_points,
        minratio=CORE_TETGEN_MINRATIO, mindihedral=CORE_TETGEN_MINDIHEDRAL,
        force_preserve_boundary=True,
    )
    n_core_boundary = len(core_plc_points)
    if not (len(core_nodes) >= n_core_boundary and np.array_equal(core_nodes[:n_core_boundary], core_plc_points)):
        raise RuntimeError(
            "Core tetgen fill did not preserve its own fixed (real BL "
            "outer surface) boundary verbatim despite "
            "force_preserve_boundary=True - the BL/core splice below "
            "assumes point-for-point preservation and cannot proceed "
            "safely"
        )
    if regions:
        oversized_max_volume = regions[0][2] * OVERSIZED_TET_FACTOR
        core_nodes, core_tets = subdivide_oversized_tetrahedra(
            core_nodes, core_tets, oversized_max_volume
        )
    core_cell_groups = (
        attribute_cells_from_trifaces(core_tets, trifaces, triface_markers, marker_to_name)
        if face_markers is not None
        else np.full(len(core_tets), '', dtype=object)
    )

    if export_core_only:
        path = export_core_only_path
        if not path:
            raise ValueError("export_core_only=True requires export_core_only_path to be set")
        _export_partial_mesh_and_exit(
            core_nodes, np.empty((0, 6), dtype=core_tets.dtype), np.empty(0, dtype=object),
            core_tets, core_cell_groups,
            path, "core-only (tetgen core fill from the real BL outer surface)",
        )

    # Final splice: bl_nodes (BL prisms, unchanged, already in their
    # own global space) + core's own NEW interior points
    # (core_nodes[:n_core_boundary] duplicates outer_nodes, already
    # present in bl_nodes at outer_offset - not re-appended).
    core_remap = np.empty(len(core_nodes), dtype=np.int64)
    core_remap[:n_core_boundary] = np.arange(outer_offset, outer_offset + n_core_boundary)
    core_remap[n_core_boundary:] = len(bl_nodes) + np.arange(len(core_nodes) - n_core_boundary)
    merged_nodes = np.vstack([bl_nodes, core_nodes[n_core_boundary:]])
    core_tets_remapped = core_remap[core_tets]

    # Prisms and tets are kept as two SEPARATE connectivity arrays (see
    # this function's docstring) rather than one vstacked array - a
    # prism's (n,6) shape can't share a row layout with a tet's (n,4)
    # anyway. No separate "transition" cell block exists anymore (see
    # this section's own opening comment) - n_transition_cells is kept
    # at 0 only because generate_hybrid_mesh's own return signature
    # still expects a "how many of merged_cells are near-wall-origin"
    # count; every tet here is core-fill-origin now.
    prism_cells = bl_prisms
    tet_cells = core_tets_remapped
    cell_groups = core_cell_groups
    n_transition_cells = 0
    logger.info(
        f"  Merged mesh: {len(merged_nodes)} nodes, "
        f"{len(prism_cells) + len(tet_cells)} cells "
        f"({len(prism_cells)} BL prisms + {len(tet_cells)} core tets)"
    )

    return (
        merged_nodes, prism_cells, tet_cells, cell_groups, n_bl_cells,
        source_vertex, topology_faces, bl_cell_groups, n_transition_cells,
    )
