"""Merged BL + tetgen-core mesh assembly for one generation attempt.

_build_merged_mesh does the actual per-attempt work generate_hybrid_mesh
(mesh_background.py) orchestrates: classify boundary groups, extrude BL
layers, tetgen-fill the remaining core volume, and splice the two into one
merged (nodes, cells) pair with per-cell source-group attribution. Split
into its own module purely to keep mesh_background.py's own file size
down - there is no independent reuse of this function outside
generate_hybrid_mesh's own retry loop (Stage B), which is why it stays
private (leading underscore) and lives right next to its only caller's
module.
"""

import numpy as np
from typing import Dict, NamedTuple, Optional, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ..structures import BoundaryMap

from .mesh_extrusion import extrude_layers
from .mesh_prism_to_tet import convert_layers_to_prisms, orient_tetrahedra
from .mesh_utils import compute_face_normals
from .mesh_domain_classify import classify_boundary_groups
from .mesh_corner_split import split_sharp_corners
from .mesh_tetgen_core import (
    build_seam_taper_scale, fill_core_volume,
    compute_local_thickness_limit, repair_nonmanifold_cells,
    attribute_cells_from_trifaces, generate_core_background_points,
    subdivide_oversized_tetrahedra,
    CORE_TETGEN_MINRATIO, CORE_TETGEN_MINDIHEDRAL, CORE_VOLUME_CAP_FRACTION,
)

# Deterministic backstop for tetgen's own volume refinement missing a
# well-shaped-but-oversized cell entirely (see subdivide_oversized_
# tetrahedra's own docstring for why this happens and why centroid
# subdivision is used). Multiplied against each call site's own
# region-target maxvol (not applied at exactly that target) so ordinary,
# expected coarse-but-legitimate grading near the target isn't churned -
# only genuine outliers (measured directly at 100x-16,000x the target) get
# split.
OVERSIZED_TET_FACTOR = 5.0

# How large a core tet is allowed to grow (as a fraction of max_cell_size**3)
# in the main "fill directly from the BL's own real outer surface" branch
# below - a single FLAT cap applied to the whole core region (tetgen's own
# distance-graded background-mesh/metric sizing segfaults in this
# environment, see fill_core_volume's own `regions` doc), so this is the
# only lever available for how abrupt the BL-outer-surface-to-core size
# jump looks. Deliberately its OWN constant, not
# mesh_tetgen_core.CORE_VOLUME_CAP_FRACTION (0.08) - that one is tuned for
# Stage B''s small local cavity retiles, a different workload with its own
# rationale (see that constant's own docstring), and reusing it here at
# first (0.08) made the transition noticeably too slow/fine-grained
# (excess core cell count) for this much larger single call; 0.2 was then
# still a bit slow/fine, 0.3 a bit too fast/coarse. 0.25 is the current
# middle ground - adjust directly here if it still isn't right.
CORE_FILL_VOLUME_CAP_FRACTION = 0.25
from .mesh_repair import compute_bl_thickness_limit_override


def _refine_large_boundary_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    markers: Optional[np.ndarray],
    max_edge_length: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Iteratively bisect edges longer than max_edge_length on the boundary surface.
    
    This prevents TetGen from generating huge boundary tets that violate the
    target cell size constraint. It also helps match the resolution of the
    BL outer surface to the core fill and improves compatibility with ANSA's
    mesh quality standards through more precise edge length control.
    """
    if max_edge_length <= 0:
        return vertices, faces, markers

    current_verts = vertices.copy()
    current_faces = faces.copy()
    current_markers = markers.copy() if markers is not None else None
    
    max_iterations = 10
    max_total_vertices = len(vertices) * 5  # Prevent memory explosion
    
    for iteration in range(max_iterations):
        if len(current_verts) > max_total_vertices:
            logger.warning(f"Boundary refinement stopped: vertex count ({len(current_verts)}) exceeded limit ({max_total_vertices})")
            break

        # Vectorized edge length calculation
        v0 = current_verts[current_faces[:, 0]]
        v1 = current_verts[current_faces[:, 1]]
        v2 = current_verts[current_faces[:, 2]]
        
        e01 = np.linalg.norm(v1 - v0, axis=1)
        e12 = np.linalg.norm(v2 - v1, axis=1)
        e20 = np.linalg.norm(v0 - v2, axis=1)
        
        max_edges = np.maximum.reduce([e01, e12, e20])
        needs_refinement_mask = max_edges > max_edge_length
        
        if not np.any(needs_refinement_mask):
            logger.info(f"Boundary refinement completed after {iteration} iterations. "
                        f"Faces: {len(faces)} -> {len(current_faces)}, "
                        f"Vertices: {len(vertices)} -> {len(current_verts)}")
            return current_verts, current_faces, current_markers
            
        # Find the longest edge for each face that needs refinement
        split_indices = np.argmax(np.stack([e01, e12, e20], axis=1)[needs_refinement_mask], axis=1)
        faces_to_split = current_faces[needs_refinement_mask]
        markers_to_split = current_markers[needs_refinement_mask] if current_markers is not None else None
        
        new_faces_list = []
        new_markers_list = []
        new_vertices_list = []
        
        # Process splits in batches to avoid index shifting issues
        # We need to map old vertex indices to new ones
        vertex_offset = len(current_verts)
        
        for i, (face, split_idx) in enumerate(zip(faces_to_split, split_indices)):
            v0, v1, v2 = face
            if split_idx == 0: # Split 0-1
                mid_coord = (current_verts[v0] + current_verts[v1]) / 2.0
                mid_idx = vertex_offset + i
                f1, f2 = [mid_idx, v1, v2], [v0, mid_idx, v2]
            elif split_idx == 1: # Split 1-2
                mid_coord = (current_verts[v1] + current_verts[v2]) / 2.0
                mid_idx = vertex_offset + i
                f1, f2 = [v0, mid_idx, v2], [v0, v1, mid_idx]
            else: # Split 2-0
                mid_coord = (current_verts[v2] + current_verts[v0]) / 2.0
                mid_idx = vertex_offset + i
                f1, f2 = [v0, v1, mid_idx], [mid_idx, v1, v2]
            
            new_faces_list.extend([f1, f2])
            new_vertices_list.append(mid_coord)
            if current_markers is not None:
                new_markers_list.extend([markers_to_split[i], markers_to_split[i]])
        
        # Add new vertices
        if new_vertices_list:
            current_verts = np.vstack([current_verts, np.array(new_vertices_list)])
        
        # Replace split faces with new ones
        remaining_faces = current_faces[~needs_refinement_mask]
        remaining_markers = current_markers[~needs_refinement_mask] if current_markers is not None else None
        
        if new_faces_list:
            current_faces = np.vstack([remaining_faces, np.array(new_faces_list, dtype=np.int32)]) if len(remaining_faces) > 0 else np.array(new_faces_list, dtype=np.int32)
            if current_markers is not None:
                current_markers = np.concatenate([remaining_markers, np.array(new_markers_list)]) if len(remaining_markers) > 0 else np.array(new_markers_list)
        else:
            current_faces = remaining_faces
            if current_markers is not None:
                current_markers = remaining_markers
                
        logger.info(f"  Refinement iteration {iteration+1}: Split {len(faces_to_split)} faces, added {len(new_vertices_list)} vertices")

    logger.info(f"Boundary refinement completed after {max_iterations} iterations (limit reached). "
                f"Faces: {len(faces)} -> {len(current_faces)}, "
                f"Vertices: {len(vertices)} -> {len(current_verts)}")
    return current_verts, current_faces, current_markers


def _export_partial_mesh_and_exit(
    nodes: np.ndarray,
    prism_cells: np.ndarray,
    prism_groups: np.ndarray,
    tet_cells: np.ndarray,
    tet_groups: np.ndarray,
    output_path: str,
    label: str,
) -> None:
    """Export a partial (BL-only/transition-only/core-only) debug mesh and
    exit the process - shared by every `--*-only` CLI flag's early-stop
    path (see cli/grid_commands.py's own `--bl-only`/`--trans-only`/
    `--core-only`). These exist to let a real, generated mesh from any one
    pipeline stage be inspected directly in a mesh viewer (ANSA etc.) -
    this session's own investigation into the BL/transition-to-core-fill
    interface repeatedly needed exactly this and had no reusable way to
    get it short of ad-hoc scripts each time.

    Every distinct non-empty group name in `prism_groups`/`tet_groups`
    becomes its own WALL boundary group; every '' (unattributed - e.g. a
    mid-stack cell with no exposed named face) is lumped into a single
    catch-all 'INTERFACE' group instead of being silently dropped, so the
    exported file always has a complete boundary partition to open.

    Args:
        nodes: (n_nodes, 3) node coordinates, meters
        prism_cells: (n_prism, 6) prism connectivity, or empty (0, 6)
        prism_groups: (n_prism,) str array parallel to prism_cells
        tet_cells: (n_tet, 4) tet connectivity, or empty (0, 4)
        tet_groups: (n_tet,) str array parallel to tet_cells
        output_path: where to write the .nas file
        label: human-readable name for this stage, used only in log lines
    """
    import sys
    from ..nas_io.nas_export import export_volume_mesh_to_nas
    from ..structures import NodeArray, PrismCells, TetrahedralCells, BoundaryMap, GridMetadata, VolumeMeshData

    logger.success(f"Exporting {label} mesh to: {output_path}")
    try:
        nodes_obj = NodeArray(x=nodes[:, 0].copy(), y=nodes[:, 1].copy(), z=nodes[:, 2].copy())

        n_prism = len(prism_cells)
        prism_cells_obj = None
        if n_prism:
            prism_volumes = PrismCells.compute_volumes(nodes_obj, prism_cells.astype(np.int32))
            prism_cells_obj = PrismCells(connectivity=prism_cells.astype(np.int32), volumes=prism_volumes)

        tet_cells32 = tet_cells.astype(np.int32)
        tet_volumes = (
            TetrahedralCells.compute_volumes(nodes_obj, tet_cells32)
            if len(tet_cells32) else np.empty(0, dtype=np.float64)
        )
        cells_obj = TetrahedralCells(connectivity=tet_cells32, volumes=tet_volumes)

        groups: Dict[str, np.ndarray] = {}
        bc_types: Dict[str, str] = {}
        interface_parts = []
        if n_prism:
            for name in np.unique(prism_groups):
                idx = np.flatnonzero(prism_groups == name).astype(np.int32)
                if name:
                    groups[name] = idx
                    bc_types[name] = 'WALL'
                else:
                    interface_parts.append(idx)
        if len(tet_cells32):
            for name in np.unique(tet_groups):
                idx = (np.flatnonzero(tet_groups == name) + n_prism).astype(np.int32)
                if name:
                    if name in groups:
                        groups[name] = np.union1d(groups[name], idx).astype(np.int32)
                    else:
                        groups[name] = idx
                        bc_types[name] = 'WALL'
                else:
                    interface_parts.append(idx)
        if interface_parts:
            groups['INTERFACE'] = np.concatenate(interface_parts).astype(np.int32)
            bc_types['INTERFACE'] = 'INTERFACE'

        boundaries_obj = BoundaryMap(groups=groups, bc_types=bc_types)
        metadata = GridMetadata(
            node_count=len(nodes), cell_count=n_prism + len(tet_cells32),
            boundary_groups=list(groups.keys()), file_format="nas",
        )
        vol_mesh = VolumeMeshData(
            nodes=nodes_obj, cells=cells_obj, boundaries=boundaries_obj,
            metadata=metadata, prism_cells=prism_cells_obj,
        )
        # export_volume_mesh_to_nas expects meters and converts to mm.
        export_volume_mesh_to_nas(vol_mesh, output_path, scale_factor=1000.0)
        logger.success(f"{label} mesh exported successfully.")
    except Exception as e:
        logger.error(f"Failed to export {label} mesh: {e}")
        import traceback
        traceback.print_exc()

    sys.exit(0)


class _BuildResult(NamedTuple):
    merged_nodes: np.ndarray
    prism_cells: np.ndarray
    tet_cells: np.ndarray
    cell_groups: np.ndarray
    n_bl_prisms: int
    source_vertex: np.ndarray
    bl_extrude_faces: np.ndarray
    bl_cell_groups: np.ndarray
    n_transition_cells: int


def _build_merged_mesh(
    surface_nodes: np.ndarray,
    surface_faces: np.ndarray,
    bounding_box: Dict[str, np.ndarray],
    surface_boundaries: 'BoundaryMap',
    growth_rate: float = 1.2,
    min_cell_size: float = 0.001,
    max_cell_size: Optional[float] = None,
    extra_thickness_limit: Optional[np.ndarray] = None,
    bl_layers: Optional[int] = None,
    export_bl_only: bool = False,
    export_bl_only_path: Optional[str] = None,
    export_core_only: bool = False,
    export_core_only_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build the merged mesh (BL prisms + TetGen core tets)."""
    bbox_min = np.asarray(bounding_box['min'], dtype=np.float64)
    bbox_max = np.asarray(bounding_box['max'], dtype=np.float64)

    # Note: surface_nodes are already in meters at this point (converted by
    # NASParser.parse). max_cell_size from CLI is also in meters. No scaling
    # is needed.

    logger.info("Step 1/4: Classifying boundary groups (extrude vs. core-only)...")
    (extrude_faces, core_faces, extruded_groups, extrude_face_groups,
     hole_points, core_face_groups, _is_closed_solid_face) = classify_boundary_groups(
        surface_nodes, surface_faces, surface_boundaries, bbox_min, bbox_max
    )

    # Marker IDs for tetgen's facet-marker mechanism (attribute_cells_from_trifaces)
    # - only needed when max_cell_size grading is active (it's the only thing
    # that switches fill_core_volume's nobisect off, which is what breaks the
    # plain node-index-matching boundary attribution for subdivided facets).
    # 0 is reserved by tetgen for "no marker" (an unmarked/interior facet).
    group_name_to_marker = {name: i + 1 for i, name in enumerate(surface_boundaries.groups.keys())}
    marker_to_name = {v: k for k, v in group_name_to_marker.items()}

    if len(extrude_faces) == 0:
        logger.warning(
            "No boundary group was eligible for BL extrusion; filling the "
            "entire closed surface directly with tetgen (no boundary layer)"
        )
        n_bl_cells = 0
        source_vertex = np.arange(len(surface_nodes))
        topology_faces = extrude_faces  # empty - no corner-splitting to do with no BL region
        face_markers = None
        regions = None
        
        # Prepare markers and regions if max_cell_size is set
        if max_cell_size is not None:
            face_group_name = np.full(len(surface_faces), '', dtype=object)
            for name, idx in surface_boundaries.groups.items():
                face_group_name[idx] = name
            face_markers = np.array(
                [group_name_to_marker.get(n, 0) for n in face_group_name], dtype=np.int32
            )
            center = surface_nodes.mean(axis=0)
            # max_cell_size is already in meters, matching surface_nodes
            target_edge_length = max_cell_size
            
            # Refine large boundary faces before TetGen
            logger.info(f"Refining boundary faces with max edge length > {target_edge_length:.4f}m...")
            proc_nodes, proc_faces, face_markers = _refine_large_boundary_faces(
                surface_nodes, surface_faces, face_markers, target_edge_length
            )
            
            regions = [(center, 1, target_edge_length ** 3 * CORE_VOLUME_CAP_FRACTION)]
            background_points = generate_core_background_points(
                proc_nodes, proc_faces, target_edge_length
            )
        else:
            proc_nodes, proc_faces = surface_nodes, surface_faces
            background_points = None

        core_nodes, core_tets, trifaces, triface_markers = fill_core_volume(
            proc_nodes, proc_faces, holes=hole_points,
            regions=regions, face_markers=face_markers,
            background_points=background_points,
            minratio=CORE_TETGEN_MINRATIO, mindihedral=CORE_TETGEN_MINDIHEDRAL,
        )
        if regions:
            oversized_max_volume = regions[0][2] * OVERSIZED_TET_FACTOR
            core_nodes, core_tets = subdivide_oversized_tetrahedra(
                core_nodes, core_tets, oversized_max_volume
            )
        merged_nodes, tet_cells = core_nodes, core_tets
        prism_cells = np.zeros((0, 6), dtype=np.int64)
        bl_cell_groups = np.zeros(0, dtype=object)
        n_transition_cells = 0
        if face_markers is not None:
            cell_groups = attribute_cells_from_trifaces(
                core_tets, trifaces, triface_markers, marker_to_name
            )
        else:
            cell_groups = np.full(len(tet_cells), '', dtype=object)

        if export_core_only:
            if not export_core_only_path:
                raise ValueError("export_core_only=True requires export_core_only_path to be set")
            _export_partial_mesh_and_exit(
                merged_nodes, prism_cells, bl_cell_groups, tet_cells, cell_groups,
                export_core_only_path, "core-only (no BL region - this is the whole mesh)",
            )
    else:
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
            if not export_bl_only_path:
                raise ValueError("export_bl_only=True requires export_bl_only_path to be set")
            logger.success(f"Exporting BL-only mesh to: {export_bl_only_path}")

            try:
                from ..nas_io.nas_export import export_volume_mesh_to_nas
                from ..structures import NodeArray, PrismCells, BoundaryMap, GridMetadata, VolumeMeshData, TetrahedralCells

                # bl_nodes carries every extruded layer (BL + transition
                # stage), but bl_prisms only indexes the BL-stage prefix of
                # them (see the convert_layers_to_prisms call above, sliced
                # to bl_split_offset + nodes_per_layer) - keep only that
                # prefix here too, or the export ends up with a trailing
                # block of orphan GRID nodes no CPENTA references.
                used_node_count = bl_split_offset + nodes_per_layer
                export_nodes = bl_nodes[:used_node_count]
                nodes_obj = NodeArray(
                    x=export_nodes[:, 0].copy(), y=export_nodes[:, 1].copy(), z=export_nodes[:, 2].copy()
                )

                # Create dummy tet cells to satisfy VolumeMeshData structure
                dummy_tets = np.empty((0, 4), dtype=np.int32)
                dummy_vols = np.empty(0, dtype=np.float64)
                cells_obj = TetrahedralCells(connectivity=dummy_tets, volumes=dummy_vols)

                prism_volumes = PrismCells.compute_volumes(nodes_obj, bl_prisms.astype(np.int32))
                prisms_obj = PrismCells(connectivity=bl_prisms.astype(np.int32), volumes=prism_volumes)

                # A BL-only export has no core tet mesh past the outermost
                # layer, so BOTH the true wall (layer 0) and the BL/core
                # interface (the last layer generated here) are "exterior"
                # faces of this prism block - lumping every prism into one
                # group made _extract_boundary_faces_by_group export both
                # surfaces under the same WALL tag (two overlapping shells).
                # Split them the same way the non-bl-only path already
                # distinguishes layer 0 (see is_layer0_prism a few lines
                # below this block): a prism's v0 is always its own bottom
                # layer's node, and layer L's nodes always occupy
                # bl_nodes[L*nodes_per_layer : (L+1)*nodes_per_layer].
                cell_layer = bl_prisms[:, 0] // nodes_per_layer
                wall_mask = cell_layer == 0
                interface_mask = cell_layer == (_effective_bl_layers - 1)

                dummy_groups = {
                    'BL_Wall': np.flatnonzero(wall_mask).astype(np.int32),
                    'BL_Interface': np.flatnonzero(interface_mask).astype(np.int32),
                }
                dummy_bc = {'BL_Wall': 'WALL', 'BL_Interface': 'INTERFACE'}
                boundaries_obj = BoundaryMap(groups=dummy_groups, bc_types=dummy_bc)

                metadata = GridMetadata(
                    node_count=len(export_nodes),
                    cell_count=n_bl_cells,
                    boundary_groups=list(dummy_groups.keys()),
                    file_format="nas"
                )

                vol_mesh = VolumeMeshData(
                    nodes=nodes_obj,
                    cells=cells_obj,
                    boundaries=boundaries_obj,
                    metadata=metadata,
                    prism_cells=prisms_obj,
                )

                # Note: export_volume_mesh_to_nas expects meters and converts to mm (scale_factor=1000)
                export_volume_mesh_to_nas(vol_mesh, export_bl_only_path, scale_factor=1000.0)
                logger.success(f"BL-only mesh exported successfully.")
            except Exception as e:
                logger.error(f"Failed to export BL mesh: {e}")
                import traceback
                traceback.print_exc()

            import sys
            sys.exit(0)

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
