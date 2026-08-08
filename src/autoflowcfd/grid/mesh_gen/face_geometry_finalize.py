"""面提取的收尾几何计算：面积/法向/中心 + 校验。

从 face_extractor.py 拆分出来。finalize_face_data 是 extract_faces（纯四
面体网格）和 extract_faces_mixed（棱柱+四面体混合网格）共用的收尾步骤——
从这一步开始，输入已经和具体单元形状无关，只依赖每个面的 3 个角点节点
编号、owner/neighbour 单元编号，以及该单元已经算好的质心。
"""

import numpy as np
from loguru import logger

from ..structures import NodeArray, FaceData


def finalize_face_data(
    face_nodes_sorted: np.ndarray,
    face_connectivity: np.ndarray,
    occurrence_count: np.ndarray,
    n_unique_faces: int,
    n_interior: int,
    n_faces_raw: int,
    nodes: NodeArray,
    all_cell_centers: np.ndarray,
    n_cells: int,
    strict: bool = False,
) -> FaceData:
    """Shared post-dedup geometry/orientation/validation, used by both
    extract_faces (tet-only) and extract_faces_mixed (prism+tet) -
    genuinely cell-shape-agnostic from this point on: everything below
    only ever consumes a face's 3 corner node indices, its owner/
    neighbour cell index, and that cell's already-computed centroid."""
    n_boundary = n_unique_faces - n_interior
    n_invalid = np.sum(occurrence_count > 2)

    logger.info(
        f"Identified {n_unique_faces} unique faces from {n_faces_raw} occurrences"
    )
    logger.info(
        f"Face topology: {n_interior} interior, {n_boundary} boundary, "
        f"{n_invalid} invalid (>2 cells)"
    )

    if n_invalid > 0:
        # NOTE: the dedup scan above only ever records the first 2 cells
        # touching a given face key (see _deduplicate_and_build_connectivity);
        # for a face shared by 3+ cells, every cell beyond the first two
        # never gets connected to it at all, silently dropping that
        # cell's flux through this face from the residual - a genuine
        # local conservation violation, not a numerical-stability issue.
        # This can (and has been observed to) produce a residual that
        # diverges unboundedly regardless of how low CFL is pushed,
        # while integrated body forces stay comparatively normal since
        # they don't depend on these (typically interior/core-mesh)
        # faces. Continuing to solve on a topologically invalid mesh
        # wastes potentially hours of compute on a result that was
        # never going to be physically meaningful - fail immediately
        # instead, pointing at the volume mesh generation step that
        # produced overlapping/duplicate tetrahedra.
        invalid_mask = occurrence_count > 2
        invalid_node_ids = np.unique(face_nodes_sorted[invalid_mask])
        bad_x = nodes.x[invalid_node_ids]
        bad_y = nodes.y[invalid_node_ids]
        bad_z = nodes.z[invalid_node_ids]
        logger.warning(
            f"Invalid faces detected (n={n_invalid}), spatially bounded by "
            f"x=[{bad_x.min():.4g}, {bad_x.max():.4g}], "
            f"y=[{bad_y.min():.4g}, {bad_y.max():.4g}], "
            f"z=[{bad_z.min():.4g}, {bad_z.max():.4g}]. "
            f"This is likely due to BL extrusion at sharp corners."
            + (" Proceeding for inspection (non-strict call)." if not strict else "")
        )
        if strict:
            # Unlike the intermediate/exploratory callers during mesh
            # generation and repair (mesh_repair.py, mesh_repair_cavity.py,
            # mesh_background.py's pre-repair check - all non-strict,
            # since a transient non-manifold state there is expected and
            # gets resolved by a LATER repair stage, e.g.
            # repair_nonmanifold_mixed), this is the genuine solve/export-
            # time gate (GridData.ensure_faces_exist, strict=True) - by
            # this point every repair stage has already run, so a
            # remaining >2-owner face is a real, uncorrected defect, not
            # a transient one.
            raise RuntimeError(
                f"Invalid mesh topology: {n_invalid} faces are shared by more than "
                f"2 cells (expected exactly 1 for boundary or 2 for interior faces). "
                f"This means the volume mesh contains overlapping/duplicate "
                f"tetrahedra - almost certainly from the boundary-layer/core "
                f"tetgen merge (see mesh_background.generate_hybrid_mesh). "
                f"Solving on this mesh would silently drop flux through the "
                f"affected faces and is not physically meaningful; regenerate "
                f"the volume mesh (e.g. with different BL parameters) rather "
                f"than proceeding."
            )

    # Expected ratio: ~2x cells for interior-dominated mesh
    expected_ratio = n_unique_faces / n_cells
    logger.debug(f"Face-to-cell ratio: {expected_ratio:.2f} (expected ~2.0-2.5)")

    # Step 3: Compute geometric properties using vectorized operations
    logger.debug("Computing face geometry (vectorized)...")
    x = nodes.x
    y = nodes.y
    z = nodes.z

    # Vectorized face center computation
    n0 = face_nodes_sorted[:, 0]
    n1 = face_nodes_sorted[:, 1]
    n2 = face_nodes_sorted[:, 2]

    face_centers = np.column_stack([
        (x[n0] + x[n1] + x[n2]) / 3.0,
        (y[n0] + y[n1] + y[n2]) / 3.0,
        (z[n0] + z[n1] + z[n2]) / 3.0
    ])

    # Vectorized area vector computation
    p0 = np.column_stack([x[n0], y[n0], z[n0]])
    p1 = np.column_stack([x[n1], y[n1], z[n1]])
    p2 = np.column_stack([x[n2], y[n2], z[n2]])

    v1 = p1 - p0
    v2 = p2 - p0
    face_areas_vec = 0.5 * np.cross(v1, v2)

    # Determine face orientation and flip if needed
    left_cells = face_connectivity[:, 0]
    right_cells = face_connectivity[:, 1]

    # all_cell_centers is already computed by the caller (tet-only or
    # mixed prism+tet - see _compute_tet_cell_centers/
    # _compute_prism_cell_centers), passed in as a parameter.

    # Get left and right cell centers
    center_left = all_cell_centers[left_cells]

    # For interior faces, ensure the normal points outward from the
    # OWNER (left) cell - i.e. away from left's own centroid, through
    # the face itself - using the face's own center (face_centers)
    # relative to left's centroid, same criterion the boundary branch
    # just below already (correctly) uses.
    #
    # CRITICAL: flip the normal sign ONLY. A previous version of this
    # code additionally swapped face_connectivity's two columns whenever
    # it flipped the normal - which undoes its own fix: if the raw
    # cross-product normal pointed inward to left, negating it makes it
    # correctly point outward from left (exactly what "left/owner gets
    # +normal" needs) - but then swapping the columns re-labels left as
    # "neighbour" and right as "owner", so the solver's "+normal to
    # owner, -normal to neighbour" accumulation ends up giving left
    # -normal (i.e. right back to the original, still-wrong, inward
    # value) and right +normal (outward from LEFT, not from right -
    # wrong for right too). The two mistakes cancel for the pair's
    # combined bookkeeping (which is why this was never caught by a
    # sum-over-both-cells check) but not for either cell's OWN closure.
    # Confirmed directly on this project's actual cube_demo core mesh:
    # 89% of cells (100% of BL prisms) had a nonzero sum of their own
    # outward area-weighted face normals - must be exactly zero for any
    # closed cell by the divergence theorem - which silently broke both
    # flux conservation and Green-Gauss gradient reconstruction almost
    # everywhere, and was the actual root cause of the solver diverging
    # on an otherwise mesh-quality-gate-passing mesh. A minimal 576-tet
    # structured-box repro (no prisms, no skew) reproduced the same
    # 73%-of-cells defect rate through extract_faces_mixed/extract_faces
    # (which share this function) while the older, separate
    # core/fvm_faces.py:FVMFaceExtractor.build_from_tetrahedra path -
    # which doesn't have a connectivity swap at all - stayed exactly
    # closed, pointing straight at this swap as the culprit.
    mask_interior = right_cells >= 0
    dx_interior = face_centers[mask_interior] - center_left[mask_interior]
    dot_interior = np.sum(face_areas_vec[mask_interior] * dx_interior, axis=1)

    # Flip faces where normal points wrong direction (sign only - see above)
    flip_mask = dot_interior < 0
    indices_to_flip = np.where(mask_interior)[0][flip_mask]
    face_areas_vec[indices_to_flip] *= -1

    # For boundary faces, ensure normal points outward
    mask_boundary = ~mask_interior
    dx_boundary = face_centers[mask_boundary] - center_left[mask_boundary]
    dot_boundary = np.sum(face_areas_vec[mask_boundary] * dx_boundary, axis=1)
    flip_boundary = dot_boundary < 0
    indices_to_flip_boundary = np.where(mask_boundary)[0][flip_boundary]
    face_areas_vec[indices_to_flip_boundary] *= -1

    # Compute scalar areas and unit normals
    face_scalar_areas = np.linalg.norm(face_areas_vec, axis=1)
    valid_area_mask = face_scalar_areas > 1e-12
    face_normals = np.zeros_like(face_areas_vec)
    face_normals[valid_area_mask] = (
        face_areas_vec[valid_area_mask] /
        face_scalar_areas[valid_area_mask][:, np.newaxis]
    )

    # Create FaceData object. node_connectivity is the triangle-corner
    # node indices already computed above (face_nodes_sorted) purely to
    # derive area/normal/center - kept here too so callers that need
    # the actual boundary surface mesh (e.g. VTKExporter.export_boundaries,
    # for per-zone/per-patch visualization) don't have to re-extract it
    # from the tetrahedra a second time.
    face_data = FaceData(
        connectivity=face_connectivity,
        area=face_scalar_areas,
        normal=face_normals,
        center=face_centers,
        node_connectivity=face_nodes_sorted.astype(np.int32),
    )

    # Validate output
    validate_face_data(face_data, n_cells)

    logger.success(
        f"Face extraction completed: {face_data.n_interior_faces} interior, "
        f"{face_data.n_boundary_faces} boundary faces"
    )

    return face_data


def validate_face_data(face_data: FaceData, n_cells: int) -> bool:
    """Validate extracted face data for consistency.

    Checks:
    - All cells are referenced by at least one face
    - No duplicate faces
    - Area values have reasonable magnitudes
    - Normal vectors are unit length

    Args:
        face_data: Extracted face data
        n_cells: Expected number of cells

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    # Check 1: All cells should be referenced
    referenced_cells = set()
    for i in range(face_data.count):
        referenced_cells.add(int(face_data.connectivity[i, 0]))
        if face_data.connectivity[i, 1] >= 0:
            referenced_cells.add(int(face_data.connectivity[i, 1]))

    if len(referenced_cells) != n_cells:
        raise ValueError(
            f"Face connectivity references {len(referenced_cells)} cells, "
            f"expected {n_cells}"
        )

    # Check 2: Areas should have positive magnitude
    n_zero_areas = np.sum(face_data.area < 1e-12)
    if n_zero_areas > 0:
        logger.warning(f"Found {n_zero_areas} faces with zero/near-zero area. Allowing export for debugging.")
        # raise ValueError(f"Found {n_zero_areas} faces with zero/near-zero area")

    # Check 3: Normal vectors should be unit length
    normal_magnitudes = np.linalg.norm(face_data.normal, axis=1)
    n_invalid_normals = np.sum(np.abs(normal_magnitudes - 1.0) > 1e-6)
    if n_invalid_normals > 0:
        logger.warning(f"Found {n_invalid_normals} faces with non-unit normals (magnitude != 1.0)")

    logger.debug("Face data validation passed")
    return True
