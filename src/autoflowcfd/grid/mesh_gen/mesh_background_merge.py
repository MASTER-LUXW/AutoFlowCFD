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

两个和 _build_merged_mesh 自身逻辑无关、只是被它调用的独立工具函数
（边界面按最大边长细分、`--*-only` 调试导出）拆到了同目录
mesh_background_merge_utils.py。

_build_merged_mesh 本身的两条分支（没有任何曲面组适合挤出边界层 / 至少
一个曲面组适合挤出边界层）进一步拆到了 mesh_background_merge_no_bl.py
和 mesh_background_merge_with_bl.py（原文件超过 400 行上限）——本文件
只保留两条分支共用的前置步骤（曲面分类、facet marker 映射表构建）和分
发逻辑。两个常量（OVERSIZED_TET_FACTOR、CORE_FILL_VOLUME_CAP_FRACTION）
被两个分支模块延迟导入使用，因此仍留在本文件里。
"""

import numpy as np
from typing import Dict, Optional, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ..structures import BoundaryMap

from .mesh_domain_classify import classify_boundary_groups
from .mesh_repair import compute_bl_thickness_limit_override

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
        from .mesh_background_merge_no_bl import _build_merged_mesh_no_bl
        return _build_merged_mesh_no_bl(
            surface_nodes, surface_faces, surface_boundaries, extrude_faces,
            hole_points, max_cell_size, group_name_to_marker, marker_to_name,
            export_core_only, export_core_only_path,
        )
    else:
        from .mesh_background_merge_with_bl import _build_merged_mesh_with_bl
        return _build_merged_mesh_with_bl(
            surface_nodes, surface_boundaries, bbox_min, bbox_max,
            extrude_faces, core_faces, extruded_groups, extrude_face_groups,
            hole_points, core_face_groups, group_name_to_marker, marker_to_name,
            growth_rate, min_cell_size, max_cell_size, extra_thickness_limit, bl_layers,
            export_bl_only, export_bl_only_path, export_core_only, export_core_only_path,
        )
