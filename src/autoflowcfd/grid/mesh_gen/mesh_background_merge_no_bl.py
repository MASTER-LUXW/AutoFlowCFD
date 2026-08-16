"""_build_merged_mesh 的"无 BL"分支：没有曲面组适合挤出边界层时，
直接对整张封闭输入曲面做一次 tetgen 填充。

从 mesh_background_merge.py 拆分出来（原文件超过 400 行上限），纯粹是
代码搬移——逻辑与 _build_merged_mesh 里原来的 `if len(extrude_faces) == 0:`
分支完全一致，只是把它变成一个独立的模块级函数，由
mesh_background_merge._build_merged_mesh 在该分支下直接调用并原样返回其
结果。
"""

import sys
import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ..structures import BoundaryMap

from .mesh_background_merge_utils import _refine_large_boundary_faces, _export_partial_mesh_and_exit
from .mesh_tetgen_core import (
    fill_core_volume, attribute_cells_from_trifaces, generate_core_background_points,
    subdivide_oversized_tetrahedra,
    CORE_TETGEN_MINRATIO, CORE_TETGEN_MINDIHEDRAL, CORE_VOLUME_CAP_FRACTION,
)


def _build_merged_mesh_no_bl(
    surface_nodes: np.ndarray,
    surface_faces: np.ndarray,
    surface_boundaries: 'BoundaryMap',
    extrude_faces: np.ndarray,
    hole_points,
    max_cell_size: Optional[float],
    group_name_to_marker: dict,
    marker_to_name: dict,
    export_core_only: bool,
    export_core_only_path: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, int]:
    """没有任何曲面组适合挤出边界层时，直接用 tetgen 填充整张封闭曲面。

    对应 mesh_background_merge._build_merged_mesh 里原来的
    `if len(extrude_faces) == 0:` 分支，逐字搬移，未改动任何数值逻辑。
    extrude_faces 在这个分支下总是空数组，直接传入是为了让
    `topology_faces = extrude_faces` 这一步和原代码逐字节一致（而不是
    在这里重新构造一个"看起来等价"的空数组）。

    Returns:
        与 _build_merged_mesh 自身完全相同的 9 元组：
        (merged_nodes, prism_cells, tet_cells, cell_groups, n_bl_cells,
        source_vertex, topology_faces, bl_cell_groups, n_transition_cells)
    """
    # OVERSIZED_TET_FACTOR 定义在 mesh_background_merge.py（本函数唯一
    # 的调用者所在文件），延迟导入以避免循环导入——本模块被
    # mesh_background_merge.py 导入，不能在模块顶层反向导入它。
    from .mesh_background_merge import OVERSIZED_TET_FACTOR

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

    return (
        merged_nodes, prism_cells, tet_cells, cell_groups, n_bl_cells,
        source_vertex, topology_faces, bl_cell_groups, n_transition_cells,
    )
