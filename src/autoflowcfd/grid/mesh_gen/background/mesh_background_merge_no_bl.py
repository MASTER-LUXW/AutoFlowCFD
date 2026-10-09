"""_build_merged_mesh 的"无 BL"分支：没有曲面组适合挤出边界层时，
直接对整张封闭输入曲面做一次 tetgen 填充。

从 mesh_background_merge.py 拆分出来（原文件超过 400 行上限），纯粹是
代码搬运——逻辑与 _build_merged_mesh 里原来的 `if len(extrude_faces) == 0:`
分支完全一致，只是把它变成一个独立的模块级函数，由
mesh_background_merge._build_merged_mesh 在该分支下直接调用并原样返回
结果。
"""

import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ...schema.grid_boundaries import BoundaryMap

from .mesh_background_merge_utils import _refine_large_boundary_faces, _export_partial_mesh_and_exit
from ..tetgen.mesh_tetgen_core import (
    fill_core_volume, generate_core_background_points,
    subdivide_oversized_tetrahedra,
    CORE_TETGEN_MINRATIO, CORE_TETGEN_MINDIHEDRAL, CORE_VOLUME_CAP_FRACTION,
)


def _build_merged_mesh_no_bl(
    surface_nodes: np.ndarray,
    surface_faces: np.ndarray,
    surface_boundaries: 'BoundaryMap',
    hole_points,
    max_cell_size: Optional[float],
    export_core_only: bool,
    export_core_only_path: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """没有曲面组适合挤出边界层时，直接用 tetgen 填充整张封闭曲面。

    对应 mesh_background_merge._build_merged_mesh 里原来的
    `if len(extrude_faces) == 0:` 分支，逐字搬运，未改动任何数值逻辑。

    Returns:
        (merged_nodes, prism_cells, tet_cells)，与 _build_merged_mesh 自身相同
    """
    # OVERSIZED_TET_FACTOR 定义在 mesh_background_merge.py（本函数唯一
    # 的调用者所在文件），延迟导入以避免循环导入——本模块被
    # mesh_background_merge.py 导入，不能在模块顶层反向导入它。
    from .mesh_background_merge import OVERSIZED_TET_FACTOR

    logger.warning(
        "No boundary group was eligible for BL extrusion; filling the "
        "entire closed surface directly with tetgen (no boundary layer)"
    )
    regions = None

    # 设置了 max_cell_size 时：先共形细分过大的边界面，再按区域限制体积（tetgen 也可能在边界上插点——
    # 外部面因此是面网格三角形的细分，边界组由最后的 map_generated_boundaries 按几何包含关系给出）
    if max_cell_size is not None:
        center = surface_nodes.mean(axis=0)
        # max_cell_size 已经是米制，与 surface_nodes 一致
        target_edge_length = max_cell_size

        # 在 TetGen 之前细化过大的边界面的边
        logger.info(f"Refining boundary faces with max edge length > {target_edge_length:.4f}m...")
        proc_nodes, proc_faces = _refine_large_boundary_faces(
            surface_nodes, surface_faces, target_edge_length
        )

        regions = [(center, 1, target_edge_length ** 3 * CORE_VOLUME_CAP_FRACTION)]
        background_points = generate_core_background_points(
            proc_nodes, proc_faces, target_edge_length
        )
    else:
        proc_nodes, proc_faces = surface_nodes, surface_faces
        background_points = None

    core_nodes, core_tets = fill_core_volume(
        proc_nodes, proc_faces, holes=hole_points,
        regions=regions,
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

    if export_core_only:
        if not export_core_only_path:
            raise ValueError("export_core_only=True requires export_core_only_path to be set")
        _export_partial_mesh_and_exit(
            merged_nodes, prism_cells, tet_cells,
            export_core_only_path, "core-only (no BL region - this is the whole mesh)",
            surface_nodes, surface_faces, surface_boundaries,
        )

    return merged_nodes, prism_cells, tet_cells
