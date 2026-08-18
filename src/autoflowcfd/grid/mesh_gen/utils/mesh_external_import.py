"""导入 + 校验 + 尽力修复"外部生成"的体网格（例如 ANSA 自身的体网格导出），
并配合其原始面网格用于边界条件归属。

这是"自带体网格"路径：用户不走本项目自己的 生成-体积 流程（BL
挤出 + tetgen/gmsh 核心填充），而是直接提供别的工具已经生成好的体网格，
外加它对应的原始面 .nas 文件（之所以需要面网格，是因为体网格本身通常
不带任何边界条件信息——见 nas_parser_volume 自己的文档字符串）。
import_external_volume_mesh 完成这三步：质量检测、尽力修复、以及（通过
其返回值，每个 solve_commands.py 入口都已经知道如何把它当作
VolumeMeshData 消费）提交给求解器。

这里的修复范围比本项目自己生成流程的完整 阶段 A/B' 窄得多：只应用
阶段 A（质量门控的拉普拉斯平滑，mesh_repair.smooth_bad_cells），而且
只作用于四面体部分。阶段 B'（局部 cavity 重新铺网）不会运行——它和本
项目自己生成流程的假设绑得更紧（例如 阶段 B' 按 tetgen 自己的质量标准
重铺 cavity，前提是这片区域本来就是 tetgen 填出来的），这些假设对外部
未知工具生成的网格不一定成立。只用 阶段 A 是最安全、适用面最广的修复：
它只会挪动本就是内部、非边界的节点，且只在不产生负体积单元时才提交这次
移动——见 smooth_bad_cells 自己的文档字符串。
"""

import numpy as np
from typing import Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ...schema.grid_data import VolumeMeshData
    from ...validation.quality_validator import MeshQualityReport


def _smooth_external_tets(
    volume_mesh: 'VolumeMeshData', max_passes: int,
) -> 'VolumeMeshData':
    """阶段 A 平滑，仅作用于四面体部分——见本模块文档字符串
    了解为何棱柱被排除在外。网格真实外表面上或棱柱/四面体
    接口上的节点都自动受保护：由于棱柱被从传给
    smooth_bad_cells 的连接关系数组中排除，一个仅与（现在不可见的）
    棱柱邻居共享的面在 compute_movable_node_mask 看来与普通
    外边界面完全一样，以相同方式被排除在移动之外——无需
    分离的 n_bl_cells 簿记。
    """
    from ...schema.grid_nodes import NodeArray
    from ...schema.grid_cells import TetrahedralCells
    from ...schema.grid_data import VolumeMeshData
    from ...schema.grid_metadata import GridMetadata
    from ...validation.quality_validator import MeshQualityValidator
    from ..repair.mesh_repair import smooth_bad_cells

    nodes = np.column_stack([volume_mesh.nodes.x, volume_mesh.nodes.y, volume_mesh.nodes.z])
    tet_conn = volume_mesh.cells.connectivity.astype(np.int64)

    validator = MeshQualityValidator()
    new_nodes, bad_mask_after, actions = smooth_bad_cells(nodes, tet_conn, validator, max_passes=max_passes)
    for action in actions:
        logger.info(f"  {action}")

    tet_vol = TetrahedralCells.compute_volumes(
        NodeArray.from_array(new_nodes),
        tet_conn,
    )
    new_cells_obj = TetrahedralCells(connectivity=tet_conn.astype(np.int32), volumes=tet_vol)

    new_prism_obj = volume_mesh.prism_cells
    if new_prism_obj is not None:
        from ...schema.grid_cells import PrismCells
        prism_vol = PrismCells.compute_volumes(
            NodeArray.from_array(new_nodes),
            new_prism_obj.connectivity,
        )
        new_prism_obj = PrismCells(connectivity=new_prism_obj.connectivity, volumes=prism_vol)

    new_nodes_obj = NodeArray.from_array(new_nodes)
    metadata = GridMetadata(
        node_count=len(new_nodes),
        cell_count=new_cells_obj.count + (new_prism_obj.count if new_prism_obj else 0),
        boundary_groups=list(volume_mesh.boundaries.groups.keys()),
        file_format=volume_mesh.metadata.file_format,
    )
    return VolumeMeshData(
        nodes=new_nodes_obj, cells=new_cells_obj, boundaries=volume_mesh.boundaries,
        metadata=metadata, prism_cells=new_prism_obj,
        surface_mesh=volume_mesh.surface_mesh,
    )


def import_external_volume_mesh(
    volume_mesh_path: str,
    surface_mesh_path: str,
    repair: bool = True,
    max_repair_passes: int = 5,
    check_overlap: bool = True,
    units: str = 'mm',
) -> Tuple['VolumeMeshData', 'MeshQualityReport']:
    """完整的"自带体网格"管线：解析、从伴生面网格归属边界组、
    质量检查，以及（若请求且需要时）尽力修复。

    Args:
        volume_mesh_path: 外部生成的体网格 .nas 文件路径
            （GRID + CTETRA + CPENTA）。
        surface_mesh_path: 体网格生成自的原始面 .nas 文件——
            为 map_boundaries_by_geometry 提供边界分组的几何信息，
            仅此而已（其自身的网格连接关系在此之后不再被使用）。
        repair: 若为 True（默认）且初始质量检查失败，运行
            阶段 A 平滑（见本模块文档字符串了解范围）并重新验证。
            若为 False，无论质量如何都按解析原样返回网格。
        max_repair_passes: 阶段 A 自身的 max_passes，原样转发。
        check_overlap: 转发给 MeshQualityValidator——物理重叠
            检查是大规模网格上最昂贵的单个检查；仅为快速初步
            查看时禁用。
        units: 转发给 parse_volume_mesh_nas——'mm'（默认，与
            NASParser 自身的默认值一致）、'm' 或 'auto'。必须与
            `surface_mesh_path` 实际使用的单位一致（NASParser 总是
            在内部将其缩放到米，无论此值如何），否则下方的几何
            边界匹配将静默失败——见 parse_volume_mesh_nas 自身的
            文档字符串。

    Returns:
        (volume_mesh, quality_report)——最终网格（若修复运行且
        有改动则已修复）及其最终质量报告。volume_mesh 是普通
        VolumeMeshData，与每个 solve_commands.py 入口点已接受的
        类型相同（例如通过 pickled 缓存文件——见
        cli/solve_commands.py 自身的 .pkl 处理）。
    """
    from ...nas_io.nas_parser_volume import parse_volume_mesh_nas
    from ...nas_io.parser_core import NASParser
    from .mesh_boundary import map_boundaries_by_geometry
    from ...validation.quality_validator import MeshQualityValidator

    volume_mesh = parse_volume_mesh_nas(volume_mesh_path, units=units)

    logger.info(f"Parsing companion surface mesh for boundary attribution: {surface_mesh_path}")
    surface_grid = NASParser(surface_mesh_path).parse()

    boundaries = map_boundaries_by_geometry(volume_mesh, surface_grid)
    volume_mesh.boundaries = boundaries
    volume_mesh.metadata.boundary_groups = list(boundaries.groups.keys())

    # 保存原始面网格数据，供参考面积（投影面积）计算使用
    surface_nodes = surface_grid.nodes.get_coordinates()  # shape=(n_nodes, 3)
    volume_mesh.surface_mesh = {
        'nodes': surface_nodes,
        'faces': surface_grid.cells.connectivity,
        'boundaries': surface_grid.boundaries
    }

    validator = MeshQualityValidator()
    logger.info("Checking external volume mesh quality (pre-repair)...")
    report = validator.validate_volume_mesh(volume_mesh, check_overlap=check_overlap)
    logger.info(f"\n{report.summary()}")

    if report.passed or not repair:
        if not report.passed:
            logger.warning(
                "Quality check failed and repair=False - returning the mesh as-is "
                "(best-effort; see report above)"
            )
        return volume_mesh, report

    logger.info("Quality check failed - attempting Stage A smoothing (tet region only)...")
    volume_mesh = _smooth_external_tets(volume_mesh, max_repair_passes)

    logger.info("Re-checking external volume mesh quality (post-repair)...")
    report = validator.validate_volume_mesh(volume_mesh, check_overlap=check_overlap)
    logger.info(f"\n{report.summary()}")
    if not report.passed:
        logger.warning(
            "Quality check still failing after Stage A - returning the best-effort "
            "repaired mesh anyway (see report above). This mesh may diverge if solved "
            "as-is; 'autoflowcfd solve steady'/'transient' will still enforce this gate "
            "before any iterations run, unless --skip-quality-check is passed."
        )
    return volume_mesh, report
