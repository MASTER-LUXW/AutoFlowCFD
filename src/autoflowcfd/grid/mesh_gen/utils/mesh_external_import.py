"""导入 + 校验 + 尽力修复"外部生成"的体网格（例如 ANSA 自身的体网格导出），
并配合其原始面网格用于边界条件归属。

这是"自带体网格"路径：用户不走本项目自己的 生成-体积 流程（BL
挤出 + tetgen/gmsh 核心填充），而是直接提供别的工具已经生成好的体网格，
外加它对应的原始面 .nas 文件（之所以需要面网格，是因为体网格本身通常
不带任何边界条件信息——见 nas_parser_volume 自己的文档字符串）。
import_external_volume_mesh 完成这三步：质量检测、尽力修复、以及返回求解命令（`cli/solve/mesh_loader.py`）
直接消费的 VolumeMeshData。

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
from typing import Dict, Optional, Tuple, TYPE_CHECKING
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
    bc_types: Optional[Dict[str, str]] = None,
    validate: bool = True,
) -> Tuple['VolumeMeshData', Optional['MeshQualityReport']]:
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
        units: 体网格与伴生面网格共用的长度单位——'mm'（默认）、'm' 或 'auto'，同时转发给
            parse_volume_mesh_nas 与面网格的 NASParser（两者都缩放到米；单位不一致时下方的几何
            边界匹配整体错位）。
        bc_types: 用户显式给出的边界条件类型 {组名: 类型}（`--bc-type`，见 `nas_io/boundary_types.py`）；
            导入后每个组都必须有确定的类型
        validate: False 时只解析与归属边界、不做质量检查（返回的报告为 None；须同时 repair=False）——求解命令把
            .nas 输入转存成 .pkl 时用，质量门在随后加载 .pkl 时做一次

    Returns:
        (volume_mesh, quality_report)——最终网格（若修复运行且
        有改动则已修复）及其最终质量报告。volume_mesh 是普通 VolumeMeshData，与求解命令读 .pkl 得到的类型相同
        （`cli/solve/mesh_loader.py`）。
    """
    from ...nas_io.nas_parser_volume import parse_volume_mesh_nas
    from ...nas_io.parser_core import NASParser
    from .mesh_boundary import map_boundaries_by_geometry
    from ...validation.quality_validator import MeshQualityValidator

    if repair and not validate:
        raise ValueError("repair=True 需要 validate=True（修复由质量检查结果触发）")
    volume_mesh = parse_volume_mesh_nas(volume_mesh_path, units=units)

    logger.info(f"Parsing companion surface mesh for boundary attribution: {surface_mesh_path}")
    # 面网格与体网格必须按同一长度单位读入（此前面网格恒按默认 'mm'，units='m' 时边界几何归属整体错位）
    surface_grid = NASParser(surface_mesh_path, units=units, bc_types=bc_types).parse()
    from ...nas_io.boundary_types import require_specified_bc_types
    require_specified_bc_types(surface_grid.boundaries)

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

    if not validate:
        return volume_mesh, None

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
