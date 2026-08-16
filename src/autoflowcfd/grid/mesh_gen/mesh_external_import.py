"""导入 + 校验 + 尽力修复"外部生成"的体网格（例如 ANSA 自身的体网格导出），
并配合其原始面网格用于边界条件归属。

这是"自带体网格"路径：用户不走本项目自己的 generate-volume 流程（BL
挤出 + tetgen/gmsh 核心填充），而是直接提供别的工具已经生成好的体网格，
外加它对应的原始面 .nas 文件（之所以需要面网格，是因为体网格本身通常
不带任何边界条件信息——见 nas_parser_volume 自己的文档字符串）。
import_external_volume_mesh 完成这三步：质量检测、尽力修复、以及（通过
其返回值，每个 solve_commands.py 入口都已经知道如何把它当作
VolumeMeshData 消费）提交给求解器。

这里的修复范围比本项目自己生成流程的完整 Stage A/B' 窄得多：只应用
Stage A（质量门控的拉普拉斯平滑，mesh_repair.smooth_bad_cells），而且
只作用于四面体部分。Stage B'（局部 cavity 重新铺网）不会运行——它和本
项目自己生成流程的假设绑得更紧（例如 Stage B' 按 tetgen 自己的质量标准
重铺 cavity，前提是这片区域本来就是 tetgen 填出来的），这些假设对外部
未知工具生成的网格不一定成立。只用 Stage A 是最安全、适用面最广的修复：
它只会挪动本就是内部、非边界的节点，且只在不产生负体积单元时才提交这次
移动——见 smooth_bad_cells 自己的文档字符串。
"""

import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ..structures import VolumeMeshData
    from ..validation.quality_validator import MeshQualityReport


def _smooth_external_tets(
    volume_mesh: 'VolumeMeshData', max_passes: int,
) -> 'VolumeMeshData':
    """Stage A smoothing restricted to the tet portion - see this module's
    own docstring for why prisms are out of scope. Nodes on the mesh's
    true exterior boundary OR on the prism/tet interface are both
    automatically protected: with prisms excluded from the connectivity
    array passed to smooth_bad_cells, a face that was only ever shared
    with a (now-invisible) prism neighbour looks exactly like an ordinary
    exterior boundary face to compute_movable_node_mask, and is excluded
    from movement the same way - no separate n_bl_cells bookkeeping
    needed.
    """
    from ..schema.grid_nodes import NodeArray
    from ..structures import TetrahedralCells, VolumeMeshData, GridMetadata
    from ..validation.quality_validator import MeshQualityValidator
    from .mesh_repair import smooth_bad_cells

    nodes = np.column_stack([volume_mesh.nodes.x, volume_mesh.nodes.y, volume_mesh.nodes.z])
    tet_conn = volume_mesh.cells.connectivity.astype(np.int64)

    validator = MeshQualityValidator()
    new_nodes, bad_mask_after, actions = smooth_bad_cells(nodes, tet_conn, validator, max_passes=max_passes)
    for action in actions:
        logger.info(f"  {action}")

    tet_vol = TetrahedralCells.compute_volumes(
        NodeArray(
            x=np.ascontiguousarray(new_nodes[:, 0]),
            y=np.ascontiguousarray(new_nodes[:, 1]),
            z=np.ascontiguousarray(new_nodes[:, 2]),
        ),
        tet_conn,
    )
    new_cells_obj = TetrahedralCells(connectivity=tet_conn.astype(np.int32), volumes=tet_vol)

    new_prism_obj = volume_mesh.prism_cells
    if new_prism_obj is not None:
        from ..structures import PrismCells
        prism_vol = PrismCells.compute_volumes(
            NodeArray(
                x=np.ascontiguousarray(new_nodes[:, 0]),
                y=np.ascontiguousarray(new_nodes[:, 1]),
                z=np.ascontiguousarray(new_nodes[:, 2]),
            ),
            new_prism_obj.connectivity,
        )
        new_prism_obj = PrismCells(connectivity=new_prism_obj.connectivity, volumes=prism_vol)

    new_nodes_obj = NodeArray(
        x=np.ascontiguousarray(new_nodes[:, 0]),
        y=np.ascontiguousarray(new_nodes[:, 1]),
        z=np.ascontiguousarray(new_nodes[:, 2]),
    )
    metadata = GridMetadata(
        node_count=len(new_nodes),
        cell_count=new_cells_obj.count + (new_prism_obj.count if new_prism_obj else 0),
        boundary_groups=list(volume_mesh.boundaries.groups.keys()),
        file_format=volume_mesh.metadata.file_format,
    )
    return VolumeMeshData(
        nodes=new_nodes_obj, cells=new_cells_obj, boundaries=volume_mesh.boundaries,
        metadata=metadata, prism_cells=new_prism_obj,
    )


def import_external_volume_mesh(
    volume_mesh_path: str,
    surface_mesh_path: str,
    repair: bool = True,
    max_repair_passes: int = 5,
    check_overlap: bool = True,
    units: str = 'mm',
) -> Tuple['VolumeMeshData', 'MeshQualityReport']:
    """Full "bring your own volume mesh" pipeline: parse, attribute
    boundary groups from the companion surface mesh, quality-check, and
    (if requested and needed) best-effort repair.

    Args:
        volume_mesh_path: Path to the externally-generated volume-mesh
            .nas file (GRID + CTETRA + CPENTA).
        surface_mesh_path: Path to the ORIGINAL surface .nas the volume
            mesh was generated from - supplies boundary-group geometry
            for map_boundaries_by_geometry, and nothing else (its own
            mesh connectivity is not reused for anything past that).
        repair: If True (default) and the initial quality check fails,
            run Stage A smoothing (see this module's own docstring for
            scope) and re-validate. If False, the mesh is returned exactly
            as parsed regardless of quality.
        max_repair_passes: Stage A's own max_passes, forwarded unchanged.
        check_overlap: Forwarded to MeshQualityValidator - physical-
            overlap checking is the most expensive single check on a
            large mesh; disable only for a quick preliminary look.
        units: Forwarded to parse_volume_mesh_nas - 'mm' (default, matches
            NASParser's own default for `surface_mesh_path`), 'm', or
            'auto'. MUST agree with whatever unit `surface_mesh_path`
            itself is actually in (NASParser always scales that one to
            metres internally regardless of this value), or geometric
            boundary matching below will silently fail to match anything -
            see parse_volume_mesh_nas's own docstring.

    Returns:
        (volume_mesh, quality_report) - the FINAL mesh (repaired, if
        repair ran and changed anything) and its own final quality
        report. volume_mesh is a plain VolumeMeshData, the same type
        every solve_commands.py entry point already accepts (e.g. via a
        pickled cache file - see cli/solve_commands.py's own .pkl
        handling).
    """
    from ..nas_io.nas_parser_volume import parse_volume_mesh_nas
    from ..nas_io.parser_core import NASParser
    from .mesh_boundary import map_boundaries_by_geometry
    from ..validation.quality_validator import MeshQualityValidator

    volume_mesh = parse_volume_mesh_nas(volume_mesh_path, units=units)

    logger.info(f"Parsing companion surface mesh for boundary attribution: {surface_mesh_path}")
    surface_grid = NASParser(surface_mesh_path).parse()

    boundaries = map_boundaries_by_geometry(volume_mesh, surface_grid)
    volume_mesh.boundaries = boundaries
    volume_mesh.metadata.boundary_groups = list(boundaries.groups.keys())

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
