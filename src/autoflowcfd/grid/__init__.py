"""网格解析与处理模块。

负责 ANSA .nas 文件解析、网格数据结构、质量校验与边界条件映射。

核心组件:
    - GridData: 主网格数据结构（SoA 布局）
    - NodeArray: SoA 格式的节点坐标
    - CellArray: 单元连接关系和类型信息
    - BoundaryMap: 边界条件映射
    - NASParser: ANSA .nas 文件解析器（v22/v23/v24）
    - GridValidator: 网格质量检查器

示例:
    >>> from autoflowcfd.grid import NASParser, GridValidator
    >>> parser = NASParser("car_model.nas")
    >>> grid = parser.parse()
    >>> validator = GridValidator(grid)
    >>> results = validator.validate()
    >>> print(f"Nodes: {grid.metadata.node_count}")
    >>> print(f"Quality check passed: {results['passed']}")
"""

# 从模块化子模块重新导出以保持向后兼容
from .schema.grid_nodes import NodeArray, CupyNodeArray
from .schema.grid_cells import CellArray, CupyCellArray, TetrahedralCells, PrismCells
from .schema.grid_boundaries import BoundaryMap
from .schema.grid_metadata import GridMetadata
from .schema.grid_faces import FaceData
from .schema.grid_data import GridData, CupyGridData, VolumeMeshData

# 解析器模块（已模块化）
from .nas_io.parser_core import NASParser
from .nas_io.nas_parser_exceptions import NASParserError, NASFormatError, NASParseError

# 其他模块
from .validation.validator import GridValidator
from .mesh_gen.tetgen.volume_mesh_generator import VolumeMeshGenerator
from .validation.quality_validator import MeshQualityValidator, MeshQualityReport
from .mesh_gen.extraction.face_extractor import FaceExtractor, extract_faces_from_tetrahedra
from .high_order.high_order_mesh import HighOrderMesh

# 新网格生成子模块（内部使用）
from .mesh_gen.utils.mesh_utils import (
    validate_surface_mesh,
    validate_bounding_box,
    compute_face_normals,
    check_reached_boundary
)
from .mesh_gen.extrusion.mesh_extrusion import extrude_layers
from .mesh_gen.extrusion.mesh_layer_step import extrude_single_layer
from .mesh_gen.tetgen.mesh_prism_to_tet import convert_layers_to_tetrahedra
from .mesh_gen.background.mesh_background import generate_hybrid_mesh
from .mesh_gen.utils.mesh_boundary import (
    identify_boundaries_from_surface,
    map_surface_boundaries
)

# NAS 导出模块
from .nas_io.nas_export import export_volume_mesh_to_nas

__all__ = [
    # 数据结构（已模块化）
    "GridData",
    "NodeArray",
    "CellArray",
    "BoundaryMap",
    "GridMetadata",
    "CupyNodeArray",
    "CupyCellArray",
    "CupyGridData",
    "TetrahedralCells",
    "PrismCells",
    "VolumeMeshData",
    "FaceData",
    # 解析器（已模块化）
    "NASParser",
    "NASParserError",
    "NASFormatError",
    "NASParseError",
    # 验证器
    "GridValidator",
    # 生成器
    "VolumeMeshGenerator",
    # 高阶网格
    "HighOrderMesh",
    # 质量验证器
    "MeshQualityValidator",
    "MeshQualityReport",
    # 面提取
    "FaceExtractor",
    "extract_faces_from_tetrahedra",
    # 内部网格生成工具（非公共 API）
    # mesh_utils
    "validate_surface_mesh",
    "validate_bounding_box",
    "compute_face_normals",
    "check_reached_boundary",
    # mesh_extrusion 边界层挤出
    "extrude_layers",
    "extrude_single_layer",
    "convert_layers_to_tetrahedra",
    # mesh_background 背景网格
    "generate_hybrid_mesh",
    # mesh_boundary 边界识别
    "identify_boundaries_from_surface",
    "map_surface_boundaries",
    # NAS 导出
    "export_volume_mesh_to_nas",
]
