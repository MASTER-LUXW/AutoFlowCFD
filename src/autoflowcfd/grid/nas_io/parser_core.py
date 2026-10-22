"""NAS 解析器核心功能。

提供 NASParser 主类，用流式解析方式读取 ANSA 导出的 .nas 文件，支持
v22、v23、v24 三个版本。
"""

import re
from pathlib import Path
from typing import Optional, Dict
from loguru import logger

from ..schema import GridData, NodeArray, GridMetadata, VolumeMeshData
from .nas_parser_exceptions import NASParserError, NASFormatError, NASParseError
from .nas_parser_nodes import parse_nodes_from_nas
from .nas_parser_cells import parse_cells_from_nas
from .nas_parser_boundary import parse_boundary_properties


class NASParser:
    """ANSA .nas 文件解析器

    解析 ANSA 生成的 Nastran 格式网格文件 (.nas)，支持
    v22、v23、v24 三个版本。使用流式方式高效处理大文件。

    Attributes:
        file_path: .nas 文件路径
        encoding: 文件编码，默认 UTF-8
        version: 检测到的 NAS 文件版本
    """

    SUPPORTED_VERSIONS = {"v22", "v23", "v24"}

    def __init__(self, file_path: str, encoding: str = 'UTF-8', units: str = 'mm',
                 bc_types: Optional[Dict[str, str]] = None):
        """初始化解析器

        Args:
            file_path: .nas 文件路径
            encoding: 文件编码，默认 UTF-8
            units: 文件中坐标的长度单位：'mm'（默认——匹配 ANSA
                的典型导出约定，按 1e-3 缩放到米）、'm'（不缩放，
                原样使用）或 'auto'（基于原始包围盒大小的启发式
                检测——见 nas_io/units.py）。以前 mm->m 缩放
                因子是无条件硬编码的，没有办法覆盖或检测文件已经
                是米，静默损坏几何（例如 1m 的车缩小到 1mm 的点，
                或反之）对于任何非毫米的文件。
            bc_types: 用户显式给出的边界条件类型 {组名: 类型}（`--bc-type`），覆盖按属性名识别的结果
                （`boundary_types.py`）；组名必须存在

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件扩展名不正确，或 units 不是 'mm'/'m'/'auto'
        """
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.version: Optional[str] = None

        from .units import UNITS_CHOICES

        if units not in UNITS_CHOICES:
            raise ValueError(f"units must be one of {UNITS_CHOICES}, got {units!r}")
        self.units = units
        #: parse() 之后为实际单位 'mm' / 'm'（units='auto' 时由包围盒判定）
        self.resolved_units: Optional[str] = None
        self.bc_types = dict(bc_types or {})

        if not self.file_path.exists():
            raise FileNotFoundError(f"NAS file not found: {file_path}")

        if self.file_path.suffix.lower() not in ['.nas', '.bdf']:
            logger.warning(
                f"File extension '{self.file_path.suffix}' is not standard "
                f"for NAS files (.nas or .bdf). Proceeding anyway."
            )

        logger.info(f"NAS parser initialized: {file_path}")

    def parse(self) -> GridData:
        """解析 NAS 面网格文件。

        解析的主入口点。协调版本检测、节点/单元/边界解析和元数据构建。只解析、不做质量校验（质量校验是
        `GridValidator`）；生成体网格走 `generate_volume_mesh_from_surface`。

        Returns:
            GridData: 面网格数据

        Raises:
            NASFormatError: 文件格式不支持或损坏
            NASParseError: 解析过程中发生错误
        """
        logger.info(f"Parsing NAS file: {self.file_path}")

        try:
            # 第 1 步：识别文件版本
            self.version = self._detect_version()
            logger.info(f"Detected NAS version: {self.version}")

            if self.version not in self.SUPPORTED_VERSIONS:
                raise NASFormatError(
                    f"Unsupported NAS version: {self.version}. "
                    f"Supported versions: {self.SUPPORTED_VERSIONS}"
                )

            # 第 2 步：解析节点（委托）
            logger.info("Parsing nodes...")
            nodes, node_id_to_index = parse_nodes_from_nas(
                str(self.file_path), self.encoding
            )
            logger.info(f"Parsed {nodes.count:,} nodes")

            if nodes.count == 0:
                raise NASParseError("No nodes found in NAS file")

            # 坐标换算到米（units='auto' 时先检测单位）
            raw_extent = float(max(
                nodes.x.max() - nodes.x.min(),
                nodes.y.max() - nodes.y.min(),
                nodes.z.max() - nodes.z.min(),
            ))

            from .units import resolve_units

            self.resolved_units, scale_factor = resolve_units(self.units, raw_extent, source=str(self.file_path))
            nodes.x = nodes.x * scale_factor
            nodes.y = nodes.y * scale_factor
            nodes.z = nodes.z * scale_factor

            # 第 3 步：解析单元（委托）
            logger.info("Parsing cells...")
            surface_cells, cell_pids = parse_cells_from_nas(
                str(self.file_path), node_id_to_index, self.encoding
            )
            logger.info(f"Parsed {surface_cells.count:,} surface cells")

            if surface_cells.count == 0:
                # 常见原因：文件其实是体网格（CTETRA/CPENTA，例如 ANSA 的体网格导出），被交给了只找 CTRIA3
                # 面三角形的路径——不在这里明确指出，就只会解析出 0 个单元、看不出原因地失败。体网格应走
                # `grid import-volume`（nas_parser_volume.py），或直接交给 `solve steady/transient`
                # （.nas 体网格 + --surface-mesh，见 cli/solve/mesh_loader.py::load_mesh_for_solver）。
                looks_like_volume_mesh = False
                try:
                    with open(self.file_path, 'r', encoding=self.encoding, errors='replace') as f:
                        for line in f:
                            if line[:8].strip() in ("CTETRA", "CPENTA"):
                                looks_like_volume_mesh = True
                                break
                except OSError:
                    pass
                if looks_like_volume_mesh:
                    raise NASParseError(
                        f"No CTRIA3 surface cells found in {self.file_path} - it contains "
                        f"CTETRA/CPENTA volume elements instead (e.g. an already-generated "
                        f"volume mesh, such as ANSA's own volume export). Use 'autoflowcfd "
                        f"grid import-volume {self.file_path} -s <original_surface.nas> "
                        f"-o <output.pkl>' instead of parsing this file directly as a "
                        f"surface mesh."
                    )
                raise NASParseError("No cells found in NAS file")

            # 第 4 步：解析边界（委托）。把每个保留下来的单元已解析出的 PID（cells_data）直接传过去，
            # 而不是让 parse_boundary_properties 再独立扫描一遍 CTRIA3：第二遍扫描不知道 parse_cells_from_nas
            # 跳过了哪些单元（引用了不存在的节点），一旦有单元被跳过，单元编号就和 surface_cells 对不上。
            logger.info("Parsing boundary conditions...")
            cells_data = list(enumerate(cell_pids.tolist()))
            boundaries = parse_boundary_properties(
                str(self.file_path),
                self.encoding,
                cell_count=surface_cells.count,
                cells_data=cells_data
            )
            from .boundary_types import apply_bc_types
            apply_bc_types(boundaries, self.bc_types)
            logger.info(f"Parsed {len(boundaries.groups)} boundary groups")

            # 第 5 步：计算包围盒
            bounding_box = self._compute_bounding_box(nodes)

            metadata = GridMetadata(
                node_count=nodes.count,
                cell_count=surface_cells.count,
                boundary_groups=list(boundaries.groups.keys()),
                file_format=self.version,
                bounding_box=bounding_box
            )
            grid_data = GridData(nodes=nodes, cells=surface_cells, boundaries=boundaries, metadata=metadata)

            logger.success(
                f"NAS file parsed successfully: "
                f"{nodes.count:,} nodes, {surface_cells.count:,} cells, "
                f"{len(boundaries.groups)} boundary groups"
            )
            return grid_data

        except NASParserError:
            raise
        except Exception as e:
            raise NASParseError(f"Unexpected error during parsing: {str(e)}") from e

    def generate_volume_mesh_from_surface(
        self,
        surface_grid: GridData,
        volume_mesh_params: Optional[Dict] = None,
    ) -> 'VolumeMeshData':
        """从已解析的表面 GridData 生成体网格。

        从 parse() 的 generate_volume_mesh=True 路径中提取出来，
        使已有解析好的 surface_grid 的调用方（例如对其运行了
        GridValidator 做生成前质量检查后）可以直接送入体网格
        生成，不需要第二次冗余的原始 NAS 文件重新解析——
        parse() 本身现在只构建表面 GridData 并委托到这里。

        Args:
            surface_grid: 已解析的表面网格（nodes/cells/
                boundaries/metadata.bounding_box）
            volume_mesh_params: 体网格生成参数（见 parse() 的
                volume_mesh_params）

        Returns:
            VolumeMeshData
        """
        from .parser_volume import generate_volume_mesh_from_surface as _generate

        return _generate(surface_grid, volume_mesh_params)

    def _detect_version(self) -> str:
        """检测NAS文件版本"""
        try:
            with open(self.file_path, 'r', encoding=self.encoding, errors='ignore') as f:
                for i, line in enumerate(f):
                    if i > 100:
                        break

                    line_upper = line.upper().strip()

                    if 'ANSA' in line_upper:
                        match = re.search(r'V(\d{2})', line_upper)
                        if match:
                            version_num = match.group(1)
                            return f"v{version_num}"

                    if 'NASTRAN' in line_upper or 'BDF' in line_upper:
                        return "v24"

                logger.warning("No version marker found in NAS file. Assuming v24 format.")
                return "v24"

        except Exception as e:
            raise NASFormatError(f"Failed to detect NAS version: {str(e)}") from e

    def _compute_bounding_box(self, nodes: NodeArray):
        """计算表面网格的精确包围盒（不加 buffer）。

        输入表面网格是一个封闭的、水密的域边界（车身 + 地面 +
        入口/出口 + 隧道/远场）——它已经定义了精确的计算域，
        所以这里返回纯节点范围，不加填充。加填充（如此函数以前
        做的）会让体网格扩展到域表面实际封闭的范围之外（例如
        直线穿过地面）。结果仅用于边界组分类
        (mesh_domain_classify) 并作为 BL 增长上限参考，从不用于
        定义填充几何。

        Returns:
            (min_x, max_x, min_y, max_y, min_z, max_z) 的元组
        """
        min_x, max_x = float(nodes.x.min()), float(nodes.x.max())
        min_y, max_y = float(nodes.y.min()), float(nodes.y.max())
        min_z, max_z = float(nodes.z.min()), float(nodes.z.max())

        logger.debug(
            f"Bounding box (exact, unpadded): "
            f"x=[{min_x:.4f}, {max_x:.4f}], y=[{min_y:.4f}, {max_y:.4f}], "
            f"z=[{min_z:.4f}, {max_z:.4f}]"
        )

        return (min_x, max_x, min_y, max_y, min_z, max_z)
