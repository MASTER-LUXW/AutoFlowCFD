"""NAS 解析器核心功能。

提供 NASParser 主类，用流式解析方式读取 ANSA 导出的 .nas 文件，支持
v22、v23、v24 三个版本。
"""

import re
from pathlib import Path
from typing import Optional, Dict
from loguru import logger

from ..structures import GridData, NodeArray, GridMetadata, VolumeMeshData
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
    
    # Below this raw (pre-scale) bounding-box max dimension, a file is
    # assumed to already be in meters when units='auto'; above it, assumed
    # to be in millimeters. A single-vehicle external-aero domain is
    # typically a few meters (car) to a few tens of meters (surrounding
    # tunnel/farfield) - the same geometry expressed in mm would read in
    # the thousands, comfortably on the other side of this threshold.
    AUTO_UNITS_MM_THRESHOLD = 50.0

    def __init__(self, file_path: str, encoding: str = 'UTF-8', units: str = 'mm'):
        """初始化解析器

        Args:
            file_path: .nas 文件路径
            encoding: 文件编码，默认 UTF-8
            units: 文件中坐标的长度单位：'mm'（默认——匹配 ANSA
                的典型导出约定，按 1e-3 缩放到米）、'm'（不缩放，
                原样使用）或 'auto'（基于原始包围盒大小的启发式
                检测——见 AUTO_UNITS_MM_THRESHOLD）。以前 mm->m 缩放
                因子是无条件硬编码的，没有办法覆盖或检测文件已经
                是米，静默损坏几何（例如 1m 的车缩小到 1mm 的点，
                或反之）对于任何非毫米的文件。

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件扩展名不正确，或 units 不是 'mm'/'m'/'auto'
        """
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.version: Optional[str] = None

        if units not in ('mm', 'm', 'auto'):
            raise ValueError(f"units must be 'mm', 'm', or 'auto', got {units!r}")
        self.units = units

        if not self.file_path.exists():
            raise FileNotFoundError(f"NAS file not found: {file_path}")
        
        if self.file_path.suffix.lower() not in ['.nas', '.bdf']:
            logger.warning(
                f"File extension '{self.file_path.suffix}' is not standard "
                f"for NAS files (.nas or .bdf). Proceeding anyway."
            )
        
        logger.info(f"NAS parser initialized: {file_path}")
    
    def parse(
        self, 
        skip_validation: bool = False,
        generate_volume_mesh: bool = False,
        volume_mesh_params: Optional[Dict] = None
    ) -> GridData:
        """解析 NAS 文件。

        解析的主入口点。协调版本检测、节点/单元/边界解析
        和元数据构建。

        Args:
            skip_validation: 是否跳过网格质量校验
            generate_volume_mesh: 是否从表面网格生成体网格
            volume_mesh_params: 体网格生成参数

        Returns:
            GridData: 网格数据对象（表面或体网格）

        Raises:
            NASFormatError: 文件格式不支持或损坏
            NASParseError: 解析过程中发生错误
        """
        logger.info(f"Parsing NAS file: {self.file_path}")
        
        try:
            # Step 1: Detect file version
            self.version = self._detect_version()
            logger.info(f"Detected NAS version: {self.version}")
            
            if self.version not in self.SUPPORTED_VERSIONS:
                raise NASFormatError(
                    f"Unsupported NAS version: {self.version}. "
                    f"Supported versions: {self.SUPPORTED_VERSIONS}"
                )
            
            # Step 2: Parse nodes (delegated)
            logger.info("Parsing nodes...")
            nodes, node_id_to_index = parse_nodes_from_nas(
                str(self.file_path), self.encoding
            )
            logger.info(f"Parsed {nodes.count:,} nodes")
            
            if nodes.count == 0:
                raise NASParseError("No nodes found in NAS file")
            
            # Convert units to meters (or detect them, if units='auto').
            raw_extent = float(max(
                nodes.x.max() - nodes.x.min(),
                nodes.y.max() - nodes.y.min(),
                nodes.z.max() - nodes.z.min(),
            ))

            if self.units == 'mm':
                scale_factor = 1e-3
            elif self.units == 'm':
                scale_factor = 1.0
            else:  # 'auto'
                if raw_extent > self.AUTO_UNITS_MM_THRESHOLD:
                    scale_factor = 1e-3
                    logger.info(
                        f"units='auto': raw bounding-box max extent={raw_extent:.4g} > "
                        f"{self.AUTO_UNITS_MM_THRESHOLD:g} -> assuming millimeters "
                        f"(scaling by 1e-3)"
                    )
                else:
                    scale_factor = 1.0
                    logger.info(
                        f"units='auto': raw bounding-box max extent={raw_extent:.4g} <= "
                        f"{self.AUTO_UNITS_MM_THRESHOLD:g} -> assuming the file is "
                        f"already in meters (no scaling)"
                    )

            nodes.x = nodes.x * scale_factor
            nodes.y = nodes.y * scale_factor
            nodes.z = nodes.z * scale_factor

            # 对转换后的（米制）域大小进行合理性检查，
            # 对照汽车外气动的合理范围。这不会改变任何值
            # ——只是把静默的单位不匹配（否则会直接进入
            # 在物理上无意义的几何上进行网格/求解）变成
            # 响亮的、可操作的警告。
            scaled_extent = raw_extent * scale_factor
            if scaled_extent < 0.1:
                logger.warning(
                    f"After unit conversion (units={self.units!r}, scale={scale_factor:g}), "
                    f"the mesh's largest bounding-box dimension is only "
                    f"{scaled_extent:.4g} m - implausibly small for automotive "
                    f"external-aero (expect several meters for the vehicle and "
                    f"tens of meters for the surrounding domain). This usually "
                    f"means the file's actual units don't match units={self.units!r} "
                    f"- try units='m' or units='auto'."
                )
            elif scaled_extent > 1000.0:
                logger.warning(
                    f"After unit conversion (units={self.units!r}, scale={scale_factor:g}), "
                    f"the mesh's largest bounding-box dimension is "
                    f"{scaled_extent:.4g} m - implausibly large for automotive "
                    f"external-aero. This usually means the file's actual units "
                    f"don't match units={self.units!r} - try units='mm'."
                )
            
            # Step 3: Parse cells (delegated)
            logger.info("Parsing cells...")
            surface_cells, cell_pids = parse_cells_from_nas(
                str(self.file_path), node_id_to_index, self.encoding
            )
            logger.info(f"Parsed {surface_cells.count:,} surface cells")

            if surface_cells.count == 0:
                # A common cause: the file is actually a VOLUME mesh
                # (CTETRA/CPENTA, e.g. ANSA's own volume export) handed to
                # a code path that only ever looks for CTRIA3 surface
                # triangles - silently parsing 0 cells and failing here
                # with no clue why, unless this is checked for and called
                # out explicitly. `grid import-volume` is the path built for
                # a volume mesh instead - see nas_parser_volume.py. (`solve
                # steady`/`transient` themselves only ever accept an
                # already-generated .pkl volume mesh, never a raw .nas file
                # of either kind - see solve_helpers.load_mesh_for_solver.)
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

            # Step 4: Parse boundaries (delegated)
            # Pass the PID already resolved for each surviving cell (cells_data)
            # instead of letting parse_boundary_properties re-scan CTRIA3 cards
            # independently. A second independent scan does not know which
            # cells parse_cells_from_nas skipped (missing node references), so
            # its cell indices would drift out of alignment with surface_cells
            # as soon as any cell is skipped.
            logger.info("Parsing boundary conditions...")
            cells_data = list(enumerate(cell_pids.tolist()))
            boundaries = parse_boundary_properties(
                str(self.file_path),
                self.encoding,
                cell_count=surface_cells.count,
                cells_data=cells_data
            )
            logger.info(f"Parsed {len(boundaries.groups)} boundary groups")

            # Step 5: Compute bounding box
            bounding_box = self._compute_bounding_box(nodes)

            # Step 6: Generate volume mesh if requested
            if generate_volume_mesh:
                metadata = GridMetadata(
                    node_count=nodes.count,
                    cell_count=surface_cells.count,
                    boundary_groups=list(boundaries.groups.keys()),
                    file_format=self.version,
                    bounding_box=bounding_box
                )
                surface_grid = GridData(
                    nodes=nodes, cells=surface_cells, boundaries=boundaries, metadata=metadata
                )
                return self.generate_volume_mesh_from_surface(surface_grid, volume_mesh_params)
            else:
                # 使用 表面 网格
                metadata = GridMetadata(
                    node_count=nodes.count,
                    cell_count=surface_cells.count,
                    boundary_groups=list(boundaries.groups.keys()),
                    file_format=self.version,
                    bounding_box=bounding_box
                )
                
                grid_data = GridData(
                    nodes=nodes,
                    cells=surface_cells,
                    boundaries=boundaries,
                    metadata=metadata
                )
                
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
