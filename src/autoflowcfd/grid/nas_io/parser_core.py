"""NAS parser core functionality.

Provides the main NASParser class with streaming parsing for ANSA .nas files,
supporting versions v22, v23, and v24.
"""

import re
from pathlib import Path
from typing import Optional, Dict
import numpy as np
from loguru import logger

from ..structures import GridData, NodeArray, GridMetadata, VolumeMeshData
from .nas_parser_exceptions import NASParserError, NASFormatError, NASParseError
from .nas_parser_nodes import parse_nodes_from_nas
from .nas_parser_cells import parse_cells_from_nas
from .nas_parser_boundary import parse_boundary_properties


class NASParser:
    """ANSA .nas文件解析器
    
    Parses ANSA-generated Nastran format mesh files (.nas), supporting
    versions v22, v23, and v24. Uses streaming approach to handle large
    files efficiently.
    
    Attributes:
        file_path: .nas文件路径
        encoding: 文件编码,默认UTF-8
        version: 检测到的NAS文件版本
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
            file_path: .nas文件路径
            encoding: 文件编码,默认UTF-8
            units: Length unit of the coordinates in the file: 'mm'
                (default - matches ANSA's typical export convention, scales
                by 1e-3 to meters), 'm' (no scaling, use as-is), or 'auto'
                (heuristic based on the raw bounding-box size - see
                AUTO_UNITS_MM_THRESHOLD). Previously the mm->m scale factor
                was hardcoded unconditionally with no way to override or
                detect a file already in meters, silently corrupting
                geometry (e.g. a 1m car shrinking to a 1mm dot, or vice
                versa) for any file not in millimeters.

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件扩展名不正确, or units is not 'mm'/'m'/'auto'
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
        """解析NAS文件
        
        Main entry point for parsing. Orchestrates version detection,
        node/cell/boundary parsing, and metadata construction.
        
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

            # Sanity-check the resulting (meters) domain size against a
            # plausible automotive external-aero range. This never changes
            # any value - it only turns a silent unit mismatch (which would
            # otherwise proceed straight into meshing/solving on a
            # physically nonsensical geometry) into a loud, actionable
            # warning.
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
                logger.info("Generating volume mesh from surface geometry...")
                
                from ..mesh_gen.volume_mesh_generator import VolumeMeshGenerator
                
                params = volume_mesh_params or {}
                
                # Optimized three-stage hybrid mesh strategy
                # Stage 1: Boundary Layer (8 layers, fine resolution for y+ control)
                # Stage 2: Transition Layer (4 layers, medium resolution)
                # Stage 3: Far-field Background (coarse Cartesian grid)
                optimized_params = {
                    'growth_rate': params.get('growth_rate', 1.2),
                    'max_layers': params.get('max_layers', 12),  # 8 BL + 4 transition
                    'min_cell_size': params.get('min_cell_size', 0.01),
                    'target_cells': params.get('target_cells', 400000),  # Balanced target
                    'max_cell_size': params.get('max_cell_size'),
                }
                
                logger.info(
                    f"Using three-stage hybrid mesh strategy:\n"
                    f"  Stage 1 (BL): 8 layers, growth_rate=1.2\n"
                    f"  Stage 2 (Transition): 4 layers, growth_rate=1.5\n"
                    f"  Stage 3 (Background): Coarse Cartesian grid\n"
                    f"  Target total cells: ~{optimized_params['target_cells']:,}"
                )
                
                generator = VolumeMeshGenerator(**optimized_params)
                
                surface_nodes_np = np.column_stack([nodes.x, nodes.y, nodes.z])
                
                volume_mesh = generator.generate_from_surface(
                    surface_nodes=surface_nodes_np,
                    surface_faces=surface_cells.connectivity,
                    bounding_box={
                        'min': np.array([bounding_box[0], bounding_box[2], bounding_box[4]]),
                        'max': np.array([bounding_box[1], bounding_box[3], bounding_box[5]])
                    },
                    method="extrusion",
                    surface_boundaries=boundaries,
                    use_hybrid_mesh=True  # Enable hybrid mesh (BL + background)
                )
                
                # Save original surface mesh data
                volume_mesh.surface_mesh = {
                    'nodes': surface_nodes_np,
                    'faces': surface_cells.connectivity,
                    'boundaries': boundaries
                }
                
                logger.success(
                    f"Volume mesh generated: {volume_mesh.node_count} nodes, "
                    f"{volume_mesh.cell_count} cells, "
                    f"total volume: {volume_mesh.total_volume:.6e} m^3"
                )
                
                return volume_mesh
            else:
                # Use surface mesh
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
        """计算表面网格的精确包围盒（不加buffer）

        The input surface mesh is a closed, watertight domain boundary (car
        body + ground + inlet/outlet + tunnel/farfield) - it already defines
        the exact computational domain, so this returns the plain node
        extent with no padding. Padding it (as this used to do) would let
        the volume mesh extend outside the domain the surface actually
        encloses (e.g. straight through the ground plane). The result is
        used only for boundary-group classification (mesh_domain_classify)
        and as a BL growth-cap reference, never to define fill geometry.

        Returns:
            Tuple of (min_x, max_x, min_y, max_y, min_z, max_z)
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
