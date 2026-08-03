"""Grid metadata structures.

Provides GridMetadata class for storing grid information including counts,
format version, and statistical data.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class GridMetadata:
    """网格元数据
    
    Contains metadata about the grid including counts, format version,
    and statistical information.
    
    Attributes:
        node_count: 节点数量
        cell_count: 单元数量
        boundary_groups: 边界组名称列表
        file_format: 文件格式版本 (e.g., "v22", "v23", "v24")
        bounding_box: 包围盒信息 (min_x, max_x, min_y, max_y, min_z, max_z), optional -
            this matches parser_core.py's _compute_bounding_box() producer order
        creation_time: 网格创建时间戳, optional
    
    Example:
        >>> metadata = GridMetadata(
        ...     node_count=1000000,
        ...     cell_count=2000000,
        ...     boundary_groups=["inlet", "outlet", "wall"],
        ...     file_format="v24"
        ... )
        >>> print(f"Grid size: {metadata.node_count} nodes, {metadata.cell_count} cells")
    """
    node_count: int
    cell_count: int
    boundary_groups: List[str]
    file_format: str
    bounding_box: Optional[Tuple[float, float, float, float, float, float]] = None
    creation_time: Optional[str] = None
    
    def __post_init__(self):
        """验证元数据合理性
        
        Raises:
            ValueError: If counts are negative
        """
        if self.node_count < 0:
            raise ValueError(f"Node count cannot be negative: {self.node_count}")
        if self.cell_count < 0:
            raise ValueError(f"Cell count cannot be negative: {self.cell_count}")
        
        logger.debug(
            f"GridMetadata: {self.node_count} nodes, {self.cell_count} cells, "
            f"format={self.file_format}"
        )
    
    def summary(self) -> str:
        """生成网格元数据摘要
        
        Returns:
            str: Human-readable summary string
        """
        lines = [
            f"Grid Metadata Summary:",
            f"  Format: {self.file_format}",
            f"  Nodes: {self.node_count:,}",
            f"  Cells: {self.cell_count:,}",
            f"  Boundary Groups: {len(self.boundary_groups)}",
        ]
        
        if self.bounding_box:
            # Order matches the producer (parser_core.py's
            # _compute_bounding_box): (min_x, max_x, min_y, max_y, min_z,
            # max_z), NOT the grouped-by-axis order this used to assume -
            # that mismatch silently mislabeled every axis in this summary.
            min_x, max_x, min_y, max_y, min_z, max_z = self.bounding_box
            lines.append(
                f"  Bounding Box: [{min_x:.3f}, {max_x:.3f}] x "
                f"[{min_y:.3f}, {max_y:.3f}] x [{min_z:.3f}, {max_z:.3f}]"
            )
        
        if self.creation_time:
            lines.append(f"  Created: {self.creation_time}")
        
        return "\n".join(lines)
