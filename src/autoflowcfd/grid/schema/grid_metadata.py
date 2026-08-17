"""网格元数据结构。

提供 GridMetadata 类，存储网格信息，包括数量统计、
格式版本和统计数据。
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class GridMetadata:
    """网格元数据
    
    包含网格的元数据，包括数量统计、格式版本和统计信息。
    
    属性:
        node_count: 节点数量
        cell_count: 单元数量
        boundary_groups: 边界组名称列表
        file_format: 文件格式版本 (e.g., "v22", "v23", "v24")
        bounding_box: 包围盒信息 (min_x, max_x, min_y, max_y, min_z, max_z), optional -
            与 parser_core.py 的 _compute_bounding_box() 输出顺序一致
        creation_time: 网格创建时间戳, optional
    
    示例:
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
        
        抛出异常:
            ValueError: 如果数量为负数
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
            str: 可读的摘要字符串
        """
        lines = [
            "Grid Metadata Summary:",
            f"  Format: {self.file_format}",
            f"  Nodes: {self.node_count:,}",
            f"  Cells: {self.cell_count:,}",
            f"  Boundary Groups: {len(self.boundary_groups)}",
        ]
        
        if self.bounding_box:
            # 顺序与生成端（parser_core.py 的 _compute_bounding_box）一致：
            # (min_x, max_x, min_y, max_y, min_z, max_z)，
            # 而非按轴分组的顺序——那种不匹配会静默错标每个轴。
            min_x, max_x, min_y, max_y, min_z, max_z = self.bounding_box
            lines.append(
                f"  Bounding Box: [{min_x:.3f}, {max_x:.3f}] x "
                f"[{min_y:.3f}, {max_y:.3f}] x [{min_z:.3f}, {max_z:.3f}]"
            )
        
        if self.creation_time:
            lines.append(f"  Created: {self.creation_time}")
        
        return "\n".join(lines)
