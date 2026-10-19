"""
AutoFlowCFD Grid - 高阶网格模块

高阶网格模块，包含高阶网格数据结构和阶数管理。
"""

from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh
from autoflowcfd.grid.high_order.high_order_mesh_order import (
    build_order_geometry,
    set_order,
)

__all__ = [
    'HighOrderMesh',
    'build_order_geometry',
    'set_order',
]
