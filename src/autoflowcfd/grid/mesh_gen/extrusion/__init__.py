"""
AutoFlowCFD Grid - 网格生成 / 挤出模块

边界层挤出模块，包含层状网格生成和衰减控制。
"""

from autoflowcfd.grid.mesh_gen.extrusion.mesh_extrusion import extrude_layers
from autoflowcfd.grid.mesh_gen.extrusion.mesh_layer_step import extrude_single_layer

__all__ = [
    'extrude_layers',
    'extrude_single_layer',
]
