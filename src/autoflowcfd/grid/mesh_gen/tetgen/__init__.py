"""
AutoFlowCFD Grid - 网格生成 / TetGen 模块

四面体网格生成模块，基于 TetGen 库的体网格生成和后处理。
"""

from autoflowcfd.grid.mesh_gen.tetgen.volume_mesh_generator import VolumeMeshGenerator

__all__ = [
    'VolumeMeshGenerator',
]
