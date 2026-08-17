"""
AutoFlowCFD Grid - 网格生成 / 背景网格模块

背景网格生成模块，包含混合网格生成和边界层处理。
"""

from autoflowcfd.grid.mesh_gen.background.mesh_background import generate_hybrid_mesh

__all__ = [
    'generate_hybrid_mesh',
]
