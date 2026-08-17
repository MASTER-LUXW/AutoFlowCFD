"""
AutoFlowCFD Grid - 网格生成 / 面提取模块

面网格提取模块，从体网格中提取面片并构建面几何信息。
"""

from autoflowcfd.grid.mesh_gen.extraction.face_extractor import (
    FaceExtractor,
    extract_faces_from_tetrahedra,
)

__all__ = [
    'FaceExtractor',
    'extract_faces_from_tetrahedra',
]
