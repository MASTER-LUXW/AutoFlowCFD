"""FVM 核心模块的向后兼容转出。

新代码请直接从 autoflowcfd.core.fvm_faces 导入。
"""

from .fvm_faces import FVMFaceExtractor

__all__ = [
    'FVMFaceExtractor',
]
