"""
AutoFlowCFD Core - Utils Module

辅助工具模块，包含 Checkpoint、Order Continuation、面图着色、壁面距离等工具函数。
"""

from autoflowcfd.core.utils.checkpoint import CheckpointManager
from autoflowcfd.core.utils.order_continuation import (
    interpolate_to_new_order,
    run_order_continuation,
)
from autoflowcfd.core.utils.face_coloring import greedy_face_coloring
from autoflowcfd.core.utils.wall_distance import WallDistanceSource
from autoflowcfd.core.utils.aero_coeffs import ReferenceAreaMixin

__all__ = [
    'CheckpointManager',
    'interpolate_to_new_order',
    'run_order_continuation',
    'greedy_face_coloring',
    'WallDistanceSource',
    'ReferenceAreaMixin',
]
