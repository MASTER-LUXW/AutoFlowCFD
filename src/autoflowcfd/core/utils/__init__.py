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
from autoflowcfd.core.utils.wall_distance import compute_wall_distance
from autoflowcfd.core.utils.aero_coeffs import ReferenceAreaMixin
from autoflowcfd.core.utils.solver_helpers import resolve_backend_type

__all__ = [
    'CheckpointManager',
    'interpolate_to_new_order',
    'run_order_continuation',
    'greedy_face_coloring',
    'compute_wall_distance',
    'ReferenceAreaMixin',
    'resolve_backend_type',
]
