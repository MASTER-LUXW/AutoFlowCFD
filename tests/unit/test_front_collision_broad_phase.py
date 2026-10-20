"""BL 前沿自交检测的广相位完备、冻结不静默放过残留相交（I-153）。

此前候选半径 = 2 x sqrt(查询面面积)：细长三角形顶点离质心远大于 sqrt(面积)，两个针形面在尖端附近相交时两侧都查
不到对方；freeze_self_colliding_nodes 迭代 5 次后、或相交面的节点全部已冻结时静默返回。
"""

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.utils.mesh_front_collision import freeze_self_colliding_nodes
from autoflowcfd.grid.mesh_gen.utils.mesh_front_collision_detect import (
    find_cross_state_colliding_faces, find_self_colliding_faces,
)

# 针 A：沿 x 长 1、宽 0.01（质心 x=1/3）；针 B：在 x=0.95 的竖直平面内沿 z 长 2，在 A 的尖端附近穿过 A
_NODES = np.array([[0, -0.005, 0], [0, 0.005, 0], [1, 0, 0],
                   [0.95, -0.002, -1], [0.95, 0.002, -1], [0.95, 0, 1]], float)
_FACES = np.array([[0, 1, 2], [3, 4, 5]])


def test_needle_triangles_crossing_near_their_tips_are_found():
    assert find_self_colliding_faces(_NODES, _FACES).tolist() == [0, 1]


def test_cross_state_needles_are_found():
    current = _NODES.copy()
    new = _NODES.copy()
    new[3:, 0] += 5.0                                  # B 的新位置远离；A 的新位置与 B 的旧位置相交
    assert find_cross_state_colliding_faces(new, current, _FACES).tolist() == [0, 1]


def test_residual_intersection_among_frozen_nodes_is_an_error():
    """相交面的节点全部已冻结：这些面都在上一层的位置上，上一层本身自交——必须报错。"""
    budget = np.zeros(6)
    with pytest.raises(RuntimeError, match="已全部冻结"):
        freeze_self_colliding_nodes(_NODES.copy(), _NODES.copy(), _FACES, budget)
