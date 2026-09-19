"""AutoFlowCFD V2.0 - `face_flux_points/locate.py` 重构(共享重心坐标
求解) + 新增 `locate_native_tet_face_point` 单元测试（Part7 阶段2
第二节）。
"""

import numpy as np

from autoflowcfd.fr.face_flux_points.locate import (
    _tet_exact_locate_on_face,
    locate_native_tet_face_point,
)
from autoflowcfd.grid.curved_mapping.curved_mapping import tet_barycentric, cube_to_tet_rst

TET_NODES = np.array([[0.1, -0.3, 0.2], [1.2, 0.0, -0.1], [-0.1, 1.1, 0.0], [0.0, 0.1, 1.3]], dtype=float)


def _random_targets_on_face(cell_nodes, face_vertex_idx, rng, n=5):
    """在给定面的三角形内随机取点（凸组合，保证落在面内）。"""
    p_i, p_j, p_k = cell_nodes[face_vertex_idx[0]], cell_nodes[face_vertex_idx[1]], cell_nodes[face_vertex_idx[2]]
    w = rng.dirichlet([1, 1, 1], size=n)
    return w @ np.array([p_i, p_j, p_k])


def test_tet_exact_locate_on_face_unchanged_after_refactor():
    """回归判据：拆分出共享函数之后，`_tet_exact_locate_on_face` 的
    行为必须与拆分前逐位相同——用真实面（fixed_axis=0,fixed_val=-1.0，
    对应局部顶点 (0,2,3)）上的随机目标点验证往返一致性（这是拆分前后
    都必须满足的性质，用来间接确认拆分没有改变数值结果）。
    """
    rng = np.random.default_rng(0)
    targets = _random_targets_on_face(TET_NODES, (0, 2, 3), rng)
    free_coords = _tet_exact_locate_on_face(TET_NODES, fixed_axis=0, fixed_val=-1.0, targets_phys=targets)

    from autoflowcfd.grid.curved_mapping.curved_mapping import map_tet_to_physical

    full = np.zeros((len(targets), 3))
    full[:, 0] = -1.0
    full[:, 1] = free_coords[:, 0]
    full[:, 2] = free_coords[:, 1]
    phys_back = map_tet_to_physical(full, TET_NODES)
    np.testing.assert_allclose(phys_back, targets, atol=1e-9)


def test_locate_native_tet_face_point_recovers_target_physical_points():
    """native 版本：解出的 (r,s,t) 经 `tet_barycentric` 映射回物理坐标，
    必须精确重合（机器精度）——四个面（排除顶点 0~3）逐一验证。"""
    rng = np.random.default_rng(1)
    for excluded_vertex in range(4):
        face_vertex_idx = tuple(v for v in range(4) if v != excluded_vertex)
        targets = _random_targets_on_face(TET_NODES, face_vertex_idx, rng)

        rst, resid = locate_native_tet_face_point(TET_NODES, excluded_vertex, targets)
        assert resid < 1e-9

        L1, L2, L3, L4 = tet_barycentric(rst[:, 0], rst[:, 1], rst[:, 2])
        p0, p1, p2, p3 = TET_NODES
        phys_back = L1[:, None] * p0 + L2[:, None] * p1 + L3[:, None] * p2 + L4[:, None] * p3
        np.testing.assert_allclose(phys_back, targets, atol=1e-9)

        # 排除的那个顶点对应的重心坐标分量必须恰好为零（点确实落在该面上）
        L = np.column_stack([L1, L2, L3, L4])
        np.testing.assert_allclose(L[:, excluded_vertex], 0.0, atol=1e-9)


def test_locate_native_tet_face_point_raises_on_non_coplanar_target():
    """目标点不在该面上时必须报错，不能静默给出错误结果。"""
    import pytest

    off_face_point = np.array([[0.5, 0.5, 0.5]])  # 四面体内部，不在任何面上
    with pytest.raises(RuntimeError):
        locate_native_tet_face_point(TET_NODES, excluded_vertex=0, targets_phys=off_face_point, char_length=0.1)
