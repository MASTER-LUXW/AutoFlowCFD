"""Q 准则（`postprocess/q_criterion.py`）。

此前的实现对单元平均速度做 Green–Gauss，却只向 owner 累加面通量、内部面没给 neighbor 减回：
两四面体网格上常数速度场在 neighbor 单元算出梯度 1.5（应为 0）。现在速度梯度与湍流模型同一实现，
解点上的 Q 按求积权重求单元积分平均。
"""

import numpy as np
import pytest

from autoflowcfd.postprocess.q_criterion import q_criterion_cell_field, q_criterion_from_velocity_gradient


def test_rigid_rotation_and_pure_strain():
    w = 3.0
    rot = np.array([[0.0, -w, 0.0], [w, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert q_criterion_from_velocity_gradient(rot) == pytest.approx(w * w)
    strain = np.diag([w, -w, 0.0])
    assert q_criterion_from_velocity_gradient(strain) == pytest.approx(-w * w)
    shear = np.array([[0.0, w, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert q_criterion_from_velocity_gradient(shear) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("order", [1, 2])
def test_cell_field_of_rigid_rotation_is_exact(order):
    """线性速度场 u = (-w y, w x, 0)：P>=1 单元内导数精确，每个单元 Q = w^2；常数场每个单元 Q = 0。"""
    from tests.unit.test_corrected_velocity_gradient import _couette_solver

    solver, Q = _couette_solver(order)
    w = 40.0
    xyz = np.asarray(solver.mesh.sps_coords)
    Q[..., 1] = -w * xyz[..., 1]
    Q[..., 2] = w * xyz[..., 0]
    Q[..., 3] = 0.0
    solver.state.Q = Q
    np.testing.assert_allclose(q_criterion_cell_field(solver), w * w, rtol=1e-9)

    Q[..., 1:4] = [12.0, -3.0, 5.0]
    solver.state.Q = Q
    np.testing.assert_allclose(q_criterion_cell_field(solver), 0.0, atol=1e-8)
