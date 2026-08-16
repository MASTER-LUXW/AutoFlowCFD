"""AutoFlowCFD V2.0 - FR 边界幽灵态构造单元测试 (BD-01)。"""

import numpy as np

from autoflowcfd.boundary.fr_ghost_state import (
    farfield_ghost_state,
    inlet_ghost_state,
    outlet_ghost_state,
    symmetry_ghost_state,
    wall_ghost_state,
)


def test_wall_no_slip_average_velocity_is_wall_velocity():
    """无滑移壁面：界面（L,R 平均）速度应恰好等于壁面速度。"""
    Q_int = np.array([[1.2, 30.0, 5.0, -2.0, 1e5]])
    normal = np.array([[0.0, 1.0, 0.0]])
    Q_ghost = wall_ghost_state(Q_int, normal, is_no_slip=True)
    avg_vel = 0.5 * (Q_int[:, 1:4] + Q_ghost[:, 1:4])
    assert np.allclose(avg_vel, 0.0, atol=1e-10)


def test_wall_slip_normal_velocity_cancels():
    """滑移壁面：界面法向速度应为零，切向速度不受影响。"""
    Q_int = np.array([[1.2, 30.0, 5.0, -2.0, 1e5]])
    normal = np.array([[0.0, 1.0, 0.0]])
    Q_ghost = wall_ghost_state(Q_int, normal, is_no_slip=False)
    avg_vel = 0.5 * (Q_int[:, 1:4] + Q_ghost[:, 1:4])
    avg_vel_n = np.sum(avg_vel * normal, axis=1)
    assert abs(avg_vel_n[0]) < 1e-10
    # 切向分量 (x,z) 不变
    assert np.isclose(Q_ghost[0, 1], Q_int[0, 1])
    assert np.isclose(Q_ghost[0, 3], Q_int[0, 3])


def test_farfield_returns_freestream_everywhere():
    Q_int = np.random.default_rng(0).uniform(size=(5, 5))
    Q_free = np.array([1.225, 30.0, 0.0, 0.0, 101325.0])
    Q_ghost = farfield_ghost_state(Q_int, Q_free)
    assert np.allclose(Q_ghost, np.tile(Q_free, (5, 1)))


def test_inlet_uses_inlet_state_on_inflow_and_interior_on_outflow():
    normal = np.array([[1.0, 0.0, 0.0]])  # 指向域外 (+x)
    Q_inlet = np.array([1.2, -10.0, 0.0, 0.0, 1e5])

    Q_int_inflow = np.array([[1.0, -5.0, 0.0, 0.0, 9e4]])  # un = -5 < 0 -> 流入
    Q_ghost = inlet_ghost_state(Q_int_inflow, Q_inlet, normal)
    assert np.allclose(Q_ghost[0], Q_inlet)

    Q_int_outflow = np.array([[1.0, 5.0, 0.0, 0.0, 9e4]])  # un = 5 > 0 -> 流出
    Q_ghost2 = inlet_ghost_state(Q_int_outflow, Q_inlet, normal)
    assert np.allclose(Q_ghost2[0], Q_int_outflow[0])


def test_outlet_fixes_pressure_on_outflow_only():
    normal = np.array([[1.0, 0.0, 0.0]])
    p_outlet = 101325.0

    Q_int_out = np.array([[1.0, 5.0, 0.0, 0.0, 9e4]])  # 流出
    Q_ghost = outlet_ghost_state(Q_int_out, p_outlet, normal)
    assert np.isclose(Q_ghost[0, 4], p_outlet)
    assert np.isclose(Q_ghost[0, 0], Q_int_out[0, 0])  # 密度延拓不变

    Q_int_in = np.array([[1.0, -5.0, 0.0, 0.0, 9e4]])  # 回流
    Q_ghost2 = outlet_ghost_state(Q_int_in, p_outlet, normal)
    assert np.isclose(Q_ghost2[0, 4], Q_int_in[0, 4])  # 回流时延拓内部压力，不强加出口压力


def test_symmetry_mirrors_normal_velocity_only():
    Q_int = np.array([[1.2, 30.0, 5.0, -2.0, 1e5]])
    normal = np.array([[0.0, 1.0, 0.0]])
    Q_ghost = symmetry_ghost_state(Q_int, normal)
    avg_vel = 0.5 * (Q_int[:, 1:4] + Q_ghost[:, 1:4])
    assert abs(avg_vel[0, 1]) < 1e-10  # 法向分量平均为零
    assert np.isclose(avg_vel[0, 0], Q_int[0, 1])  # 切向分量不变
    assert np.isclose(avg_vel[0, 2], Q_int[0, 3])
