"""AutoFlowCFD V2.0 - FR 边界幽灵态构造单元测试 (BD-01)。"""

import numpy as np
import pytest

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


def test_inlet_ghost_is_the_inlet_state_whatever_the_interior_flow_direction():
    """入口幽灵态恒为给定的入口状态。此前本侧法向流出时改用内部状态——在法向速度过零处间断。"""
    Q_inlet = np.array([1.2, -10.0, 0.0, 0.0, 1e5])
    Q_int = np.array([[1.0, -5.0, 0.0, 0.0, 9e4],          # 本侧流入
                      [1.0, 5.0, 0.0, 0.0, 9e4],           # 本侧流出
                      [1.0, 0.0, 3.0, 0.0, 9e4]])          # 法向速度为零
    np.testing.assert_array_equal(inlet_ghost_state(Q_int, Q_inlet), np.tile(Q_inlet, (3, 1)))
    per_point = np.array([[1.2, -10.0, 0.0, 0.0, 1e5], [1.1, -9.0, 1.0, 0.0, 1e5], [1.3, -8.0, 0.0, 2.0, 1e5]])
    ghost = inlet_ghost_state(Q_int, per_point)
    np.testing.assert_array_equal(ghost, per_point)
    assert ghost is not per_point


def test_outlet_pressure_is_static_on_outflow_and_total_on_backflow():
    normal = np.array([[1.0, 0.0, 0.0]])
    p_outlet = 101325.0

    Q_out = np.array([[1.0, 5.0, 2.0, 0.0, 9e4]])  # 流出
    ghost = outlet_ghost_state(Q_out, p_outlet, normal)
    assert ghost[0, 4] == p_outlet
    np.testing.assert_array_equal(ghost[0, :4], Q_out[0, :4])   # 密度与速度延拓

    Q_back = np.array([[1.2, -5.0, 2.0, 0.0, 9e4]])  # 回流：给定值按总压解释，扣掉法向动压
    ghost = outlet_ghost_state(Q_back, p_outlet, normal)
    assert ghost[0, 4] == pytest.approx(p_outlet - 0.5 * 1.2 * 25.0)
    np.testing.assert_array_equal(ghost[0, :4], Q_back[0, :4])


def test_inlet_and_outlet_ghosts_are_smooth_through_zero_normal_velocity():
    """幽灵态对本侧法向速度连续可微。此前两者都在 u_n = 0 处跳变（入口：入口状态 <-> 内部状态；出口：
    p_outlet <-> 内部压力），与无滑移壁相交处 u_n 在零附近，残差不连续，Newton 不收敛
    （`test_dual_time_newton.py::test_inner_iterations_converge_on_a_wall_bounded_channel`）。"""
    normal = np.array([[1.0, 0.0, 0.0]])
    Q_inlet = np.array([1.2, 30.0, 0.0, 0.0, 1e5])

    def at(un):
        return np.array([[1.1, un, 4.0, -2.0, 9e4]])

    eps = 1e-6
    for ghost in (lambda q: inlet_ghost_state(q, Q_inlet), lambda q: outlet_ghost_state(q, 101325.0, normal)):
        g0, gp, gm = ghost(at(0.0)), ghost(at(eps)), ghost(at(-eps))
        # 连续：跨零的变化是 O(eps)
        assert np.abs(gp - g0).max() <= 2.0 * eps and np.abs(gm - g0).max() <= 2.0 * eps
        # 一阶导数连续：两侧单边差商一致
        np.testing.assert_allclose((gp - g0) / eps, (g0 - gm) / eps, rtol=0, atol=1e-5)


def test_symmetry_mirrors_normal_velocity_only():
    Q_int = np.array([[1.2, 30.0, 5.0, -2.0, 1e5]])
    normal = np.array([[0.0, 1.0, 0.0]])
    Q_ghost = symmetry_ghost_state(Q_int, normal)
    avg_vel = 0.5 * (Q_int[:, 1:4] + Q_ghost[:, 1:4])
    assert abs(avg_vel[0, 1]) < 1e-10  # 法向分量平均为零
    assert np.isclose(avg_vel[0, 0], Q_int[0, 1])  # 切向分量不变
    assert np.isclose(avg_vel[0, 2], Q_int[0, 3])
