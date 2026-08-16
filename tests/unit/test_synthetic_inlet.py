"""AutoFlowCFD V2.0 - 合成湍流入口 (SEM) 单元测试 (BD-02)。"""

import numpy as np

from autoflowcfd.boundary.synthetic_inlet import SyntheticEddyMethod


def test_fluctuations_are_time_varying():
    """同一批 SPs 坐标，在 advance() 后应给出不同的脉动场（不是静态噪声）。"""
    rng = np.random.default_rng(0)
    sem = SyntheticEddyMethod(num_eddies=100, length_scale=0.1, seed=1)
    pos = np.column_stack([np.zeros(30), rng.uniform(-0.3, 0.3, 30), rng.uniform(-0.2, 0.2, 30)])
    mean_u = np.array([20.0, 0.0, 0.0])
    R = np.diag([1.0, 1.0, 1.0])
    sem.configure_inlet_box(pos, flow_direction=mean_u)

    u1 = sem.generate_fluctuations(pos, mean_u, R)
    sem.advance(dt=1e-3, mean_velocity=mean_u)
    u2 = sem.generate_fluctuations(pos, mean_u, R)

    assert not np.allclose(u1, u2)


def test_eddy_box_matches_real_inlet_geometry():
    """涡核影响区必须由传入的真实入口坐标决定，而不是硬编码范围。"""
    rng = np.random.default_rng(2)
    sem = SyntheticEddyMethod(num_eddies=50, length_scale=0.05, seed=3)
    pos = np.column_stack([np.full(20, 5.0), rng.uniform(100.0, 100.5, 20), rng.uniform(-1.0, -0.5, 20)])
    sem.configure_inlet_box(pos, flow_direction=np.array([1.0, 0.0, 0.0]))

    assert sem.box_min[1] <= pos[:, 1].min() and sem.box_max[1] >= pos[:, 1].max()
    assert sem.box_min[2] <= pos[:, 2].min() and sem.box_max[2] >= pos[:, 2].max()
    assert np.all(sem.eddy_centers[:, 1] >= sem.box_min[1] - 1e-9)
    assert np.all(sem.eddy_centers[:, 1] <= sem.box_max[1] + 1e-9)


def test_statistical_reynolds_stress_recovery():
    """多个时间快照统计出的协方差应逼近目标雷诺应力张量（对角占优情形）。"""
    rng = np.random.default_rng(42)
    sem = SyntheticEddyMethod(num_eddies=300, length_scale=0.08, seed=1)
    pos = np.column_stack([np.zeros(400), rng.uniform(-0.5, 0.5, 400), rng.uniform(-0.3, 0.3, 400)])
    mean_u = np.array([30.0, 0.0, 0.0])
    sem.configure_inlet_box(pos, flow_direction=mean_u)
    R_target = np.diag([4.0, 2.0, 1.0])

    samples = []
    for _ in range(150):
        sem.advance(dt=2e-4, mean_velocity=mean_u)
        u = sem.generate_fluctuations(pos, mean_u, R_target)
        samples.append(u - mean_u)
    samples = np.concatenate(samples, axis=0)
    cov = np.cov(samples.T)

    rel_err = np.abs(np.diag(cov) - np.diag(R_target)) / np.diag(R_target)
    assert np.all(rel_err < 0.2), f"Reynolds stress recovery error too large: {rel_err}"
    # 目标是对角张量，非对角项应显著小于对角项
    off_diag_max = max(abs(cov[0, 1]), abs(cov[0, 2]), abs(cov[1, 2]))
    assert off_diag_max < 0.5 * np.min(np.diag(R_target))


def test_regenerated_eddies_stay_within_box():
    """对流足够久之后，所有涡核都应经历过再生，且始终留在影响区内。"""
    rng = np.random.default_rng(5)
    sem = SyntheticEddyMethod(num_eddies=60, length_scale=0.05, seed=7)
    pos = np.column_stack([np.zeros(10), rng.uniform(-0.2, 0.2, 10), rng.uniform(-0.2, 0.2, 10)])
    mean_u = np.array([50.0, 0.0, 0.0])
    sem.configure_inlet_box(pos, flow_direction=mean_u)

    for _ in range(500):
        sem.advance(dt=1e-3, mean_velocity=mean_u)

    assert np.all(sem.eddy_centers >= sem.box_min[np.newaxis, :] - 1e-9)
    assert np.all(sem.eddy_centers <= sem.box_max[np.newaxis, :] + 1e-9)
