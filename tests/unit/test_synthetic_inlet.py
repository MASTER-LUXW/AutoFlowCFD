"""合成湍流入口 SEM（`boundary/synthetic_inlet.py`，Jarrin 2008）。

修复前（I-71 / I-74）：影响盒只沿与来流夹角最小的坐标轴外扩一个涡核尺度、横向不外扩且与坐标轴对齐（入口横向
边缘的雷诺应力只复现约 1/2、角点约 1/4）；涡核尺度取"入口包围盒最大跨度 / 10"，与来流湍流无关；每个面每次
求值在 Python 里遍历全部涡核；目标张量分解失败时静默退回对角近似。
"""

import numpy as np
import pytest

from autoflowcfd.boundary.synthetic_inlet import (
    SyntheticEddyMethod, inlet_frame, reynolds_stress_factor, sem_length_scale)


def _square_inlet(n=21, half=0.5):
    """x=0 平面上的方形入口网格点（含边与角）。"""
    y, z = np.meshgrid(np.linspace(-half, half, n), np.linspace(-half, half, n), indexing="ij")
    return np.column_stack([np.zeros(y.size), y.ravel(), z.ravel()])


def _brute_force(sem, points, sigma):
    """逐涡核逐点的直接实现（Jarrin 2008 式 4.5 / 4.8）：向量化求值的独立参照。"""
    xi = sem.to_local(points)
    out = np.zeros((points.shape[0], 3))
    for k in range(sem.num_eddies):
        s = (xi - sem.eddy_centers[k]) / sigma[:, None]
        f = np.where(np.abs(s) < 1.0, np.sqrt(1.5) * (1.0 - np.abs(s)), 0.0).prod(axis=1)
        out += sem.eddy_signs[k] * (f * np.sqrt(sem.volume / sigma ** 3))[:, None]
    return out / np.sqrt(sem.num_eddies)


def test_frame_is_right_handed_orthonormal_with_e1_along_the_flow():
    d = np.array([0.8, 0.3, -0.52])
    E = inlet_frame(d)
    np.testing.assert_allclose(E @ E.T, np.eye(3), atol=1e-15)
    assert np.linalg.det(E) == pytest.approx(1.0)
    np.testing.assert_allclose(E[0], d / np.linalg.norm(d))


def test_box_is_in_the_inlet_frame_and_extends_sigma_in_all_three_directions():
    """倾斜入口 + 攻角来流：盒子在 e1 坐标系里包住全部点、三个方向各外扩 sigma_max。"""
    rng = np.random.default_rng(1)
    pts = _square_inlet(9) @ np.array([[0.9, 0.1, 0.0], [-0.1, 0.9, 0.3], [0.0, -0.3, 0.9]])
    sigma = rng.uniform(0.02, 0.05, len(pts))
    flow = np.array([1.0, 0.4, 0.2])
    sem = SyntheticEddyMethod(pts, sigma, flow, num_eddies=50, seed=0)
    xi = sem.to_local(pts)
    np.testing.assert_allclose(sem.box_min, xi.min(axis=0) - sigma.max(), rtol=1e-14)
    np.testing.assert_allclose(sem.box_max, xi.max(axis=0) + sigma.max(), rtol=1e-14)
    assert np.all(sem.eddy_centers >= sem.box_min) and np.all(sem.eddy_centers <= sem.box_max)


def test_vectorized_evaluation_equals_the_direct_sum():
    rng = np.random.default_rng(2)
    pts = _square_inlet(15)
    sigma = rng.uniform(0.05, 0.12, len(pts))
    sem = SyntheticEddyMethod(pts, sigma, [1.0, 0.0, 0.0], num_eddies=400, seed=3)
    for _ in range(3):
        got = sem.unit_fluctuations(pts, sigma)
        np.testing.assert_allclose(got, _brute_force(sem, pts, sigma), rtol=1e-12, atol=1e-12)
        sem.advance(2e-3, [30.0, 0.0, 0.0])


def test_reynolds_stress_is_recovered_at_edges_and_corners():
    """入口中心、边中点、角点的脉动方差都恢复到目标值（修复前边约 1/2、角约 1/4）。

    独立快照：每次重新撒涡核（不同种子），统计 600 个快照的方差；每个快照的方差估计相互独立，相对标准误约 6%。
    """
    pts = _square_inlet(11)
    sigma = np.full(len(pts), 0.1)
    centre, edge, corner = 60, 5, 0                     # (0,0)、(-0.5,0)、(-0.5,-0.5)
    assert np.allclose(pts[[centre, edge, corner], 1:], [[0, 0], [-0.5, 0], [-0.5, -0.5]])
    samples = np.array([SyntheticEddyMethod(pts, sigma, [1.0, 0.0, 0.0], seed=s).unit_fluctuations(pts, sigma)
                        for s in range(600)])
    var = samples.var(axis=0)                           # (n_points, 3)
    for p in (centre, edge, corner):
        np.testing.assert_allclose(var[p], 1.0, rtol=0.2)
    cov = np.einsum("spi,spj->ij", samples, samples) / samples.shape[0] / samples.shape[1]
    assert np.abs(cov - np.diag(np.diag(cov))).max() < 0.1


def test_default_eddy_count_follows_the_jarrin_density_criterion(caplog):
    pts = _square_inlet(5)
    sigma = np.linspace(0.05, 0.1, len(pts))
    sem = SyntheticEddyMethod(pts, sigma, [1.0, 0.0, 0.0], seed=0)
    assert sem.num_eddies == sem.recommended_num_eddies == int(np.ceil(sem.volume / 0.05 ** 3))
    from loguru import logger
    handler = logger.add(lambda m: caplog.records.append(m), level="WARNING")
    try:
        SyntheticEddyMethod(pts, sigma, [1.0, 0.0, 0.0], num_eddies=10, seed=0)
    finally:
        logger.remove(handler)
    assert any("密度判据" in str(r) for r in caplog.records)


def test_wraparound_keeps_the_eddies_uniform_in_the_box():
    """长时间对流后涡核仍均匀分布在盒内（越过下游面的按越过距离回绕，不在上游面堆积）。"""
    pts = _square_inlet(5)
    sigma = np.full(len(pts), 0.1)
    sem = SyntheticEddyMethod(pts, sigma, [1.0, 0.0, 0.0], num_eddies=4000, seed=5)
    for _ in range(200):
        sem.advance(1.3e-3, [30.0, 0.0, 0.0])          # 每步 0.039 m，盒深 0.2 m
        assert np.all(sem.eddy_centers >= sem.box_min) and np.all(sem.eddy_centers <= sem.box_max)
    xi1 = (sem.eddy_centers[:, 0] - sem.box_min[0]) / (sem.box_max[0] - sem.box_min[0])
    counts, _ = np.histogram(xi1, bins=10, range=(0.0, 1.0))
    assert np.abs(counts / 400.0 - 1.0).max() < 0.2


def test_fluctuations_change_after_advance_and_evaluation_is_cached_per_step():
    from autoflowcfd.boundary.fr_ghost_state import InletSEMGhostState

    pts = _square_inlet(5)
    sigma = np.full(len(pts), 0.2)
    sem = SyntheticEddyMethod(pts, sigma, [1.0, 0.0, 0.0], seed=1)
    ghost = InletSEMGhostState(sem, {7: pts[:4], 9: pts[4:8]}, {7: sigma[:4], 9: sigma[4:8]},
                               Q_mean=np.array([1.2, 20.0, 0.0, 0.0, 1e5]), reynolds_stress=np.eye(3))
    Q = np.tile([1.2, 20.0, 0.0, 0.0, 1e5], (2, 4, 1))
    g1 = ghost.ghost_states(np.array([7, 9]), Q)
    calls = []
    orig = sem.unit_fluctuations
    sem.unit_fluctuations = lambda *a: calls.append(1) or orig(*a)
    np.testing.assert_array_equal(ghost.ghost_states(np.array([9]), Q[:1]), g1[1:])   # 同一步：不重算
    assert not calls
    sem.advance(1e-3, [20.0, 0.0, 0.0])
    g2 = ghost.ghost_states(np.array([7, 9]), Q)
    assert len(calls) == 1 and not np.allclose(g1, g2)
    np.testing.assert_array_equal(g1[..., [0, 4]], np.broadcast_to([1.2, 1e5], (2, 4, 2)))


def test_length_scale_is_the_dissipation_scale_bounded_by_resolution():
    k, omega = 0.135, 1800.0
    ell = np.sqrt(k) / (0.09 * omega)
    np.testing.assert_allclose(sem_length_scale(k, omega, 0.09, np.array([ell / 2, 3 * ell])), [ell, 3 * ell])


def test_reynolds_stress_factor_is_strict():
    R = np.array([[4.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    a = reynolds_stress_factor(R)
    np.testing.assert_allclose(a @ a.T, R, rtol=1e-14)
    assert not np.any(reynolds_stress_factor(np.zeros((3, 3))))
    with pytest.raises(np.linalg.LinAlgError):
        reynolds_stress_factor(np.diag([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError, match="不对称"):
        reynolds_stress_factor(np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
