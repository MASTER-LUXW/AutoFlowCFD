"""逐点通量核（`core/fr_operators/flux_kernels/`）对照独立写出的参考公式。

参考公式按守恒变量的定义直接写（欧拉：rho E 从守恒量出发、总焓 H = (rho E + p)/rho；粘性：Stokes 应力 +
Fourier 热流），与核里的写法（欧拉按原始变量的不含除法多项式）不同，对照的是公式本身而不是同一段代码。
2026-10-22（I-25）以前对照对象是 src 里另一份向量化实现（`fr_residual/inviscid.py::euler_physical_flux`、
`viscous_flux/pointwise.py::viscous_physical_flux`），那两份已无调用方，删除，参考公式移到这里。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.flux_kernels import (
    euler_physical_flux_batch,
    euler_physical_flux_point,
    viscous_physical_flux_point,
)
from autoflowcfd.core.thermo import GAMMA, R_AIR


def _euler_reference(Q):
    rho, vel, p = Q[0], Q[1:4], Q[4]
    rhoE = p / (GAMMA - 1.0) + 0.5 * rho * vel @ vel
    H = (rhoE + p) / rho
    F = np.zeros((3, 5))
    for i in range(3):
        F[i, 0] = rho * vel[i]
        F[i, 1:4] = rho * vel[i] * vel
        F[i, 1 + i] += p
        F[i, 4] = rho * H * vel[i]
    return F


def _viscous_reference(Q, grad_vel, grad_T, mu, Pr, mu_t, Pr_t):
    mu_total = mu + mu_t
    S = 0.5 * (grad_vel + grad_vel.T)
    tau = 2.0 * mu_total * S - 2.0 / 3.0 * mu_total * np.trace(grad_vel) * np.eye(3)
    cp = GAMMA * R_AIR / (GAMMA - 1.0)
    k = mu * cp / Pr + mu_t * cp / Pr_t
    G = np.zeros((3, 5))
    G[:, 1:4] = tau
    G[:, 4] = tau @ Q[1:4] + k * grad_T          # tau.u - q，q = -k grad T
    return G


def _random_state(rng):
    return np.array([rng.uniform(0.1, 5.0), *rng.uniform(-50, 50, size=3), rng.uniform(1e3, 5e5)])


@pytest.mark.parametrize("seed", range(20))
def test_euler_physical_flux_point_matches_reference(seed):
    Q = _random_state(np.random.default_rng(seed))
    F = euler_physical_flux_point(Q)
    assert F.shape == (3, 5)
    np.testing.assert_allclose(F, _euler_reference(Q), rtol=1e-14, atol=1e-9)


def test_euler_batch_is_the_point_kernel_row_by_row():
    rng = np.random.default_rng(7)
    Q = np.array([_random_state(rng) for _ in range(64)])
    F = euler_physical_flux_batch(Q)
    for k in range(len(Q)):
        np.testing.assert_array_equal(F[k], euler_physical_flux_point(Q[k]))


def test_euler_flux_takes_density_as_given():
    """不再有密度下限：rho = 1e-14（低于旧下限 1e-10）时能量通量仍是 u (gamma/(gamma-1) p + rho |V|^2/2)。"""
    Q = np.array([1e-14, 3.0, 0.0, 0.0, 2.0])
    F = euler_physical_flux_point(Q)
    np.testing.assert_allclose(F[0, 4], 3.0 * (GAMMA / (GAMMA - 1.0) * 2.0 + 0.5 * 1e-14 * 9.0), rtol=1e-15)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("mu_t,Pr_t", [(0.0, 0.9), (1e-3, 0.9), (5e-2, 0.85)])
def test_viscous_physical_flux_point_matches_reference(seed, mu_t, Pr_t):
    rng = np.random.default_rng(seed)
    Q = _random_state(rng)
    grad_vel = rng.uniform(-10, 10, size=(3, 3))
    grad_T = rng.uniform(-100, 100, size=3)
    mu, Pr = 1.8e-5, 0.72

    G = viscous_physical_flux_point(Q, grad_vel, grad_T, mu, Pr, mu_t, Pr_t)
    assert G.shape == (3, 5)
    np.testing.assert_allclose(G, _viscous_reference(Q, grad_vel, grad_T, mu, Pr, mu_t, Pr_t), rtol=1e-13, atol=1e-12)
