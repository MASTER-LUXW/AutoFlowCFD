"""逐点标量通量 kernel 的隔离验证 (性能优化配套)。

`fr_flux_kernels_pointwise.py` 里的 `euler_physical_flux_point`/
`viscous_physical_flux_point` 是 `fr_residual_inviscid.py::euler_physical_flux`/
`fr_viscous_flux.py::viscous_physical_flux` 的逐点 numba 重写版，供
性能优化后的逐点残差 kernel 使用。这里单独用随机输入对比两版实现，
逐位验证一致——不依赖端到端残差测试来兜底，出问题能直接定位到这里。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.flux_kernels import (
    euler_physical_flux_point,
    viscous_physical_flux_point,
)
from autoflowcfd.core.fr_residual.inviscid import euler_physical_flux
from autoflowcfd.core.fr_residual.viscous_flux import viscous_physical_flux


@pytest.mark.parametrize("seed", range(20))
def test_euler_physical_flux_point_matches_vectorized(seed):
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.1, 5.0)
    u, v, w = rng.uniform(-50, 50, size=3)
    p = rng.uniform(1e3, 5e5)
    Q = np.array([rho, u, v, w, p])

    F_vec = euler_physical_flux(Q)  # (3,5)
    F_point = euler_physical_flux_point(Q)

    assert F_point.shape == (3, 5)
    np.testing.assert_allclose(F_point, F_vec, rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("mu_t,Pr_t", [(0.0, 0.9), (1e-3, 0.9), (5e-2, 0.85)])
def test_viscous_physical_flux_point_matches_vectorized(seed, mu_t, Pr_t):
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.1, 5.0)
    u, v, w = rng.uniform(-50, 50, size=3)
    p = rng.uniform(1e3, 5e5)
    Q = np.array([rho, u, v, w, p])
    grad_vel = rng.uniform(-10, 10, size=(3, 3))
    grad_T = rng.uniform(-100, 100, size=3)
    mu = 1.8e-5
    Pr = 0.72

    G_vec = viscous_physical_flux(Q, grad_vel, grad_T, mu, Pr, mu_t=mu_t, Pr_t=Pr_t)  # (3,5)
    G_point = viscous_physical_flux_point(Q, grad_vel, grad_T, mu, Pr, mu_t, Pr_t)

    assert G_point.shape == (3, 5)
    np.testing.assert_allclose(G_point, G_vec, rtol=0, atol=1e-9)
