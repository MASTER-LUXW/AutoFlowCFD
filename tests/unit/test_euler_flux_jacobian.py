"""解析欧拉通量 Jacobian（`core/fr_residual/jacobian/pointwise.py::euler_flux_jacobian`）对逐点通量核的中心差分。

把关两件事：解析式本身正确；日后改动通量核（`flux_kernels/euler.py`）而忘记同步 Jacobian 时这里失败。
2026-10-22（I-25）以前 Jacobian 对通量做前向差分（见该模块文档）。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.flux_kernels import euler_physical_flux_point
from autoflowcfd.core.fr_residual.jacobian.pointwise import euler_flux_jacobian


def _states(seed, n=32):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0.1, 5.0, n), rng.uniform(-300, 300, (n, 3)), rng.uniform(1e3, 5e5, n)])


def _central_difference(q, v):
    """四阶中心差分（多项式通量的高阶导数有界，误差远小于判据）。"""
    h = 1e-3 * (abs(q[v]) + (np.linalg.norm(q[1:4]) + 1.0 if 1 <= v <= 3 else 0.0))
    d = np.zeros((3, 5))
    for c, s in ((8.0, 1.0), (-8.0, -1.0), (-1.0, 2.0), (1.0, -2.0)):
        x = q.copy()
        x[v] += s * h
        d += c * euler_physical_flux_point(x)
    return d / (12.0 * h)


@pytest.mark.parametrize("seed", range(4))
def test_matches_central_difference_of_the_flux_kernel(seed):
    Q = _states(seed)
    J = euler_flux_jacobian(Q)
    assert J.shape == (len(Q), 3, 5, 5)
    for k, q in enumerate(Q):
        ref = np.stack([_central_difference(q, v) for v in range(5)], axis=-1)       # (3, 5, 5)
        np.testing.assert_allclose(J[k], ref, rtol=1e-9, atol=1e-9 * np.abs(ref).max())

