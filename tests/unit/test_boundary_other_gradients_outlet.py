"""出口（Neumann）"另一侧"梯度的法向镜像只在 `boundary_other_gradients` 一处（I-16）：此前写在
`viscous_common_flux_point` 里、覆盖调用方传入的值，同一语义两处。"""

import numpy as np

from autoflowcfd.core.fr_operators.flux_kernels import (
    VBC_NEUMANN, boundary_other_gradients, mirror_normal_component, mirror_normal_derivative,
)


def test_outlet_mirrors_only_normal_derivatives():
    rng = np.random.default_rng(5)
    gv, gT, a = rng.standard_normal((3, 3)), rng.standard_normal(3), np.array([0.3, -1.2, 0.5])
    gv_x, gT_x = boundary_other_gradients(gv, gT, a, VBC_NEUMANN)
    np.testing.assert_array_equal(gv_x, mirror_normal_derivative(gv, a))
    np.testing.assert_array_equal(gT_x, mirror_normal_component(gT, a))
    n = a / np.linalg.norm(a)
    np.testing.assert_allclose(0.5 * (gv + gv_x) @ n, 0.0, atol=1e-14)       # 平均后法向导数为零
    np.testing.assert_allclose(0.5 * (gT + gT_x) @ n, 0.0, atol=1e-14)
