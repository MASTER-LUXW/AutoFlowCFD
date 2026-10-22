"""`postprocess/solution_vector.py`：后处理用的单元平均守恒解（I-93 从 core/backend/ 移来）。"""

import numpy as np
import pytest

from autoflowcfd.core.thermo import GAMMA
from autoflowcfd.postprocess.solution_vector import SolutionVector


def _conserved(rho, u, p):
    rho, u, p = (np.asarray(x, dtype=np.float64) for x in (rho, u, p))
    U = np.zeros((len(rho), 5))
    U[:, 0] = rho
    U[:, 1:4] = rho[:, None] * u
    U[:, 4] = p / (GAMMA - 1.0) + 0.5 * rho * np.sum(u ** 2, axis=1)
    return U


def test_primitive_variables_round_trip():
    u = np.array([[10.0, -2.0, 0.5], [0.0, 3.0, -4.0]])
    p = np.array([101325.0, 9.0e4])
    sol = SolutionVector(_conserved([1.2, 0.8], u, p))
    assert sol.n_cells == 2
    np.testing.assert_allclose(np.column_stack(sol.get_velocity()), u, rtol=1e-14)
    np.testing.assert_allclose(sol.get_pressure(), p, rtol=1e-14)


@pytest.mark.parametrize("data", [np.ones(5), np.ones((3, 4)), np.ones((2, 5, 1))])
def test_rejects_arrays_that_are_not_cells_by_conserved_variables(data):
    with pytest.raises(ValueError, match="守恒解"):
        SolutionVector(data)


def test_legacy_sst_checkpoint_columns_are_dropped():
    """2026-10-05 以前的 SST checkpoint 多两列（从未更新的历史槽位），只取前 5 列。"""
    U = _conserved([1.0], np.array([[5.0, 0.0, 0.0]]), [1.0e5])
    sol = SolutionVector(np.hstack([U, [[123.0, 456.0]]]))
    assert sol.data.shape == (1, 5)
    np.testing.assert_allclose(sol.get_pressure(), [1.0e5], rtol=1e-14)


def test_non_positive_density_is_reported_not_clamped():
    """此前密度被钳到 1e-10 再做除法：rho < 0 的单元速度被放大约 1e10 倍且符号翻转——发散的解被一个下限
    掩盖成看似有限的值。现在按原值换算。"""
    U = np.array([[-0.5, 1.0, 0.0, 0.0, 2.5e5]])
    u, _v, _w = SolutionVector(U).get_velocity()
    np.testing.assert_allclose(u, [-2.0], rtol=1e-14)
