"""求解循环至少推进一步（I-80）。

此前 `max_iter=0` 时定阶循环的循环体不执行，收尾的 `i + 1` 抛 NameError（`report_pseudo_time_summary` 与
`SolverResult(iterations=i + 1)`），最终残差是占位值 1e10。现在 `solve` 入口校验 `max_iter >= 1`（定阶循环与
Order Continuation 共用这个入口）。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_solver import FRSolver
from autoflowcfd.core.time_integration import TimeIntegrationScheme
from tests.validation._channel_mesh import build_channel_mesh_prism


def _solver(order):
    mesh = build_channel_mesh_prism(order, 2, 2, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL",
                              "wall_top": "WALL", "z_min": "SYMMETRY", "z_max": "SYMMETRY"}
    return FRSolver(mesh=mesh, order=order, turb_model_name="NONE", time_scheme=TimeIntegrationScheme.SSP_RK3,
                    vel_inf=10.0)


@pytest.mark.parametrize("max_iter", [0, -3])
def test_non_positive_max_iter_is_rejected(max_iter):
    s = _solver(1)
    s.order_continuation_enabled = False
    with pytest.raises(ValueError, match="max_iter"):
        s.solve(max_iter=max_iter, dt=1e-5, tol=0.0)


def test_single_step_reports_the_real_residual():
    s = _solver(1)
    s.order_continuation_enabled = False
    result = s.solve(max_iter=1, dt=1e-5, tol=0.0)
    assert result.iterations == 1
    assert np.isfinite(result.final_residual) and result.final_residual == s.residual_history[-1]
