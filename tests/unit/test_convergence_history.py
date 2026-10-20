"""逐步收敛历史（I-172，2026-10-19，`core/fr_solver/convergence_history.py`）。

此前 checkpoint 只写当前迭代号：`post convergence` 画出空图、续算后总残差从零开始记；显式时间精确推进时单步里
追加残差，一个物理步的每个子步各记一条。现在两条求解循环每步记一条（监控残差、逐方程残差、CFL、单机逐步压力
系数），写进 checkpoint，续算时接着 checkpoint 的迭代数往下记。
"""

import math
from types import SimpleNamespace

import numpy as np

from autoflowcfd.core.freestream_defaults import MU_MOLECULAR, P_INF, RHO_INF, VEL_INF
import pytest

from autoflowcfd.cli.solve.checkpoint_io.write import write_checkpoint
from autoflowcfd.core.fr_solver.convergence_history import (
    COEFFICIENT_KEYS, MONITOR_RESIDUAL, ConvergenceHistory, checkpoint_history,
)
from autoflowcfd.core.fr_solver.residual_diagnostics import MEAN_FLOW_EQUATIONS
from autoflowcfd.core.time_integration import TimeIntegrationScheme
from autoflowcfd.core.time_integration.implicit.reductions import LocalReductions
from autoflowcfd.core.time_integration.residual_norm import monitor_residuals
from autoflowcfd.core.utils.checkpoint import CheckpointManager
from autoflowcfd.fr.native_padding import real_sps_per_cell, sps_width
from tests.validation._channel_mesh import build_channel_mesh_prism

_BC = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL", "wall_top": "WALL",
       "z_min": "SYMMETRY", "z_max": "SYMMETRY"}


class TestHistoryObject:
    def test_round_trip_and_numbering_continue_from_the_checkpoint(self):
        h = ConvergenceHistory()
        h.append(1.0, [1, 2, 3, 4, 5], 10.0)
        h.append(0.5, None, 20.0, {"Cd_p": 0.3, "Cl_p": 0.1, "Cs_p": 0.0})
        assert h.iterations == [1, 2] and h.last_coefficients == {"Cd_p": 0.3, "Cl_p": 0.1, "Cs_p": 0.0}
        out = h.to_checkpoint()
        assert out["residuals"][MONITOR_RESIDUAL] == [1.0, 0.5] and out["cfl_history"] == [10.0, 20.0]
        assert out["residuals"]["rho"][0] == 1 and math.isnan(out["residuals"]["rho_E"][1])
        assert math.isnan(out["coefficients"]["Cd_p"][0]) and out["coefficients"]["Cd_p"][1] == 0.3

        r = ConvergenceHistory.from_checkpoint(out, 2)
        r.append(0.25)
        assert r.iterations == [1, 2, 3] and r.residual == [1.0, 0.5, 0.25]

    def test_no_coefficients_group_when_none_were_computed(self):
        h = ConvergenceHistory()
        h.append(1.0)
        assert "coefficients" not in h.to_checkpoint() and h.last_coefficients is None

    def test_legacy_checkpoint_history_resumes_empty_after_its_iteration(self):
        r = ConvergenceHistory.from_checkpoint({"iterations": [140]}, 140)
        r.append(3.0)
        assert r.iterations == [141] and r.residual == [3.0]

    def test_inconsistent_iteration_is_refused(self):
        h = ConvergenceHistory()
        h.append(1.0)
        with pytest.raises(ValueError, match="不符"):
            ConvergenceHistory.from_checkpoint(h.to_checkpoint(), 5)
        with pytest.raises(ValueError, match="不符"):
            checkpoint_history(SimpleNamespace(convergence_history=h), 5)


@pytest.mark.parametrize("order", [1, 2])
def test_monitor_residuals_per_equation_and_scaled_total(order):
    """逐方程 RMS 只统计真实自由度（四面体填充槽位不进）；监控残差是逐方程 RMS 除以参考量级后的均方根（I-81：
    此前直接合成原始残差的 RMS，能量方程的量级压倒其余方程）。"""
    rng = np.random.default_rng(3)
    n_sps = sps_width(order)
    is_prism = np.array([True, False, True, False])
    R = rng.standard_normal((4, n_sps, 5)) * np.array([1.0, 10.0, 10.0, 10.0, 1e4])
    n_tet = real_sps_per_cell(order)[1]
    R[~is_prism, n_tet:] = 1e30                                    # 填充槽位：变馊的值
    red = LocalReductions(np)
    scales = np.array([1.0, 10.0, 10.0, 10.0, 1e4])
    total, per_eq = monitor_residuals(red, R, is_prism, order, scales)
    real = np.concatenate([R[0], R[1, :n_tet], R[2], R[3, :n_tet]])
    np.testing.assert_allclose(per_eq, np.sqrt((real ** 2).mean(axis=0)), rtol=1e-14)
    assert total == pytest.approx(float(np.sqrt(np.mean((per_eq / scales) ** 2))), rel=1e-15)
    # 五个方程按参考量级归一化后同量级：各方程的贡献都在 O(1)，没有一个方程压倒其余方程
    assert np.all((per_eq / scales) > 0.5) and np.all((per_eq / scales) < 2.0)


def _solver(time_scheme, **kw):
    from autoflowcfd.core.fr_solver import FRSolver

    mesh = build_channel_mesh_prism(1, 3, 3, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = dict(_BC)
    s = FRSolver(mesh=mesh, order=1, turb_model_name="NONE", time_scheme=time_scheme, vel_inf=10.0, rho_inf=RHO_INF,
                 p_inf=P_INF, mu_molecular=MU_MOLECULAR, **kw)
    s.order_continuation_enabled = False
    rng = np.random.default_rng(7)
    s.state.U[..., 1:4] *= 1.0 + 0.05 * rng.standard_normal(s.state.U[..., 1:4].shape)
    s.state._update_primitives()
    return s


def test_real_solver_writes_its_history_and_resume_continues_it(tmp_path):
    s = _solver(TimeIntegrationScheme.SSP_RK3)
    result = s.solve(max_iter=4, tol=0.0)
    h = s.convergence_history
    assert h.iterations == [1, 2, 3, 4] and h.residual == s.residual_history
    assert h.residual[-1] == result.final_residual and all(c > 0 for c in h.cfl)
    assert all(math.isfinite(h.per_equation[eq][-1]) and h.per_equation[eq][-1] > 0 for eq in MEAN_FLOW_EQUATIONS)

    path = write_checkpoint(s, str(tmp_path), 4, "mesh.nas", order=1, turbulence_model="none", backend="cpu",
                            quiet=True)
    _sol, history, iteration, _meta = CheckpointManager().load(path)
    assert history["iterations"] == [1, 2, 3, 4] and history["residuals"][MONITOR_RESIDUAL] == h.residual
    assert set(history["residuals"]) == {MONITOR_RESIDUAL, *MEAN_FLOW_EQUATIONS}

    r = _solver(TimeIntegrationScheme.SSP_RK3)
    r.convergence_history = ConvergenceHistory.from_checkpoint(history, iteration)
    r.solve(max_iter=2, tol=0.0)
    assert r.convergence_history.iterations == [1, 2, 3, 4, 5, 6]


def test_time_accurate_substeps_record_one_entry_per_physical_step():
    """显式时间精确推进：一个物理步切成若干子步，历史仍是每个物理步一条（此前单步里追加，每个子步一条）。"""
    s = _solver(TimeIntegrationScheme.SSP_RK3, time_accurate=True)
    s.solve(max_iter=2, dt=2e-4, tol=0.0)
    assert s.last_substeps > 1
    assert s.convergence_history.iterations == [1, 2]


def test_per_step_coefficients_are_recorded_with_a_reference_area():
    s = _solver(TimeIntegrationScheme.SSP_RK3)
    s._reference_area = 0.01
    s.solve(max_iter=2, tol=0.0)
    coeffs = s.convergence_history.to_checkpoint()["coefficients"]
    assert set(coeffs) == set(COEFFICIENT_KEYS) and all(math.isfinite(v) for v in coeffs["Cd_p"])
