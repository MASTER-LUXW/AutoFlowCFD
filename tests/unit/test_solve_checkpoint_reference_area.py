"""真实缺陷回归测试（2026-08-22，用户在一次续算后反馈："阻力 侧力 升力没有显示"）。

`run_order_continuation` 每步日志只在 `solver._reference_area` 为正数时才附带 Cd / Cl / Cs（见
`core/utils/order_continuation/`：`getattr(solver, '_reference_area', None)`）。稳态命令
（`cli/solve/steady/command.py`）构造之后紧接着设置这个属性（显式 --reference-area，不给时由
`_compute_reference_area_auto` 按面网格的迎风投影面积自动估算）——但 `rebuild_solver_from_checkpoint` 从来不设，
续算出的求解器根本没有 `_reference_area` 属性。即使给 `solve resume` 传 `--reference-area`，也只会在整个运行
结束后由 resume() 自己的 `_report_aerodynamic_coefficients` 打印一次，每步日志里仍没有 Cd / Cl / Cs。

修复：`rebuild_solver_from_checkpoint` 与稳态命令同一逻辑——显式 `reference_area` 参数优先，否则退回
`_compute_reference_area_auto(volume_data)`——并在返回前设置 `solver._reference_area`。
"""

from types import SimpleNamespace

from tests.unit._host_state import with_host_state
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from unittest.mock import patch, MagicMock

import numpy as np

from autoflowcfd.cli.solve.checkpoint_io import write_checkpoint, rebuild_solver_from_checkpoint


def _fake_solver(n_cells=2, n_sps=1, n_vars=7, order=0):
    U = np.ones((n_cells, n_sps, n_vars))
    state = SimpleNamespace(U=U, Q=U.copy(), n_sps=n_sps, n_vars=n_vars)
    state._update_primitives = lambda: None
    return with_host_state(SimpleNamespace(
        state=state,
        order=order,
        current_order=order,
        freestream={"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0},
        wall_treatment="resolved", low_mach_precond=True,
        mu_molecular=1.8e-5, _turbulence_intensity=0.01, _viscosity_ratio=5.0, _sem_num_eddies=None,
        artificial_viscosity_enabled=False, artificial_viscosity_alpha=1.0,
        bc_overrides={},                    # 真实求解器恒有（参考面积与气动力积分同一组物体面，I-174）
        # write_checkpoint 记录时间格式（2026-09-25）：真实求解器恒有 time_integrator
        time_integrator=SimpleNamespace(scheme=TimeIntegrationScheme.SSP_RK3),
    ))


def _write_and_rebuild(tmp_path, **rebuild_kwargs):
    write_checkpoint(
        _fake_solver(), str(tmp_path), 100, "volume.nas", 0, "sst", "cpu", quiet=True,
    )
    ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"

    def _fake_frsolver(**kwargs):
        return _fake_solver(order=kwargs["order"])

    with patch(
        "autoflowcfd.cli.solve.mesh_loader.load_mesh_for_solver",
        return_value=(MagicMock(), MagicMock()),
    ), patch(
        "autoflowcfd.core.fr_solver.FRSolver", side_effect=_fake_frsolver
    ), patch(
        "autoflowcfd.cli.solve.wall_distance.compute_wall_distance_for_solver"
    ):
        return rebuild_solver_from_checkpoint(str(ckpt), **rebuild_kwargs)


class TestRebuildSolverSetsReferenceArea:
    def test_explicit_reference_area_is_used_directly(self, tmp_path):
        with patch(
            "autoflowcfd.cli.solve.aero_coefficients._compute_reference_area_auto"
        ) as mock_auto:
            solver, _, _ = _write_and_rebuild(tmp_path, reference_area=2.5)

        mock_auto.assert_not_called()
        assert solver._reference_area == 2.5

    def test_falls_back_to_auto_estimate_when_not_given(self, tmp_path):
        with patch(
            "autoflowcfd.cli.solve.aero_coefficients._compute_reference_area_auto",
            return_value=3.7,
        ) as mock_auto:
            solver, _, _ = _write_and_rebuild(tmp_path)

        mock_auto.assert_called_once()
        assert solver._reference_area == 3.7

    def test_reference_area_none_when_auto_estimate_fails(self, tmp_path):
        with patch(
            "autoflowcfd.cli.solve.aero_coefficients._compute_reference_area_auto",
            return_value=None,
        ):
            solver, _, _ = _write_and_rebuild(tmp_path)

        assert solver._reference_area is None
