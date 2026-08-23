"""Regression test for a real gap found 2026-08-22 (user reported after a
resume run: "阻力 侧力 升力没有显示" - drag/side-force/lift not showing).

`run_order_continuation`'s per-iteration log line only appends
Cd/Cl/Cs when `solver._reference_area` is a positive number (see
order_continuation.py: `getattr(solver, '_reference_area', None)`).
`solve_steady_command.py` sets this attribute right after construction
(explicit --reference-area, or auto-estimated from the surface mesh's
projected frontal area via `_compute_reference_area_auto` when not
given) - but `rebuild_solver_from_checkpoint` never set it at all, so a
resumed solver had no `_reference_area` attribute whatsoever. Even
passing `--reference-area` to `solve resume` wouldn't have produced
per-iteration Cd/Cl/Cs, only a single post-hoc value after the whole
run finished (via resume()'s own `_report_aerodynamic_coefficients`
call).

Fix: `rebuild_solver_from_checkpoint` now mirrors solve_steady_command.py's
logic exactly - explicit `reference_area` param wins, otherwise falls
back to `_compute_reference_area_auto(volume_data)` - and sets
`solver._reference_area` before returning.
"""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np

from autoflowcfd.cli.solve_checkpoint_io import write_checkpoint, rebuild_solver_from_checkpoint


def _fake_solver(n_cells=2, n_sps=1, n_vars=7, order=0):
    U = np.ones((n_cells, n_sps, n_vars))
    state = SimpleNamespace(U=U, Q=U.copy(), n_sps=n_sps, n_vars=n_vars)
    state._update_primitives = lambda: None
    return SimpleNamespace(
        state=state,
        order=order,
        current_order=order,
        freestream={"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0},
    )


def _write_and_rebuild(tmp_path, **rebuild_kwargs):
    write_checkpoint(
        _fake_solver(), str(tmp_path), 100, "volume.nas", 0, "sst", "cpu", quiet=True,
    )
    ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"

    def _fake_frsolver(**kwargs):
        return _fake_solver(order=kwargs["order"])

    with patch(
        "autoflowcfd.cli.solve_mesh_loader.load_mesh_for_solver",
        return_value=(MagicMock(), MagicMock()),
    ), patch(
        "autoflowcfd.core.FRSolver", side_effect=_fake_frsolver
    ), patch(
        "autoflowcfd.cli.solve_wall_distance.compute_wall_distance_for_solver"
    ):
        return rebuild_solver_from_checkpoint(str(ckpt), **rebuild_kwargs)


class TestRebuildSolverSetsReferenceArea:
    def test_explicit_reference_area_is_used_directly(self, tmp_path):
        with patch(
            "autoflowcfd.cli.solve_aero_coefficients._compute_reference_area_auto"
        ) as mock_auto:
            solver, _, _ = _write_and_rebuild(tmp_path, reference_area=2.5)

        mock_auto.assert_not_called()
        assert solver._reference_area == 2.5

    def test_falls_back_to_auto_estimate_when_not_given(self, tmp_path):
        with patch(
            "autoflowcfd.cli.solve_aero_coefficients._compute_reference_area_auto",
            return_value=3.7,
        ) as mock_auto:
            solver, _, _ = _write_and_rebuild(tmp_path)

        mock_auto.assert_called_once()
        assert solver._reference_area == 3.7

    def test_reference_area_none_when_auto_estimate_fails(self, tmp_path):
        with patch(
            "autoflowcfd.cli.solve_aero_coefficients._compute_reference_area_auto",
            return_value=None,
        ):
            solver, _, _ = _write_and_rebuild(tmp_path)

        assert solver._reference_area is None
