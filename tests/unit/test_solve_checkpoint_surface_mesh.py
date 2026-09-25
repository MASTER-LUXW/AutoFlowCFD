"""Unit tests for solve_checkpoint_io.py's surface_mesh persistence.

Real gap fixed here (2026-08-22): `write_checkpoint` stored `input_file`/
`order`/`turbulence_model`/`backend` in checkpoint metadata so `solve
resume` could auto-reload the volume mesh, but never stored `surface_mesh`
even though it was in scope at every call site (solve_steady_command.py,
solve_transient_command.py, solve_commands.py's own resume()) - the user
had to manually re-pass `-s <original surface mesh>` on every resume of a
run whose input_file was a raw .nas volume mesh, with no way for the CLI
to remind them what path that even was. `rebuild_solver_from_checkpoint`
now falls back to the stored value when the caller doesn't pass one
explicitly, mirroring how `backend`/`order`/`turbulence_model` already
behave.
"""

from types import SimpleNamespace

from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from autoflowcfd.cli.solve.checkpoint_io import write_checkpoint, rebuild_solver_from_checkpoint


def _fake_solver(n_cells=2, n_sps=1, n_vars=7):
    U = np.ones((n_cells, n_sps, n_vars))
    state = SimpleNamespace(U=U, Q=U.copy(), n_sps=n_sps, n_vars=n_vars)
    state._update_primitives = lambda: None
    return SimpleNamespace(
        state=state,
        freestream={"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0},
        # write_checkpoint 记录时间格式（2026-09-25）：真实求解器恒有 time_integrator
        time_integrator=SimpleNamespace(scheme=TimeIntegrationScheme.SSP_RK3),
    )


class TestWriteCheckpointSurfaceMeshPersistence:
    def test_surface_mesh_stored_when_given(self, tmp_path):
        write_checkpoint(
            _fake_solver(), str(tmp_path), 100, "volume.nas", 0, "sst", "cpu",
            surface_mesh="surface.nas", quiet=True,
        )
        import h5py
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"
        with h5py.File(ckpt, "r") as f:
            assert f["metadata"].attrs["surface_mesh"].decode("utf-8") == "surface.nas"

    def test_surface_mesh_key_omitted_when_none(self, tmp_path):
        """h5py attrs can't store None - must skip the key entirely, not
        write a null/sentinel value that would corrupt round-tripping."""
        write_checkpoint(
            _fake_solver(), str(tmp_path), 100, "volume.pkl", 0, "sst", "cpu",
            surface_mesh=None, quiet=True,
        )
        import h5py
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"
        with h5py.File(ckpt, "r") as f:
            assert "surface_mesh" not in f["metadata"].attrs


class TestRebuildSolverSurfaceMeshFallback:
    """`load_mesh_for_solver`/FRSolver construction/wall-distance are too
    heavy for a unit test (real mesh geometry) - mocked out here to isolate
    just the surface_mesh resolution logic under test."""

    def _write_and_rebuild(self, tmp_path, stored_surface_mesh, call_surface_mesh):
        write_checkpoint(
            _fake_solver(), str(tmp_path), 100, "volume.nas", 0, "sst", "cpu",
            surface_mesh=stored_surface_mesh, quiet=True,
        )
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"

        fake_mesh = MagicMock()
        fake_volume_data = MagicMock()
        fake_solver = _fake_solver()

        with patch(
            "autoflowcfd.cli.solve.mesh_loader.load_mesh_for_solver",
            return_value=(fake_mesh, fake_volume_data),
        ) as mock_load, patch(
            "autoflowcfd.core.FRSolver", return_value=fake_solver
        ), patch(
            "autoflowcfd.cli.solve.wall_distance.compute_wall_distance_for_solver"
        ):
            rebuild_solver_from_checkpoint(str(ckpt), surface_mesh=call_surface_mesh)

        return mock_load

    def test_falls_back_to_stored_surface_mesh_when_not_passed(self, tmp_path):
        mock_load = self._write_and_rebuild(
            tmp_path, stored_surface_mesh="surface.nas", call_surface_mesh=None,
        )
        assert mock_load.call_args.kwargs["surface_mesh"] == "surface.nas"

    def test_explicit_call_argument_overrides_stored_value(self, tmp_path):
        mock_load = self._write_and_rebuild(
            tmp_path, stored_surface_mesh="old_surface.nas", call_surface_mesh="new_surface.nas",
        )
        assert mock_load.call_args.kwargs["surface_mesh"] == "new_surface.nas"

    def test_none_when_neither_stored_nor_passed(self, tmp_path):
        mock_load = self._write_and_rebuild(
            tmp_path, stored_surface_mesh=None, call_surface_mesh=None,
        )
        assert mock_load.call_args.kwargs["surface_mesh"] is None
