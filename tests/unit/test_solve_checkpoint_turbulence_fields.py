"""Regression test for a real bug found 2026-08-23 during a live resume
investigation on a 791k-cell cube_demo run: `write_checkpoint` only ever
persisted the mean-flow state (`U_sps`/`Q_sps`), never
`turb_model.k_field`/`omega_field`.

`rebuild_solver_from_checkpoint` reconstructs the solver via a fresh
`FRSolver(...)` call, whose internal `SSTModelFR.__init__` unconditionally
seeds `k_field`/`omega_field` with the same uniform "just starting a solve"
guess (k=1e-6, omega=1.0) used for a brand-new run. Since the checkpoint
never carried the real, converged turbulence field, resume silently
produced a solver whose mean flow was exactly the converged state but
whose turbulence field was reset to the initial guess - a real physical
discontinuity at the resume boundary, confirmed by directly inspecting
`solver.turb_model.omega_field` after a real rebuild (uniformly 1.0,
matching the fresh-construction default rather than any spatially-varying
converged SST field).

Fix: `write_checkpoint` now also persists `k_field`/`omega_field` (via
`hasattr`, so it no-ops for `turb_model=None` or SGS-only models like LES
that don't expose these attributes); `rebuild_solver_from_checkpoint`
restores them when present, with a shape check mirroring the existing
`U_sps` check, and a safe backward-compatible fallback (keep the fresh
uniform guess, print a warning) for checkpoints written before this fix.
"""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from autoflowcfd.cli.solve.checkpoint_io import write_checkpoint, rebuild_solver_from_checkpoint


def _fake_solver(n_cells=2, n_sps=1, n_vars=7, order=0, k=None, omega=None, with_turb_model=True):
    U = np.ones((n_cells, n_sps, n_vars))
    state = SimpleNamespace(U=U, Q=U.copy(), n_sps=n_sps, n_vars=n_vars)
    state._update_primitives = lambda: None
    turb_model = None
    if with_turb_model:
        turb_model = SimpleNamespace(
            k_field=np.full((n_cells, n_sps), 1e-6) if k is None else k,
            omega_field=np.full((n_cells, n_sps), 1.0) if omega is None else omega,
        )
    return SimpleNamespace(
        state=state,
        order=order,
        current_order=order,
        turb_model=turb_model,
        freestream={"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0},
    )


class TestWriteCheckpointStoresTurbulenceFields:
    def test_k_and_omega_written_when_turb_model_present(self, tmp_path):
        k = np.array([[0.42], [0.73]])
        omega = np.array([[123.4], [567.8]])
        write_checkpoint(
            _fake_solver(n_cells=2, n_sps=1, k=k, omega=omega), str(tmp_path), 100, "volume.nas",
            order=0, turbulence_model="sst", backend="cpu", quiet=True,
        )
        import h5py
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"
        with h5py.File(ckpt, "r") as f:
            sol = f["solution"]
            assert "k_field" in sol
            assert "omega_field" in sol
            np.testing.assert_array_equal(sol["k_field"][:], k)
            np.testing.assert_array_equal(sol["omega_field"][:], omega)

    def test_no_turbulence_fields_written_when_turb_model_is_none(self, tmp_path):
        """--turbulence none: must not fabricate k_field/omega_field keys."""
        write_checkpoint(
            _fake_solver(with_turb_model=False), str(tmp_path), 100, "volume.nas",
            order=0, turbulence_model="none", backend="cpu", quiet=True,
        )
        import h5py
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"
        with h5py.File(ckpt, "r") as f:
            assert "k_field" not in f["solution"]
            assert "omega_field" not in f["solution"]


class TestRebuildRestoresTurbulenceFields:
    def _rebuild(self, ckpt_path, n_cells=2, n_sps=1, k_fresh=None, omega_fresh=None):
        """Mimics rebuild_solver_from_checkpoint's FRSolver(...) call
        returning a *freshly constructed* solver (uniform-guess turbulence
        field, distinct from whatever was checkpointed) so restoration is
        actually exercised, not just a no-op identity copy."""
        def _fake_frsolver(**kwargs):
            return _fake_solver(
                n_cells=n_cells, n_sps=n_sps, order=kwargs["order"],
                k=k_fresh, omega=omega_fresh,
            )

        with patch(
            "autoflowcfd.cli.solve.mesh_loader.load_mesh_for_solver",
            return_value=(MagicMock(), MagicMock()),
        ), patch(
            "autoflowcfd.core.FRSolver", side_effect=_fake_frsolver
        ), patch(
            "autoflowcfd.cli.solve.wall_distance.compute_wall_distance_for_solver"
        ):
            return rebuild_solver_from_checkpoint(str(ckpt_path))

    def test_resume_restores_converged_turbulence_field_not_fresh_guess(self, tmp_path):
        k_converged = np.array([[0.017], [0.055]])
        omega_converged = np.array([[842.0], [1953.0]])
        write_checkpoint(
            _fake_solver(n_cells=2, n_sps=1, k=k_converged, omega=omega_converged),
            str(tmp_path), 3000, "volume.nas",
            order=0, turbulence_model="sst", backend="cpu", quiet=True,
        )
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_003000.h5"

        # Fresh FRSolver() construction would normally seed the uniform
        # "just starting" guess - deliberately different from the
        # checkpointed values above so a silent no-restore would be caught.
        k_fresh = np.full((2, 1), 1e-6)
        omega_fresh = np.full((2, 1), 1.0)
        solver, iteration, metadata = self._rebuild(
            ckpt, k_fresh=k_fresh, omega_fresh=omega_fresh,
        )

        np.testing.assert_array_equal(solver.turb_model.k_field, k_converged)
        np.testing.assert_array_equal(solver.turb_model.omega_field, omega_converged)

    def test_old_checkpoint_without_turbulence_fields_falls_back_to_fresh_guess(self, tmp_path, capsys):
        """Pre-fix checkpoint has no k_field/omega_field at all - resume
        must not crash, and must keep whatever fresh-construction default
        FRSolver.__init__ produced (documented, backward-compatible
        degradation), while warning the user."""
        write_checkpoint(
            _fake_solver(n_cells=2, n_sps=1), str(tmp_path), 3000, "volume.nas",
            order=0, turbulence_model="sst", backend="cpu", quiet=True,
        )
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_003000.h5"
        import h5py
        with h5py.File(ckpt, "r+") as f:
            del f["solution"]["k_field"]
            del f["solution"]["omega_field"]

        k_fresh = np.full((2, 1), 1e-6)
        omega_fresh = np.full((2, 1), 1.0)
        solver, iteration, metadata = self._rebuild(
            ckpt, k_fresh=k_fresh, omega_fresh=omega_fresh,
        )

        np.testing.assert_array_equal(solver.turb_model.k_field, k_fresh)
        np.testing.assert_array_equal(solver.turb_model.omega_field, omega_fresh)
        assert "旧版本" in capsys.readouterr().out

    def test_turbulence_field_shape_mismatch_rejected(self, tmp_path):
        """U_sps shape matches (so the pre-existing mean-flow check passes
        through) but the checkpointed k_field/omega_field shape doesn't
        match the freshly-reconstructed turb_model's - must be caught by
        the new dedicated check, not silently broadcast/truncated."""
        write_checkpoint(
            _fake_solver(n_cells=2, n_sps=1, k=np.zeros((2, 1)), omega=np.ones((2, 1))),
            str(tmp_path), 3000, "volume.nas",
            order=0, turbulence_model="sst", backend="cpu", quiet=True,
        )
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_003000.h5"

        import click
        # Fresh reconstruction has matching U shape (n_cells=2) but a
        # mismatched turb_model shape (n_cells=5) - simulates a turb_model
        # whose field shape drifted independently of the mean-flow state.
        def _fake_frsolver(**kwargs):
            s = _fake_solver(n_cells=2, n_sps=1, order=kwargs["order"])
            s.turb_model = SimpleNamespace(
                k_field=np.zeros((5, 1)), omega_field=np.ones((5, 1)),
            )
            return s

        with patch(
            "autoflowcfd.cli.solve.mesh_loader.load_mesh_for_solver",
            return_value=(MagicMock(), MagicMock()),
        ), patch(
            "autoflowcfd.core.FRSolver", side_effect=_fake_frsolver
        ), patch(
            "autoflowcfd.cli.solve.wall_distance.compute_wall_distance_for_solver"
        ), pytest.raises(click.ClickException):
            rebuild_solver_from_checkpoint(str(ckpt))


class TestPhaseInitialResidualPersistence:
    """Regression test for a real bug found 2026-08-23 (same live resume
    investigation): Order Continuation's residual-drop promotion criterion
    (`initial_residual_this_order` in order_continuation.py) is a pure
    local variable with no memory of a phase's true starting residual
    across a `solve resume` process boundary - see
    order_continuation.py::run_order_continuation for the full mechanism.
    This tests the checkpoint round-trip half of the fix (write_checkpoint/
    rebuild_solver_from_checkpoint persisting `solver._phase_initial_
    residual`); the promotion-criteria behaviour itself is covered by
    tests/unit/test_order_continuation_resume.py."""

    def _rebuild(self, ckpt_path, n_cells=2, n_sps=1):
        def _fake_frsolver(**kwargs):
            return _fake_solver(n_cells=n_cells, n_sps=n_sps, order=kwargs["order"])

        with patch(
            "autoflowcfd.cli.solve.mesh_loader.load_mesh_for_solver",
            return_value=(MagicMock(), MagicMock()),
        ), patch(
            "autoflowcfd.core.FRSolver", side_effect=_fake_frsolver
        ), patch(
            "autoflowcfd.cli.solve.wall_distance.compute_wall_distance_for_solver"
        ):
            return rebuild_solver_from_checkpoint(str(ckpt_path))

    def test_written_and_restored_exactly(self, tmp_path):
        solver = _fake_solver(n_cells=2, n_sps=1)
        solver._phase_initial_residual = 12345.6789
        write_checkpoint(
            solver, str(tmp_path), 3000, "volume.nas",
            order=0, turbulence_model="sst", backend="cpu", quiet=True,
        )
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_003000.h5"

        import h5py
        with h5py.File(ckpt, "r") as f:
            assert f["metadata"].attrs["phase_initial_residual"] == pytest.approx(12345.6789)

        restored, _, _ = self._rebuild(ckpt)
        assert restored._phase_initial_residual == pytest.approx(12345.6789)

    def test_not_yet_set_is_skipped_not_written_as_none(self, tmp_path):
        """h5py attrs don't accept None - a solver that hasn't run a single
        step yet (attribute never assigned) must not blow up write_checkpoint,
        and the key must simply be absent, not written as some sentinel."""
        solver = _fake_solver(n_cells=2, n_sps=1)
        assert not hasattr(solver, "_phase_initial_residual")
        write_checkpoint(
            solver, str(tmp_path), 3000, "volume.nas",
            order=0, turbulence_model="sst", backend="cpu", quiet=True,
        )
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_003000.h5"

        import h5py
        with h5py.File(ckpt, "r") as f:
            assert "phase_initial_residual" not in f["metadata"].attrs

    def test_old_checkpoint_without_field_leaves_attribute_unset(self, tmp_path):
        """Pre-fix checkpoint has no phase_initial_residual at all - resume
        must not crash and must not fabricate a value; order_continuation.py
        detects the absence via getattr(...,None) and falls back safely
        (see test_order_continuation_resume.py for that half)."""
        solver = _fake_solver(n_cells=2, n_sps=1)
        write_checkpoint(
            solver, str(tmp_path), 3000, "volume.nas",
            order=0, turbulence_model="sst", backend="cpu", quiet=True,
        )
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_003000.h5"
        # No need to strip anything - this solver never had the attribute
        # set, so write_checkpoint already skipped it (previous test).

        restored, _, _ = self._rebuild(ckpt)
        assert getattr(restored, "_phase_initial_residual", None) is None
