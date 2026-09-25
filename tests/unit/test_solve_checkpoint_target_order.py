"""Regression test for a real bug found 2026-08-22 on a real 791k-cell
resume run: `order` and `target_order` were conflated into a single
checkpoint metadata field.

`FRSolver.__init__(order=...)` sets *both* `self.current_order` (state
shape) and `self.order` (the Order Continuation ramp target read by
`solver.solve()`'s `self.order_continuation_enabled and self.order >= 2`
gate, see fr_solver/solver.py). A checkpoint saved mid-ramp (e.g. during
the P0 phase, before any P0->P1 transition) has `current_order=0` but a
target of e.g. 2 - these genuinely differ. Storing only one value and
using it for both purposes on `resume` is a contradiction:

- Reconstruct at target_order (2): mesh/FRSolver initial state shape
  (n_cells, 27, n_vars) doesn't match the saved (n_cells, 1, n_vars)
  U_sps -> `rebuild_solver_from_checkpoint` raises "状态形状不匹配".
- Reconstruct at current_order (0) *and* leave FRSolver's `self.order`
  at 0 too (the actual observed bug, from an earlier same-day fix that
  only addressed the shape-mismatch half of this): shapes match and
  resume proceeds silently, but `self.order >= 2` is now False, so
  `solver.solve()` silently skips `_solve_with_order_continuation`
  entirely. Real symptom on the production run: the CLI's rich
  "P{n} Iter ...: Drop: ...x" per-order logging was replaced by the
  generic "Iteration N: Residual = ..." format, and the residual sat
  flat with no path to ever reach P1/P2.

Fix: `write_checkpoint` now stores `order` (current_order, for shape
matching) and `target_order` (solver.order, for continuing the ramp) as
two distinct metadata fields; `rebuild_solver_from_checkpoint` builds the
mesh/initial state at `order` but then explicitly restores
`solver.order = target_order` afterward.
"""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np

from autoflowcfd.cli.solve.checkpoint_io import write_checkpoint, rebuild_solver_from_checkpoint


def _fake_solver(n_cells=2, n_sps=1, n_vars=7, order=0):
    U = np.ones((n_cells, n_sps, n_vars))
    state = SimpleNamespace(U=U, Q=U.copy(), n_sps=n_sps, n_vars=n_vars)
    state._update_primitives = lambda: None
    return SimpleNamespace(
        state=state,
        order=order,
        current_order=order,
        # 2026-09-15：`write_checkpoint` 写出的单元平均现在只统计**真实
        # 自由度**（native 四面体的零填充槽位冻结在初值、会变馊，见
        # fr/native_padding.py::reduce_per_cell_over_real_sps），因此
        # P>=1 的替身必须给出单元类型划分。这里全当棱柱（n_prism ==
        # n_cells）——本文件测的是 target_order 的读写，单元类型不参与。
        mesh=SimpleNamespace(n_prism_cells=n_cells),
        freestream={"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0},
    )


class TestWriteCheckpointStoresTargetOrderSeparately:
    def test_target_order_stored_distinctly_from_current_order(self, tmp_path):
        # Simulates a checkpoint saved mid-ramp: current_order=0 (P0),
        # but the run's real target is P2.
        write_checkpoint(
            _fake_solver(order=0), str(tmp_path), 100, "volume.nas",
            order=0, turbulence_model="sst", backend="cpu",
            target_order=2, quiet=True,
        )
        import h5py
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"
        with h5py.File(ckpt, "r") as f:
            assert f["metadata"].attrs["order"] == 0
            assert f["metadata"].attrs["target_order"] == 2

    def test_target_order_defaults_to_order_when_not_given(self, tmp_path):
        """Old call sites / non-ramping runs where order==target_order
        should keep working without passing target_order explicitly."""
        write_checkpoint(
            _fake_solver(order=2), str(tmp_path), 100, "volume.nas",
            order=2, turbulence_model="sst", backend="cpu", quiet=True,
        )
        import h5py
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"
        with h5py.File(ckpt, "r") as f:
            assert f["metadata"].attrs["target_order"] == 2


class TestRebuildRestoresSolverOrderToTarget:
    def test_mid_ramp_checkpoint_reconstructs_shape_at_current_but_target_at_solver_order(self, tmp_path):
        write_checkpoint(
            _fake_solver(n_cells=2, n_sps=1, order=0), str(tmp_path), 100, "volume.nas",
            order=0, turbulence_model="sst", backend="cpu", target_order=2, quiet=True,
        )
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"

        # FRSolver mock: mimics the real __init__ behaviour of setting
        # both current_order and order from the single `order` kwarg it's
        # constructed with (n_sps=1, matching the checkpoint's P0 shape).
        def _fake_frsolver(**kwargs):
            s = _fake_solver(n_cells=2, n_sps=1, order=kwargs["order"])
            return s

        with patch(
            "autoflowcfd.cli.solve.mesh_loader.load_mesh_for_solver",
            return_value=(MagicMock(), MagicMock()),
        ), patch(
            "autoflowcfd.core.FRSolver", side_effect=_fake_frsolver
        ), patch(
            "autoflowcfd.cli.solve.wall_distance.compute_wall_distance_for_solver"
        ):
            solver, iteration, metadata = rebuild_solver_from_checkpoint(str(ckpt))

        assert solver.current_order == 0  # matches the saved (n_cells,1,n_vars) state shape
        assert solver.order == 2  # NOT 0 - restored to the real ramp target
        assert metadata["target_order"] == 2

    def test_old_checkpoint_without_target_order_field_falls_back_safely(self, tmp_path):
        """A checkpoint written before this fix has no target_order key at
        all - metadata.get("target_order", order) must fall back to order,
        reproducing the old (order==target_order-always) behaviour rather
        than crashing on a missing key."""
        write_checkpoint(
            _fake_solver(n_cells=2, n_sps=27, order=2), str(tmp_path), 100, "volume.nas",
            order=2, turbulence_model="sst", backend="cpu", quiet=True,
        )
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"
        # Remove the target_order key to simulate a pre-fix checkpoint.
        import h5py
        with h5py.File(ckpt, "r+") as f:
            del f["metadata"].attrs["target_order"]

        def _fake_frsolver(**kwargs):
            return _fake_solver(n_cells=2, n_sps=27, order=kwargs["order"])

        with patch(
            "autoflowcfd.cli.solve.mesh_loader.load_mesh_for_solver",
            return_value=(MagicMock(), MagicMock()),
        ), patch(
            "autoflowcfd.core.FRSolver", side_effect=_fake_frsolver
        ), patch(
            "autoflowcfd.cli.solve.wall_distance.compute_wall_distance_for_solver"
        ):
            solver, iteration, metadata = rebuild_solver_from_checkpoint(str(ckpt))

        assert solver.order == 2
        assert solver.current_order == 2
