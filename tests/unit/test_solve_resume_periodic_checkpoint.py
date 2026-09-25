"""Regression test for a real gap found 2026-08-22 (user asked directly:
"多少步存一个ckpt" - how many steps between checkpoint saves - while a
long resume run was in progress).

`solve resume` never passed a `checkpoint_callback` to `solver.solve()`,
unlike `solve steady` (which has `--checkpoint-interval` wired up via its
own `_checkpoint_cb`). A resume run wrote exactly one checkpoint, at the
very end after all `--max-iter` iterations completed - a multi-hour
resume that crashed or was interrupted partway would lose everything
back to the checkpoint it started from, with no intermediate save point.

Fix: `resume()` now has its own `--checkpoint-interval` option and builds
an equivalent callback. The callback's iteration numbers must be the
*absolute* count (checkpoint's starting `iteration` + this run's local
counter) - the local counter alone restarts from 0/1 every resume call,
and would silently collide with (overwrite) a pre-resume checkpoint file
of the same name (e.g. local iteration 500 colliding with the original
run's own checkpoint_iter_000500.h5).
"""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from autoflowcfd.cli.main import cli


def _fake_solver(current_order=0):
    solver = MagicMock()
    solver.current_order = current_order
    solver.order = current_order
    solver._reference_area = None

    def _solve(*args, **kwargs):
        cb = kwargs.get("checkpoint_callback")
        if cb is not None:
            cb(solver, 5)
            cb(solver, 10)
        return SimpleNamespace(iterations=10, final_residual=1e-3)

    solver.solve.side_effect = _solve
    return solver


class TestResumePeriodicCheckpoint:
    def test_checkpoint_callback_is_wired_and_uses_absolute_iteration(self, tmp_path):
        checkpoint_file = tmp_path / "checkpoint_iter_003000.h5"
        checkpoint_file.write_bytes(b"")

        fake_solver = _fake_solver(current_order=0)
        fake_metadata = {
            "input_file": "volume.nas",
            "order": 0,
            "turbulence_model": "sst",
            "backend": "cpu",
            "surface_mesh": "surface.nas",
        }

        with patch(
            "autoflowcfd.cli.solve.commands.rebuild_solver_from_checkpoint",
            return_value=(fake_solver, 3000, fake_metadata),
        ), patch(
            "autoflowcfd.cli.solve.commands.save_results"
        ), patch(
            "autoflowcfd.cli.solve.commands.write_checkpoint"
        ) as mock_write_checkpoint:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["solve", "resume", str(checkpoint_file), "--max-iter", "10",
                 "--checkpoint-interval", "5"],
            )

        assert result.exit_code == 0, result.output

        # solve() was called with a checkpoint_callback at all.
        assert fake_solver.solve.call_args.kwargs.get("checkpoint_callback") is not None

        # Two mid-run checkpoints (local iter 5 and 10) + one final
        # end-of-run checkpoint = 3 total write_checkpoint calls.
        assert mock_write_checkpoint.call_count == 3

        # Mid-run ones must be offset by the checkpoint's starting
        # iteration (3000), not the bare local counter (5, 10) - a bare
        # local number would collide with a pre-resume checkpoint file.
        written_iterations = [c.args[2] for c in mock_write_checkpoint.call_args_list]
        assert 3005 in written_iterations
        assert 3010 in written_iterations
        assert 5 not in written_iterations
        assert 10 not in written_iterations
