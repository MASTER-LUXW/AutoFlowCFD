"""Regression test for a real bug (found 2026-08-22, real-mesh resume):
`solve resume` re-wrote its own checkpoint using the `order` value read
from the checkpoint metadata *before* calling `solver.solve()` - but
`solver.solve()` can itself trigger an Order Continuation transition
(P0->P1 etc.) partway through, leaving `solver.current_order` ahead of
that pre-solve snapshot by the time the checkpoint is written. The next
resume would then reconstruct the mesh/FRSolver at the *wrong* (stale)
order, producing a U_sps shape that no longer matches the actually-saved
state - exactly the failure the user hit on a real 791k-cell run:

    Checkpoint 状态形状 (791492, 1, 7) 与重建求解器的状态形状
    (791492, 27, 7) 不匹配

The same stale-closure-variable bug existed in `solve_steady_command.py`'s
periodic checkpoint callback and end-of-run checkpoint, and in
`solve_transient_command.py`'s end-of-run checkpoint - all four call sites
now pass `solver.current_order` (the live value) instead of the value
captured before `solve()` ran. This test pins the `solve resume` case,
which is exercised directly through the CLI (the other three sites
require a full mesh-generation pipeline to reach solve(), too heavy for
a unit test - covered instead by inspection/consistency across all four
call sites, which were fixed identically)."""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from autoflowcfd.cli.main import cli


def _fake_solver(current_order=0):
    solver = MagicMock()
    solver.current_order = current_order
    # 真实的 rebuild_solver_from_checkpoint 总是显式设置它（真实值或 None）
    # ——换成裸 MagicMock 自动生成的属性，会让最后的
    # _report_aerodynamic_coefficients(solver, ...) 拿 MagicMock 与 0 比较，
    # 抛 TypeError。
    solver._reference_area = None

    def _solve(*args, **kwargs):
        # 模拟 Order Continuation 在这次续算自己的 solve() 调用中途从 P0 爬到 P1
        # ——current_order 在 resume() 已经把 metadata["order"] 读进局部变量
        # `order` *之后*才变化。
        solver.current_order = current_order + 1
        return SimpleNamespace(iterations=10, final_residual=1e-3)

    solver.solve.side_effect = _solve
    return solver


class TestResumeWritesLiveOrderNotStaleMetadataOrder:
    def test_checkpoint_rewritten_with_post_solve_current_order(self, tmp_path):
        checkpoint_file = tmp_path / "checkpoint_iter_000100.h5"
        checkpoint_file.write_bytes(b"")  # CliRunner 的 click.Path(exists=True) 只要求它存在

        fake_solver = _fake_solver(current_order=0)
        fake_metadata = {
            "input_file": "volume.nas",
            "order": 0,  # 求解前的过时快照，与 solve() *之前*求解器的阶数一致
            "turbulence_model": "sst",
            "backend": "cpu",
            "surface_mesh": "surface.nas",
        }

        with patch(
            "autoflowcfd.cli.solve.commands.rebuild_solver_from_checkpoint",
            return_value=(fake_solver, 100, fake_metadata),
        ), patch(
            "autoflowcfd.cli.solve.checkpoint_io.write.save_results"
        ), patch(
            "autoflowcfd.cli.solve.checkpoint_io.write.write_checkpoint"
        ) as mock_write_checkpoint:
            runner = CliRunner()
            result = runner.invoke(
                cli, ["solve", "resume", str(checkpoint_file), "--max-iter", "10"],
            )

        assert result.exit_code == 0, result.output
        assert fake_solver.current_order == 1  # 检查：solve() 确实把它改了

        # 位置参数：(solver, output_dir, iteration, input_file, order, turbulence_model, backend)
        call_args = mock_write_checkpoint.call_args
        written_order = call_args.args[4]
        assert written_order == 1, (
            f"write_checkpoint was called with the stale pre-solve order (0) "
            f"instead of solver.current_order (1) - got {written_order}"
        )
