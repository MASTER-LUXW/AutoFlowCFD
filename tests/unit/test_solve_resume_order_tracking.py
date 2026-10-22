"""真实缺陷回归测试（2026-08-22 真实网格续算时发现）：`solve resume` 重写自己的 checkpoint 时，用的是调用
`solver.solve()` **之前**从 checkpoint 元数据读出的 `order`——但 `solver.solve()` 自己可能中途触发 Order
Continuation 的阶数切换（P0 -> P1 等），写 checkpoint 时 `solver.current_order` 已经领先于求解前的那份快照。
下一次续算于是按**过时**的阶数重建网格与 FRSolver，得到的 U_sps 形状与实际保存的状态不符——正是用户在
79.1 万单元真实算例上遇到的失败：

    Checkpoint 状态形状 (791492, 1, 7) 与重建求解器的状态形状
    (791492, 27, 7) 不匹配

同样的"闭包里取了过时变量"缺陷也存在于稳态命令（`cli/solve/steady/command.py`）的周期 checkpoint 回调与收尾
checkpoint、以及瞬态命令（`cli/solve/transient.py`）的收尾 checkpoint——四处调用点现在都传
`solver.current_order`（当时的实际值），而不是 `solve()` 之前取下的值。本测试钉住 `solve resume` 这一处，
它可以直接经 CLI 触发（其余三处要走完整的网格生成流程才能到 solve()，单元测试太重——四处改法相同，由代码
一致性核对覆盖）。
"""

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
