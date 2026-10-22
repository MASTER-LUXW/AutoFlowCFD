"""真实缺陷回归测试（2026-08-22，长时间续算运行中用户直接问："多少步存一个ckpt"）。

`solve resume` 此前从不给 `solver.solve()` 传 `checkpoint_callback`，而 `solve steady` 有 `--checkpoint-interval`
接到它自己的回调上。续算只在全部 `--max-iter` 步跑完之后写一次 checkpoint——跑了几个小时的续算如果中途崩溃
或被打断，会丢掉全部进展、退回出发时的那个 checkpoint，中间没有任何保存点。

修复：`resume()` 有了自己的 `--checkpoint-interval` 选项并构造等价的回调。回调用的迭代数必须是**绝对**计数
（checkpoint 的起始 `iteration` + 本次运行的局部计数）——只用局部计数的话每次续算都从 0/1 重新数起，会与续算
之前同名的 checkpoint 文件静默冲突（覆盖），例如局部第 500 步与原运行自己的 checkpoint_iter_000500.h5 撞名。
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
            "autoflowcfd.cli.solve.checkpoint_io.write.save_results"
        ), patch(
            "autoflowcfd.cli.solve.checkpoint_io.write.write_checkpoint"
        ) as mock_write_checkpoint:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["solve", "resume", str(checkpoint_file), "--max-iter", "10",
                 "--checkpoint-interval", "5"],
            )

        assert result.exit_code == 0, result.output

        # solve() 确实带着 checkpoint_callback 被调用。
        assert fake_solver.solve.call_args.kwargs.get("checkpoint_callback") is not None

        # 两个中途 checkpoint（局部迭代 5 与 10）+ 一个结束时的最终 checkpoint
        # = 共 3 次 write_checkpoint 调用。
        assert mock_write_checkpoint.call_count == 3

        # 中途的那些必须加上 checkpoint 的起始迭代数（3000），不能用裸的局部
        # 计数（5、10）——裸的局部编号会与续算之前的 checkpoint 文件撞名。
        written_iterations = [c.args[2] for c in mock_write_checkpoint.call_args_list]
        assert 3005 in written_iterations
        assert 3010 in written_iterations
        assert 5 not in written_iterations
        assert 10 not in written_iterations
