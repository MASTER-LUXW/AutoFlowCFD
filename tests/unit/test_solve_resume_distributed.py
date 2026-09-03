"""`solve resume --n-ranks`/`--multi-gpu`/`--fully-distributed` 分布式
续算支持验证（2026-09-02，见 solve_commands.py::_resume_distributed/
solve_distributed_checkpoint_io.py 文档）。

真实缺口（用户明确要求"不允许出现完成度不是100%的功能点"后排查
发现）：此前分布式路径（CPU MPI"传统模式"/"完全分布式加载"/多GPU）
只能在 `solve steady` 跑完（或每 `--checkpoint-interval` 步）保存一次
checkpoint，`solve resume` 对 `--n-ranks`/`--multi-gpu`/
`--fully-distributed` 完全没有感知——没有任何"从分布式 checkpoint
继续跑"的机制。

与既有的 `test_solve_resume_periodic_checkpoint.py`（单机 resume
callback wiring 测试）同一个方法论：mock 掉
`rebuild_distributed_solver_from_checkpoint`（重建逻辑本身复用的
`load_mesh_for_solver`/`DistributedFRSolver`/`MultiGPUDistributedSolver`/
`distributed_save_checkpoint`/`distributed_load_checkpoint` 都已在
各自的单元测试里决定性验证过，这里只验证 CLI 层的 wiring：参数解析、
checkpoint_callback 是否正确调用、绝对迭代数是否正确、最终保存是否
调用），不重新验证网格加载/求解器构造这些底层机制本身。
"""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from autoflowcfd.cli.main import cli


def _fake_distributed_solver(solve_return=None):
    solver = MagicMock()

    def _solve(*args, **kwargs):
        cb = kwargs.get("checkpoint_callback")
        if cb is not None:
            cb(solver, 5)
            cb(solver, 10)
        return solve_return

    solver.solve.side_effect = _solve
    return solver


class TestResumeDistributedCpuTraditionalMode(object):
    """--n-ranks>1（不加 --multi-gpu/--fully-distributed）：CPU MPI
    "传统模式"。"""

    def test_checkpoint_callback_wired_with_absolute_iteration(self, tmp_path):
        checkpoint_file = tmp_path / "checkpoint_iter_002000.h5"
        checkpoint_file.write_bytes(b"")

        fake_solver = _fake_distributed_solver(solve_return=None)
        fake_metadata = {
            "input_file": "volume.nas",
            "order": 1,
            "turbulence_model": "none",
            "backend": "cpu",
            "surface_mesh": "surface.nas",
        }

        with patch(
            "autoflowcfd.cli.solve_distributed_checkpoint_io.rebuild_distributed_solver_from_checkpoint",
            return_value=(fake_solver, 2000, fake_metadata),
        ) as mock_rebuild, patch(
            "autoflowcfd.core.mpi.distributed_checkpoint.distributed_save_results"
        ), patch(
            "autoflowcfd.core.mpi.distributed_checkpoint.distributed_save_checkpoint"
        ) as mock_save_checkpoint:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["solve", "resume", str(checkpoint_file), "--max-iter", "10",
                 "--n-ranks", "2", "--checkpoint-interval", "5"],
            )

        assert result.exit_code == 0, result.output
        mock_rebuild.assert_called_once()
        assert mock_rebuild.call_args.kwargs["n_ranks"] == 2
        assert mock_rebuild.call_args.kwargs["multi_gpu"] is False
        assert mock_rebuild.call_args.kwargs["fully_distributed"] is False

        assert fake_solver.solve.call_args.kwargs.get("checkpoint_callback") is not None

        # 2 次中途保存（local iter 5/10）+ 1 次最终保存 = 3 次。
        assert mock_save_checkpoint.call_count == 3
        written_iterations = [c.args[2] for c in mock_save_checkpoint.call_args_list]
        assert 2005 in written_iterations
        assert 2010 in written_iterations
        assert 5 not in written_iterations
        assert 10 not in written_iterations


class TestResumeDistributedFullyDistributed(object):
    def test_fully_distributed_flag_passed_through(self, tmp_path):
        checkpoint_file = tmp_path / "checkpoint_iter_001000.h5"
        checkpoint_file.write_bytes(b"")

        fake_solver = _fake_distributed_solver(solve_return=None)
        fake_metadata = {
            "input_file": "volume.nas", "order": 2, "turbulence_model": "sst",
            "backend": "cpu", "surface_mesh": "surface.nas",
        }

        with patch(
            "autoflowcfd.cli.solve_distributed_checkpoint_io.rebuild_distributed_solver_from_checkpoint",
            return_value=(fake_solver, 1000, fake_metadata),
        ) as mock_rebuild, patch(
            "autoflowcfd.core.mpi.distributed_checkpoint.distributed_save_results"
        ), patch(
            "autoflowcfd.core.mpi.distributed_checkpoint.distributed_save_checkpoint"
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["solve", "resume", str(checkpoint_file), "--max-iter", "5",
                 "--n-ranks", "3", "--fully-distributed"],
            )

        assert result.exit_code == 0, result.output
        assert mock_rebuild.call_args.kwargs["fully_distributed"] is True
        assert mock_rebuild.call_args.kwargs["n_ranks"] == 3


class TestResumeDistributedMultiGpu(object):
    def test_multi_gpu_uses_gpu_checkpoint_api(self, tmp_path):
        checkpoint_file = tmp_path / "checkpoint_iter_000500.h5"
        checkpoint_file.write_bytes(b"")

        fake_solver = _fake_distributed_solver(solve_return={"final_residual": 1e-4, "converged": True})
        fake_solver.save_checkpoint_distributed.return_value = "fake_ckpt_path.h5"
        fake_metadata = {
            "input_file": "volume.nas", "order": 1, "turbulence_model": "none",
            "backend": "gpu", "surface_mesh": None,
        }

        with patch(
            "autoflowcfd.cli.solve_distributed_checkpoint_io.rebuild_distributed_solver_from_checkpoint",
            return_value=(fake_solver, 500, fake_metadata),
        ) as mock_rebuild:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["solve", "resume", str(checkpoint_file), "--max-iter", "5",
                 "--n-ranks", "2", "--multi-gpu", "--checkpoint-interval", "5"],
            )

        assert result.exit_code == 0, result.output
        assert mock_rebuild.call_args.kwargs["multi_gpu"] is True
        # GPU 版续算用求解器自身的 save_checkpoint_distributed 方法
        # （与 solve_steady_command.py 的 --multi-gpu 分支同一套 API）：
        # 中途 2 次（local iter 5/10 都能整除 checkpoint_interval=5，
        # 绝对迭代数 505/510）+ 最终 1 次 = 3 次调用。
        assert fake_solver.save_checkpoint_distributed.call_count == 3
        written_iterations = [c.args[1] for c in fake_solver.save_checkpoint_distributed.call_args_list]
        assert 505 in written_iterations
        assert 510 in written_iterations
        fake_solver.cleanup.assert_called_once()
