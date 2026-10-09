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

import pytest
from click.testing import CliRunner

from autoflowcfd.core.fr_solver.state import SolverResult

from autoflowcfd.cli.main import cli


#: 假求解器的 solve() 在第 5、10 步各调一次回调，即跑了 10 步。两个分布式后端都返回 `SolverResult`
#: （2026-10-05 起共用求解循环；CLI 用它把实际步数写进最终 checkpoint，而不是 max_iter）。
_CPU_RESULT = SolverResult(converged=False, iterations=10, final_residual=1e-4)
_GPU_RESULT = SolverResult(converged=True, iterations=10, final_residual=1e-4)


@pytest.fixture(autouse=True)
def _aero_report_calls(monkeypatch):
    """收尾气动系数报告要汇总真实求解器状态（替身求解器没有），这里只记录调用——报告本身与单机逐位一致
    由 test_distributed_aero_report.py 验证。"""
    calls = []
    monkeypatch.setattr("autoflowcfd.cli.solve.commands.report_distributed_aerodynamic_coefficients",
                        lambda solver, area: calls.append((solver, area)))
    return calls


def _fake_distributed_solver(solve_return):
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

    def test_checkpoint_callback_wired_with_absolute_iteration(self, tmp_path, _aero_report_calls):
        checkpoint_file = tmp_path / "checkpoint_iter_002000.h5"
        checkpoint_file.write_bytes(b"")

        fake_solver = _fake_distributed_solver(solve_return=_CPU_RESULT)
        fake_metadata = {
            "input_file": "volume.nas",
            "order": 1,
            "turbulence_model": "none",
            "backend": "cpu",
            "surface_mesh": "surface.nas",
        }

        with patch(
            "autoflowcfd.cli.solve.distributed_checkpoint_io.rebuild_distributed_solver_from_checkpoint",
            return_value=(fake_solver, 2000, fake_metadata),
        ) as mock_rebuild, patch(
            "autoflowcfd.core.mpi.distributed_checkpoint.distributed_save_results"
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["solve", "resume", str(checkpoint_file), "--max-iter", "10",
                 "--n-ranks", "2", "--checkpoint-interval", "5"],
            )
        # 两个分布式后端都经求解器自身的 save_checkpoint_distributed（同名同签名）
        mock_save_checkpoint = fake_solver.save_checkpoint_distributed

        assert result.exit_code == 0, result.output
        assert [c[0] for c in _aero_report_calls] == [fake_solver]   # 续算收尾报告气动系数
        mock_rebuild.assert_called_once()
        assert mock_rebuild.call_args.kwargs["n_ranks"] == 2
        assert mock_rebuild.call_args.kwargs["multi_gpu"] is False
        assert mock_rebuild.call_args.kwargs["fully_distributed"] is False

        assert fake_solver.solve.call_args.kwargs.get("checkpoint_callback") is not None

        # 2 次中途保存（local iter 5/10）+ 1 次最终保存 = 3 次。
        assert mock_save_checkpoint.call_count == 3
        written_iterations = [c.args[1] for c in mock_save_checkpoint.call_args_list]
        assert 2005 in written_iterations
        assert 2010 in written_iterations
        assert 5 not in written_iterations
        assert 10 not in written_iterations


class TestResumeDistributedFullyDistributed(object):
    def test_fully_distributed_flag_passed_through(self, tmp_path):
        checkpoint_file = tmp_path / "checkpoint_iter_001000.h5"
        checkpoint_file.write_bytes(b"")

        fake_solver = _fake_distributed_solver(solve_return=_CPU_RESULT)
        fake_metadata = {
            "input_file": "volume.nas", "order": 2, "turbulence_model": "sst",
            "backend": "cpu", "surface_mesh": "surface.nas",
        }

        with patch(
            "autoflowcfd.cli.solve.distributed_checkpoint_io.rebuild_distributed_solver_from_checkpoint",
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

        fake_solver = _fake_distributed_solver(solve_return=_GPU_RESULT)
        fake_solver.save_checkpoint_distributed.return_value = "fake_ckpt_path.h5"
        fake_metadata = {
            "input_file": "volume.nas", "order": 1, "turbulence_model": "none",
            "backend": "gpu", "surface_mesh": None,
        }

        with patch(
            "autoflowcfd.cli.solve.distributed_checkpoint_io.rebuild_distributed_solver_from_checkpoint",
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
        # 最终 checkpoint 记录实际跑到的绝对步数（500 + 10）
        assert written_iterations[-1] == 510
        fake_solver.cleanup.assert_called_once()


class TestResumeDistributedPhaseMaxIterForwarding(object):
    """真实 bug 回归测试（2026-09-05，用户直接问"--residual-drop-
    threshold phase_max_iter 可以在 resume 重置吗"发现）：`resume()`
    顶层确实解析了这两个 Order Continuation CLI 选项，但 `_resume_
    distributed` 此前的签名根本不接收它们，两处 `solver.solve(...)`
    调用也完全没有传递——用户对分布式/多GPU resume 传
    `--phase-max-iter`/`--residual-drop-threshold` 会被静默忽略，
    实际生效的永远是 `DistributedFRSolver.solve`/
    `MultiGPUDistributedSolver.solve` 自身的函数签名默认值
    （`None`/`100.0`），不是用户的真实意图。"""

    def test_cpu_traditional_mode_forwards_explicit_values(self, tmp_path):
        checkpoint_file = tmp_path / "checkpoint_iter_002000.h5"
        checkpoint_file.write_bytes(b"")

        fake_solver = _fake_distributed_solver(solve_return=_CPU_RESULT)
        fake_metadata = {
            "input_file": "volume.nas", "order": 2, "turbulence_model": "none",
            "backend": "cpu", "surface_mesh": "surface.nas",
        }

        with patch(
            "autoflowcfd.cli.solve.distributed_checkpoint_io.rebuild_distributed_solver_from_checkpoint",
            return_value=(fake_solver, 2000, fake_metadata),
        ), patch(
            "autoflowcfd.core.mpi.distributed_checkpoint.distributed_save_results"
        ), patch(
            "autoflowcfd.core.mpi.distributed_checkpoint.distributed_save_checkpoint"
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["solve", "resume", str(checkpoint_file), "--max-iter", "10",
                 "--n-ranks", "2", "--phase-max-iter", "37",
                 "--residual-drop-threshold", "250.0"],
            )

        assert result.exit_code == 0, result.output
        assert fake_solver.solve.call_args.kwargs["phase_max_iter"] == 37
        assert fake_solver.solve.call_args.kwargs["residual_drop_threshold"] == 250.0

    def test_multi_gpu_forwards_explicit_values(self, tmp_path):
        checkpoint_file = tmp_path / "checkpoint_iter_000500.h5"
        checkpoint_file.write_bytes(b"")

        fake_solver = _fake_distributed_solver(solve_return=_GPU_RESULT)
        fake_solver.save_checkpoint_distributed.return_value = "fake_ckpt_path.h5"
        fake_metadata = {
            "input_file": "volume.nas", "order": 2, "turbulence_model": "none",
            "backend": "gpu", "surface_mesh": None,
        }

        with patch(
            "autoflowcfd.cli.solve.distributed_checkpoint_io.rebuild_distributed_solver_from_checkpoint",
            return_value=(fake_solver, 500, fake_metadata),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["solve", "resume", str(checkpoint_file), "--max-iter", "5",
                 "--n-ranks", "2", "--multi-gpu",
                 "--phase-max-iter", "12", "--residual-drop-threshold", "99.0"],
            )

        assert result.exit_code == 0, result.output
        assert fake_solver.solve.call_args.kwargs["phase_max_iter"] == 12
        assert fake_solver.solve.call_args.kwargs["residual_drop_threshold"] == 99.0

    def test_cpu_traditional_mode_default_none_is_forwarded_not_dropped(self, tmp_path):
        """不传 `--phase-max-iter` 时，它仍必须作为显式的 `phase_max_iter=None`
        关键字参数到达 `solver.solve`（之后由被调方自己的默认值逻辑接管，见
        `run_order_continuation` 文档）——而不是在调用里被静默省略（从这个测试的
        角度两者无法区分，但要点是接线本身，而不只是非默认值的情形）。
        """
        checkpoint_file = tmp_path / "checkpoint_iter_002000.h5"
        checkpoint_file.write_bytes(b"")

        fake_solver = _fake_distributed_solver(solve_return=_CPU_RESULT)
        fake_metadata = {
            "input_file": "volume.nas", "order": 2, "turbulence_model": "none",
            "backend": "cpu", "surface_mesh": "surface.nas",
        }

        with patch(
            "autoflowcfd.cli.solve.distributed_checkpoint_io.rebuild_distributed_solver_from_checkpoint",
            return_value=(fake_solver, 2000, fake_metadata),
        ), patch(
            "autoflowcfd.core.mpi.distributed_checkpoint.distributed_save_results"
        ), patch(
            "autoflowcfd.core.mpi.distributed_checkpoint.distributed_save_checkpoint"
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["solve", "resume", str(checkpoint_file), "--max-iter", "10", "--n-ranks", "2"],
            )

        assert result.exit_code == 0, result.output
        assert fake_solver.solve.call_args.kwargs["phase_max_iter"] is None
        assert fake_solver.solve.call_args.kwargs["residual_drop_threshold"] == 100.0


def test_distributed_periodic_checkpoint_callback_is_shared_by_both_backends():
    """CPU MPI 与多 GPU 的中间 checkpoint 回调是同一个工厂（2026-10-05，此前 CLI 里 7 份拷贝）：按间隔调用
    求解器的 `save_checkpoint_distributed`（两个后端同名同签名），resume 时迭代数加起点偏移。"""
    import inspect
    from unittest.mock import MagicMock

    from autoflowcfd.cli.solve.distributed_checkpoint_io import distributed_periodic_checkpoint_callback
    from autoflowcfd.core.gpu.distributed.gpu_distributed import MultiGPUDistributedSolver
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

    sig_cpu = inspect.signature(DistributedFRSolver.save_checkpoint_distributed).parameters
    sig_gpu = inspect.signature(MultiGPUDistributedSolver.save_checkpoint_distributed).parameters
    assert list(sig_cpu) == list(sig_gpu)

    solver = MagicMock(current_order=1, order=2)
    solver.save_checkpoint_distributed.return_value = None
    cb = distributed_periodic_checkpoint_callback(5, "out", "mesh.pkl", "sa", surface_mesh="s.nas",
                                                  iteration_offset=3000)
    for it in range(1, 11):
        cb(solver, it)
    calls = solver.save_checkpoint_distributed.call_args_list
    assert [c.args[1] for c in calls] == [3005, 3010]
    assert all(c.args[3] == 1 and c.kwargs["target_order"] == 2 and c.kwargs["surface_mesh"] == "s.nas"
               for c in calls)
