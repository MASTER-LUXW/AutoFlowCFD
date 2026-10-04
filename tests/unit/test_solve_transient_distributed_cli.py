"""`solve transient --n-ranks`/`--multi-gpu`/`--fully-distributed` CLI
wiring 验证（2026-09-02，见 solve_transient_command.py/
solve_transient_distributed.py 文档）——此前本命令完全没有分布式支持。

与既有的 `test_solve_resume_distributed.py` 同一个方法论：mock 掉
`_solve_transient_distributed`（分布式重建/求解逻辑本身复用的
`DistributedFRSolver`/`MultiGPUDistributedSolver`/`load_mesh_for_
solver` 都已在各自的单元测试里决定性验证过），只验证 CLI 层的
wiring：参数解析、time_scheme 映射是否正确、不允许的参数组合是否
真的被拒绝。
"""

from unittest.mock import patch

from click.testing import CliRunner

from autoflowcfd.cli.main import cli
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme


def _arg(mock, name):
    """按 `_solve_transient_distributed` 的签名把被 mock 的调用绑定到参数名（此前按位置下标取，
    签名每增删一个参数就整体错位）。"""
    import inspect

    from autoflowcfd.cli.solve.transient_distributed import _solve_transient_distributed

    args, kwargs = mock.call_args
    return inspect.signature(_solve_transient_distributed).bind(*args, **kwargs).arguments[name]


class TestSolveTransientDistributedCliWiring:
    def test_n_ranks_dispatches_to_distributed_with_dual_time_scheme(self, tmp_path):
        mesh_file = tmp_path / "mesh.pkl"
        mesh_file.write_bytes(b"")

        with patch(
            "autoflowcfd.cli.solve.transient._solve_transient_distributed"
        ) as mock_distributed:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["solve", "transient", str(mesh_file), "--n-ranks", "2",
                 "--time-method", "dual-time", "--max-iter", "3"],
            )

        assert result.exit_code == 0, result.output
        mock_distributed.assert_called_once()
        assert _arg(mock_distributed, "time_scheme") == TimeIntegrationScheme.DUAL_TIME

    def test_fully_distributed_accepts_dual_time(self, tmp_path):
        """2026-09-02 起 `--fully-distributed` + `--time-method
        dual-time` 不再被拒绝——`distributed_mesh_load_v2`/
        `build_fully_distributed_rank_package` 已接入 `time_scheme`/
        `dual_time_inner_iter` 字段（见 core/mpi/distributed_mesh_
        loader.py 模块文档）。"""
        mesh_file = tmp_path / "mesh.pkl"
        mesh_file.write_bytes(b"")

        with patch(
            "autoflowcfd.cli.solve.transient._solve_transient_distributed"
        ) as mock_distributed:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["solve", "transient", str(mesh_file), "--n-ranks", "2",
                 "--fully-distributed", "--time-method", "dual-time", "--max-iter", "3"],
            )

        assert result.exit_code == 0, result.output
        mock_distributed.assert_called_once()
        assert _arg(mock_distributed, "time_scheme") == TimeIntegrationScheme.DUAL_TIME
        assert _arg(mock_distributed, "fully_distributed") is True

    def test_init_from_dispatches_with_distributed(self, tmp_path):
        """2026-09-02 起 `--init-from` + `--n-ranks`/`--multi-gpu`/
        `--fully-distributed` 不再被拒绝——`restore_distributed_state_
        from_checkpoint`（见 core/mpi/distributed_checkpoint.py 模块
        文档）已接入三条分布式路径。"""
        mesh_file = tmp_path / "mesh.pkl"
        mesh_file.write_bytes(b"")
        ckpt_file = tmp_path / "ckpt.h5"
        ckpt_file.write_bytes(b"")

        with patch(
            "autoflowcfd.cli.solve.transient._solve_transient_distributed"
        ) as mock_distributed:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["solve", "transient", str(mesh_file), "--n-ranks", "2",
                 "--init-from", str(ckpt_file), "--max-iter", "3"],
            )

        assert result.exit_code == 0, result.output
        mock_distributed.assert_called_once()
        assert _arg(mock_distributed, "init_checkpoint") == str(ckpt_file)

    def test_multi_gpu_dispatches_with_rk3_default(self, tmp_path):
        mesh_file = tmp_path / "mesh.pkl"
        mesh_file.write_bytes(b"")

        with patch(
            "autoflowcfd.cli.solve.transient._solve_transient_distributed"
        ) as mock_distributed:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["solve", "transient", str(mesh_file), "--n-ranks", "2",
                 "--multi-gpu", "--max-iter", "3"],
            )

        assert result.exit_code == 0, result.output
        mock_distributed.assert_called_once()
        assert _arg(mock_distributed, "time_scheme") == TimeIntegrationScheme.SSP_RK3
        assert _arg(mock_distributed, "multi_gpu") is True
