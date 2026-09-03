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


class TestSolveTransientDistributedCliWiring:
    def test_n_ranks_dispatches_to_distributed_with_dual_time_scheme(self, tmp_path):
        mesh_file = tmp_path / "mesh.pkl"
        mesh_file.write_bytes(b"")

        with patch(
            "autoflowcfd.cli.solve_transient_command._solve_transient_distributed"
        ) as mock_distributed:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["solve", "transient", str(mesh_file), "--n-ranks", "2",
                 "--time-method", "dual-time", "--max-iter", "3"],
            )

        assert result.exit_code == 0, result.output
        mock_distributed.assert_called_once()
        call_args = mock_distributed.call_args[0]
        # 位置参数顺序见 _solve_transient_distributed 签名（2026-09-03
        # 起不再有 tet_basis_mode，见 fr/operators.py 模块文档"删除
        # collapsed 相关内容"一节）：
        # (input_file, order, surface_mesh, skip_quality_check,
        #  time_scheme, dual_time_inner_iter, ...)
        assert call_args[4] == TimeIntegrationScheme.DUAL_TIME

    def test_fully_distributed_accepts_dual_time(self, tmp_path):
        """2026-09-02 起 `--fully-distributed` + `--time-method
        dual-time` 不再被拒绝——`distributed_mesh_load_v2`/
        `build_fully_distributed_rank_package` 已接入 `time_scheme`/
        `dual_time_inner_iter` 字段（见 core/mpi/distributed_mesh_
        loader.py 模块文档）。"""
        mesh_file = tmp_path / "mesh.pkl"
        mesh_file.write_bytes(b"")

        with patch(
            "autoflowcfd.cli.solve_transient_command._solve_transient_distributed"
        ) as mock_distributed:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["solve", "transient", str(mesh_file), "--n-ranks", "2",
                 "--fully-distributed", "--time-method", "dual-time", "--max-iter", "3"],
            )

        assert result.exit_code == 0, result.output
        mock_distributed.assert_called_once()
        call_args = mock_distributed.call_args[0]
        assert call_args[4] == TimeIntegrationScheme.DUAL_TIME
        # fully_distributed 是 _solve_transient_distributed 签名里的
        # 第 22 个位置参数（索引 21，multi_gpu 之后一位——见
        # test_multi_gpu_dispatches_with_rk3_default 同一处参数顺序
        # 注释，那里 multi_gpu=索引20；tet_basis_mode 已删除，所有索引
        # 相对旧版本整体前移 1）。
        assert call_args[21] is True

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
            "autoflowcfd.cli.solve_transient_command._solve_transient_distributed"
        ) as mock_distributed:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["solve", "transient", str(mesh_file), "--n-ranks", "2",
                 "--init-from", str(ckpt_file), "--max-iter", "3"],
            )

        assert result.exit_code == 0, result.output
        mock_distributed.assert_called_once()
        call_args = mock_distributed.call_args[0]
        # init_checkpoint 是 _solve_transient_distributed 签名里的最后
        # 一个位置参数（第 28 个，索引 27——tet_basis_mode 已删除，
        # 所有索引相对旧版本整体前移 1，见 test_multi_gpu_dispatches_
        # with_rk3_default 同一处参数顺序注释）。
        assert call_args[27] == str(ckpt_file)

    def test_multi_gpu_dispatches_with_rk3_default(self, tmp_path):
        mesh_file = tmp_path / "mesh.pkl"
        mesh_file.write_bytes(b"")

        with patch(
            "autoflowcfd.cli.solve_transient_command._solve_transient_distributed"
        ) as mock_distributed:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["solve", "transient", str(mesh_file), "--n-ranks", "2",
                 "--multi-gpu", "--max-iter", "3"],
            )

        assert result.exit_code == 0, result.output
        mock_distributed.assert_called_once()
        call_args = mock_distributed.call_args[0]
        assert call_args[4] == TimeIntegrationScheme.SSP_RK3
        # multi_gpu 是 _solve_transient_distributed 签名里第 21 个位置
        # 参数（索引 20：input_file/order/surface_mesh/skip_quality_check/
        # time_scheme/dual_time_inner_iter/turbulence_model/
        # max_iter/dt/use_eikonal/output_dir/reference_area/threads/
        # turbulence_intensity/viscosity_ratio/mu_molecular/rho_inf/
        # vel_inf/p_inf/n_ranks/multi_gpu——tet_basis_mode 已删除，
        # 所有索引相对旧版本整体前移 1）。
        assert call_args[20] is True
