"""`solve transient --n-ranks` / `--multi-gpu` / `--fully-distributed` 的 CLI 接线：参数解析与时间格式映射进入
运行参数（`cli/solve/run.py::SolveRun`），交给与稳态共用的 `run_solve`。分布式求解器的构造本身由
`tests/unit/test_distributed_build.py` 逐项核对。
"""

from unittest.mock import patch

from click.testing import CliRunner

from autoflowcfd.cli.main import cli
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme

_RUN = "autoflowcfd.cli.solve.run.run_solve"


def _invoke(tmp_path, *extra):
    mesh_file = tmp_path / "mesh.pkl"
    mesh_file.write_bytes(b"")
    with patch(_RUN) as run:
        result = CliRunner().invoke(cli, ["solve", "transient", str(mesh_file), "--max-iter", "3", *extra])
    assert result.exit_code == 0, result.output
    run.assert_called_once()
    return run.call_args.args[0]


def test_n_ranks_runs_distributed_dual_time(tmp_path):
    run = _invoke(tmp_path, "--n-ranks", "2", "--time-method", "dual-time")
    assert run.distributed and run.time_scheme is TimeIntegrationScheme.DUAL_TIME and run.time_accurate is False
    assert run.tol == 0.0 and run.max_iter == 3


def test_fully_distributed_accepts_dual_time(tmp_path):
    run = _invoke(tmp_path, "--n-ranks", "2", "--fully-distributed", "--time-method", "dual-time")
    assert run.fully_distributed is True and run.time_scheme is TimeIntegrationScheme.DUAL_TIME


def test_init_from_reaches_the_distributed_run(tmp_path):
    ckpt_file = tmp_path / "ckpt.h5"
    ckpt_file.write_bytes(b"")
    assert _invoke(tmp_path, "--n-ranks", "2", "--init-from", str(ckpt_file)).init_checkpoint == str(ckpt_file)


def test_multi_gpu_with_explicit_scheme_is_time_accurate(tmp_path):
    run = _invoke(tmp_path, "--n-ranks", "2", "--backend", "gpu", "--multi-gpu", "--time-method", "rk3",
                  "--dual-time-inner-iter", "9")
    assert run.multi_gpu is True and run.time_scheme is TimeIntegrationScheme.SSP_RK3 and run.time_accurate is True
    assert run.dual_time_inner_iter == 9


def test_physical_time_sets_the_step_count(tmp_path):
    run = _invoke(tmp_path, "--dt", "1e-3", "--physical-time", "0.01")
    assert run.max_iter == 10 and run.dt == 1e-3
