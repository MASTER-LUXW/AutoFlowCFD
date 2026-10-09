"""`solve steady/transient --config`：只用 YAML 里显式写出的字段覆盖选项默认值（2026-10-09）。

此前只要某个 CLI 参数取的是 click 默认值，就用配置对象里的同名字段覆盖——而那可能只是配置类自己的默认值：
只写了物理常数的 YAML 让稳态最大迭代步数 1000 -> 50、瞬态时间步长 1e-5 -> 1e-4、物理时间取 0.1 s、湍流模型换成
SST。见 `cli/solve/config_file.py` 模块文档。
"""

import click
import pytest

from autoflowcfd.cli.main import cli
from autoflowcfd.cli.solve.config_file import apply_config_file, load_config_file


def _context(command: str, args):
    cmd = cli.commands["solve"].commands[command]
    return cmd.make_context(command, list(args))


def _overrides(tmp_path, command, yaml_text, args=("mesh.pkl",)):
    path = tmp_path / "case.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    (tmp_path / "mesh.pkl").write_bytes(b"")
    ctx = _context(command, [str(tmp_path / a) if a.endswith(".pkl") else a for a in args])
    return apply_config_file(ctx, load_config_file(str(path)))


def test_only_explicit_yaml_keys_override(tmp_path):
    got = _overrides(tmp_path, "steady", "mode: steady\nrho_inf: 1.1\nvel_inf: 20.0\n")
    assert got == {"rho_inf": 1.1, "vel_inf": 20.0}, "max_iter/turbulence_model 等不能被配置类默认值覆盖"


def test_transient_physical_constants_only_keeps_dt_and_physical_time(tmp_path):
    got = _overrides(tmp_path, "transient", "mode: transient\nmu_molecular: 2.0e-5\n")
    assert got == {"mu_molecular": 2.0e-5}


def test_renamed_fields_and_enums_map_to_cli_values(tmp_path):
    got = _overrides(tmp_path, "transient",
                     "mode: transient\nn_threads: 8\ncfl_init: 0.02\ncfl_max: 0.05\ncfl_min: 0.01\n"
                     "total_time: 0.02\ndt: 0.0001\ntime_scheme: dual_time\nturbulence: les\n"
                     "init_from_checkpoint: steady.h5\ncheckpoint_interval: 7\n")
    assert got["threads"] == 8
    assert got["cfl_start"] == 0.02 and got["cfl_max"] == 0.05 and got["cfl_min"] == 0.01
    assert got["physical_time"] == 0.02 and got["dt"] == 0.0001
    assert got["time_method"] == "dual-time"
    assert got["turbulence_model"] == "les"
    assert got["init_checkpoint"] == "steady.h5"
    assert got["checkpoint_interval"] == 7


def test_steady_convergence_tol_maps_to_tol(tmp_path):
    got = _overrides(tmp_path, "steady", "mode: steady\nconvergence_tol: 1.0e-8\nmax_iter: 300\n")
    assert got == {"tol": 1.0e-8, "max_iter": 300}


def test_backend_auto_resolves_to_a_concrete_backend(tmp_path):
    got = _overrides(tmp_path, "steady", "mode: steady\nbackend: auto\n")
    assert got["backend"] in ("cpu", "gpu")


def test_explicit_command_line_option_wins(tmp_path):
    got = _overrides(tmp_path, "steady", "mode: steady\nrho_inf: 1.1\nmax_iter: 300\n",
                     args=("mesh.pkl", "--max-iter", "42"))
    assert got == {"rho_inf": 1.1}


def test_field_without_a_matching_option_is_rejected(tmp_path):
    """瞬态配置用于稳态命令：`dt`/`total_time` 在 `solve steady` 里没有对应参数，不能静默忽略。"""
    with pytest.raises(click.BadParameter, match="没有对应参数"):
        _overrides(tmp_path, "steady", "mode: transient\ndt: 0.0001\n")


def test_time_scheme_unsupported_by_the_command_is_rejected(tmp_path):
    with pytest.raises(click.BadParameter, match="time_scheme"):
        _overrides(tmp_path, "transient", "mode: transient\ntime_scheme: newton_krylov\n")


def test_every_config_field_has_a_command_line_counterpart():
    """配置文件是 CLI 参数的另一种写法：每个配置字段都必须对应某个选项（此前网格生成参数、
    use_wall_functions、monitor_coefficients、sample_interval、warmup_time、verbose 写了都不起作用）。"""
    import dataclasses

    from autoflowcfd.cli.solve.config_file import _CLI_NAME
    from autoflowcfd.config.solver_config import SteadyConfig, TransientConfig

    for command, cls in (("steady", SteadyConfig), ("transient", TransientConfig)):
        params = {p.name for p in cli.commands["solve"].commands[command].params}
        missing = [f.name for f in dataclasses.fields(cls) if f.init and _CLI_NAME.get(f.name, f.name) not in params]
        assert not missing, f"{cls.__name__} 字段 {missing} 在 solve {command} 里没有对应选项"
