"""配置层与 CLI 的取值范围单一来源、配置类是唯一校验点（I-183/I-163，2026-10-18）。

修复前：
* 阶数——配置层只许 1/2/3，`solve steady --order` 允许 0~3，`config init` 模板注释写 0~3：YAML 写 `order: 0` 加载失败；
* 稳态配置没有 `time_scheme`：YAML 与配置对象选不了 newton-krylov；
* `n_threads: -1` 在构造时被改写成本机核数：`config init` 模板写死生成机器的核数，CLI 的 -1（= 8）与配置层的
  -1（= 核数）不是同一个线程数；CLI 帮助写 "-1 = 4"，求解器实际取 8；
* `config/schema.py::ConfigSchema` 另做一套校验（口径不同、生产不调用）；直接构造 `SteadyConfig(backend="gpuu")`
  不报错，只有经 YAML 加载器才转换枚举；
* 模板注释表漏掉一半字段。
"""

import dataclasses

import pytest

from autoflowcfd.cli.main import cli
from autoflowcfd.config.loader import ConfigLoader
from autoflowcfd.config.solver_config import BackendType, SteadyConfig, TransientConfig, TurbulenceModel
from autoflowcfd.core.fr_solver.solver.threads import DEFAULT_N_THREADS
from autoflowcfd.core.time_integration.base import STEADY_SCHEMES, TimeIntegrationScheme as S
from autoflowcfd.core.utils.order_continuation.policy import MAX_ORDER, MIN_STEADY_ORDER, MIN_TRANSIENT_ORDER


def _option(command, name):
    return next(p for p in cli.commands["solve"].commands[command].params if p.name == name)


@pytest.mark.parametrize("command,cls,lo", [("steady", SteadyConfig, MIN_STEADY_ORDER),
                                            ("transient", TransientConfig, MIN_TRANSIENT_ORDER)])
def test_order_range_is_the_same_on_cli_and_config(command, cls, lo):
    opt = _option(command, "order").type
    assert (opt.min, opt.max) == (lo, MAX_ORDER)
    for order in range(lo, MAX_ORDER + 1):
        assert cls(order=order).order == order
    for bad in (lo - 1, MAX_ORDER + 1):
        with pytest.raises(ValueError, match="目标阶数"):
            cls(order=bad)


def test_steady_time_scheme_field_matches_cli_choices():
    from autoflowcfd.core.time_integration.base import scheme_from_name

    choices = {scheme_from_name(c) for c in _option("steady", "time_scheme").type.choices}
    assert choices == set(STEADY_SCHEMES)
    assert SteadyConfig().time_scheme is S.SSP_RK3
    assert SteadyConfig(time_scheme="newton-krylov").time_scheme is S.NEWTON_KRYLOV
    with pytest.raises(ValueError, match="时间格式"):
        SteadyConfig(time_scheme="dual-time")
    with pytest.raises(ValueError, match="时间格式"):
        TransientConfig(time_scheme="newton-krylov")


def test_resume_override_schemes_are_the_steady_schemes():
    from autoflowcfd.cli.solve.checkpoint_io.rebuild import resolve_resume_time_scheme

    for sch in STEADY_SCHEMES:
        assert resolve_resume_time_scheme(sch.value, {}) is sch


def test_n_threads_minus_one_is_kept_and_means_the_solver_default():
    assert SteadyConfig().n_threads == -1 and TransientConfig(n_threads=-1).n_threads == -1
    from autoflowcfd.cli.solve.option_help import THREADS_HELP

    assert f"-1 = {DEFAULT_N_THREADS}" in THREADS_HELP
    with pytest.raises(ValueError, match="线程数"):
        SteadyConfig(n_threads=0)


def test_direct_construction_validates_enums():
    c = SteadyConfig(backend="GPU", turbulence="SA")
    assert c.backend is BackendType.GPU and c.turbulence is TurbulenceModel.SA
    with pytest.raises(ValueError, match="后端"):
        SteadyConfig(backend="gpuu")
    with pytest.raises(ValueError, match="湍流模型"):
        TransientConfig(turbulence="sstt")
    with pytest.raises(ValueError, match="checkpoint_interval"):
        SteadyConfig(checkpoint_interval=0)


def test_no_second_validation_layer():
    import importlib.util

    assert importlib.util.find_spec("autoflowcfd.config.schema") is None


@pytest.mark.parametrize("mode,cls", [("steady", SteadyConfig), ("transient", TransientConfig)])
def test_template_comments_cover_every_field_and_template_round_trips(tmp_path, mode, cls):
    loader = ConfigLoader()
    missing = [f.name for f in dataclasses.fields(cls) if f.init and not loader._get_parameter_comment(f.name, mode)]
    assert missing == [], f"模板注释缺字段 {missing}"
    path = tmp_path / f"{mode}.yaml"
    loader.save_template(path, mode)
    text = path.read_text(encoding="utf-8")
    assert "n_threads: -1" in text, "模板不能写死生成机器的核数"
    loaded = loader.load(path)
    assert type(loaded) is cls and loaded == cls()


@pytest.mark.parametrize("command,cls", [("steady", SteadyConfig), ("transient", TransientConfig)])
def test_cli_option_defaults_are_the_config_defaults(command, cls):
    """命令行选项默认值读配置类（`cli/solve/shared_options.py`）。2026-10-19 以前各写一份：稳态最大步数 1000 vs 50、
    瞬态湍流模型 ddes vs sst、瞬态 dt 1e-5 vs 1e-4、瞬态输出目录、--gpu-device。"""
    from autoflowcfd.cli.solve.config_file import _cli_name, _cli_value

    params = {p.name: p for p in cli.commands["solve"].commands[command].params}
    default = cls()
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        p = params[_cli_name(f.name, params)]
        assert p.default == _cli_value(p, f.name, getattr(default, f.name)), (command, f.name)


def test_multi_gpu_default_device_lets_each_rank_pick_its_own(tmp_path):
    """`solve steady --multi-gpu` 不给 --gpu-device 时传 None（多 GPU 求解器按 rank % GPU 数分配）。修复前默认 0，
    每个 rank 都用 0 号 GPU，与帮助文字"每个 rank 自动分配"不符。"""
    from unittest.mock import patch

    from click.testing import CliRunner

    mesh = tmp_path / "m.pkl"
    mesh.write_bytes(b"")
    with patch("autoflowcfd.cli.solve.run.run_solve") as run:
        res = CliRunner().invoke(cli, ["solve", "steady", str(mesh), "--backend", "gpu", "--multi-gpu",
                                       "--n-ranks", "2"])
    assert res.exit_code == 0, res.output
    assert run.call_args.args[0].gpu_device is None


@pytest.mark.parametrize("field, value, option", [
    ("rho_inf", -1.0, "--rho-inf"), ("vel_inf", 0.0, "--vel-inf"), ("p_inf", -5.0, "--p-inf"),
    ("turbulence_intensity", 0.0, "--turbulence-intensity"), ("viscosity_ratio", -2.0, "--viscosity-ratio"),
    ("mu_molecular", 0.0, "--mu-molecular"),
])
def test_physical_ranges_have_one_source(field, value, option):
    """来流物理量的范围判据只有一份（I-82）：配置类与 CLI 给出同一条说明，CLI 只换成选项名。此前 SolverConfig、
    Steady/TransientConfig 与 CLI 各写一份（措辞已经各不相同）。"""
    import click

    from autoflowcfd.cli.solve.config_file import check_physical_ranges
    from autoflowcfd.config.solver_config import physical_range_errors

    [(name, message)] = physical_range_errors(**{field: value})
    assert name == field
    for cls in (SteadyConfig, TransientConfig):
        with pytest.raises(ValueError) as err:
            cls(**{field: value})
        assert message in str(err.value)
    good = dict(turbulence_intensity=0.01, viscosity_ratio=5.0, mu_molecular=1.8e-5, rho_inf=1.2, vel_inf=30.0,
                p_inf=1.0e5)
    good[field] = value
    with pytest.raises(click.BadParameter) as err:
        check_physical_ranges(**good)
    assert err.value.message == message and err.value.param_hint == option
