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
