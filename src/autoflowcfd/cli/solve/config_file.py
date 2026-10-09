"""`solve steady/transient --config <file.yaml>`：YAML 配置文件对 CLI 参数的覆盖（唯一实现）。

优先级：命令行显式给出的选项 > YAML 里**显式写出**的字段 > click 声明的默认值。

2026-10-09 以前（`physical_constants.py`）有三处缺陷：
* 只要某个 CLI 参数取的是 click 默认值，就用配置对象里的同名字段覆盖——而那可能只是配置类自己的默认值，YAML
  里根本没写。实测：只写了物理常数的 YAML 让 `solve steady` 的最大迭代步数从 1000 变成 50（`SteadyConfig.max_iter`
  默认值）、`solve transient` 的时间步长从 1e-5 放大到 1e-4，`physical_time` 取 `total_time` 的默认 0.1 s、把
  `--max-iter` 换算成几千上万步；湍流模型被换成配置默认的 SST。现在只用 YAML 里显式出现的键。
* 只处理物理常数与少数几个同名字段：`cfl_init`/`n_threads`/`checkpoint_interval`/`output_dir`/`gpu_device`/
  `backend`/`time_scheme`/`init_from_checkpoint`/`convergence_tol` 写进 YAML 不起作用。现在凡是有对应 CLI 参数的
  配置字段都生效（名字不同的经 `_CLI_NAME`，枚举经 `_cli_value` 换成 CLI 词汇）。
* YAML 里出现本命令没有的参数（例如瞬态字段写进稳态配置）时静默忽略；现在报错。
"""

import functools
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional

import click

#: 配置字段 -> CLI 参数名（其余同名）
_CLI_NAME = {
    "turbulence": "turbulence_model",
    "n_threads": "threads",
    "cfl_init": "cfl_start",
    "convergence_tol": "tol",
    "total_time": "physical_time",
    "init_from_checkpoint": "init_checkpoint",
    "time_scheme": "time_method",
}


@dataclass(frozen=True)
class ConfigFile:
    """加载后的 YAML 配置：配置对象 + YAML 里显式写出的字段名。"""

    config: Any
    keys: FrozenSet[str]


def load_config_file(path: Optional[str]) -> Optional[ConfigFile]:
    """加载 `--config` 指定的 YAML（未提供时 None）。经 `ConfigLoader` 构造配置对象（同一套校验）。"""
    if path is None:
        return None
    from autoflowcfd.config.loader import ConfigLoader

    config, keys = ConfigLoader().load_with_keys(path)
    return ConfigFile(config=config, keys=frozenset(keys))


def _cli_value(param: click.Parameter, field: str, value: Any) -> Any:
    """配置字段值 -> CLI 参数值（枚举换成该参数的 click 取值）。"""
    if field == "turbulence":
        from autoflowcfd.config.turbulence_names import turbulence_solver_name

        return turbulence_solver_name(value)
    if field == "backend":
        from autoflowcfd.cli.solve.solver_factory import resolve_backend_name

        return resolve_backend_name(getattr(value, "value", value))
    if field == "time_scheme":
        from autoflowcfd.core.time_integration.base import scheme_from_name

        scheme = scheme_from_name(value)
        for choice in param.type.choices:
            if scheme_from_name(choice) == scheme:
                return choice
        raise click.BadParameter(
            f"配置文件的 time_scheme={scheme.value} 不是本命令支持的时间格式（{', '.join(param.type.choices)}）",
            param_hint="--config")
    return value


def apply_config_file(ctx: click.Context, config_file: Optional[ConfigFile]) -> Dict[str, Any]:
    """返回 {CLI 参数名: 值}：YAML 显式给出、且该 CLI 参数本次取的是 click 默认值的项。

    Raises:
        click.BadParameter: YAML 里有本命令不接受的字段
    """
    if config_file is None:
        return {}
    from click.core import ParameterSource

    params = {p.name: p for p in ctx.command.params}
    unknown = sorted(f for f in config_file.keys if _CLI_NAME.get(f, f) not in params)
    if unknown:
        raise click.BadParameter(
            f"配置文件里的字段 {unknown} 在 `{ctx.command.name}` 命令里没有对应参数", param_hint="--config")
    out = {}
    for field in config_file.keys:
        name = _CLI_NAME.get(field, field)
        if ctx.get_parameter_source(name) == ParameterSource.DEFAULT:
            out[name] = _cli_value(params[name], field, getattr(config_file.config, field))
    return out


def config_file_overrides(callback):
    """命令函数装饰器（放在 `def` 正上方）：调用前按 `--config` 改写取默认值的参数（`apply_config_file`）。"""

    @functools.wraps(callback)
    def wrapper(**kwargs):
        ctx = click.get_current_context()
        kwargs.update(apply_config_file(ctx, load_config_file(kwargs.get("config_path"))))
        return callback(**kwargs)

    return wrapper


def check_physical_ranges(turbulence_intensity: float, viscosity_ratio: float, mu_molecular: float,
                          rho_inf: float, vel_inf: float, p_inf: float) -> None:
    """来流物理量的范围校验（`solve steady`/`solve transient` 共用；CLI 路径不经过配置类的构造校验）。"""
    checks = (
        (0.0 < turbulence_intensity <= 1.0, "湍流强度 Tu 必须在 (0, 1] 区间", "--turbulence-intensity"),
        (viscosity_ratio > 0.0, "粘性比 VR 必须 > 0", "--viscosity-ratio"),
        (mu_molecular > 0.0, "分子动力粘度必须 > 0", "--mu-molecular"),
        (rho_inf > 0.0, "自由流密度必须 > 0", "--rho-inf"),
        (vel_inf > 0.0, "自由流速度必须 > 0", "--vel-inf"),
        (p_inf > 0.0, "自由流静压必须 > 0", "--p-inf"),
    )
    for ok, message, hint in checks:
        if not ok:
            raise click.BadParameter(message, param_hint=hint)
