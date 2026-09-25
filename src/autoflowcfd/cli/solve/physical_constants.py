"""物理常量（mu_molecular/rho_inf/vel_inf/p_inf/turbulence_intensity/
viscosity_ratio）的 CLI/YAML 配置解析 —— 供 `solve steady`/`solve
transient` 共用。

此前这些"基础物理量"要么被硬编码在求解器构造参数的默认值上、要么完全
没有 CLI 选项可覆盖（例如 `solve_steady_command.py` 构造
`MultiGPUDistributedSolver` 时直接写死 `mu_molecular=1.8e-5`），用户
除了改源码没有任何办法为非标准状态（不同流体/温度/高度）设置正确的
物理常量。现在这些量都有对应的 `--xxx` CLI 选项（保留原有默认值），并
且都能从 `--config <file.yaml>` 指定的 YAML 文件里读取。

优先级：显式传入的 CLI 选项 > `--config` YAML 文件里的同名字段 > 内建
默认值。用 click 的 `get_parameter_source` 区分"用户真的敲了这个选项"
和"没敲、用的是 click 声明的默认值"，而不是简单判断值是否等于默认值
（否则用户明确指定 `--mu-molecular 1.8e-5`，恰好等于默认值，会被误判为
"未指定"进而被 YAML 值覆盖，行为不一致）。
"""

from typing import Any, Dict, Optional


def load_physical_config_if_given(config_path: Optional[str]):
    """加载 `--config` YAML 文件（未提供时返回 None）。

    复用 config/loader.py::ConfigLoader（已有的、经过验证的 YAML 加载+
    合并默认值+校验实现），不重新发明一套解析逻辑。
    """
    if config_path is None:
        return None
    from autoflowcfd.config.loader import ConfigLoader
    return ConfigLoader().load(config_path)


def resolve_physical_constants(ctx, values: Dict[str, Any], config_obj) -> Dict[str, Any]:
    """按"显式 CLI 选项 > --config YAML > 内建默认值（已经在 values 里）"
    解析一组物理常量。

    Args:
        ctx: click.Context（当前命令的上下文，用 get_parameter_source 判断
            某个参数是否被用户显式传入）
        values: {参数名: 当前值}，当前值已经是 click 解析后的结果（用户
            显式传入的值，或 click 选项声明的默认值）；参数名必须与该
            command 函数里对应形参同名
        config_obj: load_physical_config_if_given 的返回值，或 None
            （未提供 --config 时不做任何覆盖）

    Returns:
        {参数名: 最终生效值} —— 调用方用这个结果重新赋值对应的局部变量
    """
    if config_obj is None:
        return dict(values)

    from click.core import ParameterSource

    resolved = dict(values)
    for name in values:
        if ctx.get_parameter_source(name) == ParameterSource.DEFAULT and hasattr(config_obj, name):
            resolved[name] = getattr(config_obj, name)
    return resolved


# `order`/`max_iter` 用与 SteadyConfig/TransientConfig 完全相同的字段名，
# resolve_physical_constants 的通用 getattr 匹配对它们同样适用——可以直接
# 把这两个名字加进调用方传的 values 字典里复用同一个函数，不需要专门的
# 解析器。`turbulence_model` 例外：CLI/求解器用的字符串词汇
# （none/sst/ddes/iddes/wmles/les）与 SteadyConfig.turbulence 字段的
# TurbulenceModel 枚举（none/sst_kw/ddes/iddes/wmles/les）命名不完全
# 一致（仅 sst_kw vs sst 一处），必须显式映射，不能靠同名 getattr。
_TURBULENCE_ENUM_TO_CLI_STR = {
    "none": "none",
    "sst_kw": "sst",
    "ddes": "ddes",
    "iddes": "iddes",
    "wmles": "wmles",
    "les": "les",
}


def resolve_turbulence_model(ctx, turbulence_model: str, config_obj) -> str:
    """按"显式 --turbulence-model > --config YAML 的 turbulence 字段 >
    click 声明的默认值"解析湍流模型字符串（配套 resolve_physical_constants，
    但需要额外的枚举值->CLI字符串映射，见模块内 _TURBULENCE_ENUM_TO_CLI_STR
    文档）。

    Args:
        ctx: click.Context
        turbulence_model: 当前值（click 解析后的结果）
        config_obj: load_physical_config_if_given 的返回值，或 None

    Returns:
        最终生效的湍流模型字符串
    """
    if config_obj is None or not hasattr(config_obj, "turbulence"):
        return turbulence_model

    from click.core import ParameterSource

    if ctx.get_parameter_source("turbulence_model") != ParameterSource.DEFAULT:
        return turbulence_model

    config_value = config_obj.turbulence.value
    if config_value not in _TURBULENCE_ENUM_TO_CLI_STR:
        raise ValueError(
            f"--config YAML 里的 turbulence: {config_value} 在配置层可以表示，"
            f"但求解器从未真正实现——只支持 "
            f"{sorted(_TURBULENCE_ENUM_TO_CLI_STR.values())}。"
        )
    return _TURBULENCE_ENUM_TO_CLI_STR[config_value]
