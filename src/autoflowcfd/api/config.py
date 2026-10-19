"""AutoFlowCFDAPI 的配置入口：构造与加载 `SteadyConfig` / `TransientConfig`。

字段、默认值与校验只在配置类里（`config/solver_config.py`，命令行选项的默认值也读它）。2026-10-19 以前这里另有
一套默认值（稳态 max_iter=1000 而配置类 50、瞬态时间格式默认值三处三样、`total_time` 缺省为 dt*max_iter）与第二个
配置加载器（返回未校验的裸字典，格式不支持或缺 PyYAML 时返回 {} 继续）。
"""

from pathlib import Path
from typing import Union

from autoflowcfd.config.solver_config import SteadyConfig, TransientConfig


class _APIConfigMixin:
    """配置构造与加载"""

    def create_steady_config(self, **fields) -> SteadyConfig:
        """稳态配置（字段名与 YAML 一致，见 docs/CONFIGURATION_GUIDE.md；构造时校验）。"""
        return SteadyConfig(**fields)

    def create_transient_config(self, **fields) -> TransientConfig:
        """瞬态配置（字段名与 YAML 一致，见 docs/CONFIGURATION_GUIDE.md；构造时校验）。"""
        return TransientConfig(**fields)

    def load_config(self, config_file: Union[str, Path]) -> Union[SteadyConfig, TransientConfig]:
        """从 YAML 加载并校验配置（`mode` 决定类型），与 CLI `--config` 同一个加载器。"""
        from autoflowcfd.config.loader import ConfigLoader

        return ConfigLoader().load(config_file)
