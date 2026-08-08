"""配置管理模块。

处理求解器配置的解析、验证和默认值管理，使用 YAML 文件格式。

核心组件:
    - SolverConfig: 基础配置数据类
    - SteadyConfig: 定常模拟配置
    - TransientConfig: 瞬态模拟配置
    - ConfigLoader: YAML 文件加载器与验证器
    - ConfigSchema: 配置模式定义

示例:
    >>> from autoflowcfd.config import SteadyConfig, ConfigLoader
    >>> config = SteadyConfig(backend="gpu", order=3)
    >>> print(config.backend)
    'gpu'

    >>> loader = ConfigLoader()
    >>> config = loader.load("config.yaml")
"""

from .solver_config import (
    SolverConfig,
    SteadyConfig,
    TransientConfig,
    BackendType,
    TurbulenceModel,
    TimeIntegrationScheme,
)
from .loader import ConfigLoader
from .schema import ConfigSchema, validate_config

__all__ = [
    "SolverConfig",
    "SteadyConfig",
    "TransientConfig",
    "BackendType",
    "TurbulenceModel",
    "TimeIntegrationScheme",
    "ConfigLoader",
    "ConfigSchema",
    "validate_config",
]
