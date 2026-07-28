"""Configuration management module.

This module handles solver configuration parsing, validation, and
default value management using YAML files.

Key Components:
    - SolverConfig: Base configuration dataclass
    - SteadyConfig: Steady-state simulation config
    - TransientConfig: Transient simulation config
    - ConfigLoader: YAML file loader and validator
    - ConfigSchema: Configuration schema definitions

Example:
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
