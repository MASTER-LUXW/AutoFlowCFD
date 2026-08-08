"""配置模式验证。

本模块提供求解器配置的模式验证功能，确保类型安全和值范围约束。

主要组件：
    - ConfigSchema: 配置模式定义
    - validate_config: 主验证函数
    - ValidationError: 自定义验证异常

示例：
    >>> from autoflowcfd.config import SteadyConfig, validate_config
    >>> config = SteadyConfig(backend="gpu", order=3)
    >>> errors = validate_config(config)
    >>> if errors:
    ...     print(f"验证失败: {errors}")
"""

from typing import List, Optional, Union
from dataclasses import is_dataclass

from .solver_config import (
    SolverConfig,
    SteadyConfig,
    TransientConfig,
    BackendType,
    TurbulenceModel,
    TimeIntegrationScheme,
)


class ValidationError(Exception):
    """配置验证错误。"""

    def __init__(self, message: str, field: Optional[str] = None):
        """初始化验证错误。

        Args:
            message: 错误消息
            field: 导致错误的字段名
        """
        self.field = field
        super().__init__(message)


class ConfigSchema:
    """配置模式定义和验证器。

    提供静态方法验证不同配置类型，确保所有约束都得到满足。

    示例：
        >>> from autoflowcfd.config import SteadyConfig, ConfigSchema
        >>> config = SteadyConfig()
        >>> errors = ConfigSchema.validate_steady(config)
        >>> print(f"错误: {errors}")
    """

    @staticmethod
    def validate_backend(backend: Union[BackendType, str]) -> List[str]:
        """验证后端类型。

        Args:
            backend: 待验证的后端类型

        Returns:
            List[str]: 验证错误列表（有效时为空）
        """
        errors = []
        
        if isinstance(backend, str):
            try:
                backend = BackendType(backend.lower())
            except ValueError:
                errors.append(
                    f"无效的后端: '{backend}'。"
                    f"必须是以下之一: {[b.value for b in BackendType]}"
                )
                return errors
        
        if not isinstance(backend, BackendType):
            errors.append(f"后端必须是 BackendType 或 str，得到 {type(backend)}")
        
        return errors
    
    @staticmethod
    def validate_order(order: int) -> List[str]:
        """验证 FR 离散化阶数。

        Args:
            order: 待验证的阶数值

        Returns:
            List[str]: 验证错误列表（有效时为空）
        """
        errors = []
        
        if not isinstance(order, int):
            errors.append(f"阶数必须是整数，得到 {type(order)}")
            return errors
        
        if order not in [1, 2, 3]:
            errors.append(f"阶数必须是 1, 2, 或 3，得到 {order}")
        
        return errors
    
    @staticmethod
    def validate_turbulence(turbulence: Union[TurbulenceModel, str]) -> List[str]:
        """验证湍流模型。

        Args:
            turbulence: 待验证的湍流模型

        Returns:
            List[str]: 验证错误列表（有效时为空）
        """
        errors = []
        
        if isinstance(turbulence, str):
            try:
                turbulence = TurbulenceModel(turbulence.lower())
            except ValueError:
                errors.append(
                    f"无效的湍流模型: '{turbulence}'。"
                    f"必须是以下之一: {[t.value for t in TurbulenceModel]}"
                )
                return errors
        
        if not isinstance(turbulence, TurbulenceModel):
            errors.append(f"湍流模型必须是 TurbulenceModel 或 str，得到 {type(turbulence)}")
        
        return errors
    
    @staticmethod
    def validate_time_scheme(scheme: Union[TimeIntegrationScheme, str]) -> List[str]:
        """验证时间积分格式。

        Args:
            scheme: 待验证的时间格式

        Returns:
            List[str]: 验证错误列表（有效时为空）
        """
        errors = []
        
        if isinstance(scheme, str):
            try:
                scheme = TimeIntegrationScheme(scheme.lower())
            except ValueError:
                errors.append(
                    f"无效的时间格式: '{scheme}'。"
                    f"必须是以下之一: {[s.value for s in TimeIntegrationScheme]}"
                )
                return errors
        
        if not isinstance(scheme, TimeIntegrationScheme):
            errors.append(f"时间格式必须是 TimeIntegrationScheme 或 str，得到 {type(scheme)}")
        
        return errors
    
    @staticmethod
    def validate_steady(config: SteadyConfig) -> List[str]:
        """验证稳态配置。

        Args:
            config: 待验证的稳态配置

        Returns:
            List[str]: 验证错误列表（有效时为空）
        """
        errors = []
        
        # 验证基础配置
        errors.extend(ConfigSchema.validate_solver_base(config))
        
        # 验证稳态特定参数
        if config.max_iter < 1:
            errors.append(f"max_iter 必须 >= 1，得到 {config.max_iter}")
        
        if config.cfl_init <= 0:
            errors.append(f"cfl_init 必须 > 0，得到 {config.cfl_init}")
        
        if config.cfl_max <= 0:
            errors.append(f"cfl_max 必须 > 0，得到 {config.cfl_max}")
        
        if config.cfl_init > config.cfl_max:
            errors.append(
                f"cfl_init ({config.cfl_init}) 不能超过 cfl_max ({config.cfl_max})"
            )
        
        if config.convergence_tol <= 0:
            errors.append(f"convergence_tol 必须 > 0，得到 {config.convergence_tol}")
        
        return errors
    
    @staticmethod
    def validate_transient(config: TransientConfig) -> List[str]:
        """验证瞬态配置。

        Args:
            config: 待验证的瞬态配置

        Returns:
            List[str]: 验证错误列表（有效时为空）
        """
        errors = []
        
        # 验证基础配置
        errors.extend(ConfigSchema.validate_solver_base(config))
        
        # 验证瞬态特定参数
        if config.dt <= 0:
            errors.append(f"dt 必须 > 0，得到 {config.dt}")
        
        if config.total_time <= 0:
            errors.append(f"total_time 必须 > 0，得到 {config.total_time}")
        
        if config.warmup_time < 0:
            errors.append(f"warmup_time 必须 >= 0，得到 {config.warmup_time}")
        
        if config.warmup_time >= config.total_time:
            errors.append(
                f"warmup_time ({config.warmup_time}) 不能超过 total_time ({config.total_time})"
            )
        
        # 计算总步数
        total_steps = int(config.total_time / config.dt)
        if total_steps < 1:
            errors.append(
                f"总步数必须 >= 1，得到 {total_steps} "
                f"(dt={config.dt}, total_time={config.total_time})"
            )
        
        return errors
    
    @staticmethod
    def validate_solver_base(config: SolverConfig) -> List[str]:
        """验证基础求解器配置。

        Args:
            config: 待验证的基础求解器配置

        Returns:
            List[str]: 验证错误列表（有效时为空）
        """
        errors = []
        
        # 验证后端
        errors.extend(ConfigSchema.validate_backend(config.backend))
        
        # 验证阶数
        errors.extend(ConfigSchema.validate_order(config.order))
        
        # 验证湍流模型
        errors.extend(ConfigSchema.validate_turbulence(config.turbulence))
        
        # 验证 GPU 设备
        if config.gpu_device < 0:
            errors.append(f"gpu_device 必须 >= 0，得到 {config.gpu_device}")
        
        # 验证 CPU 线程数
        if config.n_threads < -1 or config.n_threads == 0:
            errors.append(f"n_threads 必须是 -1（自动）或正数，得到 {config.n_threads}")
        
        # 验证检查点间隔
        if config.checkpoint_interval < 1:
            errors.append(f"checkpoint_interval 必须 >= 1，得到 {config.checkpoint_interval}")
        
        return errors


def validate_config(config: Union[SteadyConfig, TransientConfig]) -> List[str]:
    """验证求解器配置。

    自动检测配置类型并相应验证。

    Args:
        config: 待验证的配置对象

    Returns:
        List[str]: 验证错误列表（有效时为空）

    示例：
        >>> from autoflowcfd.config import SteadyConfig, validate_config
        >>> config = SteadyConfig(backend="gpu", order=3)
        >>> errors = validate_config(config)
        >>> if errors:
        ...     for error in errors:
        ...         print(f"错误: {error}")
        ... else:
        ...     print("配置有效")
    """
    if isinstance(config, SteadyConfig):
        return ConfigSchema.validate_steady(config)
    elif isinstance(config, TransientConfig):
        return ConfigSchema.validate_transient(config)
    else:
        return [f"未知的配置类型: {type(config)}"]


def assert_valid_config(config: Union[SteadyConfig, TransientConfig]):
    """断言配置有效，无效时抛出异常。

    Args:
        config: 待验证的配置对象

    Raises:
        ValidationError: 配置无效时抛出

    示例：
        >>> from autoflowcfd.config import SteadyConfig, assert_valid_config
        >>> config = SteadyConfig()
        >>> assert_valid_config(config)  # 无效时抛出异常
    """
    errors = validate_config(config)
    if errors:
        error_msg = "\n".join([f"  - {e}" for e in errors])
        raise ValidationError(
            f"配置验证失败:\n{error_msg}"
        )
