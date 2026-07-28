"""Configuration schema validation.

This module provides schema validation for solver configurations,
ensuring type safety and value range constraints.

Key Components:
    - ConfigSchema: Configuration schema definitions
    - validate_config: Main validation function
    - ValidationError: Custom validation exception

Example:
    >>> from autoflowcfd.config import SteadyConfig, validate_config
    >>> config = SteadyConfig(backend="gpu", order=3)
    >>> errors = validate_config(config)
    >>> if errors:
    ...     print(f"Validation failed: {errors}")
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
    """Configuration validation error."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        """Initialize validation error.
        
        Args:
            message: Error message
            field: Field name that caused the error
        """
        self.field = field
        super().__init__(message)


class ConfigSchema:
    """Configuration schema definitions and validators.
    
    Provides static methods to validate different configuration types
    and ensure all constraints are satisfied.
    
    Example:
        >>> from autoflowcfd.config import SteadyConfig, ConfigSchema
        >>> config = SteadyConfig()
        >>> errors = ConfigSchema.validate_steady(config)
        >>> print(f"Errors: {errors}")
    """
    
    @staticmethod
    def validate_backend(backend: Union[BackendType, str]) -> List[str]:
        """Validate backend type.
        
        Args:
            backend: Backend type to validate
            
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        if isinstance(backend, str):
            try:
                backend = BackendType(backend.lower())
            except ValueError:
                errors.append(
                    f"Invalid backend: '{backend}'. "
                    f"Must be one of: {[b.value for b in BackendType]}"
                )
                return errors
        
        if not isinstance(backend, BackendType):
            errors.append(f"Backend must be BackendType or str, got {type(backend)}")
        
        return errors
    
    @staticmethod
    def validate_order(order: int) -> List[str]:
        """Validate FR discretization order.
        
        Args:
            order: Order value to validate
            
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        if not isinstance(order, int):
            errors.append(f"Order must be integer, got {type(order)}")
            return errors
        
        if order not in [1, 2, 3]:
            errors.append(f"Order must be 1, 2, or 3, got {order}")
        
        return errors
    
    @staticmethod
    def validate_turbulence(turbulence: Union[TurbulenceModel, str]) -> List[str]:
        """Validate turbulence model.
        
        Args:
            turbulence: Turbulence model to validate
            
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        if isinstance(turbulence, str):
            try:
                turbulence = TurbulenceModel(turbulence.lower())
            except ValueError:
                errors.append(
                    f"Invalid turbulence model: '{turbulence}'. "
                    f"Must be one of: {[t.value for t in TurbulenceModel]}"
                )
                return errors
        
        if not isinstance(turbulence, TurbulenceModel):
            errors.append(f"Turbulence must be TurbulenceModel or str, got {type(turbulence)}")
        
        return errors
    
    @staticmethod
    def validate_time_scheme(scheme: Union[TimeIntegrationScheme, str]) -> List[str]:
        """Validate time integration scheme.
        
        Args:
            scheme: Time scheme to validate
            
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        if isinstance(scheme, str):
            try:
                scheme = TimeIntegrationScheme(scheme.lower())
            except ValueError:
                errors.append(
                    f"Invalid time scheme: '{scheme}'. "
                    f"Must be one of: {[s.value for s in TimeIntegrationScheme]}"
                )
                return errors
        
        if not isinstance(scheme, TimeIntegrationScheme):
            errors.append(f"Time scheme must be TimeIntegrationScheme or str, got {type(scheme)}")
        
        return errors
    
    @staticmethod
    def validate_steady(config: SteadyConfig) -> List[str]:
        """Validate steady-state configuration.
        
        Args:
            config: Steady configuration to validate
            
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate base config
        errors.extend(ConfigSchema.validate_solver_base(config))
        
        # Validate steady-specific parameters
        if config.max_iter < 1:
            errors.append(f"max_iter must be >= 1, got {config.max_iter}")
        
        if config.cfl_init <= 0:
            errors.append(f"cfl_init must be > 0, got {config.cfl_init}")
        
        if config.cfl_max <= 0:
            errors.append(f"cfl_max must be > 0, got {config.cfl_max}")
        
        if config.cfl_init > config.cfl_max:
            errors.append(
                f"cfl_init ({config.cfl_init}) cannot exceed cfl_max ({config.cfl_max})"
            )
        
        if config.convergence_tol <= 0:
            errors.append(f"convergence_tol must be > 0, got {config.convergence_tol}")
        
        return errors
    
    @staticmethod
    def validate_transient(config: TransientConfig) -> List[str]:
        """Validate transient configuration.
        
        Args:
            config: Transient configuration to validate
            
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate base config
        errors.extend(ConfigSchema.validate_solver_base(config))
        
        # Validate transient-specific parameters
        if config.dt <= 0:
            errors.append(f"dt must be > 0, got {config.dt}")
        
        if config.total_time <= 0:
            errors.append(f"total_time must be > 0, got {config.total_time}")
        
        if config.warmup_time < 0:
            errors.append(f"warmup_time must be >= 0, got {config.warmup_time}")
        
        if config.warmup_time >= config.total_time:
            errors.append(
                f"warmup_time ({config.warmup_time}) cannot exceed total_time ({config.total_time})"
            )
        
        # Calculate total steps
        total_steps = int(config.total_time / config.dt)
        if total_steps < 1:
            errors.append(
                f"Total steps must be >= 1, got {total_steps} "
                f"(dt={config.dt}, total_time={config.total_time})"
            )
        
        return errors
    
    @staticmethod
    def validate_solver_base(config: SolverConfig) -> List[str]:
        """Validate base solver configuration.
        
        Args:
            config: Base solver configuration to validate
            
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate backend
        errors.extend(ConfigSchema.validate_backend(config.backend))
        
        # Validate order
        errors.extend(ConfigSchema.validate_order(config.order))
        
        # Validate turbulence
        errors.extend(ConfigSchema.validate_turbulence(config.turbulence))
        
        # Validate GPU device
        if config.gpu_device < 0:
            errors.append(f"gpu_device must be >= 0, got {config.gpu_device}")
        
        # Validate CPU threads
        if config.n_threads < -1 or config.n_threads == 0:
            errors.append(f"n_threads must be -1 (auto) or positive, got {config.n_threads}")
        
        # Validate checkpoint interval
        if config.checkpoint_interval < 1:
            errors.append(f"checkpoint_interval must be >= 1, got {config.checkpoint_interval}")
        
        return errors


def validate_config(config: Union[SteadyConfig, TransientConfig]) -> List[str]:
    """Validate solver configuration.
    
    Automatically detects configuration type and validates accordingly.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        List[str]: List of validation errors (empty if valid)
        
    Example:
        >>> from autoflowcfd.config import SteadyConfig, validate_config
        >>> config = SteadyConfig(backend="gpu", order=3)
        >>> errors = validate_config(config)
        >>> if errors:
        ...     for error in errors:
        ...         print(f"Error: {error}")
        ... else:
        ...     print("Configuration is valid")
    """
    if isinstance(config, SteadyConfig):
        return ConfigSchema.validate_steady(config)
    elif isinstance(config, TransientConfig):
        return ConfigSchema.validate_transient(config)
    else:
        return [f"Unknown configuration type: {type(config)}"]


def assert_valid_config(config: Union[SteadyConfig, TransientConfig]):
    """Assert that configuration is valid, raise exception if not.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValidationError: If configuration is invalid
        
    Example:
        >>> from autoflowcfd.config import SteadyConfig, assert_valid_config
        >>> config = SteadyConfig()
        >>> assert_valid_config(config)  # Raises if invalid
    """
    errors = validate_config(config)
    if errors:
        error_msg = "\n".join([f"  - {e}" for e in errors])
        raise ValidationError(
            f"Configuration validation failed:\n{error_msg}"
        )
