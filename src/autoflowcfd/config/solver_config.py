"""Solver configuration dataclasses.

This module defines the core configuration structures for AutoFlowCFD solver,
including steady-state and transient simulation configurations.

Key Components:
    - BackendType: Compute backend enumeration (cpu/gpu/auto)
    - TurbulenceModel: Turbulence model enumeration
    - TimeIntegrationScheme: Time integration scheme enumeration
    - SolverConfig: Base solver configuration
    - SteadyConfig: Steady-state specific configuration
    - TransientConfig: Transient specific configuration

Example:
    >>> from autoflowcfd.config import SteadyConfig, TransientConfig
    >>> steady = SteadyConfig(backend="gpu", order=3, max_iter=5000)
    >>> transient = TransientConfig(dt=1e-4, total_time=0.3)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Literal
import os


class BackendType(str, Enum):
    """Compute backend type enumeration."""
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


class TurbulenceModel(str, Enum):
    """Turbulence model enumeration."""
    SST_KW = "sst_kw"
    SA = "sa"
    DES = "des"
    DDES = "ddes"
    LES = "les"


class TimeIntegrationScheme(str, Enum):
    """Time integration scheme enumeration."""
    BACKWARD_EULER = "backward_euler"
    RK2 = "rk2"
    RK3 = "rk3"
    AB3 = "ab3"  # Adams-Bashforth 3rd order


@dataclass
class SolverConfig:
    """Base solver configuration.
    
    Attributes:
        backend: Compute backend (cpu/gpu/auto)
        order: FR discretization order (1/2/3)
        turbulence: Turbulence model
        gpu_device: GPU device ID (only for GPU mode)
        n_threads: CPU thread count (only for CPU mode, auto=detect)
        output_dir: Output directory path
        checkpoint_interval: Checkpoint save interval (steps)
        verbose: Enable verbose logging
        
    Example:
        >>> config = SolverConfig(backend="gpu", order=3)
        >>> print(config.backend)
        'gpu'
    """
    backend: BackendType = BackendType.AUTO
    order: int = 2
    turbulence: TurbulenceModel = TurbulenceModel.SST_KW
    gpu_device: int = 0
    n_threads: int = -1  # -1 means auto-detect
    output_dir: str = "./results"
    checkpoint_interval: int = 100
    verbose: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate order
        if self.order not in [1, 2, 3]:
            raise ValueError(f"FR order must be 1, 2, or 3, got {self.order}")
        
        # Validate gpu_device
        if self.gpu_device < 0:
            raise ValueError(f"GPU device ID must be non-negative, got {self.gpu_device}")
        
        # Validate n_threads
        if self.n_threads == -1:
            # Auto-detect CPU cores
            import multiprocessing
            self.n_threads = multiprocessing.cpu_count()
        elif self.n_threads < 1:
            raise ValueError(f"Thread count must be positive, got {self.n_threads}")
        
        # Create output directory if not exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    @property
    def is_gpu(self) -> bool:
        """Check if using GPU backend."""
        return self.backend == BackendType.GPU
    
    @property
    def is_cpu(self) -> bool:
        """Check if using CPU backend."""
        return self.backend == BackendType.CPU


@dataclass
class SteadyConfig(SolverConfig):
    """Steady-state simulation configuration.
    
    Inherits all attributes from SolverConfig and adds steady-specific parameters.
    
    Attributes:
        max_iter: Maximum iteration steps
        cfl_init: Initial CFL number
        cfl_max: Maximum CFL number
        convergence_tol: Convergence tolerance (residual)
        monitor_coefficients: Monitor aerodynamic coefficients during iteration
        
    Example:
        >>> config = SteadyConfig(
        ...     backend="gpu",
        ...     order=3,
        ...     max_iter=5000,
        ...     cfl_init=0.1,
        ...     cfl_max=5.0
        ... )
    """
    max_iter: int = 50
    cfl_init: float = 1.0
    cfl_max: float = 10.0
    convergence_tol: float = 1e-3
    monitor_coefficients: bool = True
    
    def __post_init__(self):
        """Validate steady configuration."""
        super().__post_init__()
        
        # Validate iterations
        if self.max_iter < 1:
            raise ValueError(f"Max iterations must be positive, got {self.max_iter}")
        
        # Validate CFL numbers
        if self.cfl_init <= 0:
            raise ValueError(f"Initial CFL must be positive, got {self.cfl_init}")
        if self.cfl_max <= 0:
            raise ValueError(f"Max CFL must be positive, got {self.cfl_max}")
        if self.cfl_init > self.cfl_max:
            raise ValueError(f"Initial CFL ({self.cfl_init}) cannot exceed max CFL ({self.cfl_max})")
        
        # Validate convergence tolerance
        if self.convergence_tol <= 0:
            raise ValueError(f"Convergence tolerance must be positive, got {self.convergence_tol}")


@dataclass
class TransientConfig(SolverConfig):
    """Transient simulation configuration.
    
    Inherits all attributes from SolverConfig and adds transient-specific parameters.
    
    Attributes:
        dt: Time step size (seconds)
        total_time: Total physical time (seconds)
        time_scheme: Time integration scheme
        sample_interval: Data sampling interval (steps)
        warmup_time: Warmup time to skip (seconds, for statistics)
        init_from_checkpoint: Initialize from steady-state checkpoint
        
    Example:
        >>> config = TransientConfig(
        ...     backend="gpu",
        ...     order=3,
        ...     dt=1e-4,
        ...     total_time=0.3,
        ...     time_scheme="backward_euler"
        ... )
    """
    dt: float = 1e-4
    total_time: float = 0.1
    time_scheme: TimeIntegrationScheme = TimeIntegrationScheme.BACKWARD_EULER
    sample_interval: int = 10
    warmup_time: float = 0.05
    init_from_checkpoint: Optional[str] = None
    
    def __post_init__(self):
        """Validate transient configuration."""
        super().__post_init__()
        
        # Validate time step
        if self.dt <= 0:
            raise ValueError(f"Time step must be positive, got {self.dt}")
        
        # Validate total time
        if self.total_time <= 0:
            raise ValueError(f"Total time must be positive, got {self.total_time}")
        
        # Validate warmup time
        if self.warmup_time < 0:
            raise ValueError(f"Warmup time must be non-negative, got {self.warmup_time}")
        if self.warmup_time >= self.total_time:
            raise ValueError(f"Warmup time ({self.warmup_time}) cannot exceed total time ({self.total_time})")
        
        # Calculate total steps
        self.total_steps = int(self.total_time / self.dt)
        if self.total_steps < 1:
            raise ValueError(
                f"Total steps must be at least 1, got {self.total_steps} "
                f"(dt={self.dt}, total_time={self.total_time})"
            )
    
    @property
    def n_steps(self) -> int:
        """Get total number of time steps."""
        return self.total_steps


def create_steady_config(**kwargs) -> SteadyConfig:
    """Factory function to create steady configuration with defaults.
    
    Args:
        **kwargs: Override default values
        
    Returns:
        SteadyConfig: Configured steady-state solver config
        
    Example:
        >>> config = create_steady_config(backend="gpu", max_iter=10000)
    """
    return SteadyConfig(**kwargs)


def create_transient_config(**kwargs) -> TransientConfig:
    """Factory function to create transient configuration with defaults.
    
    Args:
        **kwargs: Override default values
        
    Returns:
        TransientConfig: Configured transient solver config
        
    Example:
        >>> config = create_transient_config(dt=1e-5, total_time=0.5)
    """
    return TransientConfig(**kwargs)
