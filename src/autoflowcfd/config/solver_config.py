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
    NONE = "none"       # laminar Navier-Stokes (no turbulence model)
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
        cfl_init: Initial CFL number (recommended: 0.05-0.1 for complex grids)
        cfl_max: Maximum CFL number
        convergence_tol: Convergence tolerance (residual)
        monitor_coefficients: Monitor aerodynamic coefficients during iteration
        growth_rate: Boundary-layer geometric growth rate (surface -> volume mesh)
        max_layers: Maximum boundary-layer + transition layer count
        bl_layers: Optional override for how many of max_layers count as the
            fine boundary-layer stage before switching to the faster-growing
            transition stage (see mesh_extrusion.extrude_layers' own
            bl_layers doc). None (default) keeps the hardcoded
            min(8, max_layers) split.
        min_cell_size: First (near-wall) layer thickness, in meters
        target_cells: Target total cell count (currently only consulted by the
            pure-extrusion volume mesh path; the tetgen-based hybrid path
            ignores it)
        max_cell_size: Optional hard cap (meters) on core-region cell size,
            graded outward from the BL's own near-wall size instead of
            applied uniformly. None leaves the core fill's cell size
            unbounded (only tetgen's own shape-quality bounds apply, so
            cells can grow as large as a coarse far-field input facet, e.g.
            a sparsely-triangulated tunnel/inlet/outlet wall, allows).
        rho_inf: Freestream density (kg/m^3) - single source of truth for
            the initial condition, inlet/farfield boundary conditions, and
            Cd/Cl normalization, so the three always stay consistent.
        vel_inf: Freestream velocity magnitude (m/s), same role as rho_inf.
        p_inf: Freestream static pressure (Pa), same role as rho_inf.
        use_wall_functions: Enable Menter scalable/automatic wall
            treatment (log-law based) on WALL/GROUND boundary faces,
            instead of resolving all the way to the wall. False (default)
            preserves prior behaviour exactly - the resolved-gradient wall
            shear/k/omega treatment, which requires y+~1 at the first
            cell to be accurate. True lets a coarser near-wall mesh
            (y+ up to ~100+) still give physically meaningful skin
            friction and near-wall turbulence, at the cost of the log-law
            model's own equilibrium-boundary-layer assumption being less
            accurate than a resolved gradient in strongly separated flow.
            Default off since this is new, not yet used-in-anger physics -
            opt in explicitly rather than silently changing existing
            fine-mesh cases' results.

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
    cfl_init: float = 0.05  # Conservative default for complex grids (was 1.0)
    cfl_max: float = 10.0
    convergence_tol: float = 1e-3
    monitor_coefficients: bool = True
    growth_rate: float = 1.15
    max_layers: int = 6
    bl_layers: Optional[int] = None
    min_cell_size: float = 0.003
    target_cells: int = 500000
    max_cell_size: Optional[float] = None
    rho_inf: float = 1.225
    vel_inf: float = 30.0
    p_inf: float = 101325.0
    use_wall_functions: bool = False

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

        # Validate volume mesh parameters
        if self.growth_rate <= 1.0:
            raise ValueError(f"growth_rate must be > 1.0, got {self.growth_rate}")
        if self.max_layers < 1:
            raise ValueError(f"max_layers must be positive, got {self.max_layers}")
        if self.min_cell_size <= 0:
            raise ValueError(f"min_cell_size must be positive, got {self.min_cell_size}")
        if self.target_cells < 1:
            raise ValueError(f"target_cells must be positive, got {self.target_cells}")
        if self.max_cell_size is not None:
            if self.max_cell_size <= 0:
                raise ValueError(f"max_cell_size must be positive, got {self.max_cell_size}")
            if self.max_cell_size < self.min_cell_size:
                raise ValueError(
                    f"max_cell_size ({self.max_cell_size}) cannot be smaller than "
                    f"min_cell_size ({self.min_cell_size})"
                )

        # Validate freestream conditions
        if self.rho_inf <= 0:
            raise ValueError(f"rho_inf must be positive, got {self.rho_inf}")
        if self.vel_inf <= 0:
            raise ValueError(f"vel_inf must be positive, got {self.vel_inf}")
        if self.p_inf <= 0:
            raise ValueError(f"p_inf must be positive, got {self.p_inf}")


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
        growth_rate, max_layers, bl_layers, min_cell_size, target_cells,
            max_cell_size: Volume mesh generation parameters, same meaning
            as SteadyConfig.
        rho_inf, vel_inf, p_inf: Freestream conditions, same meaning and
            role as SteadyConfig (single source of truth for the initial
            condition, boundary conditions, and Cd/Cl normalization).
        use_wall_functions: Enable Menter scalable/automatic wall treatment
            on WALL/GROUND faces, same meaning as SteadyConfig.

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
    growth_rate: float = 1.15
    max_layers: int = 6
    bl_layers: Optional[int] = None
    min_cell_size: float = 0.003
    target_cells: int = 500000
    max_cell_size: Optional[float] = None
    rho_inf: float = 1.225
    vel_inf: float = 30.0
    p_inf: float = 101325.0
    use_wall_functions: bool = False

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

        # Validate volume mesh parameters
        if self.growth_rate <= 1.0:
            raise ValueError(f"growth_rate must be > 1.0, got {self.growth_rate}")
        if self.max_layers < 1:
            raise ValueError(f"max_layers must be positive, got {self.max_layers}")
        if self.min_cell_size <= 0:
            raise ValueError(f"min_cell_size must be positive, got {self.min_cell_size}")
        if self.target_cells < 1:
            raise ValueError(f"target_cells must be positive, got {self.target_cells}")
        if self.max_cell_size is not None:
            if self.max_cell_size <= 0:
                raise ValueError(f"max_cell_size must be positive, got {self.max_cell_size}")
            if self.max_cell_size < self.min_cell_size:
                raise ValueError(
                    f"max_cell_size ({self.max_cell_size}) cannot be smaller than "
                    f"min_cell_size ({self.min_cell_size})"
                )

        # Validate freestream conditions
        if self.rho_inf <= 0:
            raise ValueError(f"rho_inf must be positive, got {self.rho_inf}")
        if self.vel_inf <= 0:
            raise ValueError(f"vel_inf must be positive, got {self.vel_inf}")
        if self.p_inf <= 0:
            raise ValueError(f"p_inf must be positive, got {self.p_inf}")
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
