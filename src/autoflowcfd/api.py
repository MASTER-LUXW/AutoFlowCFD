"""Python API for AutoFlowCFD.

This module provides the main Python API for programmatic access to
AutoFlowCFD functionality, including grid loading, solver execution,
and post-processing.

Key Components:
    - AutoFlowCFDAPI: Main API class
    - Grid operations: parse, validate, info
    - Solver operations: run steady/transient simulations
    - Post-processing: coefficients, export, analysis

Example:
    >>> from autoflowcfd import AutoFlowCFDAPI
    >>> api = AutoFlowCFDAPI()
    >>> 
    >>> # Parse grid
    >>> grid_data = api.load_grid("sedan.nas")
    >>> 
    >>> # Run simulation
    >>> result = api.run_steady(grid_data, backend="gpu", order=3)
    >>> 
    >>> # Calculate coefficients
    >>> coeffs = api.calculate_coefficients(result)
    >>> print(f"Cd = {coeffs['Cd']:.4f}")
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
from loguru import logger

from autoflowcfd.grid import NASParser, GridValidator, GridData
from autoflowcfd.config import (
    ConfigLoader,
    SteadyConfig,
    TransientConfig,
    BackendType,
    TurbulenceModel,
    TimeIntegrationScheme,
)
from autoflowcfd.boundary import BoundaryManager
from autoflowcfd.core import FRSolver, TransientSolver


class AutoFlowCFDAPI:
    """Main API class for AutoFlowCFD.
    
    Provides high-level interface for grid processing, simulation,
    and post-processing operations.
    
    Attributes:
        verbose: Enable verbose logging
        _config_loader: Configuration loader instance
        
    Example:
        >>> api = AutoFlowCFDAPI(verbose=True)
        >>> grid = api.load_grid("model.nas")
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize API.
        
        Args:
            verbose: Enable verbose logging
            
        Example:
            >>> api = AutoFlowCFDAPI(verbose=True)
        """
        self.verbose = verbose
        self._config_loader = ConfigLoader()
        
        if verbose:
            logger.remove()
            logger.add(lambda msg: print(msg), level="DEBUG")
        
        logger.info("AutoFlowCFD API initialized")
    
    # ========================================================================
    # Grid Operations
    # ========================================================================
    
    def load_grid(
        self,
        grid_file: Union[str, Path],
        encoding: str = "UTF-8",
        validate: bool = True
    ) -> GridData:
        """Load and parse grid file.
        
        Args:
            grid_file: Path to .nas grid file
            encoding: File encoding
            validate: Whether to validate grid quality
            
        Returns:
            GridData: Parsed grid data object
            
        Raises:
            FileNotFoundError: Grid file not found
            ValueError: Invalid grid format
            
        Example:
            >>> grid = api.load_grid("sedan.nas")
            >>> print(f"Nodes: {grid.get_node_count()}")
        """
        logger.info(f"Loading grid: {grid_file}")
        
        parser = NASParser(str(grid_file), encoding=encoding)
        grid_data = parser.parse()
        
        if validate:
            logger.info("Validating grid quality...")
            validator = GridValidator(grid_data)
            quality_report = validator.validate()

            if not quality_report['passed']:
                logger.warning(
                    "Grid quality validation failed. "
                    "See quality report for details."
                )
        
        logger.info(
            f"Grid loaded: {grid_data.node_count} nodes, "
            f"{grid_data.cell_count} cells"
        )
        
        return grid_data
    
    def get_grid_info(self, grid_data: GridData) -> Dict[str, Any]:
        """Get grid information and statistics.
        
        Args:
            grid_data: Grid data object
            
        Returns:
            Dict[str, Any]: Grid statistics dictionary
            
        Example:
            >>> info = api.get_grid_info(grid)
            >>> print(info['node_count'])
        """
        info = {
            "node_count": grid_data.node_count,
            "cell_count": grid_data.cell_count,
        }
        
        if hasattr(grid_data, 'boundaries'):
            info["boundary_groups"] = {}
            for name in grid_data.boundaries.boundary_names:
                cells = grid_data.boundaries.get_boundary_cells(name)
                info["boundary_groups"][name] = len(cells)
        
        return info
    
    def validate_grid(self, grid_data: GridData) -> Dict[str, Any]:
        """Validate grid quality.
        
        Args:
            grid_data: Grid data object
            
        Returns:
            Dict[str, Any]: Quality report dictionary
            
        Example:
            >>> report = api.validate_grid(grid)
            >>> if report['passed']:
            ...     print("Grid is valid")
        """
        validator = GridValidator(grid_data)
        return validator.validate()
    
    # ========================================================================
    # Solver Operations
    # ========================================================================
    
    def run_steady(
        self,
        grid_data: GridData,
        backend: str = "auto",
        order: int = 3,
        turbulence: str = "sst_kw",
        max_iter: int = 5000,
        output_dir: str = "./results",
        **kwargs
    ) -> Any:
        """Run steady-state RANS simulation.
        
        Args:
            grid_data: Grid data object
            backend: Compute backend (cpu/gpu/auto)
            order: FR discretization order (1/2/3)
            turbulence: Turbulence model (sst_kw/sa)
            max_iter: Maximum iterations
            output_dir: Output directory
            **kwargs: Additional configuration parameters
            
        Returns:
            SolverResult: Simulation result object
            
        Raises:
            ValueError: Invalid parameters
            
        Example:
            >>> result = api.run_steady(
            ...     grid,
            ...     backend="gpu",
            ...     order=3,
            ...     max_iter=3000
            ... )
            >>> print(f"Converged: {result.converged}")
        """
        logger.info("Starting steady-state simulation")
        
        # Create configuration
        config = SteadyConfig(
            backend=BackendType(backend),
            order=order,
            turbulence=TurbulenceModel(turbulence),
            max_iter=max_iter,
            output_dir=output_dir,
            **kwargs
        )
        
        # Create boundary manager if boundaries exist
        if hasattr(grid_data, 'boundaries'):
            bc_manager = BoundaryManager(grid_data.boundaries)
            # TODO: Add default boundary conditions
        else:
            bc_manager = None
        
        # Create and run solver
        solver = FRSolver(grid_data, config)
        result = solver.solve()
        
        logger.info(
            f"Simulation complete: {result.iterations} iterations, "
            f"converged={result.converged}"
        )
        
        return result
    
    def run_transient(
        self,
        grid_data: GridData,
        backend: str = "auto",
        order: int = 3,
        mode: str = "des",
        time_integration: str = "backward_euler",
        physical_time: float = 0.3,
        dt: float = 1e-4,
        output_dir: str = "./transient_results",
        init_from: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Run transient LES/DES simulation.
        
        Args:
            grid_data: Grid data object
            backend: Compute backend
            order: FR order
            mode: Turbulence mode (des/ddes/les)
            time_integration: Time integration scheme
            physical_time: Total physical time (seconds)
            dt: Time step size
            output_dir: Output directory
            init_from: Initialize from checkpoint file
            **kwargs: Additional configuration parameters
            
        Returns:
            TransientResult: Transient simulation result
            
        Example:
            >>> result = api.run_transient(
            ...     grid,
            ...     mode="ddes",
            ...     physical_time=0.3,
            ...     dt=1e-4
            ... )
        """
        logger.info(f"Starting transient simulation: t={physical_time}s")
        
        # Map mode to turbulence model
        turbulence_map = {
            'des': TurbulenceModel.DES,
            'ddes': TurbulenceModel.DDES,
            'les': TurbulenceModel.LES,
        }
        
        # Create configuration
        config = TransientConfig(
            backend=BackendType(backend),
            order=order,
            turbulence=turbulence_map[mode],
            time_scheme=TimeIntegrationScheme(time_integration),
            dt=dt,
            total_time=physical_time,
            output_dir=output_dir,
            **kwargs
        )
        
        # Create solver
        solver = TransientSolver(grid_data, config)
        
        # Initialize from checkpoint if specified
        if init_from:
            logger.info(f"Initializing from checkpoint: {init_from}")
            solver.load_checkpoint(init_from)
        
        # Run simulation
        result = solver.solve()
        
        logger.info(
            f"Transient simulation complete: {result.time_steps} steps, "
            f"physical time={result.physical_time:.6f}s"
        )
        
        return result
    
    def resume_simulation(
        self,
        checkpoint_file: Union[str, Path],
        grid_data = None,
        config = None,
        max_iter: int = 5000,
        output_dir: Optional[str] = None,
        backend: Optional[str] = None
    ) -> Any:
        """Resume simulation from checkpoint.
        
        Args:
            checkpoint_file: Path to checkpoint file
            grid_data: Grid data (required for resume)
            config: Solver configuration (optional, will load from checkpoint if not provided)
            max_iter: Additional iterations to run
            output_dir: Override output directory
            backend: Backend override ("cpu" or "gpu")
            
        Returns:
            SteadyResult: Simulation result
            
        Raises:
            FileNotFoundError: Checkpoint file not found
            ValueError: Missing required parameters
            
        Example:
            >>> # Basic resume
            >>> result = api.resume_simulation("results/checkpoint.h5", grid_data)
            >>> 
            >>> # Resume with more iterations and GPU
            >>> result = api.resume_simulation(
            ...     "checkpoint.h5", 
            ...     grid_data, 
            ...     max_iter=2000,
            ...     backend="gpu"
            ... )
        """
        from pathlib import Path
        from .core.checkpoint import CheckpointManager
        
        logger.info(f"Resuming from checkpoint: {checkpoint_file}")
        
        # Validate inputs
        checkpoint_path = Path(checkpoint_file)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        if grid_data is None:
            raise ValueError(
                "grid_data is required for resume operation. "
                "Please provide the grid data used in the original simulation."
            )
        
        if config is None:
            logger.warning(
                "No config provided. Using default configuration. "
                "This may cause issues if checkpoint was created with different settings."
            )
            # TODO: Load config from checkpoint or use defaults
            from .config.solver_config import SteadyConfig
            config = SteadyConfig()
        
        # Override output directory if specified
        if output_dir:
            config.output_dir = output_dir
        
        # Override backend if specified
        if backend:
            config.backend.value = backend
        
        try:
            # Create solver
            from .core.solver_steady import FRSolver
            solver = FRSolver(grid_data, config)
            
            # Load checkpoint
            solution, history, iteration, metadata = solver.checkpoint_manager.load(
                checkpoint_path,
                target_backend=backend
            )
            
            # Set initial solution
            solver.solution = solution
            
            logger.info(f"Resumed from iteration {iteration}")
            logger.info(f"Running additional {max_iter} iterations...")
            
            # Continue solving
            result = solver.solve(max_iter=max_iter + iteration)
            
            return result
            
        except Exception as e:
            logger.error(f"Resume simulation failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to resume simulation: {e}")

    # ========================================================================
    # Post-processing Operations
    # ========================================================================
    
    def calculate_coefficients(
        self,
        result: Any,
        reference_area: float = 2.2,
        reference_length: float = 4.5,
        density: float = 1.225,
        velocity: float = 30.0
    ) -> Dict[str, float]:
        """Calculate aerodynamic coefficients.
        
        Args:
            result: Solver result object
            reference_area: Reference area (m²)
            reference_length: Reference length (m)
            density: Air density (kg/m³)
            velocity: Free-stream velocity (m/s)
            
        Returns:
            Dict[str, float]: Aerodynamic coefficients
            
        Example:
            >>> coeffs = api.calculate_coefficients(result)
            >>> print(f"Cd = {coeffs['Cd']:.4f}")
        """
        # TODO: Implement coefficient calculation
        logger.warning("Coefficient calculation not yet implemented")
        
        return {
            "Cd": 0.0,
            "Cl": 0.0,
            "Cm": 0.0,
            "Cs": 0.0,
        }
    
    def export_vtk(
        self,
        result: Any,
        output_file: Union[str, Path],
        variables: Optional[list] = None
    ) -> None:
        """Export results to VTK format.
        
        Args:
            result: Solver result object
            output_file: Output VTK file path
            variables: Variables to export (None for all)
            
        Example:
            >>> api.export_vtk(result, "output.vtk")
        """
        # TODO: Implement VTK export
        logger.warning("VTK export not yet implemented")
        raise NotImplementedError("VTK export not yet implemented")
    
    def get_convergence_history(self, result: Any) -> Dict[str, list]:
        """Get convergence history.
        
        Args:
            result: Solver result object
            
        Returns:
            Dict[str, list]: Convergence data
            
        Example:
            >>> history = api.get_convergence_history(result)
            >>> iterations = history['iterations']
            >>> residuals = history['residuals']
        """
        # TODO: Implement convergence history extraction
        logger.warning("Convergence history extraction not yet implemented")
        
        return {
            "iterations": [],
            "residuals": [],
        }
    
    # ========================================================================
    # Configuration Management
    # ========================================================================
    
    def load_config(self, config_file: Union[str, Path]) -> Any:
        """Load configuration from YAML file.
        
        Args:
            config_file: Path to YAML configuration file
            
        Returns:
            SteadyConfig or TransientConfig: Configuration object
            
        Example:
            >>> config = api.load_config("simulation.yaml")
        """
        return self._config_loader.load(str(config_file))
    
    def create_steady_config(
        self,
        backend: str = "auto",
        order: int = 3,
        turbulence: str = "sst_kw",
        **kwargs
    ) -> SteadyConfig:
        """Create steady-state configuration.
        
        Args:
            backend: Compute backend
            order: FR order
            turbulence: Turbulence model
            **kwargs: Additional parameters
            
        Returns:
            SteadyConfig: Configuration object
        """
        return SteadyConfig(
            backend=BackendType(backend),
            order=order,
            turbulence=TurbulenceModel(turbulence),
            **kwargs
        )
    
    def create_transient_config(
        self,
        backend: str = "auto",
        order: int = 3,
        mode: str = "des",
        dt: float = 1e-4,
        total_time: float = 0.3,
        **kwargs
    ) -> TransientConfig:
        """Create transient configuration.
        
        Args:
            backend: Compute backend
            order: FR order
            mode: Turbulence mode
            dt: Time step size
            total_time: Total physical time
            **kwargs: Additional parameters
            
        Returns:
            TransientConfig: Configuration object
        """
        turbulence_map = {
            'des': TurbulenceModel.DES,
            'ddes': TurbulenceModel.DDES,
            'les': TurbulenceModel.LES,
        }
        
        return TransientConfig(
            backend=BackendType(backend),
            order=order,
            turbulence=turbulence_map[mode],
            dt=dt,
            total_time=total_time,
            **kwargs
        )
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_version(self) -> str:
        """Get AutoFlowCFD version.
        
        Returns:
            str: Version string
        """
        from . import __version__
        return __version__
    
    def check_environment(self) -> Dict[str, Any]:
        """Check system environment.
        
        Returns:
            Dict[str, Any]: Environment information
        """
        import sys
        import platform
        
        info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "autoflowcfd_version": self.get_version(),
        }
        
        # Check GPU availability
        try:
            import cupy as cp
            test_array = cp.array([1, 2, 3])
            info["gpu_available"] = True
        except Exception:
            info["gpu_available"] = False
        
        return info
