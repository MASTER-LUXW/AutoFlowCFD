"""Boundary condition management module.

This module handles boundary condition application, including built-in
types (velocity_inlet, pressure_outlet, wall, symmetry, slip_wall) and custom BC plugins.

Key Components:
    - BoundaryManager: BC application manager with auto/manual/hybrid modes
    - YAMLConfigLoader: YAML configuration file loader
    - BoundaryTypeMapper: Automatic boundary type mapper
    - Built-in BCs: VelocityInletBC, PressureOutletBC, WallBC, SymmetryBC, SlipWallBC
    - Custom BC extension interface via register_boundary_condition decorator

Example:
    >>> from autoflowcfd.boundary import BoundaryManager
    >>> from autoflowcfd.grid import BoundaryMap
    >>> 
    >>> # Create boundary map
    >>> bmap = BoundaryMap()
    >>> bmap.add_boundary("INLET", [0, 1, 2])
    >>> bmap.add_boundary("BODY", list(range(3, 100)))
    >>> 
    >>> # Create manager and configure boundaries
    >>> bc_manager = BoundaryManager(bmap)
    >>> bc_manager.auto_configure()  # Auto mode
    >>> # or
    >>> bc_manager.configure_from_yaml("config.yaml")  # Manual mode
    >>> # or
    >>> bc_manager.hybrid_configure("config.yaml")  # Hybrid mode
    >>> 
    >>> # Apply to solution
    >>> bc_manager.apply_all(solution, time=0.0)
"""

from .conditions import (
    BaseBC,
    InletBC,
    OutletBC,
    WallBC,
    GroundBC,
    FarfieldBC,
    SymmetryBC,
    BodyBC,
    register_boundary_condition,
    get_boundary_condition_class,
    create_boundary_condition,
)
from .outlet_bc import OutletCharacteristicBC, OutletSpongeBC
from .manager import BoundaryManager
from .config import (
    YAMLConfigLoader,
    BoundaryTypeMapper,
    ParameterValidator,
    ConfigurationError,
)

__all__ = [
    # Manager
    "BoundaryManager",
    
    # Configuration
    "YAMLConfigLoader",
    "BoundaryTypeMapper",
    "ParameterValidator",
    "ConfigurationError",
    
    # Base class
    "BaseBC",
    
    # Built-in BCs
    "InletBC",
    "OutletBC",
    "WallBC",
    "GroundBC",
    "FarfieldBC",
    "SymmetryBC",
    "BodyBC",
    
    # Advanced outlet BCs
    "OutletCharacteristicBC",
    "OutletSpongeBC",
    
    # Extension mechanism
    "register_boundary_condition",
    "get_boundary_condition_class",
    "create_boundary_condition",
]
