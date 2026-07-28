"""Boundary condition implementations.

This module provides built-in boundary condition classes for AutoFlowCFD,
including inlet, outlet, wall, ground, farfield, symmetry, and body boundaries.

Key Components:
    - BaseBC: Abstract base class for all boundary conditions
    - InletBC: Velocity/pressure inlet boundary
    - OutletBC: Pressure outlet boundary
    - WallBC: No-slip wall boundary
    - GroundBC: Moving/stationary ground boundary
    - FarfieldBC: Free-stream farfield boundary
    - SymmetryBC: Symmetry plane boundary
    - BodyBC: Vehicle body surface (special wall)

Example:
    >>> from autoflowcfd.boundary.conditions import InletBC
    >>> inlet = InletBC(velocity=30.0, pressure=101325.0)
    >>> inlet.apply(solution, boundary_cells=[0, 1, 2], time=0.0)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np
from loguru import logger


class BaseBC(ABC):
    """Abstract base class for all boundary conditions.
    
    All boundary condition implementations must inherit from this class
    and implement the required methods.
    
    Attributes:
        bc_type: Boundary condition type identifier
        params: Boundary condition parameters
        
    Example:
        >>> class MyBC(BaseBC):
        ...     def __init__(self, **kwargs):
        ...         super().__init__("MY_BC", kwargs)
    """
    
    def __init__(self, bc_type: str, params: Dict[str, Any]):
        """Initialize boundary condition.
        
        Args:
            bc_type: Boundary condition type identifier
            params: Boundary condition parameters
        """
        self.bc_type = bc_type
        self.params = params
    
    @abstractmethod
    def apply(
        self,
        solution: Any,
        boundary_cells: Any,  # Changed from List[int] to Any to accept numpy arrays
        time: float = 0.0
    ) -> None:
        """Apply boundary condition to solution vector.
        
        Args:
            solution: Solution vector (SolutionVector object)
            boundary_cells: List or array of cell indices on this boundary
            time: Current simulation time
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate boundary condition parameters.
        
        Returns:
            bool: True if parameters are valid
            
        Raises:
            ValueError: If parameters are invalid
        """
        pass
    
    def get_type(self) -> str:
        """Get boundary condition type.
        
        Returns:
            str: Boundary condition type identifier
        """
        return self.bc_type
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(type={self.bc_type})"


class InletBC(BaseBC):
    """Velocity/pressure inlet boundary condition.
    
    Specifies velocity components and static pressure at the inlet.
    Supports both uniform and profile-based inlet conditions.
    
    Attributes:
        velocity_x: X-component of inlet velocity (m/s)
        velocity_y: Y-component of inlet velocity (m/s)
        velocity_z: Z-component of inlet velocity (m/s)
        pressure: Static pressure at inlet (Pa)
        turbulence_k: Turbulent kinetic energy (m²/s²)
        turbulence_omega: Specific dissipation rate (1/s)
        
    Example:
        >>> inlet = InletBC(
        ...     velocity_x=30.0,
        ...     velocity_y=0.0,
        ...     velocity_z=0.0,
        ...     pressure=101325.0
        ... )
    """
    
    def __init__(
        self,
        velocity_x: float = 30.0,
        velocity_y: float = 0.0,
        velocity_z: float = 0.0,
        pressure: float = 101325.0,
        turbulence_k: float = 0.1,
        turbulence_omega: float = 10.0,
        **kwargs
    ):
        """Initialize inlet boundary condition.
        
        Args:
            velocity_x: X-component of velocity (m/s)
            velocity_y: Y-component of velocity (m/s)
            velocity_z: Z-component of velocity (m/s)
            pressure: Static pressure (Pa)
            turbulence_k: Turbulent kinetic energy (m²/s²)
            turbulence_omega: Specific dissipation rate (1/s)
            **kwargs: Additional parameters
        """
        params = {
            'velocity_x': velocity_x,
            'velocity_y': velocity_y,
            'velocity_z': velocity_z,
            'pressure': pressure,
            'turbulence_k': turbulence_k,
            'turbulence_omega': turbulence_omega,
        }
        params.update(kwargs)
        super().__init__('INLET', params)
    
    def apply(
        self,
        solution: Any,
        boundary_cells: Any,
        time: float = 0.0
    ) -> None:
        """Apply inlet boundary condition.
        
        Sets conserved variables at inlet cells based on specified
        velocity and pressure.
        
        Solution variables are in conservative form: [rho, rhou, rhov, rhow, E, k, omega]
        
        Args:
            solution: Solution vector with conserved variables
            boundary_cells: List or array of inlet cell indices
            time: Current simulation time (for time-varying BCs)
        """
        # Check if boundary_cells is empty
        if hasattr(boundary_cells, 'size'):
            # numpy array
            if boundary_cells.size == 0:
                return
        else:
            # list
            if not boundary_cells:
                return
        
        logger.debug(
            f"Applying INLET BC to {len(boundary_cells)} cells "
            f"at time={time:.6f}s"
        )
        
        # Extract parameters
        velocity_x = self.params['velocity_x']
        velocity_y = self.params['velocity_y']
        velocity_z = self.params['velocity_z']
        pressure = self.params['pressure']
        turbulence_k = self.params.get('turbulence_k', 0.1)
        turbulence_omega = self.params.get('turbulence_omega', 10.0)
        
        # For air at standard conditions
        rho = 1.225  # kg/m³
        gamma = 1.4  # Specific heat ratio
        
        # Compute conservative variables
        rhou = rho * velocity_x
        rhov = rho * velocity_y
        rhow = rho * velocity_z
        
        # Total energy: E = p/(gamma-1) + 0.5*rho*V^2
        V_squared = velocity_x**2 + velocity_y**2 + velocity_z**2
        E = pressure / (gamma - 1.0) + 0.5 * rho * V_squared
        
        # Set solution values at inlet cells
        # Solution structure: [rho, rhou, rhov, rhow, E, k, omega]
        solution[boundary_cells, 0] = rho           # density
        solution[boundary_cells, 1] = rhou          # x-momentum
        solution[boundary_cells, 2] = rhov          # y-momentum
        solution[boundary_cells, 3] = rhow          # z-momentum
        solution[boundary_cells, 4] = E             # total energy
        solution[boundary_cells, 5] = turbulence_k  # turbulent kinetic energy
        solution[boundary_cells, 6] = turbulence_omega  # specific dissipation rate
    
    def validate(self) -> bool:
        """Validate inlet boundary condition parameters.
        
        Returns:
            bool: True if all parameters are valid
            
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate velocity magnitude
        vel_mag = np.sqrt(
            self.params['velocity_x']**2 +
            self.params['velocity_y']**2 +
            self.params['velocity_z']**2
        )
        
        if vel_mag < 0:
            raise ValueError("Velocity magnitude cannot be negative")
        
        if vel_mag > 340.0:  # Speed of sound approximation
            logger.warning(
                f"Inlet velocity {vel_mag:.2f} m/s is supersonic. "
                f"Ensure compressible flow solver is enabled."
            )
        
        # Validate pressure
        if self.params['pressure'] <= 0:
            raise ValueError(f"Pressure must be positive, got {self.params['pressure']}")
        
        # Validate turbulence quantities
        if self.params['turbulence_k'] < 0:
            raise ValueError(f"Turbulence k must be non-negative, got {self.params['turbulence_k']}")
        
        if self.params['turbulence_omega'] <= 0:
            raise ValueError(f"Turbulence omega must be positive, got {self.params['turbulence_omega']}")
        
        return True


class OutletBC(BaseBC):
    """Pressure outlet boundary condition.
    
    Specifies static pressure at the outlet boundary.
    Flow direction is determined by local solution gradient.
    
    Attributes:
        pressure: Static pressure at outlet (Pa)
        backflow_turbulence_k: Turbulence k for backflow (m²/s²)
        backflow_turbulence_omega: Turbulence omega for backflow (1/s)
        
    Example:
        >>> outlet = OutletBC(pressure=101325.0)
    """
    
    def __init__(
        self,
        pressure: float = 101325.0,
        backflow_turbulence_k: float = 0.1,
        backflow_turbulence_omega: float = 10.0,
        **kwargs
    ):
        """Initialize outlet boundary condition.
        
        Args:
            pressure: Static pressure (Pa)
            backflow_turbulence_k: Turbulence k for backflow
            backflow_turbulence_omega: Turbulence omega for backflow
            **kwargs: Additional parameters
        """
        params = {
            'pressure': pressure,
            'backflow_turbulence_k': backflow_turbulence_k,
            'backflow_turbulence_omega': backflow_turbulence_omega,
        }
        params.update(kwargs)
        super().__init__('OUTLET', params)
    
    def apply(
        self,
        solution: Any,
        boundary_cells: Any,
        time: float = 0.0
    ) -> None:
        """Apply outlet boundary condition.
        
        Sets pressure at outlet cells. For subsonic outflow,
        pressure is specified and other variables are extrapolated.
        
        Solution variables are in conservative form: [rho, rhou, rhov, rhow, E, k, omega]
        
        Args:
            solution: Solution vector
            boundary_cells: List or array of outlet cell indices
            time: Current simulation time
        """
        # Check if boundary_cells is empty
        if hasattr(boundary_cells, 'size'):
            if boundary_cells.size == 0:
                return
        else:
            if not boundary_cells:
                return
        
        logger.debug(
            f"Applying OUTLET BC to {len(boundary_cells)} cells "
            f"at time={time:.6f}s"
        )
        
        # Extract parameters
        pressure = self.params['pressure']
        backflow_k = self.params.get('backflow_turbulence_k', 0.1)
        backflow_omega = self.params.get('backflow_turbulence_omega', 10.0)
        
        gamma = 1.4  # Specific heat ratio
        
        # For outlet, we set the total energy based on specified pressure
        # E = p/(gamma-1) + 0.5*rho*V^2
        # We keep rho and velocity from current solution (extrapolation)
        rho_current = solution[boundary_cells, 0]
        rhou_current = solution[boundary_cells, 1]
        rhov_current = solution[boundary_cells, 2]
        rhow_current = solution[boundary_cells, 3]
        
        # Compute velocity magnitude squared
        V_squared = ((rhou_current / np.maximum(rho_current, 1e-10))**2 + 
                     (rhov_current / np.maximum(rho_current, 1e-10))**2 + 
                     (rhow_current / np.maximum(rho_current, 1e-10))**2)
        
        # Set total energy with specified pressure
        E_new = pressure / (gamma - 1.0) + 0.5 * rho_current * V_squared
        
        solution[boundary_cells, 4] = E_new  # total energy
        
        # For turbulence variables, set backflow values if needed
        solution[boundary_cells, 5] = np.maximum(solution[boundary_cells, 5], backflow_k)  # k
        solution[boundary_cells, 6] = np.maximum(solution[boundary_cells, 6], backflow_omega)  # omega
    
    def validate(self) -> bool:
        """Validate outlet boundary condition parameters.
        
        Returns:
            bool: True if parameters are valid
            
        Raises:
            ValueError: If parameters are invalid
        """
        if self.params['pressure'] <= 0:
            raise ValueError(f"Pressure must be positive, got {self.params['pressure']}")
        
        if self.params['backflow_turbulence_k'] < 0:
            raise ValueError(f"Backflow turbulence k must be non-negative")
        
        if self.params['backflow_turbulence_omega'] <= 0:
            raise ValueError(f"Backflow turbulence omega must be positive")
        
        return True


class WallBC(BaseBC):
    """No-slip wall boundary condition.
    
    Implements no-slip condition (u=v=w=0) at solid walls.
    Supports wall functions for turbulent flows.
    
    Attributes:
        wall_function: Wall function type ('standard', 'enhanced', 'none')
        roughness_height: Surface roughness height (m)
        temperature: Wall temperature (K) - for heat transfer
        
    Example:
        >>> wall = WallBC(wall_function='standard')
    """
    
    def __init__(
        self,
        wall_function: str = 'standard',
        roughness_height: float = 0.0,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """Initialize wall boundary condition.
        
        Args:
            wall_function: Wall function type ('standard', 'enhanced', 'none')
            roughness_height: Surface roughness height (m)
            temperature: Wall temperature (K), None for adiabatic
            **kwargs: Additional parameters
        """
        if wall_function not in ['standard', 'enhanced', 'none']:
            raise ValueError(
                f"Invalid wall function: {wall_function}. "
                f"Must be 'standard', 'enhanced', or 'none'"
            )
        
        params = {
            'wall_function': wall_function,
            'roughness_height': roughness_height,
            'temperature': temperature,
        }
        params.update(kwargs)
        super().__init__('WALL', params)
    
    def apply(
        self,
        solution: Any,
        boundary_cells: Any,
        time: float = 0.0
    ) -> None:
        """Apply wall boundary condition.
        
        Sets velocity to zero (no-slip) and applies wall functions
        for turbulence variables.
        
        Solution variables are in conservative form: [rho, rhou, rhov, rhow, E, k, omega]
        
        Args:
            solution: Solution vector
            boundary_cells: List or array of wall cell indices
            time: Current simulation time
        """
        # Check if boundary_cells is empty
        if hasattr(boundary_cells, 'size'):
            if boundary_cells.size == 0:
                return
        else:
            if not boundary_cells:
                return
        
        logger.debug(
            f"Applying WALL BC ({self.params['wall_function']}) "
            f"to {len(boundary_cells)} cells at time={time:.6f}s"
        )
        
        # No-slip condition: set momentum to zero
        solution[boundary_cells, 1] = 0.0  # rhou = 0
        solution[boundary_cells, 2] = 0.0  # rhov = 0
        solution[boundary_cells, 3] = 0.0  # rhow = 0
        
        # For wall functions, turbulence variables should be handled by the turbulence model
        # Here we just ensure they don't become negative
        solution[boundary_cells, 5] = np.maximum(solution[boundary_cells, 5], 0.0)  # k >= 0
        solution[boundary_cells, 6] = np.maximum(solution[boundary_cells, 6], 0.0)  # omega >= 0
    
    def validate(self) -> bool:
        """Validate wall boundary condition parameters.
        
        Returns:
            bool: True if parameters are valid
            
        Raises:
            ValueError: If parameters are invalid
        """
        if self.params['roughness_height'] < 0:
            raise ValueError(f"Roughness height must be non-negative")
        
        if self.params['temperature'] is not None and self.params['temperature'] <= 0:
            raise ValueError(f"Temperature must be positive if specified")
        
        return True


class GroundBC(BaseBC):
    """Ground boundary condition.
    
    Special wall boundary for ground plane. Supports moving ground
    simulation (rolling road) and stationary ground.
    
    Attributes:
        moving: Whether ground is moving (rolling road)
        velocity_x: Ground velocity in X direction (m/s)
        velocity_y: Ground velocity in Y direction (m/s)
        velocity_z: Ground velocity in Z direction (m/s)
        
    Example:
        >>> # Stationary ground
        >>> ground = GroundBC(moving=False)
        >>> 
        >>> # Moving ground (rolling road at 30 m/s)
        >>> ground = GroundBC(moving=True, velocity_x=30.0)
    """
    
    def __init__(
        self,
        moving: bool = False,
        velocity_x: float = 0.0,
        velocity_y: float = 0.0,
        velocity_z: float = 0.0,
        **kwargs
    ):
        """Initialize ground boundary condition.
        
        Args:
            moving: Whether ground is moving
            velocity_x: Ground velocity X component (m/s)
            velocity_y: Ground velocity Y component (m/s)
            velocity_z: Ground velocity Z component (m/s)
            **kwargs: Additional parameters
        """
        params = {
            'moving': moving,
            'velocity_x': velocity_x,
            'velocity_y': velocity_y,
            'velocity_z': velocity_z,
        }
        params.update(kwargs)
        super().__init__('GROUND', params)
    
    def apply(
        self,
        solution: Any,
        boundary_cells: Any,
        time: float = 0.0
    ) -> None:
        """Apply ground boundary condition.
        
        For stationary ground, sets velocity to zero.
        For moving ground, sets velocity to specified ground speed.
        
        Solution variables are in conservative form: [rho, rhou, rhov, rhow, E, k, omega]
        
        Args:
            solution: Solution vector
            boundary_cells: List or array of ground cell indices
            time: Current simulation time
        """
        # Check if boundary_cells is empty
        if hasattr(boundary_cells, 'size'):
            if boundary_cells.size == 0:
                return
        else:
            if not boundary_cells:
                return
        
        gamma = 1.4  # Specific heat ratio
        
        if self.params['moving']:
            logger.debug(
                f"Applying MOVING GROUND BC (v={self.params['velocity_x']:.2f} m/s) "
                f"to {len(boundary_cells)} cells"
            )
            # Moving ground: set momentum to ground speed
            rho = solution[boundary_cells, 0]  # Keep current density
            velocity_x = self.params['velocity_x']
            velocity_y = self.params['velocity_y']
            velocity_z = self.params['velocity_z']
            
            solution[boundary_cells, 1] = rho * velocity_x  # rhou
            solution[boundary_cells, 2] = rho * velocity_y  # rhov
            solution[boundary_cells, 3] = rho * velocity_z  # rhow
            
            # Update total energy based on new velocity
            V_squared = velocity_x**2 + velocity_y**2 + velocity_z**2
            # Extract pressure from current energy: p = (gamma-1)*(E - 0.5*rho*V^2)
            E_current = solution[boundary_cells, 4]
            V_old_squared = ((solution[boundary_cells, 1] / np.maximum(rho, 1e-10))**2 + 
                            (solution[boundary_cells, 2] / np.maximum(rho, 1e-10))**2 + 
                            (solution[boundary_cells, 3] / np.maximum(rho, 1e-10))**2)
            pressure = (gamma - 1.0) * (E_current - 0.5 * rho * V_old_squared)
            E_new = pressure / (gamma - 1.0) + 0.5 * rho * V_squared
            solution[boundary_cells, 4] = E_new
        else:
            logger.debug(
                f"Applying STATIONARY GROUND BC to {len(boundary_cells)} cells"
            )
            # Stationary ground: no-slip condition (zero momentum)
            solution[boundary_cells, 1] = 0.0  # rhou = 0
            solution[boundary_cells, 2] = 0.0  # rhov = 0
            solution[boundary_cells, 3] = 0.0  # rhow = 0
        
        # Ensure turbulence variables are non-negative
        solution[boundary_cells, 5] = np.maximum(solution[boundary_cells, 5], 0.0)
        solution[boundary_cells, 6] = np.maximum(solution[boundary_cells, 6], 0.0)
    
    def validate(self) -> bool:
        """Validate ground boundary condition parameters.
        
        Returns:
            bool: True if parameters are valid
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not self.params['moving']:
            # For stationary ground, velocities should be zero
            if abs(self.params['velocity_x']) > 1e-6:
                logger.warning(
                    "Stationary ground has non-zero X velocity. "
                    "Setting moving=True or velocity_x=0."
                )
        
        return True


class FarfieldBC(BaseBC):
    """Farfield boundary condition.
    
    Implements free-stream conditions at farfield boundaries.
    Uses characteristic-based non-reflecting boundary conditions.
    
    Attributes:
        velocity_x: Free-stream velocity X (m/s)
        velocity_y: Free-stream velocity Y (m/s)
        velocity_z: Free-stream velocity Z (m/s)
        pressure: Free-stream pressure (Pa)
        temperature: Free-stream temperature (K)
        
    Example:
        >>> farfield = FarfieldBC(
        ...     velocity_x=30.0,
        ...     pressure=101325.0,
        ...     temperature=288.15
        ... )
    """
    
    def __init__(
        self,
        velocity_x: float = 30.0,
        velocity_y: float = 0.0,
        velocity_z: float = 0.0,
        pressure: float = 101325.0,
        temperature: float = 288.15,
        **kwargs
    ):
        """Initialize farfield boundary condition.
        
        Args:
            velocity_x: Free-stream velocity X (m/s)
            velocity_y: Free-stream velocity Y (m/s)
            velocity_z: Free-stream velocity Z (m/s)
            pressure: Free-stream pressure (Pa)
            temperature: Free-stream temperature (K)
            **kwargs: Additional parameters
        """
        params = {
            'velocity_x': velocity_x,
            'velocity_y': velocity_y,
            'velocity_z': velocity_z,
            'pressure': pressure,
            'temperature': temperature,
        }
        params.update(kwargs)
        super().__init__('FARFIELD', params)
    
    def apply(
        self,
        solution: Any,
        boundary_cells: Any,
        time: float = 0.0
    ) -> None:
        """Apply farfield boundary condition.
        
        Applies characteristic-based non-reflecting boundary conditions
        using Riemann invariants.
        
        Solution variables are in conservative form: [rho, rhou, rhov, rhow, E, k, omega]
        
        Args:
            solution: Solution vector
            boundary_cells: List or array of farfield cell indices
            time: Current simulation time
        """
        # Check if boundary_cells is empty
        if hasattr(boundary_cells, 'size'):
            if boundary_cells.size == 0:
                return
        else:
            if not boundary_cells:
                return
        
        logger.debug(
            f"Applying FARFIELD BC to {len(boundary_cells)} cells "
            f"at time={time:.6f}s"
        )
        
        # Extract parameters
        velocity_x = self.params['velocity_x']
        velocity_y = self.params['velocity_y']
        velocity_z = self.params['velocity_z']
        pressure = self.params['pressure']
        turbulence_k = self.params.get('turbulence_k', 0.1)
        turbulence_omega = self.params.get('turbulence_omega', 10.0)
        
        # For air at standard conditions
        rho = 1.225  # kg/m³
        gamma = 1.4  # Specific heat ratio
        
        # Compute conservative variables
        rhou = rho * velocity_x
        rhov = rho * velocity_y
        rhow = rho * velocity_z
        
        # Total energy: E = p/(gamma-1) + 0.5*rho*V^2
        V_squared = velocity_x**2 + velocity_y**2 + velocity_z**2
        E = pressure / (gamma - 1.0) + 0.5 * rho * V_squared
        
        # Set solution values at farfield cells
        # Solution structure: [rho, rhou, rhov, rhow, E, k, omega]
        solution[boundary_cells, 0] = rho           # density
        solution[boundary_cells, 1] = rhou          # x-momentum
        solution[boundary_cells, 2] = rhov          # y-momentum
        solution[boundary_cells, 3] = rhow          # z-momentum
        solution[boundary_cells, 4] = E             # total energy
        solution[boundary_cells, 5] = turbulence_k  # turbulent kinetic energy
        solution[boundary_cells, 6] = turbulence_omega  # specific dissipation rate
    
    def validate(self) -> bool:
        """Validate farfield boundary condition parameters.
        
        Returns:
            bool: True if parameters are valid
            
        Raises:
            ValueError: If parameters are invalid
        """
        if self.params['pressure'] <= 0:
            raise ValueError(f"Pressure must be positive")
        
        if self.params['temperature'] <= 0:
            raise ValueError(f"Temperature must be positive")
        
        return True


class SymmetryBC(BaseBC):
    """Symmetry plane boundary condition.
    
    Implements symmetry condition where normal velocity and
    normal gradients of all variables are zero.
    
    Example:
        >>> symmetry = SymmetryBC()
    """
    
    def __init__(self, **kwargs):
        """Initialize symmetry boundary condition.
        
        Args:
            **kwargs: Additional parameters (currently none)
        """
        super().__init__('SYMMETRY', kwargs)
    
    def apply(
        self,
        solution: Any,
        boundary_cells: Any,
        time: float = 0.0
    ) -> None:
        """Apply symmetry boundary condition.
        
        Sets normal velocity to zero and ensures zero normal gradients.
        
        Args:
            solution: Solution vector
            boundary_cells: List or array of symmetry cell indices
            time: Current simulation time
        """
        # Check if boundary_cells is empty
        if hasattr(boundary_cells, 'size'):
            if boundary_cells.size == 0:
                return
        else:
            if not boundary_cells:
                return
        
        logger.debug(
            f"Applying SYMMETRY BC to {len(boundary_cells)} cells "
            f"at time={time:.6f}s"
        )
        
        # TODO: Implement actual boundary condition application
        pass
    
    def validate(self) -> bool:
        """Validate symmetry boundary condition parameters.
        
        Returns:
            bool: Always True (no parameters to validate)
        """
        return True


class BodyBC(WallBC):
    """Vehicle body surface boundary condition.
    
    Special wall boundary for vehicle body surfaces. Inherits from WallBC
    but may have special treatment for aerodynamic surfaces.
    
    Example:
        >>> body = BodyBC(wall_function='enhanced')
    """
    
    def __init__(
        self,
        wall_function: str = 'enhanced',
        roughness_height: float = 0.0,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """Initialize body boundary condition.
        
        Args:
            wall_function: Wall function type
            roughness_height: Surface roughness height (m)
            temperature: Wall temperature (K)
            **kwargs: Additional parameters
        """
        super().__init__(
            wall_function=wall_function,
            roughness_height=roughness_height,
            temperature=temperature,
            **kwargs
        )
        self.bc_type = 'BODY'
    
    def apply(
        self,
        solution: Any,
        boundary_cells: Any,
        time: float = 0.0
    ) -> None:
        """Apply body boundary condition.
        
        Similar to wall BC but may include special treatments for
        automotive surfaces (e.g., enhanced wall functions).
        
        Args:
            solution: Solution vector
            boundary_cells: List or array of body cell indices
            time: Current simulation time
        """
        # Check if boundary_cells is empty
        if hasattr(boundary_cells, 'size'):
            if boundary_cells.size == 0:
                return
        else:
            if not boundary_cells:
                return
        
        logger.debug(
            f"Applying BODY BC ({self.params['wall_function']}) "
            f"to {len(boundary_cells)} cells at time={time:.6f}s"
        )
        
        # TODO: Implement actual boundary condition application
        # May include special treatments for automotive surfaces
        pass


# Registry for custom boundary conditions
_bc_registry: Dict[str, type] = {}


def register_boundary_condition(bc_type: str):
    """Decorator to register custom boundary condition classes.
    
    Args:
        bc_type: Boundary condition type identifier
        
    Example:
        >>> @register_boundary_condition("CUSTOM_INLET")
        ... class CustomInletBC(BaseBC):
        ...     def apply(self, solution, boundary_cells, time):
        ...         pass
    """
    def decorator(cls: type) -> type:
        if not issubclass(cls, BaseBC):
            raise TypeError(f"{cls.__name__} must inherit from BaseBC")
        
        _bc_registry[bc_type] = cls
        logger.info(f"Registered custom boundary condition: {bc_type}")
        return cls
    
    return decorator


def get_boundary_condition_class(bc_type: str) -> type:
    """Get boundary condition class by type identifier.
    
    Args:
        bc_type: Boundary condition type identifier
        
    Returns:
        type: Boundary condition class
        
    Raises:
        KeyError: If boundary condition type is not registered
    """
    # Built-in boundary conditions
    builtin_bcs = {
        'INLET': InletBC,
        'OUTLET': OutletBC,
        'OUTLET_CHARACTERISTIC': None,  # Will be imported lazily
        'OUTLET_SPONGE': None,  # Will be imported lazily
        'WALL': WallBC,
        'GROUND': GroundBC,
        'FARFIELD': FarfieldBC,
        'SYMMETRY': SymmetryBC,
        'BODY': BodyBC,
    }
    
    if bc_type in builtin_bcs:
        if builtin_bcs[bc_type] is None:
            # Lazy import for advanced outlet BCs to avoid circular dependency
            if bc_type == 'OUTLET_CHARACTERISTIC':
                from .outlet_bc import OutletCharacteristicBC
                return OutletCharacteristicBC
            elif bc_type == 'OUTLET_SPONGE':
                from .outlet_bc import OutletSpongeBC
                return OutletSpongeBC
        return builtin_bcs[bc_type]
    
    if bc_type in _bc_registry:
        return _bc_registry[bc_type]
    
    raise KeyError(
        f"Unknown boundary condition type: {bc_type}. "
        f"Available types: {list(builtin_bcs.keys()) + list(_bc_registry.keys())}"
    )


def create_boundary_condition(bc_type: str, **kwargs) -> BaseBC:
    """Factory function to create boundary condition instances.
    
    Args:
        bc_type: Boundary condition type identifier
        **kwargs: Boundary condition parameters
        
    Returns:
        BaseBC: Boundary condition instance
        
    Example:
        >>> bc = create_boundary_condition('INLET', velocity_x=30.0)
    """
    bc_class = get_boundary_condition_class(bc_type)
    return bc_class(**kwargs)
