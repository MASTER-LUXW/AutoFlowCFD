"""Characteristic-based outlet boundary condition for subsonic/supersonic flows.

Implements non-reflecting boundary conditions using characteristic theory
to minimize spurious wave reflections at domain exits.

Key Features:
    - Automatic subsonic/supersonic detection
    - Pressure relaxation for subsonic outlets
    - Zero-gradient extrapolation for supersonic outlets
    - Entropy-consistent implementation

References:
    - Thompson, K.W. "Time-dependent boundary conditions for hyperbolic systems", 1987
    - Poinsot & Lele, "Boundary conditions for direct simulations", 1992
"""

import numpy as np
from typing import Dict, Any
from loguru import logger

from .conditions import BaseBC


class OutletCharacteristicBC(BaseBC):
    """Characteristic-based outlet boundary condition.
    
    This BC uses Riemann invariant theory to determine which variables
    should be specified and which should be extrapolated from interior.
    
    For subsonic outflow (Ma < 1):
        - One characteristic enters domain → specify pressure
        - Other characteristics leave → extrapolate velocity, density
    
    For supersonic outflow (Ma >= 1):
        - All characteristics leave → no BC needed (pure extrapolation)
    
    Attributes:
        pressure_ref: Reference pressure for subsonic outlets (Pa)
        relaxation_factor: Pressure relaxation factor (0-1)
        ma_threshold: Mach number threshold for sub/supersonic split
    """
    
    def __init__(
        self,
        pressure_ref: float = 101325.0,
        relaxation_factor: float = 0.1,
        ma_threshold: float = 1.0,
        **kwargs
    ):
        """Initialize characteristic outlet BC.
        
        Args:
            pressure_ref: Target static pressure at outlet (Pa)
            relaxation_factor: How aggressively to enforce pressure (0-1)
                              Lower = more stable, Higher = faster convergence
            ma_threshold: Mach number threshold to distinguish sub/supersonic
            **kwargs: Additional parameters passed to BaseBC
        """
        super().__init__('OUTLET_CHARACTERISTIC', kwargs)
        
        self.pressure_ref = pressure_ref
        self.relaxation_factor = relaxation_factor
        self.ma_threshold = ma_threshold
        
        logger.info(
            f"OutletCharacteristicBC initialized: p_ref={pressure_ref:.1f} Pa, "
            f"relaxation={relaxation_factor:.2f}"
        )
    
    def validate(self) -> bool:
        """Validate boundary condition parameters.
        
        Returns:
            True if all parameters are valid
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not (0 < self.pressure_ref < 1e7):
            raise ValueError(f"Invalid pressure_ref: {self.pressure_ref}")
        
        if not (0 <= self.relaxation_factor <= 1):
            raise ValueError(f"Invalid relaxation_factor: {self.relaxation_factor}")
        
        if not (0.1 <= self.ma_threshold <= 2.0):
            raise ValueError(f"Invalid ma_threshold: {self.ma_threshold}")
        
        return True
    
    def apply(self, solution: np.ndarray, boundary_cells: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Apply characteristic-based outlet boundary condition.
        
        Args:
            solution: Current solution array, shape=(n_cells, n_vars)
            boundary_cells: Indices of cells on outlet boundary
            time: Current simulation time (unused for steady-state)
            
        Returns:
            Modified solution with BC applied
        """
        gamma = 1.4  # Specific heat ratio for air
        
        for cell_idx in boundary_cells:
            # Extract conservative variables
            rho = solution[cell_idx, 0]
            rhou = solution[cell_idx, 1]
            rhov = solution[cell_idx, 2]
            rhow = solution[cell_idx, 3]
            E = solution[cell_idx, 4]
            
            # Check for invalid state
            if rho < 1e-6 or not np.isfinite(rho):
                logger.warning(f"Invalid density at outlet cell {cell_idx}: {rho}")
                continue
            
            # Compute primitive variables
            u = rhou / rho
            v = rhov / rho
            w = rhow / rho
            
            V2 = u*u + v*v + w*w
            p = (gamma - 1.0) * (E - 0.5 * rho * V2)
            
            # Ensure positive pressure
            if not np.isfinite(p) or p < 100.0:
                p = 100.0
            
            # Compute sound speed and Mach number
            a = np.sqrt(gamma * p / rho)
            Ma = np.sqrt(V2) / a if a > 1e-6 else 0.0
            
            if Ma < self.ma_threshold:
                # Subsonic outlet: apply pressure relaxation
                solution = self._apply_subsonic_outlet(
                    solution, cell_idx, rho, u, v, w, p, E, gamma
                )
            else:
                # Supersonic outlet: pure extrapolation (no modification)
                # All characteristics are outgoing, so interior values are valid
                pass
        
        return solution
    
    def _apply_subsonic_outlet(
        self,
        solution: np.ndarray,
        cell_idx: int,
        rho: float,
        u: float,
        v: float,
        w: float,
        p_current: float,
        E: float,
        gamma: float
    ) -> np.ndarray:
        """Apply subsonic outlet condition with pressure relaxation.
        
        For subsonic flow, one Riemann invariant enters the domain,
        corresponding to pressure. We relax pressure toward reference
        while keeping other variables from interior extrapolation.
        
        Args:
            solution: Solution array to modify
            cell_idx: Cell index
            rho: Density
            u, v, w: Velocity components
            p_current: Current pressure
            E: Total energy
            gamma: Specific heat ratio
            
        Returns:
            Modified solution
        """
        # Relax pressure toward reference value
        p_new = self.pressure_ref * self.relaxation_factor + \
                p_current * (1.0 - self.relaxation_factor)
        
        # Ensure pressure stays in physical range
        p_new = max(100.0, min(p_new, 1e6))
        
        # Recompute total energy with new pressure
        # E = p/(γ-1) + 0.5*rho*V²
        V2 = u*u + v*v + w*w
        E_new = p_new / (gamma - 1.0) + 0.5 * rho * V2
        
        # Update solution
        solution[cell_idx, 4] = E_new
        
        # Log occasional updates for debugging
        if hasattr(self, '_log_counter'):
            self._log_counter += 1
        else:
            self._log_counter = 1
        
        if self._log_counter % 100 == 0:
            logger.debug(
                f"Outlet cell {cell_idx}: p={p_current:.1f} → {p_new:.1f} Pa "
                f"(ref={self.pressure_ref:.1f}, relax={self.relaxation_factor:.2f})"
            )
        
        return solution


class OutletSpongeBC(BaseBC):
    """Sponge layer outlet boundary condition.
    
    Adds artificial damping in a buffer zone near the outlet to absorb
    outgoing waves and prevent reflections. The damping strength increases
    gradually from zero at the sponge start to maximum at the outlet.
    
    Attributes:
        damping_strength: Maximum damping coefficient (0-1)
        sponge_fraction: Fraction of domain length used as sponge layer
        coordinate_axis: Axis along which sponge is applied (0=x, 1=y, 2=z)
    """
    
    def __init__(
        self,
        damping_strength: float = 0.5,
        sponge_fraction: float = 0.1,
        coordinate_axis: int = 0,
        **kwargs
    ):
        """Initialize sponge layer outlet BC.
        
        Args:
            damping_strength: Maximum damping strength (0=no damping, 1=strong)
            sponge_fraction: Fraction of domain used as sponge layer
            coordinate_axis: Coordinate axis for sponge direction
            **kwargs: Additional parameters
        """
        super().__init__('OUTLET_SPONGE', kwargs)
        
        self.damping_strength = damping_strength
        self.sponge_fraction = sponge_fraction
        self.coordinate_axis = coordinate_axis
        
        logger.info(
            f"OutletSpongeBC initialized: strength={damping_strength:.2f}, "
            f"fraction={sponge_fraction:.2f}, axis={coordinate_axis}"
        )
    
    def validate(self) -> bool:
        """Validate boundary condition parameters.
        
        Returns:
            True if all parameters are valid
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not (0 <= self.damping_strength <= 1):
            raise ValueError(f"Invalid damping_strength: {self.damping_strength}")
        
        if not (0 < self.sponge_fraction < 1):
            raise ValueError(f"Invalid sponge_fraction: {self.sponge_fraction}")
        
        if self.coordinate_axis not in [0, 1, 2]:
            raise ValueError(f"Invalid coordinate_axis: {self.coordinate_axis}")
        
        return True
    
    def apply(
        self,
        solution: np.ndarray,
        boundary_cells: np.ndarray,
        time: float = 0.0,
        cell_centers: np.ndarray = None,
        domain_bounds: Dict[str, float] = None
    ) -> np.ndarray:
        """Apply sponge layer damping to residuals.
        
        Note: This BC modifies residuals rather than solution directly.
        It should be called after residual computation but before update.
        
        Args:
            solution: Current solution (unused, kept for interface consistency)
            boundary_cells: Cells in sponge region
            time: Simulation time
            cell_centers: Cell center coordinates (required)
            domain_bounds: Domain extent {min, max} along sponge axis
            
        Returns:
            Modified solution (unchanged, actual damping applied to residuals)
        """
        if cell_centers is None or domain_bounds is None:
            logger.warning("SpongeBC requires cell_centers and domain_bounds")
            return solution
        
        # This BC doesn't modify solution directly
        # Damping is applied through residual modification in solver
        return solution
    
    def compute_damping_factor(
        self,
        cell_center: np.ndarray,
        domain_min: float,
        domain_max: float
    ) -> float:
        """Compute local damping factor based on position.
        
        Damping increases quadratically from 0 at sponge_start to 
        damping_strength at outlet.
        
        Args:
            cell_center: Cell center coordinates
            domain_min: Minimum coordinate of domain
            domain_max: Maximum coordinate (outlet location)
            
        Returns:
            Damping factor in [0, damping_strength]
        """
        x = cell_center[self.coordinate_axis]
        
        # Sponge starts at this fraction from outlet
        sponge_start = domain_min + (1.0 - self.sponge_fraction) * (domain_max - domain_min)
        
        if x <= sponge_start:
            # Outside sponge layer: no damping
            return 0.0
        
        # Inside sponge layer: quadratic increase
        normalized_pos = (x - sponge_start) / (domain_max - sponge_start)
        damping = self.damping_strength * normalized_pos ** 2
        
        return min(damping, self.damping_strength)
