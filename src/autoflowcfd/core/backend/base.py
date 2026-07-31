"""Backend abstract base class for CPU/GPU computation."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SolutionVector:
    """Solution vector data structure.

    Stores the flow field solution in CONSERVED form for all cells (this is
    what the solver actually integrates and what checkpoints save under
    `solution/conserved`):
    - data[:, 0]: rho (density)
    - data[:, 1:4]: rho*u, rho*v, rho*w (momentum)
    - data[:, 4]: rho*E (total energy density)
    - data[:, 5:7]: rho*k, rho*omega (turbulence, optional)

    The get_velocity()/get_pressure()/get_turbulence() accessors below
    convert these to the PRIMITIVE quantities their names promise (actual
    velocity, static pressure, k and omega) - they used to return the raw
    conserved columns unconverted (e.g. "velocity" was really momentum,
    "pressure" was really total energy density), silently mislabeling
    values by orders of magnitude for any caller. Kept for backward
    compatibility, but note the solver's own residual/BC code does NOT use
    these - it derives primitives inline with its own gamma/floor
    conventions (see e.g. core/aero_coeffs.py).

    Attributes:
        data: Solution array, shape=(n_cells, n_variables)
        n_cells: Number of cells
        n_variables: Number of variables per cell
    """
    data: Optional[np.ndarray] = None
    n_cells: int = 0
    n_variables: int = 5

    # Ratio of specific heats, matching the equation of state used
    # throughout the solver (e.g. core/aero_coeffs.py).
    GAMMA = 1.4
    _RHO_FLOOR = 1e-10

    def __post_init__(self):
        """Initialize data array if not provided"""
        if self.data is None and self.n_cells > 0:
            self.data = np.zeros((self.n_cells, self.n_variables))

    @property
    def shape(self):
        """Get shape of solution array"""
        if self.data is not None:
            return self.data.shape
        return (0, 0)

    def get_density(self) -> np.ndarray:
        """Get density field"""
        if self.data is not None and self.data.shape[1] > 0:
            return self.data[:, 0]
        return np.array([])

    def get_velocity(self) -> tuple:
        """Get primitive velocity components (u, v, w), i.e. momentum/rho."""
        if self.data is not None and self.data.shape[1] >= 4:
            rho = np.maximum(self.data[:, 0], self._RHO_FLOOR)
            return (self.data[:, 1] / rho, self.data[:, 2] / rho, self.data[:, 3] / rho)
        return (np.array([]), np.array([]), np.array([]))

    def get_pressure(self) -> np.ndarray:
        """Get static pressure via the ideal-gas equation of state,
        p = (gamma-1) * (rho*E - 0.5*rho*|V|^2) - NOT the raw rho*E column."""
        if self.data is not None and self.data.shape[1] >= 5:
            rho = np.maximum(self.data[:, 0], self._RHO_FLOOR)
            rhoE = self.data[:, 4]
            V_sq = (self.data[:, 1]**2 + self.data[:, 2]**2 + self.data[:, 3]**2) / rho**2
            return (self.GAMMA - 1.0) * (rhoE - 0.5 * rho * V_sq)
        return np.array([])

    def get_turbulence(self) -> tuple:
        """Get primitive turbulence quantities (k, omega), i.e. their
        conserved (rho*k, rho*omega) columns divided by density. Returns
        two empty arrays if this solution has no turbulence columns."""
        if self.data is not None and self.data.shape[1] >= 7:
            rho = np.maximum(self.data[:, 0], self._RHO_FLOOR)
            return (self.data[:, 5] / rho, self.data[:, 6] / rho)
        return (np.array([]), np.array([]))


class BackendBase(ABC):
    """Abstract base class for solver backends.
    
    This class defines the interface that all computational backends
    (CPU/Numba, GPU/CUDA) must implement. It provides a unified API
    for the FR solver to interact with different hardware accelerators.
    
    Attributes:
        backend_type: Type identifier ('cpu' or 'gpu')
        available: Whether the backend is available on current system
        device_info: Hardware information dictionary
    """
    
    def __init__(self):
        """Initialize backend base class."""
        self.backend_type = "base"
        self.available = False
        self.device_info: Dict[str, Any] = {}
    
    @abstractmethod
    def initialize(
        self,
        n_cells: int,
        n_nodes: int,
        n_variables: int = 5
    ) -> None:
        """Allocate memory and initialize data structures.
        
        Args:
            n_cells: Number of cells in mesh
            n_nodes: Number of nodes in mesh
            n_variables: Number of solution variables (default: 5 for compressible flow)
        """
        pass
    
    @abstractmethod
    def compute_flux(
        self,
        solution: np.ndarray,
        cell_connectivity: np.ndarray,
        face_normals: np.ndarray,
        gamma: float = 1.4
    ) -> np.ndarray:
        """Compute numerical flux at all cell interfaces.
        
        Args:
            solution: Solution vector, shape=(n_cells, n_variables)
            cell_connectivity: Cell connectivity array
            face_normals: Face normal vectors
            gamma: Specific heat ratio
            
        Returns:
            Flux tensor at interfaces
        """
        pass
    
    @abstractmethod
    def compute_residuals(
        self,
        solution: np.ndarray,
        flux: np.ndarray,
        cell_volumes: np.ndarray,
        boundary_mask: np.ndarray
    ) -> np.ndarray:
        """Compute residuals from flux divergence.
        
        Args:
            solution: Current solution state
            flux: Computed fluxes at interfaces
            cell_volumes: Cell volumes
            boundary_mask: Boundary condition mask
            
        Returns:
            Residual vector, shape=(n_cells, n_variables)
        """
        pass
    
    @abstractmethod
    def update_solution(
        self,
        solution: np.ndarray,
        residuals: np.ndarray,
        dt: float,
        cfl: float
    ) -> np.ndarray:
        """Update solution using time integration scheme.
        
        Args:
            solution: Current solution
            residuals: Computed residuals
            dt: Time step size
            cfl: CFL number
            
        Returns:
            Updated solution
        """
        pass
    
    @abstractmethod
    def apply_boundary_conditions(
        self,
        solution: np.ndarray,
        boundary_map: Dict[str, np.ndarray],
        bc_params: Dict[str, Any]
    ) -> np.ndarray:
        """Apply boundary conditions to solution.
        
        Args:
            solution: Solution vector
            boundary_map: Mapping of boundary names to cell indices
            bc_params: Boundary condition parameters
            
        Returns:
            Solution with boundary conditions applied
        """
        pass
    
    @abstractmethod
    def synchronize(self) -> None:
        """Synchronize data (important for GPU async operations)."""
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Get hardware device information.
        
        Returns:
            Dictionary containing device specifications
        """
        pass
    
    def cleanup(self) -> None:
        """Release allocated resources."""
        pass
