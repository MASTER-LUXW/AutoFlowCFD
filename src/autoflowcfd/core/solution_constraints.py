"""Solution constraint handler for FVM solver.

This module applies physical constraints to the solution field to prevent
numerical blow-up during iterations.

Key Components:
    - SolutionConstraintHandler: Enforces physical bounds on solution variables
"""

import numpy as np
from loguru import logger


class SolutionConstraintHandler:
    """Applies physical constraints to solution variables."""
    
    def __init__(self, gamma: float = 1.4):
        self.gamma = gamma
    
    def apply_constraints(self, solution: np.ndarray):
        """Apply physical constraints to all cells.
        
        Ensures:
        - Density > 1e-6 kg/m³
        - Pressure > 100 Pa
        - Turbulent kinetic energy > 1e-10
        - Specific dissipation rate in [1e-6, 1e6]
        
        Args:
            solution: Solution array, shape=(n_cells, 7)
        """
        n_cells = len(solution)
        
        for cell_idx in range(n_cells):
            rho, rhou, rhov, rhow, E, k, omega = solution[cell_idx]
            
            # Clamp density
            rho = max(rho, 1e-6)
            
            # Compute velocity
            u = rhou / rho
            v = rhov / rho
            w = rhow / rho
            
            # Compute pressure
            p = (self.gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2 + w**2))
            
            # Clamp pressure and recompute energy if needed
            if p < 100.0:
                E = 100.0 / (self.gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
                p = 100.0
            
            # Clamp turbulence variables
            k = max(k, 1e-10)
            omega = max(omega, 1e-6)
            omega = min(omega, 1e6)
            
            # Update solution
            solution[cell_idx] = np.array([rho, rhou, rhov, rhow, E, k, omega])
