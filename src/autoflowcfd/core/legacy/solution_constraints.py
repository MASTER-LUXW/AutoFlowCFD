"""Solution constraint handler for FVM solver.

This module applies physical constraints to the solution field to prevent
numerical blow-up during iterations.

Key Components:
    - SolutionConstraintHandler: Enforces physical bounds on solution variables

NOT CURRENTLY USED: FRSolver.solve() (solver_steady.py) never constructs
SolutionConstraintHandler - equivalent positivity enforcement is done by
time_integration.enforce_positivity() after every RK stage instead.
"""

import numpy as np
from loguru import logger


class SolutionConstraintHandler:
    """Applies physical constraints to solution variables."""
    
    def __init__(self, gamma: float = 1.4):
        self.gamma = gamma
    
    def apply_constraints(self, solution: np.ndarray):
        """Apply physical constraints to all cells (vectorised).

        Ensures density > floor and pressure >= floor (projecting energy while
        preserving velocity), and keeps the conservative turbulence variables
        rho*k, rho*omega non-negative.  Turbulence variables are stored in
        conservative form in slots 5 and 6.

        Args:
            solution: Solution array, shape=(n_cells, 7)
        """
        # === CRITICAL: Clip velocity magnitude to prevent kinetic energy blow-up ===
        MAX_VELOCITY = 1e4  # 10 km/s - physically reasonable upper bound for external aerodynamics
        
        rho = np.maximum(solution[:, 0], 1e-6)
        solution[:, 0] = rho

        vel = solution[:, 1:4] / rho[:, None]
        
        # Clip velocity magnitude
        vel_mag = np.sqrt(np.sum(vel**2, axis=1))
        clip_mask = vel_mag > MAX_VELOCITY
        if np.any(clip_mask):
            clip_factor = MAX_VELOCITY / vel_mag[clip_mask]
            vel[clip_mask] *= clip_factor[:, None]
            # Update momentum with clipped velocities
            solution[clip_mask, 1:4] = (rho[clip_mask, None] * vel[clip_mask])
        
        ke = 0.5 * rho * np.sum(vel**2, axis=1)
        p = (self.gamma - 1.0) * (solution[:, 4] - ke)

        low = p < 100.0
        if np.any(low):
            solution[low, 4] = 100.0 / (self.gamma - 1.0) + ke[low]

        # Conservative turbulence variables: rho*k >= 0, rho*omega in [rho*1e-6, rho*1e6].
        solution[:, 5] = np.maximum(solution[:, 5], 0.0)
        solution[:, 6] = np.clip(solution[:, 6], rho * 1e-6, rho * 1e6)
