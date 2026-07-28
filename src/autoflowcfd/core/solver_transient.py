"""Helper functions for transient solver.

This module contains utility functions used by the transient solver
to keep the main solver class focused and reduce code complexity.
"""

import numpy as np
import h5py
import os
from typing import Tuple, Dict, Any, Optional


class TransientSolverHelpers:
    """Collection of helper methods for transient solver."""
    
    @staticmethod
    def extract_velocities(solution: np.ndarray) -> np.ndarray:
        """Extract velocity components from solution vector.
        
        Args:
            solution: Solution vector [rho, rho*u, rho*v, rho*w, E]
            
        Returns:
            Velocity array [u, v, w]
        """
        rho = solution[:, 0]
        rhou = solution[:, 1]
        rhov = solution[:, 2]
        rhow = solution[:, 3]
        
        # Avoid division by zero
        rho_safe = np.maximum(rho, 1e-10)
        
        u = rhou / rho_safe
        v = rhov / rho_safe
        w = rhow / rho_safe
        
        return np.column_stack([u, v, w])
    
    @staticmethod
    def compute_characteristic_lengths(grid_data) -> np.ndarray:
        """Compute characteristic length scales for cells.
        
        Args:
            grid_data: Grid data structure
            
        Returns:
            Characteristic lengths
        """
        # Use cube root of cell volume as characteristic length
        volumes = grid_data.cell_volumes
        lengths = np.cbrt(volumes)
        
        return lengths
    
    @staticmethod
    def compute_aero_coefficients(
        solution: np.ndarray,
        grid_data
    ) -> tuple:
        """Compute drag and lift coefficients.
        
        Args:
            solution: Solution vector
            grid_data: Grid data
            
        Returns:
            Tuple of (Cd, Cl)
        """
        # Simplified coefficient computation
        # Production code would integrate pressure and shear stress
        
        rho_inf = 1.225  # Freestream density
        u_inf = 30.0     # Freestream velocity
        q_inf = 0.5 * rho_inf * u_inf**2  # Dynamic pressure
        A_ref = 2.0      # Reference area (m²)
        
        # Extract pressure from solution
        gamma = 1.4
        rho = solution[:, 0]
        rhou = solution[:, 1]
        rhov = solution[:, 2]
        rhow = solution[:, 3]
        E = solution[:, 4]
        
        u = rhou / np.maximum(rho, 1e-10)
        v = rhov / np.maximum(rho, 1e-10)
        w = rhow / np.maximum(rho, 1e-10)
        
        p = (gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2 + w**2))
        
        # Simplified force integration (placeholder)
        F_drag = np.sum(p) * 0.01  # Placeholder
        F_lift = np.sum(p) * 0.001  # Placeholder
        
        Cd = F_drag / (q_inf * A_ref)
        Cl = F_lift / (q_inf * A_ref)
        
        return float(Cd), float(Cl)
    
    @staticmethod
    def save_checkpoint(
        solution: np.ndarray,
        current_time: float,
        n_steps: int,
        last_checkpoint_time: float
    ) -> Tuple[Optional[str], float]:
        """Save simulation checkpoint.
        
        Args:
            solution: Current solution
            current_time: Current simulation time
            n_steps: Current step count
            last_checkpoint_time: Time of last checkpoint
            
        Returns:
            Tuple of (checkpoint file path, updated last checkpoint time)
        """
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        filename = f"checkpoint_t{current_time:.6f}.h5"
        filepath = os.path.join(checkpoint_dir, filename)
        
        try:
            with h5py.File(filepath, 'w') as f:
                f.create_dataset('solution', data=solution)
                f.create_dataset('time', data=current_time)
                f.create_dataset('n_steps', data=n_steps)
            
            print(f"[Checkpoint] Saved to {filepath}")
            
            return filepath, current_time
        
        except Exception as e:
            print(f"[Checkpoint] Warning: Failed to save checkpoint: {e}")
            return None, last_checkpoint_time
"""Container for transient simulation results."""

from typing import Dict, Optional, List
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TransientResult:
    """Container for transient simulation results.
    
    Attributes:
        solution_final: Final solution state
        total_time: Total physical time simulated
        n_steps: Number of time steps completed
        cd_history: Drag coefficient history
        cl_history: Lift coefficient history
        time_stamps: Time stamps for each step
        checkpoint_path: Path to last checkpoint
    """
    solution_final: np.ndarray
    total_time: float
    n_steps: int
    cd_history: List[float] = field(default_factory=list)
    cl_history: List[float] = field(default_factory=list)
    time_stamps: List[float] = field(default_factory=list)
    checkpoint_path: Optional[str] = None
    
    def get_mean_coefficients(self) -> Dict[str, float]:
        """Compute time-averaged aerodynamic coefficients.
        
        Returns:
            Dictionary with mean Cd and Cl
        """
        if len(self.cd_history) == 0:
            return {"Cd": 0.0, "Cl": 0.0}
        
        # Skip initial transient (first 20%)
        n_skip = int(len(self.cd_history) * 0.2)
        
        cd_mean = float(np.mean(self.cd_history[n_skip:]))
        cl_mean = float(np.mean(self.cl_history[n_skip:]))
        
        return {"Cd": cd_mean, "Cl": cl_mean}
    
    def get_rms_coefficients(self) -> Dict[str, float]:
        """Compute RMS fluctuations of coefficients.
        
        Returns:
            Dictionary with RMS Cd' and Cl'
        """
        if len(self.cd_history) < 10:
            return {"Cd_rms": 0.0, "Cl_rms": 0.0}
        
        n_skip = int(len(self.cd_history) * 0.2)
        
        cd_rms = float(np.std(self.cd_history[n_skip:]))
        cl_rms = float(np.std(self.cl_history[n_skip:]))
        
        return {"Cd_rms": cd_rms, "Cl_rms": cl_rms}
"""Transient solver main loop with statistics collection.

This module provides backward compatibility by re-exporting from submodules.
For new code, import directly from:
    - autoflowcfd.core.transient_result
    - autoflowcfd.core.transient_solver_base
    - autoflowcfd.core.transient_solver_loop
"""

# Re-export from submodules for backward compatibility
from .transient_result import TransientResult
from .transient_solver_loop import TransientSolver

__all__ = [
    'TransientResult',
    'TransientSolver',
]
