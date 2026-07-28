"""Transient solver helper methods.

Provides utility functions for transient simulations including
aerodynamic coefficient computation and checkpoint management.
"""

import numpy as np
from typing import Tuple, Optional


class TransientSolverHelpers:
    """Helper methods for transient solver operations."""
    
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
    ) -> Tuple[float, float]:
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
            n_steps: Current step number
            last_checkpoint_time: Time of last checkpoint
            
        Returns:
            Tuple of (checkpoint_path, updated_last_checkpoint_time)
        """
        import os
        import h5py
        
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
