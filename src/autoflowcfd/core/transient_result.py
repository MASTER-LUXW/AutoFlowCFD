"""Transient solver result container and statistics.

Provides data structures for storing transient simulation results
with time-averaged statistics computation.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


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
