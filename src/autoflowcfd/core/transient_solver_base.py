"""Transient solver base class with initialization and configuration.

Provides the foundation for transient simulations including
backend setup, parameter management, and helper integration.
"""

import numpy as np
from typing import Dict


class TransientSolverBase:
    """Base class for transient solver with common functionality."""
    
    def __init__(
        self,
        backend,
        fr_scheme,
        time_integrator,
        convergence_monitor,
        dt: float = 1e-5,
        total_time: float = 0.1,
        sampling_interval: float = 1e-4,
        checkpoint_interval: float = 0.01
    ):
        """Initialize transient solver base.
        
        Args:
            backend: Computational backend instance
            fr_scheme: FR discretization scheme
            time_integrator: Time integration manager
            convergence_monitor: Convergence monitor
            dt: Time step size (s)
            total_time: Total physical time to simulate (s)
            sampling_interval: Data sampling interval (s)
            checkpoint_interval: Checkpoint save interval (s)
        """
        self.backend = backend
        self.fr_scheme = fr_scheme
        self.time_integrator = time_integrator
        self.convergence_monitor = convergence_monitor
        
        self.dt = dt
        self.total_time = total_time
        self.sampling_interval = sampling_interval
        self.checkpoint_interval = checkpoint_interval
        
        self.current_time = 0.0
        self.n_steps = 0
        
        # Statistics collectors
        self.cd_history = []
        self.cl_history = []
        self.time_stamps = []
        
        # Checkpoint tracking
        self.last_checkpoint_time = 0.0
        self.checkpoint_path = None
    
    def _should_sample(self) -> bool:
        """Check if should collect statistics at current time."""
        time_since_last = self.current_time % self.sampling_interval
        return abs(time_since_last - self.sampling_interval) < self.dt / 2
    
    def _should_checkpoint(self) -> bool:
        """Check if should save checkpoint."""
        time_since_checkpoint = self.current_time - self.last_checkpoint_time
        return time_since_checkpoint >= self.checkpoint_interval
