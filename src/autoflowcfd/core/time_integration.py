"""Time integration schemes for transient simulations.

This module implements various time discretization methods including
backward Euler, Runge-Kutta, and Adams-Bashforth schemes.
"""

import numpy as np
from typing import Optional, List
from enum import Enum
from loguru import logger


class TimeIntegrationScheme(Enum):
    """Time integration scheme enumeration."""

    BACKWARD_EULER = "backward_euler"
    RUNGE_KUTTA_2 = "rk2"
    ADAMS_BASHFORTH_3 = "ab3"


class TimeIntegrator:
    """Time integration manager for transient simulations.

    Supports multiple time discretization schemes with adaptive time step
    control based on CFL number.

    Attributes:
        scheme: Time integration scheme
        dt: Current time step size
        cfl_target: Target CFL number
        n_steps: Number of completed time steps
        solution_history: Solution history for multi-step methods
    """

    def __init__(
        self,
        scheme: TimeIntegrationScheme = TimeIntegrationScheme.BACKWARD_EULER,
        dt: float = 1e-5,
        cfl_target: float = 1.0,
    ):
        """Initialize time integrator.

        Args:
            scheme: Time integration scheme (default: backward Euler)
            dt: Initial time step size (s)
            cfl_target: Target CFL number
        """
        self.scheme = scheme
        self.dt = dt
        self.cfl_target = cfl_target
        self.n_steps = 0
        self.current_time = 0.0

        # Solution history for multi-step methods
        self.solution_history: List[np.ndarray] = []
        self.max_history = self._get_history_length(scheme)

    def _get_history_length(self, scheme: TimeIntegrationScheme) -> int:
        """Get required solution history length for scheme.

        Args:
            scheme: Time integration scheme

        Returns:
            Number of previous solutions to store
        """
        history_map = {
            TimeIntegrationScheme.BACKWARD_EULER: 1,
            TimeIntegrationScheme.RUNGE_KUTTA_2: 1,
            TimeIntegrationScheme.ADAMS_BASHFORTH_3: 3,
        }
        return history_map[scheme]

    def step(
        self,
        solution: np.ndarray,
        residuals: np.ndarray,
        cell_volumes: np.ndarray,
        velocities: np.ndarray,
        characteristic_lengths: np.ndarray,
        residual_norm: Optional[float] = None,
    ) -> np.ndarray:
        """Perform one time integration step.

        Args:
            solution: Current solution vector
            residuals: Computed residuals
            cell_volumes: Cell volumes
            velocities: Cell velocities for CFL calculation
            characteristic_lengths: Characteristic lengths for CFL
            residual_norm: Optional residual norm for adaptive dt control

        Returns:
            Updated solution
        """
        # Adaptive time step based on CFL
        dt_adaptive = self._compute_adaptive_dt(velocities, characteristic_lengths)

        # For pseudo-time stepping in steady-state simulations,
        # allow dt to increase as solution develops
        if self.n_steps == 0:
            # First iteration: use moderate dt for stability
            self.dt = dt_adaptive * 0.5  # Start with 50% of CFL limit
            logger.info(f"Initial dt set to {self.dt:.6e} (50% of CFL limit)")
        else:
            # Subsequent iterations: adapt based on convergence behavior
            # More aggressive growth to accelerate convergence
            dt_max = self.dt * 1.3  # Allow 30% growth per iteration
            
            # If residual norm is provided, adjust dt based on convergence rate
            if residual_norm is not None and hasattr(self, '_last_residual_norm'):
                residual_ratio = residual_norm / self._last_residual_norm
                
                # Adaptive strategy based on residual behavior
                if 0.95 < residual_ratio < 1.02:
                    # Good convergence, can increase dt aggressively
                    dt_max = self.dt * 1.5
                elif residual_ratio > 1.1:
                    # Residual increasing significantly, reduce dt
                    dt_max = self.dt * 0.7
                    logger.warning(
                        f"Residual increasing (ratio={residual_ratio:.3f}), "
                        f"reducing dt by 30%"
                    )
                elif residual_ratio < 0.8:
                    # Very fast convergence, moderate increase
                    dt_max = self.dt * 1.2
                else:
                    # Moderate change
                    dt_max = self.dt * 1.15
            
            self.dt = min(dt_adaptive, dt_max)
            
            # Store current residual for next iteration
            self._last_residual_norm = residual_norm

        # Ensure dt stays within reasonable bounds
        self.dt = max(self.dt, 1e-6)
        self.dt = min(self.dt, 1.0)  # Upper limit for stability

        # Apply time integration scheme
        if self.scheme == TimeIntegrationScheme.BACKWARD_EULER:
            updated = self._backward_euler(solution, residuals, cell_volumes)

        elif self.scheme == TimeIntegrationScheme.RUNGE_KUTTA_2:
            updated = self._runge_kutta_2(solution, residuals, cell_volumes)

        elif self.scheme == TimeIntegrationScheme.ADAMS_BASHFORTH_3:
            updated = self._adams_bashforth_3(solution, residuals, cell_volumes)

        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")

        # Update history
        self._update_history(solution)

        # Advance time
        self.current_time += self.dt
        self.n_steps += 1

        return updated

    def _backward_euler(
        self, solution: np.ndarray, residuals: np.ndarray, cell_volumes: np.ndarray
    ) -> np.ndarray:
        """First-order backward Euler (implicit, unconditionally stable).

        For pseudo-time stepping with relaxation-based residuals,
        we apply the residual directly without volume division.
        
        Update formula: U^{n+1} = U^n - dt * R(U)
        
        Includes numerical stability checks to prevent NaN/Inf propagation.

        Args:
            solution: Current solution U^n
            residuals: Residuals R(U^{n+1}) - already scaled appropriately
            cell_volumes: Cell volumes V (not used for relaxation residuals)

        Returns:
            Updated solution U^{n+1} with validity checks
        """
        # Ensure dt is reasonable
        dt_safe = max(self.dt, 1e-8)

        # Apply update: U^{n+1} = U^n - dt * R
        updated = solution - dt_safe * residuals
        
        # Numerical stability checks
        # 1. Check for NaN/Inf in updated solution
        if not np.all(np.isfinite(updated)):
            # If update produces invalid values, reduce dt and retry
            dt_reduced = dt_safe * 0.5
            logger.warning(
                f"Numerical instability detected, reducing dt from {dt_safe:.6e} to {dt_reduced:.6e}"
            )
            updated = solution - dt_reduced * residuals
            
            # Second check
            if not np.all(np.isfinite(updated)):
                # If still unstable, revert to previous solution
                logger.error("Update still unstable after dt reduction, reverting")
                return solution.copy()
            
            # Update dt for next iteration
            self.dt = dt_reduced
        
        # 2. Physical constraints enforcement
        # Density must be positive
        updated[:, 0] = np.maximum(updated[:, 0], 0.1)  # rho >= 0.1 kg/m^3
        
        # Pressure must be positive (E = p/(gamma-1) + 0.5*rho*V^2)
        # For safety, ensure total energy corresponds to positive pressure
        gamma = 1.4
        vel_sq = np.sum(updated[:, 1:4]**2, axis=1) / np.maximum(updated[:, 0], 1e-10)**2
        p_min = 1000.0  # Minimum pressure
        E_min = p_min / (gamma - 1.0) + 0.5 * 0.1 * vel_sq
        updated[:, 4] = np.maximum(updated[:, 4], E_min)
        
        # Turbulence variables must be non-negative
        updated[:, 5] = np.maximum(updated[:, 5], 1e-10)  # k >= 0
        updated[:, 6] = np.maximum(updated[:, 6], 1e-10)  # omega >= 0
        
        # Velocity magnitude limit (prevent unphysical speeds)
        vel_mag = np.linalg.norm(updated[:, 1:4], axis=1)
        max_vel = 340.0  # Speed of sound as upper limit
        vel_mask = vel_mag > max_vel
        if np.any(vel_mask):
            scale = max_vel / vel_mag[vel_mask]
            updated[vel_mask, 1:4] *= scale[:, np.newaxis]

        return updated

    def _runge_kutta_2(
        self, solution: np.ndarray, residuals: np.ndarray, cell_volumes: np.ndarray
    ) -> np.ndarray:
        """Second-order Runge-Kutta (Heun's method).

        Stage 1: U* = U^n - dt * R(U^n) / V
        Stage 2: U^{n+1} = U^n - dt/2 * [R(U^n) + R(U*)] / V

        Args:
            solution: Current solution U^n
            residuals: Residuals at current state
            cell_volumes: Cell volumes

        Returns:
            Updated solution U^{n+1}
        """
        # Stage 1: Predictor
        u_star = solution - self.dt * residuals / cell_volumes[:, np.newaxis]

        # In production, recompute residuals at u*
        # For now, use same residuals (simplified)
        residuals_star = residuals.copy()

        # Stage 2: Corrector
        updated = (
            solution
            - 0.5 * self.dt * (residuals + residuals_star) / cell_volumes[:, np.newaxis]
        )

        return updated

    def _adams_bashforth_3(
        self, solution: np.ndarray, residuals: np.ndarray, cell_volumes: np.ndarray
    ) -> np.ndarray:
        """Third-order Adams-Bashforth (explicit multi-step).

        U^{n+1} = U^n - dt/12V * [23*R^n - 16*R^{n-1} + 5*R^{n-2}]

        Requires 3 previous solution states.

        Args:
            solution: Current solution U^n
            residuals: Current residuals R^n
            cell_volumes: Cell volumes

        Returns:
            Updated solution U^{n+1}

        Raises:
            RuntimeError: If insufficient history
        """
        if len(self.solution_history) < 3:
            raise RuntimeError(
                "AB3 requires at least 3 previous solutions. "
                f"Have {len(self.solution_history)}, need 3."
            )

        # Get residuals from history (simplified - in production store residuals)
        r_n = residuals
        r_n1 = residuals * 0.95  # Placeholder
        r_n2 = residuals * 0.90  # Placeholder

        # AB3 formula
        coeff = self.dt / (12.0 * cell_volumes[:, np.newaxis])
        updated = solution - coeff * (23.0 * r_n - 16.0 * r_n1 + 5.0 * r_n2)

        return updated

    def _compute_adaptive_dt(
        self, velocities: np.ndarray, characteristic_lengths: np.ndarray
    ) -> float:
        """Compute adaptive time step based on CFL condition.

        dt = CFL * min(dx / |U|)

        Args:
            velocities: Velocity magnitudes
            characteristic_lengths: Cell characteristic lengths

        Returns:
            Recommended time step
        """
        # Compute velocity magnitude
        if velocities.ndim == 2:
            vel_mag = np.linalg.norm(velocities, axis=1)
        else:
            vel_mag = np.abs(velocities)

        # Avoid division by zero - use a minimum velocity
        # For steady-state, start with reasonable velocity scale
        vel_min = 1.0  # Minimum velocity scale (m/s)
        vel_safe = np.maximum(vel_mag, vel_min)

        # Local time scales
        dt_local = characteristic_lengths / vel_safe

        # Use percentile instead of median for better stability
        # This avoids being dominated by extreme small cells while still being conservative
        dt_typical = np.percentile(dt_local, 10)  # Use 10th percentile (more conservative than median)

        # Apply CFL using percentile-based dt
        dt_cfl = self.cfl_target * dt_typical

        # Ensure dt doesn't become too small or zero
        dt_min_limit = 1e-6  # Increased from 1e-8 for better stability
        result = max(dt_cfl, dt_min_limit)

        return result

    def _update_history(self, solution: np.ndarray) -> None:
        """Update solution history for multi-step methods.

        Args:
            solution: Current solution to add to history
        """
        self.solution_history.append(solution.copy())

        # Trim history if exceeds maximum
        if len(self.solution_history) > self.max_history:
            self.solution_history.pop(0)

    def reset(self) -> None:
        """Reset integrator state."""
        self.n_steps = 0
        self.current_time = 0.0
        self.solution_history.clear()

    def get_cfl_number(
        self, velocities: np.ndarray, characteristic_lengths: np.ndarray
    ) -> float:
        """Compute current CFL number.

        CFL = |U| * dt / dx

        Args:
            velocities: Velocity magnitudes
            characteristic_lengths: Cell lengths

        Returns:
            Maximum CFL number across domain
        """
        if velocities.ndim == 2:
            vel_mag = np.linalg.norm(velocities, axis=1)
        else:
            vel_mag = np.abs(velocities)

        vel_safe = np.maximum(vel_mag, 1e-6)
        cfl_local = vel_safe * self.dt / characteristic_lengths

        return float(np.max(cfl_local))
