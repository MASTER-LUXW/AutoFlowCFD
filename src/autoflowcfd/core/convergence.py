"""Convergence monitoring and adaptive CFL control.

This module provides real-time convergence monitoring, residual tracking,
and adaptive CFL number adjustment for stable simulations.

NOT CURRENTLY USED: FRSolver.solve() (solver_steady.py) never constructs
ConvergenceMonitor - it hand-rolls equivalent residual-trend/CFL-adaptation
logic inline in its iteration loop instead.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ConvergenceHistory:
    """Container for convergence history data.
    
    Attributes:
        iterations: Iteration numbers
        residuals: Residual norms per iteration
        cd_history: Drag coefficient history
        cl_history: Lift coefficient history
        cfl_history: CFL number history
    """
    iterations: List[int] = field(default_factory=list)
    residuals: List[float] = field(default_factory=list)
    cd_history: List[float] = field(default_factory=list)
    cl_history: List[float] = field(default_factory=list)
    cfl_history: List[float] = field(default_factory=list)
    
    def add_entry(
        self,
        iteration: int,
        residual: float,
        cd: Optional[float] = None,
        cl: Optional[float] = None,
        cfl: Optional[float] = None
    ) -> None:
        """Add convergence data entry.
        
        Args:
            iteration: Iteration number
            residual: Residual norm
            cd: Drag coefficient (optional)
            cl: Lift coefficient (optional)
            cfl: CFL number (optional)
        """
        self.iterations.append(iteration)
        self.residuals.append(residual)
        
        if cd is not None:
            self.cd_history.append(cd)
        if cl is not None:
            self.cl_history.append(cl)
        if cfl is not None:
            self.cfl_history.append(cfl)
    
    def export_csv(self, filepath: str) -> None:
        """Export convergence history to CSV file.
        
        Args:
            filepath: Output CSV file path
        """
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration', 'Residual', 'Cd', 'Cl', 'CFL'])
            
            n_entries = len(self.iterations)
            for i in range(n_entries):
                cd = self.cd_history[i] if i < len(self.cd_history) else ''
                cl = self.cl_history[i] if i < len(self.cl_history) else ''
                cfl = self.cfl_history[i] if i < len(self.cfl_history) else ''
                
                writer.writerow([
                    self.iterations[i],
                    self.residuals[i],
                    cd,
                    cl,
                    cfl
                ])


class ConvergenceMonitor:
    """Monitors simulation convergence and controls adaptive CFL.
    
    Tracks residual norms, aerodynamic coefficients, and automatically
    adjusts CFL number for optimal convergence rate.
    
    Attributes:
        history: Convergence history container
        initial_residual: Initial residual norm
        convergence_threshold: Residual reduction threshold
        max_iterations: Maximum allowed iterations
        converged: Whether simulation has converged
        cfl_current: Current CFL number
        cfl_min: Minimum CFL number
        cfl_max: Maximum CFL number
    """
    
    def __init__(
        self,
        convergence_threshold: float = 1e-3,
        max_iterations: int = 5000,
        cfl_initial: float = 0.1,
        cfl_min: float = 0.01,
        cfl_max: float = 5.0,
        check_interval: int = 10
    ):
        """Initialize convergence monitor.
        
        Args:
            convergence_threshold: Residual reduction factor for convergence
            max_iterations: Maximum iteration limit
            cfl_initial: Initial CFL number
            cfl_min: Minimum CFL number
            cfl_max: Maximum CFL number
            check_interval: Check convergence every N iterations
        """
        self.history = ConvergenceHistory()
        self.initial_residual: Optional[float] = None
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.check_interval = check_interval
        
        # CFL adaptation parameters
        self.cfl_current = cfl_initial
        self.cfl_min = cfl_min
        self.cfl_max = cfl_max
        
        # Status flags
        self.converged = False
        self.diverged = False
        self.current_iteration = 0
        
        # Coefficient monitoring
        self.cd_window: List[float] = []
        self.cl_window: List[float] = []
        self.coeff_window_size = 100
    
    def update(
        self,
        residuals: np.ndarray,
        cd: Optional[float] = None,
        cl: Optional[float] = None
    ) -> bool:
        """Update monitor with new iteration data.
        
        Args:
            residuals: Current residual vector
            cd: Drag coefficient (optional)
            cl: Lift coefficient (optional)
            
        Returns:
            True if should continue iterating, False if converged/diverged
        """
        self.current_iteration += 1
        
        # Compute residual norm
        residual_norm = np.linalg.norm(residuals)
        
        # Store initial residual
        if self.initial_residual is None:
            self.initial_residual = residual_norm
        
        # Record history
        self.history.add_entry(
            iteration=self.current_iteration,
            residual=residual_norm,
            cd=cd,
            cl=cl,
            cfl=self.cfl_current
        )
        
        # Update coefficient windows
        if cd is not None:
            self.cd_window.append(cd)
            if len(self.cd_window) > self.coeff_window_size:
                self.cd_window.pop(0)
        
        if cl is not None:
            self.cl_window.append(cl)
            if len(self.cl_window) > self.coeff_window_size:
                self.cl_window.pop(0)
        
        # Check convergence periodically
        if self.current_iteration % self.check_interval == 0:
            self._check_convergence(residual_norm)
            self._adapt_cfl(residual_norm)
        
        # Check termination conditions
        if self.converged:
            print(f"[Convergence] Converged at iteration {self.current_iteration}")
            return False
        
        if self.diverged:
            print(f"[Convergence] Diverged at iteration {self.current_iteration}")
            return False
        
        if self.current_iteration >= self.max_iterations:
            print(f"[Convergence] Reached maximum iterations ({self.max_iterations})")
            return False
        
        return True
    
    def _check_convergence(self, residual_norm: float) -> None:
        """Check if simulation has converged.
        
        Convergence criteria:
        1. Residual reduced by threshold factor
        2. Coefficients stabilized (if available)
        
        Args:
            residual_norm: Current residual norm
        """
        if self.initial_residual is None:
            return
        
        # Residual reduction check
        reduction_factor = residual_norm / self.initial_residual
        
        if reduction_factor < self.convergence_threshold:
            # Check coefficient stability
            coeff_stable = self._check_coefficient_stability()
            
            if coeff_stable or len(self.cd_window) < 10:
                self.converged = True
    
    def _check_coefficient_stability(self) -> bool:
        """Check if aerodynamic coefficients have stabilized.
        
        Returns:
            True if coefficients vary less than 0.1% over window
        """
        if len(self.cd_window) < 10:
            return False
        
        # Compute coefficient variation
        cd_mean = np.mean(self.cd_window[-10:])
        cd_std = np.std(self.cd_window[-10:])
        
        # Relative variation
        if cd_mean > 1e-6:
            cd_variation = cd_std / cd_mean
        else:
            cd_variation = cd_std
        
        # Converged if variation < 0.1%
        return cd_variation < 0.001
    
    def _adapt_cfl(self, residual_norm: float) -> None:
        """Adaptively adjust CFL number based on convergence behavior.
        
        Strategy:
        - If residuals decreasing smoothly: increase CFL
        - If residuals oscillating: decrease CFL
        - If residuals increasing: significantly decrease CFL
        
        Args:
            residual_norm: Current residual norm
        """
        if len(self.history.residuals) < 3:
            return
        
        # Get recent residual trend (use last 5 points for better signal)
        n_points = min(5, len(self.history.residuals))
        recent = self.history.residuals[-n_points:]
        
        # Compute trend using log scale for better sensitivity to large changes
        if recent[0] > 1e-30:
            log_trend = np.log(recent[-1] / recent[0]) / (n_points - 1)
        else:
            log_trend = 0.0
        
        # === AGGRESSIVE DIVERGENCE PREVENTION ===
        # Check for exponential growth
        if len(recent) >= 3:
            growth_rates = [recent[i+1]/max(recent[i], 1e-30) for i in range(len(recent)-1)]
            avg_growth = np.mean(growth_rates)
            
            # If growing exponentially, drastically reduce CFL
            if avg_growth > 10.0:
                self.cfl_current = max(self.cfl_current * 0.1, self.cfl_min)
                print(f"[CFL ADJUST] Exponential growth detected (rate={avg_growth:.2f}), "
                      f"drastically reducing CFL: {self.cfl_current:.4f}")
                return
            
            elif avg_growth > 2.0:
                self.cfl_current = max(self.cfl_current * 0.3, self.cfl_min)
                print(f"[CFL ADJUST] Rapid growth detected (rate={avg_growth:.2f}), "
                      f"reducing CFL: {self.cfl_current:.4f}")
                return
        
        # Adapt CFL based on log trend
        if log_trend < -0.2:  # Rapid decrease (>20% per step)
            # Increase CFL for faster convergence
            self.cfl_current = min(self.cfl_current * 1.3, self.cfl_max)
        
        elif log_trend > 0.1:  # Increasing
            # Decrease CFL for stability
            self.cfl_current = max(self.cfl_current * 0.4, self.cfl_min)
        
        elif abs(log_trend) < 0.02:  # Stagnant
            # Slight increase to accelerate
            self.cfl_current = min(self.cfl_current * 1.05, self.cfl_max)
        
        # Log CFL adjustment
        if len(self.history.residuals) % 10 == 0:
            print(f"[CFL INFO] Iteration {len(self.history.residuals)}, "
                  f"CFL={self.cfl_current:.4f}, log_trend={log_trend:.4f}")
    
    def check_divergence(self, residual_norm: float) -> bool:
        """Check if simulation is diverging.
        
        Divergence detected if:
        - Residual increases continuously for 50 iterations
        - Residual exceeds 100x initial value
        
        Args:
            residual_norm: Current residual norm
            
        Returns:
            True if diverging
        """
        if self.initial_residual is None:
            return False
        
        # Absolute divergence check
        if residual_norm > 100.0 * self.initial_residual:
            self.diverged = True
            return True
        
        # Trend-based divergence check
        if len(self.history.residuals) >= 50:
            recent = self.history.residuals[-50:]
            
            # Check if monotonically increasing
            increasing = all(
                recent[i+1] > recent[i] 
                for i in range(len(recent)-1)
            )
            
            if increasing:
                self.diverged = True
                return True
        
        return False
    
    def get_convergence_rate(self) -> float:
        """Compute average convergence rate.
        
        Returns:
            Convergence rate (residual reduction per iteration)
        """
        if len(self.history.residuals) < 2:
            return 0.0
        
        # Use last 100 iterations
        n_samples = min(100, len(self.history.residuals))
        recent = self.history.residuals[-n_samples:]
        
        # Logarithmic slope
        log_residuals = np.log(np.array(recent) + 1e-16)
        iterations = np.arange(len(log_residuals))
        
        # Linear fit
        if len(iterations) > 1:
            slope = np.polyfit(iterations, log_residuals, 1)[0]
        else:
            slope = 0.0
        
        return float(slope)
    
    def export_report(self, filepath: str) -> None:
        """Export convergence report to file.
        
        Args:
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("AutoFlowCFD Convergence Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total Iterations: {self.current_iteration}\n")
            f.write(f"Converged: {self.converged}\n")
            f.write(f"Diverged: {self.diverged}\n\n")
            
            if self.initial_residual is not None:
                final_residual = self.history.residuals[-1]
                reduction = final_residual / self.initial_residual
                
                f.write(f"Initial Residual: {self.initial_residual:.6e}\n")
                f.write(f"Final Residual:   {final_residual:.6e}\n")
                f.write(f"Reduction Factor: {reduction:.6e}\n\n")
            
            if len(self.cd_window) > 0:
                cd_mean = np.mean(self.cd_window)
                cd_std = np.std(self.cd_window)
                
                f.write(f"Drag Coefficient (Cd):\n")
                f.write(f"  Mean: {cd_mean:.6f}\n")
                f.write(f"  Std:  {cd_std:.6f}\n\n")
            
            f.write(f"Final CFL Number: {self.cfl_current:.4f}\n")
            f.write(f"Convergence Rate: {self.get_convergence_rate():.6e}\n")
        
        print(f"[Report] Convergence report saved to {filepath}")
