"""Wall function models for near-wall turbulence treatment.

This module implements wall functions to handle boundary layer flows,
supporting industrial RANS meshes with y+ = 30-100.
"""

import numpy as np
from typing import Tuple


class WallFunctionModel:
    """Wall function model for RANS simulations.
    
    Supports both standard wall functions (log-law) and enhanced wall treatment
    for improved accuracy across different y+ ranges.
    
    Attributes:
        kappa: Von Karman constant
        E: Log-law constant
        y_plus_min: Minimum y+ for log-law validity
        y_plus_max: Maximum y+ for log-law validity
    """
    
    def __init__(
        self,
        kappa: float = 0.41,
        E: float = 9.8,
        y_plus_range: Tuple[float, float] = (30.0, 100.0)
    ):
        """Initialize wall function model.
        
        Args:
            kappa: Von Karman constant
            E: Log-law additive constant
            y_plus_range: Valid y+ range (min, max)
        """
        self.kappa = kappa
        self.E = E
        self.y_plus_min = y_plus_range[0]
        self.y_plus_max = y_plus_range[1]
    
    def compute_y_plus(
        self,
        u_tau: np.ndarray,
        y_distance: np.ndarray,
        nu: float
    ) -> np.ndarray:
        """Compute dimensionless wall distance y+.
        
        Args:
            u_tau: Friction velocity (m/s)
            y_distance: Distance to wall (m)
            nu: Kinematic viscosity (m²/s)
            
        Returns:
            y+ values
        """
        y_plus = u_tau * y_distance / nu
        
        return y_plus
    
    def compute_friction_velocity(
        self,
        tau_wall: np.ndarray,
        rho: float
    ) -> np.ndarray:
        """Compute friction velocity from wall shear stress.
        
        Args:
            tau_wall: Wall shear stress (Pa)
            rho: Fluid density (kg/m³)
            
        Returns:
            Friction velocity (m/s)
        """
        u_tau = np.sqrt(np.abs(tau_wall) / rho)
        
        return u_tau
    
    def standard_log_law(
        self,
        y_plus: np.ndarray
    ) -> np.ndarray:
        """Standard logarithmic law of the wall.
        
        u+ = (1/kappa) * ln(y+) + E
        
        Valid for y+ in [30, 100].
        
        Args:
            y_plus: Dimensionless wall distance
            
        Returns:
            Dimensionless velocity u+
        """
        # Clamp y+ to valid range
        y_plus_clamped = np.clip(y_plus, self.y_plus_min, self.y_plus_max)
        
        # Log-law
        u_plus = (1.0 / self.kappa) * np.log(y_plus_clamped) + self.E
        
        return u_plus
    
    def enhanced_wall_treatment(
        self,
        y_plus: np.ndarray
    ) -> np.ndarray:
        """Enhanced wall treatment for full y+ range.
        
        Combines viscous sublayer (linear) and log-law regions smoothly.
        
        Args:
            y_plus: Dimensionless wall distance
            
        Returns:
            Dimensionless velocity u+
        """
        # Viscous sublayer: u+ = y+ (for y+ < 11)
        u_plus_viscous = y_plus
        
        # Log-law region: u+ = (1/kappa)*ln(y+) + E (for y+ > 30)
        u_plus_log = (1.0 / self.kappa) * np.log(y_plus) + self.E
        
        # Blending function (smooth transition)
        # Use Spalding's formula for smooth blending
        u_plus_blended = self._spalding_formula(y_plus)
        
        return u_plus_blended
    
    def _spalding_formula(
        self,
        y_plus: np.ndarray
    ) -> np.ndarray:
        """Spalding's unified wall law.
        
        Provides smooth transition from viscous sublayer to log-law.
        
        Args:
            y_plus: Dimensionless wall distance
            
        Returns:
            Dimensionless velocity u+
        """
        # Spalding's formula: y+ = u+ + exp(-kappa*E) * [exp(kappa*u+) - 1 - kappa*u+ - (kappa*u+)²/2]
        # Inverted numerically
        
        # Initial guess
        u_plus = y_plus.copy()
        
        # Newton-Raphson iteration (simplified)
        for _ in range(5):
            f = u_plus + np.exp(-self.kappa * self.E) * \
                (np.exp(self.kappa * u_plus) - 1 - self.kappa * u_plus - 
                 0.5 * (self.kappa * u_plus)**2) - y_plus
            
            df_du = 1 + np.exp(-self.kappa * self.E) * \
                    (self.kappa * np.exp(self.kappa * u_plus) - self.kappa - 
                     self.kappa**2 * u_plus)
            
            u_plus = u_plus - f / (df_du + 1e-12)
        
        return u_plus
    
    def apply_wall_bc(
        self,
        velocity_parallel: np.ndarray,
        y_distance: np.ndarray,
        nu: float,
        rho: float,
        method: str = "standard"
    ) -> np.ndarray:
        """Apply wall boundary condition using wall functions.
        
        Args:
            velocity_parallel: Velocity parallel to wall (m/s)
            y_distance: Distance to wall (m)
            nu: Kinematic viscosity (m²/s)
            rho: Density (kg/m³)
            method: 'standard' or 'enhanced'
            
        Returns:
            Wall shear stress (Pa)
        """
        # Estimate friction velocity iteratively
        u_tau = np.ones_like(velocity_parallel) * 0.1  # Initial guess
        
        for _ in range(3):
            y_plus = self.compute_y_plus(u_tau, y_distance, nu)
            
            if method == "standard":
                u_plus = self.standard_log_law(y_plus)
            else:  # enhanced
                u_plus = self.enhanced_wall_treatment(y_plus)
            
            # Update friction velocity
            u_plus_safe = np.maximum(u_plus, 1e-6)
            u_tau_new = velocity_parallel / u_plus_safe
            
            # Relaxation
            u_tau = 0.7 * u_tau + 0.3 * u_tau_new
        
        # Compute wall shear stress
        tau_wall = rho * u_tau**2
        
        return tau_wall
    
    def validate_y_plus(
        self,
        y_plus: np.ndarray
    ) -> dict:
        """Validate y+ distribution for mesh quality assessment.
        
        Args:
            y_plus: y+ values at wall-adjacent cells
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "mean_y_plus": float(np.mean(y_plus)),
            "min_y_plus": float(np.min(y_plus)),
            "max_y_plus": float(np.max(y_plus)),
            "std_y_plus": float(np.std(y_plus)),
            "cells_in_range": int(np.sum((y_plus >= self.y_plus_min) & 
                                        (y_plus <= self.y_plus_max))),
            "total_cells": len(y_plus),
            "percentage_valid": float(np.mean((y_plus >= self.y_plus_min) & 
                                             (y_plus <= self.y_plus_max)) * 100)
        }
        
        return stats
