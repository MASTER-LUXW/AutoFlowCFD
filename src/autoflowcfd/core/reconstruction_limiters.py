"""Slope limiter functions for MUSCL reconstruction.

Implements various TVD limiters to prevent oscillations near discontinuities.
All limiters follow the form: phi(r) where r is the ratio of consecutive gradients.

References:
    - Van Leer, B. "On the relation between TVD and flux limiters", 1991
    - Toro, E.F. "Riemann Solvers and Numerical Methods", Chapter 9
"""

import numpy as np
from typing import Callable
from enum import Enum
from loguru import logger


class LimiterType(Enum):
    """Available slope limiter types."""
    NONE = "none"              # No limiting (pure linear reconstruction)
    MINMOD = "minmod"          # MinMod limiter (most dissipative)
    VAN_LEER = "van_leer"      # Van Leer limiter (balanced)
    SUPERBEE = "superbee"      # SuperBee limiter (least dissipative)
    MC = "mc"                  # MC limiter (modified centered)


class SlopeLimiters:
    """Collection of slope limiter functions.
    
    All limiters return a value in [0, 2] that scales the gradient.
    """
    
    @staticmethod
    def minmod(r: float) -> float:
        """MinMod limiter - most dissipative but most stable.
        
        phi(r) = max(0, min(1, r))
        
        Characteristics:
        - Very stable, prevents all oscillations
        - Reduces to first-order at extrema
        - Good for strong shocks
        
        Args:
            r: Gradient ratio r = (u_i - u_{i-1}) / (u_{i+1} - u_i)
            
        Returns:
            Limiter value in [0, 1]
        """
        return max(0.0, min(1.0, r))
    
    @staticmethod
    def van_leer(r: float) -> float:
        """Van Leer limiter - balanced accuracy and stability.
        
        phi(r) = (r + |r|) / (1 + |r|)
        
        Characteristics:
        - Second-order accurate in smooth regions
        - TVD (Total Variation Diminishing)
        - Good compromise for most flows
        
        Args:
            r: Gradient ratio
            
        Returns:
            Limiter value in [0, 2]
        """
        if abs(r) < 1e-10:
            return 0.0
        return (r + abs(r)) / (1.0 + abs(r))
    
    @staticmethod
    def superbee(r: float) -> float:
        """SuperBee limiter - least dissipative, highest resolution.
        
        phi(r) = max(0, min(2*r, 1), min(r, 2))
        
        Characteristics:
        - Sharpest shock capture
        - May produce slight oscillations
        - Best for contact discontinuities
        
        Args:
            r: Gradient ratio
            
        Returns:
            Limiter value in [0, 2]
        """
        return max(0.0, min(2.0 * r, 1.0), min(r, 2.0))
    
    @staticmethod
    def mc(r: float) -> float:
        """MC (Modified Centered) limiter.
        
        phi(r) = max(0, min(2*r, (1+r)/2, 2))
        
        Characteristics:
        - Less compressive than SuperBee
        - Better for smooth extrema
        
        Args:
            r: Gradient ratio
            
        Returns:
            Limiter value in [0, 2]
        """
        return max(0.0, min(2.0 * r, 0.5 * (1.0 + r), 2.0))
    
    @staticmethod
    def get_limiter(limiter_type: LimiterType) -> Callable[[float], float]:
        """Get limiter function by type.
        
        Args:
            limiter_type: Type of limiter
            
        Returns:
            Limiter function phi(r)
        """
        limiters = {
            LimiterType.NONE: lambda r: 1.0,
            LimiterType.MINMOD: SlopeLimiters.minmod,
            LimiterType.VAN_LEER: SlopeLimiters.van_leer,
            LimiterType.SUPERBEE: SlopeLimiters.superbee,
            LimiterType.MC: SlopeLimiters.mc,
        }
        
        return limiters[limiter_type]


# Numba-accelerated Van Leer limiter
try:
    from numba import njit
    
    @njit(cache=True)
    def _van_leer_limiter_numba(r: float) -> float:
        """Van Leer limiter function for Numba."""
        if r <= 0:
            return 0.0
        return (2.0 * r) / (1.0 + r)
        
except ImportError:
    logger.warning("Numba not available for limiter, using fallback")
    
    def _van_leer_limiter_numba(r: float) -> float:
        """Fallback Van Leer limiter without Numba."""
        if r <= 0:
            return 0.0
        return (2.0 * r) / (1.0 + r)
