"""FVM flux computation using HLLC Riemann solver.

Implements numerical flux calculation for compressible flows with turbulence modeling.
Supports 7-equation system: continuity, momentum (3), energy, k, omega.

NOT CURRENTLY USED: FRSolver.solve() (solver_steady.py) uses the HLLC
implementation in fvm_viscous_residual.py (ViscousRANSResidual._hllc)
instead. This module's FVMFluxCalculator is only reachable when Numba is
unavailable, via FVMResidualComputer's Python fallback path - itself unused
by the live steady solve. A fix here will not affect solve() behaviour.
"""

import numpy as np


class FVMFluxCalculator:
    """Computes numerical fluxes using HLLC Riemann solver."""
    
    def __init__(self, gamma: float = 1.4):
        self.gamma = gamma
    
    def compute_flux(self, U_left: np.ndarray, U_right: np.ndarray,
                     normal: np.ndarray, area: float) -> np.ndarray:
        """Compute HLLC flux across a face.
        
        Args:
            U_left: Left state [rho, rhou, rhov, rhow, E, k, omega]
            U_right: Right state
            normal: Unit normal vector
            area: Face area
            
        Returns:
            Flux vector scaled by area
        """
        rho_L, rhou_L, rhov_L, rhow_L, E_L, k_L, omega_L = U_left
        rho_R, rhou_R, rhov_R, rhow_R, E_R, k_R, omega_R = U_right
        
        # Primitive variables
        u_L = rhou_L / max(rho_L, 1e-10)
        v_L = rhov_L / max(rho_L, 1e-10)
        w_L = rhow_L / max(rho_L, 1e-10)
        
        u_R = rhou_R / max(rho_R, 1e-10)
        v_R = rhov_R / max(rho_R, 1e-10)
        w_R = rhow_R / max(rho_R, 1e-10)
        
        # Pressure
        p_L = (self.gamma - 1.0) * (E_L - 0.5 * rho_L * (u_L**2 + v_L**2 + w_L**2))
        p_R = (self.gamma - 1.0) * (E_R - 0.5 * rho_R * (u_R**2 + v_R**2 + w_R**2))
        p_L = max(p_L, 100.0)
        p_R = max(p_R, 100.0)
        
        # Speed of sound
        a_L = np.sqrt(self.gamma * p_L / max(rho_L, 1e-10))
        a_R = np.sqrt(self.gamma * p_R / max(rho_R, 1e-10))
        
        # Normal velocity
        u_n_L = u_L * normal[0] + v_L * normal[1] + w_L * normal[2]
        u_n_R = u_R * normal[0] + v_R * normal[1] + w_R * normal[2]
        
        # HLLC wave speeds
        S_L = min(u_n_L - a_L, u_n_R - a_R)
        S_R = max(u_n_L + a_L, u_n_R + a_R)
        
        # Contact speed
        if abs(S_R - S_L) > 1e-10:
            S_star = (p_R - p_L + rho_L * u_n_L * (S_L - u_n_L) - 
                     rho_R * u_n_R * (S_R - u_n_R)) / \
                     (rho_L * (S_L - u_n_L) - rho_R * (S_R - u_n_R))
        else:
            S_star = 0.5 * (u_n_L + u_n_R)
        
        # Compute flux
        if S_L >= 0:
            flux = self._convective_flux(U_left, normal)
        elif S_R <= 0:
            flux = self._convective_flux(U_right, normal)
        else:
            if S_star >= 0:
                U_star = self._star_state(U_left, S_L, S_star, normal)
                flux = self._convective_flux(U_left, normal) + S_L * (U_star - U_left)
            else:
                U_star = self._star_state(U_right, S_R, S_star, normal)
                flux = self._convective_flux(U_right, normal) + S_R * (U_star - U_right)
        
        return flux * area
    
    def _convective_flux(self, U: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Compute convective flux."""
        rho, rhou, rhov, rhow, E, k, omega = U
        
        u = rhou / max(rho, 1e-10)
        v = rhov / max(rho, 1e-10)
        w = rhow / max(rho, 1e-10)
        
        p = (self.gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2 + w**2))
        p = max(p, 100.0)
        
        u_n = u * normal[0] + v * normal[1] + w * normal[2]
        
        flux = np.zeros(7)
        flux[0] = rho * u_n
        flux[1] = rho * u * u_n + p * normal[0]
        flux[2] = rho * v * u_n + p * normal[1]
        flux[3] = rho * w * u_n + p * normal[2]
        flux[4] = (E + p) * u_n
        flux[5] = k * u_n
        flux[6] = omega * u_n
        
        return flux
    
    def _star_state(self, U: np.ndarray, S_wave: float, 
                   S_star: float, normal: np.ndarray) -> np.ndarray:
        """Compute star region state."""
        rho, rhou, rhov, rhow, E, k, omega = U
        
        u = rhou / max(rho, 1e-10)
        v = rhov / max(rho, 1e-10)
        w = rhow / max(rho, 1e-10)
        
        p = (self.gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2 + w**2))
        p = max(p, 100.0)
        
        u_n = u * normal[0] + v * normal[1] + w * normal[2]
        
        rho_star = rho * (S_wave - u_n) / (S_wave - S_star)
        rho_star = max(rho_star, 1e-6)
        
        u_star = S_star * normal[0] + (u - u_n * normal[0])
        v_star = S_star * normal[1] + (v - u_n * normal[1])
        w_star = S_star * normal[2] + (w - u_n * normal[2])
        
        rhou_star = rho_star * u_star
        rhov_star = rho_star * v_star
        rhow_star = rho_star * w_star
        
        E_star = rho_star * (p / max(rho, 1e-10)) / (self.gamma - 1.0) + \
                 0.5 * rho_star * (u_star**2 + v_star**2 + w_star**2)
        
        k_star = k * (S_wave - u_n) / (S_wave - S_star)
        omega_star = omega * (S_wave - u_n) / (S_wave - S_star)
        
        return np.array([rho_star, rhou_star, rhov_star, rhow_star, 
                        E_star, k_star, omega_star])
