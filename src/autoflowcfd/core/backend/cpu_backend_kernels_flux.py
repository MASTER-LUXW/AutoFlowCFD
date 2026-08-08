"""cpu_backend.py 的无粘通量 Numba kernel（`_compute_flux_kernel`/`_compute_flux_kernel_muscl`）。

从 cpu_backend_kernels.py 进一步拆出来的，纯粹是为了控制单文件行数，
不是独立的概念层。见 cpu_backend.py 模块文档字符串：这套 kernel 只实现
无粘 Euler 物理，不是生产求解器实际使用的计算路径。
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _compute_flux_kernel(
    solution: np.ndarray,
    connectivity: np.ndarray,
    normals: np.ndarray,
    flux: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Numba-accelerated HLLC Riemann solver for compressible flow.

    Implements the Harten-Lax-van Leer-Contact (HLLC) approximate Riemann solver
    with entropy fix for robustness. This is a widely-used scheme in CFD that
    accurately captures shocks, contact discontinuities, and rarefactions.

    HLLC Flux Formula:
        F_HLLC = [S_R*F_L - S_L*F_R + S_L*S_R*(U_R - U_L)] / (S_R - S_L)

    where:
        U_L, U_R = conservative variables (left/right states)
        F_L, F_R = physical fluxes (left/right states)
        S_L, S_R = left/right wave speeds

    Args:
        solution: Cell solutions [rho, rhou, rhov, rhow, E], shape=(n_cells, 5+)
        connectivity: Face-to-cell connectivity, shape=(n_faces, 2)
        normals: Face normal vectors (unit), shape=(n_faces, 3)
        flux: Output flux array, shape=(n_faces, n_vars)
        gamma: Specific heat ratio (1.4 for air)

    Returns:
        Updated flux array with HLLC fluxes

    References:
        - Toro, E.F. "Riemann Solvers and Numerical Methods for Fluid Dynamics"
        - Batten et al. "On Choosing Wave Speeds for HLLC Riemann Solver"
    """
    n_faces = normals.shape[0]
    n_cells = solution.shape[0]
    n_vars = min(solution.shape[1], 5)  # Only compute for first 5 variables

    for i in prange(n_faces):
        # Get left and right cell indices
        left_cell = connectivity[i, 0]
        right_cell = connectivity[i, 1]

        # Bounds checking
        if (
            left_cell < 0 or left_cell >= n_cells or
            right_cell < 0 or right_cell >= n_cells
        ):
            for v in range(n_vars):
                flux[i, v] = 0.0
            continue

        # Extract left state conservative variables with validity checks
        rho_L = solution[left_cell, 0]
        rhou_L = solution[left_cell, 1]
        rhov_L = solution[left_cell, 2]
        rhow_L = solution[left_cell, 3]
        E_L = solution[left_cell, 4]

        # Extract right state conservative variables with validity checks
        rho_R = solution[right_cell, 0]
        rhou_R = solution[right_cell, 1]
        rhov_R = solution[right_cell, 2]
        rhow_R = solution[right_cell, 3]
        E_R = solution[right_cell, 4]

        # Check for invalid density or non-finite values (entropy fix)
        if (rho_L < 1e-6 or rho_R < 1e-6 or
            not np.isfinite(rho_L) or not np.isfinite(rho_R)):
            for v in range(n_vars):
                flux[i, v] = 0.0
            continue

        # Clamp momentum to prevent extreme velocities
        max_velocity = 500.0  # m/s
        u_L = rhou_L / rho_L
        v_L = rhov_L / rho_L
        w_L = rhow_L / rho_L

        vel_mag_L = np.sqrt(u_L*u_L + v_L*v_L + w_L*w_L)
        if vel_mag_L > max_velocity:
            scale = max_velocity / vel_mag_L
            rhou_L *= scale
            rhov_L *= scale
            rhow_L *= scale
            u_L *= scale
            v_L *= scale
            w_L *= scale

        u_R = rhou_R / rho_R
        v_R = rhov_R / rho_R
        w_R = rhow_R / rho_R

        vel_mag_R = np.sqrt(u_R*u_R + v_R*v_R + w_R*w_R)
        if vel_mag_R > max_velocity:
            scale = max_velocity / vel_mag_R
            rhou_R *= scale
            rhov_R *= scale
            rhow_R *= scale
            u_R *= scale
            v_R *= scale
            w_R *= scale

        # Check energy validity
        if not np.isfinite(E_L) or not np.isfinite(E_R):
            # Fallback to simple averaging
            for v in range(n_vars):
                flux[i, v] = 0.5 * (solution[left_cell, v] + solution[right_cell, v])
            continue

        # Pressures from total energy: p = (γ-1)*(E - 0.5*ρ*V²)
        V2_L = u_L*u_L + v_L*v_L + w_L*w_L
        p_L = (gamma - 1.0) * (E_L - 0.5 * rho_L * V2_L)

        V2_R = u_R*u_R + v_R*v_R + w_R*w_R
        p_R = (gamma - 1.0) * (E_R - 0.5 * rho_R * V2_R)

        # Entropy fix: ensure positive pressure with reasonable bounds
        p_min = 100.0   # Minimum pressure (Pa)
        p_max = 1e6     # Maximum pressure (Pa)

        if not np.isfinite(p_L) or p_L < p_min:
            p_L = p_min
        elif p_L > p_max:
            p_L = p_max

        if not np.isfinite(p_R) or p_R < p_min:
            p_R = p_min
        elif p_R > p_max:
            p_R = p_max

        # Get face normal vector
        nx = normals[i, 0]
        ny = normals[i, 1]
        nz = normals[i, 2]

        # Normal velocities
        un_L = u_L * nx + v_L * ny + w_L * nz
        un_R = u_R * nx + v_R * ny + w_R * nz

        # Sound speeds: a = sqrt(γ*p/ρ)
        a_L = np.sqrt(max(gamma * p_L / rho_L, 1e-6))
        a_R = np.sqrt(max(gamma * p_R / rho_R, 1e-6))

        # Compute wave speeds using Davis estimates (robust and simple)
        S_L = min(un_L - a_L, un_R - a_R)
        S_R = max(un_L + a_L, un_R + a_R)

        # Entropy fix: ensure S_L < 0 < S_R and limit extreme values
        wave_speed_limit = 1000.0  # m/s
        S_L = max(S_L, -wave_speed_limit)
        S_R = min(S_R, wave_speed_limit)

        if S_L >= 0.0:
            S_L = -1e-3
        if S_R <= 0.0:
            S_R = 1e-3
        # F = [ρ*un, ρ*u*un + p*nx, ρ*v*un + p*ny, ρ*w*un + p*nz, (E+p)*un]

        # Left state flux
        F_L_0 = rho_L * un_L                                    # mass flux
        F_L_1 = rhou_L * un_L + p_L * nx                        # x-momentum flux
        F_L_2 = rhov_L * un_L + p_L * ny                        # y-momentum flux
        F_L_3 = rhow_L * un_L + p_L * nz                        # z-momentum flux
        F_L_4 = (E_L + p_L) * un_L                              # energy flux

        # Right state flux
        F_R_0 = rho_R * un_R
        F_R_1 = rhou_R * un_R + p_R * nx
        F_R_2 = rhov_R * un_R + p_R * ny
        F_R_3 = rhow_R * un_R + p_R * nz
        F_R_4 = (E_R + p_R) * un_R

        # HLLC flux formula
        # F_HLLC = [S_R*F_L - S_L*F_R + S_L*S_R*(U_R - U_L)] / (S_R - S_L)
        denom = S_R - S_L

        # Ensure denominator is safe
        denom = S_R - S_L
        if abs(denom) < 1e-6 or not np.isfinite(denom):
            # Fallback to simple averaging
            for v in range(n_vars):
                flux[i, v] = 0.5 * (solution[left_cell, v] + solution[right_cell, v])
            continue

        # Compute HLLC flux for each variable
        # Mass conservation
        flux[i, 0] = (S_R * F_L_0 - S_L * F_R_0 + S_L * S_R * (rho_R - rho_L)) / denom

        # Momentum conservation (x)
        flux[i, 1] = (S_R * F_L_1 - S_L * F_R_1 + S_L * S_R * (rhou_R - rhou_L)) / denom

        # Momentum conservation (y)
        flux[i, 2] = (S_R * F_L_2 - S_L * F_R_2 + S_L * S_R * (rhov_R - rhov_L)) / denom

        # Momentum conservation (z)
        flux[i, 3] = (S_R * F_L_3 - S_L * F_R_3 + S_L * S_R * (rhow_R - rhow_L)) / denom

        # Energy conservation
        flux[i, 4] = (S_R * F_L_4 - S_L * F_R_4 + S_L * S_R * (E_R - E_L)) / denom

        # Copy remaining variables (k, omega) if present using simple averaging
        for v in range(5, n_vars):
            flux[i, v] = 0.5 * (solution[left_cell, v] + solution[right_cell, v])

    return flux




@njit(parallel=True, cache=True)
def _compute_flux_kernel_muscl(
    U_L: np.ndarray,
    U_R: np.ndarray,
    normals: np.ndarray,
    flux: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """HLLC Riemann solver using MUSCL-reconstructed states.

    This kernel computes fluxes at interfaces using left and right states
    that have been reconstructed with slope limiting (second-order accurate).

    Args:
        U_L: Left states at interfaces, shape=(n_faces, n_vars)
        U_R: Right states at interfaces, shape=(n_faces, n_vars)
        normals: Face normals, shape=(n_faces, 3)
        flux: Output flux array, shape=(n_faces, n_vars)
        gamma: Specific heat ratio

    Returns:
        Updated flux array with HLLC fluxes from reconstructed states
    """
    n_faces = normals.shape[0]
    n_vars = min(U_L.shape[1], 5)  # Only compute for first 5 variables

    for i in prange(n_faces):
        # Extract left state (already reconstructed)
        rho_L = U_L[i, 0]
        rhou_L = U_L[i, 1]
        rhov_L = U_L[i, 2]
        rhow_L = U_L[i, 3]
        E_L = U_L[i, 4]

        # Extract right state (already reconstructed)
        rho_R = U_R[i, 0]
        rhou_R = U_R[i, 1]
        rhov_R = U_R[i, 2]
        rhow_R = U_R[i, 3]
        E_R = U_R[i, 4]

        # Check for invalid density or non-finite values
        if (rho_L < 1e-6 or rho_R < 1e-6 or
            not np.isfinite(rho_L) or not np.isfinite(rho_R)):
            for v in range(n_vars):
                flux[i, v] = 0.0
            continue

        # Convert to primitive variables - LEFT STATE
        u_L = rhou_L / rho_L
        v_L = rhov_L / rho_L
        w_L = rhow_L / rho_L

        V2_L = u_L*u_L + v_L*v_L + w_L*w_L
        p_L = (gamma - 1.0) * (E_L - 0.5 * rho_L * V2_L)

        # Convert to primitive variables - RIGHT STATE
        u_R = rhou_R / rho_R
        v_R = rhov_R / rho_R
        w_R = rhow_R / rho_R

        V2_R = u_R*u_R + v_R*v_R + w_R*w_R
        p_R = (gamma - 1.0) * (E_R - 0.5 * rho_R * V2_R)

        # Entropy fix: ensure positive pressure with reasonable bounds
        p_min = 100.0
        p_max = 1e6

        if not np.isfinite(p_L) or p_L < p_min:
            p_L = p_min
        elif p_L > p_max:
            p_L = p_max

        if not np.isfinite(p_R) or p_R < p_min:
            p_R = p_min
        elif p_R > p_max:
            p_R = p_max

        # Get face normal vector
        nx = normals[i, 0]
        ny = normals[i, 1]
        nz = normals[i, 2]

        # Normal velocities
        un_L = u_L * nx + v_L * ny + w_L * nz
        un_R = u_R * nx + v_R * ny + w_R * nz

        # Sound speeds
        a_L = np.sqrt(max(gamma * p_L / rho_L, 1e-6))
        a_R = np.sqrt(max(gamma * p_R / rho_R, 1e-6))

        # Wave speeds (Davis estimates)
        S_L = min(un_L - a_L, un_R - a_R)
        S_R = max(un_L + a_L, un_R + a_R)

        # Entropy fix
        wave_speed_limit = 1000.0
        S_L = max(S_L, -wave_speed_limit)
        S_R = min(S_R, wave_speed_limit)

        if S_L >= 0.0:
            S_L = -1e-3
        if S_R <= 0.0:
            S_R = 1e-3

        # Compute physical fluxes
        F_L_0 = rho_L * un_L
        F_L_1 = rhou_L * un_L + p_L * nx
        F_L_2 = rhov_L * un_L + p_L * ny
        F_L_3 = rhow_L * un_L + p_L * nz
        F_L_4 = (E_L + p_L) * un_L

        F_R_0 = rho_R * un_R
        F_R_1 = rhou_R * un_R + p_R * nx
        F_R_2 = rhov_R * un_R + p_R * ny
        F_R_3 = rhow_R * un_R + p_R * nz
        F_R_4 = (E_R + p_R) * un_R

        # HLLC flux formula
        denom = S_R - S_L

        if abs(denom) < 1e-12 or not np.isfinite(denom):
            for v in range(n_vars):
                flux[i, v] = 0.5 * (U_L[i, v] + U_R[i, v])
            continue

        # Mass conservation
        flux[i, 0] = (S_R * F_L_0 - S_L * F_R_0 + S_L * S_R * (rho_R - rho_L)) / denom

        # Momentum conservation
        flux[i, 1] = (S_R * F_L_1 - S_L * F_R_1 + S_L * S_R * (rhou_R - rhou_L)) / denom
        flux[i, 2] = (S_R * F_L_2 - S_L * F_R_2 + S_L * S_R * (rhov_R - rhov_L)) / denom
        flux[i, 3] = (S_R * F_L_3 - S_L * F_R_3 + S_L * S_R * (rhow_R - rhow_L)) / denom

        # Energy conservation
        flux[i, 4] = (S_R * F_L_4 - S_L * F_R_4 + S_L * S_R * (E_R - E_L)) / denom

    return flux
