"""Weiss-Smith style low-Mach-number preconditioning for the density-based
compressible steady solver.

At low Mach number, a non-preconditioned compressible scheme must resolve
acoustic waves (speed ~ sqrt(gamma*p/rho), e.g. ~340 m/s in air at STP)
even though the physical flow of interest moves far slower (e.g. ~30 m/s
for typical automotive external aero, M~0.09). This couples an
artificially tiny pseudo-time step (CFL limited by the acoustic wave, not
the much slower convective one) with excess numerical dissipation from the
HLLC upwinding (whose numerical viscosity also scales with the acoustic
wave speed), degrading both convergence speed and solution robustness -
most severely near stagnation points and recirculation/separation, where
the LOCAL Mach number can be near zero even in an overall M~0.1 flow
(observed directly: a bluff body's separated wake collapsing to
near-vacuum density before the true root cause, an outlet backflow
instability, was found and fixed separately).

Preconditioning replaces the physical acoustic eigenvalues (un +/- a) with
a rescaled pair whose spread is controlled by a locally-clipped Mach
number beta, decoupling the pseudo-time march (and the HLLC flux's
numerical dissipation) from the true acoustic speed. This leaves the
converged steady-state residual (R=0) mathematically unchanged: HLLC's
flux is still exactly consistent (F(U,U)=F(U) for any U) regardless of how
the SL/SR wave-speed estimates are computed, so at convergence this
produces the identical answer - just via a far better-conditioned pseudo-
time path.

Reference: Weiss & Smith (1995), "Preconditioning Applied to Variable and
Constant Density Flows", AIAA Journal 33(11):2050-2057.
"""

from __future__ import annotations

import numpy as np


def preconditioned_acoustic_eigs(
    un: np.ndarray,
    a: np.ndarray,
    mach_ref: float,
    k: float = 1.1,
):
    """Replace the raw acoustic eigenvalues (un+a, un-a) with their
    Weiss-Smith preconditioned equivalents.

    Args:
        un: local normal (or characteristic) velocity component, signed
        a: local physical speed of sound (must be > 0)
        mach_ref: reference (freestream) Mach number - beta^2 is clipped
            to never drop below k*mach_ref^2, so it stays well-behaved
            (not over-relaxed into near-incompressible-limit stiffness)
            even exactly at a stagnation point where the LOCAL Mach
            number is zero
        k: safety-margin multiplier on mach_ref^2 (Weiss-Smith recommend
            ~1.1-1.2; keeps beta^2 from sitting exactly on its floor,
            which would make the preconditioner degenerate)

    Returns:
        (lambda_plus, lambda_minus, c_precond): the two preconditioned
        acoustic eigenvalues, and the effective (reduced) acoustic speed
        sqrt(beta2)*a - a drop-in replacement for `a` in a spectral-
        radius-based CFL time-step estimate. At beta2=1 (M >= mach_ref
        i.e. transonic/supersonic-ish regions), this reduces exactly to
        (un+a, un-a, a) - the unpreconditioned values - so preconditioning
        only relaxes the stiffness where the flow is genuinely slow.
    """
    a_safe = np.maximum(a, 1e-30)
    mach_local2 = (un / a_safe) ** 2
    beta2 = np.clip(np.maximum(mach_local2, k * mach_ref ** 2), 1e-10, 1.0)

    lam_center = un * (1.0 + beta2) / 2.0
    radius = np.sqrt(((1.0 - beta2) * un / 2.0) ** 2 + beta2 * a_safe ** 2)

    lam_plus = lam_center + radius
    lam_minus = lam_center - radius
    c_precond = np.sqrt(beta2) * a_safe
    return lam_plus, lam_minus, c_precond
