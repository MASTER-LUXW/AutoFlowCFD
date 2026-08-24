"""直接测试 AUSM+up 在壁面配置下的质量通量"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

# 不用 numba，直接用 Python 重新实现 AUSM+up 的关键公式
def ausm_up_mass_flux_python(rhoL, rhoR, uL, uR, pL, pR, normal, mach_ref):
    """Python 版 AUSM+up 质量通量（无 numba）"""
    gamma = 1.4
    alpha = 0.1875
    beta = 0.5
    Kp = 0.25
    sigma_p = 1.0
    _WEISS_SMITH_K = 1.1
    
    unL = uL * normal[0] + 0.0 * normal[1] + 0.0 * normal[2]
    unR = uR * normal[0] + 0.0 * normal[1] + 0.0 * normal[2]
    
    aL = np.sqrt(max(gamma * pL / rhoL, 1e-10))
    aR = np.sqrt(max(gamma * pR / rhoR, 1e-10))
    a_half = 0.5 * (aL + aR)
    rho_half = 0.5 * (rhoL + rhoR)
    
    Mbar2 = (unL**2 + unR**2) / (2.0 * a_half**2)
    M0_sq = min(1.0, max(Mbar2, mach_ref**2))
    fa = np.sqrt(M0_sq) * (2.0 - np.sqrt(M0_sq))
    fa = max(fa, 1e-6)
    
    beta2 = min(1.0, max(max(Mbar2, _WEISS_SMITH_K * mach_ref**2), 1e-10))
    sqrt_beta2 = np.sqrt(beta2)
    a_half_p = sqrt_beta2 * a_half
    
    M_L = unL / max(a_half_p, 1e-10)
    M_R = unR / max(a_half_p, 1e-10)
    
    print(f"  a_half = {a_half:.4f}, a_half_p = {a_half_p:.4f}")
    print(f"  Mbar2 = {Mbar2:.6f}, beta2 = {beta2:.6f}")
    print(f"  M_L = {M_L:.6f}, M_R = {M_R:.6f}")
    
    # M+ and M-
    def M_plus(M):
        if abs(M) >= 1:
            return 0.5 * (M + abs(M))
        else:
            return 0.25 * (M + 1)**2 + alpha * (M**2 - 1)**2
    
    def M_minus(M):
        if abs(M) >= 1:
            return 0.5 * (M - abs(M))
        else:
            return -0.25 * (M - 1)**2 - alpha * (M**2 - 1)**2
    
    Mp = M_plus(M_L)
    Mm = M_minus(M_R)
    M_half = Mp + Mm
    
    print(f"  M+(M_L) = {Mp:.6f}")
    print(f"  M-(M_R) = {Mm:.6f}")
    print(f"  M_half = {M_half:.6f}")
    
    Mp_term = -(Kp / fa) * max(1.0 - sigma_p * Mbar2, 0.0) * (pR - pL) / (rho_half * a_half_p**2)
    print(f"  Mp = {Mp_term:.6e}")
    
    mass_flux = 0.5 * (rhoL * a_half_p + rhoR * a_half_p) * (M_half + Mp_term)
    return mass_flux

print("=" * 60)
print("Case 1: No-slip wall, un_L = 33.33 (flow toward wall)")
print("=" * 60)
mf1 = ausm_up_mass_flux_python(
    rhoL=1.225, rhoR=1.225,
    uL=33.33, uR=-33.33,  # ghost: velocity mirrored
    pL=101325.0, pR=101325.0,
    normal=np.array([1.0, 0.0, 0.0]),
    mach_ref=0.097,
)
print(f"  mass_flux = {mf1:.6e}")

print("\n" + "=" * 60)
print("Case 2: No-slip wall, un_L = 0 (flow parallel to wall)")
print("=" * 60)
mf2 = ausm_up_mass_flux_python(
    rhoL=1.225, rhoR=1.225,
    uL=0.0, uR=0.0,  # both zero
    pL=101325.0, pR=101325.0,
    normal=np.array([1.0, 0.0, 0.0]),
    mach_ref=0.097,
)
print(f"  mass_flux = {mf2:.6e}")

print("\n" + "=" * 60)
print("Case 3: Slip wall, un_L = 3.0 (small normal velocity)")
print("=" * 60)
mf3 = ausm_up_mass_flux_python(
    rhoL=1.225, rhoR=1.225,
    uL=3.0, uR=-3.0,  # ghost: normal velocity mirrored
    pL=101325.0, pR=101325.0,
    normal=np.array([1.0, 0.0, 0.0]),
    mach_ref=0.097,
)
print(f"  mass_flux = {mf3:.6e}")

print("\n" + "=" * 60)
print("Case 4: Internal face, qL = qR (uniform flow)")
print("=" * 60)
mf4 = ausm_up_mass_flux_python(
    rhoL=1.225, rhoR=1.225,
    uL=33.33, uR=33.33,  # same state
    pL=101325.0, pR=101325.0,
    normal=np.array([1.0, 0.0, 0.0]),
    mach_ref=0.097,
)
print(f"  mass_flux = {mf4:.6e}")
print(f"  Expected: rho*u = {1.225*33.33:.2f}")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print(f"  Wall (un=33.33):  mass_flux = {mf1:.6e} (should be 0)")
print(f"  Wall (un=0):      mass_flux = {mf2:.6e} (should be 0)")
print(f"  Slip (un=3.0):    mass_flux = {mf3:.6e} (should be 0)")
print(f"  Internal (uniform): mass_flux = {mf4:.6e} (should be {1.225*33.33:.2f})")
