"""Run a few iterations to trace divergence in detail."""
import sys
import numpy as np
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from autoflowcfd.cli.solve.mesh_loader import load_mesh_for_solver
from autoflowcfd.core.fr_solver.solver import FRSolver
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme

VOLUME = r"C:\Users\luxw_\Desktop\AutoFlowCFD\cube_demo\cube_demo_volume.nas"
SURFACE = r"C:\Users\luxw_\Desktop\AutoFlowCFD\cube_demo\cube_demo.nas"

print("Loading mesh...")
mesh, volume_data = load_mesh_for_solver(
    VOLUME, order=0, surface_mesh=SURFACE, skip_quality_check=True
)

print("Creating solver...")
solver = FRSolver(
    mesh=mesh, backend="cpu", order=0,
    turb_model_name="sst",
    time_scheme=TimeIntegrationScheme.SSP_RK3,
    n_threads=4,
)

# Compute wall distance (needed for SST)
from autoflowcfd.cli.solve.wall_distance import compute_wall_distance_for_solver
compute_wall_distance_for_solver(solver, volume_data)

# Check wall distance
wd = solver.wall_distance
print(f"\n=== Wall Distance ===")
if wd is not None:
    print(f"  min={wd.min():.6e}  max={wd.max():.6e}  mean={wd.mean():.6e}")
    print(f"  zero count: {(wd < 1e-15).sum()}")
    print(f"  < 1e-8 count: {(wd < 1e-8).sum()}")
else:
    print("  NOT SET!")

# Check state before stepping
Q = solver.state.Q
U = solver.state.U
print(f"\n=== Pre-step state check ===")
print(f"  rho: {Q[0,0,0]:.6f}")
print(f"  u:   {Q[0,0,1]:.6f}")
print(f"  p:   {Q[0,0,4]:.2f}")
print(f"  U[0,0,4] (rho*e): {U[0,0,4]:.2f}")
print(f"  k:   {Q[0,0,5]:.6e}")
print(f"  omega: {Q[0,0,6]:.6e}")

# Check CFL / dt_local
print(f"\n=== CFL / dt_local ===")
dt_local = solver._compute_local_time_step()
print(f"  dt_local: min={dt_local.min():.6e}  max={dt_local.max():.6e}  mean={dt_local.mean():.6e}")
print(f"  dt_local median: {np.median(dt_local):.6e}")

# Check turbulent viscosity
mu_t = solver._get_turbulent_viscosity_field()
if mu_t is not None:
    print(f"\n=== Turbulent viscosity ===")
    print(f"  mu_t: min={mu_t.min():.6e}  max={mu_t.max():.6e}  mean={mu_t.mean():.6e}")
    print(f"  mu_t/mu_mol ratio: max={mu_t.max()/solver.mu_molecular:.2e}")
else:
    print("\n  mu_t_field is None")

# Run 5 steps manually, checking residual at each stage
print(f"\n=== Manual stepping (dt=1e-3 as CLI) ===")
for i in range(5):
    try:
        solver.state._update_primitives()
        
        # Compute inviscid residual
        res_inv = solver.compute_inviscid_residual()
        
        if i == 0:
            print(f"\n  --- Initial residual (before any step) ---")
            print(f"  Inviscid residual shape: {res_inv.shape}")
            for v in range(min(7, res_inv.shape[2])):
                col = res_inv[:, 0, v]
                finite_mask = np.isfinite(col)
                n_finite = finite_mask.sum()
                if n_finite > 0:
                    print(f"    var{v}: min={col[finite_mask].min():.6e}  max={col[finite_mask].max():.6e}  "
                          f"rms={np.sqrt(np.mean(col[finite_mask]**2)):.6e}  "
                          f"non-finite={int((~finite_mask).sum())}")
                else:
                    print(f"    var{v}: ALL non-finite!")
        
        # Take step
        dt = 1e-3
        res = solver.step(dt)
        
        # Check state after step
        Q = solver.state.Q
        U = solver.state.U
        rho_min, rho_max = Q[:,:,0].min(), Q[:,:,0].max()
        p_min, p_max = Q[:,:,4].min(), Q[:,:,4].max()
        
        print(f"\n  --- After step {i+1}: residual_norm={res:.6e} ---")
        print(f"  rho: [{rho_min:.6e}, {rho_max:.6e}]")
        print(f"  p:   [{p_min:.6e}, {p_max:.6e}]")
        
        if Q.shape[2] >= 7:
            k_min, k_max = Q[:,:,5].min(), Q[:,:,5].max()
            w_min, w_max = Q[:,:,6].min(), Q[:,:,6].max()
            print(f"  k:   [{k_min:.6e}, {k_max:.6e}]")
            print(f"  omega: [{w_min:.6e}, {w_max:.6e}]")
        
        # Check for NaN/Inf
        if np.any(~np.isfinite(Q)):
            n_nan = np.sum(~np.isfinite(Q))
            print(f"  *** NaN/Inf in state: {n_nan} values ***")
            # Find which variable is first to blow up
            for v in range(Q.shape[2]):
                n_bad = (~np.isfinite(Q[:,:,v])).sum()
                if n_bad > 0:
                    print(f"    var{v}: {n_bad} non-finite values")
            break
            
    except Exception as e:
        print(f"  Step {i+1} failed: {e}")
        import traceback
        traceback.print_exc()
        break

print("\nDone.")
