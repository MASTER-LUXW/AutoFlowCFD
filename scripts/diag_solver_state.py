"""Diagnose solver divergence: check freestream, initialization, CFL."""
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
    n_threads=-1,
)

print("\n=== Freestream conditions ===")
for k, v in solver.freestream.items():
    print(f"  {k}: {v}")

print(f"\n=== State initialization ===")
Q = solver.state.Q
print(f"  Shape: {Q.shape}  (n_cells, n_sps, n_vars)")
n_vars = Q.shape[2]
var_names = ["rho", "rho*u", "rho*v", "rho*w", "E", "k", "omega"][:n_vars]
for v_idx in range(min(5, n_vars)):
    col = Q[:, 0, v_idx]
    print(f"  {var_names[v_idx]:>8s}: min={col.min():.6e}  max={col.max():.6e}  mean={col.mean():.6e}")

# Check derived quantities
rho = Q[:, 0, 0]
rhou = Q[:, 0, 1]
rhov = Q[:, 0, 2]
rhow = Q[:, 0, 3]
E = Q[:, 0, 4]
gamma = 1.4
p = (gamma - 1) * (E - 0.5 * (rhou**2 + rhov**2 + rhow**2) / rho)
print(f"\n  Derived pressure: min={p.min():.6e}  max={p.max():.6e}  mean={p.mean():.6e}")
print(f"  Expected p_inf:   {solver.freestream['p_inf']:.6e}")
T = p / (rho * 287.0)
print(f"  Derived temperature: min={T.min():.4f}  max={T.max():.4f}  mean={T.mean():.4f}")
a = np.sqrt(gamma * 287.0 * T)
vel = np.sqrt((rhou/rho)**2 + (rhov/rho)**2 + (rhow/rho)**2)
M = vel / a
print(f"  Mach number: min={M.min():.6f}  max={M.max():.6f}  mean={M.mean():.6f}")

print(f"\n=== Mesh info ===")
print(f"  n_cells: {mesh.n_cells}")
print(f"  n_prism_cells: {mesh.n_prism_cells}")
print(f"  boundary_bc_types: {dict(mesh.boundary_bc_types) if mesh.boundary_bc_types else None}")

# Check SST turbulence state
if n_vars >= 7:
    k_vals = Q[:, 0, 5]
    omega_vals = Q[:, 0, 6]
    print(f"\n=== SST Turbulence State ===")
    print(f"  k:     min={k_vals.min():.6e}  max={k_vals.max():.6e}")
    print(f"  omega: min={omega_vals.min():.6e}  max={omega_vals.max():.6e}")
    # Check for negative values
    print(f"  k negative count: {(k_vals < 0).sum()}")
    print(f"  omega negative count: {(omega_vals < 0).sum()}")

# Check wall distance field
if hasattr(solver, 'wall_distance') and solver.wall_distance is not None:
    wd = solver.wall_distance
    print(f"\n=== Wall Distance ===")
    print(f"  min={wd.min():.6e}  max={wd.max():.6e}  mean={wd.mean():.6e}")
    print(f"  zero count: {(wd < 1e-15).sum()}")
else:
    print(f"\n=== Wall Distance: NOT SET ===")

# Check CFL / time step
print(f"\n=== CFL Configuration ===")
print(f"  solver.cfl_base: {getattr(solver, 'cfl_base', 'N/A')}")
print(f"  solver.current_order: {solver.current_order}")

print("\nDone.")
