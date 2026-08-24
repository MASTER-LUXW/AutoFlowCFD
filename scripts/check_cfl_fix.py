"""Quick check: compare CFL dt before and after the face-based fix."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

# Load mesh and set up solver
from autoflowcfd.cli.solve_mesh_loader import load_mesh_for_solver
from autoflowcfd.core.fr_solver.solver import FRSolver

volume_nas = r"C:\Users\luxw_\Desktop\AutoFlowCFD\cube_demo\cube_demo_volume.nas"
surface_nas = r"C:\Users\luxw_\Desktop\AutoFlowCFD\cube_demo\cube_demo.nas"

print("Loading mesh...")
mesh, _ = load_mesh_for_solver(volume_nas, order=0, surface_mesh=surface_nas)
print(f"  n_cells={mesh.n_cells}, n_prism={mesh.n_prism_cells}")

solver = FRSolver(mesh=mesh, order=0, turb_model_name="SST", backend="cpu")

# Compute CFL dt with the new face-based method
print("\nComputing face-based CFL dt...")
from autoflowcfd.core.fr_solver.cfl import compute_local_time_step
dt_new = compute_local_time_step(solver)
dt_flat = dt_new[:, 0]  # P0: 1 SP per cell

print(f"\n=== Face-based CFL dt (NEW) ===")
print(f"  dt range: [{dt_flat.min():.4e}, {dt_flat.max():.4e}]")
print(f"  dt mean:  {dt_flat.mean():.4e}")
print(f"  dt median: {np.median(dt_flat):.4e}")

# Compare with old V^(1/3) method
volumes = mesh.get_all_cell_volumes()
h_old = np.power(np.abs(volumes), 1.0 / 3.0)
rho = solver.state.Q[:, 0, 0]
vel = solver.state.Q[:, 0, 1]
a = np.sqrt(1.4 * solver.state.Q[:, 0, 4] / rho)
wave_speed_old = vel + a  # simplified (no preconditioning for comparison)
dt_old = 0.1 * h_old / wave_speed_old

print(f"\n=== V^(1/3) CFL dt (OLD) ===")
print(f"  dt range: [{dt_old.min():.4e}, {dt_old.max():.4e}]")
print(f"  dt mean:  {dt_old.mean():.4e}")
print(f"  dt median: {np.median(dt_old):.4e}")

# Prism vs Tet comparison
n_prism = mesh.n_prism_cells
print(f"\n=== Prism vs Tet dt comparison (NEW) ===")
print(f"  Prism dt: min={dt_flat[:n_prism].min():.4e}, max={dt_flat[:n_prism].max():.4e}, mean={dt_flat[:n_prism].mean():.4e}")
print(f"  Tet dt:   min={dt_flat[n_prism:].min():.4e}, max={dt_flat[n_prism:].max():.4e}, mean={dt_flat[n_prism:].mean():.4e}")
print(f"  Ratio (tet mean / prism mean): {dt_flat[n_prism:].mean() / dt_flat[:n_prism].mean():.2f}")

print(f"\n=== Prism vs Tet dt comparison (OLD) ===")
print(f"  Prism dt: min={dt_old[:n_prism].min():.4e}, max={dt_old[:n_prism].max():.4e}, mean={dt_old[:n_prism].mean():.4e}")
print(f"  Tet dt:   min={dt_old[n_prism:].min():.4e}, max={dt_old[n_prism:].max():.4e}, mean={dt_old[n_prism:].mean():.4e}")
print(f"  Ratio (tet mean / prism mean): {dt_old[n_prism:].mean() / dt_old[:n_prism].mean():.2f}")

print("\nDone!")
