"""Deep diagnosis: check Jacobians, metric terms, viscous residual."""
import sys
import numpy as np
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from autoflowcfd.cli.solve_mesh_loader import load_mesh_for_solver
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

from autoflowcfd.cli.solve_wall_distance import compute_wall_distance_for_solver
compute_wall_distance_for_solver(solver, volume_data)

# Check Jacobians
print(f"\n=== Jacobian Analysis (P0) ===")
det_jacs = mesh.jacobians["det_jacs"]
print(f"  det_jacs shape: {det_jacs.shape}")
det_j = det_jacs.reshape(mesh.n_cells, -1)
print(f"  det_j per cell (P0, col 0): min={det_j[:,0].min():.6e}  max={det_j[:,0].max():.6e}")
print(f"  det_j < 1e-10: {(det_j[:,0] < 1e-10).sum()}")
print(f"  det_j < 1e-8: {(det_j[:,0] < 1e-8).sum()}")
print(f"  det_j < 1e-6: {(det_j[:,0] < 1e-6).sum()}")

# Check metric flux scale
mfs = solver._get_metric_flux_scale()
print(f"\n=== Metric Flux Scale ===")
print(f"  shape: {mfs.shape}")
mfs_flat = mfs.reshape(mesh.n_cells, -1)
print(f"  P0 col 0: min={mfs_flat[:,0].min():.6e}  max={mfs_flat[:,0].max():.6e}")

# Check cell volumes
vols = mesh.get_all_cell_volumes()
print(f"\n=== Cell Volumes ===")
print(f"  min={vols.min():.6e}  max={vols.max():.6e}  mean={vols.mean():.6e}")
print(f"  negative volumes: {(vols < 0).sum()}")
print(f"  volume ratio max/min: {vols.max()/max(vols.min(), 1e-30):.2e}")

# Check dt_local components
print(f"\n=== dt_local breakdown ===")
dt_local = solver._compute_local_time_step()
print(f"  dt_local: min={dt_local.min():.6e}  max={dt_local.max():.6e}")

# Now compute inviscid AND viscous residuals
solver.state._update_primitives()
res_inv = solver.compute_inviscid_residual()
res_visc = solver.compute_viscous_residual()

print(f"\n=== Initial Inviscid Residual ===")
for v in range(min(7, res_inv.shape[2])):
    col = res_inv[:, 0, v]
    finite = np.isfinite(col)
    if finite.sum() > 0:
        print(f"  var{v}: rms={np.sqrt(np.mean(col[finite]**2)):.6e}  non-finite={int((~finite).sum())}")

print(f"\n=== Initial Viscous Residual ===")
for v in range(min(7, res_visc.shape[2])):
    col = res_visc[:, 0, v]
    finite = np.isfinite(col)
    if finite.sum() > 0:
        print(f"  var{v}: rms={np.sqrt(np.mean(col[finite]**2)):.6e}  non-finite={int((~finite).sum())}")

print(f"\n=== Total Residual (inv+visc) ===")
res_total = res_inv + res_visc
for v in range(min(7, res_total.shape[2])):
    col = res_total[:, 0, v]
    finite = np.isfinite(col)
    if finite.sum() > 0:
        print(f"  var{v}: rms={np.sqrt(np.mean(col[finite]**2)):.6e}")

# Check where the largest residuals are located
print(f"\n=== Residual localization ===")
res_mag = np.sqrt(np.sum(res_total[:, 0, :5]**2, axis=1))
top_idx = np.argsort(res_mag)[-10:]
print(f"  Top 10 residual cells:")
for idx in top_idx[::-1]:
    vol = vols[idx]
    det_j_val = det_j[idx, 0]
    print(f"    cell {idx}: |res|={res_mag[idx]:.6e}  vol={vol:.6e}  det_j={det_j_val:.6e}")

# Check prism vs tet cells
n_prism = mesh.n_prism_cells
print(f"\n=== Prism vs Tet residual ===")
prism_res = np.mean(res_mag[:n_prism])
tet_res = np.mean(res_mag[n_prism:])
print(f"  Prism mean |res|: {prism_res:.6e}  ({n_prism} cells)")
print(f"  Tet mean |res|:   {tet_res:.6e}  ({mesh.n_cells - n_prism} cells)")
print(f"  Ratio prism/tet:  {prism_res/max(tet_res, 1e-30):.2e}")

# Now take one step and check what happens
print(f"\n=== After step 1 ===")
res = solver.step(1e-3)
solver.state._update_primitives()

res_inv2 = solver.compute_inviscid_residual()
res_visc2 = solver.compute_viscous_residual()
res_total2 = res_inv2 + res_visc2

print(f"  Inviscid residual RMS per var:")
for v in range(min(7, res_inv2.shape[2])):
    col = res_inv2[:, 0, v]
    finite = np.isfinite(col)
    if finite.sum() > 0:
        print(f"    var{v}: rms={np.sqrt(np.mean(col[finite]**2)):.6e}")

print(f"  Viscous residual RMS per var:")
for v in range(min(7, res_visc2.shape[2])):
    col = res_visc2[:, 0, v]
    finite = np.isfinite(col)
    if finite.sum() > 0:
        print(f"    var{v}: rms={np.sqrt(np.mean(col[finite]**2)):.6e}")

# Check state after step 1
Q = solver.state.Q
U = solver.state.U
print(f"\n  State after step 1:")
print(f"  rho: [{Q[:,:,0].min():.6e}, {Q[:,:,0].max():.6e}]")
print(f"  u:   [{Q[:,:,1].min():.6e}, {Q[:,:,1].max():.6e}]")
print(f"  p:   [{Q[:,:,4].min():.6e}, {Q[:,:,4].max():.6e}]")

# Check if any cells have very large dt_local * residual
dt_res = dt_local * np.sqrt(np.sum(res_total[:, 0, :5]**2, axis=1))
print(f"\n  dt * |res| (solution change estimate):")
print(f"  min={dt_res.min():.6e}  max={dt_res.max():.6e}  mean={dt_res.mean():.6e}")
top_dt_res_idx = np.argsort(dt_res)[-5:]
for idx in top_dt_res_idx[::-1]:
    print(f"    cell {idx}: dt*|res|={dt_res[idx]:.6e}  dt_local={dt_local[idx,0]:.6e}")

print("\nDone.")
