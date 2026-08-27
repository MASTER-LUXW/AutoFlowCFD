"""诊断 cube_demo 收敛慢：重建求解器，实测步内行为（残差比率、dt 限制分解、湍流场演化）。"""
import sys
import numpy as np

sys.path.insert(0, r"d:\myWorkspace\AutoFlowCFD\src")

from autoflowcfd.cli.solve_checkpoint_io import rebuild_solver_from_checkpoint
from autoflowcfd.core.fr_solver.cfl import compute_local_time_step

CKPT = r"C:\Users\luxw_\Desktop\AutoFlowCFD\cube_demo\steady_results\checkpoints\checkpoint_iter_000100.h5"

solver, iteration, metadata = rebuild_solver_from_checkpoint(CKPT)
print(f"iteration={iteration}, order={metadata.get('order')}, backend={metadata.get('backend')}")

tm = solver.turb_model
Q = solver.state.Q
vel = Q[:, :, 1:4]
vmag = np.sqrt((vel**2).sum(-1))  # (n_cells, n_sps)
print(f"\n=== 平均流场发展程度（iter {iteration}）===")
print(f"|vel|: min={vmag.min():.3f} median={np.median(vmag):.3f} max={vmag.max():.3f} (来流 {solver.freestream['vel_inf']})")
print(f"k 场: min={tm.k_field.min():.4e} max={tm.k_field.max():.4e}")

CFL = solver._cfl_controller.cfl_number if getattr(solver, '_cfl_controller', None) else 0.1
dt_local = compute_local_time_step(solver)
print(f"\n=== 局部时间步长（CFL={CFL}）===")
print(f"dt: min={dt_local.min():.3e} median={np.median(dt_local):.3e} max={dt_local.max():.3e}")

# --- 三项限制分解（复刻 cfl.py 公式，P0 order_factor=1）---
rho = Q[:, :, 0]
p = Q[:, :, 4]
a = np.sqrt(np.maximum(1.4 * p / np.maximum(rho, 1e-10), 1e-10))
wave_speed = np.maximum(vmag + a, 1e-10)

volumes = solver.mesh.get_all_cell_volumes()
fc = solver.mesh.face_connectivity
spectral = np.zeros(solver.mesh.n_cells)
ffp = solver.mesh.face_flux_points
# 面谱半径：每面取面积加权的波速
areas = fc.area
np.add.at(spectral, fc.owner_cell, (np.abs((vel[fc.owner_cell, 0] * fc.normal).sum(-1)) + a[fc.owner_cell, 0]) * areas)
mask_int = ~fc.is_boundary
nc = fc.neighbor_cell[mask_int]
np.add.at(spectral, nc, (np.abs((vel[nc, 0] * fc.normal[mask_int]).sum(-1)) + a[nc, 0]) * areas[mask_int])
dt_adv = CFL * volumes / np.maximum(spectral, 1e-30)

mu_mol = solver.mu_molecular
mu_eff = mu_mol + rho * np.maximum(tm.nu_t, 0.0)
Lc2 = np.power(np.abs(volumes), 2.0 / 3.0)
dt_visc = 0.25 * CFL * rho[:, 0] * Lc2 / np.maximum(mu_eff[:, 0], 1e-30)

metric_flux_scale = solver._get_metric_flux_scale()
n_sps = solver.mesh.n_sps_per_cell
det_jacs = solver.mesh.jacobians["det_jacs"].reshape(solver.mesh.n_cells, n_sps)
dt_geo = CFL * np.abs(det_jacs[:, 0]) / np.maximum(metric_flux_scale[:, 0] * wave_speed[:, 0], 1e-300)

for nm, arr in [("advective", dt_adv), ("viscous", dt_visc), ("geometric", dt_geo)]:
    idx = np.argmin(arr)
    print(f"dt_{nm:9s}: min={arr.min():.3e} median={np.median(arr):.3e}  "
          f"(min在cell {idx}, vol={volumes[idx]:.2e}, detJ={det_jacs[idx,0]:.2e})")

# 全局最小 dt 处三项对比
ig = np.unravel_index(np.argmin(dt_local), dt_local.shape)[0]
print(f"\n全局最小 dt 单元 {ig}: adv={dt_adv[ig]:.3e} visc={dt_visc[ig]:.3e} geo={dt_geo[ig]:.3e}")
bind = np.argmin(np.stack([dt_adv, dt_visc, dt_geo]), axis=0)
print(f"限制因素占比: advective={(bind==0).mean()*100:.1f}%  viscous={(bind==1).mean()*100:.1f}%  geometric={(bind==2).mean()*100:.1f}%")

# 物理时间尺度
L, U = 0.5, solver.freestream['vel_inf']
t_pass = L / U
print(f"\n特征流过时间 ~{t_pass:.4f}s; 中位 dt={np.median(dt_local):.2e}s → 每步推进 {np.median(dt_local)/t_pass*100:.4f}% 流过时间")
print(f"100 步累计 ~{100*np.median(dt_local)/t_pass:.4f} 个流过时间（稳态收敛通常需 100+ 个）")

print("\n=== 实测推进 20 步（验证新控制器）===")
k0 = tm.k_field.copy()
prev = None
import time
for i in range(20):
    t0 = time.time()
    res = solver.step(dt=1e-3)
    ratio = res / prev if prev else float('nan')
    kchg = np.abs(tm.k_field - k0).max() / max(k0.max(), 1e-30)
    cfl_now = solver._cfl_controller.cfl_number if getattr(solver, '_cfl_controller', None) else 0.1
    print(f"step {i+1:2d}: residual={res:.6e}  ratio={ratio:.4f}  CFL={cfl_now:.3f}  "
          f"Δk(rel)={kchg:.2e}  ({time.time()-t0:.1f}s)")
    prev = res
