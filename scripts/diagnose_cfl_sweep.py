"""CFL 扫描诊断：测试不同 CFL 下的稳定性。

分别用 CFL=0.1, 0.05, 0.01, 0.005 跑 5 步，定位稳定/不稳定边界。
同时检查 positivity limiter 的触发频率和程度。
"""
import sys
import numpy as np
sys.path.insert(0, "src")

# Monkey-patch CFL before importing solver
import autoflowcfd.core.fr_solver.cfl as cfl_module
_original_compute = cfl_module.compute_local_time_step

def patched_compute(solver):
    dt = _original_compute(solver)
    return dt * _cfl_scale

_cfl_scale = 1.0
cfl_module.compute_local_time_step = patched_compute

from autoflowcfd.cli.solve.mesh_loader import load_mesh_for_solver
from autoflowcfd.cli.solve.wall_distance import compute_wall_distance_for_solver
from autoflowcfd.core import FRSolver

NAS_VOLUME = r"C:\Users\luxw_\Desktop\AutoFlowCFD\cube_demo\cube_demo_volume.nas"
NAS_SURFACE = r"C:\Users\luxw_\Desktop\AutoFlowCFD\cube_demo\cube_demo.nas"

print("加载网格...")
mesh, volume_data = load_mesh_for_solver(NAS_VOLUME, order=0, surface_mesh=NAS_SURFACE)
print(f"网格: {mesh.n_cells} cells")

N_STEPS = 5
CFL_VALUES = [0.1, 0.05, 0.01, 0.005]

for cfl_target in CFL_VALUES:
    _cfl_scale = cfl_target / 0.1  # 原始 CFL=0.1，按比例缩放
    
    print(f"\n{'='*70}")
    print(f"CFL = {cfl_target} (scale={_cfl_scale})")
    print(f"{'='*70}")
    
    solver = FRSolver(
        mesh=mesh, backend="cpu", order=0,
        turb_model_name="none", n_threads=4,
    )
    compute_wall_distance_for_solver(solver, volume_data)
    
    # 检查实际 dt 分布
    dt_local = solver._compute_local_time_step()
    print(f"  dt: min={dt_local.min():.3e}, max={dt_local.max():.3e}, "
          f"mean={dt_local.mean():.3e}")
    
    U_init = solver.state.U.copy()
    
    for step_i in range(N_STEPS):
        U_before = solver.state.U.copy()
        res = solver.step(1e-3)
        
        U_after = solver.state.U.copy()
        dU = U_after - U_before
        
        # 物理量
        solver.state._update_primitives()
        Q = solver.state.Q
        neg_rho = int(np.sum(Q[:,:,0] <= 0))
        neg_p = int(np.sum(Q[:,:,4] <= 0))
        rho_min = Q[:,:,0].min()
        rho_max = Q[:,:,0].max()
        p_min = Q[:,:,4].min()
        p_max = Q[:,:,4].max()
        vel_max = np.sqrt(Q[:,:,1]**2 + Q[:,:,2]**2 + Q[:,:,3]**2).max()
        
        # positivity limiter 触发统计
        rho_clipped = int(np.sum(U_after[:, 0] < U_before[:, 0] * 0.5))  # 密度下降>50%
        has_nan = not np.all(np.isfinite(U_after))
        
        status = "OK"
        if has_nan: status = "NaN!"
        elif neg_rho > 0: status = f"neg_rho={neg_rho}"
        elif rho_clipped > 0: status = f"rho_clip={rho_clipped}"
        
        print(f"  Step {step_i+1}: res={res:.3e} | rho=[{rho_min:.3e},{rho_max:.3e}] "
              f"p=[{p_min:.1e},{p_max:.1e}] vel_max={vel_max:.1f} | {status}")
        
        if has_nan or (not np.all(np.isfinite(U_after))):
            print(f"  *** 发散 ***")
            break

print(f"\n{'='*70}")
print("诊断完成")
print(f"{'='*70}")
