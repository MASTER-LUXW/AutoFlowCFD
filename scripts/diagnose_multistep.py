"""多步诊断：定位发散起始点。

分别用 turb=none 和 turb=sst 跑 10 步，逐步对比残差增长。
"""
import sys
import numpy as np
sys.path.insert(0, "src")

from autoflowcfd.cli.solve.mesh_loader import load_mesh_for_solver
from autoflowcfd.cli.solve.wall_distance import compute_wall_distance_for_solver
from autoflowcfd.core import FRSolver

NAS_VOLUME = r"C:\Users\luxw_\Desktop\AutoFlowCFD\cube_demo\cube_demo_volume.nas"
NAS_SURFACE = r"C:\Users\luxw_\Desktop\AutoFlowCFD\cube_demo\cube_demo.nas"

print("加载网格...")
mesh, volume_data = load_mesh_for_solver(NAS_VOLUME, order=0, surface_mesh=NAS_SURFACE)
print(f"网格: {mesh.n_cells} cells")

N_STEPS = 10

for turb_name in ["none", "sst"]:
    print(f"\n{'='*70}")
    print(f"多步测试: turbulence={turb_name}, {N_STEPS} steps")
    print(f"{'='*70}")
    
    solver = FRSolver(
        mesh=mesh, backend="cpu", order=0,
        turb_model_name=turb_name, n_threads=4,
    )
    compute_wall_distance_for_solver(solver, volume_data)
    
    # 记录初始状态
    U0 = solver.state.U.copy()
    
    for step_i in range(N_STEPS):
        try:
            res = solver.step(1e-3)
            
            # 检查状态
            U = solver.state.U
            has_nan = np.any(~np.isfinite(U))
            
            # 物理量检查
            solver.state._update_primitives()
            Q = solver.state.Q
            neg_rho = np.sum(Q[:,:,0] <= 0)
            neg_p = np.sum(Q[:,:,4] <= 0)
            rho_range = f"[{Q[:,:,0].min():.4e}, {Q[:,:,0].max():.4e}]"
            p_range = f"[{Q[:,:,4].min():.4e}, {Q[:,:,4].max():.4e}]"
            vel_max = np.sqrt(Q[:,:,1]**2 + Q[:,:,2]**2 + Q[:,:,3]**2).max()
            
            # 湍流场
            turb_info = ""
            if turb_name == "sst":
                k_max = solver.turb_model.k_field.max()
                omega_max = solver.turb_model.omega_field.max()
                nu_t_max = solver.turb_model.nu_t.max() if solver.turb_model.nu_t is not None else 0
                turb_info = f" | k_max={k_max:.3e} omega_max={omega_max:.3e} nu_t_max={nu_t_max:.3e}"
            
            flags = ""
            if has_nan: flags += " NaN!"
            if neg_rho > 0: flags += f" neg_rho={neg_rho}"
            if neg_p > 0: flags += f" neg_p={neg_p}"
            
            print(f"  Step {step_i+1:2d}: res={res:.6e} | vel_max={vel_max:.2f} | "
                  f"rho={rho_range} p={p_range}{turb_info}{flags}")
            
            if has_nan or neg_rho > 100 or neg_p > 100:
                print(f"  *** 发散检测停止 ***")
                break
                
        except Exception as e:
            print(f"  Step {step_i+1}: FAILED - {e}")
            break

# ============================================================
# 额外诊断：检查 SSP-RK3 中间 stage 的行为
# ============================================================
print(f"\n{'='*70}")
print("额外诊断: 检查 SSP-RK3 中间 stage 稳定性")
print(f"{'='*70}")

solver2 = FRSolver(
    mesh=mesh, backend="cpu", order=0,
    turb_model_name="sst", n_threads=4,
)
compute_wall_distance_for_solver(solver2, volume_data)

# 手动执行 SSP-RK3 的各个 stage，检查每个 stage 后的状态
from autoflowcfd.core.fr_solver.step import step as step_func

# 先检查 dt_local 分布
dt_local = solver2._compute_local_time_step()
print(f"\ndt_local 分布:")
print(f"  min={dt_local.min():.6e}, max={dt_local.max():.6e}, mean={dt_local.mean():.6e}")
print(f"  dt_max/dt_min = {dt_local.max()/dt_local.min():.1f}")

# 检查 CFL 数（dt_local / 特征时间尺度）
volumes = mesh.get_all_cell_volumes()
n_prism = mesh.n_prism_cells
print(f"\n单元体积分布:")
print(f"  Prism: min={volumes[:n_prism].min():.6e}, max={volumes[:n_prism].max():.6e}")
print(f"  Tet:   min={volumes[n_prism:].min():.6e}, max={volumes[n_prism:].max():.6e}")

# 检查谱半径（1/dt_local * CFL）
# spectral = CFL * V / dt → 从 dt 反推
CFL = 0.1
spectral = CFL * volumes[:, None] / np.maximum(dt_local, 1e-30)
print(f"\n谱半径 (1/dt) 分布:")
print(f"  min={spectral.min():.6e}, max={spectral.max():.6e}")
print(f"  max/min ratio = {spectral.max()/spectral.min():.1f}")

# 检查壁面最近单元的 dt
wall_dist = solver2.wall_distance
near_wall = wall_dist[:, 0] < 0.01  # 距离 < 0.01m 的单元
print(f"\n近壁单元 (d<0.01m): {np.sum(near_wall)} cells")
if np.any(near_wall):
    dt_near = dt_local[near_wall, 0]
    dt_far = dt_local[~near_wall, 0]
    print(f"  近壁 dt: min={dt_near.min():.6e}, max={dt_near.max():.6e}")
    print(f"  远壁 dt: min={dt_far.min():.6e}, max={dt_far.max():.6e}")
    print(f"  近壁/远壁 dt 比 = {dt_near.min()/dt_far.min():.6e}")

print(f"\n{'='*70}")
print("诊断完成")
print(f"{'='*70}")
