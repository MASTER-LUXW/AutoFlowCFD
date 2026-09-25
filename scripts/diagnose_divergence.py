"""诊断 P0 阶段第1步发散原因。

隔离三个可能的发散源：
1. 平均流（无粘）残差本身不稳定
2. SST 湍流源项刚性导致 k/omega 爆炸
3. CFL 时间步长仍然过大
"""
import sys
import numpy as np
sys.path.insert(0, "src")

from autoflowcfd.cli.solve.mesh_loader import load_mesh_for_solver
from autoflowcfd.cli.solve.wall_distance import compute_wall_distance_for_solver
from autoflowcfd.core import FRSolver
from autoflowcfd.core.fr_solver.state import FRState
from autoflowcfd.fr.operators import generate_fr_operators

NAS_VOLUME = r"C:\Users\luxw_\Desktop\AutoFlowCFD\cube_demo\cube_demo_volume.nas"
NAS_SURFACE = r"C:\Users\luxw_\Desktop\AutoFlowCFD\cube_demo\cube_demo.nas"

print("=" * 70)
print("Step 0: 加载网格")
print("=" * 70)
mesh, volume_data = load_mesh_for_solver(NAS_VOLUME, order=0, surface_mesh=NAS_SURFACE)
ops = generate_fr_operators(0)

print(f"\n网格加载完成: {mesh.n_cells} cells, P0 (1 SP/cell)")

# ============================================================
# Test 1: 无湍流模型 — 隔离平均流稳定性
# ============================================================
print("\n" + "=" * 70)
print("Test 1: 无湍流模型 (turbulence=none)")
print("=" * 70)

solver_none = FRSolver(
    mesh=mesh, backend="cpu", order=0,
    turb_model_name="none", n_threads=4,
)
compute_wall_distance_for_solver(solver_none, volume_data)

# 检查 CFL 分布
dt_local = solver_none._compute_local_time_step()
print(f"\nCFL dt 分布 (turb=none):")
print(f"  min  = {dt_local.min():.6e}")
print(f"  max  = {dt_local.max():.6e}")
print(f"  mean = {dt_local.mean():.6e}")
print(f"  median = {np.median(dt_local):.6e}")
print(f"  P1   = {np.percentile(dt_local, 1):.6e}")
print(f"  P5   = {np.percentile(dt_local, 5):.6e}")

# 初始状态
U0 = solver_none.state.U.copy()
print(f"\n初始状态 U 范围:")
print(f"  rho: [{U0[:,:,0].min():.6e}, {U0[:,:,0].max():.6e}]")
print(f"  rho*u: [{U0[:,:,1].min():.6e}, {U0[:,:,1].max():.6e}]")
print(f"  rho*v: [{U0[:,:,2].min():.6e}, {U0[:,:,2].max():.6e}]")
print(f"  rho*w: [{U0[:,:,3].min():.6e}, {U0[:,:,3].max():.6e}]")
print(f"  rho*E: [{U0[:,:,4].min():.6e}, {U0[:,:,4].max():.6e}]")

# 计算初始残差（不推进状态）
solver_none.state._update_primitives()
from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr
res0 = compute_inviscid_residual_fr(
    solver_none.state.U, solver_none.mesh, solver_none.ops,
    boundary_ghost_provider=solver_none.boundary_ghost_provider,
    mach_ref=solver_none.freestream["mach_ref"],
)
print(f"\n初始无粘残差 (step 之前):")
print(f"  shape: {res0.shape}")
print(f"  L2 norm: {np.linalg.norm(res0):.6e}")
print(f"  RMS norm: {np.linalg.norm(res0) / np.sqrt(res0.size):.6e}")
print(f"  max |res|: {np.max(np.abs(res0)):.6e}")
print(f"  per-variable RMS:")
for v in range(5):
    rms = np.sqrt(np.mean(res0[:,:,v]**2))
    print(f"    var[{v}]: {rms:.6e}")

# 跑一步
print("\n--- 执行第 1 步 (turb=none) ---")
try:
    res_after = solver_none.step(1e-3)
    print(f"  残差范数: {res_after:.6e}")
    
    U1 = solver_none.state.U.copy()
    dU = U1 - U0
    print(f"\n状态变化 dU = U1 - U0:")
    for v in range(5):
        print(f"    var[{v}]: max|dU|={np.max(np.abs(dU[:,:,v])):.6e}, "
              f"RMS={np.sqrt(np.mean(dU[:,:,v]**2)):.6e}")
    
    # 检查是否有 NaN/Inf
    has_nan = np.any(np.isnan(U1))
    has_inf = np.any(np.isinf(U1))
    print(f"\n  NaN: {has_nan}, Inf: {has_inf}")
    
    # 检查物理量
    solver_none.state._update_primitives()
    Q = solver_none.state.Q
    print(f"\n  更新后物理量范围:")
    print(f"    rho: [{Q[:,:,0].min():.6e}, {Q[:,:,0].max():.6e}]")
    vel_mag = np.sqrt(Q[:,:,1]**2 + Q[:,:,2]**2 + Q[:,:,3]**2)
    print(f"    |vel|: [{vel_mag.min():.6e}, {vel_mag.max():.6e}]")
    print(f"    p: [{Q[:,:,4].min():.6e}, {Q[:,:,4].max():.6e}]")
    
    # 负压力/密度检查
    neg_rho = np.sum(Q[:,:,0] <= 0)
    neg_p = np.sum(Q[:,:,4] <= 0)
    print(f"\n  负密度单元数: {neg_rho}")
    print(f"  负压力单元数: {neg_p}")
    
except Exception as e:
    print(f"  步骤失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Test 2: SST 湍流模型 — 检查湍流源项是否主导发散
# ============================================================
print("\n" + "=" * 70)
print("Test 2: SST 湍流模型 (turbulence=sst)")
print("=" * 70)

solver_sst = FRSolver(
    mesh=mesh, backend="cpu", order=0,
    turb_model_name="sst", n_threads=4,
)
compute_wall_distance_for_solver(solver_sst, volume_data)

# 检查初始湍流场
print(f"\n初始湍流场:")
print(f"  k:     [{solver_sst.turb_model.k_field.min():.6e}, {solver_sst.turb_model.k_field.max():.6e}]")
print(f"  omega: [{solver_sst.turb_model.omega_field.min():.6e}, {solver_sst.turb_model.omega_field.max():.6e}]")
if hasattr(solver_sst.turb_model, 'nu_t') and solver_sst.turb_model.nu_t is not None:
    print(f"  nu_t:  [{solver_sst.turb_model.nu_t.min():.6e}, {solver_sst.turb_model.nu_t.max():.6e}]")
else:
    print(f"  nu_t:  None or not initialized")

# CFL 分布
dt_local_sst = solver_sst._compute_local_time_step()
print(f"\nCFL dt 分布 (turb=sst):")
print(f"  min  = {dt_local_sst.min():.6e}")
print(f"  max  = {dt_local_sst.max():.6e}")
print(f"  mean = {dt_local_sst.mean():.6e}")

# 保存初始湍流场
k0 = solver_sst.turb_model.k_field.copy()
omega0 = solver_sst.turb_model.omega_field.copy()

# 跑一步
print("\n--- 执行第 1 步 (turb=sst) ---")
try:
    res_sst = solver_sst.step(1e-3)
    print(f"  残差范数: {res_sst:.6e}")
    
    k1 = solver_sst.turb_model.k_field
    omega1 = solver_sst.turb_model.omega_field
    print(f"\n湍流场变化:")
    print(f"  k:     [{k1.min():.6e}, {k1.max():.6e}]")
    print(f"  omega: [{omega1.min():.6e}, {omega1.max():.6e}]")
    dk = k1 - k0
    domega = omega1 - omega0
    print(f"  dk max:     {np.max(np.abs(dk)):.6e}, RMS: {np.sqrt(np.mean(dk**2)):.6e}")
    print(f"  domega max: {np.max(np.abs(domega)):.6e}, RMS: {np.sqrt(np.mean(domega**2)):.6e}")
    print(f"  omega 放大倍数: max={omega1.max()/max(omega0.max(),1e-30):.1f}x")
    
    if hasattr(solver_sst.turb_model, 'nu_t') and solver_sst.turb_model.nu_t is not None:
        print(f"  nu_t:  [{solver_sst.turb_model.nu_t.min():.6e}, {solver_sst.turb_model.nu_t.max():.6e}]")
    
    has_nan_sst = np.any(np.isnan(solver_sst.state.U))
    has_inf_sst = np.any(np.isinf(solver_sst.state.U))
    print(f"\n  NaN: {has_nan_sst}, Inf: {has_inf_sst}")
    
except Exception as e:
    print(f"  步骤失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Test 3: 检查 dt 与单元体积的关系
# ============================================================
print("\n" + "=" * 70)
print("Test 3: dt 与单元类型/体积的关系")
print("=" * 70)

volumes = mesh.get_all_cell_volumes()
n_prism = mesh.n_prism_cells
n_tet = mesh.n_cells - n_prism

print(f"\n单元统计: {n_prism} prism + {n_tet} tet = {mesh.n_cells} total")

# prism vs tet 体积分布
v_prism = volumes[:n_prism]
v_tet = volumes[n_prism:]
print(f"\nPrism 体积: min={v_prism.min():.6e}, max={v_prism.max():.6e}, mean={v_prism.mean():.6e}")
print(f"Tet   体积: min={v_tet.min():.6e}, max={v_tet.max():.6e}, mean={v_tet.mean():.6e}")

# dt vs 体积
dt_flat = dt_local[:, 0]  # P0 only has 1 SP
dt_prism = dt_flat[:n_prism]
dt_tet = dt_flat[n_prism:]
print(f"\nPrism dt: min={dt_prism.min():.6e}, max={dt_prism.max():.6e}, mean={dt_prism.mean():.6e}")
print(f"Tet   dt: min={dt_tet.min():.6e}, max={dt_tet.max():.6e}, mean={dt_tet.mean():.6e}")

# 最小 dt 的单元位置
min_dt_idx = np.argmin(dt_flat)
min_dt_cell_type = "prism" if min_dt_idx < n_prism else "tet"
print(f"\n最小 dt 单元: index={min_dt_idx}, type={min_dt_cell_type}")
print(f"  dt = {dt_flat[min_dt_idx]:.6e}")
print(f"  volume = {volumes[min_dt_idx]:.6e}")
print(f"  V^(1/3) = {volumes[min_dt_idx]**(1/3):.6e}")

# 最大 dt 的单元位置
max_dt_idx = np.argmax(dt_flat)
max_dt_cell_type = "prism" if max_dt_idx < n_prism else "tet"
print(f"\n最大 dt 单元: index={max_dt_idx}, type={max_dt_cell_type}")
print(f"  dt = {dt_flat[max_dt_idx]:.6e}")
print(f"  volume = {volumes[max_dt_idx]:.6e}")

# dt 分布直方图（对数尺度）
print(f"\ndt 对数分布:")
log_dt = np.log10(dt_flat)
for lo in range(-10, 0):
    hi = lo + 1
    count = np.sum((log_dt >= lo) & (log_dt < hi))
    if count > 0:
        bar = "#" * min(count // 100, 60)
        print(f"  [1e{lo}, 1e{hi}): {count:6d} {bar}")

print("\n" + "=" * 70)
print("诊断完成")
print("=" * 70)
