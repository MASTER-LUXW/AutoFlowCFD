"""最小化测试：验证 P0 kernel 的 wall_mask 是否生效"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ['NUMBA_CACHE'] = '0'

import numpy as np
from autoflowcfd.core.fr_residual.inviscid_p0_kernel import _p0_inviscid_kernel
from autoflowcfd.core.fr_operators.kernels import compute_ausm_up_flux

# 简单测试：两个单元，一个面
# 单元 0 (内部) 和单元 1 (壁面幽灵)
n_cells = 2
n_faces = 1

owner_cell = np.array([0], dtype=np.int64)
neighbor_cell = np.array([-1], dtype=np.int64)  # 边界面
is_boundary = np.array([True])
unit_normals = np.array([[1.0, 0.0, 0.0]])  # 法向沿 x
area_weights = np.array([1.0])  # 单位面积

# 内部态：自由来流
Q_all = np.array([
    [1.225, 33.33, 0.0, 0.0, 101325.0],  # cell 0: interior
    [1.225, 33.33, 0.0, 0.0, 101325.0],  # cell 1: unused for boundary
], dtype=np.float64)

# 幽灵态：速度反号 (壁面)
Q_ghost = np.array([
    [1.225, -33.33, 0.0, 0.0, 101325.0],  # cell 0 ghost: velocity mirrored
], dtype=np.float64)

cell_volumes = np.array([1e-6, 1e-6], dtype=np.float64)
mach_ref = 0.097

# 测试 1：wall_mask = False (不修复)
wall_mask_off = np.array([False])
res_off = _p0_inviscid_kernel(
    owner_cell, neighbor_cell, is_boundary,
    unit_normals, area_weights,
    Q_all, Q_ghost, cell_volumes,
    n_cells, 1, mach_ref,
    wall_mask_off,
)
print("wall_mask=False (修复前):")
print(f"  residual[0] = {res_off[0, 0]}")
print(f"  mass_flux (density) = {res_off[0, 0, 0]:.6e}")
print(f"  energy flux = {res_off[0, 0, 4]:.6e}")

# 测试 2：wall_mask = True (修复后)
wall_mask_on = np.array([True])
res_on = _p0_inviscid_kernel(
    owner_cell, neighbor_cell, is_boundary,
    unit_normals, area_weights,
    Q_all, Q_ghost, cell_volumes,
    n_cells, 1, mach_ref,
    wall_mask_on,
)
print("\nwall_mask=True (修复后):")
print(f"  residual[0] = {res_on[0, 0]}")
print(f"  mass_flux (density) = {res_on[0, 0, 0]:.6e}")
print(f"  energy flux = {res_on[0, 0, 4]:.6e}")

# 对比
print("\n对比:")
print(f"  density 变化: {res_off[0,0,0]:.6e} -> {res_on[0,0,0]:.6e}")
print(f"  energy 变化:  {res_off[0,0,4]:.6e} -> {res_on[0,0,4]:.6e}")

if abs(res_on[0, 0, 0]) < 1e-15 and abs(res_on[0, 0, 4]) < 1e-15:
    print("\n✅ kernel 修复生效：质量和能量通量已归零")
else:
    print("\n❌ kernel 修复未生效！")

# 也验证 AUSM+up 直接计算
print("\n--- AUSM+up 直接验证 ---")
Q_L = np.array([1.225, 33.33, 0.0, 0.0, 101325.0])
Q_R = np.array([1.225, -33.33, 0.0, 0.0, 101325.0])
normal = np.array([1.0, 0.0, 0.0])
flux = compute_ausm_up_flux(Q_L, Q_R, normal, mach_ref)
print(f"AUSM+up flux (no-slip wall, un_L=33.33): {flux}")
print(f"  mass_flux = {flux[0]:.6e} (应该为 0，但 AUSM+up 给出非零值)")

Q_L2 = np.array([1.225, 0.0, 0.0, 0.0, 101325.0])
Q_R2 = np.array([1.225, 0.0, 0.0, 0.0, 101325.0])
flux2 = compute_ausm_up_flux(Q_L2, Q_R2, normal, mach_ref)
print(f"\nAUSM+up flux (un_L=0): {flux2}")
print(f"  mass_flux = {flux2[0]:.6e} (应该为 0)")
