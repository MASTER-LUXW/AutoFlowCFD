"""验证壁面零质量通量修复效果。

修复前：均匀流场初始残差中，壁面棱柱单元残差 ~6e9（非物理质量通量导致）
修复后：壁面残差应大幅下降（只剩压力项，均匀压力下应接近零）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from autoflowcfd.cli.solve_mesh_loader import load_mesh_for_solver


def main():
    volume_nas = r"C:\Users\luxw_\Desktop\AutoFlowCFD\cube_demo\cube_demo_volume.nas"
    surface_nas = r"C:\Users\luxw_\Desktop\AutoFlowCFD\cube_demo\cube_demo.nas"

    print("=" * 70)
    print("加载网格...")
    mesh, _vol_data = load_mesh_for_solver(
        volume_nas, order=0, surface_mesh=surface_nas,
    )
    print(f"  n_cells={mesh.n_cells}, n_prism={mesh.n_prism_cells}")

    # --- 构造求解器环境（简化版） ---
    from autoflowcfd.core.fr_solver.solver import FRSolver
    from autoflowcfd.core.fr_solver.state import FRState

    rho_inf = 1.225
    vel_inf = 33.33
    p_inf = 101325.0
    mach_ref = vel_inf / np.sqrt(1.4 * p_inf / rho_inf)

    solver = FRSolver(
        mesh=mesh, order=0, turb_model_name="SST",
        backend="cpu", rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
    )

    state = solver.state
    ghost_provider = solver.boundary_ghost_provider

    # --- 检查壁面标记 ---
    n_faces = mesh.face_connectivity.n_faces
    if hasattr(ghost_provider, 'get_zero_mass_flux_mask'):
        wall_mask = ghost_provider.get_zero_mass_flux_mask(n_faces)
        n_wall_faces = np.sum(wall_mask)
        n_bnd_faces = np.sum(mesh.face_connectivity.is_boundary)
        print(f"\n壁面标记统计:")
        print(f"  总面数: {n_faces}")
        print(f"  边界面数: {n_bnd_faces}")
        print(f"  壁面/对称面数: {n_wall_faces}")
        print(f"  壁面占比: {n_wall_faces / max(1, n_bnd_faces) * 100:.1f}%")
    else:
        print("\n  [警告] ghost_provider 没有 get_zero_mass_flux_mask 方法!")
        wall_mask = np.zeros(n_faces, dtype=bool)

    # --- 计算 P0 残差 ---
    print("\n计算 P0 无粘残差...")
    U = state.U.copy()
    from autoflowcfd.core.fr_residual.inviscid_p0 import compute_inviscid_residual_fv_p0
    residual = compute_inviscid_residual_fv_p0(U, mesh, ghost_provider, mach_ref)

    res = residual[:, 0, :]  # (n_cells, 5)

    # --- 分析结果 ---
    print("\n" + "=" * 70)
    print("修复后残差分析")
    print("=" * 70)

    # 全场 RMS
    rms = np.linalg.norm(res) / max(1, np.sqrt(res.size))
    print(f"\n全场 RMS 残差: {rms:.6e}")

    # 按变量
    for v, name in enumerate(["density", "momentum_x", "momentum_y", "momentum_z", "energy"]):
        v_rms = np.sqrt(np.mean(res[:, v] ** 2))
        print(f"  {name:15s}: RMS = {v_rms:.6e}")

    # 棱柱 vs 四面体
    n_prism = mesh.n_prism_cells
    res_prism = res[:n_prism]
    res_tet = res[n_prism:]

    prism_norm = np.sqrt(np.mean(np.sum(res_prism ** 2, axis=1)))
    tet_norm = np.sqrt(np.mean(np.sum(res_tet ** 2, axis=1)))

    print(f"\n棱柱单元 mean |res|: {prism_norm:.6e}  ({n_prism} cells)")
    print(f"四面体单元 mean |res|: {tet_norm:.6e}  ({mesh.n_cells - n_prism} cells)")
    if tet_norm > 0:
        print(f"Ratio prism/tet: {prism_norm / tet_norm:.2e}")

    # Top-10 残差细胞
    cell_norms = np.sqrt(np.sum(res ** 2, axis=1))
    top10_idx = np.argsort(cell_norms)[-10:][::-1]
    print(f"\nTop-10 残差细胞:")
    for i, idx in enumerate(top10_idx):
        is_prism = "PRISM" if idx < n_prism else "TET"
        print(f"  #{i+1}: cell {idx:8d} ({is_prism:5s}), |res|={cell_norms[idx]:.6e}, "
              f"vol={mesh.cell_volumes[idx]:.4e}")

    # 壁面细胞 vs 内部细胞
    bnd_cells = set()
    for f in range(n_faces):
        if mesh.face_connectivity.is_boundary[f]:
            bnd_cells.add(int(mesh.face_connectivity.owner_cell[f]))
    bnd_mask_cells = np.zeros(n_cells, dtype=bool)
    for c in bnd_cells:
        bnd_mask_cells[c] = True

    bnd_norm = np.sqrt(np.mean(np.sum(res[bnd_mask_cells] ** 2, axis=1)))
    int_norm = np.sqrt(np.mean(np.sum(res[~bnd_mask_cells] ** 2, axis=1)))
    print(f"\n边界细胞 mean |res|: {bnd_norm:.6e}  ({np.sum(bnd_mask_cells)} cells)")
    print(f"内部细胞 mean |res|: {int_norm:.6e}  ({np.sum(~bnd_mask_cells)} cells)")

    print("\n" + "=" * 70)
    print("验证标准:")
    print("  - 均匀流场初始残差应接近机器精度 (~1e-10 或更小)")
    print("  - 棱柱/四面体残差比应接近 1（而非修复前的 1e14）")
    print("  - 如果残差仍然很大，说明修复不完整")
    print("=" * 70)


if __name__ == "__main__":
    main()
