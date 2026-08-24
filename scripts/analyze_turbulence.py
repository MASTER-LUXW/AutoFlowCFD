"""深入分析 SST 湍流变量 k/omega 的空间分布和增长模式。"""
import sys, os
sys.path.insert(0, r"d:\myWorkspace\AutoFlowCFD\src")

import numpy as np
import h5py

ckpt_dir = r"C:\Users\luxw_\Desktop\AutoFlowCFD\cube_demo\steady_results\checkpoints"

# 对比 iter 100 vs 1500 的 k/omega 空间分布
for it in [100, 500, 1000, 1500]:
    path = os.path.join(ckpt_dir, f"checkpoint_iter_{it:06d}.h5")
    with h5py.File(path, "r") as f:
        k_field = f["solution/k_field"][:].ravel()
        omega_field = f["solution/omega_field"][:].ravel()
        nu_t = f["solution/nu_t"][:].ravel()
        conserved = f["solution/conserved"][:]
        rho = conserved[:, 0]
        rho_k = conserved[:, 5]
        rho_omega = conserved[:, 6]
        
        print(f"\n=== Iter {it} ===")
        print(f"  k_field:  min={np.min(k_field):.4e}, max={np.max(k_field):.4e}, "
              f"mean={np.mean(k_field):.4e}, median={np.median(k_field):.4e}")
        print(f"  omega:    min={np.min(omega_field):.4e}, max={np.max(omega_field):.4e}, "
              f"mean={np.mean(omega_field):.4e}, median={np.median(omega_field):.4e}")
        print(f"  nu_t:     min={np.min(nu_t):.4e}, max={np.max(nu_t):.4e}, "
              f"mean={np.mean(nu_t):.4e}")
        
        # k 的分布：多少单元超过阈值
        thresholds_k = [1e-3, 1e0, 1e3, 1e10, 1e50, 1e100, 1e200, 1e300]
        for th in thresholds_k:
            n = np.sum(k_field > th)
            if n > 0:
                print(f"    k > {th:.0e}: {n} cells ({100*n/len(k_field):.2f}%)")
        
        # 高 k 单元的位置特征
        high_k_mask = k_field > 1e10
        if np.any(high_k_mask):
            high_k_rho = rho[high_k_mask]
            print(f"  高k单元 ({np.sum(high_k_mask)} 个): rho range [{np.min(high_k_rho):.4f}, {np.max(high_k_rho):.4f}]")
        
        # 检查 k_field 与 conserved rho_k 的关系
        # k_field 是原始变量，rho_k = rho * k_field
        k_from_cons = rho_k / np.maximum(rho, 1e-30)
        max_diff = np.max(np.abs(k_field - k_from_cons))
        print(f"  k_field vs rho_k/rho 最大差异: {max_diff:.4e}")
        
        # omega 分布
        thresholds_omega = [1e0, 1e3, 1e10, 1e50, 1e100, 1e200, 1e300]
        for th in thresholds_omega:
            n = np.sum(omega_field > th)
            if n > 0:
                print(f"    omega > {th:.0e}: {n} cells ({100*n/len(omega_field):.2f}%)")

# 检查 Q_sps 中的 k/omega（解点值 vs 单元均值）
print("\n\n=== Q_sps vs k_field 对比 (iter 100) ===")
with h5py.File(os.path.join(ckpt_dir, "checkpoint_iter_000100.h5"), "r") as f:
    Q = f["solution/Q_sps"][:]  # (n_cells, n_sps, 7)
    k_sps = Q[:, :, 5]  # k at solution points
    k_cell = f["solution/k_field"][:].ravel()  # (n_cells,)
    print(f"  Q_sps k: min={np.min(k_sps):.4e}, max={np.max(k_sps):.4e}, mean={np.mean(k_sps):.4e}")
    print(f"  k_field: min={np.min(k_cell):.4e}, max={np.max(k_cell):.4e}, mean={np.mean(k_cell):.4e}")
    # P0 应该两者一致
    print(f"  差异: max={np.max(np.abs(k_sps[:,0] - k_cell)):.4e}")
