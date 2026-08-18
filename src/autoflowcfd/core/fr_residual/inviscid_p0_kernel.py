"""
AutoFlowCFD V2.0 - P0 有限体积无粘残差 numba 并行 kernel

从 inviscid_p0.py 拆出。将原纯 Python 逐面循环（188 万个面，每步迭代
~25s）替换为 numba `@njit(parallel=True)` + `prange` 并行 kernel，
与 P≥1 界面项 kernel (inviscid_kernel.py) 一致的加速策略。

性能目标：791K 单元 / 188 万面网格上，从 ~25s/次 降至 ~1-2s/次。

算法与原 inviscid_p0.py 完全一致：
- 每个面：AUSM+up 黎曼求解 → 面积加权通量 → scatter-add 到 owner/neighbor
- 边界面：ghost_provider 在 kernel 外部预计算为 Q_ghost 数组
- per-thread buffer 避免写冲突（同 P≥1 kernel 方案）

验证：与原 Python 路径逐位对比（残差 L∞ 差 < 1e-14，纯浮点重排误差）。
"""

import numpy as np
from numba import njit, prange, get_thread_id

from autoflowcfd.core.fr_operators.kernels import compute_ausm_up_flux


@njit(cache=True, parallel=True)
def _p0_inviscid_kernel(
    owner_cell,       # int64 (n_faces,)
    neighbor_cell,    # int64 (n_faces,)，边界面为 -1
    is_boundary,      # bool (n_faces,)
    unit_normals,     # float64 (n_faces, 3)，单位法向量
    area_weights,     # float64 (n_faces,)，面积权重
    Q_all,            # float64 (n_cells, 5)，原始变量
    Q_ghost,          # float64 (n_cells, 5)，预计算的边界幽灵态
    cell_volumes,     # float64 (n_cells,)
    n_cells,
    n_threads,
):
    """P0 有限体积无粘残差并行 kernel。

    对每个面执行 AUSM+up 黎曼求解，将面积加权通量 scatter-add 到
    owner/neighbor 单元的 per-thread buffer（避免写冲突），
    最后由调用方归约得到最终残差。

    per-thread buffer 内存：n_threads × n_cells × 5 × 8 bytes
    4 threads × 791K × 5 × 8 ≈ 126 MB（可接受）
    """
    n_faces = owner_cell.shape[0]

    # per-thread buffer，零初始化
    residual_per_thread = np.zeros((n_threads, n_cells, 5), dtype=np.float64)

    for f in prange(n_faces):
        tid = get_thread_id()
        oc = owner_cell[f]

        # Owner 状态（5 个分量逐元素复制，避免 numba 对切片赋值的限制）
        Q_o = np.empty(5, dtype=np.float64)
        for v in range(5):
            Q_o[v] = Q_all[oc, v]

        # Neighbor / 幽灵态
        if is_boundary[f]:
            Q_n = np.empty(5, dtype=np.float64)
            for v in range(5):
                Q_n[v] = Q_ghost[oc, v]
        else:
            nc = neighbor_cell[f]
            Q_n = np.empty(5, dtype=np.float64)
            for v in range(5):
                Q_n[v] = Q_all[nc, v]

        # 法向量（unit normal）
        normal = np.empty(3, dtype=np.float64)
        normal[0] = unit_normals[f, 0]
        normal[1] = unit_normals[f, 1]
        normal[2] = unit_normals[f, 2]

        # AUSM+up 黎曼求解（返回单位面积通量）
        F_common_n = compute_ausm_up_flux(Q_o, Q_n, normal)

        # 面积加权通量
        aw = area_weights[f]
        flux0 = F_common_n[0] * aw
        flux1 = F_common_n[1] * aw
        flux2 = F_common_n[2] * aw
        flux3 = F_common_n[3] * aw
        flux4 = F_common_n[4] * aw

        # Owner：通量流出为正（法向量由 owner 指向 neighbor）
        inv_vol_o = 1.0 / cell_volumes[oc]
        residual_per_thread[tid, oc, 0] -= flux0 * inv_vol_o
        residual_per_thread[tid, oc, 1] -= flux1 * inv_vol_o
        residual_per_thread[tid, oc, 2] -= flux2 * inv_vol_o
        residual_per_thread[tid, oc, 3] -= flux3 * inv_vol_o
        residual_per_thread[tid, oc, 4] -= flux4 * inv_vol_o

        # Neighbor：通量流入（符号相反）
        if not is_boundary[f]:
            nc = neighbor_cell[f]
            inv_vol_n = 1.0 / cell_volumes[nc]
            residual_per_thread[tid, nc, 0] += flux0 * inv_vol_n
            residual_per_thread[tid, nc, 1] += flux1 * inv_vol_n
            residual_per_thread[tid, nc, 2] += flux2 * inv_vol_n
            residual_per_thread[tid, nc, 3] += flux3 * inv_vol_n
            residual_per_thread[tid, nc, 4] += flux4 * inv_vol_n

    return residual_per_thread
