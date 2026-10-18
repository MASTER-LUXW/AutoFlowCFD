"""
AutoFlowCFD V2.0 - P0 有限体积无粘残差 numba 并行 kernel

从 inviscid_p0.py 拆出。将原纯 Python 逐面循环（188 万个面，每次残差求值
~25s）替换为 numba `@njit(parallel=True)` + `prange` 并行 kernel，
与 P≥1 界面项 kernel (inviscid_kernel.py) 一致的加速策略。

性能目标：791K 单元 / 188 万面网格上，从 ~25s/次 降至 ~1-2s/次。

算法与原 inviscid_p0.py 完全一致：
- 每个面：AUSM+up 黎曼求解 → 面积加权通量 → scatter-add 到 owner/neighbor
- 边界面：ghost_provider 在 kernel 外部预计算为 Q_ghost 数组
- 按面着色：调用方逐颜色调用，每次只处理一个颜色组 `face_indices`（同色面的 owner/neighbor 单元互不重复，`fr_operators/face_kernels.py` 的面着色），组内 `prange` 直接累加进共享输出，没有写冲突、不需要每线程缓冲与归约。与 P>=1 界面核同一套颜色分组；2026-10-18 以前这里是每线程缓冲版（内存 n_threads*n_cells*5）。

验证：与原 Python 路径逐位对比（残差 L∞ 差 < 1e-14，纯浮点重排误差）。
"""

import numpy as np
from numba import njit, prange

from autoflowcfd.core.fr_operators.kernels import compute_ausm_up_flux


@njit(cache=True, parallel=True)
def _p0_inviscid_kernel(
    owner_cell,       # int64 (n_faces,)
    neighbor_cell,    # int64 (n_faces,)，边界面为 -1
    is_boundary,      # bool (n_faces,)
    unit_normals,     # float64 (n_faces, 3)，单位法向量
    area_weights,     # float64 (n_faces,)，面积权重
    Q_all,            # float64 (n_cells, 5)，原始变量
    Q_ghost,          # float64 (n_faces, 5)，预计算的幽灵态（按面索引，
                      # 不是按 owner 单元索引——见 inviscid_p0.py::
                      # _precompute_ghost_states 文档"真实 bug 修复"一节）
    cell_volumes,     # float64 (n_cells,)
    face_indices,     # int64，本颜色组的面
    residual,         # float64 (n_cells, 5)，就地累加
    mach_ref,         # float，见 kernels.py::compute_ausm_up_flux 文档
) -> None:
    """P0 有限体积无粘残差并行 kernel。

    对颜色组里的每个面执行 AUSM+up 黎曼求解，把面积加权通量累加到 owner/neighbor 单元的残差。

    按面着色：调用方逐颜色调用，每次只处理一个颜色组 `face_indices`（同色面的 owner/neighbor 单元互不重复，`fr_operators/face_kernels.py` 的面着色），组内 `prange` 直接累加进共享输出，没有写冲突、不需要每线程缓冲与归约。与 P>=1 界面核同一套颜色分组；2026-10-18 以前这里是每线程缓冲版（内存 n_threads*n_cells*5）。
    """
    for fi in prange(face_indices.shape[0]):
        f = face_indices[fi]
        oc = owner_cell[f]

        # Owner 状态（5 个分量逐元素复制，避免 numba 对切片赋值的限制）
        Q_o = np.empty(5, dtype=np.float64)
        for v in range(5):
            Q_o[v] = Q_all[oc, v]

        # Neighbor / 幽灵态
        if is_boundary[f]:
            Q_n = np.empty(5, dtype=np.float64)
            for v in range(5):
                Q_n[v] = Q_ghost[f, v]
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
        F_common_n = compute_ausm_up_flux(Q_o, Q_n, normal, mach_ref)

        # 混合拆分面的两半在面几何里已经是两个独立的面（inviscid_p0.py::_extract_p0_face_geometry），这里不另做处理

        # 面积加权通量
        aw = area_weights[f]
        flux0 = F_common_n[0] * aw
        flux1 = F_common_n[1] * aw
        flux2 = F_common_n[2] * aw
        flux3 = F_common_n[3] * aw
        flux4 = F_common_n[4] * aw

        # Owner：通量流出为正（法向量由 owner 指向 neighbor）
        inv_vol_o = 1.0 / cell_volumes[oc]
        residual[oc, 0] -= flux0 * inv_vol_o
        residual[oc, 1] -= flux1 * inv_vol_o
        residual[oc, 2] -= flux2 * inv_vol_o
        residual[oc, 3] -= flux3 * inv_vol_o
        residual[oc, 4] -= flux4 * inv_vol_o

        # Neighbor：通量流入（符号相反）
        if not is_boundary[f]:
            nc = neighbor_cell[f]
            inv_vol_n = 1.0 / cell_volumes[nc]
            residual[nc, 0] += flux0 * inv_vol_n
            residual[nc, 1] += flux1 * inv_vol_n
            residual[nc, 2] += flux2 * inv_vol_n
            residual[nc, 3] += flux3 * inv_vol_n
            residual[nc, 4] += flux4 * inv_vol_n

