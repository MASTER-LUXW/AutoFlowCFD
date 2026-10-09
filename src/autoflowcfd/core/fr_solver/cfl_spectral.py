"""局部 CFL 步长的对流谱半径：逐解点的面求和（CPU 与 CPU MPI 共用；GPU 版见
`core/gpu/gpu_time_integration.py::compute_local_cfl_step_gpu`，同一公式）。

    S[c, sp] = sum_{f in faces(c)} (|u[c, sp] . n_f| + c_s[c, sp]) A_f

`c_s` 是物理声速或低马赫预处理后的有效声速（由调用方给出）。每个解点一列，列与列之间无耦合，按解点并行。
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def face_spectral_sum_kernel(u, v, w, sound, owner, neighbor, is_boundary, unit_normal, area):
    """见模块文档。`u/v/w/sound`: (n_cells, n_sps)；面数组长度 n_faces；返回 (n_cells, n_sps)。"""
    n_cells, n_sps = sound.shape
    n_faces = owner.shape[0]
    out = np.zeros((n_sps, n_cells))
    for sp in prange(n_sps):
        col = out[sp]
        for f in range(n_faces):
            nx, ny, nz, a_f = unit_normal[f, 0], unit_normal[f, 1], unit_normal[f, 2], area[f]
            o = owner[f]
            col[o] += (abs(u[o, sp] * nx + v[o, sp] * ny + w[o, sp] * nz) + sound[o, sp]) * a_f
            if not is_boundary[f]:
                nb = neighbor[f]
                col[nb] += (abs(u[nb, sp] * nx + v[nb, sp] * ny + w[nb, sp] * nz) + sound[nb, sp]) * a_f
    return np.ascontiguousarray(out.T)
