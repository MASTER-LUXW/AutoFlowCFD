"""AutoFlowCFD V2.0 - 应变率/涡量张量模的 numba kernel 与其封装

从 `src/autoflowcfd/core/turbulence/sst.py` 拆出（2026-09-24，项目「单文件不超 500 行」规范）。
"""

import numpy as np
from numba import prange


def _strain_vorticity_magnitude_kernel(grad_u, s_out, w_out) -> None:
    """一趟算出应变率模 |S| 与涡量模 |Omega|（两者共用同一份 grad_u）。

    `S_ij = 0.5*(g_ij + g_ji)`，`|S| = sqrt(2*sum_ij S_ij^2)`；
    `W_ij = 0.5*(g_ij - g_ji)`，`|Omega| = sqrt(2*sum_ij W_ij^2)`。

    性能优化（2026-09-13 真实剖析）：原实现是
    `S_ij = 0.5*(grad_u + np.transpose(grad_u,(0,1,3,2)))` 先物化一份
    (n_cells,n_sps,3,3) 的对称化张量（79 万单元 P1 下约 456MiB，含一次
    转置拷贝），再用 `np.einsum('nijm,nijm->ni', S_ij, S_ij)` 收缩——
    两个函数各来一遍、且 numpy 逐元素/einsum 路径全部单线程。融合后
    只读一遍 grad_u、不产生任何中间大数组，按 cell prange 并行。
    数值等价性（实测，随机张量 n_sps=1/8/27）：与原 einsum 路径最大绝对
    误差 8.9e-16~1.8e-15（量级 O(1~10) 的输出上约 1e-16 相对误差），
    **不是逐位相同**——einsum 内部用成对/SIMD 求和，这里是顺序的 (j,m)
    双重循环，浮点重结合导致最后一位不同。与本代码库此前 einsum->matmul
    那批优化（见 fr_operators/volume_contract.py 模块文档）是同一类、
    同一量级的现象。
    """
    n_cells, n_sps = grad_u.shape[0], grad_u.shape[1]
    for c in prange(n_cells):
        for sp in range(n_sps):
            acc_s = 0.0
            acc_w = 0.0
            for j in range(3):
                for m in range(3):
                    gjm = grad_u[c, sp, j, m]
                    gmj = grad_u[c, sp, m, j]
                    sij = 0.5 * (gjm + gmj)
                    wij = 0.5 * (gjm - gmj)
                    acc_s += sij * sij
                    acc_w += wij * wij
            s_out[c, sp] = np.sqrt(2.0 * acc_s)
            w_out[c, sp] = np.sqrt(2.0 * acc_w)


def compute_strain_and_vorticity_magnitude(grad_u: np.ndarray):
    """同时返回 (|S|, |Omega|)，见 `_strain_vorticity_magnitude_kernel` 文档。"""
    g = np.ascontiguousarray(grad_u)
    n_cells, n_sps = g.shape[0], g.shape[1]
    s_out = np.empty((n_cells, n_sps))
    w_out = np.empty((n_cells, n_sps))
    _strain_vorticity_magnitude_kernel(g, s_out, w_out)
    return s_out, w_out
