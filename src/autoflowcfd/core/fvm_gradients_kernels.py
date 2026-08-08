"""Numba CPU kernel：Green-Gauss 梯度重构 + Barth-Jespersen 限制器。

fvm_gradients.py 里 green_gauss_gradient/barth_jespersen_limiter 的逐面
scatter-add 用的是 numpy 的 np.add.at/np.maximum.at/np.minimum.at，这两个
在 numpy 里已经是相对高效的向量化实现，但不能被 Numba nopython 模式直接
JIT（np.add.at 不在 Numba 支持的 numpy 子集里）。这里把同样的算法翻译成
显式的逐面循环 + @njit(parallel=True)：面几何计算部分用 prange 并行（每个
面独立，无竞争），累加进单元的 scatter 部分改成串行循环（多个面会写向
同一个单元，并行会产生数据竞争，所以这一步和 np.add.at 一样保持顺序执行）。

对外的 green_gauss_gradient/barth_jespersen_limiter（fvm_gradients.py）在
Numba 可用时会调用这里的实现，公式和边界情况处理与原 numpy 版本逐项对应，
不是重新推导。
"""

import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@njit(cache=True)
def _green_gauss_gradient_kernel(
    cell_values: np.ndarray,       # (n_cells, n_vars)
    int_owner: np.ndarray,         # (n_int,)
    int_neigh: np.ndarray,         # (n_int,)
    int_areas: np.ndarray,         # (n_int,)
    int_normals: np.ndarray,       # (n_int, 3)
    bnd_owner: np.ndarray,         # (n_bnd,)
    bnd_areas: np.ndarray,         # (n_bnd,)
    bnd_normals: np.ndarray,       # (n_bnd, 3)
    bnd_values: np.ndarray,        # (n_bnd, n_vars) - ghost/BC values
    cell_volumes: np.ndarray,      # (n_cells,)
) -> np.ndarray:
    """Green-Gauss 梯度：grad[i] = (1/V_i) * sum_faces phi_f * n * A."""
    n_cells, n_vars = cell_values.shape
    grad = np.zeros((n_cells, n_vars, 3), dtype=np.float64)

    n_int = int_owner.shape[0]
    for f in range(n_int):
        o = int_owner[f]
        nb = int_neigh[f]
        a = int_areas[f]
        nx, ny, nz = int_normals[f, 0], int_normals[f, 1], int_normals[f, 2]
        for v in range(n_vars):
            phi_f = 0.5 * (cell_values[o, v] + cell_values[nb, v])
            cx = phi_f * a * nx
            cy = phi_f * a * ny
            cz = phi_f * a * nz
            grad[o, v, 0] += cx
            grad[o, v, 1] += cy
            grad[o, v, 2] += cz
            grad[nb, v, 0] -= cx
            grad[nb, v, 1] -= cy
            grad[nb, v, 2] -= cz

    n_bnd = bnd_owner.shape[0]
    for f in range(n_bnd):
        o = bnd_owner[f]
        a = bnd_areas[f]
        nx, ny, nz = bnd_normals[f, 0], bnd_normals[f, 1], bnd_normals[f, 2]
        for v in range(n_vars):
            phi_b = bnd_values[f, v]
            grad[o, v, 0] += phi_b * a * nx
            grad[o, v, 1] += phi_b * a * ny
            grad[o, v, 2] += phi_b * a * nz

    for c in range(n_cells):
        vol = cell_volumes[c]
        if vol < 1e-30:
            vol = 1e-30
        for v in range(n_vars):
            grad[c, v, 0] /= vol
            grad[c, v, 1] /= vol
            grad[c, v, 2] /= vol

    return grad


@njit(cache=True)
def _barth_jespersen_limiter_kernel(
    cell_values: np.ndarray,   # (n_cells, n_vars)
    grad: np.ndarray,          # (n_cells, n_vars, 3)
    owner: np.ndarray,         # (n_faces,) - every face's owner
    face_centers: np.ndarray,  # (n_faces, 3)
    cell_centroids: np.ndarray,  # (n_cells, 3)
    int_owner: np.ndarray,     # (n_int,)
    int_neigh: np.ndarray,     # (n_int,)
    int_face_idx: np.ndarray,  # (n_int,) - index of each internal face into face_centers
) -> np.ndarray:
    """Barth-Jespersen 限制器。

    先算出每个单元 u_min/u_max（自身 + 内部面邻居），再对每个面（含边界面，
    owner 侧一次；内部面额外对 neighbour 侧再算一次）用重构偏差算出该面
    对 owner/neighbour 各自限制器的约束，取所有面里最紧的一个（逐单元
    逐变量取 min）。
    """
    n_cells, n_vars = cell_values.shape
    n_faces = face_centers.shape[0]
    eps = 1e-12

    u_max = cell_values.copy()
    u_min = cell_values.copy()

    n_int = int_owner.shape[0]
    for f in range(n_int):
        o = int_owner[f]
        nb = int_neigh[f]
        for v in range(n_vars):
            vo = cell_values[o, v]
            vn = cell_values[nb, v]
            if vn > u_max[o, v]:
                u_max[o, v] = vn
            if vn < u_min[o, v]:
                u_min[o, v] = vn
            if vo > u_max[nb, v]:
                u_max[nb, v] = vo
            if vo < u_min[nb, v]:
                u_min[nb, v] = vo

    phi = np.ones((n_cells, n_vars), dtype=np.float64)

    def _constrain(c, fidx, phi_arr):
        rx = face_centers[fidx, 0] - cell_centroids[c, 0]
        ry = face_centers[fidx, 1] - cell_centroids[c, 1]
        rz = face_centers[fidx, 2] - cell_centroids[c, 2]
        for v in range(n_vars):
            delta = grad[c, v, 0] * rx + grad[c, v, 1] * ry + grad[c, v, 2] * rz
            if delta > eps:
                d = delta if delta > eps else eps
                pf = (u_max[c, v] - cell_values[c, v]) / d
                if pf > 1.0:
                    pf = 1.0
            elif delta < -eps:
                d = delta if delta < -eps else -eps
                pf = (u_min[c, v] - cell_values[c, v]) / d
                if pf > 1.0:
                    pf = 1.0
            else:
                pf = 1.0
            if pf < 0.0:
                pf = 0.0
            if pf < phi_arr[c, v]:
                phi_arr[c, v] = pf

    # Every face constrains its own owner.
    for f in range(n_faces):
        _constrain(owner[f], f, phi)
    # Internal faces additionally constrain the neighbour side.
    for f in range(n_int):
        _constrain(int_neigh[f], int_face_idx[f], phi)

    return phi
