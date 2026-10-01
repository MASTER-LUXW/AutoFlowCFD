"""AutoFlowCFD V2.0 - 多层预处理的 Galerkin 粗算子、转移算子与装配矩阵的矩阵-向量乘。

细层矩阵就是块预处理已经装配好的那一个（`block_ilu.py` 模块文档）：

    A = blockdiag(J_cc + I/dtau) + {J_cy}

粗空间：每个聚合体 a、每个守恒变量 v 一个基向量，在 a 内全部单元的全部**真实**
解点上该变量取 1（零填充槽位不在粗空间里）。于是

    (P^T r)[a, v] = sum_{c in a} sum_{s 真实} r[c, s, v]
    (P z)[c, s, v] = z[agg(c), v]              （零填充槽位取 0）
    A_c = P^T A P：每个对角块/耦合块按 (行变量, 列变量) 求和，累加到
          (agg(行单元), agg(列单元)) 上；对角上再加真实解点的 1/dtau 之和。

块内行序与块预处理相同：`i = 解点 * n_var + 变量`。
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def _diag_triplets(diag_data, diag_off, row_dof, inv_dtau, agg, n_sps, n_var, rows, cols, vals):
    n_cells = row_dof.shape[0]
    nv2 = n_var * n_var
    for c in prange(n_cells):
        m = row_dof[c]
        base = diag_off[c]
        S = np.zeros((n_var, n_var))
        for i in range(m):
            vi = i % n_var
            for j in range(m):
                S[vi, j % n_var] += diag_data[base + i * m + j]
            S[vi, vi] += inv_dtau[c * n_sps + i // n_var]
        a = agg[c]
        o = c * nv2
        for v in range(n_var):
            for w in range(n_var):
                rows[o + v * n_var + w] = a * n_var + v
                cols[o + v * n_var + w] = a * n_var + w
                vals[o + v * n_var + w] = S[v, w]


@njit(cache=True, parallel=True)
def coupling_triplets(indptr, cols_c, offset, data, row_dof, col_dof, agg_row, agg_col, n_var, rows, cols, vals):
    """耦合块按 (行变量, 列变量) 求和，记到 `(agg_row[行单元], agg_col[列单元])`。行列分开给出
    自由度数与粗节点映射：rank 内耦合两者相同，跨 rank 耦合的列是 halo 单元（`global_coarse.py`）。"""
    n_cells = indptr.shape[0] - 1
    nv2 = n_var * n_var
    for c in prange(n_cells):
        m = row_dof[c]
        for k in range(indptr[c], indptr[c + 1]):
            y = cols_c[k]
            my = col_dof[y]
            o = offset[k]
            S = np.zeros((n_var, n_var))
            for i in range(m):
                vi = i % n_var
                for j in range(my):
                    S[vi, j % n_var] += data[o + i * my + j]
            t = k * nv2
            for v in range(n_var):
                for w in range(n_var):
                    rows[t + v * n_var + w] = agg_row[c] * n_var + v
                    cols[t + v * n_var + w] = agg_col[y] * n_var + w
                    vals[t + v * n_var + w] = S[v, w]


@njit(cache=True)
def restrict(r, row_dof, agg, n_agg, n_sps, n_var):
    """`P^T r`，`r` 是扁平 `(n_cells * n_sps * n_var,)`。"""
    out = np.zeros(n_agg * n_var)
    for c in range(row_dof.shape[0]):
        base = c * n_sps * n_var
        a = agg[c] * n_var
        for i in range(row_dof[c]):
            out[a + i % n_var] += r[base + i]
    return out


@njit(cache=True, parallel=True)
def prolong(zc, row_dof, agg, n_sps, n_var, out):
    """`out = P zc`（零填充槽位置 0）。"""
    for c in prange(row_dof.shape[0]):
        base = c * n_sps * n_var
        a = agg[c] * n_var
        m = row_dof[c]
        for i in range(n_sps * n_var):
            out[base + i] = zc[a + i % n_var] if i < m else 0.0


@njit(cache=True, parallel=True)
def assembled_matvec(x, diag_data, diag_off, row_dof, inv_dtau, indptr, cols, offset, data, n_sps, n_var, out):
    """`out = A x`，A 见模块文档（零填充行只有 `x / dtau`）。"""
    stride = n_sps * n_var
    for c in prange(row_dof.shape[0]):
        base = c * stride
        m = row_dof[c]
        db = diag_off[c]
        for i in range(stride):
            out[base + i] = x[base + i] * inv_dtau[c * n_sps + i // n_var]
        for i in range(m):
            acc = 0.0
            for j in range(m):
                acc += diag_data[db + i * m + j] * x[base + j]
            out[base + i] += acc
        for k in range(indptr[c], indptr[c + 1]):
            y = cols[k]
            my = row_dof[y]
            o = offset[k]
            yb = y * stride
            for i in range(m):
                acc = 0.0
                for j in range(my):
                    acc += data[o + i * my + j] * x[yb + j]
                out[base + i] += acc


def coarse_triplets(diag_data, diag_off, row_dof, inv_dtau, struct, agg, n_sps: int, n_var: int):
    """`P^T A P` 的未合并三元组 `(rows, cols, vals)`（粗节点编号即 `agg` 的值）。`struct` 为
    None 时只有对角块。"""
    n_cells = row_dof.shape[0]
    nv2 = n_var * n_var
    n_diag = n_cells * nv2
    n_coup = 0 if struct is None else struct.cols.size * nv2
    rows = np.empty(n_diag + n_coup, dtype=np.int64)
    cols = np.empty(n_diag + n_coup, dtype=np.int64)
    vals = np.empty(n_diag + n_coup, dtype=np.float64)
    _diag_triplets(diag_data, diag_off, row_dof, inv_dtau, agg, n_sps, n_var,
                   rows[:n_diag], cols[:n_diag], vals[:n_diag])
    if n_coup:
        coupling_triplets(struct.indptr, struct.cols, struct.offset, struct.data, row_dof, row_dof, agg, agg,
                           n_var, rows[n_diag:], cols[n_diag:], vals[n_diag:])
    return rows, cols, vals


def galerkin_coarse_matrix(diag_data, diag_off, row_dof, inv_dtau, struct, agg, n_agg: int,
                           n_sps: int, n_var: int):
    """`A_c = P^T A P`（scipy CSC，`(n_agg*n_var)^2`）。`struct` 为 None 时只有对角块。"""
    import scipy.sparse as sp

    rows, cols, vals = coarse_triplets(diag_data, diag_off, row_dof, inv_dtau, struct, agg, n_sps, n_var)
    n = n_agg * n_var
    return sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
