"""AutoFlowCFD V2.0 - 单元块 ILU 预处理（多色序、对角修正型 D-ILU）。

## 为什么块 Jacobi 在大 CFL 下不够

块 Jacobi 只看单元自身的块，单元之间的耦合全部留给 GMRES。伪时间步长大时
`I/dtau + J` 由 `J` 主导，单元间耦合（对流的上下游、声学与低马赫压力的椭圆型
耦合）占主导，迭代数随 CFL 与网格尺寸一起增长。实测（432 单元通道 P1 完整
Jacobian，`rtol=0.01`）：

    CFL 倍数       1     10    100    1000   10000
    块 Jacobi      6     28    109     156     167
    块 SGS         3     12     47      61      65
    块 ILU(0)      3      9     29      47      50

plate_demo P0 阶段 CFL 升到约 500 以后块 Jacobi 下 GMRES 用满 200 次仍未达到
容差，CFL 随即塌回 27（长测日志）。

## 做法

矩阵 `A = blockdiag(J_cc + I/dtau) + {J_cy}`（面邻居耦合块来自解析 Jacobian
`fr_residual/jacobian/coupling.py`，P0 来自距离 2 着色差分 `cell_blocks.py`）。
按单元排序名次 `rank` 做对角修正型不完全分解（D-ILU，Pommerell 1992）：

    D~_c = D_c - sum_{y 邻 c, rank(y) < rank(c)} J_cy D~_y^{-1} J_yc
    M    = (D~ + L) D~^{-1} (D~ + U)

`L`/`U` 就是原耦合块（名次低/高的邻居）。只修正对角块、不引入填充——面邻接图上
一个单元的两个面邻居几乎从不互为面邻居，ILU(0) 在这张图上本来就只修正对角块。

作用：
    前代  y_c = D~_c^{-1} (r_c - sum_{low} J_cy y_y)          名次由低到高
    回代  z_c = y_c - D~_c^{-1} sum_{high} J_cz z_z             名次由高到低

## 排序：RCM，不是多色（2026-09-26）

此前按距离 1 着色排序（同色单元互不相邻、批内并行）。多色序是 ILU 最差的排序
之一：每个单元的"已消去"邻居只是颜色更低的那几个，对流的上下游链被颜色切碎。
plate_demo P0（17.9 万单元）CFL 640 下 GMRES 152 次、CFL 1000 以上用满 200 次，
P0 阶段因此收敛不深。改为 CFD Newton-Krylov 的标准排序 Reverse Cuthill-McKee
（Pueyo & Zingg 1998 对 ILU 排序的比较），在耦合图本身上算（`scipy.sparse.csgraph`）。

并行靠**波前分层**：`level(c) = 1 + max{level(y) : y 邻 c, rank(y) < rank(c)}`，
同层单元之间没有依赖，分解与前代/回代按层分批、层内并行（回代反序）。核函数只
比较名次，对任意排序都成立。

零填充槽位（原生基的填充解点）不在任何块里，它们的残差行恒为零，`A` 在那里
就是 `I/dtau`，预处理保持对角形式 `dtau * v`（与块 Jacobi 相同）。
"""

import numpy as np
from numba import njit, prange

from .preconditioner import PseudoTransientDiagonal


class BlockCouplingStructure:
    """按单元的耦合块 CSR（行单元 -> 列单元、块在扁平数组中的偏移）。"""

    __slots__ = ("indptr", "cols", "offset", "data", "n_real", "row_dof", "n_var", "rank", "levels")

    def __init__(self, coupling, n_cells, n_real, n_var: int = 5):
        """`coupling`：耦合块组（`groups` 里每组有 `rows/cols/blocks`）；`n_real`：
        `(n_cells,)` 每单元真实解点数；`n_var`：每个解点的未知量个数。"""
        rows = np.concatenate([g.rows for g in coupling.groups]) if coupling.groups else np.zeros(0, np.int64)
        cols = np.concatenate([g.cols for g in coupling.groups]) if coupling.groups else np.zeros(0, np.int64)
        sizes = np.concatenate([np.full(g.rows.size, g.blocks.shape[1] * g.blocks.shape[2], dtype=np.int64)
                                for g in coupling.groups]) if coupling.groups else np.zeros(0, np.int64)
        src_off = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        data = np.empty(int(sizes.sum()), dtype=np.float32)
        pos = 0
        for g in coupling.groups:
            flat = g.blocks.reshape(-1)
            data[pos:pos + flat.size] = flat
            pos += flat.size
        order = np.lexsort((cols, rows))
        self.cols = np.ascontiguousarray(cols[order])
        self.offset = np.ascontiguousarray(src_off[order])
        counts = np.bincount(rows, minlength=n_cells)
        self.indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.data = data
        self.n_real = np.ascontiguousarray(n_real, dtype=np.int64)
        self.n_var = int(n_var)
        self.row_dof = self.n_var * self.n_real
        self.rank, self.levels = rcm_wavefronts(self.indptr, self.cols, int(n_cells))


def rcm_wavefronts(indptr, cols, n_cells: int):
    """耦合图上的 RCM 名次 `rank`（`(n_cells,)`）与波前层（按层的单元数组列表，
    层内互不依赖），见模块文档"排序"。"""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    graph = sp.csr_matrix((np.ones(cols.size, dtype=np.int8), cols, indptr), shape=(n_cells, n_cells))
    perm = np.asarray(reverse_cuthill_mckee(graph, symmetric_mode=True), dtype=np.int64)
    rank = np.empty(n_cells, dtype=np.int64)
    rank[perm] = np.arange(n_cells)
    level = _wavefront_levels(perm, rank, indptr, cols)
    order = np.argsort(level, kind="stable")
    bounds = np.searchsorted(level[order], np.arange(int(level.max()) + 2 if n_cells else 1))
    return rank, [order[bounds[k]:bounds[k + 1]] for k in range(bounds.size - 1)]


@njit(cache=True)
def _wavefront_levels(perm, rank, indptr, cols):
    level = np.zeros(perm.size, dtype=np.int64)
    for c in perm:
        lv = 0
        for k in range(indptr[c], indptr[c + 1]):
            y = cols[k]
            if rank[y] < rank[c] and level[y] + 1 > lv:
                lv = level[y] + 1
        level[c] = lv
    return level


@njit(cache=True)
def _invert_small(a):
    """`a` (m, m) float64 的逆（部分选主元 Gauss-Jordan）。"""
    m = a.shape[0]
    w = np.zeros((m, 2 * m))
    for i in range(m):
        for j in range(m):
            w[i, j] = a[i, j]
        w[i, m + i] = 1.0
    for col in range(m):
        piv = col
        best = abs(w[col, col])
        for r in range(col + 1, m):
            if abs(w[r, col]) > best:
                best = abs(w[r, col])
                piv = r
        if piv != col:
            for j in range(2 * m):
                t = w[col, j]
                w[col, j] = w[piv, j]
                w[piv, j] = t
        inv_p = 1.0 / w[col, col]
        for j in range(2 * m):
            w[col, j] *= inv_p
        for r in range(m):
            if r != col:
                f = w[r, col]
                if f != 0.0:
                    for j in range(2 * m):
                        w[r, j] -= f * w[col, j]
    out = np.empty((m, m))
    for i in range(m):
        for j in range(m):
            out[i, j] = w[i, m + j]
    return out


@njit(cache=True, parallel=True)
def _factor_level(cells, rank, diag_data, diag_off, inv_data, indptr, cols, offset, data, row_dof, inv_dtau,
                  n_sps, n_var):
    """一个波前层的 `D~_c^{-1}`（写入 `inv_data`，float32）。"""
    for ci in prange(cells.shape[0]):
        c = cells[ci]
        m = row_dof[c]
        a = np.empty((m, m))
        base = diag_off[c]
        for i in range(m):
            for j in range(m):
                a[i, j] = diag_data[base + i * m + j]
        for i in range(m):
            a[i, i] += inv_dtau[c * n_sps + i // n_var]
        for k in range(indptr[c], indptr[c + 1]):
            y = cols[k]
            if rank[y] >= rank[c]:
                continue
            my = row_dof[y]
            # 找 J_yc
            kyc = -1
            for kk in range(indptr[y], indptr[y + 1]):
                if cols[kk] == c:
                    kyc = kk
                    break
            if kyc < 0:
                continue
            ocy, oyc, oinv = offset[k], offset[kyc], diag_off[y]
            # T = J_cy D~_y^{-1}   (m x my)
            T = np.zeros((m, my))
            for i in range(m):
                for p in range(my):
                    acc = 0.0
                    for q in range(my):
                        acc += data[ocy + i * my + q] * inv_data[oinv + q * my + p]
                    T[i, p] = acc
            for i in range(m):
                for j in range(m):
                    acc = 0.0
                    for p in range(my):
                        acc += T[i, p] * data[oyc + p * m + j]
                    a[i, j] -= acc
        inv = _invert_small(a)
        for i in range(m):
            for j in range(m):
                inv_data[base + i * m + j] = inv[i, j]


@njit(cache=True, parallel=True)
def _forward_level(cells, rank, x, y, inv_data, diag_off, indptr, cols, offset, data, row_dof, n_sps, n_var):
    for ci in prange(cells.shape[0]):
        c = cells[ci]
        m = row_dof[c]
        rhs = np.empty(m)
        for i in range(m):
            rhs[i] = x[c * n_sps * n_var + i]
        for k in range(indptr[c], indptr[c + 1]):
            yc = cols[k]
            if rank[yc] >= rank[c]:
                continue
            my = row_dof[yc]
            o = offset[k]
            for i in range(m):
                acc = 0.0
                for j in range(my):
                    acc += data[o + i * my + j] * y[yc * n_sps * n_var + j]
                rhs[i] -= acc
        b = diag_off[c]
        for i in range(m):
            acc = 0.0
            for j in range(m):
                acc += inv_data[b + i * m + j] * rhs[j]
            y[c * n_sps * n_var + i] = acc


@njit(cache=True, parallel=True)
def _backward_level(cells, rank, z, inv_data, diag_off, indptr, cols, offset, data, row_dof, n_sps, n_var):
    for ci in prange(cells.shape[0]):
        c = cells[ci]
        m = row_dof[c]
        s = np.zeros(m)
        for k in range(indptr[c], indptr[c + 1]):
            yc = cols[k]
            if rank[yc] <= rank[c]:
                continue
            my = row_dof[yc]
            o = offset[k]
            for i in range(m):
                acc = 0.0
                for j in range(my):
                    acc += data[o + i * my + j] * z[yc * n_sps * n_var + j]
                s[i] += acc
        b = diag_off[c]
        for i in range(m):
            acc = 0.0
            for j in range(m):
                acc += inv_data[b + i * m + j] * s[j]
            z[c * n_sps * n_var + i] -= acc


class BlockILUPreconditioner(PseudoTransientDiagonal):
    """`M = (D~ + L) D~^{-1} (D~ + U)` 的逆作用；接口与对角预处理相同。

    分解与前代/回代在主机上做（numba）。GPU 后端的向量是 cupy 数组时，作用前后
    各做一次主机-设备传输（整场一个向量，plate_demo P1 约 43 MB）。
    """

    __slots__ = ("_struct", "_inv", "_diag_off", "_n_sps", "_xp")

    @classmethod
    def from_cell_blocks(cls, jac, struct: BlockCouplingStructure, dtau_flat):
        """由 `CellBlockJacobian`（对角块）与耦合结构构造。"""
        def host(a):
            return a.get() if hasattr(a, "get") else np.asarray(a)
        n_cells = jac.prism_cells.size + jac.tet_cells.size
        bp, bt = host(jac.blocks_prism), host(jac.blocks_tet)
        sp, st = bp.shape[1] * bp.shape[2], bt.shape[1] * bt.shape[2]
        diag_off = np.empty(n_cells, dtype=np.int64)
        diag_off[jac.prism_cells] = np.arange(jac.prism_cells.size) * sp
        diag_off[jac.tet_cells] = jac.prism_cells.size * sp + np.arange(jac.tet_cells.size) * st
        diag = np.concatenate([bp.reshape(-1), bt.reshape(-1)]).astype(np.float32)
        return cls(diag, diag_off, struct, dtau_flat, jac.n_sps)

    def __init__(self, diag_data, diag_off, struct: BlockCouplingStructure, dtau_flat, n_sps: int):
        from autoflowcfd.core.utils.array_module import array_module

        self._xp = array_module(dtau_flat)
        super().__init__(dtau_flat, struct.n_var)
        self._struct = struct
        self._n_sps = int(n_sps)
        self._diag_off = diag_off
        self._inv = np.empty_like(diag_data)
        dtau_host = self.dtau.get() if hasattr(self.dtau, "get") else self.dtau
        inv_dtau = 1.0 / np.asarray(dtau_host, dtype=np.float64)
        s = struct
        for cells in s.levels:
            _factor_level(cells, s.rank, diag_data, diag_off, self._inv, s.indptr, s.cols, s.offset,
                          s.data, s.row_dof, inv_dtau, self._n_sps, s.n_var)

    def apply(self, v_flat):
        out = super().apply(v_flat)              # 零填充槽位：dtau * v
        if self._xp is not np:
            v_flat, out = v_flat.get(), out.get()
        s = self._struct
        x = np.ascontiguousarray(v_flat, dtype=np.float64).reshape(-1)
        y = np.ascontiguousarray(out, dtype=np.float64).reshape(-1).copy()
        for cells in s.levels:
            _forward_level(cells, s.rank, x, y, self._inv, self._diag_off, s.indptr, s.cols, s.offset,
                           s.data, s.row_dof, self._n_sps, s.n_var)
        for cells in reversed(s.levels):
            _backward_level(cells, s.rank, y, self._inv, self._diag_off, s.indptr, s.cols, s.offset,
                            s.data, s.row_dof, self._n_sps, s.n_var)
        y = y.reshape(out.shape)
        return y if self._xp is np else self._xp.asarray(y)
