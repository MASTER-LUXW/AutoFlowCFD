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

矩阵 `A = blockdiag(J_cc + I/dtau) + {J_cy}`（面邻居耦合块来自解析 Jacobian，
`fr_residual/jacobian/coupling.py`）。按距离 1 着色排序单元（同色单元互不
相邻，与块 Jacobi 装配同一份着色），做对角修正型不完全分解（D-ILU，
Pommerell 1992）：

    D~_c = D_c - sum_{y 邻 c, color(y) < color(c)} J_cy D~_y^{-1} J_yc
    M    = (D~ + L) D~^{-1} (D~ + U)

`L`/`U` 就是原耦合块（颜色低/高的邻居）。只修正对角块、不引入填充：多色序下
同色单元互不耦合，前代/回代与分解都按颜色分批、批内逐单元并行。

作用：
    前代  y_c = D~_c^{-1} (r_c - sum_{low} J_cy y_y)          颜色由低到高
    回代  z_c = y_c - D~_c^{-1} sum_{high} J_cz z_z             颜色由高到低

零填充槽位（原生基的填充解点）不在任何块里，它们的残差行恒为零，`A` 在那里
就是 `I/dtau`，预处理保持对角形式 `dtau * v`（与块 Jacobi 相同）。
"""

import numpy as np
from numba import njit, prange

from .preconditioner import PseudoTransientDiagonal


class BlockCouplingStructure:
    """按单元的耦合块 CSR（行单元 -> 列单元、块在扁平数组中的偏移）。"""

    __slots__ = ("indptr", "cols", "offset", "data", "n_real", "row_dof")

    def __init__(self, coupling, n_cells, n_real):
        """`coupling`：`CouplingBlocks`；`n_real`：`(n_cells,)` 每单元真实解点数。"""
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
        self.row_dof = 5 * self.n_real


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
def _factor_color(cells, color, diag_data, diag_off, inv_data, indptr, cols, offset, data, row_dof, inv_dtau,
                  n_sps):
    """一个颜色批次的 `D~_c^{-1}`（写入 `inv_data`，float32）。"""
    for ci in prange(cells.shape[0]):
        c = cells[ci]
        m = row_dof[c]
        a = np.empty((m, m))
        base = diag_off[c]
        for i in range(m):
            for j in range(m):
                a[i, j] = diag_data[base + i * m + j]
        for i in range(m):
            a[i, i] += inv_dtau[c * n_sps + i // 5]
        for k in range(indptr[c], indptr[c + 1]):
            y = cols[k]
            if color[y] >= color[c]:
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
def _forward_color(cells, color, x, y, inv_data, diag_off, indptr, cols, offset, data, row_dof, n_sps):
    for ci in prange(cells.shape[0]):
        c = cells[ci]
        m = row_dof[c]
        rhs = np.empty(m)
        for i in range(m):
            rhs[i] = x[c * n_sps * 5 + i]
        for k in range(indptr[c], indptr[c + 1]):
            yc = cols[k]
            if color[yc] >= color[c]:
                continue
            my = row_dof[yc]
            o = offset[k]
            for i in range(m):
                acc = 0.0
                for j in range(my):
                    acc += data[o + i * my + j] * y[yc * n_sps * 5 + j]
                rhs[i] -= acc
        b = diag_off[c]
        for i in range(m):
            acc = 0.0
            for j in range(m):
                acc += inv_data[b + i * m + j] * rhs[j]
            y[c * n_sps * 5 + i] = acc


@njit(cache=True, parallel=True)
def _backward_color(cells, color, z, inv_data, diag_off, indptr, cols, offset, data, row_dof, n_sps):
    for ci in prange(cells.shape[0]):
        c = cells[ci]
        m = row_dof[c]
        s = np.zeros(m)
        for k in range(indptr[c], indptr[c + 1]):
            yc = cols[k]
            if color[yc] <= color[c]:
                continue
            my = row_dof[yc]
            o = offset[k]
            for i in range(m):
                acc = 0.0
                for j in range(my):
                    acc += data[o + i * my + j] * z[yc * n_sps * 5 + j]
                s[i] += acc
        b = diag_off[c]
        for i in range(m):
            acc = 0.0
            for j in range(m):
                acc += inv_data[b + i * m + j] * s[j]
            z[c * n_sps * 5 + i] -= acc


class BlockILUPreconditioner(PseudoTransientDiagonal):
    """`M = (D~ + L) D~^{-1} (D~ + U)` 的逆作用；接口与对角预处理相同。

    分解与前代/回代在主机上做（numba）。GPU 后端的向量是 cupy 数组时，作用前后
    各做一次主机-设备传输（整场一个向量，plate_demo P1 约 43 MB）。
    """

    __slots__ = ("_struct", "_color_batches", "_color", "_inv", "_diag_off", "_n_sps", "_xp")

    @classmethod
    def from_cell_blocks(cls, jac, struct: BlockCouplingStructure, colors, dtau_flat):
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
        return cls(diag, diag_off, struct, colors, dtau_flat, jac.n_sps)

    def __init__(self, diag_data, diag_off, struct: BlockCouplingStructure, colors, dtau_flat, n_sps: int):
        from autoflowcfd.core.utils.array_module import array_module

        self._xp = array_module(dtau_flat)
        super().__init__(dtau_flat, 5)
        self._struct = struct
        self._n_sps = int(n_sps)
        self._color = np.ascontiguousarray(colors, dtype=np.int64)
        n_colors = int(self._color.max()) + 1 if self._color.size else 0
        self._color_batches = [np.nonzero(self._color == k)[0].astype(np.int64) for k in range(n_colors)]
        self._diag_off = diag_off
        self._inv = np.empty_like(diag_data)
        dtau_host = self.dtau.get() if hasattr(self.dtau, "get") else self.dtau
        inv_dtau = 1.0 / np.asarray(dtau_host, dtype=np.float64)
        s = struct
        for cells in self._color_batches:
            if cells.size:
                _factor_color(cells, self._color, diag_data, diag_off, self._inv, s.indptr, s.cols, s.offset,
                              s.data, s.row_dof, inv_dtau, self._n_sps)

    def apply(self, v_flat):
        out = super().apply(v_flat)              # 零填充槽位：dtau * v
        if self._xp is not np:
            v_flat, out = v_flat.get(), out.get()
        s = self._struct
        x = np.ascontiguousarray(v_flat, dtype=np.float64).reshape(-1)
        y = np.ascontiguousarray(out, dtype=np.float64).reshape(-1).copy()
        for cells in self._color_batches:
            if cells.size:
                _forward_color(cells, self._color, x, y, self._inv, self._diag_off, s.indptr, s.cols, s.offset,
                               s.data, s.row_dof, self._n_sps)
        for cells in reversed(self._color_batches):
            if cells.size:
                _backward_color(cells, self._color, y, self._inv, self._diag_off, s.indptr, s.cols, s.offset,
                                s.data, s.row_dof, self._n_sps)
        y = y.reshape(out.shape)
        return y if self._xp is np else self._xp.asarray(y)
