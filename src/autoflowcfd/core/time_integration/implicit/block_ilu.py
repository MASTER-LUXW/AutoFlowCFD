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
        self.indptr, self.cols, self.offset, self.data = coupling_csr(coupling, n_cells)
        self._set_rows(n_real, n_var)

    def _set_rows(self, n_real, n_var: int):
        """逐行单元的真实自由度数与 RCM 名次 / 波前层（两个构造入口共用）。"""
        self.n_real = np.ascontiguousarray(n_real, dtype=np.int64)
        self.n_var = int(n_var)
        self.row_dof = self.n_var * self.n_real
        self.rank, self.levels = rcm_wavefronts(self.indptr, self.cols, int(self.n_real.size))

    @classmethod
    def from_csr(cls, indptr, cols, offset, data, n_real, n_var: int):
        """由现成的块 CSR（行内列已排序、不含对角块）构造；多层预处理的粗层用它。"""
        obj = cls.__new__(cls)
        obj.indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        obj.cols = np.ascontiguousarray(cols, dtype=np.int64)
        obj.offset = np.ascontiguousarray(offset, dtype=np.int64)
        obj.data = np.ascontiguousarray(data, dtype=np.float32)
        obj._set_rows(n_real, n_var)
        return obj


def coupling_csr(coupling, n_rows: int):
    """耦合块组（`groups` 里每组有 `rows/cols/blocks`）-> 块 CSR `(indptr, cols, offset, data)`：
    行内按列排序，`offset` 是块在扁平 float32 `data` 里的起点（块内行主序）。"""
    groups = coupling.groups
    rows = np.concatenate([g.rows for g in groups]) if groups else np.zeros(0, np.int64)
    cols = np.concatenate([g.cols for g in groups]) if groups else np.zeros(0, np.int64)
    sizes = np.concatenate([np.full(g.rows.size, g.blocks.shape[1] * g.blocks.shape[2], dtype=np.int64)
                            for g in groups]) if groups else np.zeros(0, np.int64)
    src_off = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    data = np.empty(int(sizes.sum()), dtype=np.float32)
    pos = 0
    for g in groups:
        flat = g.blocks.reshape(-1)
        data[pos:pos + flat.size] = flat
        pos += flat.size
    order = np.lexsort((cols, rows))
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=int(n_rows)))]).astype(np.int64)
    return indptr, np.ascontiguousarray(cols[order], dtype=np.int64), np.ascontiguousarray(src_off[order]), data


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
def _block64(flat, start, rows, cols):
    """扁平 float32 存储里从 `start` 起的 `rows x cols` 块（行主序）-> 连续 float64 矩阵。"""
    return flat[start:start + rows * cols].astype(np.float64).reshape((rows, cols))


@njit(cache=True, parallel=True)
def _factor_level(cells, rank, diag_data, diag_off, inv_data, indptr, cols, offset, data, row_dof, inv_dtau,
                  n_sps, n_var):
    """一个波前层的 `D~_c^{-1}`（写入 `inv_data`，float32；计算在 float64 里做）。

    块乘与求逆走 BLAS/LAPACK（`np.dot`、`np.linalg.inv`）：P3 原生棱柱平均流块 200x200 时，此前的
    手写三重循环与 Gauss-Jordan 一次分解 11.1 s（湍流平板 3072 单元，占 P3 Newton 步耗时 75%）。
    """
    for ci in prange(cells.shape[0]):
        c = cells[ci]
        m = row_dof[c]
        base = diag_off[c]
        a = _block64(diag_data, base, m, m)
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
            # D~_c -= J_cy D~_y^{-1} J_yc
            a -= np.dot(np.dot(_block64(data, offset[k], m, my), _block64(inv_data, diag_off[y], my, my)),
                        _block64(data, offset[kyc], my, m))
        inv = np.linalg.inv(a).ravel()
        for i in range(m * m):
            inv_data[base + i] = inv[i]


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


def factor_block_ilu(diag_data, diag_off, struct: BlockCouplingStructure, inv_dtau, n_sps: int):
    """D-ILU 分解：返回各单元 `D~_c^{-1}`（与 `diag_data` 同布局，float32）。

    `inv_dtau`：逐（单元, 解点）的 `1/dtau`，加到对角块对角线上；多层预处理的粗层
    已把 PTC 项并进对角块，传全零。"""
    inv = np.empty_like(diag_data)
    for cells in struct.levels:
        _factor_level(cells, struct.rank, diag_data, diag_off, inv, struct.indptr, struct.cols,
                      struct.offset, struct.data, struct.row_dof, inv_dtau, int(n_sps), struct.n_var)
    return inv


def solve_block_ilu(inv, diag_off, struct: BlockCouplingStructure, x, y, n_sps: int):
    """`y <- M^{-1} x` 的前代 + 回代（`y` 进入时须已含零填充槽位的值，真实行被覆盖）。"""
    for cells in struct.levels:
        _forward_level(cells, struct.rank, x, y, inv, diag_off, struct.indptr, struct.cols, struct.offset,
                       struct.data, struct.row_dof, int(n_sps), struct.n_var)
    for cells in reversed(struct.levels):
        _backward_level(cells, struct.rank, y, inv, diag_off, struct.indptr, struct.cols, struct.offset,
                        struct.data, struct.row_dof, int(n_sps), struct.n_var)
    return y


def flatten_cell_blocks(jac):
    """`CellBlockJacobian` 的对角块 -> 主机上的扁平 float32 数组与逐单元偏移
    `(diag_data, diag_off)`（块内行主序；块 ILU 与多层预处理共用）。"""
    def host(a):
        return a.get() if hasattr(a, "get") else np.asarray(a)
    n_cells = jac.prism_cells.size + jac.tet_cells.size
    bp, bt = host(jac.blocks_prism), host(jac.blocks_tet)
    sp, st = bp.shape[1] * bp.shape[2], bt.shape[1] * bt.shape[2]
    diag_off = np.empty(n_cells, dtype=np.int64)
    diag_off[jac.prism_cells] = np.arange(jac.prism_cells.size) * sp
    diag_off[jac.tet_cells] = jac.prism_cells.size * sp + np.arange(jac.tet_cells.size) * st
    diag = np.concatenate([bp.reshape(-1), bt.reshape(-1)]).astype(np.float32)
    return diag, diag_off


class BlockILUPreconditioner(PseudoTransientDiagonal):
    """`M = (D~ + L) D~^{-1} (D~ + U)` 的逆作用；接口与对角预处理相同。

    分解与前代/回代在主机上做（numba）。GPU 后端的向量是 cupy 数组时，作用前后
    各做一次主机-设备传输（整场一个向量，plate_demo P1 约 43 MB）。
    """

    __slots__ = ("_struct", "_inv", "_diag_off", "_n_sps", "_xp")

    def __init__(self, diag_data, diag_off, struct: BlockCouplingStructure, dtau_flat, n_sps: int):
        from autoflowcfd.core.utils.array_module import array_module

        self._xp = array_module(dtau_flat)
        super().__init__(dtau_flat, struct.n_var)
        self._struct = struct
        self._n_sps = int(n_sps)
        self._diag_off = diag_off
        dtau_host = self.dtau.get() if hasattr(self.dtau, "get") else self.dtau
        self._inv = factor_block_ilu(diag_data, diag_off, struct, 1.0 / np.asarray(dtau_host, dtype=np.float64),
                                     self._n_sps)

    def apply(self, v_flat):
        out = super().apply(v_flat)              # 零填充槽位：dtau * v
        if self._xp is not np:
            v_flat, out = v_flat.get(), out.get()
        x = np.ascontiguousarray(v_flat, dtype=np.float64).reshape(-1)
        y = np.ascontiguousarray(out, dtype=np.float64).reshape(-1).copy()
        solve_block_ilu(self._inv, self._diag_off, self._struct, x, y, self._n_sps)
        y = y.reshape(out.shape)
        return y if self._xp is np else self._xp.asarray(y)
