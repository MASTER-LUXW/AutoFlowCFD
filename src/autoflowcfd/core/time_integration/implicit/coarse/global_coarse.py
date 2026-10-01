"""AutoFlowCFD V2.0 - 分布式块预处理的全局粗校正（跨 rank 的粗空间，各 rank 冗余直接解）。

## 为什么需要

各 rank 的块 ILU / 多层预处理只用两端都在本 rank 的耦合块（rank 间按块 Jacobi 式分解，
`fr_residual/jacobian/backend.py::select_rows`）。多层预处理的价值正在于全局的光滑模态
（plate_demo P0 单机：每步 GMRES 160 -> 51 次，`multilevel.py` 模块文档），而这些模态
跨越 rank 边界：rank 内的粗层看不到 rank 间的耦合，分区越多越接近单纯的块 ILU。

## 粗空间与粗矩阵

* 各 rank 在本地单元图上聚合（与本 rank 多层预处理同一组确定性聚合，继续粗化到不超过
  `COARSEST_DOF / n_var / n_ranks` 个节点；聚合体不跨 rank），按 rank 顺序编号成全局粗节点
  （`allgather` 各 rank 的节点数得到偏移）；
* `A_g = P_g^T A P_g`：本 rank 的对角块、rank 内耦合块与**跨 rank 耦合块**（行单元在本
  rank、列单元是 halo，装配时保留，见 `select_rows` / `cell_blocks.py`）按
  (全局粗行, 全局粗列, 行变量, 列变量) 求和成三元组；halo 单元的全局粗编号经与残差同一次
  halo 交换取得（`CoarseCommContext.compact_cell_values`）。三元组 `allgather` 后各 rank 拼出
  逐位相同的同一个矩阵（拼接顺序是 rank 顺序），各自做一次稀疏 LU。

## 作用（乘性，粗校正在前）

    z1 = P_g A_g^{-1} P_g^T r          P_g^T r：本 rank 那一段 + allreduce 求和
    z  = z1 + M_loc (r - A z1)

`A z1` 里跨 rank 耦合块乘的是 halo 单元上的 `z1`，它就是全局粗解的对应分量——每个 rank
都持有完整的粗解，不需要再交换。`M_loc` 是本 rank 的块 ILU 或多层预处理。每次作用一次
`allreduce`，全部 rank 的作用次数一致（GMRES 每次迭代一次）。单进程（`n_ranks == 1`）不用
本模块：本地多层预处理的最粗层就是全局的。
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numba import njit, prange

from ..block_ilu import coupling_csr
from ..preconditioner import PseudoTransientDiagonal
from .galerkin import coupling_triplets, coarse_triplets, prolong, restrict
from .multilevel import COARSEST_DOF, _Level, preconditioner_hierarchy


@dataclass(frozen=True)
class CoarseCommContext:
    """全局粗校正需要的集合通信与 halo 映射（由分布式后端构造，测试里可注入线程模拟）。

    allgather: 一维数组按 rank 顺序拼接（各 rank 长度可不同；dtype 为 int64 / float64）。
    allreduce_sum: 同形状一维 float64 数组逐元素求和。
    compact_cell_values: `(n_local,)` 逐单元值 -> `(n_compact,)` 紧凑空间（local+halo，与残差
        和装配器同一次 halo 交换与换序）里的逐单元值；耦合块的列单元编号就在这个空间里。
    """

    rank: int
    n_ranks: int
    allgather: Callable
    allreduce_sum: Callable
    compact_cell_values: Callable


class CrossRankCoupling:
    """跨 rank 耦合块（行单元在本 rank、列单元是 halo）的块 CSR；列是紧凑空间编号。"""

    __slots__ = ("indptr", "cols", "offset", "data", "col_dof")

    def __init__(self, coupling, n_local: int, n_compact: int):
        self.indptr, self.cols, self.offset, self.data = coupling_csr(coupling, n_local)
        self.col_dof = np.zeros(int(n_compact), dtype=np.int64)
        for g in coupling.groups:
            self.col_dof[g.cols] = g.blocks.shape[2]


@njit(cache=True, parallel=True)
def _cross_matvec_add(indptr, cols, offset, data, row_dof, col_dof, gid_col, zc, n_sps, n_var, out):
    """`out += J_cross (P_g zc)`：halo 列单元上的 `P_g zc` 就是全局粗解在其粗节点上的值。"""
    stride = n_sps * n_var
    for c in prange(indptr.shape[0] - 1):
        m = row_dof[c]
        base = c * stride
        for k in range(indptr[c], indptr[c + 1]):
            y = cols[k]
            my = col_dof[y]
            o = offset[k]
            g = gid_col[y] * n_var
            for i in range(m):
                acc = 0.0
                for j in range(my):
                    acc += data[o + i * my + j] * zc[g + j % n_var]
                out[base + i] += acc


class GlobalCoarseSpace:
    """全局粗空间的拓扑部分（缓存对象的生命周期内不变）：本地单元 -> 全局粗节点。"""

    __slots__ = ("ctx", "gid_local", "gid_compact", "offset", "n_mine", "n_global")

    def __init__(self, struct, high_order: bool, ctx: CoarseCommContext):
        self.ctx = ctx
        target = max(1, COARSEST_DOF // struct.n_var // ctx.n_ranks)
        comp = np.arange(struct.n_real.size, dtype=np.int64)
        n_mine = comp.size
        for agg, n_agg in preconditioner_hierarchy(struct, high_order, target):
            comp = agg[comp]
            n_mine = n_agg
        counts = np.asarray(ctx.allgather(np.array([n_mine], dtype=np.int64)), dtype=np.int64)
        self.offset = int(counts[:ctx.rank].sum())
        self.n_mine = int(n_mine)
        self.n_global = int(counts.sum())
        self.gid_local = comp + self.offset
        compact = np.asarray(ctx.compact_cell_values(self.gid_local.astype(np.float64)), dtype=np.float64)
        self.gid_compact = np.rint(compact).astype(np.int64)


class GlobalCoarseCorrection(PseudoTransientDiagonal):
    """`z = z1 + M_loc(r - A z1)`，`z1 = P_g A_g^{-1} P_g^T r`（见模块文档）；接口与对角预处理相同，
    向量是 cupy 数组时在主机上作用。"""

    __slots__ = ("_space", "_local", "_fine", "_cross", "_lu", "_n_var", "_xp")

    def __init__(self, space: GlobalCoarseSpace, local_factory: Callable, diag, off, struct,
                 cross: CrossRankCoupling, dtau_flat, n_sps: int):
        """`local_factory(dtau_host)`：按主机端 dtau 构造本 rank 的块 ILU / 多层预处理。"""
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import splu

        from autoflowcfd.core.utils.array_module import array_module

        self._xp = array_module(dtau_flat)
        super().__init__(dtau_flat, struct.n_var)
        dtau_host = np.asarray(dtau_flat.get() if hasattr(dtau_flat, "get") else dtau_flat,
                               dtype=np.float64).ravel()
        nv = self._n_var = struct.n_var
        self._space, self._cross = space, cross
        self._local = local_factory(dtau_host)
        self._fine = _Level(diag, off, struct, 1.0 / dtau_host, n_sps, factor=False)
        rows, cols, vals = coarse_triplets(diag, off, struct.row_dof, 1.0 / dtau_host, struct, space.gid_local,
                                           n_sps, nv)
        n_x = cross.cols.size * nv * nv
        xr, xc, xv = np.empty(n_x, np.int64), np.empty(n_x, np.int64), np.empty(n_x)
        if n_x:
            coupling_triplets(cross.indptr, cross.cols, cross.offset, cross.data, struct.row_dof, cross.col_dof,
                               space.gid_local, space.gid_compact, nv, xr, xc, xv)
        n = space.n_global * nv
        # 先在本 rank 合并重复项，只交换合并后的三元组
        mine = coo_matrix((np.concatenate([vals, xv]), (np.concatenate([rows, xr]), np.concatenate([cols, xc]))),
                          shape=(n, n)).tocsr().tocoo()
        ctx = space.ctx
        A_g = coo_matrix((ctx.allgather(np.ascontiguousarray(mine.data, dtype=np.float64)),
                          (ctx.allgather(np.ascontiguousarray(mine.row, dtype=np.int64)),
                           ctx.allgather(np.ascontiguousarray(mine.col, dtype=np.int64)))), shape=(n, n))
        self._lu = splu(A_g.tocsc())

    @property
    def flexible(self) -> bool:
        return self._local.flexible

    def apply(self, v_flat):
        shape = v_flat.shape
        x = np.ascontiguousarray(v_flat.get() if self._xp is not np else v_flat, dtype=np.float64).reshape(-1)
        sp, s, nv = self._space, self._fine.struct, self._n_var
        n_sps = self._fine.n_sps
        lo, hi = sp.offset * nv, (sp.offset + sp.n_mine) * nv
        rc = np.zeros(sp.n_global * nv)
        rc[lo:hi] = restrict(x, s.row_dof, sp.gid_local - sp.offset, sp.n_mine, n_sps, nv)
        zc = self._lu.solve(np.asarray(sp.ctx.allreduce_sum(rc), dtype=np.float64))
        z1 = np.empty_like(x)
        prolong(zc[lo:hi], s.row_dof, sp.gid_local - sp.offset, n_sps, nv, z1)
        az1 = self._fine.matvec(z1)
        c = self._cross
        _cross_matvec_add(c.indptr, c.cols, c.offset, c.data, s.row_dof, c.col_dof, sp.gid_compact, zc, n_sps,
                          nv, az1)
        z = z1 + self._local.apply((x - az1).reshape(-1, nv)).reshape(-1)
        z = z.reshape(shape)
        return z if self._xp is np else self._xp.asarray(z)
