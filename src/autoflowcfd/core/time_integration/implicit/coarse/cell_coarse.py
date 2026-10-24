"""AutoFlowCFD V2.0 - 块 Jacobi 档的单元级叠加性粗校正（P>=1 平均流）。

## 为什么需要（cube_demo 粗网格 SA P2，第 300 步状态，同一 Newton 线性系统，restart 40、eta 0.1）

P2 大网格上平均流的面邻居耦合块放不下（每对单元 90x90 量级，合计几十 GB），平均流只能用块 Jacobi
（`block_budget.py` 的"两份 / 单份"档），没有块 ILU 档的多层粗空间（`multilevel.py`）。块 Jacobi 只在单元内
起作用，大 CFL 下全局的光滑模态（低马赫下的压力-速度耦合、尾迹里的长程对流）只能靠 GMRES 一次一层地传：

    CFL      块 Jacobi                          + 本文件的粗校正
    1000     42 次 228 s，非线性比 0.984        29 次 172 s，非线性比 0.964
    4000     294 次停在 0.551，非线性比 1.544   277 次达到 0.1，非线性比 0.956

CFL 4000 时块 Jacobi 的线性求解停滞、整步残差变大（该步被拒、CFL 被打回）；加粗校正后同一系统可解、步可
接受——线性求解不再给 CFL 增长封顶。

## 粗空间

每单元一个节点、`n_var` 个守恒变量。延拓 `P` 把节点值放到单元全部真实解点上（零填充槽位取 0），限制 `R`
按逐解点权重 `w` 求和：

    (R v)[c] = sum_s w[c, s] v[c, s]
    A_c = R (J + I/dtau) P：  对角块 sum_s w_s sum_t J_cc[s,:,t,:] + diag(sum_s w_s / dtau_s)
                              耦合块 sum_s w_s sum_t J_cy[s,:,t,:]（面邻居单元对）

`J_cc` 与 `J_cy` 的单元对 5x5 约化由解析装配器在面核里流式算出、不存完整耦合块
（`fr_residual/jacobian/coarse.py`），伪时间项在这里按当前 `dtau` 加。

## 限制权重（`restriction_weights`）

单元积分权重（解点求积权重 x det J，正性限制器的同一组 `W`：`R` 作用在残差上就是单元净通量，`A_c` 即以
FR 迹为面状态的有限体积通量 Jacobian）在该单元全部非负时用它；有负权重时改用真实解点上的均匀权重（同样合计
为单元体积，就是正性限制器给 Newton 残差范数的那一组 `norm_weights`，不另算一份）。P2 四面体顶点的积分权重是 -V/20（二次 Lagrange 顶点基函数的积分），P2 棱柱三角形顶点的
是 0。2026-10-16 同一线性系统（CFL 2600）的对照：

    权重                                全局线性残差（GMRES 次数）
    全部单元积分权重                    0.827（64，停滞）
    全部均匀（P^T）                     0.289（82，停滞）
    棱柱积分权重 + 四面体均匀（本规则）  0.095（117，达到）

负权重让 `R` 不再是正的平均，粗层方程在这些单元上与细层的守恒关系错位。均匀权重合计为单元体积而不是取 1：
粗层 K 循环的内层 GMRES 在粗层残差的 2 范数里最小化，两类单元的行须同一量级（棱柱行是积分权重、量级为
体积）。2026-10-24 第 300 步状态 CFL 1000：四面体行取 1 时 GMRES 29 次，按体积缩放后 24 次（块 Jacobi 42 次）。

## 粗层近似逆与组合

粗矩阵上用现有的聚合多层 K 循环（`multilevel.py`，每节点 1 个"解点"、没有 p 层；节点数不超过最粗层上限时
直接稀疏 LU）。与细层块 Jacobi **加性**组合：

    z = M_bj^{-1} v + P W R v

乘性（粗校正在前、块 Jacobi 后光滑，细层残差用矩阵自由 `J`）同一系统更差且每次作用多一次残差求值；粗层内
再套 GMRES 解到 1e-2 没有收益（2026-10-16）。K 循环让作用非线性，但非线性部分落在 `P` 的值域里，GMRES 只为
它存每单元 `n_var` 个数（`krylov/gmres.py` 模块文档"部分灵活"）。

## 分布式

各 rank 的粗矩阵行是本 rank 的单元；列在 halo 上的单元对（跨 rank 粗块）由装配器单独给出
（`fr_residual/jacobian/coarse.py::select_coarse_rows`）。多 rank 时粗层近似逆是以单元级粗矩阵为细层的全局
粗校正（`global_coarse.py::GlobalCoarseCorrection`）：本 rank 的 K 循环（或 LU）处理 rank 内耦合，跨 rank 粗块
进各 rank 冗余直接解的全局粗矩阵——与块 ILU 档"本 rank 多层 + 全局粗校正"同一结构。限制 / 延拓只涉及本 rank
单元，每次作用一次 `allreduce`。
"""

import numpy as np

from ..blocks.block_ilu import BlockCouplingStructure
from ..blocks.preconditioner import PseudoTransientDiagonal


def restriction_weights(quad_weights, uniform_weights):
    """逐（单元, 槽位）的限制权重（模块文档"限制权重"）。

    Args:
        quad_weights: `(n_cells, n_sps)` 解点求积权重 x det J（零填充槽位的值不用）。
        uniform_weights: `(n_cells, n_sps)` 真实解点上合计为单元体积的均匀权重、零填充槽位为 0（正性限制器的
            `norm_weights`）；真实解点掩码即其正值处。

    Raises:
        ValueError: 有单元没有正的均匀权重（体积非正）
    """
    from autoflowcfd.core.utils.array_module import array_module

    xp = array_module(quad_weights)
    uniform = xp.asarray(uniform_weights, dtype=xp.float64)
    real = uniform > 0.0
    if not bool(xp.all(real.any(axis=1))):
        raise ValueError("均匀权重（单元体积 / 真实解点数）必须为正")
    w = xp.where(real, xp.asarray(quad_weights, dtype=xp.float64), 0.0)
    negative = (w < 0.0).any(axis=1)
    return xp.where(negative[:, None], uniform, w)


def mean_coarse_available(n_real_prism: int, n_real_tet: int) -> bool:
    """平均流块 Jacobi 档能否叠加单元级粗校正：P>=1（每单元不止一个真实解点；P0 的块 Jacobi 本身就是单元级，
    有面邻居耦合块时走块 ILU）。单机与分布式都可用（模块文档"分布式"）。"""
    return max(int(n_real_prism), int(n_real_tet)) > 1


def mean_coarse_weights(positivity):
    """限制权重（`restriction_weights`）：积分权重与均匀权重都取正性限制器的那一份（与状态同一数组模块）。"""
    return restriction_weights(positivity.W, positivity.norm_weights)


class CellCoarseTopology:
    """粗矩阵的稀疏结构（面邻居单元对，行内按列排序）与聚合层次：只依赖网格，跨装配复用。"""

    __slots__ = ("n_nodes", "n_var", "rows", "cols", "order", "indptr", "hierarchy", "cross_rows", "cross_cols",
                 "_global_space")

    def __init__(self, n_nodes: int, rows, cols, n_var: int = 5, cross_rows=None, cross_cols=None):
        """`rows/cols`：装配器给出的单元对（每对一次、不含对角），顺序就是之后各次装配给出块的顺序。
        `cross_rows/cross_cols`：分布式的跨 rank 单元对（行为本 rank 单元、列为 halo 紧凑编号），单机为 None。"""
        from .aggregation import build_hierarchy
        from .multilevel import COARSEST_DOF

        self.n_nodes, self.n_var = int(n_nodes), int(n_var)
        self.rows = np.ascontiguousarray(rows, dtype=np.int64)
        self.cols = np.ascontiguousarray(cols, dtype=np.int64)
        if np.any(self.rows == self.cols):
            raise ValueError("粗层耦合块的单元对不能含对角（对角块单独给出）")
        self.order = np.lexsort((self.cols, self.rows))
        r, c = self.rows[self.order], self.cols[self.order]
        if r.size > 1 and np.any((r[1:] == r[:-1]) & (c[1:] == c[:-1])):
            raise ValueError("粗层耦合块的单元对有重复")
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(r, minlength=self.n_nodes))]).astype(np.int64)
        coarsest = max(1, COARSEST_DOF // self.n_var)
        self.hierarchy = build_hierarchy(self.indptr, c, self.n_nodes, coarsest) if self.n_nodes > coarsest else []
        self.cross_rows = None if cross_rows is None else np.ascontiguousarray(cross_rows, dtype=np.int64)
        self.cross_cols = None if cross_cols is None else np.ascontiguousarray(cross_cols, dtype=np.int64)
        self._global_space = None

    def matches(self, rows, cols, cross_rows=None, cross_cols=None) -> bool:
        def same(a, b):
            if a is None or b is None:
                return a is None and b is None
            return np.asarray(a).shape == b.shape and np.array_equal(a, b)
        return (same(rows, self.rows) and same(cols, self.cols) and same(cross_rows, self.cross_rows)
                and same(cross_cols, self.cross_cols))

    def global_space(self, struct, ctx):
        """多 rank 的全局粗空间（拓扑量，首次构造时各 rank 同时调用：其中有 allgather 与 halo 交换）。"""
        from .global_coarse import GlobalCoarseSpace

        if self._global_space is None:
            self._global_space = GlobalCoarseSpace(struct, False, ctx)
        return self._global_space

    def structure(self, blocks) -> BlockCouplingStructure:
        """本次装配的耦合块（与 `rows/cols` 同序，`(n_pairs, n_var, n_var)`）-> 块 CSR。"""
        nv2 = self.n_var * self.n_var
        data = np.ascontiguousarray(np.asarray(blocks)[self.order], dtype=np.float32).reshape(-1)
        return BlockCouplingStructure.from_csr(
            self.indptr, self.cols[self.order], np.arange(self.rows.size, dtype=np.int64) * nv2, data,
            np.ones(self.n_nodes, dtype=np.int64), self.n_var)


class CellCoarseMatrix:
    """一次装配的粗矩阵（不含伪时间项）：`diag (n_nodes, n_var, n_var)` 与面邻居耦合块的块 CSR；分布式另有跨 rank
    粗块（与 `topology.cross_rows/cross_cols` 同序）。"""

    __slots__ = ("topology", "diag", "struct", "cross_blocks")

    def __init__(self, topology: CellCoarseTopology, diag, blocks, cross_blocks=None):
        self.topology = topology
        self.diag = np.ascontiguousarray(diag, dtype=np.float64)
        self.struct = topology.structure(blocks)
        if (cross_blocks is None) != (topology.cross_rows is None):
            raise ValueError("跨 rank 粗块与拓扑不一致（分布式两者都有、单机都没有）")
        self.cross_blocks = None if cross_blocks is None else np.ascontiguousarray(cross_blocks, dtype=np.float32)

    def ptc_diagonal(self, weights, dtau_flat, n_sps: int):
        """`sum_s w_s / dtau_s`（`(n_nodes,)`，主机 float64）。"""
        from autoflowcfd.core.utils.array_module import to_host

        w = to_host(weights).reshape(self.topology.n_nodes, n_sps)
        dtau = to_host(dtau_flat).reshape(self.topology.n_nodes, n_sps)
        return (w / dtau).sum(axis=1)

    def scipy_matrix(self, inv_dtau=None):
        """整个粗矩阵（scipy CSR，测试与诊断用）。"""
        import scipy.sparse as sps

        n, nv, s = self.topology.n_nodes, self.topology.n_var, self.struct
        rows = np.repeat(np.arange(n), np.diff(s.indptr))
        blocks = s.data.reshape(-1, nv, nv).astype(np.float64)
        diag = self.diag.copy()
        if inv_dtau is not None:
            diag += inv_dtau[:, None, None] * np.eye(nv)
        row_node = np.concatenate([np.repeat(np.arange(n), nv * nv), np.repeat(rows, nv * nv)])
        col_node = np.concatenate([np.repeat(np.arange(n), nv * nv), np.repeat(s.cols, nv * nv)])
        a = np.tile(np.repeat(np.arange(nv), nv), n + rows.size)
        b = np.tile(np.tile(np.arange(nv), nv), n + rows.size)
        vals = np.concatenate([diag.reshape(-1), blocks.reshape(-1)])
        return sps.csr_matrix((vals, (row_node * nv + a, col_node * nv + b)), shape=(n * nv, n * nv))


class _CellLU(PseudoTransientDiagonal):
    """节点数不超过最粗层上限时的精确粗解（稀疏 LU）；接口与对角预处理相同（全局粗校正的局部部分也用它）。"""

    __slots__ = ("_lu",)

    def __init__(self, matrix: CellCoarseMatrix, dtau_node):
        from scipy.sparse.linalg import splu

        super().__init__(dtau_node, matrix.topology.n_var)
        self._lu = splu(matrix.scipy_matrix(1.0 / np.asarray(dtau_node, dtype=np.float64)).tocsc())

    def apply(self, v_flat):
        return self._lu.solve(np.ascontiguousarray(v_flat, dtype=np.float64).reshape(-1)).reshape(v_flat.shape)


class _LocalCellSolver:
    """本 rank 粗矩阵的近似逆工厂（`dtau_node -> 预处理子`）：K 循环或 LU。做成类而不是闭包（项目规范）。"""

    __slots__ = ("matrix",)

    def __init__(self, matrix: CellCoarseMatrix):
        self.matrix = matrix

    def __call__(self, dtau_node):
        from .multilevel import MultilevelPreconditioner

        m, topo = self.matrix, self.matrix.topology
        if not topo.hierarchy:
            return _CellLU(m, dtau_node)
        off = np.arange(topo.n_nodes, dtype=np.int64) * topo.n_var * topo.n_var
        return MultilevelPreconditioner(m.diag.reshape(-1).astype(np.float32), off, m.struct, topo.hierarchy,
                                        dtau_node, 1)


class CellCoarseSolve:
    """给定 `dtau` 下粗矩阵 `A_c` 的近似逆：聚合多层 K 循环，节点数不超过最粗层上限时稀疏 LU；多 rank 时外面叠
    以单元级粗矩阵为细层的全局粗校正（模块文档"分布式"）。"""

    __slots__ = ("_op", "n_var")

    def __init__(self, matrix: CellCoarseMatrix, inv_dtau, ctx=None):
        """`ctx`：分布式的 `CoarseCommContext`（单机为 None）；多 rank 时构造是集体调用。"""
        topo = matrix.topology
        self.n_var = topo.n_var
        dtau_node = 1.0 / np.asarray(inv_dtau, dtype=np.float64)
        local = _LocalCellSolver(matrix)
        if ctx is None or ctx.n_ranks <= 1:
            self._op = local(dtau_node)
            return
        from autoflowcfd.core.fr_residual.jacobian.coupling import CouplingBlocks, CouplingGroup

        from .global_coarse import CrossRankCoupling, GlobalCoarseCorrection

        if topo.cross_rows is None or topo.cross_cols is None or matrix.cross_blocks is None:
            raise ValueError("多 rank 的单元级粗校正需要跨 rank 粗块（装配器分布式分支给出）")
        space = topo.global_space(matrix.struct, ctx)
        cross = CrossRankCoupling(
            CouplingBlocks(groups=[CouplingGroup(row_is_prism=False, col_is_prism=False, rows=topo.cross_rows,
                                                 cols=topo.cross_cols, blocks=matrix.cross_blocks)]),
            topo.n_nodes, space.gid_compact.size)
        off = np.arange(topo.n_nodes, dtype=np.int64) * self.n_var * self.n_var
        self._op = GlobalCoarseCorrection(space, local, matrix.diag.reshape(-1).astype(np.float32), off,
                                          matrix.struct, cross, dtau_node, 1)

    def solve(self, rc):
        """`rc`：`(n_nodes, n_var)` 主机数组 -> 同形近似解。"""
        return np.asarray(self._op.apply(np.ascontiguousarray(rc, dtype=np.float64))).reshape(rc.shape)


class CoarseCorrectedPreconditioner:
    """`z = M_base^{-1} v + P W R v`（模块文档"粗层近似逆与组合"）；接口与 `PseudoTransientDiagonal` 相同，
    另给出部分灵活的拆分（`apply_split` / `apply_linear` / `coarse_rows`，`krylov/linear_solve.py`）。
    做成类而不是闭包：在整个 Krylov 求解期间存活（项目规范）。"""

    __slots__ = ("base", "coarse", "weights", "real", "n_sps", "coarse_rows", "_xp")

    flexible = True
    nonlinear_columns = None

    def __init__(self, base: PseudoTransientDiagonal, coarse: CellCoarseSolve, weights, real, n_sps: int):
        """`weights/real`：`(n_cells, n_sps)` 限制权重与真实解点掩码（与状态同一数组模块）。"""
        from autoflowcfd.core.utils.array_module import array_module

        self.base, self.coarse = base, coarse
        self._xp = array_module(weights)
        self.weights = weights
        self.real = self._xp.asarray(real, dtype=bool)
        self.n_sps = int(n_sps)
        n_cells = self.real.shape[0]
        cell = self._xp.repeat(self._xp.arange(n_cells, dtype=self._xp.int64), self.n_sps)
        self.coarse_rows = self._xp.where(self.real.reshape(-1), cell, -1)

    @property
    def n_coarse_cols(self) -> int:
        return self.coarse.n_var

    @property
    def dtau(self):
        return self.base.dtau

    def _coarse_vector(self, v):
        from autoflowcfd.core.utils.array_module import to_host

        xp, nv = self._xp, self.coarse.n_var
        rc = (self.weights[:, :, None] * v.reshape(self.real.shape[0], self.n_sps, nv)).sum(axis=1)
        zc = self.coarse.solve(np.ascontiguousarray(to_host(rc), dtype=np.float64))
        return zc if xp is np else xp.asarray(zc)

    def prolong_add(self, z, c):
        """原地 `z += P c`（`z`：`(n_cells * n_sps, n_var)`）。"""
        nv = self.coarse.n_var
        z3 = z.reshape(self.real.shape[0], self.n_sps, nv)
        z3 += self._xp.where(self.real[:, :, None], c[:, None, :], 0.0)

    def apply_split(self, v):
        return self.base.apply(v), self._coarse_vector(v)

    def apply_linear(self, v):
        return self.base.apply(v)

    def apply(self, v):
        z, c = self.apply_split(v)
        self.prolong_add(z, c)
        return z
