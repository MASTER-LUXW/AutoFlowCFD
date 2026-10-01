"""AutoFlowCFD V2.0 - 多层预处理：聚合粗空间 + 粗层 K 循环 + 各层块 ILU 光滑。

## 为什么需要（plate_demo P0，17.9 万单元，检查点续跑 12 步，CFL 1e3 -> 1e4）

大 CFL 下 `I/dtau + J` 由 `J` 主导，全局的光滑模态（低马赫下的压力-速度椭圆型耦合、
对流的长程传播）块 ILU 只能一层一层地传。接近收敛时残差里剩下的恰恰是这些模态：

    CFL 1e4、eta = 0.1        块 ILU      多层（本文件）
    每步 GMRES 次数             155~164      48~52
    每步墙钟                    43~46 s      18~19 s
    11 步合计                   498 s        226 s

**判断难度必须用真实右端项 `-R`**：用随机解造的右端项，块 ILU 在同一矩阵上 9 次就到
1e-1，真实运行却要 160 次——慢模态在随机向量里只占很小的分量。Blasius P2（1152 单元，
只有 p 层）CFL 1e4 段 GMRES 合计 413 -> 293 次，但单次残差求值太便宜，墙钟不降
（62 s -> 66 s）；plate_demo P1 在其真实 CFL（约 30）下块 ILU 6 次就到 1e-2，两者持平。

## 层次（`preconditioner_hierarchy`）

P>=1 先做一层 p 粗化（每单元一个节点：单元内全部真实解点上取同一值），再在单元面邻接
图上逐层聚合（每层约 6 倍粗化，`aggregation.py`），最粗层不超过 `COARSEST_DOF` 个未知量、
做稀疏直接解。`A_0` 是块预处理已经装配好的矩阵（对角块 + 面邻居耦合块 + `I/dtau`），
`A_{k+1} = P_k^T A_k P_k`（Galerkin，`galerkin.py`），粗层每节点 `n_var` 个未知量、PTC 项
已并进对角块。各层矩阵依赖 `dtau`，每个 `dtau` 档重新构造与分解；层次只依赖拓扑。

## 作用：K 循环（Notay & Vassilevski 2008）

    第 k 层两层作用    T_k(r) = z1 + M_k^{-1}(r - A_k z1)，  z1 = P_k W_{k+1}(P_k^T r)
    粗层近似逆         W_k   = 以 T_k 为预处理的 K_CYCLE_INNER 步灵活 GMRES（k >= 1）
    最粗层             W_L   = A_L^{-1}（稀疏 LU）

`M_k` 是该层的块 ILU（细层就是块 ILU 预处理子本身）。粗层只做一次 ILU 光滑时近似逆太弱：
粗层精确求解（25313 个聚合体，LU 构造 425 s）的两层法要 90 次，V(1,0) 循环 142 次，
V(1,1) 发散（ILU 的 Richardson 迭代谱半径大于 1），W 循环 145 次；K 循环 3 步内层 101 次、
8 步 90 次，与精确粗层持平。内层步数（plate_demo P0 后期系统，到 1e-1）：1 步 153 次、
2 步 58 次、3 步 51 次、5 步 45 次（单次作用更贵，墙钟反而多），取 3。内层 Krylov 让
预处理随右端项变化，外层 GMRES 走灵活模式（`gmres.py`，`flexible = True`）。
"""

import numpy as np

from ..block_ilu import BlockCouplingStructure, BlockILUPreconditioner, factor_block_ilu, solve_block_ilu
from ..gmres import gmres_right
from ..preconditioner import PseudoTransientDiagonal
from .galerkin import assembled_matvec, galerkin_coarse_matrix, prolong, restrict

#: 最粗层未知量上限：不超过它就做稀疏直接解（plate_demo P0 实测 8e3 量级的 LU 分解
#: 与回代都在 0.1 s 以内；聚合层次的最粗节点数按它除以 n_var 取）。
COARSEST_DOF = 8000

#: 粗层（第 1 层起、最粗层除外）上内层灵活 GMRES 的固定步数。
K_CYCLE_INNER = 3


def preconditioner_hierarchy(struct: BlockCouplingStructure, high_order: bool, coarsest_nodes: int = None):
    """块预处理的聚合层次（拓扑量，缓存在 `BlockJacobiCache` 里跨步复用）。

    P>=1（`high_order`）先做一层 p 粗化：每单元一个节点（单元内全部真实解点上取同一值，
    即单元均值模态），再在单元面邻接图上逐层聚合，直到节点数不超过 `coarsest_nodes`
    （缺省 `COARSEST_DOF/n_var`）。返回空列表表示单元数已不超过最粗层上限且没有 p 层
    可做（P0 小网格），用块 ILU 本身。聚合是确定性的：上限更小时得到的层次以上限更大
    时的层次为前缀（分布式全局粗层据此与本 rank 的多层共用同一组聚合，`global_coarse.py`）。
    """
    from .aggregation import build_hierarchy

    if coarsest_nodes is None:
        coarsest_nodes = max(1, COARSEST_DOF // struct.n_var)
    n = struct.n_real.size
    levels = [(np.arange(n, dtype=np.int64), n)] if high_order else []
    return levels + build_hierarchy(struct.indptr, struct.cols, n, coarsest_nodes)


class _Level:
    """一层块稀疏矩阵（对角块扁平存储 + 耦合块 CSR）及其 ILU 分解。"""

    __slots__ = ("diag", "off", "struct", "inv_dtau", "n_sps", "inv")

    def __init__(self, diag, off, struct, inv_dtau, n_sps, factor: bool):
        self.diag, self.off, self.struct = diag, off, struct
        self.inv_dtau = inv_dtau
        self.n_sps = int(n_sps)
        self.inv = factor_block_ilu(diag, off, struct, inv_dtau, self.n_sps) if factor else None

    def matvec(self, x):
        s = self.struct
        out = np.empty_like(x)
        assembled_matvec(x, self.diag, self.off, s.row_dof, self.inv_dtau, s.indptr, s.cols, s.offset, s.data,
                         self.n_sps, s.n_var, out)
        return out

    def coarse_matrix(self, agg, n_agg):
        s = self.struct
        return galerkin_coarse_matrix(self.diag, self.off, s.row_dof, self.inv_dtau, s, agg, n_agg,
                                      self.n_sps, s.n_var)


def _level_from_scipy(A, n_var: int) -> _Level:
    """Galerkin 粗矩阵（scipy）-> 块稀疏层（每节点 1 个解点，PTC 已在对角块里）。"""
    B = A.tobsr(blocksize=(n_var, n_var))
    B.sum_duplicates()
    B.sort_indices()
    n = B.shape[0] // n_var
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(B.indptr))
    cols = np.asarray(B.indices, dtype=np.int64)
    blocks = np.asarray(B.data, dtype=np.float64)
    on_diag = rows == cols
    diag = np.zeros((n, n_var, n_var))
    diag[rows[on_diag]] = blocks[on_diag]
    off_rows, off_cols, off_blocks = rows[~on_diag], cols[~on_diag], blocks[~on_diag]
    indptr = np.concatenate([[0], np.cumsum(np.bincount(off_rows, minlength=n))]).astype(np.int64)
    struct = BlockCouplingStructure.from_csr(
        indptr, off_cols, np.arange(off_cols.size, dtype=np.int64) * n_var * n_var, off_blocks.reshape(-1),
        np.ones(n, dtype=np.int64), n_var)
    off = np.arange(n, dtype=np.int64) * n_var * n_var
    return _Level(diag.reshape(-1).astype(np.float32), off, struct, np.zeros(n), 1, factor=True)


class MultilevelPreconditioner(PseudoTransientDiagonal):
    """接口与对角预处理相同（`add_ptc_term` / `apply`）；向量是 cupy 数组时在主机上作用。"""

    __slots__ = ("_fine", "_smoother", "_levels", "_hierarchy", "_lu", "_n_var", "_xp")

    flexible = True             # K 循环的粗层内层 Krylov 让作用随右端项变化

    def __init__(self, diag, off, struct: BlockCouplingStructure, hierarchy, dtau_flat, n_sps: int):
        """`diag/off`：细层对角块（`block_ilu.flatten_cell_blocks`）；`hierarchy`：
        `preconditioner_hierarchy` 的结果（至少一层）。"""
        from scipy.sparse.linalg import splu

        from autoflowcfd.core.utils.array_module import array_module

        if not hierarchy:
            raise ValueError("多层预处理至少需要一层聚合（单元数已不超过最粗层上限时用块 ILU 本身）")
        self._xp = array_module(dtau_flat)
        super().__init__(dtau_flat, struct.n_var)
        dtau_host = np.asarray(dtau_flat.get() if hasattr(dtau_flat, "get") else dtau_flat,
                               dtype=np.float64).ravel()
        self._n_var = struct.n_var
        self._fine = _Level(diag, off, struct, 1.0 / dtau_host, n_sps, factor=False)
        # 细层光滑沿用块 ILU 预处理子（它处理零填充槽位：dtau * v）
        self._smoother = BlockILUPreconditioner(diag, off, struct, dtau_host, n_sps)
        self._hierarchy = hierarchy
        self._levels = []
        level = self._fine
        for k, (agg, n_agg) in enumerate(hierarchy):
            A_c = level.coarse_matrix(agg, n_agg)
            if k == len(hierarchy) - 1:
                self._lu = splu(A_c.tocsc())
            else:
                level = _level_from_scipy(A_c, self._n_var)
                self._levels.append(level)

    def _smooth(self, k: int, level, r):
        if k == 0:
            return self._smoother.apply(r.reshape(-1, self._n_var)).reshape(-1)
        return solve_block_ilu(level.inv, level.off, level.struct, r, np.zeros_like(r), 1)

    def _two_grid(self, k: int, level, r):
        """第 k 层的一次两层作用：粗校正在前、一次 ILU 后光滑（见模块文档）。"""
        agg, n_agg = self._hierarchy[k]
        s = level.struct
        rc = restrict(r, s.row_dof, agg, n_agg, level.n_sps, s.n_var)
        z = np.empty_like(r)
        prolong(self._solve_level(k + 1, rc), s.row_dof, agg, level.n_sps, s.n_var, z)
        return z + self._smooth(k, level, r - level.matvec(z))

    def _solve_level(self, k: int, r):
        """近似 `A_k^{-1} r`（k >= 1）：最粗层直接解，其余层做 `K_CYCLE_INNER` 步以本层
        两层作用为预处理的灵活 GMRES（K 循环）。"""
        if k == len(self._hierarchy):
            return self._lu.solve(r)
        level = self._levels[k - 1]
        x, _, info, _ = gmres_right(level.matvec, r, lambda v: self._two_grid(k, level, v), rtol=0.0,
                                    restart=K_CYCLE_INNER, max_iter=K_CYCLE_INNER, flexible=True)
        if info < 0:
            # 粗层出现非有限值：如实传给外层 GMRES（它据此判线性求解失败、缩 dtau），
            # 不以零校正之类的值静默顶替
            return np.full_like(r, np.nan)
        return x

    def apply(self, v_flat):
        shape = v_flat.shape
        x = v_flat.get() if self._xp is not np else v_flat
        z = self._two_grid(0, self._fine, np.ascontiguousarray(x, dtype=np.float64).reshape(-1)).reshape(shape)
        return z if self._xp is np else self._xp.asarray(z)
