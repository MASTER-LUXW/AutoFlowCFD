"""AutoFlowCFD V2.0 - PTC-Newton-Krylov 的单元块 Jacobi 预处理（着色有限差分装配）。

## 为什么需要（真实测量）

`preconditioner.py` 的逐 SP 对角预处理 `P = I/dtau` 只在 `dtau` 小的时候
好用：`dtau` 大时 `I/dtau + J` 由 `J` 主导，而 FR/DG 的 `J` 在单元内部是
稠密的（高阶模态之间的刚性耦合），对角预处理完全看不到它。plate_demo
P1 层流（17.9 万单元，预处理 NK，CFL 34，同一状态、同一 `rtol=0.1`）：

    对角预处理     GMRES 300 次仍未达到容差（真实相对残差 0.126），925 s
    单元块 Jacobi   GMRES  20 次达到（0.067），65 s（含每次迭代的块作用）

同一次运行里 SER 律把 CFL 推到 30~50 时，对角预处理下每个 Newton 步
要 60~150 次 GMRES（每步 3~7 分钟），成了整个隐式路径的瓶颈。

## 装配：着色有限差分，成本与单元数无关

块 `J_cc = dR_c / dU_c`（单元 `c` 的全部真实解点 x 全部变量）。按面
相邻关系给单元做**距离 1 着色**（相邻单元不同色），同色单元**同时**
扰动同一个 `(解点 s, 变量 v)`：单元 `c` 的残差只受它自己与面邻居的状态
影响，而面邻居与它不同色、没有被扰动，于是

    [R(U + h e_{色k,s,v}) - R(U)] / h   在色 k 的每个单元 c 上
                                         恰好是 J_cc 的第 (s,v) 列

总残差求值次数 = `色数 x 真实解点数上限 x 变量数`，**与单元数无关**
（plate_demo P1：5 色 x 6 x 5 = 150 次）。旧文档里"每单元 `n_sps*n_var`
次残差求值、成本不可接受"的论证漏掉了着色，已同步更正。

**近似之处（如实说明）**：BR1 粘性项的梯度提升让单元 `c` 的残差还依赖
**距离 2** 的单元（邻居的梯度用到邻居的邻居）；距离 1 着色下，同色的
距离 2 单元的扰动会以粘性耦合的量级混入 `J_cc` 的列。预处理子不需要
精确——GMRES 对 `A` 本身仍是精确求解，预处理只影响迭代数——上面的实测
就是在这个近似下得到的。距离 2 着色色数约是距离 1 的 3~4 倍，装配成本
同比上升，不划算。

## 复用与刷新

`J_cc` 只依赖状态，不依赖 `dtau`：每个 Newton 步只按新的 `dtau` 重新
求逆（逐块 `J_cc + diag(1/dtau)`，廉价），`J_cc` 本身跨步复用，直到

* 已经复用了 `MAX_AGE` 个 Newton 步；或
* 上一步的 GMRES 迭代数超过"刚装配完那一步"的 `2 倍 + 10`（线性化
  已经明显过时）；或
* 上一步没有被接受（`theta == 0`）；或
* **本步**的 GMRES 在上一条的迭代预算内没有达到容差（`stale_budget` /
  `refresh`，由 `jfnk.py` 当场调用）。此前只按上一步的迭代数决定下一步是否
  重装配：plate_demo P1+SST 上湍流块在一步里从 10 次跳到 200 次（用满、只到
  0.29），平均流同一步也用满 200 次，那一步耗时 1173 s——过时的预处理把整步
  的迭代预算烧完，重装配要等下一步才发生。

冻结 Jacobian 的预处理是隐式 CFD 的标准做法；过时的预处理子仍然是合法
的预处理子，只是迭代数上升，由上面的刷新判据兜住。

## 内存

逐单元只存**真实**自由度（原生基的零填充槽位不进块，见
`fr/native_padding.py::real_sps_per_cell`）。`J_cc` 与逆各一份，float32
存储（逐块求逆在 float64 下做）：预处理子的精度只影响迭代数，不影响
GMRES 解的精度。超过 `MAX_BYTES` 时不装配、退回对角预处理并打 WARNING
（例如 P3 棱柱一块 200x200，79 万单元就要上百 GB）。
"""

from typing import Optional

import numpy as np
from loguru import logger
from numba import njit, prange

from .preconditioner import PseudoTransientDiagonal
from .reductions import LocalReductions

#: 一份 `J_cc` 最多复用多少个 Newton 步。
MAX_AGE = 20

#: 迭代数刷新判据：`iters > REFRESH_FACTOR * 基线 + REFRESH_SLACK`。
REFRESH_FACTOR = 2.0
REFRESH_SLACK = 10

#: 块 `J_cc` + 逆两份（float32）允许的总字节数。8 GiB：本项目开发机与常见
#: 工作站上与求解器本身（P1 79 万单元状态约 0.2 GB/份 x 数十份工作数组）
#: 共存的上限。
MAX_BYTES = 8 * 2 ** 30

_SQRT_EPS = float(np.sqrt(np.finfo(np.float64).eps))


def greedy_cell_coloring(owner_cell: np.ndarray, neighbor_cell: np.ndarray,
                         n_cells: int) -> np.ndarray:
    """按面相邻关系的贪心距离 1 着色，返回 `(n_cells,)` 的色号。

    `neighbor_cell < 0` 的是边界面（不产生相邻关系）。色数不超过最大
    邻居数 + 1（棱柱 5 面、四面体 4 面）。
    """
    import scipy.sparse as sp

    own = np.asarray(owner_cell, dtype=np.int64)
    nb = np.asarray(neighbor_cell, dtype=np.int64)
    inner = nb >= 0
    a, b = own[inner], nb[inner]
    adj = sp.coo_matrix((np.ones(2 * a.size, dtype=np.int8), (np.r_[a, b], np.r_[b, a])),
                        shape=(n_cells, n_cells)).tocsr()
    indptr = adj.indptr.astype(np.int64)
    max_degree = int(np.diff(indptr).max()) if n_cells else 0
    return _greedy_color_csr(indptr, adj.indices.astype(np.int64), n_cells, max_degree + 2)


@njit(cache=True)
def _greedy_color_csr(indptr, indices, n_cells, n_mark):
    color = -np.ones(n_cells, dtype=np.int64)
    mark = -np.ones(n_mark, dtype=np.int64)       # mark[k] == c：色 k 已被 c 的邻居占用
    for c in range(n_cells):
        for j in range(indptr[c], indptr[c + 1]):
            k = color[indices[j]]
            if k >= 0:
                mark[k] = c
        k = 0
        while mark[k] == c:
            k += 1
        color[c] = k
    return color


def estimate_block_bytes(n_prism: int, n_tet: int, n_real_prism: int,
                         n_real_tet: int, n_var: int) -> int:
    """`J_cc` + 逆两份（float32）的总字节数。"""
    bp, bt = n_real_prism * n_var, n_real_tet * n_var
    return 2 * 4 * (n_prism * bp * bp + n_tet * bt * bt)


class CellBlockJacobian:
    """单元对角块 `J_cc = dR_c/dU_c`（棱柱、四面体各一组，只含真实自由度）。

    后端无关：状态/残差/块在 `red.xp`（numpy 或 cupy）上；着色与单元类型
    划分是主机端 numpy。**分布式约束**：每次残差求值都是集体操作，所有
    rank 必须调用同样多次——所以"本色本解点有没有要扰动的单元"用
    `red.max` 取全局结论，而不是本地没有就跳过（那会让各 rank 的调用
    次数不一致而死锁）。着色必须是全局一致的（同色单元跨 rank 也不相邻）。
    """

    __slots__ = ("n_sps", "n_var", "prism_cells", "tet_cells", "n_real_prism",
                 "n_real_tet", "blocks_prism", "blocks_tet", "n_residual_evals", "xp")

    def __init__(self, residual, u0_flat, r0_flat, scales: np.ndarray, *, n_sps: int,
                 cell_is_prism: np.ndarray, n_real_prism: int, n_real_tet: int,
                 colors: np.ndarray, red: LocalReductions = None):
        red = red if red is not None else LocalReductions()
        xp = self.xp = red.xp
        u0 = xp.ascontiguousarray(u0_flat, dtype=xp.float64)
        r0 = xp.ascontiguousarray(r0_flat, dtype=xp.float64)
        n_var = u0.shape[1]
        cell_is_prism = np.asarray(cell_is_prism, dtype=bool)
        n_cells = cell_is_prism.size
        if u0.shape[0] != n_cells * n_sps:
            raise ValueError(f"状态行数 {u0.shape[0]} != n_cells*n_sps = {n_cells}*{n_sps}")
        scales = np.asarray(scales, dtype=np.float64).ravel()
        self.n_sps, self.n_var = n_sps, n_var
        self.prism_cells = np.nonzero(cell_is_prism)[0]
        self.tet_cells = np.nonzero(~cell_is_prism)[0]
        self.n_real_prism, self.n_real_tet = n_real_prism, n_real_tet
        bp, bt = n_real_prism * n_var, n_real_tet * n_var
        self.blocks_prism = xp.zeros((self.prism_cells.size, bp, bp), dtype=xp.float32)
        self.blocks_tet = xp.zeros((self.tet_cells.size, bt, bt), dtype=xp.float32)
        # 单元号 -> 在各自块数组里的下标
        slot = np.empty(n_cells, dtype=np.int64)
        slot[self.prism_cells] = np.arange(self.prism_cells.size)
        slot[self.tet_cells] = np.arange(self.tet_cells.size)

        # 差分步长：与 `MatrixFreeJacobian` 同一取法——在按参考量级无量纲化
        # 的空间里取 `sqrt(eps_mach) * (1 + rms(U~))`，再换回物理量纲。
        u0_rms_scaled = red.rms(u0 / xp.asarray(scales)[None, :])
        h = _SQRT_EPS * (1.0 + u0_rms_scaled) * scales

        colors = np.asarray(colors, dtype=np.int64)
        n_colors = int(red.max(xp.asarray([colors.max() if colors.size else -1]))) + 1
        n_eval = 0
        for k in range(n_colors):
            in_k = colors == k
            groups = []
            for cells, blocks, n_real in ((np.nonzero(in_k & cell_is_prism)[0], self.blocks_prism, n_real_prism),
                                          (np.nonzero(in_k & ~cell_is_prism)[0], self.blocks_tet, n_real_tet)):
                if cells.size:
                    rows = cells[:, None] * n_sps + np.arange(n_real)[None, :]
                    groups.append((xp.asarray(slot[cells]), blocks, n_real,
                                   xp.asarray(rows), cells.size))
            local_s = max((g[2] for g in groups), default=0)
            n_s = int(red.max(xp.asarray([local_s])))       # 全局一致的调用次数
            for s in range(n_s):
                for v in range(n_var):
                    up = u0.copy()
                    for _, _, n_real, rows, _ in groups:
                        if s < n_real:
                            up[rows[:, s], v] += h[v]
                    dr = (xp.asarray(residual(up), dtype=xp.float64) - r0) / h[v]
                    n_eval += 1
                    col = s * n_var + v
                    for slots, blocks, n_real, rows, nc in groups:
                        if s < n_real:
                            blocks[slots, :, col] = dr[rows.ravel()].reshape(nc, n_real * n_var)
        self.n_residual_evals = n_eval


class CellBlockJacobiPreconditioner(PseudoTransientDiagonal):
    """`M = blockdiag(J_cc + I/dtau)` 的逆作用；接口与对角预处理相同。

    零填充槽位不在任何块里：那些行的残差恒为零、Jacobian 行也为零，
    `A` 在那里就是 `I/dtau`，所以它们的预处理保持对角形式 `dtau * v`
    （继承自 `PseudoTransientDiagonal.apply`，先整体作用再被块覆盖）。
    """

    __slots__ = ("_jac", "_inv_prism", "_inv_tet", "_rows_prism", "_rows_tet")

    def __init__(self, jac: CellBlockJacobian, dtau_flat):
        super().__init__(dtau_flat, jac.n_var)
        self._jac = jac
        xp = jac.xp
        n_sps, nv = jac.n_sps, jac.n_var
        self._rows_prism = xp.asarray(jac.prism_cells[:, None] * n_sps + np.arange(jac.n_real_prism)[None, :])
        self._rows_tet = xp.asarray(jac.tet_cells[:, None] * n_sps + np.arange(jac.n_real_tet)[None, :])
        self._inv_prism = self._invert(jac.blocks_prism, self._rows_prism, nv)
        self._inv_tet = self._invert(jac.blocks_tet, self._rows_tet, nv)

    def _invert(self, blocks, rows, nv: int):
        xp = self._jac.xp
        if blocks.shape[0] == 0:
            return xp.empty_like(blocks)
        diag = xp.ascontiguousarray(xp.repeat(1.0 / self.dtau[rows], nv, axis=1))  # 与块内 (s,v) 序一致
        if xp is np:
            out = np.empty_like(blocks)
            _invert_blocks_plus_diag(blocks, diag, out)
            return out
        # cupy：一次批量 cuSOLVER 调用（GPU 上没有 CPU LAPACK 逐矩阵调用的开销）
        a = blocks.astype(xp.float64)
        idx = xp.arange(blocks.shape[1])
        a[:, idx, idx] += diag
        return xp.linalg.inv(a).astype(xp.float32)

    def apply(self, v_flat):
        out = super().apply(v_flat)
        xp = self._jac.xp
        nv = self._jac.n_var
        for inv, rows in ((self._inv_prism, self._rows_prism), (self._inv_tet, self._rows_tet)):
            if inv.shape[0]:
                # float32 作用：逆以 float32 存储，把向量也降到 float32 再乘，避免
                # 混合精度 matmul 每次把整份逆隐式提升成 float64（实测 5 倍慢）
                x = v_flat[rows].reshape(rows.shape[0], -1, 1).astype(xp.float32)
                out[rows] = xp.matmul(inv, x).reshape(rows.shape[0], rows.shape[1], nv)
        return out


@njit(parallel=True, cache=True)
def _invert_blocks_plus_diag(blocks, diag, out):
    """逐块 `inv(blocks[b] + diag(diag[b]))`：float64 Gauss-Jordan、部分选主元，
    结果写回 float32 的 `out`。

    **为什么不用 `np.linalg.inv` 的批量形式**：它对每个小矩阵单独走一次
    LAPACK，调用开销与 BLAS 线程调度远大于 30x30 本身的计算量——实测
    10.7 万个 30x30 块要 107 s（本核并行版不到 1 s），在每个 Newton 步都要
    按新 `dtau` 重新求逆的前提下不可接受。
    """
    n_blk, m, _ = blocks.shape
    for b in prange(n_blk):
        a = np.empty((m, 2 * m))
        for i in range(m):
            for j in range(m):
                a[i, j] = blocks[b, i, j]
                a[i, m + j] = 0.0
            a[i, i] += diag[b, i]
            a[i, m + i] = 1.0
        for col in range(m):
            piv = col
            best = abs(a[col, col])
            for r in range(col + 1, m):
                if abs(a[r, col]) > best:
                    best = abs(a[r, col])
                    piv = r
            if piv != col:
                for j in range(2 * m):
                    tmp = a[col, j]
                    a[col, j] = a[piv, j]
                    a[piv, j] = tmp
            inv_p = 1.0 / a[col, col]
            for j in range(2 * m):
                a[col, j] *= inv_p
            for r in range(m):
                if r != col:
                    f = a[r, col]
                    if f != 0.0:
                        for j in range(2 * m):
                            a[r, j] -= f * a[col, j]
        for i in range(m):
            for j in range(m):
                out[b, i, j] = a[i, m + j]


class BlockJacobiCache:
    """跨 Newton 步持有 `J_cc` 并决定何时重装配（刷新判据见模块文档）。

    换阶（Order Continuation）后 `n_sps` 与块尺寸都变了，调用方必须丢弃
    整个缓存对象重建（`order_continuation/interp.py` 与 forcing 状态一起
    失效）。
    """

    __slots__ = ("cell_is_prism", "colors", "n_sps", "n_real_prism", "n_real_tet",
                 "jac", "age", "baseline_iters", "last_iters", "last_accepted",
                 "disabled_reason", "n_builds", "red")

    def __init__(self, *, cell_is_prism, colors: np.ndarray, n_sps: int,
                 n_real_prism: int, n_real_tet: int, n_var: int,
                 red: LocalReductions = None):
        """`colors`：块装配用的单元着色（单机 `greedy_cell_coloring`；分布式是
        全局一致着色里本 rank 那一段，保证同色单元跨 rank 也不相邻）。"""
        self.red = red if red is not None else LocalReductions()
        self.cell_is_prism = np.asarray(cell_is_prism, dtype=bool)
        self.n_sps, self.n_real_prism, self.n_real_tet = n_sps, n_real_prism, n_real_tet
        self.jac: Optional[CellBlockJacobian] = None
        self.age = 0
        self.baseline_iters = None
        self.last_iters = None
        self.last_accepted = True
        self.n_builds = 0
        n_prism = int(self.cell_is_prism.sum())
        need = estimate_block_bytes(n_prism, self.cell_is_prism.size - n_prism,
                                    n_real_prism, n_real_tet, n_var)
        if need > MAX_BYTES:
            self.disabled_reason = (
                f"单元块 Jacobi 需要 {need / 2 ** 30:.1f} GiB（上限 {MAX_BYTES / 2 ** 30:.0f} GiB），"
                f"改用逐 SP 对角预处理——大 CFL 下 GMRES 迭代数会显著上升")
            logger.warning("[NK] " + self.disabled_reason)
            self.colors = None
        else:
            self.disabled_reason = None
            self.colors = np.asarray(colors, dtype=np.int64)

    def _needs_rebuild(self) -> bool:
        if self.jac is None or self.age >= MAX_AGE or not self.last_accepted:
            return True
        if self.baseline_iters is not None and self.last_iters is not None:
            return self.last_iters > REFRESH_FACTOR * self.baseline_iters + REFRESH_SLACK
        return False

    def stale_budget(self) -> Optional[int]:
        """复用中的 `J_cc`（`age > 0`）在本步的 GMRES 迭代预算：超过它就说明
        线性化已明显过时（与跨步刷新判据同一个阈值）。刚装配的块、或没有
        基线时返回 None（用满全部预算）。"""
        if self.disabled_reason is not None or self.jac is None or self.age == 0:
            return None
        if self.baseline_iters is None:
            return None
        return int(REFRESH_FACTOR * self.baseline_iters + REFRESH_SLACK)

    def refresh(self, residual, u0_flat, r0_flat, scales) -> None:
        """当场按本步基态重装配（本步 GMRES 超出 `stale_budget` 时调用）。"""
        self._build(residual, u0_flat, r0_flat, scales, reason="本步 GMRES 超出过时预算")

    def begin_step(self, residual, u0_flat, r0_flat, scales) -> None:
        """每个 Newton 步开始时调用一次：按刷新判据决定是否重装配 `J_cc`。

        必须与 `preconditioner()` 分开：一个 Newton 步内 `dtau` 逐档缩小
        重试时每档都要重新求逆，但 `J_cc` 只依赖基态，不能每档重装配。
        """
        if self.disabled_reason is not None or not self._needs_rebuild():
            return
        self._build(residual, u0_flat, r0_flat, scales, reason="刷新判据")

    def _build(self, residual, u0_flat, r0_flat, scales, *, reason: str) -> None:
        import time

        t0 = time.time()
        self.jac = CellBlockJacobian(
            residual, u0_flat, r0_flat, scales, n_sps=self.n_sps,
            cell_is_prism=self.cell_is_prism, n_real_prism=self.n_real_prism,
            n_real_tet=self.n_real_tet, colors=self.colors, red=self.red)
        self.age = 0
        self.baseline_iters = None
        self.last_iters = None
        self.last_accepted = True
        self.n_builds += 1
        logger.info(f"[NK] 单元块 Jacobian 重装配（第 {self.n_builds} 次，{reason}，"
                    f"{self.jac.n_residual_evals} 次残差求值，{time.time() - t0:.1f}s）")

    def preconditioner(self, dtau_flat: np.ndarray, n_var: int):
        """给定 `dtau` 下的预处理子（`begin_step` 之后调用，可多次）。"""
        if self.disabled_reason is not None:
            return PseudoTransientDiagonal(dtau_flat, n_var)
        return CellBlockJacobiPreconditioner(self.jac, dtau_flat)

    def record(self, gmres_iters: int, accepted: bool) -> None:
        """一个 Newton 步结束后调用：更新刷新判据的依据。"""
        if self.disabled_reason is not None:
            return
        if self.baseline_iters is None:
            self.baseline_iters = gmres_iters
        self.last_iters = gmres_iters
        self.last_accepted = accepted
        self.age += 1
