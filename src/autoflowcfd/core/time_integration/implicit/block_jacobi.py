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

## 装配

着色有限差分装配（`cell_blocks.py::CellBlockJacobian`，成本与单元数无关、
P0 顺带给出面邻居耦合块）或解析装配（下一节）。

## 解析装配（P>=1 的默认）

差分装配要 `色数 x 真实解点数 x 5` 次整场残差求值，P2 起不可承受（plate_demo
P2 450 次、单次 5.5 s）。P>=1 由各后端传入解析装配器
（`fr_residual/jacobian`，逐点导数与参考算子的收缩，同一个矩阵、代价约十几次
残差求值），它同时给出面邻居耦合块，合计预算放得下（`plan_block_mode`）时预处理
换成块 ILU（`block_ilu.py`，大 CFL 下 GMRES 迭代数约为块 Jacobi 的 0.4 倍）。
P0 与解析装配不覆盖的离散（见 `fr_residual/jacobian/backend.py::unsupported_reason`）
仍用差分装配。

## 复用与刷新

`J_cc` 只依赖状态，不依赖 `dtau`：每个 Newton 步只按新的 `dtau` 重新
求逆（逐块 `J_cc + diag(1/dtau)`，廉价），`J_cc` 本身跨步复用，直到

* 已经复用了 `MAX_AGE` 个 Newton 步；或
* 上一步的 GMRES 迭代数超过"刚装配完那一步"的基线加上**一次装配折合的迭代数**
  （`装配耗时 / 单次 GMRES 迭代耗时`，两者都是实测值，下限 `REFRESH_SLACK_MIN`）：
  多出来的迭代比重装配一次还贵时才重装配。此前是固定的 `2 倍 + 10`（差分）/
  `1.5 倍 + 3`（解析）：plate_demo P1 基线 3 次时阈值只有 7 次，GMRES 8 次就触发
  一次 9~11 s 的装配，而多一次迭代只要约 1 s，单步耗时从约 20 s 涨到 40~63 s；
  反过来 P0 差分装配（15 s）对 0.1 s 的迭代又刷新得太勤。分布式下两个耗时取全局
  最大值（阈值决定 GMRES 迭代上限，各 rank 必须一致）；或
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
GMRES 解的精度。

合计预算（`block_budget.py`，平均流与湍流共享）放不下两份、放得下一份时用**单份**（冻结 dtau）：
装配后立即按当时的 `dtau` 原地求逆、不再保留 `J_cc`；之后每步直接用这份逆，
当前 `dtau` 与求逆时的 `dtau`（几何平均比值）相差超过 `DTAU_REBUILD_RATIO` 倍才
重装配。预处理子里的 `dtau` 略微过时只影响迭代数（它仍是合法的预处理子），迭代
数上升由上面的刷新判据兜住。plate_demo P3（1.99 万棱柱块 200x200 + 15.9 万四面体块
100x100）两份 19.1 GiB、一份 9.5 GiB：与求解器本身（残差峰值约 11.8 GiB）和紧凑
Krylov 基（约 4.8 GiB）共存于 31.5 GB 开发机只能是一份。

一份也放不下时不装配、退回对角预处理并打 WARNING（例如 P3 棱柱一块 200x200，
79 万单元就要上百 GB）。
"""

from typing import Optional

import numpy as np
from loguru import logger
from numba import njit, prange

from . import block_budget
from .block_budget import block_mode_bytes, plan_block_mode
from .cell_blocks import CellBlockJacobian
from .preconditioner import PseudoTransientDiagonal
from .reductions import LocalReductions

#: 一份 `J_cc` 最多复用多少个 Newton 步。
MAX_AGE = 20

#: 迭代数刷新判据的下限松弛：`iters > 基线 + max(REFRESH_SLACK_MIN, 装配耗时/单次迭代耗时)`
#: （见模块文档"复用与刷新"）。
REFRESH_SLACK_MIN = 3

#: 单次 GMRES 迭代耗时的指数滑动平均权重（新值占比）。
_ITER_SECONDS_EMA = 0.5

#: 单份模式：当前 dtau 与求逆时 dtau 的几何平均比值超过这个倍数（任一方向）就重装配。
DTAU_REBUILD_RATIO = 2.0

class CellBlockJacobiPreconditioner(PseudoTransientDiagonal):
    """`M = blockdiag(J_cc + I/dtau)` 的逆作用；接口与对角预处理相同。

    零填充槽位不在任何块里：那些行的残差恒为零、Jacobian 行也为零，
    `A` 在那里就是 `I/dtau`，所以它们的预处理保持对角形式 `dtau * v`
    （继承自 `PseudoTransientDiagonal.apply`，先整体作用再被块覆盖）。
    """

    __slots__ = ("_jac", "_inv_prism", "_inv_tet", "_rows_prism", "_rows_tet")

    def __init__(self, jac: CellBlockJacobian, dtau_flat, inverted: bool = False):
        """`inverted=True`：`jac` 的块已经是 `inv(J_cc + diag(1/dtau_build))`（单份模式，
        `invert_blocks_in_place`），直接使用；`dtau_flat` 只用于零填充行的对角部分。"""
        super().__init__(dtau_flat, jac.n_var)
        self._jac = jac
        self._rows_prism, self._rows_tet = _block_rows(jac)
        if inverted:
            self._inv_prism, self._inv_tet = jac.blocks_prism, jac.blocks_tet
        else:
            self._inv_prism = _inverted_blocks(jac, jac.blocks_prism, self._rows_prism, self.dtau, out=None)
            self._inv_tet = _inverted_blocks(jac, jac.blocks_tet, self._rows_tet, self.dtau, out=None)

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


def _block_rows(jac: CellBlockJacobian):
    """棱柱块、四面体块各自覆盖的状态行（真实解点）。"""
    xp, n_sps = jac.xp, jac.n_sps
    return (xp.asarray(jac.prism_cells[:, None] * n_sps + np.arange(jac.n_real_prism)[None, :]),
            xp.asarray(jac.tet_cells[:, None] * n_sps + np.arange(jac.n_real_tet)[None, :]))


def _inverted_blocks(jac: CellBlockJacobian, blocks, rows, dtau, out):
    """逐块 `inv(blocks + diag(1/dtau))`（float32）；`out is blocks` 时原地。"""
    xp, nv = jac.xp, jac.n_var
    if blocks.shape[0] == 0:
        return blocks if out is not None else xp.empty_like(blocks)
    diag = xp.ascontiguousarray(xp.repeat(1.0 / dtau[rows], nv, axis=1))  # 与块内 (s,v) 序一致
    if xp is np:
        res = np.empty_like(blocks) if out is None else out
        _invert_blocks_plus_diag(blocks, diag, res)
        return res
    # cupy：一次批量 cuSOLVER 调用（GPU 上没有 CPU LAPACK 逐矩阵调用的开销）
    a = blocks.astype(xp.float64)
    idx = xp.arange(blocks.shape[1])
    a[:, idx, idx] += diag
    inv = xp.linalg.inv(a).astype(xp.float32)
    if out is None:
        return inv
    out[...] = inv
    return out


def invert_blocks_in_place(jac: CellBlockJacobian, dtau_flat) -> None:
    """单份模式：把 `jac` 的块原地换成 `inv(J_cc + diag(1/dtau))`（见模块文档"内存"）。"""
    dtau = jac.xp.asarray(dtau_flat, dtype=jac.xp.float64).ravel()
    rows_prism, rows_tet = _block_rows(jac)
    _inverted_blocks(jac, jac.blocks_prism, rows_prism, dtau, out=jac.blocks_prism)
    _inverted_blocks(jac, jac.blocks_tet, rows_tet, dtau, out=jac.blocks_tet)


@njit(parallel=True, cache=True)
def _invert_blocks_plus_diag(blocks, diag, out):
    """逐块 `inv(blocks[b] + diag(diag[b]))`：float64 Gauss-Jordan、部分选主元，
    结果写回 float32 的 `out`。

    **为什么不用 `np.linalg.inv` 的批量形式**：它对每个小矩阵单独走一次
    LAPACK，调用开销与 BLAS 线程调度远大于 30x30 本身的计算量——实测
    10.7 万个 30x30 块要 107 s（本核并行版不到 1 s），在每个 Newton 步都要
    按新 `dtau` 重新求逆的前提下不可接受。

    `out` 可以就是 `blocks`（原地）：每块先整块读入局部数组再写回。
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
                 "disabled_reason", "n_builds", "red", "n_var", "assembler", "use_ilu", "coupling",
                 "_coupling_graph_fn", "_coupling_graph", "build_seconds", "iter_seconds",
                 "single_copy", "_inverted_dtau", "_coarse", "cross")

    def __init__(self, *, cell_is_prism, colors: np.ndarray, n_sps: int,
                 n_real_prism: int, n_real_tet: int, n_var: int,
                 red: LocalReductions = None, coupling_graph=None, with_turbulence: bool = False,
                 global_coarse=None):
        """`colors`：块装配用的单元着色（单机 `coloring.greedy_cell_coloring`；分布式
        是全局一致着色里本 rank 那一段，保证同色单元跨 rank 也不相邻）。

        `assembler`（每个 Newton 步由调用方设置，见 `begin_step`）：解析装配器
        `(u0_flat, r0_flat) -> (blocks_prism, blocks_tet)`；为 None 时用着色差分装配。

        `coupling_graph`：可选回调 `() -> coloring.CouplingGraph`，只在首次需要时调用。
        每单元一个真实解点（P0）且没有解析装配器时，差分装配按它的距离 2 着色同时给出
        面邻居耦合块，预处理用块 ILU（见 `cell_blocks.py` 模块文档）。

        `with_turbulence`：平均流缓存是否与隐式湍流缓存共享预算（`plan_block_mode`）。

        `global_coarse`（`coarse.CoarseCommContext`，分布式后端给出）：块 ILU 档在本 rank 的
        预处理之外再叠全局粗校正（`coarse/global_coarse.py`），装配器须同时给出跨 rank 耦合块。
        """
        self.red = red if red is not None else LocalReductions()
        self.n_var = int(n_var)
        self.assembler = None
        self.coupling = None
        self._coupling_graph_fn = coupling_graph
        self._coupling_graph = None
        self.build_seconds = None
        self.iter_seconds = None
        self.single_copy = False
        self._inverted_dtau = None
        self.cross = None
        self.cell_is_prism = np.asarray(cell_is_prism, dtype=bool)
        self.n_sps, self.n_real_prism, self.n_real_tet = n_sps, n_real_prism, n_real_tet
        self.jac: Optional[CellBlockJacobian] = None
        self.age = 0
        self.baseline_iters = None
        self.last_iters = None
        self.last_accepted = True
        self.n_builds = 0
        sizes = (int(self.cell_is_prism.sum()), int((~self.cell_is_prism).sum()), n_real_prism, n_real_tet)
        mode = plan_block_mode(self.n_var, *sizes, with_turbulence=with_turbulence)
        # 块 ILU 要面邻居耦合块（解析装配或 P0 差分装配给出）
        self.use_ilu = mode == "ilu"
        self.single_copy = mode == "single"
        self.disabled_reason = None
        from .coarse import CoarsePreconditionerFactory
        self._coarse = CoarsePreconditionerFactory(max(n_real_prism, n_real_tet) > 1, global_coarse)
        self.colors = np.asarray(colors, dtype=np.int64)
        if mode is None:
            self.disabled_reason = (
                f"单元块 Jacobi（n_var={self.n_var}）一份也需要 "
                f"{block_mode_bytes('single', *sizes, self.n_var) / 2 ** 30:.1f} GiB，超出预处理合计预算 "
                f"{block_budget.PRECOND_TOTAL_BYTES / 2 ** 30:.0f} GiB，改用逐 SP 对角预处理——大 CFL 下 GMRES "
                f"迭代数会显著上升")
            logger.warning("[NK] " + self.disabled_reason)
            self.colors = None
        else:
            logger.info(f"[NK] 单元块预处理（n_var={self.n_var}）：{mode}，"
                        f"{block_mode_bytes(mode, *sizes, self.n_var) / 2 ** 30:.2f} GiB"
                        f"（合计预算 {block_budget.PRECOND_TOTAL_BYTES / 2 ** 30:.0f} GiB）")

    def _refresh_threshold(self, baseline: int) -> int:
        slack = REFRESH_SLACK_MIN
        if self.build_seconds is not None and self.iter_seconds:
            slack = max(slack, self.build_seconds / self.iter_seconds)
        return int(baseline + slack)

    def _needs_rebuild(self, dtau_flat) -> bool:
        if self.jac is None or self.age >= MAX_AGE or not self.last_accepted:
            return True
        if self.single_copy and self._dtau_drifted(dtau_flat):
            return True
        if self.baseline_iters is not None and self.last_iters is not None:
            return self.last_iters > self._refresh_threshold(self.baseline_iters)
        return False

    def _dtau_drifted(self, dtau_flat) -> bool:
        """单份模式：当前 dtau 相对求逆时 dtau 的几何平均比值是否超过 `DTAU_REBUILD_RATIO`
        （几何平均不让少数被局部缩小 dtau 的行触发整场重装配）。分布式下取全局平均。"""
        xp = self.red.xp
        log_ratio = xp.log(xp.asarray(dtau_flat, dtype=xp.float64).ravel() / self._inverted_dtau)
        mean = self.red.sum(log_ratio) / max(self.red.count(log_ratio), 1.0)
        return abs(mean) > np.log(DTAU_REBUILD_RATIO)

    def stale_budget(self) -> Optional[int]:
        """复用中的 `J_cc`（`age > 0`）在本步的 GMRES 迭代预算：超过它就说明
        线性化已明显过时（与跨步刷新判据同一个阈值）。刚装配的块、或没有
        基线时返回 None（用满全部预算）。"""
        if self.disabled_reason is not None or self.jac is None or self.age == 0:
            return None
        if self.baseline_iters is None:
            return None
        return self._refresh_threshold(self.baseline_iters)

    def refresh(self, residual, u0_flat, r0_flat, scales, dtau_flat) -> None:
        """当场按本步基态重装配（本步 GMRES 超出 `stale_budget` 时调用）。"""
        self._build(residual, u0_flat, r0_flat, scales, dtau_flat, reason="本步 GMRES 超出过时预算")

    def begin_step(self, residual, u0_flat, r0_flat, scales, dtau_flat) -> None:
        """每个 Newton 步开始时调用一次：按刷新判据决定是否重装配 `J_cc`。

        必须与 `preconditioner()` 分开：一个 Newton 步内 `dtau` 逐档缩小
        重试时每档都要重新求逆，但 `J_cc` 只依赖基态，不能每档重装配。

        `dtau_flat`：本步的伪时间步长（缩档之前）。单份模式据它判断存下的逆是否已经
        过时（`_dtau_drifted`），重装配时也按它求逆。
        """
        if self.disabled_reason is not None or not self._needs_rebuild(dtau_flat):
            return
        self._build(residual, u0_flat, r0_flat, scales, dtau_flat, reason="刷新判据")

    def _build(self, residual, u0_flat, r0_flat, scales, dtau_flat, *, reason: str) -> None:
        import time

        t0 = time.time()
        # 先释放旧块再装配：新旧两份同时存在时 P3 单份模式要 2 x 8.9 GiB（plate_demo 实测
        # 步内 refresh 时 OOM）；调用方同样先丢掉持有旧块的预处理对象（jfnk.py）
        self.jac = None
        self.coupling = None
        self.cross = None
        if self.assembler is not None:
            self.assembler.want_coupling = self.use_ilu
            out = self.assembler(u0_flat, r0_flat)
            blocks_prism, blocks_tet = out[0], out[1]
            if self.use_ilu:
                from .block_ilu import BlockCouplingStructure
                n_real = np.where(self.cell_is_prism, self.n_real_prism, self.n_real_tet)
                self.coupling = BlockCouplingStructure(out[2], self.cell_is_prism.size, n_real, self.n_var)
                self.cross = self._coarse.cross_coupling(out[3] if len(out) > 3 else None, self.coupling)
            self.jac = CellBlockJacobian.from_blocks(
                blocks_prism, blocks_tet, n_sps=self.n_sps, n_var=self.n_var,
                cell_is_prism=self.cell_is_prism, n_real_prism=self.n_real_prism,
                n_real_tet=self.n_real_tet, xp=self.red.xp)
        else:
            graph = self._fd_coupling_graph()
            self.jac = CellBlockJacobian(
                residual, u0_flat, r0_flat, scales, n_sps=self.n_sps,
                cell_is_prism=self.cell_is_prism, n_real_prism=self.n_real_prism,
                n_real_tet=self.n_real_tet, colors=self.colors, red=self.red, coupling_graph=graph)
            if graph is not None:
                from .block_ilu import BlockCouplingStructure
                self.coupling = BlockCouplingStructure(
                    self.jac.coupling, self.cell_is_prism.size, np.ones(self.cell_is_prism.size, np.int64),
                    self.n_var)
                self.cross = self._coarse.cross_coupling(self.jac.cross_coupling, self.coupling)
        if self.single_copy:
            invert_blocks_in_place(self.jac, dtau_flat)
            self._inverted_dtau = self.red.xp.asarray(dtau_flat, dtype=self.red.xp.float64).ravel().copy()
        self.age = 0
        self.baseline_iters = None
        self.last_iters = None
        self.last_accepted = True
        self.n_builds += 1
        self.build_seconds = self.red.max(self.red.xp.asarray([time.time() - t0]))
        logger.info(f"[NK] 单元块 Jacobian 重装配（第 {self.n_builds} 次，{reason}，"
                    f"{'块 ILU' if self.coupling is not None else '块 Jacobi'}"
                    f"{'（单份，冻结 dtau）' if self.single_copy else ''}，"
                    + ("解析装配" if self.assembler is not None
                       else f"差分装配 {self.jac.n_residual_evals} 次残差求值")
                    + f"，{time.time() - t0:.1f}s）")

    def _fd_coupling_graph(self):
        """差分装配要不要同时截取耦合块：P0（每单元一个真实解点）、内存允许块 ILU、
        后端给了耦合图时返回 `CouplingGraph`，否则 None（只装配对角块）。"""
        if self._coupling_graph_fn is None or not self.use_ilu or max(self.n_real_prism, self.n_real_tet) != 1:
            return None
        if self._coupling_graph is None:
            self._coupling_graph = self._coupling_graph_fn()
        return self._coupling_graph

    @property
    def flexible(self) -> bool:
        """`preconditioner()` 给出的是否是非线性预处理（多层 K 循环），GMRES 据此走灵活模式。
        `begin_step` 之后才有定论（耦合结构由首次装配给出）。"""
        return self.disabled_reason is None and self.coupling is not None and self._coarse.flexible(self.coupling)

    def preconditioner(self, dtau_flat: np.ndarray, n_var: int):
        """给定 `dtau` 下的预处理子（`begin_step` 之后调用，可多次）。块 ILU 档交给
        `coarse/selection.py`（本地多层或块 ILU，分布式再叠全局粗校正）。"""
        if self.disabled_reason is not None:
            return PseudoTransientDiagonal(dtau_flat, n_var)
        if self.coupling is not None:
            return self._coarse.make(self.jac, self.coupling, self.cross, dtau_flat)
        return CellBlockJacobiPreconditioner(self.jac, dtau_flat, inverted=self.single_copy)

    def record(self, gmres_iters: int, accepted: bool, gmres_seconds: Optional[float] = None) -> None:
        """一个 Newton 步结束后调用：更新刷新判据的依据。

        `gmres_seconds`：得到 `gmres_iters` 的那次 GMRES 求解的墙钟耗时（本 rank），
        用来估计单次迭代耗时（全局取最大，见模块文档"复用与刷新"）。
        """
        if self.disabled_reason is not None:
            return
        if gmres_seconds is not None and gmres_iters > 0:
            per_iter = self.red.max(self.red.xp.asarray([gmres_seconds / gmres_iters]))
            self.iter_seconds = (per_iter if self.iter_seconds is None
                                 else _ITER_SECONDS_EMA * per_iter + (1.0 - _ITER_SECONDS_EMA) * self.iter_seconds)
        if self.baseline_iters is None:
            self.baseline_iters = gmres_iters
        self.last_iters = gmres_iters
        self.last_accepted = accepted
        self.age += 1
