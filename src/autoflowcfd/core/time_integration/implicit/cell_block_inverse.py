"""AutoFlowCFD V2.0 - 单元块 `J_cc + I/dtau` 的逐块求逆与块 Jacobi 预处理作用。

从 `block_jacobi.py` 拆出（2026-10-15，单文件不超 500 行）；块的装配、档位、刷新与复用（`BlockJacobiCache`）
仍在那里。逐块求逆在 float64 下做、存回 float32（`block_jacobi.py` 模块文档"内存"）。
"""

import numpy as np
from numba import njit, prange

from .cell_blocks import CellBlockJacobian
from .preconditioner import PseudoTransientDiagonal


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
        """`v_flat` 可以是非连续视图（块对角组合传入列切片）。"""
        out = super().apply(v_flat)
        xp = self._jac.xp
        nv = self._jac.n_var
        for inv, rows in ((self._inv_prism, self._rows_prism), (self._inv_tet, self._rows_tet)):
            if not inv.shape[0]:
                continue
            if xp is np:
                _apply_inverse_blocks(inv, rows, v_flat, out)
                continue
            # cupy：float32 批量 matmul（逆以 float32 存储，向量也降到 float32 再乘，避免混合精度 matmul
            # 把整份逆隐式提升成 float64）
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
def _apply_inverse_blocks(inv, rows, v, out):
    """`out[rows[b, s], k] = sum_{t,l} inv[b, s*nv+k, t*nv+l] * v[rows[b, t], l]`，按块并行，float64 累加。

    块内行序与装配一致（`(解点, 变量)`，`_inverted_blocks` 的 `diag` 同一顺序）。此前 CPU 也走 float32 批量
    `matmul`：先把块内向量 gather 成 `(n_blk, m, 1)` 再降精度、乘完再 scatter 回去，三份全尺寸临时数组，
    numpy 的批量 matmul 又不随核数并行——cube_demo 粗网格（28.6 万单元）P2 每次作用约 0.6 s。这里直接按
    行号读写，逆仍是 float32 存储，乘加在 float64 里做（比 float32 累加更准；预处理的作用因此在 float32
    舍入量级上与旧路径不同，不影响被求解的线性系统）。
    """
    n_blk, m, _ = inv.shape
    n_real = rows.shape[1]
    nv = v.shape[1]
    for b in prange(n_blk):
        for s in range(n_real):
            rs = rows[b, s]
            for k in range(nv):
                i = s * nv + k
                acc = 0.0
                for t in range(n_real):
                    rt = rows[b, t]
                    for l in range(nv):
                        acc += inv[b, i, t * nv + l] * v[rt, l]
                out[rs, k] = acc


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
