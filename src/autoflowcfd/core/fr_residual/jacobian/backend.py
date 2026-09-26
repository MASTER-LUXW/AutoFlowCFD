"""AutoFlowCFD V2.0 - 解析单元块装配器的后端适配（全部后端共用）。

块 Jacobi / 块 ILU 缓存（`time_integration/implicit/block_jacobi.py`）每个 Newton
步向装配器要一次块：`assembler(u0_flat, r0_flat)`，`u0/r0` 是 Newton 行（本 rank
的单元、Newton 变量 5 个）上的状态与残差，数组模块可以是 numpy 或 cupy。装配
在主机上做（`assemble.py` 是 numpy + numba），结果是主机 float32 数组，由缓存
按自己的数组模块上传。

各后端只差两件事，由构造参数给出：

* **状态从 Newton 行扩展到残差的单元空间**：单机即恒等；分布式要先 halo 交换
  再换到"棱柱在前"的 local+halo 紧凑排列（与分布式残差
  `distributed_compute.py` 同一套 `perm`），`compact_state` 给出这一步；
* **块的行单元在紧凑空间里的下标** `row_compact`（分布式是 `inv_perm[:n_local]`），
  对角块按它取出、耦合块只保留两端都是本 rank 单元的那些（halo 耦合由 rank 间
  的块 Jacobi 式分解忽略，与块 Jacobi 同一个近似）。

装配器持有本步冻结的涡粘，所以每个 Newton 步新建（做成类而不是闭包，项目规范）。
"""

from typing import Callable, Optional

import numpy as np

from autoflowcfd.core.fr_operators.kernels import resolve_ausm_precond_mode

from .assemble import MeanFlowLinearization, assemble_mean_flow_blocks
from .coupling import CouplingBlocks, CouplingGroup


def _host(a):
    if isinstance(a, np.ndarray):
        return a
    get = getattr(a, "get", None)
    return get() if get is not None else np.asarray(a)


def unsupported_reason(*, order: int, entropy_stable_volume: bool = False,
                       artificial_viscosity: bool = False, wmles: bool = False) -> Optional[str]:
    """解析装配不覆盖当前离散时返回原因（块 Jacobi 回到着色差分装配，精确、只是贵）。"""
    if int(order) < 1:
        return "P0 走有限体积特化核（差分装配只要 色数 x 5 次残差求值）"
    if entropy_stable_volume:
        return "熵稳定两点通量体积项"
    if artificial_viscosity:
        return "Persson-Peraire 人工粘性（粘度随状态变化、带质量扩散通道）"
    if wmles:
        return "WMLES 壁面应力修正"
    return None


class MeanFlowBlockAssembler:
    """`(u0_flat, r0_flat) -> (blocks_prism, blocks_tet[, coupling])`（主机 float32）。"""

    __slots__ = ("ctx", "n_sps", "compact_state", "row_compact", "want_coupling")

    def __init__(self, *, mesh, ops, ghost_provider, mu, mach_ref, low_mach, mu_t, n_sps: int,
                 flat=None, compact_state: Optional[Callable] = None, row_compact=None,
                 want_coupling: bool = False):
        self.ctx = MeanFlowLinearization(
            mesh=mesh, ops=ops, ghost_provider=ghost_provider, mu=float(mu), mach_ref=float(mach_ref),
            precond_mode=resolve_ausm_precond_mode(), low_mach=bool(low_mach),
            mu_t=None if mu_t is None else np.ascontiguousarray(_host(mu_t), dtype=np.float64), flat=flat)
        self.n_sps = int(n_sps)
        self.compact_state = compact_state
        self.row_compact = None if row_compact is None else np.asarray(row_compact, dtype=np.int64)
        self.want_coupling = bool(want_coupling)

    def __call__(self, u0_flat, r0_flat):
        u_dev = u0_flat.reshape(-1, self.n_sps, u0_flat.shape[-1])
        r = _host(r0_flat).reshape(u_dev.shape[0], self.n_sps, -1)
        if self.row_compact is None:
            return assemble_mean_flow_blocks(self.ctx, _host(u_dev), residual=r,
                                             want_coupling=self.want_coupling)
        # halo 交换在状态自己的数组模块上做（多 GPU 的交换器收 cupy 数组），再搬回主机
        U = np.ascontiguousarray(_host(self.compact_state(u_dev)))
        R = np.zeros(U.shape[:2] + (r.shape[-1],))
        R[self.row_compact] = r
        out = assemble_mean_flow_blocks(self.ctx, U, residual=R, want_coupling=self.want_coupling)
        return self._rows_only(out, int(self.ctx.mesh.n_prism_cells))

    def _rows_only(self, out, n_prism_compact):
        return select_rows(out, self.row_compact, n_prism_compact, self.want_coupling)


def select_rows(out, row_compact, n_prism_compact: int, want_coupling: bool):
    """紧凑空间（local+halo）装配结果 -> 只含 Newton 行单元（local）的块。

    对角块按 `row_compact` 取出（棱柱/四面体各自按行单元的原生顺序）；耦合块只保留
    两端都是本 rank 单元的那些（与 rank 间块 Jacobi 式分解同一个近似），单元号换成
    行单元下标。平均流与 k-omega 装配器共用。
    """
    rc = row_compact
    is_p = rc < n_prism_compact
    bp_c, bt_c = out[0], out[1]
    blocks = (np.ascontiguousarray(bp_c[rc[is_p]]), np.ascontiguousarray(bt_c[rc[~is_p] - n_prism_compact]))
    if not want_coupling:
        return blocks
    local = -np.ones(bp_c.shape[0] + bt_c.shape[0], dtype=np.int64)
    local[rc] = np.arange(rc.size)
    groups = []
    for g in out[2].groups:
        lr, lc = local[g.rows], local[g.cols]
        keep = (lr >= 0) & (lc >= 0)
        if keep.any():
            groups.append(CouplingGroup(row_is_prism=g.row_is_prism, col_is_prism=g.col_is_prism,
                                        rows=lr[keep], cols=lc[keep], blocks=g.blocks[keep]))
    return blocks + (CouplingBlocks(groups=groups),)
