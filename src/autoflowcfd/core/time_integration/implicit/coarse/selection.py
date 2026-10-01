"""AutoFlowCFD V2.0 - 块 ILU 档的预处理构造（本地多层 / 块 ILU，分布式再叠全局粗校正）。

`BlockJacobiCache`（`block_jacobi.py`）在块 ILU 档（有面邻居耦合块）时把构造交给这里：

* 本 rank 有粗层可做（`preconditioner_hierarchy` 非空）时用多层预处理，否则用块 ILU 本身；
* 分布式（`CoarseCommContext.n_ranks > 1`）时外面再叠一层全局粗校正（`global_coarse.py`）。

聚合层次与全局粗空间只依赖拓扑，首次需要时构造，缓存对象的生命周期内复用（换阶时缓存
整体重建，见 `BlockJacobiCache` 文档）。
"""

from typing import Optional

from ..block_ilu import BlockILUPreconditioner, flatten_cell_blocks
from .global_coarse import CoarseCommContext, CrossRankCoupling, GlobalCoarseCorrection, GlobalCoarseSpace
from .multilevel import MultilevelPreconditioner, preconditioner_hierarchy


class CoarsePreconditionerFactory:
    """见模块文档。做成类而不是闭包：随缓存对象长期存活，只持有拓扑量（项目规范）。"""

    __slots__ = ("high_order", "ctx", "_hierarchy", "_space")

    def __init__(self, high_order: bool, ctx: Optional[CoarseCommContext] = None):
        self.high_order = bool(high_order)
        self.ctx = ctx if ctx is not None and ctx.n_ranks > 1 else None
        self._hierarchy = None
        self._space = None

    def hierarchy(self, struct):
        if self._hierarchy is None:
            self._hierarchy = preconditioner_hierarchy(struct, self.high_order)
        return self._hierarchy

    def flexible(self, struct) -> bool:
        """作用是否随调用变化（本地多层的 K 循环），GMRES 据此走灵活模式。"""
        return bool(self.hierarchy(struct))

    def _global_space(self, struct) -> GlobalCoarseSpace:
        if self._space is None:
            self._space = GlobalCoarseSpace(struct, self.high_order, self.ctx)
        return self._space

    def cross_coupling(self, blocks, struct) -> Optional[CrossRankCoupling]:
        """装配给出的跨 rank 耦合块 -> 块 CSR（单进程返回 None）。装配是集体调用，各 rank 在
        这里同时首次构造全局粗空间（其中有 allgather 与 halo 交换）。"""
        if self.ctx is None:
            return None
        if blocks is None:
            raise ValueError("分布式块 ILU 档需要装配器给出跨 rank 耦合块（全局粗矩阵 P^T A P 的 rank 间部分）")
        space = self._global_space(struct)
        return CrossRankCoupling(blocks, struct.n_real.size, space.gid_compact.size)

    def make(self, jac, struct, cross: Optional[CrossRankCoupling], dtau_flat):
        diag, off = flatten_cell_blocks(jac)
        hier = self.hierarchy(struct)

        def local(dtau):
            if hier:
                return MultilevelPreconditioner(diag, off, struct, hier, dtau, jac.n_sps)
            return BlockILUPreconditioner(diag, off, struct, dtau, jac.n_sps)

        if self.ctx is None:
            return local(dtau_flat)
        return GlobalCoarseCorrection(self._global_space(struct), local, diag, off, struct, cross, dtau_flat,
                                      jac.n_sps)
