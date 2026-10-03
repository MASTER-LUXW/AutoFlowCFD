"""AutoFlowCFD V2.0 - 紧凑空间场的 halo 行改取所属 rank 的值（CPU-MPI 与多 GPU 共用）。

分布式求值在紧凑空间（local + 一层 halo，"棱柱在前"排列）上进行。只依赖单元自身
解点的量在 halo 行上也是对的；**依赖面邻居的量**（提升修正梯度等）在 halo 行上不完整
——halo 单元外侧的面不在本 rank 的面集合里。这类量算完后用本对象刷新：取 local 段
（`inv_perm` 换回原生排列的前 `n_local` 行）、经 halo 交换拿到所属 rank 算好的 halo
值、再按 `perm` 换回紧凑空间。

CPU `HaloExchange` 与 GPU `GPUHaloExchange` 的 `exchange` 都接受任意每解点尾部形状，
`perm/inv_perm` 在对应的数组模块上即可，同一个类两个后端共用。随求值视图长期存活，
做成类而不是闭包，只持有交换器与两个换序（项目规范）。
"""


class CompactHaloRefresh:
    """`field_compact (n_compact, n_sps, ...)` -> halo 行取所属 rank 值后的同形数组。"""

    __slots__ = ("exchange", "perm", "inv_perm", "n_local")

    def __init__(self, halo_exchange, perm, inv_perm, n_local: int):
        self.exchange = halo_exchange.exchange
        self.perm, self.inv_perm, self.n_local = perm, inv_perm, int(n_local)

    def __call__(self, field_compact):
        return self.exchange(field_compact[self.inv_perm][:self.n_local])[self.perm]
