"""AutoFlowCFD V2.0 - 多层预处理的粗空间：单元耦合图上的分层贪心聚合。

一层聚合（Vanek 等的 plain aggregation）：

1. 按单元顺序扫描，一个单元与它全部面邻居都尚未归属时，单元及其邻居组成一个
   新聚合体（"根 + 一圈邻居"，三维面邻接下约 5~6 个单元）；
2. 剩下的单元并入任一已归属的面邻居所在的聚合体（只看第 1 步的结果，保证每个
   聚合体在图上连通）；没有已归属邻居的孤立单元自成一体。

聚合体之间按"至少一条细图边相连"构成粗图，在粗图上重复同一过程，得到多层预处理
的层次（`build_hierarchy`）。所有操作只依赖拓扑，与状态无关，换阶前一直复用。
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _aggregate_once(indptr, cols, n):
    agg = np.full(n, -1, dtype=np.int64)
    n_agg = 0
    for c in range(n):
        if agg[c] >= 0:
            continue
        free = True
        for k in range(indptr[c], indptr[c + 1]):
            if agg[cols[k]] >= 0:
                free = False
                break
        if not free:
            continue
        agg[c] = n_agg
        for k in range(indptr[c], indptr[c + 1]):
            agg[cols[k]] = n_agg
        n_agg += 1
    first = agg.copy()
    for c in range(n):
        if first[c] >= 0:
            continue
        target = -1
        for k in range(indptr[c], indptr[c + 1]):
            y = cols[k]
            if first[y] >= 0:
                target = first[y]
                break
        if target < 0:
            agg[c] = n_agg
            n_agg += 1
        else:
            agg[c] = target
    return agg, n_agg


def _coarse_graph(indptr, cols, agg, n_agg):
    """聚合体之间的邻接（去掉自环、去重），CSR。"""
    import scipy.sparse as sp

    rows = np.repeat(np.arange(indptr.size - 1, dtype=np.int64), np.diff(indptr))
    ra, ca = agg[rows], agg[cols]
    keep = ra != ca
    g = sp.csr_matrix((np.ones(int(keep.sum()), dtype=np.int8), (ra[keep], ca[keep])),
                      shape=(n_agg, n_agg))
    g.sum_duplicates()
    return np.asarray(g.indptr, dtype=np.int64), np.asarray(g.indices, dtype=np.int64)


def build_hierarchy(indptr, cols, n_cells: int, coarsest_nodes: int):
    """多层聚合层次：`[(agg_0, n_1), (agg_1, n_2), ...]`，`agg_k` 把第 k 层节点映到
    第 k+1 层（第 0 层是单元）。每层只做一次聚合（约 5~7 倍粗化），直到节点数不超过
    `coarsest_nodes`，或图上已无边可聚。"""
    if coarsest_nodes < 1:
        raise ValueError(f"coarsest_nodes 必须 >= 1，收到 {coarsest_nodes}")
    ip, cs = np.asarray(indptr, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    n = int(n_cells)
    levels = []
    while n > coarsest_nodes:
        agg, n_new = _aggregate_once(ip, cs, n)
        if n_new >= n:          # 图已无边（全是孤立点），无法再聚合
            break
        levels.append((agg, int(n_new)))
        ip, cs = _coarse_graph(ip, cs, agg, n_new)
        n = int(n_new)
    return levels
