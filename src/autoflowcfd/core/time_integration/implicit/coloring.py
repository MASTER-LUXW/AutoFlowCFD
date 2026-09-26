"""AutoFlowCFD V2.0 - 隐式步单元块装配用的单元着色与残差模板单元对。

* **距离 1 着色**（`greedy_cell_coloring`）：面相邻单元不同色。块 Jacobi 的着色
  差分装配（`cell_blocks.py::CellBlockJacobian`）与多色块 ILU 的排序都用它——
  残差只经本单元与面邻居，同色单元同时扰动时每个单元只看到自己的扰动。
* **距离 2 着色**（`distance2_cell_coloring`）：面相邻或共享一个面邻居的单元
  不同色。同色单元同时扰动时，任一单元最多只有**一个**被扰动的面邻居，于是差分
  出来的残差变化同时给出对角块 `J_cc` 与面邻居耦合块 `J_cy`（见
  `CellBlockJacobian` 的 `coupling_graph` 参数）。只用于 P0：每色要
  `真实解点数 x 变量数` 次残差求值，P0 每单元一个解点，P>=1 由解析装配给耦合块。
* **模板单元对**（`stencil_pairs`）：残差的面邻居对（有向、两个方向、去重），
  即耦合块的非零结构。

分布式：着色必须在**全局**面连接关系上算、各 rank 取本地一段（同色单元跨 rank
也不相邻 / 不共享邻居），见 `mpi/distributed_implicit.py`。
"""

from dataclasses import dataclass

import numpy as np
from numba import njit


def _adjacency_csr(owner_cell, neighbor_cell, n_cells: int):
    """面相邻关系的对称 CSR（去重，不含自环）。`neighbor_cell < 0` 的是边界面。"""
    import scipy.sparse as sp

    own = np.asarray(owner_cell, dtype=np.int64)
    nb = np.asarray(neighbor_cell, dtype=np.int64)
    inner = (nb >= 0) & (own != nb)
    a, b = own[inner], nb[inner]
    adj = sp.coo_matrix((np.ones(2 * a.size, dtype=np.int8), (np.r_[a, b], np.r_[b, a])),
                        shape=(n_cells, n_cells)).tocsr()
    adj.sum_duplicates()
    return adj.indptr.astype(np.int64), adj.indices.astype(np.int64)


def greedy_cell_coloring(owner_cell: np.ndarray, neighbor_cell: np.ndarray,
                         n_cells: int) -> np.ndarray:
    """按面相邻关系的贪心距离 1 着色，返回 `(n_cells,)` 的色号。

    色数不超过最大邻居数 + 1（棱柱 5 面、四面体 4 面）。
    """
    indptr, indices = _adjacency_csr(owner_cell, neighbor_cell, n_cells)
    max_degree = int(np.diff(indptr).max()) if n_cells else 0
    return _greedy_color_csr(indptr, indices, n_cells, max_degree + 2)


def distance2_cell_coloring(owner_cell: np.ndarray, neighbor_cell: np.ndarray,
                            n_cells: int) -> np.ndarray:
    """按面相邻关系的贪心距离 2 着色，返回 `(n_cells,)` 的色号。

    两个单元同色当且仅当它们既不相邻、也没有公共面邻居。色数不超过
    `1 + d + d(d-1)`（`d` 为最大邻居数）；四面体网格实测约 20~30 色。
    """
    indptr, indices = _adjacency_csr(owner_cell, neighbor_cell, n_cells)
    max_degree = int(np.diff(indptr).max()) if n_cells else 0
    return _greedy_color_d2_csr(indptr, indices, n_cells, max_degree * max_degree + 2)


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


@njit(cache=True)
def _greedy_color_d2_csr(indptr, indices, n_cells, n_mark):
    color = -np.ones(n_cells, dtype=np.int64)
    mark = -np.ones(n_mark, dtype=np.int64)       # mark[k] == c：色 k 已被 c 的距离 <= 2 单元占用
    for c in range(n_cells):
        for j in range(indptr[c], indptr[c + 1]):
            y = indices[j]
            k = color[y]
            if k >= 0:
                mark[k] = c
            for jj in range(indptr[y], indptr[y + 1]):
                k = color[indices[jj]]
                if k >= 0:
                    mark[k] = c
        k = 0
        while mark[k] == c:
            k += 1
        color[c] = k
    return color


def stencil_pairs(owner_cell: np.ndarray, neighbor_cell: np.ndarray, n_cells: int):
    """残差模板的有向面邻居对 `(rows, cols)`（两个方向、去重、按行再按列排序）。"""
    indptr, indices = _adjacency_csr(owner_cell, neighbor_cell, n_cells)
    rows = np.repeat(np.arange(n_cells, dtype=np.int64), np.diff(indptr))
    return rows, indices


@dataclass
class CouplingGraph:
    """差分装配耦合块所需的全部结构（Newton 行单元编号）。

    Attributes:
        rows, cols: 耦合块 `J_{rows[k], cols[k]}` 的有向单元对（本 rank 两端都在的）。
        colors: 距离 2 着色（全局一致着色的本地一段）。
    """
    rows: np.ndarray
    cols: np.ndarray
    colors: np.ndarray


def coupling_graph_from_faces(owner_cell, neighbor_cell, n_cells: int) -> CouplingGraph:
    """单机：由面记录的 `(owner, neighbor)` 直接给出模板单元对与距离 2 着色。"""
    rows, cols = stencil_pairs(owner_cell, neighbor_cell, n_cells)
    return CouplingGraph(rows=rows, cols=cols,
                         colors=distance2_cell_coloring(owner_cell, neighbor_cell, n_cells))
