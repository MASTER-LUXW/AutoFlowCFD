"""BJ 型邻居极值判据测试（test_bounds_sensor*.py）共用的一维网格构造。"""

import numpy as np


def _line_mesh(n_cells, n_sps=2):
    """一维单元链的面连接：cell i 与 cell i+1 相邻，两端是边界面。

    返回 (owner, neighbor, is_boundary)。
    """
    owner = list(range(n_cells - 1)) + [0, n_cells - 1]
    neigh = list(range(1, n_cells)) + [-1, -1]
    bnd = [False] * (n_cells - 1) + [True, True]
    return (np.array(owner), np.array(neigh), np.array(bnd, dtype=bool))
