"""四面体 / 棱柱单元朝向修正：`signed_tet_volume`、`fix_tet_orientation`、`fix_prism_orientation`。

加载高阶网格时（`grid/high_order/high_order_mesh.py`）对全部单元统一朝向，面连接关系与原生基的面局部坐标都依赖
这一约定。只依赖 numpy；包级别（`autoflowcfd.grid.curved_mapping`）导出后两个函数。
"""

import numpy as np


def signed_tet_volume(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """四面体有符号体积（六分之一混合积）。"""
    return float(np.dot(np.cross(p1 - p0, p2 - p0), p3 - p0)) / 6.0


def fix_tet_orientation(node_ids: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """确保四面体节点顺序对应正的有符号体积（正 Jacobian 的前提条件）。

    若有符号体积为负（左手系排列，网格生成器输出的常见问题），
    交换节点 1、2 以翻转朝向；返回可能被重排后的 node_ids 副本。
    """
    p = nodes[node_ids]
    vol = signed_tet_volume(p[0], p[1], p[2], p[3])
    if vol < 0:
        node_ids = node_ids.copy()
        node_ids[[1, 2]] = node_ids[[2, 1]]
    return node_ids


def fix_prism_orientation(node_ids: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """确保棱柱节点顺序 (v0,v1,v2,w0,w1,w2) 对应正体积。

    用棱柱分解为 3 个四面体（v0,v1,v2,w0), (v1,v2,w0,w1), (v2,w0,w1,w2)
    的体积之和判断朝向；若为负，交换底面和顶面的节点 1、2（同步交换保持
    "顶点 i 正上方是顶点 i+3" 的对应关系不被破坏）。
    """
    p = nodes[node_ids]
    v0, v1, v2, w0, w1, w2 = p
    vol = (
        signed_tet_volume(v0, v1, v2, w0)
        + signed_tet_volume(v1, v2, w0, w1)
        + signed_tet_volume(v2, w0, w1, w2)
    )
    if vol < 0:
        node_ids = node_ids.copy()
        node_ids[[1, 2]] = node_ids[[2, 1]]
        node_ids[[4, 5]] = node_ids[[5, 4]]
    return node_ids
