"""四面体/棱柱单元朝向修正（从 curved_mapping.py 拆分）。

从 curved_mapping.py 中拆分出来（原文件超过 400 行的项目约定上限）：
signed_tet_volume / fix_tet_orientation /
fix_prism_orientation 是"单元朝向修正"这一个独立算法阶段（原文件里也
已经用同名注释分节隔开），只依赖 numpy，不与 CurvedMapping 类、Duffy
坍缩坐标变换、解析精确雅可比等其余内容共享任何状态，是天然的拆分边界。
curved_mapping.py 在模块顶层原样重新导出这三个名字，任何既有的
`从 autoflowcfd.grid.curved_mapping.curved_mapping 导入 fix_tet_orientation` 之类
导入路径（high_order_mesh.py 等）不受影响，逻辑/数值结果完全未改动。
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
