"""四面体/棱柱单元朝向修正（从 curved_mapping.py 拆分）。

从 curved_mapping.py 中拆分出来（原文件超过 400 行的项目约定上限）：
signed_tet_volume / fix_tet_orientation / decompose_prism_to_tets /
fix_prism_orientation 是"单元朝向修正"这一个独立算法阶段（原文件里也
已经用同名注释分节隔开），只依赖 numpy，不与 CurvedMapping 类、Duffy
坍缩坐标变换、解析精确雅可比等其余内容共享任何状态，是天然的拆分边界。
curved_mapping.py 在模块顶层原样重新导出这四个名字，任何既有的
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


def decompose_prism_to_tets(node_ids: np.ndarray) -> np.ndarray:
    """把一个棱柱 (v0,v1,v2,w0,w1,w2)（局部数组，可能已被 fix_prism_orientation
    重排）分解为 3 个四面体，与 grid/mesh_gen/mesh_prism_to_tet.py::
    convert_layers_to_tetrahedra 生成核心区四面体网格所用的规则完全一致：
    按 全局 节点编号对棱柱底面三角形排序 v0'<v1'<v2'（保持 w 侧对应关系
    不变），取
        T1 = (v0', v1', v2', w2')
        T2 = (v0', v1', w1', w2')
        T3 = (v0', w0', w1', w2')

    只依赖共享四边形侧面的 4 个 全局 节点编号（与本棱柱局部数组的存储
    顺序、朝向修正历史无关），因此与网格中任何用同一规则生成的相邻单元
    （棱柱或四面体）在共享侧面上比特级一致——已在真实网格上数值验证：
    对角线选取等价于"连接该四边形 4 个角点中 全局 编号最小与最大的
    两点"，329126 处内部面比对结果零例外（层间节点编号单调，w 层编号
    恒大于其下方对应 v 层编号）。

    用于 high_order_mesh.py 里把"四边形侧面被拆分给 2 个不同相邻单元"
    （棱柱边界层与四面体核心区过渡处、必然出现的拓扑情形）的少数棱柱
    （实测约5%）转成四面体，从根本上消除"同一 owner 单元、同一立方体面
    对应 2 条不同 face_connectivity 记录，各自独立参与残差组装导致重复
    计正"或"各自只匹配到其中一个真实相邻单元"的两类错误——而不是在
    FR 残差组装或 Flux 点 匹配算法层面做任何近似/容差放宽。

    Returns:
        (3,4) int 数组，3 个四面体的节点编号（未做符号体积/朝向修正，
        调用方需按需自行调用 fix_tet_orientation）。
    """
    v_tri = np.asarray(node_ids[:3])
    w_tri = np.asarray(node_ids[3:])
    order = np.argsort(v_tri)
    sv0, sv1, sv2 = v_tri[order]
    sw0, sw1, sw2 = w_tri[order]
    return np.array(
        [
            [sv0, sv1, sv2, sw2],
            [sv0, sv1, sw1, sw2],
            [sv0, sw0, sw1, sw2],
        ]
    )


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
