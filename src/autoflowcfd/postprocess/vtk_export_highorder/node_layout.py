"""AutoFlowCFD V2.0 - VTK Lagrange 单元的节点枚举与坐标反演。

从 `vtk_export_highorder.py`（678 行）拆出（2026-09-20，项目"单文件不超
500 行"规范）。纯搬家，逻辑未改。递归单纯形节点排序的依据（直接读 VTK
源码核实）见 `_simplex_multi_indices` 与 `_tet_vtk_node_barycentrics`
的文档。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

_VTK_LAGRANGE_TETRAHEDRON = 71
_VTK_LAGRANGE_WEDGE = 73

# 四面体：递归节点排序方案（见下方"递归单纯形节点排序"一节）已实现，
# 在 order=3 上用物理空间三次解析场决定性验证到 VTK 自身形函数重构的
# 探测点误差 ~1e-9（机器精度级）——真正达标。order=4 用同样方法测过：
# 节点值误差 4.2e-3（不是机器精度），说明 order>=4 时"每个面的顶点排列
# 顺序"开始真正影响面内部点位置（order=3 每面恰好1个内部点=面形心，与
# 顶点排列无关，是特例——见 _TET_FACES_VTK 文档），而本次没有独立核实
# 清楚 VTK 期望的面内部点在面自身坐标系下的精确排列约定。按项目"不能
# 静默简化/退化"要求，只开放到已经决定性验证过的 order=3，不虚报
# order=4+ 也"支持"。
_MAX_SUPPORTED_ORDER_TET = 3
_MAX_SUPPORTED_ORDER_WEDGE = 2
# 向后兼容旧名字（此前唯一调用方是测试文件，两处都已 import 这个名字）——
# 现在只代表 wedge 的上限，tet 走 _MAX_SUPPORTED_ORDER_TET。
_MAX_SUPPORTED_ORDER = _MAX_SUPPORTED_ORDER_WEDGE


# ============================================================================
# 递归单纯形节点排序（VTK Lagrange 单纯形单元通用方案）
#
# VTK 的高阶 Lagrange 单纯形单元（三角形/四面体，`vtkLagrangeTriangle`/
# `vtkLagrangeTetra`）用同一套递归规则枚举节点：先角点，再每条棱的
# 内部点（从低编号顶点到高编号顶点），再（四面体）每个面的内部点（按该
# 面自己的三角形递归规则），最后（三角形的面内部/四面体的体内部）用同一
# 规则递归到一个更小的单纯形——把重心坐标（多重指标 (i0,...,id)，
# 非负整数，和为 order）里所有分量都严格大于 0 的那些点，各自减 1 后
# 恰好是一个 order-(dim+1) 阶、同维度单纯形的**全部**节点（角点+棱+面+
# 内部），递归下去直至 order 不足以再有内部点为止。
#
# 已用组合计数验证（内部点数分别应为 C(order-1,2)（三角形）/
# C(order-1,3)（四面体）——递归定义自动满足，见开发过程记录）；数值
# 正确性由本模块已有的"物理空间解析场往返"测试方法在 order=3 上决定性
# 验证（见 test_vtk_export_highorder.py）。
#
# 四面体面顶点分组：VTK 标准 `vtkTetra::GetFace` 约定
# faces = [(0,1,3), (1,2,3), (2,0,3), (0,2,1)]（前3个面共享顶点3，
# 第4个面是"底面"）。order=3 时每个面恰好1个内部点——该点在归一化重心
# 坐标下必为 (1/3,1/3,1/3)（面自身的形心），与面内3个顶点的排列顺序
# 无关，因此 order=3 这个最直接的目标阶数不依赖对这份面顶点分组细节
# 排列顺序的验证，只依赖分组本身（哪3个顶点组成哪个面）正确——已通过
# 决定性数值测试确认。order>=4 时面内部点不再是形心，顶点排列顺序才会
# 真正影响结果，若后续需要 order>=4 请先补充针对性验证。
# ============================================================================

_TET_FACES_VTK = [(0, 1, 3), (1, 2, 3), (2, 0, 3), (0, 2, 1)]


def _simplex_edges(dim: int) -> List[Tuple[int, int]]:
    if dim == 2:
        return [(0, 1), (1, 2), (2, 0)]
    if dim == 3:
        return [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    raise ValueError(f"Unsupported simplex dim={dim}")


def _simplex_multi_indices(dim: int, order: int) -> List[Tuple[int, ...]]:
    """按 VTK 递归方案枚举 dim-单纯形（2=三角形，3=四面体）、给定
    `order` 的全部节点，每个节点是长度 dim+1 的非负整数多重指标（重心
    坐标 * order），和恒为 `order`。返回顺序：角点、各棱内部点、（仅
    dim==3）各面内部点、单纯形自身内部点（递归）——见模块文档。
    """
    n = dim + 1
    if order == 0:
        return [tuple([0] * n)]

    corners = []
    for v in range(n):
        idx = [0] * n
        idx[v] = order
        corners.append(tuple(idx))
    pts = list(corners)

    for (a, b) in _simplex_edges(dim):
        for k in range(1, order):
            idx = [0] * n
            idx[a] = order - k
            idx[b] = k
            pts.append(tuple(idx))

    if dim == 3:
        for face in _TET_FACES_VTK:
            for fi in _simplex_interior_only(2, order):
                idx = [0, 0, 0, 0]
                for local_pos, vert in enumerate(face):
                    idx[vert] = fi[local_pos]
                pts.append(tuple(idx))

    if order >= n:
        sub = _simplex_multi_indices(dim, order - n)
        pts.extend(tuple(x + 1 for x in s) for s in sub)

    return pts


def _simplex_interior_only(dim: int, order: int) -> List[Tuple[int, ...]]:
    """`_simplex_multi_indices` 的"仅内部点"子集（不含角点/棱/面），用于
    四面体的面内部点递归引用某个面自身的三角形内部点。"""
    n = dim + 1
    if order < n:
        return []
    sub = _simplex_multi_indices(dim, order - n)
    return [tuple(x + 1 for x in s) for s in sub]

_TET_CORNER_BARYCENTRICS = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
_TET_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]

_WEDGE_TRI_CORNER_BARYCENTRICS = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])
_WEDGE_BOTTOM_EDGES = [(0, 1), (1, 2), (2, 0)]
# 四边形侧面中心点与底三角形边的对应关系：面 (0,1,4,3) 对应底边 (0,1)，
# 面 (1,2,5,4) 对应底边 (1,2)，面 (2,0,3,5) 对应底边 (2,0)——已用真实
# 构造+pyvista 往返采样验证（对照组：打乱这个对应关系后采样结果偏离）。
_WEDGE_QUAD_FACES = [(0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]


def _tet_vtk_node_barycentrics(order: int) -> np.ndarray:
    """按 VTK_LAGRANGE_TETRAHEDRON 节点顺序枚举各节点的重心坐标
    (L1,L2,L3,L4)，形状 (n_nodes,4)。

    2026-09-02 起改用通用递归单纯形节点排序（`_simplex_multi_indices`，
    见模块文档"递归单纯形节点排序"一节）。order<=2 时与此前手写实现
    逐位等价（角点+棱内部点，无面/体内部点）；order=3 起新增面内部点
    （每面1个，物理上就是面形心，与面顶点排列顺序无关，已用本模块既有
    的物理空间解析场决定性验证方法确认到机器精度，见
    test_vtk_export_highorder.py）。order>=4 用同样方法测过，节点值
    误差 4.2e-3（不是机器精度）——说明面内部点在 order>=4 时的排列
    对面顶点顺序敏感，本次没有独立核实清楚，因此只开放到 order=3，
    见 `_MAX_SUPPORTED_ORDER_TET` 文档。
    """
    if order < 1:
        raise ValueError(f"order 必须 >= 1，收到 {order}")
    if order > _MAX_SUPPORTED_ORDER_TET:
        raise NotImplementedError(
            f"VTK_LAGRANGE_TETRAHEDRON 导出目前只验证到 order="
            f"{_MAX_SUPPORTED_ORDER_TET}（order>=4 时面内部点排列对面"
            f"顶点顺序敏感，本次未独立核实，见模块文档）。收到 order={order}。"
        )
    multi_idx = _simplex_multi_indices(3, order)
    return np.array(multi_idx, dtype=np.float64) / order


def _wedge_vtk_node_layout(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """按 VTK_LAGRANGE_WEDGE 节点顺序枚举各节点的 (三角形重心坐标,
    挤出方向分数 in [0,1])，形状分别为 (n_nodes,3) 和 (n_nodes,)。见模块
    文档"节点排序约定"（18 点二次方案）。
    """
    if order < 1:
        raise ValueError(f"order 必须 >= 1，收到 {order}")
    if order > _MAX_SUPPORTED_ORDER:
        raise NotImplementedError(
            f"VTK_LAGRANGE_WEDGE 导出目前只验证/实现到 order="
            f"{_MAX_SUPPORTED_ORDER}（order>=3 需要三角形面/体内部点的"
            f"递归排序方案，见模块文档）。收到 order={order}。"
        )
    tri = list(_WEDGE_TRI_CORNER_BARYCENTRICS)
    tri_list = []
    z_list = []
    # 6 顶点：底三角形 (z=0)，顶三角形 (z=1)
    for i in range(3):
        tri_list.append(tri[i]); z_list.append(0.0)
    for i in range(3):
        tri_list.append(tri[i]); z_list.append(1.0)

    if order >= 2:
        for z_val in (0.0, 1.0):
            for (a, b) in _WEDGE_BOTTOM_EDGES:
                for k in range(1, order):
                    frac = k / order
                    tri_list.append((1.0 - frac) * tri[a] + frac * tri[b])
                    z_list.append(z_val)
        # 竖直边
        for i in range(3):
            for k in range(1, order):
                frac = k / order
                tri_list.append(tri[i].copy())
                z_list.append(frac)
        # 四边形侧面内部点：每面 (order-1)^2 个，order==2 时恰好每面 1 个
        # （对应文献里的 nqfdof=(order-1)^2，本模块只验证/实现 order<=2，
        # 见模块文档 wedge 21 vs 18 点方案的说明）。
        n_qfdof = (order - 1) ** 2
        if n_qfdof > 0:
            for (a, b) in _WEDGE_BOTTOM_EDGES:
                tri_list.append(0.5 * (tri[a] + tri[b]))
                z_list.append(0.5)

    return np.array(tri_list), np.array(z_list)


def _tet_barycentric_to_cube(bary: np.ndarray) -> np.ndarray:
    """四面体重心坐标 (L1,L2,L3,L4) -> 本项目参考立方体坐标 (a,b,c)。

    对 `curved_mapping.py::cube_to_tet_rst` + `tet_barycentric` 的解析
    求逆（已用正向映射数值往返验证到机器精度，见开发过程记录）：

        c = 2*L4 - 1
        b = 2*L3/(L1+L2+L3) - 1   （L1+L2+L3≈0 时任取 b=0——该处
            L4≈1，是四面体的单一顶点 v4，整条 b 轴在此坍缩为一点，
            取值不影响物理位置）
        a = (L2-L1)/(L1+L2)        （L1+L2≈0 时任取 a=0——该处
            L3+L4≈1 且 L1=L2=0，是坍缩坐标 b=+1 棱上的点，整条 a 轴
            在此坍缩为一条棱，取值不影响物理位置）

    Args:
        bary: (n,4) 重心坐标

    Returns:
        (n,3) 参考立方体坐标 (a,b,c)
    """
    eps = 1e-12
    L1, L2, L3, L4 = bary[:, 0], bary[:, 1], bary[:, 2], bary[:, 3]

    c = 2.0 * L4 - 1.0

    sum_123 = L1 + L2 + L3
    b = np.where(sum_123 > eps, 2.0 * L3 / np.maximum(sum_123, eps) - 1.0, 0.0)

    sum_12 = L1 + L2
    a = np.where(sum_12 > eps, (L2 - L1) / np.maximum(sum_12, eps), 0.0)

    return np.column_stack([a, b, c])


def _tri_barycentric_to_cube_ab(tri_bary: np.ndarray) -> np.ndarray:
    """三角形重心坐标 (l1,l2,l3) -> 本项目参考立方体 (a,b) 分量（棱柱的
    c 分量由挤出方向分数直接给出，见 `_build_vtk_lagrange_export_data`）。

    对 `curved_mapping.py::cube_to_tri_rs` + `tri_barycentric` 的解析
    求逆，与 `_tet_barycentric_to_cube` 同一套 Duffy 三角形子结构（已用
    正向映射数值往返验证）：

        b = 2*l3 - 1
        a = (l2-l1)/(l1+l2)   （l1+l2≈0 时任取 a=0，该处 l3≈1，是三角形
            单一顶点，整条 a 轴在此坍缩为一点）

    Args:
        tri_bary: (n,3) 三角形重心坐标 (l1,l2,l3)

    Returns:
        (n,2) 参考立方体坐标 (a,b)
    """
    eps = 1e-12
    l1, l2, l3 = tri_bary[:, 0], tri_bary[:, 1], tri_bary[:, 2]

    b = 2.0 * l3 - 1.0
    sum_12 = l1 + l2
    a = np.where(sum_12 > eps, (l2 - l1) / np.maximum(sum_12, eps), 0.0)

    return np.column_stack([a, b])
