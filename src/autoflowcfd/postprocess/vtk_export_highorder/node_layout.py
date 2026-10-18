"""AutoFlowCFD V2.0 - VTK Lagrange 单元的节点枚举。

VTK 的高阶 Lagrange 单元要求节点位于等距参数网格上（`vtkLagrangeInterpolation::EvaluateShapeFunctions`
把节点 j 固定在参数 j/order），节点的**编号顺序**由各单元类自己规定：

* 四面体（VTK_LAGRANGE_TETRAHEDRON）：递归单纯形方案——角点、各棱内部点（低号顶点 -> 高号顶点）、各面内部点
  （按该面自身的三角形递归规则）、体内部点（递归到 order-4 阶的四面体），见 `_simplex_multi_indices`；
* 三棱柱（VTK_LAGRANGE_WEDGE）：`vtkHigherOrderWedge::PointIndexFromIJK` 的显式编号——角点（底 0,1,2、顶 3,4,5）、
  底面三条棱、顶面三条棱、三条竖棱、底/顶三角形面内部（逐行）、三个四边形侧面内部（张量序）、体内部
  （逐层三角形），见 `_wedge_point_index`。

两者都用 VTK 自身的形函数在单元内部非节点位置重新采样、对照物理空间解析场验证
（`tests/unit/test_vtk_export_highorder.py`）。
"""

from typing import List, Tuple

import numpy as np

_VTK_LAGRANGE_TETRAHEDRON = 71
_VTK_LAGRANGE_WEDGE = 73

#: 四面体导出验证到的最高阶数：order>=4 时面内部点不止一个，其排列依赖 VTK 对面顶点顺序的约定，未独立核实
#: （同样的解析场方法测出节点值误差 4.2e-3），显式拒绝。求解器最高阶数为 3。
_MAX_SUPPORTED_ORDER_TET = 3
#: 三棱柱导出验证到的最高阶数（与求解器最高阶数相同）
_MAX_SUPPORTED_ORDER_WEDGE = 3

#: VTK 四面体的面顶点分组（`vtkTetra::GetFace`）
_TET_FACES_VTK = [(0, 1, 3), (1, 2, 3), (2, 0, 3), (0, 2, 1)]


def _simplex_edges(dim: int) -> List[Tuple[int, int]]:
    if dim == 2:
        return [(0, 1), (1, 2), (2, 0)]
    if dim == 3:
        return [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    raise ValueError(f"Unsupported simplex dim={dim}")


def _simplex_multi_indices(dim: int, order: int) -> List[Tuple[int, ...]]:
    """VTK 递归方案下 dim-单纯形（2=三角形，3=四面体）的全部节点，每个节点是和为 `order` 的非负整数多重指标
    （重心坐标 x order）。顺序：角点、各棱内部点、（dim==3）各面内部点、内部点（递归）。"""
    n = dim + 1
    if order == 0:
        return [tuple([0] * n)]
    pts = []
    for v in range(n):
        idx = [0] * n
        idx[v] = order
        pts.append(tuple(idx))
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
        pts.extend(tuple(x + 1 for x in s) for s in _simplex_multi_indices(dim, order - n))
    return pts


def _simplex_interior_only(dim: int, order: int) -> List[Tuple[int, ...]]:
    """`_simplex_multi_indices` 的仅内部点子集（四面体面内部点引用面自身的三角形内部点）。"""
    n = dim + 1
    if order < n:
        return []
    return [tuple(x + 1 for x in s) for s in _simplex_multi_indices(dim, order - n)]


def _tet_vtk_node_barycentrics(order: int) -> np.ndarray:
    """VTK_LAGRANGE_TETRAHEDRON 节点顺序下各节点的重心坐标 (L0,L1,L2,L3)，`(n_nodes, 4)`。"""
    if order < 1:
        raise ValueError(f"order 必须 >= 1，收到 {order}")
    if order > _MAX_SUPPORTED_ORDER_TET:
        raise NotImplementedError(
            f"VTK_LAGRANGE_TETRAHEDRON 导出只验证到 order={_MAX_SUPPORTED_ORDER_TET}（order>=4 时面内部点的排列依赖"
            f"面顶点顺序约定，未独立核实），收到 order={order}")
    return np.array(_simplex_multi_indices(3, order), dtype=np.float64) / order


def _triangle_dof_offset(order: int, i: int, j: int) -> int:
    """三角形内部点 (i, j)（i, j >= 1，i + j < order）的逐行序号（`vtkHigherOrderWedge` 的 triangleDOFOffset）。"""
    return i + order * (j - 1) - (j * (j + 1)) // 2


def _wedge_point_index(i: int, j: int, k: int, order: int) -> int:
    """三棱柱节点 (i, j, k) 的 VTK 编号（`vtkHigherOrderWedge::PointIndexFromIJK`，三个方向同阶）。

    (i, j) 是三角形上的格点（顶点 1 在 i = order，顶点 2 在 j = order），k 是挤出方向格点（0 为底面）。
    """
    rm1 = tm1 = order - 1
    ibdy, jbdy, ijbdy, kbdy = i == 0, j == 0, i + j == order, k in (0, order)
    nbdy = int(ibdy) + int(jbdy) + int(ijbdy) + int(kbdy)

    def corner():
        return 0 if (ibdy and jbdy) else (1 if (jbdy and ijbdy) else 2)

    if nbdy == 3:                                         # 角点
        return corner() + (3 if k else 0)
    offset = 6
    if nbdy == 2:                                         # 棱
        if not kbdy:                                      # 竖棱
            return offset + 6 * rm1 + (k - 1) + corner() * tm1
        offset += 3 * rm1 if k == order else 0            # 顶面三条棱在底面三条棱之后
        if jbdy:
            return offset + i - 1
        offset += rm1
        if ijbdy:
            return offset + j - 1
        offset += rm1
        return offset + (order - j - 1)
    offset += 6 * rm1 + 3 * tm1
    ntf = (rm1 - 1) * rm1 // 2                            # 三角形面内部点数
    nqf = rm1 * tm1                                       # 四边形面内部点数
    if nbdy == 1:                                         # 面
        if kbdy:
            return offset + (ntf if k > 0 else 0) + _triangle_dof_offset(order, i, j)
        offset += 2 * ntf
        if jbdy:
            return offset + (i - 1) + rm1 * (k - 1)
        offset += nqf
        if ijbdy:
            return offset + (order - i - 1) + rm1 * (k - 1)
        offset += nqf
        return offset + (order - j - 1) + rm1 * (k - 1)              # i-normal 面：沿 j 递减（顶点 2 -> 顶点 0）
    return offset + 2 * ntf + 3 * nqf + _triangle_dof_offset(order, i, j) + ntf * (k - 1)


def _wedge_vtk_node_layout(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """VTK_LAGRANGE_WEDGE 节点顺序下各节点的 (三角形重心坐标 (l0,l1,l2) `(n,3)`, 挤出方向分数 `(n,)` in [0,1])。"""
    if order < 1:
        raise ValueError(f"order 必须 >= 1，收到 {order}")
    if order > _MAX_SUPPORTED_ORDER_WEDGE:
        raise NotImplementedError(
            f"VTK_LAGRANGE_WEDGE 导出只验证到 order={_MAX_SUPPORTED_ORDER_WEDGE}（求解器最高阶数），收到 order={order}")
    n_nodes = (order + 1) * (order + 1) * (order + 2) // 2
    tri = np.full((n_nodes, 3), np.nan)
    z = np.full(n_nodes, np.nan)
    for k in range(order + 1):
        for j in range(order + 1):
            for i in range(order + 1 - j):
                p = _wedge_point_index(i, j, k, order)
                if not np.isnan(z[p]):
                    raise RuntimeError(f"wedge 节点编号冲突：({i},{j},{k}) -> {p}")
                tri[p] = ((order - i - j) / order, i / order, j / order)
                z[p] = k / order
    if np.isnan(z).any():
        raise RuntimeError("wedge 节点编号不完整")
    return tri, z
