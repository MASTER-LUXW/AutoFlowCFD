"""VTK 高阶 Lagrange 单元导出。

区别于单元平均导出（`vtk_export.py`）：每个单元导出为一个 VTK 高阶 Lagrange 单元（VTK_LAGRANGE_TETRAHEDRON 71、
VTK_LAGRANGE_WEDGE 73），节点值是 FR 解多项式在 VTK 等距节点上的精确取值，ParaView 里能看到单元内部的真实分布。

要点：

1. VTK Lagrange 单元的节点必须在等距参数网格上（`vtkLagrangeInterpolation::EvaluateShapeFunctions` 把节点 j
   固定在参数 j/order），而 FR 解点不在这些位置——先用解所在的原生模态基把解插值到 VTK 节点
   （`interp_matrices.py`）；
2. 节点编号按 VTK 各单元类的规定（`node_layout.py`：四面体递归单纯形方案、三棱柱 `PointIndexFromIJK`）；
3. 必须给出 `HigherOrderDegrees` 单元数组（三棱柱缺它时 VTK 底层崩溃）。

已验证阶数：三棱柱 1~3、四面体 1~3（求解器最高阶数为 3；四面体 order>=4 的面内部点排列未核实，显式拒绝）。
验证方法：物理空间多项式解析场赋到解点，导出后既比较节点值，又用 VTK 自身形函数在单元内部多个非节点位置重新
采样（`tests/unit/test_vtk_export_highorder.py`）。

    node_layout.py     VTK 节点编号
    interp_matrices.py 解点 -> VTK 节点插值矩阵与顶点形函数
    export.py          `export_highorder_vtk`
"""

from .export import export_highorder_vtk, export_solver_highorder_vtk  # noqa: F401

__all__ = ["export_highorder_vtk", "export_solver_highorder_vtk"]
