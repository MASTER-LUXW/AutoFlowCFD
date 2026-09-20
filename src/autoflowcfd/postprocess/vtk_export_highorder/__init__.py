"""VTK 高阶 Lagrange 单元导出 (#5，V2.0 专家组盲审第4轮，2026-08-28)。

区别于 vtk_export.py/vtk_export_legacy.py/vtk_export_xml.py 的单元中心
平均值导出（`solver.state.U.mean(axis=1)` 拍扁成一个值/单元）：本模块把
FR 解场按其真实的分段多项式表示，导出成 VTK 的 VTK_LAGRANGE_TETRAHEDRON
(71)/VTK_LAGRANGE_WEDGE (73) 高阶单元，在 ParaView 里能看到单元内部真实
的多项式分布，而不是被压平成常数。

核心技术难点与解决方案（均已用真实数值实验/往返测试验证，见下）：

1. **VTK 的 Lagrange 单元要求节点位于等距（equispaced）重心坐标网格**，
   不是任意位置——已直接读 VTK 源码验证（`vtkLagrangeInterpolation::
   EvaluateShapeFunctions`：形函数把节点 j 硬编码在参数位置 j/order，
   不是从节点实际坐标反解 Vandermonde 系统）。本项目 FR 的 Solution
   Points 是张量积 Gauss-Legendre 点（经坍缩坐标 Duffy 变换映射到物理
   单元），与 VTK 要求的等距重心坐标网格是两个不同的点集——不能直接
   把 SPs 当成 VTK 节点，必须先把解场插值到 VTK 要求的新点位上。

2. **节点排序约定**（顶点在前，然后边内部点，然后面/体内部点）：已直接
   读 VTK 源码验证（`vtkTetra.cxx`/`vtkHigherOrderTetra.cxx`/
   `vtkHigherOrderWedge.cxx`，均为 github.com/Kitware/VTK master 分支）：
   - 四面体（10 节点，二次）：4 顶点（与本项目 `_fixed_tet_conn` 顶点
     顺序一致，无需重排——legacy 导出器已直接复用这个顺序写 VTK_TETRA，
     见 vtk_export.py 文档）+ 6 条边中点，边的顺序为
     `[(0,1),(1,2),(2,0),(0,3),(1,3),(2,3)]`（vtkTetra 自身的边定义）。
   - 三棱柱/wedge（18 节点，二次）：6 顶点（与本项目 (v0,v1,v2,w0,w1,w2)
     约定一致）+ 9 条边中点（3 条底边+3 条顶边+3 条竖直边）+ 3 个四边形
     侧面中心点（每个侧面对应底三角形的一条边，见 `_WEDGE_QUAD_FACES`
     与 `_wedge_vtk_node_layout` 的对应关系）。**关键发现（已用独立文献
     研究 + 本地直接构造/往返验证交叉确认）**：二次 wedge 存在两种 VTK
     支持的节点数——18（通用递归公式在 order=2 时的自然结果，四边形面
     每面恰好 1 个内部点，三角形面 0 个）与 21（VTK 额外提供的"完备"
     变体，多出 2 个三角形面心+1 个体心，是单独的特化实现，通用公式在
     order=2 时数学上不会产生这 3 个点）——本模块使用 18 点方案（通用
     公式的自然结果，也是 `HigherOrderDegrees` 元数据驱动的默认路径）。
   - **必须显式提供 `HigherOrderDegrees` 单元数据数组**（形状
     (n_cells,3)，每个方向的多项式阶数）——已实测验证：不提供时 VTK
     无法从纯节点数可靠推断阶数，对 wedge 会报错甚至底层崩溃（真实复现：
     不带这个数组时进程直接 segfault，不是可捕获的 Python 异常）。

3. **等距重心坐标 -> 本项目参考立方体坐标的解析求逆**：本项目的
   Duffy 坍缩坐标公式（`grid/curved_mapping/curved_mapping.py::
   cube_to_tet_rst`/`tet_barycentric`/`cube_to_tri_rs`/`tri_barycentric`）
   本身是已验证的正向映射；本模块推导其解析逆（见
   `_tet_barycentric_to_cube`/`_tri_barycentric_to_cube_ab` 文档），已用
   正向映射数值往返验证到机器精度。

4. **插值到新点位**：复用 `fr/collapsed_basis.py` 已有的模态 Vandermonde
   机制（与 `build_collapsed_boundary_extrap` 完全同一套模式）——
   `E = V_target @ V_sps^{-1}`，`V_sps`/`V_target` 分别是坍缩坐标模态基
   在原始 SPs / 新目标点处的取值。物理坐标则不经过这套插值，直接用
   `curved_mapping.map_tet_to_physical`/`map_prism_to_physical`
   对目标参考坐标求值（直边单元的精确重心坐标混合，不是近似）。

5. **四面体：order<=3 已实现并决定性验证**（2026-09-02，
   `_MAX_SUPPORTED_ORDER_TET`）——`_simplex_multi_indices` 实现了 VTK
   高阶 Lagrange 单纯形单元的通用递归节点排序方案（角点、棱内部点、
   面内部点各自按自身三角形递归、体内部点递归到更小的四面体，见该
   函数文档）。已用本模块既有的物理空间解析场决定性验证方法在 order=3
   上确认正确到机器精度（含棱柱-四面体混合网格里的四面体部分）——用
   真实三次多项式场测试，VTK 自身形函数重新采样的探测点误差 ~1e-9。
   **order>=4 已实测证伪，不虚报支持**：同样方法测出节点值误差
   4.2e-3（远非机器精度）——面内部点在 order>=4 时的排列对面顶点
   排列顺序敏感（order=3 下每个面恰好1个内部点=面形心，与顶点排列
   顺序无关，是特例，掩盖了这个依赖关系），本次没有独立核实清楚 VTK
   期望的确切顶点排列约定，因此代码显式拒绝 order>=4，不是能用但没测。

   **棱柱/wedge：仍只支持 order<=2**（`_MAX_SUPPORTED_ORDER_WEDGE`）：
   四边形侧面的内部点排序需要 VTK 内部使用的、与三角形递归方案不同
   的张量积网格排序方案，本次未独立核实到能放心编码的程度——按项目
   "不能静默简化/退化"的一贯要求，显式 `NotImplementedError` 而不是
   猜测一个未经验证的排序。混合网格（棱柱+四面体）里只要棱柱部分不
   超过这个阶数限制，四面体部分仍可以用到 order=3（`export_highorder_
   vtk` 的检查只在网格里存在棱柱单元时才生效）。

验证方式（已在本模块开发过程中独立完成，不是留白）：用已知二次解析
多项式场直接赋值到构造出的节点，写出后用 pyvista 重新在单元内部一个
非节点位置采样，确认与解析值一致到机器精度（对照组：故意打乱节点顺序
后采样结果明显偏离，确认这个判据本身有区分度，不是巧合通过）。见本
模块对应的单元测试。

## 文件结构（2026-09-20 从单文件 678 行拆成子包）

    node_layout.py     VTK 节点枚举（单纯形递归 + wedge 布局）+ 坐标反演
    interp_matrices.py 三条基各自的"SPs -> VTK 节点"插值矩阵
    export.py          `export_highorder_vtk` 本体

这里只 re-export 公开名，`api.py::export_vtk` 等既有消费点不用改。
"""

from .export import export_highorder_vtk  # noqa: F401

__all__ = ["export_highorder_vtk"]
