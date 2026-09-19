"""
AutoFlowCFD V2.0 - FR 算子生成器主接口

本模块是 Flux Reconstruction 方法算子生成的统一入口，
整合了点集生成、矩阵算子计算等功能。

核心功能：
1. 统一的FR算子生成接口
2. 算子数据容器定义

坍缩坐标四面体基已删除（2026-09-03，用户明确要求"删除 collapsed 相关
内容"，前提是 native 已在同一天的排查中决定性验证覆盖 CPU/GPU 全部
残差链路——含此前被误判为缺失的 over-integration 组合，见
`native_tet_basis_deletion_2026_09_03` 记忆）：四面体不再有
`tet_basis_mode` 选择，永远走 native 单纯形基（Part6/7/8 文档"路径C"，
`fr/native_tet/basis.py`）——不是"默认改成 native"，是坍缩坐标
（Duffy 变换）四面体基函数构造本身（`collapsed_basis.py::
build_collapsed_diff_matrices("tet",...)`/`build_collapsed_boundary_
extrap("tet",...)`/`modal_filter.py::build_tet_modal_filter`）不再被
调用，这部分计算真正被删除，不是被绕开。棱柱不受影响——棱柱没有
"collapsed vs native"的选择，坍缩坐标是棱柱唯一、正确的构造方式（三棱柱
=三角形×直线，不存在四面体那种坍缩坐标退化轴条件数问题），
`collapsed_basis.py`/`modal_filter.py` 对 "prism" 的调用原样保留。

`D_3d_tet`/`filter_tet` 两个字段现在直接**别名**到对应的 native 填充
版本（`D_native_tet_padded`/`filter_native_tet_padded`，两者形状恒等，
是"零填充块对角"设计从一开始就保证的，见 native_padding.py 文档）；
`boundary_extrap_tet` 保留为占位零矩阵字典（形状与之前一致），只是为了
不用同步修改 `core/fr_operators/face_kernels.py` 里"无条件按 (celltype,
axis,side) 读取 boundary_extrap_tet/prism 拼成统一查找表"这一段代码——
四面体面在 `with_native_face_codes` 静态翻译（现在总是执行，见
`grid/connectivity/face_connectivity.py`/`grid/high_order/high_order_
mesh.py`）之后恒为 `tet_native_v*` 编码（>=6），这些占位行在生产路径上
永远不会被真正读取（`_native_self_extrap` 等函数按 `is_native` 掩码
丢弃 collapsed 分支的结果，只是 NumPy/CuPy 花式索引要求索引本身合法，
见 gpu_inviscid.py 模块文档"提前无条件求值导致越界"一节的同一原理）。

## 文件分工（2026-09-19 拆包）

    container.py   `FROperators` 数据容器与它的取算子方法
    build.py       `generate_fr_operators` 构建流程

本 `__init__.py` re-export 三个既有公开名（`FROperators`/
`generate_fr_operators`/`gauss_legendre`），所以全仓库
`from autoflowcfd.fr.operators import ...` 一个字都不用改。
"""

from ..quadrature_points import gauss_legendre, gauss_lobatto  # noqa: F401
from .container import FROperators  # noqa: F401
from .build import generate_fr_operators  # noqa: F401

__all__ = ["FROperators", "generate_fr_operators",
           "gauss_legendre", "gauss_lobatto"]
