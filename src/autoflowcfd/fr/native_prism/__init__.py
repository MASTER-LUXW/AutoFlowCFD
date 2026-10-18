"""AutoFlowCFD V2.0 - 棱柱原生基（三角形 PKD/Dubiner ⊗ 直线 Legendre）。

## 这个子包在解决什么

棱柱长期使用**坍缩坐标基**（`fr/collapsed_basis.py::prism_modal_basis_and_grad`，
在 (a,b) 截面上做 Duffy 三角形坍缩）。实测三条后果：

1. 微分矩阵元素量级随阶数爆炸（`max|D_3d_prism|` P1 2.05 -> P3 560.1）；
2. 自由流保持性只有 ~1e-9，且严格等于 `eps * max|D| / det(J)` —— 舍入被
   算子量级放大，**不降低 `max|D|` 就改不掉**；
3. **P2/P3 离散不稳定**：Couette 的精确解是 `y` 的线性函数、在各阶空间里
   都精确可表示，从精确解出发跑 1600 步后 P1 保持到 6.62e-9、P2 偏离
   1.78e-2、P3 第 5 步发散。不是分辨率问题，是离散本身。

四面体当年有完全同类的病理，解法是整套换成 native PKD 基。本子包是棱柱侧
的同一条路。

## 为什么只剩原生基（坍缩棱柱基 2026-09-20 退出默认、2026-09-23 删除）

删除前的负控制实测（代码删了，证据仍然有效）：

    判据                              坍缩        原生
    真实网格 179,237 单元 P1+LES      第148步 nan  246步单调降2.5x
    等熵涡 P2（800 棱柱、CFL 0.1）    第23步非有限 150步降到初值0.557
    Couette 精确保持性 P2             1.78e-2     8.40e-9（210万倍）
    伪横流最小复现 P2 CFL 0.1         第82步 nan   400步有限 5.25e-2
    GCL 残差 P2 / P3                  1.5e-15 /    7.1e-17 /
                                      4.5e-14      1.4e-16
    过积分后体积项误差 P2             9.5e-3      2.9e-12（机器零）
    cond(V) oo=4 / max|D| oo=4        4.72e5 /    6.33e1 /
                                      33928.8      7.7
    粘性算子动量块 max Re             +5.80e+00   +0.00e+00

坍缩档还与内部面 IP 罚项数学上不兼容：罚项须经 DG 提升 `M^-1 ∮`，原生路径用 `lift_native` 满足，坍缩路径的界面项
走 1D Radau/VCJH 修正函数，同一个罚项把动量块从严格耗散变成 10 个增长模态。2026-10-18 起迁移期开关
`AFCFD_PRISM_BASIS`（`mode.py`）一并删除；checkpoint 仍记录棱柱基并拒绝非原生基写的文件
（`core/utils/checkpoint_load.py::_check_prism_basis`）。

## 文件划分（按功能逻辑）

    triangle_basis.py  三角形 PKD/Dubiner 基 + Warp & Blend 节点
    basis.py           棱柱 = 三角形 ⊗ 直线：节点/Vandermonde/微分算子/
                       几何映射/解析雅可比
    face.py            5 个面：通量点、体积->面外插、DG 提升、adj 行、
                       立方体面对应表
    interp_numba.py    numba 侧的"目标点 -> 插值矩阵"（残差 kernel 消费）

## 与坍缩基的两条可复用的支点事实（实测）

1. **两条基的几何映射逐位恒等**：`map_prism_to_physical(a,b,c)` 与
   `map_native_prism_to_physical(cube_to_tri_rs(a,b), c)` 在 2000 个随机点上
   `max|diff|` **恰好 0.0**。所以面点定位层（含 numba Newton 预计算）不必
   重写，只需要把定位结果换算过去、再换成原生 Vandermonde。
2. `det_collapsed = det_native * (1-b)/2`（吻合 2.7e-15）。
"""

from .basis import (
    build_native_prism_nodes,
    build_native_prism_operators,
    build_native_prism_vandermonde,
    map_native_prism_to_physical,
    native_prism_exact_jacobian,
    native_prism_n_sps,
    restricted_prism_modes,
)
from .face import (
    NATIVE_PRISM_FACE_TO_CUBE_FACE,
    PRISM_FACE_IDS,
    build_all_native_prism_face_operators,
    build_native_prism_boundary_extrap,
    build_native_prism_lift,
    cube_face_to_native_prism_face,
    native_prism_face_adj_rows,
    native_prism_face_points,
    native_prism_face_points_physical,
    native_prism_mode_norm_squared,
)
from .triangle_basis import (
    build_native_tri_vandermonde,
    eval_tri_modes,
    restricted_tri_modes,
    rs_to_ab,
    simplex2d_grad,
    simplex2d_value,
    warp_blend_nodes_2d,
)

__all__ = [
    # mode
    # triangle
    "build_native_tri_vandermonde",
    "eval_tri_modes",
    "restricted_tri_modes",
    "rs_to_ab",
    "simplex2d_grad",
    "simplex2d_value",
    "warp_blend_nodes_2d",
    # basis
    "build_native_prism_nodes",
    "build_native_prism_operators",
    "build_native_prism_vandermonde",
    "map_native_prism_to_physical",
    "native_prism_exact_jacobian",
    "native_prism_n_sps",
    "restricted_prism_modes",
    # face
    "NATIVE_PRISM_FACE_TO_CUBE_FACE",
    "PRISM_FACE_IDS",
    "build_all_native_prism_face_operators",
    "build_native_prism_boundary_extrap",
    "build_native_prism_lift",
    "cube_face_to_native_prism_face",
    "native_prism_face_adj_rows",
    "native_prism_face_points",
    "native_prism_face_points_physical",
    "native_prism_mode_norm_squared",
]
