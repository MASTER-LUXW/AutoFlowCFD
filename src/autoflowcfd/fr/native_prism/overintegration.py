"""AutoFlowCFD V2.0 - 原生棱柱基体积项去混叠（过积分）算子。

去混叠的动机与基无关：欧拉通量 x 度量项的真实多项式次数远高于求解阶数，直接在 coarse SPs 上微分等价于先混叠
再求导（实测 P2 残差可以是真值的 43~62 倍）。原生棱柱基**必须有**这套算子：
`volume_contract.get_overintegration_context` 要求六个 `overint_*` 算子全部非 None。

三件套（`basis.py` 的受限 PKD 三角形模态 x 挤出方向 Legendre + Warp&Blend 三角形节点 x Gauss 挤出节点）：

1. `interp_c2f`：**COARSE 阶数**模态基在 FINE 点上取值 —— Q 本身次数
   <=order，这一步是精确插值，不引入混叠。
2. `D_fine`：**FINE 阶数**原生棱柱微分矩阵（直接复用
   `build_native_prism_operators(over_order)`，不是新公式；含
   `enforce_constant_annihilation`，自由流保持性依赖它）。
3. `restrict_f2c`：**FINE 阶数**模态基在 COARSE 点上取值 —— 把微分后
   的场从细网格插值回 coarse SPs（湍流 k-omega 输运用；平均流用下面的 L2
   投影 `build_native_prism_l2_projection`，两者的分工见
   `native_tet/overintegration.py` 模块文档"限制：平均流投影、湍流插值"一节）。

## 阶数与细点

**（一）不设条件数上限。** 原生棱柱的模态 Vandermonde 条件良好（`tests/unit/test_native_prism_basis.py`）：

    基          oo   cond(V)     max|D|
    原生棱柱     4   6.33e+01       7.7
    原生棱柱     8   1.86e+03      24.7

（坍缩坐标张量积基在 oo=4 时 cond(V)=4.72e5、`max|D|`=33928.8，那是它需要上限的原因。）去混叠精度在
`oo = 2*order` 处断崖式下降，所以原生棱柱与原生四面体取同一个做法：实际阶数由 `rule*order` 决定（默认
rule=2，P1/P2/P3 = 2/4/6，`fr/overintegration_order.py`）。

**（二）细点度量必须逐点求值，不能广播。** 直边**四面体**的 Jacobian 是逐单元常数，所以
`get_overintegration_context` 的四面体段直接取第 0 列广播；棱柱即便直边也一般随点变化（只有顶面是底面纯平移
的右棱柱才恒定），所以 `grid/high_order/high_order_mesh_order.py` 的 `build_order_geometry` 必须在**本模块给出
的原生细点**上逐点求 Jacobian（`compute_native_prism_jacobians(..., ref_pts=...)`）。

**（三）细网格轴不填充**：`n_sps_per_cell_fine` 直接取原生细点数 `(oo+1)^2 (oo+2)/2`（填充槽位恒为零、零贡献，
却会让整条过积分链在空点上白算，`D_fine` 的收缩是 O(n_fine^2)）。粗网格轴的存储宽度见 `fr/native_padding.py`。
"""

from typing import Tuple

import numpy as np

from ..quadrature_points import gauss_legendre
from .basis import (
    build_native_prism_operators,
    build_native_prism_vandermonde,
)
# 阶数策略（上限常量/env/`rule*order` 规则）**不在本模块**：它统一住在
# `fr/overintegration_order.py`，那里是棱柱两档与四面体共用的唯一事实
# 来源。本模块只负责矩阵构造，不做 re-export —— 全仓库唯一的调用点
# （`fr/operators/build.py`）只需要下面这个构造函数，而阶数是由
# `overintegration_order.overintegration_order` 在那里
# 直接解析的。多一层 re-export 就多一份要同步的名单。


def build_native_prism_overintegration_operators(
    order: int, over_order: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造原生棱柱体积项去混叠三件套。

    Args:
        order: 当前求解阶数 P（coarse）
        over_order: 过积分阶数（由 `overintegration_order`
            给出）

    Returns:
        `(ref_rst_fine, interp_c2f, D_fine, restrict_f2c)`：
        `ref_rst_fine` 形状 `(n_fine,3)`；`interp_c2f` 形状
        `(n_fine,n_coarse)`；`D_fine` 形状 `(n_fine,n_fine,3)`；
        `restrict_f2c` 形状 `(n_coarse,n_fine)` —— 与原生四面体同一形状
        约定，`fr_residual/inviscid.py` 的过积分分支按同一套
        `contract_shared_operator_1axis/2axis` 消费，不需要改那段通用
        代码。这里的 `n_coarse`/`n_fine` 是**原生真实**节点数，粗轴填充
        到全局张量积宽度在 `fr/operators/` 里用
        `native_padding.pad_native_matrix_to_global` 完成（与四面体同
        一条路径）。
    """
    from scipy.linalg import lu_factor, lu_solve

    ref_coarse, _ = build_native_prism_operators(order)
    ref_fine, D_fine = build_native_prism_operators(over_order)

    # `build_native_prism_vandermonde(m, pts)` 就是"第 m 阶模态集在任意
    # 点上取值"，跨阶数直接可用 —— 不需要像四面体那样另写一个
    # `_native_modal_vandermonde` 包装（那边是因为模态求值没有这样的
    # 现成入口）。
    V_coarse_at_coarse = build_native_prism_vandermonde(order, ref_coarse)[0]
    V_coarse_at_fine = build_native_prism_vandermonde(order, ref_fine)[0]
    V_fine_at_fine = build_native_prism_vandermonde(over_order, ref_fine)[0]

    # `interp = V_at_target @ V_at_source^{-1}`，用 `lu_factor(V.T)` 后
    # 转置求解（`lu_solve(lu, B.T).T` 等价于 `B @ V^{-1}`，避免显式求逆）。
    lu_coarse = lu_factor(V_coarse_at_coarse.T)
    interp_c2f = lu_solve(lu_coarse, V_coarse_at_fine.T).T

    restrict_f2c = lu_solve(lu_factor(V_fine_at_fine.T),
                            build_native_prism_vandermonde(over_order, ref_coarse)[0].T).T

    return ref_fine, interp_c2f, D_fine, restrict_f2c


def build_native_prism_l2_projection(order: int, over_order: int) -> np.ndarray:
    """细层（over_order 次）节点值 -> 求解空间（order 次）节点值的参考单元 L2 投影
    `M_c^{-1} ∫ φ_i g`，`(n_coarse, n_fine)`（平均流体积项用，见模块文档第 3 条）。

    两套节点 Lagrange 基在 Duffy 求积点上取值。被积函数（粗 x 细）在三角形与挤出方向
    各至多 order+over_order 次，Duffy 因子再加 1 次，一维 n 点 Gauss 精确到 2n-1 次，
    n = order+over_order+2 有富余。
    """
    from scipy.linalg import lu_factor, lu_solve

    ref_coarse, _ = build_native_prism_operators(order)
    ref_fine, _ = build_native_prism_operators(over_order)
    xq, wq = _prism_duffy_rule(order + over_order + 2)
    L_coarse = lu_solve(lu_factor(build_native_prism_vandermonde(order, ref_coarse)[0].T),
                        build_native_prism_vandermonde(order, xq)[0].T).T
    L_fine = lu_solve(lu_factor(build_native_prism_vandermonde(over_order, ref_fine)[0].T),
                      build_native_prism_vandermonde(over_order, xq)[0].T).T
    return np.linalg.solve(L_coarse.T @ (wq[:, None] * L_coarse), L_coarse.T @ (wq[:, None] * L_fine))


def _prism_duffy_rule(n: int):
    """参考棱柱上的 Duffy 张量 Gauss 求积 `(点 (n^3,3), 权重 (n^3,))`。"""
    a_1d, w_1d = gauss_legendre(n)
    a, b, t = (x.ravel() for x in np.meshgrid(a_1d, a_1d, a_1d, indexing="ij"))
    wa, wb, wt = (x.ravel() for x in np.meshgrid(w_1d, w_1d, w_1d, indexing="ij"))
    pts = np.stack([(1.0 + a) * (1.0 - b) / 2.0 - 1.0, b, t], axis=1)
    return pts, wa * wb * wt * (1.0 - b) / 2.0
