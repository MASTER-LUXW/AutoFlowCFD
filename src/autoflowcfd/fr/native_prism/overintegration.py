"""AutoFlowCFD V2.0 - 原生棱柱基（`AFCFD_PRISM_BASIS=native`）体积项
去混叠（过积分）算子。

去混叠的物理动机与基无关，完整记录在
`fr/collapsed_basis.py::build_overintegration_operators`（欧拉通量 x
度量项的真实多项式次数远高于求解阶数，直接在 coarse SPs 上微分等价于
先混叠再求导，真实数值实验测得 P2 残差可以是真值的 43~62 倍）。原生
棱柱基同样需要它 —— 而且**必须有**：`volume_contract.
get_overintegration_context` 要求六个 `overint_*` 算子全部非 None，缺
任何一个会让整条过积分链（**含四面体段**）静默退回 coarse 路径。

三件套与坍缩版本逐项对应，只是换成原生基（`basis.py` 的受限 PKD 三角形
模态 x 挤出方向 Legendre + Warp&Blend 三角形节点 x Gauss 挤出节点）：

1. `interp_c2f`：**COARSE 阶数**模态基在 FINE 点上取值 —— Q 本身次数
   <=order，这一步是精确插值，不引入混叠。
2. `D_fine`：**FINE 阶数**原生棱柱微分矩阵（直接复用
   `build_native_prism_operators(over_order)`，不是新公式；含
   `enforce_constant_annihilation`，自由流保持性依赖它）。
3. `restrict_f2c`：**FINE 阶数**模态基在 COARSE 点上取值 —— 把微分后
   的场从细网格精确插值回 coarse SPs。**不是**模态截断投影：过积分的
   目的正是要保留非线性通量的高阶内容，不能在算完导数后又截断掉。

## 与坍缩棱柱的两个关键差异

**（一）不继承 `OVERINTEGRATION_MAX_ORDER = 3`。** 那个上限是坍缩坐标
模态 Vandermonde 的**数值条件数**极限，不是去混叠本身的要求。直接实测
（`tests/unit/test_native_prism_overintegration.py`）：

    基          oo   cond(V)     max|D|
    坍缩棱柱     3   6.19e+03     560.1
    坍缩棱柱     4   4.72e+05   33928.8   <- 条件数崩掉的地方
    原生棱柱     4   6.33e+01       7.7
    原生棱柱     8   1.86e+03      24.7   <- 比坍缩 oo=4 还好 250 倍

而去混叠精度在 `oo = 2*order` 处断崖式下降，所以原生棱柱与原生四面体
取同一个做法：**不额外设限**，实际阶数由 `rule*order` 决定（默认
rule=2，P1/P2/P3 = 2/4/6）。上限常量存在只为留一个可调旋钮。

**（二）细点度量必须逐点求值，不能广播。** 直边**四面体**的 Jacobian
是逐单元常数，所以 `get_overintegration_context` 的四面体段直接取第 0
列广播；棱柱即便直边也一般随点变化（只有顶面是底面纯平移的右棱柱才
恒定），所以 `high_order_mesh_order.build_order_geometry` 在原生档下
必须在**本模块给出的原生细点**上逐点求 Jacobian（见那里的
`compute_native_prism_jacobians(..., ref_pts=...)` 调用）。

**（三）细网格轴不填充、`n_sps_per_cell_fine` 直接取原生细点数。**
原生棱柱在 `over_order` 下只有 `(oo+1)^2 (oo+2)/2` 个真实细点（oo=2:
18 vs 坍缩 27；oo=4: 75 vs 125），填充槽位恒为零、零贡献，却让整条
过积分链在空点上白算，其中 `D_fine` 的收缩是 O(n_fine^2)。粗网格轴
仍然必须填充到 `(order+1)^3`，因为 `Q` 数组是那个宽度的填充布局（同
`native_tet/overintegration.py` 的说明）。
"""

from typing import Tuple

import numpy as np

from .basis import (
    build_native_prism_operators,
    build_native_prism_vandermonde,
)
# 阶数策略（上限常量/env/`rule*order` 规则）**不在本模块**：它统一住在
# `fr/overintegration_order.py`，那里是棱柱两档与四面体共用的唯一事实
# 来源。本模块只负责矩阵构造，不做 re-export —— 全仓库唯一的调用点
# （`fr/operators/build.py`）只需要下面这个构造函数，而阶数是由
# `overintegration_order.resolve_prism_overintegration_order` 在那里
# 直接解析的。多一层 re-export 就多一份要同步的名单。


def build_native_prism_overintegration_operators(
    order: int, over_order: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造原生棱柱体积项去混叠三件套。

    Args:
        order: 当前求解阶数 P（coarse）
        over_order: 过积分阶数（由 `resolve_prism_overintegration_order`
            给出）

    Returns:
        `(ref_rst_fine, interp_c2f, D_fine, restrict_f2c)`：
        `ref_rst_fine` 形状 `(n_fine,3)`；`interp_c2f` 形状
        `(n_fine,n_coarse)`；`D_fine` 形状 `(n_fine,n_fine,3)`；
        `restrict_f2c` 形状 `(n_coarse,n_fine)` —— 与坍缩版本完全对应的
        形状约定，`fr_residual/inviscid.py` 的过积分分支按同一套
        `contract_shared_operator_1axis/2axis` 消费，不需要改那段通用
        代码。这里的 `n_coarse`/`n_fine` 是**原生真实**节点数，粗轴填充
        到全局张量积宽度在 `fr/operators.py` 里用
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
    V_fine_at_coarse = build_native_prism_vandermonde(over_order, ref_coarse)[0]

    # `interp = V_at_target @ V_at_source^{-1}`，用 `lu_factor(V.T)` 后
    # 转置求解（`lu_solve(lu, B.T).T` 等价于 `B @ V^{-1}`，避免显式求逆）。
    lu_coarse = lu_factor(V_coarse_at_coarse.T)
    interp_c2f = lu_solve(lu_coarse, V_coarse_at_fine.T).T

    lu_fine = lu_factor(V_fine_at_fine.T)
    restrict_f2c = lu_solve(lu_fine, V_fine_at_coarse.T).T

    return ref_fine, interp_c2f, D_fine, restrict_f2c
