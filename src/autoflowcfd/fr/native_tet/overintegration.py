"""AutoFlowCFD V2.0 - native 四面体（路径C）体积项去混叠（过积分）算子。

背景：`fr/collapsed_basis.py::build_overintegration_operators` 文档
完整记录了去混叠的物理动机（欧拉通量×度量项的非线性乘积真实多项式
次数远高于求解阶数，直接在 coarse SPs 上微分等价于先混叠再求导，真实
数值实验测得 P2 残差可以是真值的 43~62 倍）——这个机制与坍缩坐标/
native 哪套基函数无关，native 单纯形基同样需要它。
（历史：本段曾写"`inviscid.py` 目前对 native 网格完全跳过这一步
（`jacobians_fine=None`）"——那是本文件刚建立时的状态，早已过时：native
网格在 order>=1 时无条件构造 `jacobians_fine` 与本文件的三件套。）

本文件的三件套与 `build_overintegration_operators` 逐项对应，只是
换成 native 单纯形基（`native_tet/basis.py` 的受限 PKD 模态 +
Warp&Blend 节点），数学结构完全一致：
1. `interp_c2f`：**COARSE 阶数**模态基在 FINE 点上取值——Q 本身次数
   <=order，这一步是精确插值，不引入混叠。
2. `D_fine`：**FINE 阶数**原生单纯形微分矩阵（直接复用
   `build_native_tet_operators(over_order)`，不是新公式）。
3. `restrict_f2c`：**FINE 阶数**模态基在 COARSE 点上取值——把微分后
   的场从细网格精确插值（不是模态截断投影：过积分的目的正是要保留
   非线性通量的高阶内容,不能在算完导数后又把它截断掉,这一步只是把
   已经算好的、真实值得信赖的细网格结果在 COARSE 点上原样取值）回
   coarse SPs。

与坍缩坐标版本的关键差异：native 单纯形基节点数随阶数增长的方式不同
（`(order+1)(order+2)(order+3)/6` vs `(order+1)^3`），**过积分阶数也已于
2026-09-17 与棱柱解耦**（`resolve_tet_overintegration_order`）。

**这是两件独立的事，2026-09-16 前本文件把它们混成了一件**：

  * `rule*order`（默认 2x）是**与基无关**的去混叠经验法则（二次非线性
    的标准做法），native 照搬是对的；实测误差在 `oo = 2*order` 处断崖式
    下降，低于它基本拿不到收益（P3: oo=5 3.37e-3 vs oo=6 4.80e-6）；
  * 而 `OVERINTEGRATION_MAX_ORDER = 3` 这个**上限**是坍缩坐标模态
    Vandermonde 的**数值条件数**极限（cond 在 N=4 达约 1e14、`max|D|`
    暴涨约 6.3 万倍，见 collapsed_basis.py 该常量处的完整记录），
    native 没有理由继承它——实测 native 在 over_order=6 才 cond=3856、
    `max|D|` 从 3 到 6 只长 3.5 倍（
    `tests/unit/test_native_tet_overintegration_conditioning.py`）。

曾经有第三个理由（本段旧文字）：`jacobians_fine` 是棱柱/四面体共用一个
`n_sps_per_cell_fine` 维度的合并数组，"按较大者分配"会让 plate_demo P2
的该数组从约 1.86 GB 涨到约 3.63 GB。**那条论证是错的**：四面体段根本不
需要更宽的数组——直边四面体的细点度量是一个逐单元常数的原样广播，取第 0
列广播到本段自己的 `n_fine_tet` 即可（见 `core/fr_operators/
volume_contract.get_overintegration_context`）。

现在 P1/P2/P3 都取到理想的 `2*order`（2/4/6）。放开上限本身会抬高自由流
保持性的误差底（`D_fine` 对常数的零化残余随阶数增长），已同批用
`fr/diff_matrix_consistency.enforce_constant_annihilation` 消掉，真实网格
实测见 `tests/unit/test_tet_overintegration_cap_raised.py`。
"""

from typing import Optional, Tuple

import numpy as np

from ..overintegration_order import (
    resolve_native_overintegration_max_order,
    resolve_native_overintegration_order,
)
from .basis import (
    build_native_tet_operators,
    restricted_tet_modes,
    simplex3d_value,
    rst_to_abc,
)


def _native_modal_vandermonde(ref_rst: np.ndarray, modes) -> np.ndarray:
    """给定一批参考坐标 (r,s,t) 和一组 (i,j,k) 模态索引，返回
    Vandermonde 矩阵 (n_pts, n_modes)——`build_native_tet_lift`/
    `build_native_tet_boundary_extrap` 已经在各自函数体内重复写过
    这个三行代码，这里单独提出来给过积分算子复用，避免第四份重复。
    """
    a, b, c = rst_to_abc(ref_rst[:, 0], ref_rst[:, 1], ref_rst[:, 2])
    return np.column_stack([simplex3d_value(a, b, c, i, j, k) for (i, j, k) in modes])


def build_native_tet_overintegration_operators(
    order: int, over_order: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造 native 四面体体积项去混叠三件套，与
    `build_overintegration_operators("tet", order, over_order, ...)`
    同样的用途和消费方式（`fr_residual/inviscid.py` 的过积分分支）。

    Args:
        order: 当前求解阶数 P（coarse）
        over_order: 过积分阶数（`min(2*order, OVERINTEGRATION_MAX_ORDER)`，
            与坍缩坐标版本同一个经验法则，见模块文档）

    Returns:
        (ref_rst_fine, interp_c2f, D_fine, restrict_f2c)：
        `ref_rst_fine` 形状 (n_fine,3)；`interp_c2f` 形状
        (n_fine,n_coarse)；`D_fine` 形状 (n_fine,n_fine,3)；
        `restrict_f2c` 形状 (n_coarse,n_fine)——与坍缩坐标版本完全
        对应的形状约定，`inviscid.py` 的过积分分支可以按同一套
        `contract_shared_operator_1axis/2axis` 消费，不需要改动那段
        通用代码本身。
    """
    from scipy.linalg import lu_factor, lu_solve

    ref_coarse, _ = build_native_tet_operators(order)
    ref_fine, D_fine = build_native_tet_operators(over_order)

    modes_coarse = restricted_tet_modes(order)
    modes_fine = restricted_tet_modes(over_order)

    V_coarse_sps = _native_modal_vandermonde(ref_coarse, modes_coarse)
    V_coarse_at_fine = _native_modal_vandermonde(ref_fine, modes_coarse)
    V_fine_at_fine = _native_modal_vandermonde(ref_fine, modes_fine)
    V_fine_at_coarse = _native_modal_vandermonde(ref_coarse, modes_fine)

    lu_coarse = lu_factor(V_coarse_sps.T)
    interp_c2f = lu_solve(lu_coarse, V_coarse_at_fine.T).T

    lu_fine = lu_factor(V_fine_at_fine.T)
    restrict_f2c = lu_solve(lu_fine, V_fine_at_coarse.T).T

    return ref_fine, interp_c2f, D_fine, restrict_f2c

#: native 四面体过积分阶数的**独立**上限（2026-09-17）。
#:
#: 为什么与棱柱分开：`collapsed_basis.OVERINTEGRATION_MAX_ORDER = 3` 是
#: **坍缩坐标模态 Vandermonde 的数值条件数极限**（cond 在 N=4 达约 1e14、
#: `max|D|` 暴涨约 6.3 万倍），不是去混叠本身的要求。native 受限 PKD 基
#: 实测在 over_order=6 才 cond=3856、`max|D|` 从 3 到 6 只长 3.5 倍
#: （`tests/unit/test_native_tet_overintegration_conditioning.py`），那条
#: 论证对它不成立。
#:
#: 上限的代价是实测过的（`tests/unit/test_overintegration_cap_cost.py`，
#: 以 over_order=8 为参照的体积项去混叠相对误差）：
#:
#:     P2  oo=3（旧上限）2.38e-2  ->  oo=4（理想）7.01e-6   3400 倍
#:     P3  oo=3（=order，完全无操作）9.17e-2 -> oo=6 4.92e-6  18600 倍
#:
#: 取 6 的含义是"不再额外设限"——实际阶数由 `rule*order` 的经验法则决定
#: （默认 rule=2，所以 P1 取 2、P2 取 4、P3 取 6）。
#:
#: **不再有布局约束**（2026-09-17 第二次改动）：四面体段的细点度量现在从
#: `jacobians_fine` 的**第 0 列广播**而不是切前 n_fine_tet 列（直边四面体
#: 逐单元常数，第 0 列就是那个常数，见 `core/fr_operators/volume_contract.
#: get_overintegration_context`）。第一版的"切列"要求 `n_fine_tet <=
#: (oo_prism+1)^3`，把 P3 夹到 5；而去混叠误差在 `oo = 2*order` 处断崖式
#: 下降，P3 被夹住只拿到 18.6 倍中的 13000 倍：
#:
#:     P3  oo=3  6.26e-2   oo=4  2.66e-2   oo=5  3.37e-3   oo=6  4.80e-6
#:
#: 所以现在 P1/P2/P3 都取到理想的 `2*order`。
NATIVE_TET_OVERINTEGRATION_MAX_ORDER = 6


def native_tet_n_fine(over_order: int) -> int:
    """native 四面体在 `over_order` 下的**真实**细点数。"""
    oo = int(over_order)
    return (oo + 1) * (oo + 2) * (oo + 3) // 6


def resolve_tet_overintegration_max_order() -> int:
    """读 `AFCFD_TET_OVERINT_MAX_ORDER`，默认
    `NATIVE_TET_OVERINTEGRATION_MAX_ORDER`。

    实现在 `fr/overintegration_order.py`（与原生棱柱共用同一份
    env 解析，见该模块文档"为什么单独一个模块"）；这里只是保留四面体
    自己的公开名与默认值。
    """
    return resolve_native_overintegration_max_order(
        "AFCFD_TET_OVERINT_MAX_ORDER", NATIVE_TET_OVERINTEGRATION_MAX_ORDER)


def resolve_tet_overintegration_order(
    order: int, n_fine_layout: Optional[int] = None
) -> int:
    """定四面体实际用的 over_order（`rule*order` 与自身上限取 min）。

    Args:
        order: 求解阶数 P
        n_fine_layout: **已废弃、被忽略**。第一版有第三条"布局约束"
            `native_tet_n_fine(oo) <= (oo_prism+1)^3`，因为四面体段的细点
            度量是切共用数组的前 n_fine_tet 列。现在改成第 0 列广播
            （`volume_contract.get_overintegration_context`），那条约束
            消失。形参保留只为不破坏既有调用点/测试的签名，传什么都不影响
            返回值。

    Returns:
        实际 over_order（>= order；等于 order 时过积分退化为恒等）。
    """
    return resolve_native_overintegration_order(
        order, "AFCFD_TET_OVERINT_MAX_ORDER",
        NATIVE_TET_OVERINTEGRATION_MAX_ORDER)
