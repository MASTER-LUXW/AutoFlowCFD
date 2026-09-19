"""AutoFlowCFD V2.0 - 原生（非坍缩）基过积分阶数的**共享**解析逻辑。

## 为什么单独一个模块

`fr/collapsed_basis.py::OVERINTEGRATION_MAX_ORDER = 3` 是**坍缩坐标模态
Vandermonde 的数值条件数极限**（cond 在 N=4 达约 1e14、`max|D|` 暴涨约
6.3 万倍，见该常量处的完整记录），不是去混叠本身的要求。原生基没有这个
问题，所以四面体（2026-09-17）和棱柱（2026-09-19）各自有**自己的**上限
环境变量。

两者的 env 解析与"`rule*order` 取 min 再对 order 取 max"这套规则逐字
相同。此前只有四面体一份，写在 `native_tet_overintegration.py` 里；棱柱
接入时把它抄第二遍是本项目明确禁止的（同一语义两份实现，改一份漏一份
已经出过真实缺陷——滤波档双解析器、CFL 三处硬编码兜底）。所以这里做成
参数化的唯一实现，两个基各自只保留一行"传自己的 env 名与默认值"的薄
包装。

## 实测依据（为什么原生基可以不设额外上限）

    基          阶数  cond(V)    max|D|
    坍缩棱柱    oo=3  6.19e+03    560.1
    坍缩棱柱    oo=4  4.72e+05  33928.8   <- 条件数崩掉的地方
    原生棱柱    oo=4  6.33e+01      7.7
    原生棱柱    oo=8  1.86e+03     24.7   <- 比坍缩 oo=4 还好 250 倍
    原生四面体  oo=6  3.86e+03      -     （见 native_tet_overintegration）

而去混叠精度在 `oo = 2*order` 处断崖式下降（四面体 P3: oo=5 3.37e-3 vs
oo=6 4.80e-6，18600 倍），所以"不额外设限、让 `rule*order` 说话"是两个
原生基共同的正确取法。
"""

import os

#: 常识护栏上界：`n_fine` 随阶数三次增长、体积项收缩是 O(n_fine^2)，
#: 超过这个值不是数值失效而是没有实用意义。两个原生基共用同一个护栏。
NATIVE_OVERINT_SANITY_MAX = 8


def resolve_native_overintegration_max_order(env_name: str, default: int) -> int:
    """读 `env_name` 指定的环境变量，缺省返回 `default`。

    非法取值直接报错、不静默回退——静默回退会让一次拼写错误伪装成默认
    行为，把 A/B 的两条运行悄悄变成同一档（同一原则见
    `fr_operators/kernels.py::resolve_ausm_precond_mode`；本项目已经因为
    "两条'不同 CFL'给出逐位相同轨迹"吃过一次亏）。
    """
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return int(default)
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(f"{env_name}={raw!r} 不是整数") from None
    if not (1 <= v <= NATIVE_OVERINT_SANITY_MAX):
        raise ValueError(
            f"{env_name}={v} 超出 [1, {NATIVE_OVERINT_SANITY_MAX}]。上界"
            f"不是数值极限而是常识护栏：n_fine 随阶数三次增长、体积项"
            f"收缩是 O(n_fine^2)")
    return v


def resolve_native_overintegration_order(
    order: int, env_name: str, default_max: int
) -> int:
    """定某个原生基实际使用的 `over_order`。

    两条约束：
      1. 去混叠经验法则 `rule * order`（**与基无关**，见
         `collapsed_basis.resolve_overintegration_order_rule`）；
      2. 该基自己的上限（`env_name`，默认 `default_max`）。

    Returns:
        实际 `over_order`（>= order；等于 order 时过积分退化为恒等）。
    """
    from .collapsed_basis import resolve_overintegration_order_rule

    oo = min(resolve_overintegration_order_rule() * int(order),
             resolve_native_overintegration_max_order(env_name, default_max))
    return max(oo, int(order))


# ---------------------------------------------------------------------------
# 棱柱：跨基档的唯一入口
# ---------------------------------------------------------------------------
#
# 棱柱的 `over_order` 与"细点数"此前在两个文件里各算一遍同一个式子
# （`fr/operators.py::generate_fr_operators` 与 `grid/high_order/
# high_order_mesh_order.py::build_order_geometry`），两处的注释里都写着
# "必须与另一处逐字一致，否则算子形状与 mesh.jacobians_fine 对不上"。
# 那种"靠注释维持一致"的写法在本项目已经出过真实缺陷（滤波档双解析器、
# CFL 三处硬编码兜底），而且现在还要再叠一层原生/坍缩分档 —— 所以合并成
# 下面这两个函数，两个调用点都改成引用它们。

#: 原生棱柱过积分阶数上限，含义是"不再额外设限"（实际阶数由
#: `rule*order` 决定）。`fr/collapsed_basis.py::OVERINTEGRATION_MAX_ORDER
#: = 3` 是**坍缩坐标** Vandermonde 的条件数极限，原生基实测 oo=8 的
#: cond(V)=1.86e3 比坍缩 oo=4 的 4.72e5 还小 250 倍（见模块文档表格），
#: 没有理由继承它。与四面体取同一个值是因为是同一个结论，不是抄默认值
#: —— 两个基各有自己的 env，可以独立调。
NATIVE_PRISM_OVERINTEGRATION_MAX_ORDER = 6

#: 原生棱柱过积分上限的环境变量名（唯一事实来源，别处只引用这个常量）。
NATIVE_PRISM_OVERINT_MAX_ORDER_ENV = "AFCFD_PRISM_OVERINT_MAX_ORDER"


def resolve_prism_overintegration_max_order() -> int:
    """原生档下棱柱过积分阶数上限（读
    `AFCFD_PRISM_OVERINT_MAX_ORDER`，默认
    `NATIVE_PRISM_OVERINTEGRATION_MAX_ORDER`）。"""
    return resolve_native_overintegration_max_order(
        NATIVE_PRISM_OVERINT_MAX_ORDER_ENV,
        NATIVE_PRISM_OVERINTEGRATION_MAX_ORDER)


def resolve_prism_overintegration_order(order: int) -> int:
    """棱柱实际使用的 `over_order`，**按当前生效的棱柱基分档**。

    * 坍缩档：`min(rule*order, collapsed_basis.OVERINTEGRATION_MAX_ORDER)`
      —— 那个上限是坍缩 Vandermonde 的条件数极限（放宽到 4 会让 P2 均匀
      自由流残差从 1.06e-5 恶化到 5.6e-3，实测记录在该常量处）。
    * 原生档：`min(rule*order, AFCFD_PRISM_OVERINT_MAX_ORDER)`，默认上限
      6 = 不额外设限，所以 P1/P2/P3 拿到理想的 2/4/6。

    `order == 0` 时返回 0（P0 走独立的有限体积残差路径，没有可去混叠的
    内容）。
    """
    from .collapsed_basis import (
        OVERINTEGRATION_MAX_ORDER, resolve_overintegration_order_rule,
    )
    from .native_prism.mode import prism_basis_is_native

    order = int(order)
    if order <= 0:
        return 0
    if prism_basis_is_native():
        return resolve_native_overintegration_order(
            order, NATIVE_PRISM_OVERINT_MAX_ORDER_ENV,
            NATIVE_PRISM_OVERINTEGRATION_MAX_ORDER)
    return max(order, min(resolve_overintegration_order_rule() * order,
                          OVERINTEGRATION_MAX_ORDER))


def prism_n_fine(over_order: int) -> int:
    """棱柱在 `over_order` 下的细点数，**按当前生效的棱柱基分档**。

    这个值同时是 `mesh.n_sps_per_cell_fine`（`jacobians_fine` 的每单元
    布局宽度）：

    * 坍缩档 `(oo+1)^3`（张量积立方体点）；
    * 原生档 `(oo+1)^2 (oo+2)/2`（Warp&Blend 三角形点 x Gauss 挤出点）
      —— **不填充到立方体宽度**，填充槽位恒为零、零贡献，却让整条过积分
      链在空点上白算，而 `D_fine` 的收缩是 O(n_fine^2)。

    四面体段不受这个宽度约束：它的细点度量是逐单元常数、只取第 0 列广播
    （见 `core/fr_operators/volume_contract.get_overintegration_context`）。
    """
    from .native_prism.basis import native_prism_n_sps
    from .native_prism.mode import prism_basis_is_native

    oo = int(over_order)
    if oo <= 0:
        return 0
    if prism_basis_is_native():
        return native_prism_n_sps(oo)
    return (oo + 1) ** 3
